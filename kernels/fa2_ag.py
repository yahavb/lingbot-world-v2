# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch-callable wrappers for the fused all-gather + FlashAttention-2 kernel."""

from __future__ import annotations

import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from kernels.flash_attention2_trn3_allgather import FlashAttentionConfig
from kernels.flash_attention2_trn3_allgather import flash_attention2_trn3_allgather
from nki.compiler import CompileConfig

ATTENTION_TILE_SIZE = 128

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))

# Sk tiles per Sk block (KV DMA size / softmax merge period).
SK_TILES_PER_BLK = int(os.environ.get("FA2_AG_SK_TILES_PER_BLK", "4"))

# FlashAttentionGeometry requires sq_tiles_per_blk > max_update_offset // sk_tiles_per_blk (= 3).
_MIN_SQ_TILES_PER_BLK = 4


def _sq_tiles_per_blk(seq_q: int) -> int:
    """Sq tiles per block: cover all of Q so K/V stream once per chunk."""
    sq_tiles = -(-seq_q // ATTENTION_TILE_SIZE)  # ceil
    return max(sq_tiles, _MIN_SQ_TILES_PER_BLK)


def _run(q_hbm, k_chunks, v_chunks, k_shard, v_shard, replica_group,
         softmax_scale, kv_chunk, sk_tiles_per_blk, k_dest, v_dest):
    """Stage around the collective and call the kernel body.

    Tiling config is built here (not closed over) because it depends on Sq,
    which is only known once the query tensor is in hand.
    """
    n_heads, _, d = k_shard.shape
    seq_q = q_hbm.shape[2]

    v_last_gathered = nl.ndarray(shape=(n_heads, kv_chunk, d), dtype=q_hbm.dtype,
                                 buffer=nl.shared_hbm)

    config = FlashAttentionConfig(
        sq_tiles_per_blk=_sq_tiles_per_blk(seq_q),
        sk_tiles_per_blk=sk_tiles_per_blk,
    )

    out = flash_attention2_trn3_allgather(
        q_hbm, list(k_chunks), list(v_chunks),
        k_shard, v_shard,
        k_dest, v_last_gathered,
        replica_group, softmax_scale, config,
    )

    nisa.dma_copy(dst=v_dest, src=v_last_gathered)
    return out, k_dest, v_dest


# One fixed tensor signature per leading-chunk count. Generating them dynamically
# breaks the NKI tracer.

@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def fa2_ag_0(
    query, key_shard, value_shard, key_destination, value_destination,
    replica_group, softmax_scale, kv_chunk, sk_tiles_per_blk,
):
    return _run(
        query, (), (), key_shard, value_shard, replica_group,
        softmax_scale, kv_chunk, sk_tiles_per_blk, key_destination, value_destination)


@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def fa2_ag_1(
    query, key_0, value_0, key_shard, value_shard,
    key_destination, value_destination, replica_group,
    softmax_scale, kv_chunk, sk_tiles_per_blk,
):
    return _run(
        query, (key_0,), (value_0,), key_shard, value_shard, replica_group,
        softmax_scale, kv_chunk, sk_tiles_per_blk, key_destination, value_destination)


@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def fa2_ag_2(
    query, key_0, key_1, value_0, value_1, key_shard, value_shard,
    key_destination, value_destination, replica_group,
    softmax_scale, kv_chunk, sk_tiles_per_blk,
):
    return _run(
        query, (key_0, key_1), (value_0, value_1), key_shard, value_shard,
        replica_group, softmax_scale, kv_chunk, sk_tiles_per_blk,
        key_destination, value_destination)


@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def fa2_ag_3(
    query, key_0, key_1, key_2, value_0, value_1, value_2,
    key_shard, value_shard, key_destination, value_destination,
    replica_group, softmax_scale, kv_chunk, sk_tiles_per_blk,
):
    return _run(
        query, (key_0, key_1, key_2), (value_0, value_1, value_2),
        key_shard, value_shard, replica_group, softmax_scale,
        kv_chunk, sk_tiles_per_blk, key_destination, value_destination)


_JITTED_AG_KERNELS_BY_LEAD_COUNT = {
    0: fa2_ag_0,
    1: fa2_ag_1,
    2: fa2_ag_2,
    3: fa2_ag_3,
}
_AG_KERNELS_BY_LEAD_COUNT = {
    lead_count: kernel[LNC]
    for lead_count, kernel in _JITTED_AG_KERNELS_BY_LEAD_COUNT.items()
}


def build_fa2_ag_kernels(replica_group, softmax_scale, kv_chunk,
                         sk_tiles_per_blk: int = SK_TILES_PER_BLK):
    """Build four tensor-only callables, keyed by leading-chunk count."""
    def bind(kern):
        def call(*tensors):
            return kern(*tensors, replica_group, softmax_scale, kv_chunk, sk_tiles_per_blk)
        return call

    return {
        lead_count: bind(kernel)
        for lead_count, kernel in _AG_KERNELS_BY_LEAD_COUNT.items()
    }


def ag_fa2_attention(
    query,
    key,
    value,
    k_lead,
    v_lead,
    kernel_call,
    k_dest,
    v_dest,
):
    """Run fused AG FlashAttention-2 from the model's ``[B,S,N,d]`` layout."""
    batch, seq_q, heads, head_dim = query.shape
    assert head_dim == ATTENTION_TILE_SIZE, (
        f"d must be {ATTENTION_TILE_SIZE}, got {head_dim}")

    # [B,S,N,d] -> kernel layout
    q_kernel = query.squeeze(0).permute(1, 2, 0)  # [N,d,S]
    k_kernel = key.squeeze(0).transpose(0, 1)     # [N,S,d]
    v_kernel = value.squeeze(0).transpose(0, 1)   # [N,S,d]

    out, k_alias, v_alias = kernel_call(
        q_kernel, *k_lead, *v_lead, k_kernel, v_kernel, k_dest, v_dest)
    out = out.reshape(batch, heads, seq_q, head_dim).transpose(1, 2)
    return out, k_alias, v_alias
