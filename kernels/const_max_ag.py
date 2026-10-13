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

"""Torch-callable wrappers for the fused all-gather + constant-max attention kernel.

Ported verbatim from the upstream kernel wrapper (const_max_ag.py)
(the reference author/improved_const_max_kernel @, with only the import path changed. Do not
reformat: keeping it byte-comparable to theirs is what makes future diffs readable.

Preconditions, all checked at the call site in wan/modules/attention.py:
  head_dim == 128                 lingbot: 5120/40 = 128
  kv_chunk % 128 == 0             no longer required -- the updated kernel masks a sub-128
                                  tail tile via last_tile_extent, so non-aligned chunk
                                  widths are supported
  total chunks <= 4               only const_max_ag_0..3 exist, because lists cannot
                                  cross the Torch->NKI boundary
  QK-RMSNorm across heads         lingbot: _tp_rms_norm normalises over full_dim=5120

softmax_max is an ASSUMED BOUND (15.0), not a measured one -- the reference implementation's config.py calls
it "safe for QK-RMSNorm with across_heads" and their own CLI help says the accuracy is
still being tuned and to distrust the pixels. the reference implementation measured const-max ON vs OFF across
6 prompt sets at 720p and found the output visually indistinguishable at 152.9 vs 209.6
ms/forward, but that is an eyeball result, not numeric parity -- still compare the mp4
against the attention_cte run at the same shape.
"""

from __future__ import annotations

import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch

from kernels.attention_const_max_allgather import attention_const_max_allgather

ATTENTION_TILE_SIZE = 128

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))

def _run(q_hbm, k_chunks, v_chunks, k_shard, v_shard, replica_group,
         softmax_max, softmax_scale, kv_chunk, k_dest, v_dest):
    """Stage around the collective and call the kernel body.

    k_dest [N, d, kv_chunk] is passed as k_last_gathered — the kernel transpose-stores
    into it (regular DMA, not a collective). V needs a private gather buffer because
    the all-gather writes v_last_gathered directly and CC can't target NEFF I/O.
    """
    n_heads, _, d = k_shard.shape

    v_last_gathered = nl.ndarray(shape=(n_heads, kv_chunk, d), dtype=q_hbm.dtype,
                                 buffer=nl.shared_hbm)

    out = attention_const_max_allgather(
        q_hbm, list(k_chunks), list(v_chunks),
        k_shard, v_shard,
        k_dest, v_last_gathered,
        replica_group, softmax_max, softmax_scale,
    )

    nisa.dma_copy(dst=v_dest, src=v_last_gathered)
    return out, k_dest, v_dest


# One fixed tensor signature per leading-chunk count. Lists cannot cross the
# Torch -> NKI boundary. Keep these wrappers explicit: generating them dynamically
# obscures signatures from the NKI tracer and has caused name-resolution failures.

@nki.jit
def const_max_ag_0(
    query, key_shard, value_shard, key_destination, value_destination,
    replica_group, softmax_max, softmax_scale, kv_chunk,
):
    return _run(
        query, (), (), key_shard, value_shard, replica_group,
        softmax_max, softmax_scale, kv_chunk, key_destination, value_destination)


@nki.jit
def const_max_ag_1(
    query, key_0, value_0, key_shard, value_shard,
    key_destination, value_destination, replica_group,
    softmax_max, softmax_scale, kv_chunk,
):
    return _run(
        query, (key_0,), (value_0,), key_shard, value_shard, replica_group,
        softmax_max, softmax_scale, kv_chunk, key_destination, value_destination)


@nki.jit
def const_max_ag_2(
    query, key_0, key_1, value_0, value_1, key_shard, value_shard,
    key_destination, value_destination, replica_group,
    softmax_max, softmax_scale, kv_chunk,
):
    return _run(
        query, (key_0, key_1), (value_0, value_1), key_shard, value_shard,
        replica_group, softmax_max, softmax_scale, kv_chunk,
        key_destination, value_destination)


@nki.jit
def const_max_ag_3(
    query, key_0, key_1, key_2, value_0, value_1, value_2,
    key_shard, value_shard, key_destination, value_destination,
    replica_group, softmax_max, softmax_scale, kv_chunk,
):
    return _run(
        query, (key_0, key_1, key_2), (value_0, value_1, value_2),
        key_shard, value_shard, replica_group, softmax_max, softmax_scale,
        kv_chunk, key_destination, value_destination)


_JITTED_AG_KERNELS_BY_LEAD_COUNT = {
    0: const_max_ag_0,
    1: const_max_ag_1,
    2: const_max_ag_2,
    3: const_max_ag_3,
}
_AG_KERNELS_BY_LEAD_COUNT = {
    lead_count: kernel[LNC]
    for lead_count, kernel in _JITTED_AG_KERNELS_BY_LEAD_COUNT.items()
}


def build_const_max_ag_kernels(replica_group, softmax_max, softmax_scale, kv_chunk):
    """Build four tensor-only callables with opaque constants closed over eagerly."""
    def bind(kern):
        def call(*tensors):
            return kern(*tensors, replica_group, softmax_max, softmax_scale, kv_chunk)
        return call

    return {
        lead_count: bind(kernel)
        for lead_count, kernel in _AG_KERNELS_BY_LEAD_COUNT.items()
    }


def ag_const_max_attention(
    query,
    key,
    value,
    k_lead,
    v_lead,
    kernel_call,
    k_dest,
    v_dest,
):
    """Run fused AG attention from the model's `[B,S,N,d]` layout.

    `k_lead/v_lead` contain 0..3 preallocated full chunks. `key/value` are this
    rank's new shard. `kernel_call` selects one of the four total-chunk forms and
    carries replica-group/max/scale constants. `k_dest/v_dest` are the final
    preallocated cache chunks and are returned as must-alias outputs.
    """
    batch, seq_q, heads, head_dim = query.shape
    assert head_dim == ATTENTION_TILE_SIZE, (
        f"d must be {ATTENTION_TILE_SIZE}, got {head_dim}")

    # Model Q/K/V are [B,S,N,d]. Pass transpose views directly to the NKI custom
    # call and let the Torch-NeuronX lowering choose the physical transformation. This
    # removes explicit copy_ operations and persistent staging buffers.
    q_kernel = query.squeeze(0).permute(1, 2, 0)  # [N,d,S]
    k_kernel = key.squeeze(0).transpose(0, 1)     # [N,S,d]
    v_kernel = value.squeeze(0).transpose(0, 1)   # [N,S,d]

    out, k_alias, v_alias = kernel_call(
        q_kernel, *k_lead, *v_lead, k_kernel, v_kernel, k_dest, v_dest)
    out = out.reshape(batch, heads, seq_q, head_dim).transpose(1, 2)
    return out, k_alias, v_alias
