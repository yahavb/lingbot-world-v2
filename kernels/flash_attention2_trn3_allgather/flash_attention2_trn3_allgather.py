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

"""All-gather FlashAttention-2 forward: ``O = softmax(Q @ K^T * scale) @ V``.

Online-softmax variant of ``attention_const_max_allgather``. The running max, sum and numerator
are rescaled across chunks, so chunks are the innermost loop and only one Sq block's state is
resident at a time.

Two deviations from the nkilib version (both required by the chunked-KV call pattern):

1. **Collectives hoisted when ``num_chunks < 3``** — upstream fires the all-gather inside chunk 1's
   steady state and reads it at the last chunk, which requires >= 3 chunks. Callers use 1..4
   chunks as the KV window fills.

2. **Q/O sharded on Sq only, never on heads** — head sharding caused corruption at TP=4 because
   both LNC cores must issue identical collectives on whole tensors.
"""

from typing import Callable, Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.allocator import sizeinbytes
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import is_trn3_b1
from nkilib.experimental import neurotile as nt

from .config import D_TILE, SQ_TILE, SV_TILE, FlashAttentionConfig, FlashAttentionGeometry
from nki.compiler import CompileConfig

from .flash_attention_fwd import FlashAttentionFwd
from .remainder_stream import RemainderBlockStream, remainder_block_stream

# Below this many total chunks the all-gather cannot be overlapped into the chunk loop -- it would be
# issued at or after the chunk that consumes it. See deviation 1 in the module docstring.
_MIN_CHUNKS_TO_OVERLAP_COLLECTIVES = 3


class _SyncHwdgeStream:
    """Wraps a :class:`RemainderBlockStream` so every load uses hardware DGE on the sync engine.

    ``load`` ignores the ``dge_mode`` / ``engine`` the caller passes and forces
    ``dge_mode=nisa.dge_mode.hwdge`` on ``nisa.engine.sync``.
    """

    def __init__(self, stream: RemainderBlockStream):
        self._stream = stream

    def load(self, index: int, dge_mode=None, engine=None):
        return self._stream.load(index, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


class _KCacheStoreStream:
    """Transposed K stream that also writes each loaded block into the D-major ``[D, Sk_i]`` cache."""

    def __init__(self, stream: RemainderBlockStream, dst: RemainderBlockStream):
        self._stream = stream
        self._dst = dst

    def load(self, index: int, dge_mode=None, engine=None):
        loaded = self._stream.load(index, dge_mode=dge_mode, engine=engine)
        self._dst.store(index, loaded[0].data, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        return loaded


@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def flash_attention2_trn3_allgather(
    q: nl.NkiTensor,
    k_chunks: list,
    v_chunks: list,
    k_last: nl.NkiTensor,
    v_last: nl.NkiTensor,
    k_last_gathered: nl.NkiTensor,
    v_last_gathered: nl.NkiTensor,
    replica_group: ncc.ReplicaGroup,
    scale: float,
    config: Optional[FlashAttentionConfig] = None,
) -> nl.NkiTensor:
    """FlashAttention-2 forward over gathered + all-gathered K/V.

    ``k_chunks`` / ``v_chunks`` are already-gathered (full) K/V chunks. ``k_last`` / ``v_last`` are this
    rank's shard of one more chunk: at the start of the kernel a per-head all-gather reconstructs that
    chunk across ranks (written into the ``k_last_gathered`` / ``v_last_gathered`` output buffers) and
    appends it to the chunk list. Unmasked bidirectional online-softmax attention is then computed over
    all chunks.

    Dimensions:
        B: batch x heads, collapsed into one axis (Q and K/V share it -- no GQA).
        D: head dimension. A positive multiple of 128, at most 512.
        Sq: query sequence length. Arbitrary.
        Sk_i: per-chunk key/value sequence length. Equal across chunks.

    Args:
        q (nl.NkiTensor): ``[B, D, Sq]`` bf16 @ HBM.
        k_chunks (list): already-gathered Key chunks, each ``[B, D, Sk_i]`` bf16 @ HBM.
        v_chunks (list): already-gathered Value chunks, each ``[B, Sk_i, D]`` bf16 @ HBM.
        k_last (nl.NkiTensor): ``[B, Sk_local, D]`` bf16 @ HBM, this rank's Key shard.
        v_last (nl.NkiTensor): ``[B, Sk_local, D]`` bf16 @ HBM, this rank's Value shard.
        k_last_gathered (nl.NkiTensor): ``[B, D, Sk_i]`` bf16 @ HBM output buffer -- the D-major
            (``[N, d, S]``) cache the all-gathered last chunk is persisted into.
        v_last_gathered (nl.NkiTensor): ``[B, Sk_i, D]`` bf16 @ HBM output buffer for gathered ``v_last``.
        replica_group (ReplicaGroup): ranks participating in the ``k_last`` / ``v_last`` all-gather.
        scale (float): attention-score scale.
        config (Optional[FlashAttentionConfig]): tiling configuration; its ``Sk`` is the per-chunk
            ``Sk_i``, not the total. Derived from ``D`` / ``Sq`` / ``Sk_i`` if omitted.

    Returns:
        nl.NkiTensor: ``[B, Sq, D]`` bf16 @ shared HBM, normalized attention output.

    Notes:
        * **Trn3 B1 only.** ``nisa.exponential`` is a VectorE instruction absent on Trn3 A0 and earlier.
        * All chunks (including the gathered shard) must share ``Sk_i``: one ``FlashAttentionFwd``
          instance sizes its buffers and walk from a single ``config``, and reusing it per chunk
          requires identical tiling geometry.
        * Compile with ``--internal-skip-backend-allocation-opt-nki`` (the primitive hand-places PSUM).

    Pseudocode:
        all_gather(k_last -> k_last_gathered_sd, v_last -> v_last_gathered)   # reconstruct last chunk
        chunks = gathered_chunks + [k_last_gathered_sd, v_last_gathered]

        for each head:
            state[head] = fwd.reset_state(); fwd.get_state()   # running max / sum / out = -inf, 0, 0

        for each chunk:                                        # chunk-outer: one chunk across all heads
            for each head:
                fwd.set_state(state[head])                     # restore this head's running state
                fwd.run_pipeline(Q_block, K_chunk, V_chunk)    # online-softmax fold, no reset/normalize
                state[head] = fwd.get_state()                  # save it back

        for each head:
            fwd.set_state(state[head]); fwd.normalize(O_block) # running_out / running_sum
    """
    # nisa.exponential is VectorE and needs gen4 sub-version 1; trn3 A0 (sub-version 0) lacks it.
    kernel_assert(
        is_trn3_b1(),
        "flash_attention2_trn3_allgather requires Trn3 B1 (NeuronCore-v4 sub-version 1) for nisa.exponential",
    )
    B, D, Sq = q.shape

    kernel_assert(
        k_last_gathered.shape[1] == D, f"k_last_gathered head dim {k_last_gathered.shape[1]} must match q D={D}"
    )
    Sk_i = k_last_gathered.shape[2]
    k_last_gathered_sd = nl.ndarray((B, Sk_i, D), dtype=k_last.dtype, buffer=nl.shared_hbm)

    k_chunks = k_chunks + [k_last_gathered_sd]
    v_chunks = v_chunks + [v_last_gathered]

    num_chunks = len(k_chunks)
    kernel_assert(num_chunks == len(v_chunks), f"K/V chunk count mismatch: {num_chunks} vs {len(v_chunks)}")
    # The all-gathered last chunk is [B, Sk_i, D]; the earlier gathered chunks are D-major [B, D, Sk_i].
    for chunk_idx in range(num_chunks):
        if chunk_idx == num_chunks - 1:
            Bk, Sk_chunk, d_k = k_chunks[chunk_idx].shape
        else:
            Bk, d_k, Sk_chunk = k_chunks[chunk_idx].shape
        Bv, Sv_chunk, d_v = v_chunks[chunk_idx].shape
        kernel_assert(Bk == B and Bv == B, f"chunk {chunk_idx} batch must match Q, got {Bk=}, {Bv=}, {B=}")
        kernel_assert(d_k == D and d_v == D, f"chunk {chunk_idx} head dim must match Q, got {d_k=}, {d_v=}, {D=}")
        kernel_assert(Sk_chunk == Sk_i and Sv_chunk == Sk_i, f"all chunks must share Sk_i={Sk_i}, chunk {chunk_idx}")

    if config == None:
        config = FlashAttentionConfig()
    # Resolve the tuning knobs against this shape; config.Sk is the per-chunk Sk_i.
    geometry = config.geometry(D=D, Sq=Sq, Sk=Sk_i)
    # dma_transpose on the last chunk's K load requires a 2-byte element.
    kernel_assert(sizeinbytes(k_chunks[0].dtype) == 2, "k must be a 2-byte dtype for the transposed load")

    out = nl.ndarray(shape=(B, Sq, D), dtype=q.dtype, buffer=nl.shared_hbm)

    # Shard the independent (B, Sq) work across LNC cores; K/V chunks stay replicated.
    q_local, o_local, k_chunks_local, v_chunks_local, k_cache_local = _shard_inputs(
        q, k_chunks, v_chunks, out, k_last_gathered
    )

    def start_head_collectives(head_idx: int) -> None:
        _allgather_head(head_idx, k_last, k_last_gathered_sd, v_last, v_last_gathered, replica_group)

    _attention_fa2_compute(
        q_local, k_chunks_local, v_chunks_local, o_local, k_cache_local, scale, geometry, start_head_collectives
    )

    return out


def _allgather_head(
    head_idx: int,
    k_shard: nl.NkiTensor,
    k_out: nl.NkiTensor,
    v_shard: nl.NkiTensor,
    v_out: nl.NkiTensor,
    replica_group: ncc.ReplicaGroup,
) -> None:
    """All-gather one head's K and V shards into the gathered output buffers, K then V."""
    ncc.all_gather(dsts=[k_out[head_idx]], srcs=[k_shard[head_idx]], replica_group=replica_group, collective_dim=0)
    ncc.all_gather(dsts=[v_out[head_idx]], srcs=[v_shard[head_idx]], replica_group=replica_group, collective_dim=0)


def _shard_inputs(q: nl.NkiTensor, k_chunks: list, v_chunks: list, out: nl.NkiTensor, k_cache: nl.NkiTensor) -> tuple:
    """Shard Q and O on Sq across LNC cores; K/V chunks and the K cache stay whole on every core.

    Both B (heads) and Sq are fully concurrent, so either could carry the split. Sq is used
    unconditionally -- even when uneven -- because it leaves every core's view of K/V, of the
    collective sources/destinations, and of the K-cache store identical. Head sharding instead gives
    each core its own slice of those, which is what corrupted TP=4 in the const-max kernel; here it
    would additionally leave the upper heads un-gathered, because the per-head collective loop is
    driven by the *local* head count while indexing the *whole* K/V tensors.

    Uneven splits are fine -- the last core takes the remainder and the Q/O/KV tiling all handle a
    partial final tile.

    Returns:
        (q_local, o_local, k_chunks_local, v_chunks_local, k_cache_local) -- sliced views for this core.
    """
    _, _, Sq = q.shape
    num_cores = nl.num_programs()
    core_id = nl.program_id(0)

    kernel_assert(Sq >= num_cores, f"Sq ({Sq}) must be at least num_cores ({num_cores}) to shard on Sq")

    sq_per_core = Sq // num_cores
    sq_start = core_id * sq_per_core
    # Last core takes the remainder
    sq_end = Sq if (core_id == num_cores - 1) else (sq_start + sq_per_core)

    q_local = q[:, :, sq_start:sq_end]
    o_local = out[:, sq_start:sq_end, :]
    k_chunks_local = k_chunks
    v_chunks_local = v_chunks
    k_cache_local = k_cache

    return q_local, o_local, k_chunks_local, v_chunks_local, k_cache_local


def _attention_fa2_compute(
    q: nl.NkiTensor,
    k_chunks: list,
    v_chunks: list,
    out: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    scale: float,
    geometry: FlashAttentionGeometry,
    start_head_collectives: Optional[Callable[[int], None]] = None,
) -> None:
    """Core attention -- agnostic to LNC sharding. Operates on this core's local Q/O slice.

    Args:
        q: ``[B, D, Sq_local]`` bf16 @ HBM.
        k_chunks: list of K chunks; gathered chunks are D-major ``[B, D, Sk_i]``, the final
            all-gathered chunk is ``[B, Sk_i, D]`` (transposed on load). All bf16 @ HBM.
        v_chunks: list of V chunks, each ``[B, Sk_i, D]`` bf16 @ HBM.
        out: ``[B, Sq_local, D]`` bf16 @ HBM (output).
        k_cache: ``[B, D, Sk_i]`` bf16 @ HBM, the D-major cache the last (all-gathered) chunk is
            persisted into as it streams through the transposed load.
        scale: attention-score scale.
        geometry: tiling/pipeline geometry resolved against the shape; ``geometry.Sk`` is the per-chunk Sk_i.
        start_head_collectives: optional callable ``(head_idx) -> None`` that issues one head's
            all-gather. With at least ``_MIN_CHUNKS_TO_OVERLAP_COLLECTIVES`` chunks it is invoked at
            the start of each head's second chunk, inside the pipeline's steady-state
            ``nl.no_reorder()`` block, so the collective overlaps compute. With fewer chunks it is
            invoked for every head before the chunk loop -- see deviation 1 in the module docstring.
    """
    B, D, _ = q.shape
    num_chunks = len(k_chunks)
    fwd = FlashAttentionFwd(scale=scale, geometry=geometry)

    # Too few chunks to hide the collective behind chunk 1's compute: issue everything up front,
    # before any chunk (and therefore before the gathered chunk) is streamed.
    hoist_collectives = start_head_collectives != None and num_chunks < _MIN_CHUNKS_TO_OVERLAP_COLLECTIVES
    if hoist_collectives:
        for head_idx in range(B):
            start_head_collectives(head_idx)

    # Chunk-outer walk within each Sq block: one chunk is streamed across every head before the next,
    # so each (head, Sq block) carries its online-softmax state between chunks via an opaque HBM
    # FlashAttentionState token. Sq is split into blocks on the outside (Sq may be sharded across LNC
    # cores, so derive the block count from the local output); K/V chunks are re-streamed per Sq block.
    # The all-gather and the K-cache store run only on the first Sq block -- their results persist in
    # the gathered / cache HBM buffers for the remaining blocks.
    num_sq_blks = nt.blocks(out[0], tile_size=(SQ_TILE, D), block_size=(geometry.sq_tiles_per_blk, 1)).shape[0]
    for sq_blk_idx in range(num_sq_blks):
        q_blocks = []
        o_sbufs = []
        o_hbm_blocks = []
        states = []
        for head_idx in range(B):
            o_blks = nt.blocks(out[head_idx], tile_size=(SQ_TILE, D), block_size=(geometry.sq_tiles_per_blk, 1))
            o_hbm_blk = o_blks[sq_blk_idx, 0]
            o_sbuf = nt.alloc_tiles(
                tile_size=(SQ_TILE, D), grid=(o_hbm_blk.shape[0], 1), buffer_type=nl.sbuf, dtype=out.dtype
            )

            # MM1 contracts over D, so Q arrives with D on the partition axis.
            q_blks = nt.blocks(
                q[head_idx],
                tile_size=(D_TILE, SQ_TILE),
                block_size=(geometry.d_tiles, geometry.sq_tiles_per_blk),
            )
            q_src_blk = q_blks[0, sq_blk_idx]
            q_block = nt.alloc_tiles(
                tile_size=(D_TILE, SQ_TILE),
                grid=(q_src_blk.shape[0], q_src_blk.shape[1]),
                buffer_type=nl.sbuf,
                dtype=q.dtype,
            )
            # Sq is padded up to a whole tile in SBUF and narrowed only here and at the store.
            for d_tile_idx in range(q_src_blk.shape[0]):
                for sq_tile_idx in range(q_src_blk.shape[1]):
                    sq_cols = q_src_blk[d_tile_idx, sq_tile_idx].element_shape[1]
                    q_src_blk[d_tile_idx, sq_tile_idx].load(
                        dst=q_block[d_tile_idx, sq_tile_idx].data[:, 0:sq_cols],
                    )

            fwd.reset_state()
            states.append(fwd.get_state())  # allocate this head's state token once, seeded with the reset state
            q_blocks.append(q_block)
            o_sbufs.append(o_sbuf)
            o_hbm_blocks.append(o_hbm_blk)

        for chunk_idx in range(num_chunks):
            for head_idx in range(B):
                # K and V are cut at the same SV_TILE granule: V maps Sk on partitions, where a block
                # mixing whole and partial tiles is not allowed.
                if chunk_idx == num_chunks - 1:
                    # All-gathered last chunk is [Sk_i, D]: transpose on load to put D on partitions.
                    k_stream = remainder_block_stream(
                        k_chunks[chunk_idx][head_idx],
                        block_size=(geometry.sk_tiles_per_blk, geometry.d_tiles),
                        tile_size=(geometry.sk_tile, D_TILE),
                        dim=0,
                        buffer_count=geometry.kv_buffer_count,
                        align=SV_TILE,
                        transpose=True,
                    )
                    # Persist the gathered last chunk into the D-major cache once, on the first Sq block.
                    if sq_blk_idx == 0:
                        k_dst = remainder_block_stream(
                            k_cache[head_idx],
                            block_size=(geometry.d_tiles, geometry.sk_tiles_per_blk),
                            tile_size=(D_TILE, geometry.sk_tile),
                            dim=1,
                            buffer_count=geometry.kv_buffer_count,
                            align=SV_TILE,
                        )
                        k_stream = _KCacheStoreStream(k_stream, k_dst)
                else:
                    # Gathered chunks are already D-major [D, Sk_i]: load without a transpose, cut along Sk.
                    k_stream = remainder_block_stream(
                        k_chunks[chunk_idx][head_idx],
                        block_size=(geometry.d_tiles, geometry.sk_tiles_per_blk),
                        tile_size=(D_TILE, geometry.sk_tile),
                        dim=1,
                        buffer_count=geometry.kv_buffer_count,
                        align=SV_TILE,
                    )
                v_stream = _SyncHwdgeStream(
                    remainder_block_stream(
                        v_chunks[chunk_idx][head_idx],
                        block_size=(geometry.sk_blk // SV_TILE, 1),
                        tile_size=(SV_TILE, geometry.D),
                        dim=0,
                        buffer_count=geometry.kv_buffer_count,
                        align=SV_TILE,
                    )
                )
                # At a head's second chunk on the first Sq block, kick off its collectives inside the
                # pipeline's steady state. Later Sq blocks reuse the already-gathered chunk. Skipped
                # entirely when the collectives were hoisted above.
                callback = None
                if not hoist_collectives and chunk_idx == 1 and start_head_collectives != None and sq_blk_idx == 0:
                    callback = _bind_head(start_head_collectives, head_idx)
                # Restore this head's running state, fold the chunk in, save it back into the same token.
                fwd.set_state(states[head_idx])
                fwd.run_pipeline(q_block=q_blocks[head_idx], k_stream=k_stream, v_stream=v_stream, callback=callback)
                fwd.get_state(states[head_idx])

        for head_idx in range(B):
            fwd.set_state(states[head_idx])
            fwd.normalize(o_sbufs[head_idx])
            o_hbm_blk = o_hbm_blocks[head_idx]
            # Per tile, because a short final Sq tile stores fewer rows than o_sbuf's uniform tile has.
            for o_tile_idx in range(o_hbm_blk.shape[0]):
                o_hbm_tile = o_hbm_blk[o_tile_idx, 0]
                o_hbm_tile.store(
                    o_sbufs[head_idx][o_tile_idx, 0].data[0 : o_hbm_tile.element_shape[0], :],
                )


def _bind_head(fn: Callable[[int], None], head_idx: int) -> Callable[[], None]:
    """Close ``head_idx`` over ``fn`` eagerly, giving the zero-arg callback the pipeline wants.

    A closure rather than ``functools.partial``: the NKI tracer resolves the callback by walking the
    Python call, and a plain nested ``def`` keeps that walk on ordinary function objects.
    """

    def _run() -> None:
        fn(head_idx)

    return _run
