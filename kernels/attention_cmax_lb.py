"""Constant-offset self-attention for lingbot (LINGBOT_CMAX): softmax(Q K^T - c) V with a fixed offset c per call.

The compute follows kernels/attention_const_max_allgather.py (K^T Q scores, the exponentials written straight into
the layout of P @ [V | 1], whose last column accumulates the row sums: no row-max reduction and no transpose of the
probabilities), without its collectives and without its clip of the scores at 70, which would change rows whose
scaled scores exceed it. Q is pre-scaled [N, d, Sq], K chunks are [N, d, Sk_i] and V chunks [N, Sk_i, d], in HBM.

Exact softmax attention up to rounding as long as every row max m satisfies m - c < ~78 (no overflow of the
exponentials or of their fp32 sums over the window) and m - c > ~-85 (no underflow of the row's probabilities), so c
comes from row-max ranges measured per layer (LINGBOT_ROWMAX_GRAPH). Layers whose range does not fit stay exact.

A call either normalizes its result ([N, Sq, d], q's dtype) or returns the accumulator, fp32 [N, Sq, d + 1]
(unnormalized output and row sums), which a later call over other keys continues (acc_in) before normalizing.
"""
import nki
import nki.isa as nisa
import nki.language as nl

from kernels.attention_const_max_allgather import (
    _KV_TILES_PER_BLOCK,
    _KVBlockStream,
    _QBlockLoaderHeadMajor,
    _StreamWrapper,
    _normalize,
    _prepare_softmax_max,
)
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from nkilib.experimental import neurotile as nt

_P = 128
_O_TILE_SIZE = 128
_MM1_BUFFER_BANKS = 2
_MAX_O_AUG_BANKS_CAP = 8


def _core(k_stream, v_stream, q_block, o_aug_psums, neg_max):
    """O_aug[q, :] += sum over the stream's keys k of exp(s[k, q] - c) * [V[k, :], 1] for one query block, pipelined
    as upstream (MM1 and exp run prologue_depth tiles ahead of MM2); the exp reads the scores from PSUM."""
    d = _P
    num_blocks = k_stream.num_blocks
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]
    total_tiles = k_stream.num_tiles
    last_tile_idx = total_tiles - 1
    last_tile_extent = k_stream.last_tile_extent
    prologue_depth = min(8, total_tiles)
    num_bufs = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_bufs, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)
    probs_all = nl.ndarray(shape=(_P, num_bufs, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)
    kv_tile_idx = 0
    for block_idx in range(num_blocks):
        k_block = k_stream.load(block_idx)[0]
        v_block = v_stream.load(block_idx)
        tiles_in_block = k_block.shape[0]
        with nl.no_reorder():
            for local_tile in range(tiles_in_block):
                buf = kv_tile_idx % num_bufs
                tile_extent = last_tile_extent if kv_tile_idx == last_tile_idx else _P
                nisa.tensor_copy(
                    dst=v_aug_all[0:tile_extent, buf, :d],
                    src=v_block[local_tile, 0].data[0:tile_extent, :],
                    engine=nisa.engine.vector,
                )
                scores = nl.ndarray(shape=(_P, sq_block_size), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=scores[0:tile_extent, :],
                    stationary=k_block[local_tile].data[:, 0:tile_extent],
                    moving=q_block,
                )
                nisa.activation(
                    dst=probs_all[0:tile_extent, buf, :],
                    data=scores[0:tile_extent, :],
                    op=nl.exp,
                    scale=1.0,
                    bias=neg_max,
                )
                if kv_tile_idx >= prologue_depth:
                    consume_idx = kv_tile_idx - prologue_depth
                    cbuf = consume_idx % num_bufs
                    extent = last_tile_extent if consume_idx == last_tile_idx else _P
                    for ot_idx in range(num_output_tiles):
                        rows = o_aug_psums[ot_idx].data.shape[0]
                        nisa.nc_matmul(
                            dst=o_aug_psums[ot_idx].data,
                            stationary=probs_all[0:extent, cbuf, nl.ds(ot_idx * _O_TILE_SIZE, rows)],
                            moving=v_aug_all[0:extent, cbuf, :],
                            accumulate=consume_idx != 0,
                        )
                kv_tile_idx += 1
    kernel_assert(kv_tile_idx == total_tiles, f"stream yielded {kv_tile_idx} KV tiles, expected {total_tiles}")
    with nl.no_reorder():
        for i in range(prologue_depth):
            drain_idx = total_tiles - prologue_depth + i
            dbuf = drain_idx % num_bufs
            extent = last_tile_extent if drain_idx == last_tile_idx else _P
            for ot_idx in range(num_output_tiles):
                rows = o_aug_psums[ot_idx].data.shape[0]
                nisa.nc_matmul(
                    dst=o_aug_psums[ot_idx].data,
                    stationary=probs_all[0:extent, dbuf, nl.ds(ot_idx * _O_TILE_SIZE, rows)],
                    moving=v_aug_all[0:extent, dbuf, :],
                    accumulate=drain_idx != 0,
                )


class _KStream(_StreamWrapper):
    """K stream of a pre-transposed [d, Sk] chunk (the upstream stream wrapper, hardware DGE on the sync engine)."""


def _cmax(q_hbm, k_chunks, v_chunks, softmax_max, acc_in=None, normalize=True):
    """[N, d, Sq] pre-scaled queries over every chunk's keys; heads split over the logical core's programs.
    normalize: [N, Sq, d] in q's dtype, else the fp32 accumulator [N, Sq, d + 1]. acc_in: an earlier call's
    accumulator over other keys, continued here."""
    n_heads, d, seq_q = q_hbm.shape
    kernel_assert(d == _P, f"d must be {_P}, got {d}")
    if normalize:
        out = nl.ndarray(shape=(n_heads, seq_q, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
    else:
        out = nl.ndarray(shape=(n_heads, seq_q, d + 1), dtype=nl.float32, buffer=nl.shared_hbm)
    _, n_prgs, prg_id = get_verified_program_sharding_info("attention_cmax_lb", max_sharding=2)
    kernel_assert(n_heads % n_prgs == 0, f"{n_heads=} must be divisible by {n_prgs=}")
    per = n_heads // n_prgs
    h0 = prg_id * per
    neg_max = _prepare_softmax_max(softmax_max)

    max_o_aug_banks = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, _MAX_O_AUG_BANKS_CAP)
    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block = max_o_aug_banks * _O_TILE_SIZE if has_bf16_psum else nl.tile_size.gemm_moving_fmax
    o_tiles_per_block = max_sq_block // _O_TILE_SIZE
    num_o_tiles = div_ceil(seq_q, _O_TILE_SIZE)

    q_blocks_by_head = []
    for head in range(per):
        loader = _QBlockLoaderHeadMajor(q_hbm[h0 + head], max_sq_block_size=max_sq_block, softmax_scale=None)
        q_blocks_by_head.append([loader.load(block_idx) for block_idx in range(loader.num_blocks)])
    num_q_blocks = len(q_blocks_by_head[0])

    acc = nl.ndarray(shape=(_O_TILE_SIZE, per, num_o_tiles, d + 1), dtype=nl.float32, buffer=nl.sbuf)
    if acc_in is not None:
        for head in range(per):
            for o_tile in range(num_o_tiles):
                rows = min(_O_TILE_SIZE, seq_q - o_tile * _O_TILE_SIZE)
                nisa.dma_copy(
                    dst=acc[:rows, head, o_tile, :],
                    src=acc_in[h0 + head, nl.ds(o_tile * _O_TILE_SIZE, rows), :],
                )

    for chunk_idx in range(len(k_chunks)):
        for head in range(per):
            k_stream = _KStream(
                _KVBlockStream(
                    k_chunks[chunk_idx][h0 + head], block_size=(1, _KV_TILES_PER_BLOCK), tiling_dim=1, transpose=False
                )
            )
            v_stream = _StreamWrapper(
                _KVBlockStream(v_chunks[chunk_idx][h0 + head], block_size=(_KV_TILES_PER_BLOCK, 1), tiling_dim=0)
            )
            for q_block_idx in range(num_q_blocks):
                q_block, sq_block_size = q_blocks_by_head[head][q_block_idx]
                num_output_tiles = div_ceil(sq_block_size, _O_TILE_SIZE)
                kernel_assert(
                    num_output_tiles <= max_o_aug_banks,
                    f"Q block too large: {num_output_tiles} O_aug tiles, {max_o_aug_banks} PSUM banks",
                )
                o_aug_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(sq_block_size, d + 1))
                _core(k_stream, v_stream, q_block, o_aug_psums, neg_max)
                for ot_idx in range(num_output_tiles):
                    rows = o_aug_psums[ot_idx].data.shape[0]
                    o_tile = q_block_idx * o_tiles_per_block + ot_idx
                    view = acc[:rows, head, o_tile, :]
                    if chunk_idx == 0 and acc_in is None:
                        nisa.tensor_copy(dst=view, src=o_aug_psums[ot_idx].data, engine=nisa.engine.vector)
                    else:
                        nisa.tensor_tensor(dst=view, data1=view, data2=o_aug_psums[ot_idx].data, op=nl.add)

    for head in range(per):
        for o_tile in range(num_o_tiles):
            rows = min(_O_TILE_SIZE, seq_q - o_tile * _O_TILE_SIZE)
            if normalize:
                o_sb = _normalize(acc[:rows, head, o_tile, :])
                nisa.dma_copy(dst=out[h0 + head, nl.ds(o_tile * _O_TILE_SIZE, rows), :], src=o_sb)
            else:
                nisa.dma_copy(dst=out[h0 + head, nl.ds(o_tile * _O_TILE_SIZE, rows), :], src=acc[:rows, head, o_tile, :])
    return out


# One fixed signature per chunk count and mode (lists cannot cross the torch -> NKI boundary).
@nki.jit
def attention_cmax_1(q, k0, v0, softmax_max):
    return _cmax(q, [k0], [v0], softmax_max)


@nki.jit
def attention_cmax_2(q, k0, k1, v0, v1, softmax_max):
    return _cmax(q, [k0, k1], [v0, v1], softmax_max)


@nki.jit
def attention_cmax_4(q, k0, k1, k2, k3, v0, v1, v2, v3, softmax_max):
    return _cmax(q, [k0, k1, k2, k3], [v0, v1, v2, v3], softmax_max)


@nki.jit
def attention_cmax_part_1(q, k0, v0, softmax_max):
    return _cmax(q, [k0], [v0], softmax_max, normalize=False)


@nki.jit
def attention_cmax_part_2(q, k0, k1, v0, v1, softmax_max):
    return _cmax(q, [k0, k1], [v0, v1], softmax_max, normalize=False)


@nki.jit
def attention_cmax_part_3(q, k0, k1, k2, v0, v1, v2, softmax_max):
    return _cmax(q, [k0, k1, k2], [v0, v1, v2], softmax_max, normalize=False)


@nki.jit
def attention_cmax_final_1(q, k0, v0, acc, softmax_max):
    return _cmax(q, [k0], [v0], softmax_max, acc_in=acc)
