"""Constant-offset self-attention for lingbot, whole-chunk loads (LINGBOT_CMAX=2): see kernels/attention_cmax_lb.py.

Same math as attention_cmax_lb (softmax(Q K^T - c) V through P @ [V | 1], fp32 accumulators), with the loads
reorganized: per (chunk, head) one copy brings the chunk's K into SBUF as [d, Sk], and two bring its V straight into
the [128 keys, tiles, d + 1] layout MM2 multiplies (the ones column set once per buffer), so no per-tile copies. The
copies read K in the CP all-gather's rank-block memory ([k_blocks, N, d, Sk / k_blocks], k_blocks > 1) and V in
token-major memory ([Sk, N, d], v_token_major) directly.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl

from kernels.attention_const_max_allgather import _EPS, _normalize, _prepare_softmax_max
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from nkilib.experimental import neurotile as nt

_P = 128
_O_TILE_SIZE = 128
_PROLOGUE = 8  # MM1 + exp run this many key tiles ahead of MM2

# LINGBOT_CMAX_DGE=hwdge_sync|hwdge_act: issue every copy of the kernel from the Sync / Scalar engine's hardware DGE
# and set the V ones column with the Vector engine, so the kernel puts no instruction on GpSimd. With the
# compiler's choice the copies went to GpSimd, whose in-order queue also carries the collectives' triggers: in the
# block profile the new chunk's V all-gather was triggered only after the cached-chunk call's GpSimd work (~713 us,
# 140 us after the K all-gather ended), with every engine idle from 720 to 893 us waiting for it.
_DGE_MODE = os.environ.get("LINGBOT_CMAX_DGE", "")
_DGE_ENGINE = {"hwdge_sync": nisa.engine.sync, "hwdge_act": nisa.engine.scalar}.get(_DGE_MODE)
assert _DGE_MODE in ("", "hwdge_sync", "hwdge_act"), f"LINGBOT_CMAX_DGE={_DGE_MODE!r}"


def _dma(dst, src):
    if _DGE_ENGINE is None:
        nisa.dma_copy(dst=dst, src=src)
    else:
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=_DGE_ENGINE)


def _memset(dst, value):
    if _DGE_ENGINE is None:
        nisa.memset(dst, value)
    else:
        nisa.memset(dst, value, engine=nisa.engine.vector)


def _add(acc_view, psum):
    """acc_view += psum (fp32)."""
    if _DGE_ENGINE is None:
        nisa.tensor_tensor(dst=acc_view, data1=acc_view, data2=psum, op=nl.add)
    else:
        nisa.tensor_tensor(dst=acc_view, data1=acc_view, data2=psum, op=nl.add, engine=nisa.engine.vector)


def _normalize_acc(o_aug):
    """[rows, d + 1] fp32 accumulator -> [rows, d] bf16 O / max(sum, eps): attention_const_max_allgather._normalize,
    on the Vector engine under LINGBOT_CMAX_DGE."""
    if _DGE_ENGINE is None:
        return _normalize(o_aug)
    rows, d = o_aug.shape[0], o_aug.shape[1] - 1
    sum_exp = nl.ndarray(shape=(rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sum_exp, data=o_aug[:, d : d + 1], op0=nl.maximum, operand0=_EPS, engine=nisa.engine.vector)
    inv = nl.ndarray(shape=(rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv, data=sum_exp)
    o_out = nl.ndarray(shape=(rows, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_out, data=o_aug[:, :d], op0=nl.multiply, operand0=inv, engine=nisa.engine.vector)
    return o_out


def _load_k(k_sb, k, head, n_heads, d, seq_k, k_blocks):
    """k_sb [d, seq_k] <- this head's keys, plain [N, d, Sk] or rank-blocked [k_blocks, N, d, Sk / k_blocks]."""
    if k_blocks == 1:
        _dma(dst=k_sb[:, 0:seq_k], src=k[head, :, :])
    else:
        lb = seq_k // k_blocks
        _dma(
            dst=k_sb.ap(pattern=[[seq_k, d], [lb, k_blocks], [1, lb]], offset=0),
            src=k.ap(pattern=[[lb, d], [n_heads * d * lb, k_blocks], [1, lb]], offset=head * d * lb),
        )


def _load_v(v_sb, v, head, n_heads, d, seq_k, n_tiles, token_major):
    """v_sb [128, n_tiles, d + 1][:, t, :d] <- keys t * 128 + partition of this head; V plain [N, Sk, d] or
    token-major memory [Sk, N, d]. Full tiles in one copy, the partial last tile in another."""
    row = n_heads * d if token_major else d  # elements between consecutive keys of one head
    base = head * d if token_major else head * seq_k * d
    n_full = seq_k // _P
    tail = seq_k - n_full * _P
    width = n_tiles * (d + 1)
    if n_full > 0:
        _dma(
            dst=v_sb.ap(pattern=[[width, _P], [d + 1, n_full], [1, d]], offset=0),
            src=v.ap(pattern=[[row, _P], [_P * row, n_full], [1, d]], offset=base),
        )
    if tail > 0:
        _dma(
            dst=v_sb.ap(pattern=[[width, tail], [1, d]], offset=n_full * (d + 1)),
            src=v.ap(pattern=[[row, tail], [1, d]], offset=base + n_full * _P * row),
        )


def _chunk_head(q_sb, k_sb, v_sb, seq_k, n_tiles, seq_q, o_psums, neg_max):
    """o_psums[q tile] = sum over the chunk's key tiles of exp(K^T Q - c)^T @ [V | 1], MM2 trailing MM1 + exp by
    _PROLOGUE tiles so the Tensor engine always has ready probabilities."""
    depth = min(_PROLOGUE, n_tiles)
    n_bufs = depth + 1
    probs = nl.ndarray(shape=(_P, n_bufs, seq_q), dtype=nl.bfloat16, buffer=nl.sbuf)
    n_o_tiles = o_psums.shape[0]

    def extent(tile):
        return min(_P, seq_k - tile * _P)

    def mm2(tile):
        buf = tile % n_bufs
        ext = extent(tile)
        for ot in range(n_o_tiles):
            rows = o_psums[ot].data.shape[0]
            nisa.nc_matmul(
                dst=o_psums[ot].data,
                stationary=probs[0:ext, buf, nl.ds(ot * _O_TILE_SIZE, rows)],
                moving=v_sb[0:ext, tile, :],
                accumulate=tile != 0,
            )

    with nl.no_reorder():
        for tile in range(n_tiles):
            ext = extent(tile)
            scores = nl.ndarray(shape=(_P, seq_q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=scores[0:ext, :], stationary=k_sb[:, nl.ds(tile * _P, ext)], moving=q_sb)
            bias = neg_max if isinstance(neg_max, (int, float)) else neg_max[0:ext, :]
            nisa.activation(dst=probs[0:ext, tile % n_bufs, :], data=scores[0:ext, :], op=nl.exp, scale=1.0,
                            bias=bias)
            if tile >= depth:
                mm2(tile - depth)
        for tile in range(n_tiles - depth, n_tiles):
            mm2(tile)


def _cmax2(q_hbm, k_chunks, v_chunks, softmax_max, acc_in=None, normalize=True, k_blocks=1, v_token_major=False):
    """[N, d, Sq] pre-scaled queries (Sq <= 512) over every chunk's keys; heads split over the logical core's
    programs. normalize: [N, Sq, d] in q's dtype, else the fp32 accumulator [N, Sq, d + 1]; acc_in continues one.
    K chunks have the logical shape [N, d, Sk] and V chunks [N, Sk, d], whatever their memory (k_blocks,
    v_token_major)."""
    n_heads, d, seq_q = q_hbm.shape
    kernel_assert(d == _P, f"d must be {_P}, got {d}")
    kernel_assert(seq_q <= nl.tile_size.gemm_moving_fmax, f"seq_q={seq_q} must be <= {nl.tile_size.gemm_moving_fmax}")
    if normalize:
        out = nl.ndarray(shape=(n_heads, seq_q, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
    else:
        out = nl.ndarray(shape=(n_heads, seq_q, d + 1), dtype=nl.float32, buffer=nl.shared_hbm)
    _, n_prgs, prg_id = get_verified_program_sharding_info("attention_cmax2_lb", max_sharding=2)
    kernel_assert(n_heads % n_prgs == 0, f"{n_heads=} must be divisible by {n_prgs=}")
    per = n_heads // n_prgs
    h0 = prg_id * per
    neg_max = _prepare_softmax_max(softmax_max)
    n_o_tiles = div_ceil(seq_q, _O_TILE_SIZE)

    q_sbs = []
    for head in range(per):
        q_sb = nl.ndarray(shape=(d, seq_q), dtype=nl.bfloat16, buffer=nl.sbuf)
        _dma(dst=q_sb, src=q_hbm[h0 + head, :, :])
        q_sbs.append(q_sb)

    acc = nl.ndarray(shape=(_O_TILE_SIZE, per, n_o_tiles, d + 1), dtype=nl.float32, buffer=nl.sbuf)
    if acc_in is not None:
        for head in range(per):
            for ot in range(n_o_tiles):
                rows = min(_O_TILE_SIZE, seq_q - ot * _O_TILE_SIZE)
                _dma(dst=acc[:rows, head, ot, :], src=acc_in[h0 + head, nl.ds(ot * _O_TILE_SIZE, rows), :])

    for chunk_idx in range(len(k_chunks)):
        k, v = k_chunks[chunk_idx], v_chunks[chunk_idx]
        seq_k = k.shape[2]
        kernel_assert(seq_k % k_blocks == 0, f"k_blocks={k_blocks} must divide the chunk's {seq_k} keys")
        n_tiles = div_ceil(seq_k, _P)
        for head in range(per):
            k_sb = nl.ndarray(shape=(d, seq_k), dtype=nl.bfloat16, buffer=nl.sbuf)
            v_sb = nl.ndarray(shape=(_P, n_tiles, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            _memset(v_sb[:, :, d : d + 1], 1.0)
            _load_k(k_sb, k, h0 + head, n_heads, d, seq_k, k_blocks)
            _load_v(v_sb, v, h0 + head, n_heads, d, seq_k, n_tiles, v_token_major)
            o_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(seq_q, d + 1))
            _chunk_head(q_sbs[head], k_sb, v_sb, seq_k, n_tiles, seq_q, o_psums, neg_max)
            for ot in range(n_o_tiles):
                rows = o_psums[ot].data.shape[0]
                view = acc[:rows, head, ot, :]
                if chunk_idx == 0 and acc_in is None:
                    nisa.tensor_copy(dst=view, src=o_psums[ot].data, engine=nisa.engine.vector)
                else:
                    _add(view, o_psums[ot].data)

    for head in range(per):
        for ot in range(n_o_tiles):
            rows = min(_O_TILE_SIZE, seq_q - ot * _O_TILE_SIZE)
            dst = out[h0 + head, nl.ds(ot * _O_TILE_SIZE, rows), :]
            if normalize:
                _dma(dst=dst, src=_normalize_acc(acc[:rows, head, ot, :]))
            else:
                _dma(dst=dst, src=acc[:rows, head, ot, :])
    return out


# One fixed signature per chunk count and mode (lists cannot cross the torch -> NKI boundary); k_blocks and
# v_token_major are compile-time ints / bools.
@nki.jit
def cmax2_full_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0], [v0], softmax_max, k_blocks=k_blocks, v_token_major=v_token_major)


@nki.jit
def cmax2_full_4(q, k0, k1, k2, k3, v0, v1, v2, v3, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0, k1, k2, k3], [v0, v1, v2, v3], softmax_max, k_blocks=k_blocks, v_token_major=v_token_major)


@nki.jit
def cmax2_part_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0], [v0], softmax_max, normalize=False, k_blocks=k_blocks, v_token_major=v_token_major)


@nki.jit
def cmax2_part_2(q, k0, k1, v0, v1, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0, k1], [v0, v1], softmax_max, normalize=False, k_blocks=k_blocks,
                  v_token_major=v_token_major)


@nki.jit
def cmax2_part_3(q, k0, k1, k2, v0, v1, v2, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0, k1, k2], [v0, v1, v2], softmax_max, normalize=False, k_blocks=k_blocks,
                  v_token_major=v_token_major)


@nki.jit
def cmax2_final_1(q, k0, v0, acc, softmax_max, k_blocks, v_token_major):
    return _cmax2(q, [k0], [v0], softmax_max, acc_in=acc, k_blocks=k_blocks, v_token_major=v_token_major)
