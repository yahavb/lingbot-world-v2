"""Constant-offset self-attention for lingbot, whole-chunk loads (LINGBOT_CMAX=2): see kernels/attention_cmax_lb.py.

Same math as attention_cmax_lb (softmax(Q K^T - c) V through P @ [V | 1], fp32 accumulators), with the loads
reorganized: per (chunk, head) one copy brings the chunk's K into SBUF as [d, Sk], and two bring its V straight into
the [128 keys, tiles, d + 1] layout MM2 multiplies (the ones column set once per buffer), so no per-tile copies. The
copies read K in the CP all-gather's rank-block memory ([k_blocks, N, d, Sk / k_blocks], k_blocks > 1) and V in
token-major memory ([Sk, N, d], v_token_major) directly. More than 384 queries (CP8: 544) run as several passes over
each (chunk, head)'s SBUF-resident K and V, with the PSUM tiles pinned (see _PSUM_PIN_ENV).

The guarded entry points (*_g*, LINGBOT_CMAX_GUARD) check their own result: see _guard_check and _redo_block.
"""
import math
import os

import nki
import nki.isa as nisa
import nki.language as nl

from kernels.attention_const_max_allgather import _EPS, _normalize, _prepare_softmax_max
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE, div_ceil, get_verified_program_sharding_info
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


_PASS_MAX = 384  # queries per pass over a chunk's keys: at most 3 output tiles (CP16 runs 288 in one pass)
_MAX_Q = 640  # 5 output tiles, all on pinned banks 0-4
# PSUM placement. With more than one pass (CP8: 544 queries) every PSUM tile is pinned to a fixed bank: the passes'
# output tiles on banks 0-4, the MM1 scores alternating on banks 5 and 6, bank 7 unused. With compiler-managed
# placement the 544-query kernel was exact on one core but wrote inf / NaN on random ranks of the 32-rank block,
# deterministically for a given graph; pinned it is exact on all 32 ranks. One pass (CP16: 288) keeps compiler
# placement, the configuration validated there. LINGBOT_CMAX_PSUM_PIN=0 / 1 forces either.
_PSUM_PIN_ENV = os.environ.get("LINGBOT_CMAX_PSUM_PIN", "")
assert _PSUM_PIN_ENV in ("", "0", "1"), f"LINGBOT_CMAX_PSUM_PIN={_PSUM_PIN_ENV!r}"
_SCORE_BANKS = (5, 6)


def _query_passes(seq_q):
    """(first query, count) of each pass over a chunk's key tiles: one pass up to _PASS_MAX queries, else passes
    of 256 (two whole output tiles) and a last one of at most _PASS_MAX, e.g. 544 (CP8) = 256 + 288. MM1 streams
    a pass's queries as one moving operand (at most 512 columns, one fp32 PSUM bank)."""
    if seq_q <= _PASS_MAX:
        return [(0, seq_q)]
    passes, r0 = [], 0
    while seq_q - r0 > _PASS_MAX:
        passes.append((r0, 2 * _O_TILE_SIZE))
        r0 += 2 * _O_TILE_SIZE
    passes.append((r0, seq_q - r0))
    return passes


def _chunk_head(q_sb, k_sb, v_sb, seq_k, n_tiles, seq_q, o_psums, neg_max, pin=False):
    """o_psums[q tile] = sum over the chunk's key tiles of exp(K^T Q - c)^T @ [V | 1], MM2 trailing MM1 + exp by
    _PROLOGUE tiles so the Tensor engine always has ready probabilities. seq_q <= _PASS_MAX (see _query_passes)."""
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
            if pin:
                scores = nl.ndarray(shape=(_P, seq_q), dtype=nl.float32, buffer=nl.psum,
                                    address=(0, _SCORE_BANKS[tile % 2] * PSUM_BANK_SIZE))
            else:
                scores = nl.ndarray(shape=(_P, seq_q), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=scores[0:ext, :], stationary=k_sb[:, nl.ds(tile * _P, ext)], moving=q_sb)
            bias = neg_max if isinstance(neg_max, (int, float)) else neg_max[0:ext, :]
            nisa.activation(dst=probs[0:ext, tile % n_bufs, :], data=scores[0:ext, :], op=nl.exp, scale=1.0,
                            bias=bias)
            if tile >= depth:
                mm2(tile - depth)
        for tile in range(n_tiles - depth, n_tiles):
            mm2(tile)


# THE GUARD (the *_g* entry points). The constant offset c makes softmax(Q K^T - c) V exact only while no row's
# exponentials overflow and its largest one does not underflow; that was measured on one clip, not guaranteed. So the
# final call checks every row's sum S = sum_k exp(s_k - c) (the ones column of the accumulator): S in [e^-60, e^78]
# means nothing overflowed (every term and every partial sum <= e^78) and the largest term is >= e^-60 / 17280 keys,
# a normal float, so the result is the exact softmax up to rounding, whatever the input; NaN fails the check. A block
# (head, 128 query rows) with any row outside the band is recomputed in the same kernel, behind a runtime branch,
# over every chunk's keys (the call gets the cached chunks too) with each row's own maximum as the offset: one pass
# for the row maxima, one for exp(K^T q - m)^T @ [V | 1]. The two physical cores of a logical core must take every
# runtime branch the same number of times (a branch taken on one core only hangs the logical core: NRT_TIMEOUT,
# kiro/nki_branch_test.py), so they exchange their block flags (sendrecv) and redo block slot (head i of each core,
# query tile t) on both when either core's block is out of band. The flags output gives per block 2 (out of band,
# redone), 1 (redone as the other core's partner) or 0.
_BAND_LO = math.exp(-60.0)
_BAND_HI = math.exp(78.0)
_GUARD_BANK = 7      # pinned placement: the check's count and the redo's row-max transpose (the fast path leaves it)
_MAX_GROUP = 512     # keys per row-max matmul (one fp32 PSUM bank)


def _psum(shape, bank, pin):
    if pin:
        return nl.ndarray(shape=shape, dtype=nl.float32, buffer=nl.psum, address=(0, bank * PSUM_BANK_SIZE))
    return nl.ndarray(shape=shape, dtype=nl.float32, buffer=nl.psum)


def _guard_check(acc, per, n_o_tiles, seq_q, real_q, d, n_prgs, prg_id, pin, flags):
    """Band check of every row sum in the finished accumulator acc [128, per, n_o_tiles, d + 1] (block b = head *
    n_o_tiles + query tile). Query rows real_q .. seq_q - 1 are padding (LINGBOT_TOKEN_PAD) and pass: their outputs
    are dropped, so an out-of-band padding row must not redo the block's real rows. Writes this core's row of flags
    [n_prgs, per * n_o_tiles] and returns (redo [1, B] int32 per block slot, any [1, 1] int32), identical on both
    cores."""
    n_b = per * n_o_tiles
    # column d of every block, [128, n_b] (one free dim: the blocks are d + 1 apart). The last query tile's unused
    # rows hold 1.0 (set when acc was allocated).
    sums = acc.ap(pattern=[[n_b * (d + 1), _P], [d + 1, n_b]], offset=d)
    le = nl.ndarray(shape=(_P, n_b), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=le, data=sums, op0=nl.less_equal, operand0=_BAND_HI)
    ok = nl.ndarray(shape=(_P, n_b), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=ok, data=sums, op0=nl.greater_equal, operand0=_BAND_LO, op1=nl.multiply,
                              operand1=le)
    pidx = None
    for t in range(n_o_tiles):
        p0 = real_q - t * _O_TILE_SIZE              # first padding row of query tile t, in tile partitions
        if p0 >= min(_O_TILE_SIZE, seq_q - t * _O_TILE_SIZE):
            continue                                 # no padding rows in this tile
        if pidx is None:
            pidx = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=pidx, pattern=[[0, 1]], offset=0, channel_multiplier=1)      # partition index
        pad = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pad, data=pidx, op0=nl.greater_equal, operand0=float(max(p0, 0)))
        tile_t = ok.ap(pattern=[[n_b, _P], [n_o_tiles, per]], offset=t)                 # tile t of every head
        nisa.tensor_scalar(dst=tile_t, data=tile_t, op0=nl.maximum, operand0=pad)
    ones = nl.ndarray(shape=(_P, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    _memset(ones, 1.0)
    n_ok = _psum((1, n_b), _GUARD_BANK, pin)
    nisa.nc_matmul(dst=n_ok, stationary=ones, moving=ok)                 # rows in band, per block
    bad = nl.ndarray(shape=(1, n_b), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bad, data=n_ok, op0=nl.less, operand0=_P - 0.5)
    sel = bad
    if n_prgs > 1:
        other = nl.ndarray(shape=(1, n_b), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(src=bad, dst=other, send_to_rank=1 - prg_id, recv_from_rank=1 - prg_id, pipe_id=0,
                      dma_engine=nisa.dma_engine.gpsimd_dma)
        sel = nl.ndarray(shape=(1, n_b), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sel, data1=bad, data2=other, op=nl.maximum)
    code = nl.ndarray(shape=(1, n_b), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=code, data1=bad, data2=sel, op=nl.add)
    _dma(dst=flags[prg_id:prg_id + 1, :], src=code)
    redo = nl.ndarray(shape=(1, n_b), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=redo, src=sel)
    any_f = nl.ndarray(shape=(1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=any_f, data=sel, op=nl.maximum, axis=(1,))
    any_i = nl.ndarray(shape=(1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=any_i, src=any_f)
    return redo, any_i


def _redo_block(q_hbm, chunks, acc_view, head, n_heads, d, r0, rows, k_blocks, v_token_major, pin):
    """acc_view [rows, d + 1] <- exp(K^T q - m)^T @ [V | 1] for queries r0 .. r0 + rows of head over every chunk's
    keys, each query row offset by its own maximum m (rounded to bf16: any offset near the maximum is exact): pass 1
    the row maxima ([rows, 512] score groups), pass 2 the accumulation (the offset added in PSUM by a one-row
    matmul). The output loop then normalizes it like any other block."""
    q_sb = nl.ndarray(shape=(d, rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    _dma(dst=q_sb, src=q_hbm[head, :, nl.ds(r0, rows)])
    m_run = nl.ndarray(shape=(rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    m_grp = nl.ndarray(shape=(rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    n_grp = 0
    for k, _ in chunks:
        seq_k = k.shape[2]
        k_sb = nl.ndarray(shape=(d, seq_k), dtype=nl.bfloat16, buffer=nl.sbuf)
        _load_k(k_sb, k, head, n_heads, d, seq_k, k_blocks)
        for g in range(div_ceil(seq_k, _MAX_GROUP)):
            w = min(_MAX_GROUP, seq_k - g * _MAX_GROUP)
            sc = _psum((_P, _MAX_GROUP), _SCORE_BANKS[n_grp % 2], pin)
            nisa.nc_matmul(dst=sc[0:rows, 0:w], stationary=q_sb, moving=k_sb[:, nl.ds(g * _MAX_GROUP, w)])
            if n_grp == 0:
                nisa.tensor_reduce(dst=m_run, data=sc[0:rows, 0:w], op=nl.maximum, axis=(1,))
            else:
                nisa.tensor_reduce(dst=m_grp, data=sc[0:rows, 0:w], op=nl.maximum, axis=(1,))
                nisa.tensor_tensor(dst=m_run, data1=m_run, data2=m_grp, op=nl.maximum)
            n_grp += 1
    m_row = _psum((1, rows), _GUARD_BANK, pin)
    nisa.nc_transpose(dst=m_row, data=m_run, engine=nisa.tensor_engine)
    neg_m = nl.ndarray(shape=(1, rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_m, data=m_row, op0=nl.multiply, operand0=-1.0)
    ones = nl.ndarray(shape=(1, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    _memset(ones, 1.0)
    o_ps = _psum((rows, d + 1), 0, pin)
    n_mm = 0
    for k, v in chunks:
        seq_k = k.shape[2]
        n_tiles = div_ceil(seq_k, _P)
        k_sb = nl.ndarray(shape=(d, seq_k), dtype=nl.bfloat16, buffer=nl.sbuf)
        v_sb = nl.ndarray(shape=(_P, n_tiles, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
        _memset(v_sb[:, :, d:d + 1], 1.0)
        _load_k(k_sb, k, head, n_heads, d, seq_k, k_blocks)
        _load_v(v_sb, v, head, n_heads, d, seq_k, n_tiles, v_token_major)
        probs = nl.ndarray(shape=(_P, 2, rows), dtype=nl.bfloat16, buffer=nl.sbuf)
        for tile in range(n_tiles):
            ext = min(_P, seq_k - tile * _P)
            sc = _psum((_P, rows), _SCORE_BANKS[tile % 2], pin)
            nisa.nc_matmul(dst=sc[0:ext, :], stationary=k_sb[:, nl.ds(tile * _P, ext)], moving=q_sb,
                           accumulate=False)
            nisa.nc_matmul(dst=sc[0:ext, :], stationary=ones[0:1, 0:ext], moving=neg_m, accumulate=True)
            nisa.activation(dst=probs[0:ext, tile % 2, :], data=sc[0:ext, :], op=nl.exp)
            nisa.nc_matmul(dst=o_ps, stationary=probs[0:ext, tile % 2, :], moving=v_sb[0:ext, tile, :],
                           accumulate=n_mm > 0)
            n_mm += 1
    nisa.tensor_copy(dst=acc_view, src=o_ps, engine=nisa.engine.vector)


def _cmax2(q_hbm, k_chunks, v_chunks, softmax_max, acc_in=None, normalize=True, k_blocks=1, v_token_major=False,
           guard=None, real_q=None):
    """[N, d, Sq] pre-scaled queries (Sq <= 640, in the passes of _query_passes) over every chunk's keys; heads
    split over the logical core's programs. normalize: [N, Sq, d] in q's dtype, else the fp32 accumulator
    [N, Sq, d + 1]; acc_in continues one. K chunks have the logical shape [N, d, Sk] and V chunks [N, Sk, d],
    whatever their memory (k_blocks, v_token_major). guard (normalize only): the (K, V) of every chunk the rows
    attend, the cached ones first; returns (out, flags [n_prgs, per * n_o_tiles]) with the check of _guard_check
    (rows real_q .. Sq - 1 are padding, not checked; None: none) and out-of-band blocks redone (_redo_block)."""
    n_heads, d, seq_q = q_hbm.shape
    kernel_assert(d == _P, f"d must be {_P}, got {d}")
    kernel_assert(seq_q <= _MAX_Q, f"seq_q={seq_q} must be <= {_MAX_Q}")
    passes = _query_passes(seq_q)
    pin = len(passes) > 1 if _PSUM_PIN_ENV == "" else _PSUM_PIN_ENV == "1"
    kernel_assert(not pin or div_ceil(seq_q, _O_TILE_SIZE) <= _SCORE_BANKS[0],
                  f"pinned PSUM: {div_ceil(seq_q, _O_TILE_SIZE)} output tiles exceed banks 0-{_SCORE_BANKS[0] - 1}")
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

    # q_sbs[head][pass]: each pass's queries in an SBUF tensor of their own, so MM1's moving operand is always a
    # whole tensor (one pass up to _PASS_MAX queries: the whole [d, Sq], as before).
    q_sbs = []
    for head in range(per):
        per_pass = []
        for r0, rows_p in passes:
            q_sb = nl.ndarray(shape=(d, rows_p), dtype=nl.bfloat16, buffer=nl.sbuf)
            _dma(dst=q_sb, src=q_hbm[h0 + head, :, :] if rows_p == seq_q else q_hbm[h0 + head, :, nl.ds(r0, rows_p)])
            per_pass.append(q_sb)
        q_sbs.append(per_pass)

    acc = nl.ndarray(shape=(_O_TILE_SIZE, per, n_o_tiles, d + 1), dtype=nl.float32, buffer=nl.sbuf)
    if guard is not None and seq_q < n_o_tiles * _O_TILE_SIZE:
        # the guard checks all 128 rows of every query tile: the last tile's unused rows (never written) must pass
        _memset(acc[:, :, n_o_tiles - 1, d:d + 1], 1.0)
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
            for pi, (r0, rows_p) in enumerate(passes):
                # one pass of the chunk's key tiles per query range; K and V stay in SBUF across passes
                if pin:
                    b0 = r0 // _O_TILE_SIZE
                    o_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(rows_p, d + 1),
                                           bank_ids=tuple(range(b0, b0 + div_ceil(rows_p, _O_TILE_SIZE))))
                else:
                    o_psums = nt.psum_pool(tile_size=(_O_TILE_SIZE, d + 1), element_shape=(rows_p, d + 1))
                _chunk_head(q_sbs[head][pi], k_sb, v_sb, seq_k, n_tiles, rows_p, o_psums, neg_max, pin=pin)
                for j in range(div_ceil(rows_p, _O_TILE_SIZE)):
                    rows = o_psums[j].data.shape[0]
                    view = acc[:rows, head, r0 // _O_TILE_SIZE + j, :]
                    if chunk_idx == 0 and acc_in is None:
                        nisa.tensor_copy(dst=view, src=o_psums[j].data, engine=nisa.engine.vector)
                    else:
                        _add(view, o_psums[j].data)

    if guard is not None:
        kernel_assert(normalize, "the guard checks the final call (normalize=True)")
        real_q = seq_q if real_q is None else int(real_q)
        kernel_assert(0 < real_q <= seq_q, f"real_q={real_q} must be in 1 .. seq_q={seq_q}")
        flags = nl.ndarray(shape=(n_prgs, per * n_o_tiles), dtype=nl.float32, buffer=nl.shared_hbm)
        redo, any_redo = _guard_check(acc, per, n_o_tiles, seq_q, real_q, d, n_prgs, prg_id, pin, flags)
        # Runtime branches, skipped unless some block slot is out of band: any slot, then each slot. A redone block
        # replaces its accumulator, which the output loop below normalizes like the others. Placed before the
        # output loop: after it, the branch's engine reconvergence waited for the output copies (CP16, 4 chunks:
        # +62 us per call against +52 here).
        reg_any = nisa.register_alloc()
        nisa.register_load(reg_any, any_redo)

        def _redo_any(i):
            for head in range(per):
                for ot in range(n_o_tiles):
                    reg_b = nisa.register_alloc()
                    nisa.register_load(reg_b, redo[0:1, head * n_o_tiles + ot:head * n_o_tiles + ot + 1])

                    def _redo_one(j, head=head, ot=ot):
                        rows = min(_O_TILE_SIZE, seq_q - ot * _O_TILE_SIZE)
                        _redo_block(q_hbm, guard, acc[:rows, head, ot, :], h0 + head, n_heads, d,
                                    ot * _O_TILE_SIZE, rows, k_blocks, v_token_major, pin)
                    nl.fori_loop(0, reg_b, _redo_one)
        nl.fori_loop(0, reg_any, _redo_any)

    for head in range(per):
        for ot in range(n_o_tiles):
            rows = min(_O_TILE_SIZE, seq_q - ot * _O_TILE_SIZE)
            dst = out[h0 + head, nl.ds(ot * _O_TILE_SIZE, rows), :]
            if normalize:
                _dma(dst=dst, src=_normalize_acc(acc[:rows, head, ot, :]))
            else:
                _dma(dst=dst, src=acc[:rows, head, ot, :])
    if guard is None:
        return out
    return out, flags


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


# Guarded (LINGBOT_CMAX_GUARD): the new chunk (k0, v0), the accumulator of the cached chunks, and those chunks
# (lk*, lv*), which only a redo reads; real_q (compile-time) = the real query rows (the rest are padding). Return
# (out, flags).
@nki.jit
def cmax2_full_1g(q, k0, v0, softmax_max, k_blocks, v_token_major, real_q):
    return _cmax2(q, [k0], [v0], softmax_max, k_blocks=k_blocks, v_token_major=v_token_major, guard=[(k0, v0)],
                  real_q=real_q)


@nki.jit
def cmax2_final_1g1(q, k0, v0, acc, lk0, lv0, softmax_max, k_blocks, v_token_major, real_q):
    return _cmax2(q, [k0], [v0], softmax_max, acc_in=acc, k_blocks=k_blocks, v_token_major=v_token_major,
                  guard=[(lk0, lv0), (k0, v0)], real_q=real_q)


@nki.jit
def cmax2_final_1g2(q, k0, v0, acc, lk0, lk1, lv0, lv1, softmax_max, k_blocks, v_token_major, real_q):
    return _cmax2(q, [k0], [v0], softmax_max, acc_in=acc, k_blocks=k_blocks, v_token_major=v_token_major,
                  guard=[(lk0, lv0), (lk1, lv1), (k0, v0)], real_q=real_q)


@nki.jit
def cmax2_final_1g3(q, k0, v0, acc, lk0, lk1, lk2, lv0, lv1, lv2, softmax_max, k_blocks, v_token_major, real_q):
    return _cmax2(q, [k0], [v0], softmax_max, acc_in=acc, k_blocks=k_blocks, v_token_major=v_token_major,
                  guard=[(lk0, lv0), (lk1, lv1), (lk2, lv2), (k0, v0)], real_q=real_q)
