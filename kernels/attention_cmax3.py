"""Constant-offset self-attention with a 512-query P@V (LINGBOT_CMAX=3): softmax(Q K^T - c) V for the layers whose
row maxima stay in the band of kernels/attention_cmax2_lb.py (c = LINGBOT_CMAX_OFFSET, the same _CMAX_LAYERS).

The 2-chip block profile (TP4 x CP2, 2176 queries x 17280 keys per rank) had the chained exact kernel
(attention_cte_lb) at ~2.3 ms with four engines near their limits at once: Tensor 70% (P@V streams 128 query
columns per instruction, ~2x the cost per FLOP of a 512-column one, perf/pe_mm_bench.py), Scalar 62% (the row-max
evict pass), the probability transposes' DMA queue 73%, Vector 47% (exp). Here:
  MM1  S [128 q, 512 k] = Q^T-tile (stationary) x K (512 moving columns), fp32 PSUM (pinned banks)
  exp  P = exp(S - c) straight from PSUM into bf16 SBUF with the row sums fused (no max pass), alternating
       Scalar (activation) and Vector (exponential)
  DMA  P -> P^T [128 k, 4, 512 q] (attention_cte_lb's 4-D transpose form), four query groups side by side
  MM2  O^T [d, 512 q] += V-tile (stationary [128 k, d]) x P^T (512 moving columns), fp32 PSUM
so one P@V instruction covers four query groups. Output d-major: O^T [N, d, Sq] (the chained kernel's layout).

Layouts: q [N, d, Sq] pre-scaled bf16 (cmax_prep_q); K chunk [N, d, Sk] logical, memory plain or rank-blocked
[k_blocks, N, d, Sk / k_blocks]; V chunk [N, Sk, d] logical, memory plain or token-major [Sk, N, d]. Partial calls
return (o [N, d, Sq] fp32 unnormalized, s [N, Sq] fp32 row sums); the final call adds one more chunk to them (the
offset is the same constant, so partial sums just add) and normalizes. Heads split over the logical core's programs.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE, div_ceil, get_verified_program_sharding_info

_P = 128
_KB = 512          # keys per MM1 instruction (one fp32 PSUM bank of scores)
_SG = 4            # query groups (of 128) per P@V instruction
_SUM_GROUP = int(os.environ.get("LINGBOT_C3_SUMGROUP", "8"))   # key tiles per bf16 partial row sum
# experiment knobs (kiro): MM1 + exp lead over P@V in key tiles; row sums on the Vector engine (vec) or as a
# ones-matmul per key tile on the Tensor engine (pe); exp on Scalar for every tile (0) or every other tile on
# Vector (1)
_DEPTH = int(os.environ.get("LINGBOT_C3_DEPTH", "2"))
_SUMMODE = os.environ.get("LINGBOT_C3_SUMMODE", "vec")
_EXPALT = os.environ.get("LINGBOT_C3_EXPALT", "0") == "1"
_EXP_RED = os.environ.get("LINGBOT_C3_EXP_RED", "0") == "1"
_O_BANKS = (0, 1)  # O^T accumulators, ping-pong across query super-groups
_S_BANKS = (2, 3, 4, 5, 6, 7)   # S^T tiles (rotating) and the per-super-group sums broadcast
_SUM_BANK = 7
if _SUMMODE in ("pe", "pef"):
    _S_BANKS = (2, 3, 4, 5, 6)


# LINGBOT_C3_GPS_EVERY=k: every k-th key tile's row-sum add runs on GpSimd (its own chain) instead of Vector;
# LINGBOT_C3_DVE_EXP_EVERY=k: every k-th key tile's exp on Vector (exponential) instead of Scalar (0: none)
_GPS_EVERY = int(os.environ.get("LINGBOT_C3_GPS_EVERY", "0") or 0)
_DVE_EXP_EVERY = int(os.environ.get("LINGBOT_C3_DVE_EXP_EVERY", "0") or 0)
# LINGBOT_C3_FLUSH_LATE=1 (pef): a finished partial sum's ones-matmul is issued with the P@V of the same key tile
# (DEPTH items later) instead of right after its last add, so the in-order Tensor queue never waits on that add
_FLUSH_LATE = os.environ.get("LINGBOT_C3_FLUSH_LATE", "0") == "1"
_ENG = {"dve": nisa.engine.vector, "gps": nisa.engine.gpsimd}


def _sum_engines():
    return ("dve", "gps") if _GPS_EVERY else ("dve",)


def _sum_engine(ti):
    return "gps" if (_GPS_EVERY and ti % _GPS_EVERY == _GPS_EVERY - 1) else "dve"


def _chain_index(ti, n_t):
    """(index of key tile ti within its engine's chain, the chain's length)."""
    eng = _sum_engine(ti)
    mine = [t_ for t_ in range(n_t) if _sum_engine(t_) == eng]
    return mine.index(ti), len(mine)


def _psum(shape, bank):
    return nl.ndarray(shape=shape, dtype=nl.float32, buffer=nl.psum, address=(0, bank * PSUM_BANK_SIZE))


# LINGBOT_C3_LOAD_PRIO=0..3: DMA QoS priority of the per-head Q / K / V loads (lower is higher); LINGBOT_C3_LOAD_DGE=
# act | sync | sw (Q and K) and LINGBOT_C3_VLOAD_DGE (V): descriptors from the Scalar or Sync engine's hardware DGE,
# or GpSimd's software DGE, instead of the compiler's choice. In the block profile the first head's Q load waited
# ~150 us behind the V projection's activation transposes on the Sync queue while the CP K/V gathers held the DMA rings.
_LOAD_PRIO = os.environ.get("LINGBOT_C3_LOAD_PRIO", "")
_LOAD_DGE = os.environ.get("LINGBOT_C3_LOAD_DGE", "")
_VLOAD_DGE = os.environ.get("LINGBOT_C3_VLOAD_DGE", "")


def _load_kw(v=False):
    kw = {}
    if _LOAD_PRIO != "":
        kw["priority"] = int(_LOAD_PRIO)
    mode = _VLOAD_DGE if v else _LOAD_DGE
    if mode in ("act", "sync"):
        kw["dge_mode"] = nisa.dge_mode.hwdge
        kw["engine"] = nisa.engine.scalar if mode == "act" else nisa.engine.sync
    elif mode == "sw":
        kw["dge_mode"] = nisa.dge_mode.swdge
    return kw


def _load_k(k_sb, k, head, n_heads, d, seq_k, k_blocks):
    if k_blocks == 1:
        nisa.dma_copy(dst=k_sb[:, 0:seq_k], src=k[head, :, :], **_load_kw())
    else:
        lb = seq_k // k_blocks
        nisa.dma_copy(dst=k_sb.ap(pattern=[[seq_k, d], [lb, k_blocks], [1, lb]], offset=0),
                      src=k.ap(pattern=[[lb, d], [n_heads * d * lb, k_blocks], [1, lb]], offset=head * d * lb),
                      **_load_kw())


def _load_v(v_sb, v, head, n_heads, d, seq_k, n_tiles, token_major):
    """v_sb [128, n_tiles, d][:, t, :] <- keys t * 128 + partition of this head."""
    row = n_heads * d if token_major else d
    base = head * d if token_major else head * seq_k * d
    n_full = seq_k // _P
    tail = seq_k - n_full * _P
    width = n_tiles * d
    if n_full > 0:
        nisa.dma_copy(dst=v_sb.ap(pattern=[[width, _P], [d, n_full], [1, d]], offset=0),
                      src=v.ap(pattern=[[row, _P], [_P * row, n_full], [1, d]], offset=base), **_load_kw(v=True))
    if tail > 0:
        nisa.dma_copy(dst=v_sb.ap(pattern=[[width, tail], [1, d]], offset=n_full * d),
                      src=v.ap(pattern=[[row, tail], [1, d]], offset=base + n_full * _P * row), **_load_kw(v=True))


def _cmax3(q_hbm, k_chunks, v_chunks, offset, k_blocks, v_token_major, o_in=None, s_in=None, normalize=True,
           tok_cols=0):
    n_heads, d, seq_q = q_hbm.shape
    kernel_assert(d == _P, f"d must be {_P}, got {d}")
    kernel_assert(seq_q % _P == 0, f"seq_q={seq_q} must be a multiple of {_P}")
    _, n_prgs, prg_id = get_verified_program_sharding_info("attention_cmax3", max_sharding=2)
    kernel_assert(n_heads % n_prgs == 0, f"{n_heads=} must be divisible by {n_prgs=}")
    per = n_heads // n_prgs
    h0 = prg_id * per
    c = float(offset)
    tok = int(tok_cols or 0)
    kernel_assert(not tok or normalize, "tok_cols: normalized calls only")
    n_qt = seq_q // _P
    if normalize and tok:
        # LINGBOT_C3_TOKOUT: token-major [Sq, tok_cols] (heads x d, then zero columns up to the o-projection's
        # padded K), each head's normalized O^T transposed on the Tensor engine in the epilogue into obuf
        kernel_assert(tok >= n_heads * d and (tok - n_heads * d) % n_prgs == 0, f"{tok=} vs {n_heads}x{d}")
        o_out = nl.ndarray(shape=(seq_q, tok), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
        s_out = None
        obuf = nl.ndarray(shape=(_P, n_qt, per * d), dtype=q_hbm.dtype, buffer=nl.sbuf)
        pw = (tok - n_heads * d) // n_prgs
        if pw > 0:
            zb = nl.ndarray(shape=(_P, n_qt, pw), dtype=q_hbm.dtype, buffer=nl.sbuf)
            nisa.memset(dst=zb, value=0.0)
            nisa.dma_copy(dst=o_out.ap(pattern=[[tok, _P], [_P * tok, n_qt], [1, pw]], offset=n_heads * d + prg_id * pw),
                          src=zb[:, 0:n_qt, 0:pw])
    elif normalize:
        o_out = nl.ndarray(shape=(n_heads, d, seq_q), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
        s_out = None
    else:
        o_out = nl.ndarray(shape=(n_heads, d, seq_q), dtype=nl.float32, buffer=nl.shared_hbm)
        s_out = nl.ndarray(shape=(n_heads, seq_q), dtype=nl.float32, buffer=nl.shared_hbm)
    n_ch = len(k_chunks)
    seq_ks = [int(k.shape[2]) for k in k_chunks]
    n_vts = [div_ceil(sk, _P) for sk in seq_ks]
    tiles = [(ci, t) for ci in range(n_ch) for t in range(n_vts[ci])]     # 128-key tiles of every chunk
    n_t = len(tiles)
    n_sg = div_ceil(seq_q, _SG * _P)
    ones = nl.ndarray(shape=(_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ones_b = nl.ndarray(shape=(_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    red_sink = nl.ndarray(shape=(_P, 8), dtype=nl.float32, buffer=nl.sbuf)

    q_bufs = [nl.ndarray(shape=(d, seq_q), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    k_bufs = [[nl.ndarray(shape=(d, seq_ks[ci]), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]
    v_bufs = [[nl.ndarray(shape=(_P, n_vts[ci], d), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]

    def load_head(hl):
        st, head = hl % 2, h0 + hl
        nisa.dma_copy(dst=q_bufs[st], src=q_hbm[head, :, :], **_load_kw())
        for ci in range(n_ch):
            _load_k(k_bufs[st][ci], k_chunks[ci], head, n_heads, d, seq_ks[ci], k_blocks)
            _load_v(v_bufs[st][ci], v_chunks[ci], head, n_heads, d, seq_ks[ci], n_vts[ci], v_token_major)

    # One work item per (head, query super-group of 512, 128-key tile). Scores come out key-major,
    # S^T [128 keys, 512 queries] (stationary: the key tile [d, 128]; moving: the super-group's queries), so the
    # probabilities exp(S^T - c) are already P@V's moving operand: no transposes. The row sums (over keys, i.e.
    # partitions) accumulate elementwise on the Vector engine into sp [128, 512] fp32, reduced over partitions by
    # one ones-matmul per super-group, which also broadcasts them to all d partitions of O^T.
    items = [(hl, sg, ti) for hl in range(per) for sg in range(n_sg) for ti in range(n_t)]
    cnt = {"s": 0}
    sg_state = {}
    flush_q = {}       # LINGBOT_C3_FLUSH_LATE: item -> partial sums whose ones-matmul goes out with its P@V

    def stage1(item):
        hl, sg, ti = item
        st = hl % 2
        ci, t = tiles[ti]
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        if ti == 0:
            if _SUMMODE == "pef":
                sg_state[(hl, sg)] = (_psum((d, _SG * _P), _O_BANKS[(hl * n_sg + sg) % len(_O_BANKS)]),
                                      {"first": True, "ps": _psum((d, _SG * _P), _SUM_BANK),
                                       "chains": {eng: {"n": 0, "prev": None, "spb": None}
                                                  for eng in _sum_engines()}})
            else:
                sg_state[(hl, sg)] = (_psum((d, _SG * _P), _O_BANKS[(hl * n_sg + sg) % len(_O_BANKS)]),
                                      {eng: (nl.ndarray(shape=(_P, _SG * _P), dtype=nl.float32, buffer=nl.sbuf),
                                             nl.ndarray(shape=(_P, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf))
                                       for eng in _sum_engines()})
        sums = sg_state[(hl, sg)][1]
        w = min(_P, seq_ks[ci] - t * _P)
        s_ps = _psum((_P, _SG * _P), _S_BANKS[cnt["s"] % len(_S_BANKS)])
        cnt["s"] += 1
        nisa.nc_matmul(s_ps[0:w, 0:nq], stationary=k_bufs[st][ci][:, nl.ds(t * _P, w)],
                       moving=q_bufs[st][:, nl.ds(q0, nq)])
        pt = nl.ndarray(shape=(_P, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf)
        if (_EXPALT and ti % 2 == 1) or (_DVE_EXP_EVERY and ti % _DVE_EXP_EVERY == _DVE_EXP_EVERY - 1):
            nisa.exponential(dst=pt[0:w, 0:nq], src=s_ps[0:w, 0:nq], max_value=c)
        elif _EXP_RED:
            # LINGBOT_C3_EXP_RED=1 (experiment): the same exp with a (discarded) fused free-dim sum
            nisa.activation(dst=pt[0:w, 0:nq], op=nl.exp, data=s_ps[0:w, 0:nq], bias=-c, scale=1.0,
                            reduce_op=nl.add, reduce_res=red_sink[0:w, (cnt["s"] % 8):(cnt["s"] % 8) + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
        else:
            nisa.activation(dst=pt[0:w, 0:nq], op=nl.exp, data=s_ps[0:w, 0:nq], bias=-c, scale=1.0)
        if _SUMMODE == "pe":
            return pt
        if _SUMMODE == "pef":
            # bf16 partial sums of _SUM_GROUP full key tiles on Vector (the first add takes two tiles, no copy),
            # reduced over partitions and accumulated in fp32 PSUM by a ones-matmul per group (Tensor engine);
            # a partial key tile goes to the matmul alone. With LINGBOT_C3_GPS_EVERY, every k-th key tile's add
            # runs on a second chain on GpSimd, flushed into the same PSUM accumulator.
            def _flush(src, rows):
                if _FLUSH_LATE:
                    flush_q.setdefault(item, []).append((src, rows))
                    return
                nisa.nc_matmul(sums["ps"][:, 0:nq], stationary=ones_b[0:rows, :], moving=src[0:rows, 0:nq],
                               accumulate=not sums["first"])
                sums["first"] = False
            eng = _sum_engine(ti)
            ch, e_ = sums["chains"][eng], _ENG[eng]
            if w < _P:
                _flush(pt, w)
            elif ch["n"] == 0:
                ch["prev"], ch["n"] = pt, 1
            elif ch["n"] == 1:
                spb = nl.ndarray(shape=(_P, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=spb[:, 0:nq], data1=ch["prev"][:, 0:nq], data2=pt[:, 0:nq], op=nl.add,
                                   engine=e_)
                ch["spb"], ch["n"] = spb, 2
            else:
                nisa.tensor_tensor(dst=ch["spb"][:, 0:nq], data1=ch["spb"][:, 0:nq], data2=pt[:, 0:nq],
                                   op=nl.add, engine=e_)
                ch["n"] += 1
            j, n_j = _chain_index(ti, n_t)
            if ch["n"] and (ch["n"] == _SUM_GROUP or j == n_j - 1):
                _flush(ch["prev"] if ch["n"] == 1 else ch["spb"], _P)
                ch["n"] = 0
            return pt
        # sums over keys: per engine chain, bf16 partial sums of _SUM_GROUP of its key tiles (2-byte adds) folded
        # into that chain's fp32 accumulator; the chains add up in the epilogue
        eng = _sum_engine(ti)
        acc32, accb = sums[eng]
        j, n_j = _chain_index(ti, n_t)
        e_ = _ENG[eng]
        if j % _SUM_GROUP == 0:
            if w < _P:
                nisa.memset(dst=accb[:, 0:nq], value=0.0, engine=e_)
                nisa.tensor_copy(dst=accb[0:w, 0:nq], src=pt[0:w, 0:nq], engine=e_)
            else:
                nisa.tensor_copy(dst=accb[:, 0:nq], src=pt[:, 0:nq], engine=e_)
        else:
            nisa.tensor_tensor(dst=accb[0:w, 0:nq], data1=accb[0:w, 0:nq], data2=pt[0:w, 0:nq], op=nl.add,
                               engine=e_)
        if j % _SUM_GROUP == _SUM_GROUP - 1 or j == n_j - 1:
            if j < _SUM_GROUP:
                nisa.tensor_copy(dst=acc32[:, 0:nq], src=accb[:, 0:nq], engine=e_)
            else:
                nisa.tensor_tensor(dst=acc32[:, 0:nq], data1=acc32[:, 0:nq], data2=accb[:, 0:nq], op=nl.add,
                                   engine=e_)
        return pt

    def stage2(item, pt):
        hl, sg, ti = item
        st, head = hl % 2, h0 + hl
        ci, t = tiles[ti]
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        o_ps, sums = sg_state[(hl, sg)][:2]
        w = min(_P, seq_ks[ci] - t * _P)
        nisa.nc_matmul(o_ps[:, 0:nq], stationary=v_bufs[st][ci][0:w, t, :], moving=pt[0:w, 0:nq],
                       accumulate=ti != 0)
        for src, rows in flush_q.pop(item, []):
            nisa.nc_matmul(sums["ps"][:, 0:nq], stationary=ones_b[0:rows, :], moving=src[0:rows, 0:nq],
                           accumulate=not sums["first"])
            sums["first"] = False
        if _SUMMODE == "pe":
            if ti == 0:
                sg_state[(hl, sg)] = sg_state[(hl, sg)] + (_psum((d, _SG * _P), _SUM_BANK),)
            nisa.nc_matmul(sg_state[(hl, sg)][2][:, 0:nq], stationary=ones_b[0:w, :], moving=pt[0:w, 0:nq],
                           accumulate=ti != 0)
        if ti != n_t - 1:
            return
        # epilogue of this (head, super-group): sums over keys = ones^T @ sp, on every partition
        if _SUMMODE == "pe":
            b_ps = sg_state[(hl, sg)][2]
        elif _SUMMODE == "pef":
            b_ps = sums["ps"]
        else:
            engs = _sum_engines()
            sp = sums[engs[0]][0]
            for eng in engs[1:]:
                nisa.tensor_tensor(dst=sp[:, 0:nq], data1=sp[:, 0:nq], data2=sums[eng][0][:, 0:nq], op=nl.add,
                                   engine=nisa.engine.vector)
            b_ps = _psum((d, _SG * _P), _S_BANKS[cnt["s"] % len(_S_BANKS)])
            cnt["s"] += 1
            nisa.nc_matmul(b_ps[:, 0:nq], stationary=ones, moving=sp[:, 0:nq])
        o_sb = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        s_bc = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        if o_in is not None:
            o_prev = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=o_prev[:, 0:nq], src=o_in[head, :, nl.ds(q0, nq)])
            nisa.tensor_tensor(dst=o_sb[:, 0:nq], data1=o_ps[:, 0:nq], data2=o_prev[:, 0:nq], op=nl.add)
            s_prev = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s_prev[:, 0:nq], src=s_in.ap(pattern=[[0, d], [1, nq]], offset=head * seq_q + q0))
            nisa.tensor_tensor(dst=s_bc[:, 0:nq], data1=b_ps[:, 0:nq], data2=s_prev[:, 0:nq], op=nl.add)
        else:
            nisa.tensor_copy(dst=o_sb[:, 0:nq], src=o_ps[:, 0:nq])
            nisa.tensor_copy(dst=s_bc[:, 0:nq], src=b_ps[:, 0:nq])
        if not normalize:
            nisa.dma_copy(dst=o_out[head, :, nl.ds(q0, nq)], src=o_sb[:, 0:nq])
            nisa.dma_copy(dst=s_out.ap(pattern=[[seq_q, 1], [1, nq]], offset=head * seq_q + q0), src=s_bc[0:1, 0:nq])
            return
        rcp = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp[:, 0:nq], data=s_bc[:, 0:nq])
        o_bf = nl.ndarray(shape=(d, _SG * _P), dtype=q_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o_bf[:, 0:nq], data1=o_sb[:, 0:nq], data2=rcp[:, 0:nq], op=nl.multiply)
        if tok:
            # O^T [d, nq] -> nq/128 tiles [128 q, d] in this super-group's O bank (read above), then obuf
            nqt = nq // _P
            ob_bank = _O_BANKS[(hl * n_sg + sg) % len(_O_BANKS)]
            tp_ps = nl.ndarray(shape=(_P, _SG * _P), dtype=q_hbm.dtype, buffer=nl.psum,
                               address=(0, ob_bank * PSUM_BANK_SIZE))
            for i in range(nqt):
                nisa.nc_transpose(dst=tp_ps[:, i * _P:(i + 1) * _P], data=o_bf[:, i * _P:(i + 1) * _P],
                                  engine=nisa.engine.tensor)
            qt0 = q0 // _P
            nisa.tensor_copy(dst=obuf[:, qt0:qt0 + nqt, hl * d:(hl + 1) * d],
                             src=tp_ps[:, 0:nqt * _P].reshape((_P, nqt, _P)), engine=nisa.engine.vector)
            return
        nisa.dma_copy(dst=o_out[head, :, nl.ds(q0, nq)], src=o_bf[:, 0:nq])

    load_head(0)
    depth = _DEPTH      # MM1 + exp run this many key tiles ahead of P@V
    pend = []
    for item in items:
        hl, sg, ti = item
        pend.append((item, stage1(item)))
        if len(pend) > depth:
            stage2(*pend.pop(0))
        if sg == 0 and ti == depth and hl + 1 < per:
            load_head(hl + 1)          # after head hl - 1's last P@V (issued above, depth items late)
    while pend:
        stage2(*pend.pop(0))
    if normalize and tok:
        nisa.dma_copy(dst=o_out.ap(pattern=[[tok, _P], [_P * tok, n_qt], [1, per * d]], offset=h0 * d),
                      src=obuf[:, 0:n_qt, 0:per * d])
    if normalize:
        return o_out
    return o_out, s_out


# LINGBOT_C3_PAIR=1: two full key tiles per exp and per row-sum add. The block profile (bb_pf, 2-chip) put the
# exp (Scalar, [128, 512] fp32 PSUM -> bf16) at ~129 ns fixed + 0.82 ns per element and the bf16 add (Vector) at
# ~196 ns fixed + 0.42 ns per element, with Scalar 92% busy inside the attention call: one instruction over two
# PSUM banks (S^T of two key tiles) and one add over both tiles' probabilities cut the fixed costs in half.
# PSUM: bank 0 O^T (or 0-1 with LINGBOT_C3_PAIR_OB=2), bank 1 (7) the row sums, the rest slots of two S^T banks.
# Partial key tiles (a chunk's last 96 keys) run alone, their sums flushed straight to PSUM.
_PAIR = os.environ.get("LINGBOT_C3_PAIR", "0") == "1"
_PAIR_OB = int(os.environ.get("LINGBOT_C3_PAIR_OB", "1") or 1)


def _pair_engine(u):
    return "gps" if (_GPS_EVERY and u % _GPS_EVERY == _GPS_EVERY - 1) else "dve"


def _cmax3u(q_hbm, k_chunks, v_chunks, offset, k_blocks, v_token_major, o_in=None, s_in=None, normalize=True):
    n_heads, d, seq_q = q_hbm.shape
    kernel_assert(d == _P, f"d must be {_P}, got {d}")
    kernel_assert(seq_q % _P == 0, f"seq_q={seq_q} must be a multiple of {_P}")
    _, n_prgs, prg_id = get_verified_program_sharding_info("attention_cmax3", max_sharding=2)
    kernel_assert(n_heads % n_prgs == 0, f"{n_heads=} must be divisible by {n_prgs=}")
    per = n_heads // n_prgs
    h0 = prg_id * per
    c = float(offset)
    if normalize:
        o_out = nl.ndarray(shape=(n_heads, d, seq_q), dtype=q_hbm.dtype, buffer=nl.shared_hbm)
        s_out = None
    else:
        o_out = nl.ndarray(shape=(n_heads, d, seq_q), dtype=nl.float32, buffer=nl.shared_hbm)
        s_out = nl.ndarray(shape=(n_heads, seq_q), dtype=nl.float32, buffer=nl.shared_hbm)
    n_ch = len(k_chunks)
    seq_ks = [int(k.shape[2]) for k in k_chunks]
    n_vts = [div_ceil(sk, _P) for sk in seq_ks]
    tiles = [(ci, t) for ci in range(n_ch) for t in range(n_vts[ci])]
    widths = [min(_P, seq_ks[ci] - t * _P) for ci, t in tiles]
    units, held = [], None
    for ti, w in enumerate(widths):
        if w < _P:
            units.append((ti,))
        elif held is None:
            held = ti
        else:
            units.append((held, ti))
            held = None
    if held is not None:
        units.append((held,))
    n_u = len(units)
    pair_us = [u for u in range(n_u) if len(units[u]) == 2]
    chain_of = {eng: [u for u in pair_us if _pair_engine(u) == eng] for eng in ("dve", "gps")}
    chain_pos = {u: (chain_of[_pair_engine(u)].index(u), len(chain_of[_pair_engine(u)])) for u in pair_us}
    engs = ("dve", "gps") if _GPS_EVERY else ("dve",)
    if _PAIR_OB == 2:
        o_banks, sum_bank, slots = (0, 1), 7, (2, 4)
    else:
        o_banks, sum_bank, slots = (0,), 1, (2, 4, 6)
    n_sg = div_ceil(seq_q, _SG * _P)
    ones_b = nl.ndarray(shape=(_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)

    q_bufs = [nl.ndarray(shape=(d, seq_q), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    k_bufs = [[nl.ndarray(shape=(d, seq_ks[ci]), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]
    v_bufs = [[nl.ndarray(shape=(_P, n_vts[ci], d), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]

    def load_head(hl):
        st, head = hl % 2, h0 + hl
        nisa.dma_copy(dst=q_bufs[st], src=q_hbm[head, :, :], **_load_kw())
        for ci in range(n_ch):
            _load_k(k_bufs[st][ci], k_chunks[ci], head, n_heads, d, seq_ks[ci], k_blocks)
            _load_v(v_bufs[st][ci], v_chunks[ci], head, n_heads, d, seq_ks[ci], n_vts[ci], v_token_major)

    items = [(hl, sg, u) for hl in range(per) for sg in range(n_sg) for u in range(n_u)]
    cnt = {"s": 0}
    sg_state = {}
    flush_q = {}

    def stage1(item):
        hl, sg, u = item
        st = hl % 2
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        if u == 0:
            sg_state[(hl, sg)] = (_psum((d, _SG * _P), o_banks[(hl * n_sg + sg) % len(o_banks)]),
                                  {"first": True, "ps": _psum((d, _SG * _P), sum_bank),
                                   "chains": {eng: {"n": 0, "prev": None, "spb": None} for eng in engs}})
        sums = sg_state[(hl, sg)][1]
        unit = units[u]
        slot = slots[cnt["s"] % len(slots)]
        cnt["s"] += 1
        s_ps = nl.ndarray(shape=(_P, 2, _SG * _P), dtype=nl.float32, buffer=nl.psum,
                          address=(0, slot * PSUM_BANK_SIZE))
        for j, ti in enumerate(unit):
            ci, t = tiles[ti]
            w = widths[ti]
            nisa.nc_matmul(s_ps[0:w, j, 0:nq], stationary=k_bufs[st][ci][:, nl.ds(t * _P, w)],
                           moving=q_bufs[st][:, nl.ds(q0, nq)])
        pt = nl.ndarray(shape=(_P, 2, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf)
        w0 = widths[unit[0]]
        if len(unit) == 2:
            src, dst = s_ps[:, 0:2, 0:nq], pt[:, 0:2, 0:nq]
        else:
            src, dst = s_ps[0:w0, 0, 0:nq], pt[0:w0, 0, 0:nq]
        if _DVE_EXP_EVERY and u % _DVE_EXP_EVERY == _DVE_EXP_EVERY - 1:
            nisa.exponential(dst=dst, src=src, max_value=c)
        else:
            nisa.activation(dst=dst, op=nl.exp, data=src, bias=-c, scale=1.0)

        def flush(buf, j, rows):
            flush_q.setdefault(item, []).append((buf, j, rows))

        if len(unit) == 1:
            flush(pt, 0, w0)
        else:
            eng = _pair_engine(u)
            ch, e_ = sums["chains"][eng], _ENG[eng]
            if ch["n"] == 0:
                ch["prev"], ch["n"] = pt, 1
            elif ch["n"] == 1:
                spb = nl.ndarray(shape=(_P, 2, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=spb[:, 0:2, 0:nq], data1=ch["prev"][:, 0:2, 0:nq], data2=pt[:, 0:2, 0:nq],
                                   op=nl.add, engine=e_)
                ch["spb"], ch["n"] = spb, 2
            else:
                nisa.tensor_tensor(dst=ch["spb"][:, 0:2, 0:nq], data1=ch["spb"][:, 0:2, 0:nq],
                                   data2=pt[:, 0:2, 0:nq], op=nl.add, engine=e_)
                ch["n"] += 1
            j_, n_j = chain_pos[u]
            if ch["n"] == _SUM_GROUP or j_ == n_j - 1:
                buf = ch["prev"] if ch["n"] == 1 else ch["spb"]
                flush(buf, 0, _P)
                flush(buf, 1, _P)
                ch["n"] = 0
        return pt

    def stage2(item, pt):
        hl, sg, u = item
        st, head = hl % 2, h0 + hl
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        o_ps, sums = sg_state[(hl, sg)]
        for j, ti in enumerate(units[u]):
            ci, t = tiles[ti]
            w = widths[ti]
            nisa.nc_matmul(o_ps[:, 0:nq], stationary=v_bufs[st][ci][0:w, t, :], moving=pt[0:w, j, 0:nq],
                           accumulate=(u != 0 or j != 0))
        for buf, j, rows in flush_q.pop(item, []):
            nisa.nc_matmul(sums["ps"][:, 0:nq], stationary=ones_b[0:rows, :], moving=buf[0:rows, j, 0:nq],
                           accumulate=not sums["first"])
            sums["first"] = False
        if u != n_u - 1:
            return
        b_ps = sums["ps"]
        o_sb = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        s_bc = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        if o_in is not None:
            o_prev = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=o_prev[:, 0:nq], src=o_in[head, :, nl.ds(q0, nq)])
            nisa.tensor_tensor(dst=o_sb[:, 0:nq], data1=o_ps[:, 0:nq], data2=o_prev[:, 0:nq], op=nl.add)
            s_prev = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=s_prev[:, 0:nq], src=s_in.ap(pattern=[[0, d], [1, nq]], offset=head * seq_q + q0))
            nisa.tensor_tensor(dst=s_bc[:, 0:nq], data1=b_ps[:, 0:nq], data2=s_prev[:, 0:nq], op=nl.add)
        else:
            nisa.tensor_copy(dst=o_sb[:, 0:nq], src=o_ps[:, 0:nq])
            nisa.tensor_copy(dst=s_bc[:, 0:nq], src=b_ps[:, 0:nq])
        if not normalize:
            nisa.dma_copy(dst=o_out[head, :, nl.ds(q0, nq)], src=o_sb[:, 0:nq])
            nisa.dma_copy(dst=s_out.ap(pattern=[[seq_q, 1], [1, nq]], offset=head * seq_q + q0), src=s_bc[0:1, 0:nq])
            return
        rcp = nl.ndarray(shape=(d, _SG * _P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rcp[:, 0:nq], data=s_bc[:, 0:nq])
        o_bf = nl.ndarray(shape=(d, _SG * _P), dtype=q_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o_bf[:, 0:nq], data1=o_sb[:, 0:nq], data2=rcp[:, 0:nq], op=nl.multiply)
        nisa.dma_copy(dst=o_out[head, :, nl.ds(q0, nq)], src=o_bf[:, 0:nq])

    load_head(0)
    depth = _DEPTH      # MM1 + exp run this many units ahead of P@V
    pend = []
    for item in items:
        hl, sg, u = item
        pend.append((item, stage1(item)))
        if len(pend) > depth:
            stage2(*pend.pop(0))
        if sg == 0 and u == depth and hl + 1 < per:
            load_head(hl + 1)
    while pend:
        stage2(*pend.pop(0))
    if normalize:
        return o_out
    return o_out, s_out


_IMPL = _cmax3u if _PAIR else _cmax3


@nki.jit
def cmax3_full_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _IMPL(q, [k0], [v0], softmax_max, k_blocks, v_token_major, normalize=True)


@nki.jit
def cmax3_part_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _IMPL(q, [k0], [v0], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_part_2(q, k0, k1, v0, v1, softmax_max, k_blocks, v_token_major):
    return _IMPL(q, [k0, k1], [v0, v1], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_part_3(q, k0, k1, k2, v0, v1, v2, softmax_max, k_blocks, v_token_major):
    return _IMPL(q, [k0, k1, k2], [v0, v1, v2], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_final_1(q, k0, v0, o_acc, s_acc, softmax_max, k_blocks, v_token_major):
    return _IMPL(q, [k0], [v0], softmax_max, k_blocks, v_token_major, o_in=o_acc, s_in=s_acc, normalize=True)


@nki.jit
def cmax3_full_tok_1(q, k0, v0, softmax_max, k_blocks, v_token_major, tok_cols):
    """LINGBOT_C3_TOKOUT: cmax3_full_1 with the token-major [Sq, tok_cols] output (zero columns past n x d)."""
    return _cmax3(q, [k0], [v0], softmax_max, k_blocks, v_token_major, normalize=True, tok_cols=tok_cols)


@nki.jit
def cmax3_final_tok_1(q, k0, v0, o_acc, s_acc, softmax_max, k_blocks, v_token_major, tok_cols):
    """LINGBOT_C3_TOKOUT: cmax3_final_1 with the token-major [Sq, tok_cols] output (zero columns past n x d)."""
    return _cmax3(q, [k0], [v0], softmax_max, k_blocks, v_token_major, o_in=o_acc, s_in=s_acc, normalize=True,
                  tok_cols=tok_cols)
