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
_O_BANKS = (0, 1)  # O^T accumulators, ping-pong across query super-groups
_S_BANKS = (2, 3, 4, 5, 6, 7)   # S^T tiles (rotating) and the per-super-group sums broadcast
_SUM_BANK = 7
if _SUMMODE == "pe":
    _S_BANKS = (2, 3, 4, 5, 6)


def _psum(shape, bank):
    return nl.ndarray(shape=shape, dtype=nl.float32, buffer=nl.psum, address=(0, bank * PSUM_BANK_SIZE))


def _load_k(k_sb, k, head, n_heads, d, seq_k, k_blocks):
    if k_blocks == 1:
        nisa.dma_copy(dst=k_sb[:, 0:seq_k], src=k[head, :, :])
    else:
        lb = seq_k // k_blocks
        nisa.dma_copy(dst=k_sb.ap(pattern=[[seq_k, d], [lb, k_blocks], [1, lb]], offset=0),
                      src=k.ap(pattern=[[lb, d], [n_heads * d * lb, k_blocks], [1, lb]], offset=head * d * lb))


def _load_v(v_sb, v, head, n_heads, d, seq_k, n_tiles, token_major):
    """v_sb [128, n_tiles, d][:, t, :] <- keys t * 128 + partition of this head."""
    row = n_heads * d if token_major else d
    base = head * d if token_major else head * seq_k * d
    n_full = seq_k // _P
    tail = seq_k - n_full * _P
    width = n_tiles * d
    if n_full > 0:
        nisa.dma_copy(dst=v_sb.ap(pattern=[[width, _P], [d, n_full], [1, d]], offset=0),
                      src=v.ap(pattern=[[row, _P], [_P * row, n_full], [1, d]], offset=base))
    if tail > 0:
        nisa.dma_copy(dst=v_sb.ap(pattern=[[width, tail], [1, d]], offset=n_full * d),
                      src=v.ap(pattern=[[row, tail], [1, d]], offset=base + n_full * _P * row))


def _cmax3(q_hbm, k_chunks, v_chunks, offset, k_blocks, v_token_major, o_in=None, s_in=None, normalize=True):
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
    tiles = [(ci, t) for ci in range(n_ch) for t in range(n_vts[ci])]     # 128-key tiles of every chunk
    n_t = len(tiles)
    n_sg = div_ceil(seq_q, _SG * _P)
    ones = nl.ndarray(shape=(_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ones_b = nl.ndarray(shape=(_P, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)

    q_bufs = [nl.ndarray(shape=(d, seq_q), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
    k_bufs = [[nl.ndarray(shape=(d, seq_ks[ci]), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]
    v_bufs = [[nl.ndarray(shape=(_P, n_vts[ci], d), dtype=nl.bfloat16, buffer=nl.sbuf) for ci in range(n_ch)]
              for _ in range(2)]

    def load_head(hl):
        st, head = hl % 2, h0 + hl
        nisa.dma_copy(dst=q_bufs[st], src=q_hbm[head, :, :])
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

    def stage1(item):
        hl, sg, ti = item
        st = hl % 2
        ci, t = tiles[ti]
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        if ti == 0:
            sg_state[(hl, sg)] = (_psum((d, _SG * _P), _O_BANKS[(hl * n_sg + sg) % len(_O_BANKS)]),
                                  nl.ndarray(shape=(_P, _SG * _P), dtype=nl.float32, buffer=nl.sbuf),
                                  nl.ndarray(shape=(_P, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf))
        _, sp, spb = sg_state[(hl, sg)]
        w = min(_P, seq_ks[ci] - t * _P)
        s_ps = _psum((_P, _SG * _P), _S_BANKS[cnt["s"] % len(_S_BANKS)])
        cnt["s"] += 1
        nisa.nc_matmul(s_ps[0:w, 0:nq], stationary=k_bufs[st][ci][:, nl.ds(t * _P, w)],
                       moving=q_bufs[st][:, nl.ds(q0, nq)])
        pt = nl.ndarray(shape=(_P, _SG * _P), dtype=nl.bfloat16, buffer=nl.sbuf)
        if _EXPALT and ti % 2 == 1:
            nisa.exponential(dst=pt[0:w, 0:nq], src=s_ps[0:w, 0:nq], max_value=c)
        else:
            nisa.activation(dst=pt[0:w, 0:nq], op=nl.exp, data=s_ps[0:w, 0:nq], bias=-c, scale=1.0)
        if _SUMMODE == "pe":
            return pt
        # sums over keys: bf16 partial sums of _SUM_GROUP key tiles (2-byte adds), folded into the fp32 sp
        if ti % _SUM_GROUP == 0:
            kernel_assert(w == _P, "a sum group starts on a full key tile")
            nisa.tensor_copy(dst=spb[:, 0:nq], src=pt[:, 0:nq], engine=nisa.engine.vector)
        else:
            nisa.tensor_tensor(dst=spb[0:w, 0:nq], data1=spb[0:w, 0:nq], data2=pt[0:w, 0:nq], op=nl.add,
                               engine=nisa.engine.vector)
        if ti % _SUM_GROUP == _SUM_GROUP - 1 or ti == n_t - 1:
            if ti < _SUM_GROUP:
                nisa.tensor_copy(dst=sp[:, 0:nq], src=spb[:, 0:nq], engine=nisa.engine.vector)
            else:
                nisa.tensor_tensor(dst=sp[:, 0:nq], data1=sp[:, 0:nq], data2=spb[:, 0:nq], op=nl.add,
                                   engine=nisa.engine.vector)
        return pt

    def stage2(item, pt):
        hl, sg, ti = item
        st, head = hl % 2, h0 + hl
        ci, t = tiles[ti]
        q0 = sg * _SG * _P
        nq = min(_SG * _P, seq_q - q0)
        o_ps, sp, _ = sg_state[(hl, sg)]
        w = min(_P, seq_ks[ci] - t * _P)
        nisa.nc_matmul(o_ps[:, 0:nq], stationary=v_bufs[st][ci][0:w, t, :], moving=pt[0:w, 0:nq],
                       accumulate=ti != 0)
        if _SUMMODE == "pe":
            if ti == 0:
                sg_state[(hl, sg)] = sg_state[(hl, sg)] + (_psum((d, _SG * _P), _SUM_BANK),)
            nisa.nc_matmul(sg_state[(hl, sg)][3][:, 0:nq], stationary=ones_b[0:w, :], moving=pt[0:w, 0:nq],
                           accumulate=ti != 0)
        if ti != n_t - 1:
            return
        # epilogue of this (head, super-group): sums over keys = ones^T @ sp, on every partition
        if _SUMMODE == "pe":
            b_ps = sg_state[(hl, sg)][3]
        else:
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
    if normalize:
        return o_out
    return o_out, s_out


@nki.jit
def cmax3_full_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _cmax3(q, [k0], [v0], softmax_max, k_blocks, v_token_major, normalize=True)


@nki.jit
def cmax3_part_1(q, k0, v0, softmax_max, k_blocks, v_token_major):
    return _cmax3(q, [k0], [v0], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_part_2(q, k0, k1, v0, v1, softmax_max, k_blocks, v_token_major):
    return _cmax3(q, [k0, k1], [v0, v1], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_part_3(q, k0, k1, k2, v0, v1, v2, softmax_max, k_blocks, v_token_major):
    return _cmax3(q, [k0, k1, k2], [v0, v1, v2], softmax_max, k_blocks, v_token_major, normalize=False)


@nki.jit
def cmax3_final_1(q, k0, v0, o_acc, s_acc, softmax_max, k_blocks, v_token_major):
    return _cmax3(q, [k0], [v0], softmax_max, k_blocks, v_token_major, o_in=o_acc, s_in=s_acc, normalize=True)
