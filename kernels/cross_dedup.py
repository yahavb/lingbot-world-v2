"""LINGBOT_CROSS_KERNEL=1: the deduplicated cross-attention (LINGBOT_CROSS_DEDUP, wan/modules/attention.py
cross_attention_dedup) as one NKI kernel instead of the block graph's fp32 matmuls + softmax.

  out[s, h, :] = softmax_k(q[s, h] . k[h, :, k] / sqrt(d) + bias[k]) @ v[h, k, :]     over the first Kc keys

Exact softmax (row max) in the query-major layout, four heads per instruction group so the Vector / Scalar
instructions run on [128, 4 x 128] tiles: per (4 heads, 128 queries)
  Tensor  4 transposes of the token-major q tile -> q^T [d, 4, q] (PSUM -> SBUF on Scalar), 4 score matmuls
          S_h [q, Kc] = q_h^T (stationary) x K_h [d, Kc], one ones x bias matmul adding the keys' bias to all of them,
          4 transposes of the probabilities, 4 P @ V matmuls O_h [q, d] = P_h^T (stationary) x V_h [Kc, d]
  Vector  row max of all four heads (one reduce), the probabilities' transposes out of PSUM, 1 / sum, normalize
  Scalar  exp(S - max) with the row sums fused (one instruction per head)
2-chip TP-free cross-attention (40 heads, 544 shard queries, Kc = 128): the graph's version spent ~120 us of the
block's critical path between the q and o projections (bb_tp2 profile: dot.229 / dot.256 / add.232 ...).
Layouts: q [Sq, N * d] token-major (the q projection's [1, Sq, N, d] flattened), k [N, d, L] and v [N, L, d] (the
N-major cross cache, LINGBOT_CROSS_NMAJOR), bias [Kc] fp32; out [Sq, N * d] bf16. Heads split over the two cores.
"""
import math

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import PSUM_BANK_SIZE, div_ceil, get_verified_program_sharding_info

_P = 128
_G = 4          # heads per instruction group


def _psum(shape, dtype, bank):
    return nl.ndarray(shape=shape, dtype=dtype, buffer=nl.psum, address=(0, bank * PSUM_BANK_SIZE))


@nki.jit
def cross_dedup_kernel(q, k, v, bias, n_heads):
    sq, nd = q.shape
    n = int(n_heads)
    d = nd // n
    kc = int(bias.shape[0])
    kernel_assert(d == _P, f"head_dim must be {_P}, got {d}")
    kernel_assert(kc == _P, f"the deduplicated slice must be {_P} keys, got {kc}")
    _, n_prgs, prg_id = get_verified_program_sharding_info("cross_dedup", max_sharding=2)
    kernel_assert(n % (n_prgs * _G) == 0, f"{n=} heads vs {n_prgs} programs x {_G}")
    per = n // n_prgs
    h_base = prg_id * per
    scale = 1.0 / math.sqrt(d)
    out = nl.ndarray(shape=(sq, nd), dtype=q.dtype, buffer=nl.shared_hbm)

    ones = nl.ndarray(shape=(1, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    # the keys' bias / scale as one row, repeated for the group's heads: [1, G * Kc]. The scores stay unscaled
    # (q . k + bias / scale) and the exp applies the scale, so q is not rounded twice.
    braw = nl.ndarray(shape=(1, _G * kc), dtype=nl.float32, buffer=nl.sbuf)
    for g in range(_G):
        nisa.dma_copy(dst=braw[0:1, g * kc:(g + 1) * kc], src=bias.ap(pattern=[[0, 1], [1, kc]], offset=0))
    brow = nl.ndarray(shape=(1, _G * kc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=brow[0:1, 0:_G * kc], data=braw[0:1, 0:_G * kc], op0=nl.multiply, operand0=1.0 / scale)

    n_qt = div_ceil(sq, _P)
    for gi in range(per // _G):
        h0 = h_base + gi * _G
        # K_h [d, Kc] and V_h [Kc, d] of the group's heads
        ks = nl.ndarray(shape=(_P, _G, kc), dtype=k.dtype, buffer=nl.sbuf)
        vs = nl.ndarray(shape=(_P, _G, d), dtype=v.dtype, buffer=nl.sbuf)
        L = int(k.shape[2])
        nisa.dma_copy(dst=ks[:, 0:_G, 0:kc], src=k.ap(pattern=[[L, d], [d * L, _G], [1, kc]], offset=h0 * d * L))
        Lv = int(v.shape[1])
        nisa.dma_copy(dst=vs[:, 0:_G, 0:d], src=v.ap(pattern=[[d, kc], [Lv * d, _G], [1, d]], offset=h0 * Lv * d))
        for t in range(n_qt):
            q0 = t * _P
            sz = min(_P, sq - q0)
            bk = (gi * n_qt + t) % 2
            # q tile [sz, G * d] token-major -> q^T per head [d, sz] (scaled by 1 / sqrt(d) on the copy out)
            qs = nl.ndarray(shape=(_P, _G * d), dtype=q.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=qs[0:sz, 0:_G * d], src=q[nl.ds(q0, sz), nl.ds(h0 * d, _G * d)])
            qt_ps = _psum((_P, _G * _P), q.dtype, 0 + bk)
            for g in range(_G):
                nisa.nc_transpose(dst=qt_ps[0:d, g * _P:g * _P + sz], data=qs[0:sz, g * d:(g + 1) * d],
                                  engine=nisa.engine.tensor)
            qt = nl.ndarray(shape=(_P, _G, _P), dtype=q.dtype, buffer=nl.sbuf)
            nisa.activation(dst=qt[0:d, 0:_G, 0:sz], op=nl.copy,
                            data=qt_ps[0:d, 0:_G * _P].reshape((d, _G, _P))[0:d, 0:_G, 0:sz])
            # scores S_h [sz, Kc] + bias
            s_ps = _psum((_P, _G * kc), nl.float32, 2 + bk)
            for g in range(_G):
                nisa.nc_matmul(s_ps[0:sz, g * kc:(g + 1) * kc], stationary=qt[0:d, g, 0:sz],
                               moving=ks[0:d, g, 0:kc], accumulate=False)
            nisa.nc_matmul(s_ps[0:sz, 0:_G * kc], stationary=ones[0:1, 0:sz], moving=brow[0:1, 0:_G * kc],
                           accumulate=True)
            # row max per head, exp with the row sums
            mr = nl.ndarray(shape=(_P, _G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mr[0:sz, 0:_G], op=nl.maximum,
                               data=s_ps[0:sz, 0:_G * kc].reshape((sz, _G, kc)), axis=2)
            mx = nl.ndarray(shape=(_P, _G), dtype=nl.float32, buffer=nl.sbuf)     # -scale * row max
            nisa.tensor_scalar(dst=mx[0:sz, 0:_G], data=mr[0:sz, 0:_G], op0=nl.multiply, operand0=-scale)
            pr = nl.ndarray(shape=(_P, _G * kc), dtype=q.dtype, buffer=nl.sbuf)
            sm = nl.ndarray(shape=(_P, _G), dtype=nl.float32, buffer=nl.sbuf)
            for g in range(_G):
                nisa.activation(dst=pr[0:sz, g * kc:(g + 1) * kc], op=nl.exp, data=s_ps[0:sz, g * kc:(g + 1) * kc],
                                bias=mx[0:sz, g:g + 1], scale=scale, reduce_op=nl.add, reduce_res=sm[0:sz, g:g + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            # P_h^T [Kc, sz]
            pt_ps = _psum((_P, _G * _P), q.dtype, 4 + bk)
            for g in range(_G):
                nisa.nc_transpose(dst=pt_ps[0:kc, g * _P:g * _P + sz], data=pr[0:sz, g * kc:(g + 1) * kc],
                                  engine=nisa.engine.tensor)
            pt = nl.ndarray(shape=(_P, _G, _P), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pt[0:kc, 0:_G, 0:sz],
                             src=pt_ps[0:kc, 0:_G * _P].reshape((kc, _G, _P))[0:kc, 0:_G, 0:sz],
                             engine=nisa.engine.vector)
            # O_h [sz, d] = P_h @ V_h, normalized
            o_ps = _psum((_P, _G * d), nl.float32, 6 + bk)
            for g in range(_G):
                nisa.nc_matmul(o_ps[0:sz, g * d:(g + 1) * d], stationary=pt[0:kc, g, 0:sz], moving=vs[0:kc, g, 0:d],
                               accumulate=False)
            rs = nl.ndarray(shape=(_P, _G), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=rs[0:sz, 0:_G], data=sm[0:sz, 0:_G])
            ob = nl.ndarray(shape=(_P, _G * d), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ob[0:sz, 0:_G * d].reshape((sz, _G, d)),
                               data1=o_ps[0:sz, 0:_G * d].reshape((sz, _G, d)),
                               data2=rs[0:sz, 0:_G].expand_dim(2).broadcast(2, d), op=nl.multiply)
            nisa.dma_copy(dst=out[nl.ds(q0, sz), nl.ds(h0 * d, _G * d)], src=ob[0:sz, 0:_G * d])
    return out
