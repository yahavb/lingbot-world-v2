"""Self-attention Q/K post-processing after the MXFP8 projections (LINGBOT_QK_POST=1), replacing the compiler's
bias add + TP RMSNorm + RoPE chain (two sum-of-squares all-reduces and ~10 elementwise ops, ~0.5 ms between the
q/k/v kernels and the attention at the 2-chip shape, with no engine above 45% busy).

  qk_ss(q, k, bq, bk)        -> ss [L, 2] fp32: sum over this rank's features of (q + bq)^2 and (k + bk)^2
  (all-reduce ss over the TP group, one collective for both)
  qk_norm_rope(q, k, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps) -> (q', k') bf16 [L, F]:
      t = bf16((x + b) * rsqrt(ss * inv_dim + eps)) * w, then RoPE on interleaved pairs (t[2j], t[2j+1]) of every
      head with cos/sin [L, C] (C = head_dim / 2; rows past the real tokens carry cos 1, sin 0)

Rows (tokens) on partitions, features contiguous; LNC2 splits the rows. Same math as _tp_rms_norm +
causal_rope_apply_rotation, with fp32 bias add and RoPE.
"""
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

P = 128


def _bcast_rows(vec, F, dtype):
    """[1, F] HBM -> [P, F] SBUF, the same row on every partition."""
    t = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=vec.ap(pattern=[[0, P], [1, F]], offset=0))
    return t


@nki.jit
def _qk_ss(q, k, bq, bk):
    L, F = q.shape
    ss = nl.ndarray((L, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    bq_sb = _bcast_rows(bq, F, bq.dtype)
    bk_sb = _bcast_rows(bk, F, bk.dtype)
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        acc = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(2):
            src = q if j == 0 else k
            b_sb = bq_sb if j == 0 else bk_sb
            x = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:sz, 0:F], src=src[nl.ds(r_base + t * P, sz), 0:F])
            y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=x[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
            sq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sq[0:sz, 0:F], op=nl.square, data=y[0:sz, 0:F], reduce_op=nl.add,
                            reduce_res=acc[0:sz, j:j + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.dma_copy(dst=ss[nl.ds(r_base + t * P, sz), 0:2], src=acc[0:sz, 0:2])
    return ss


@nki.jit
def _qk_norm_rope(q, k, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps):
    L, F = q.shape
    C = cos.shape[1]
    H = F // (2 * C)
    oq = nl.ndarray((L, F), dtype=q.dtype, buffer=nl.shared_hbm)
    ok = nl.ndarray((L, F), dtype=k.dtype, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = (_bcast_rows(bq, F, bq.dtype), _bcast_rows(bk, F, bk.dtype))
    w_sb = (_bcast_rows(wq, F, wq.dtype), _bcast_rows(wk, F, wk.dtype))
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        r0 = r_base + t * P
        cs = nl.ndarray((P, C), dtype=cos.dtype, buffer=nl.sbuf)
        sn = nl.ndarray((P, C), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs[0:sz, 0:C], src=cos[nl.ds(r0, sz), 0:C])
        nisa.dma_copy(dst=sn[0:sz, 0:C], src=sin[nl.ds(r0, sz), 0:C])
        s2 = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s2[0:sz, 0:2], src=ss[nl.ds(r0, sz), 0:2])
        rr = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr[0:sz, 0:2], op=nl.rsqrt, data=s2[0:sz, 0:2], scale=inv_dim, bias=eps)
        cb = cs[0:sz, 0:C].expand_dim(1).broadcast(1, H)        # [sz, H, C]
        sb = sn[0:sz, 0:C].expand_dim(1).broadcast(1, H)
        for j in range(2):
            src = q if j == 0 else k
            dst = oq if j == 0 else ok
            x = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:sz, 0:F], src=src[nl.ds(r0, sz), 0:F])
            y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=x[0:sz, 0:F], data2=b_sb[j][0:sz, 0:F], op=nl.add)
            tn = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)       # bf16, as _tp_rms_norm's type_as(x)
            nisa.activation(dst=tn[0:sz, 0:F], op=nl.copy, data=y[0:sz, 0:F], scale=rr[0:sz, j:j + 1])
            tw = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tw[0:sz, 0:F], data1=tn[0:sz, 0:F], data2=w_sb[j][0:sz, 0:F], op=nl.multiply)
            t4 = tw[0:sz, 0:F].reshape((sz, H, C, 2))
            ev, od = t4.select(3, 0), t4.select(3, 1)                # [sz, H, C] each
            a = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            bb = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            o = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            o4 = o[0:sz, 0:F].reshape((sz, H, C, 2))
            # even: e cos - o sin
            nisa.tensor_tensor(dst=a[0:sz], data1=ev, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=bb[0:sz], data1=od, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=o4.select(3, 0), data1=a[0:sz], data2=bb[0:sz], op=nl.subtract)
            # odd: e sin + o cos
            c2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            d2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=c2[0:sz], data1=ev, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=d2[0:sz], data1=od, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=o4.select(3, 1), data1=c2[0:sz], data2=d2[0:sz], op=nl.add)
            nisa.dma_copy(dst=dst[nl.ds(r0, sz), 0:F], src=o[0:sz, 0:F])
    return oq, ok


qk_ss = wrap_nki(_qk_ss)
qk_norm_rope = wrap_nki(_qk_norm_rope)
