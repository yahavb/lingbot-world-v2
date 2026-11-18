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
    return _qk_ss_body((q, 0), (k, 0), bq, bk)


@nki.jit
def _qk_ss3(qkv, bq, bk):
    """LINGBOT_MX_QKV_FUSED + QK_POST: _qk_ss on the fused projection's output [L, 3F] (q, k, v column blocks)."""
    F = bq.shape[1]
    return _qk_ss_body((qkv, 0), (qkv, F), bq, bk)


def _qk_ss_body(qs, ks, bq, bk):
    L = qs[0].shape[0]
    F = bq.shape[1]
    q, k = qs[0], ks[0]
    ss = nl.ndarray((L, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    bq_sb = _bcast_rows(bq, F, bq.dtype)
    bk_sb = _bcast_rows(bk, F, bk.dtype)
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        acc = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(2):
            src, c0 = qs if j == 0 else ks
            b_sb = bq_sb if j == 0 else bk_sb
            x = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:sz, 0:F], src=src[nl.ds(r_base + t * P, sz), c0:c0 + F])
            y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=x[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
            sq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sq[0:sz, 0:F], op=nl.square, data=y[0:sz, 0:F], reduce_op=nl.add,
                            reduce_res=acc[0:sz, j:j + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.dma_copy(dst=ss[nl.ds(r_base + t * P, sz), 0:2], src=acc[0:sz, 0:2])
    return ss


@nki.jit
def _qk_norm_rope(q, k, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps):
    return _qk_norm_rope_body((q, 0), (k, 0), bq, bk, ss, wq, wk, cos, sin, inv_dim, eps)


@nki.jit
def _qk_norm_rope3(qkv, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps):
    """LINGBOT_MX_QKV_FUSED + QK_POST: _qk_norm_rope on the fused projection's output [L, 3F]."""
    F = bq.shape[1]
    return _qk_norm_rope_body((qkv, 0), (qkv, F), bq, bk, ss, wq, wk, cos, sin, inv_dim, eps)


def _qk_norm_rope_body(qs, ks, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps):
    q, k = qs[0], ks[0]
    L = q.shape[0]
    F = bq.shape[1]
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
            src, c0 = qs if j == 0 else ks
            dst = oq if j == 0 else ok
            x = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:sz, 0:F], src=src[nl.ds(r0, sz), c0:c0 + F])
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


@nki.jit
def _x_ss(x, b):
    """One tensor's (x + b)^2 row sums [L, 1] (LINGBOT_QK_POST=2: q's chain ahead of k's)."""
    L, F = x.shape
    ss = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = _bcast_rows(b, F, b.dtype)
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        acc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        xt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:sz, 0:F], src=x[nl.ds(r_base + t * P, sz), 0:F])
        y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=xt[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
        sq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:sz, 0:F], op=nl.square, data=y[0:sz, 0:F], reduce_op=nl.add,
                        reduce_res=acc[0:sz, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.dma_copy(dst=ss[nl.ds(r_base + t * P, sz), 0:1], src=acc[0:sz, 0:1])
    return ss


@nki.jit
def _x_norm_rope(x, b, ss, w, cos, sin, inv_dim, eps):
    """One tensor's bias + RMSNorm + RoPE (the per-tensor half of _qk_norm_rope)."""
    L, F = x.shape
    C = cos.shape[1]
    H = F // (2 * C)
    out = nl.ndarray((L, F), dtype=x.dtype, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = _bcast_rows(b, F, b.dtype)
    w_sb = _bcast_rows(w, F, w.dtype)
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        r0 = r_base + t * P
        cs = nl.ndarray((P, C), dtype=cos.dtype, buffer=nl.sbuf)
        sn = nl.ndarray((P, C), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs[0:sz, 0:C], src=cos[nl.ds(r0, sz), 0:C])
        nisa.dma_copy(dst=sn[0:sz, 0:C], src=sin[nl.ds(r0, sz), 0:C])
        s1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s1[0:sz, 0:1], src=ss[nl.ds(r0, sz), 0:1])
        rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr[0:sz, 0:1], op=nl.rsqrt, data=s1[0:sz, 0:1], scale=inv_dim, bias=eps)
        cb = cs[0:sz, 0:C].expand_dim(1).broadcast(1, H)
        sb = sn[0:sz, 0:C].expand_dim(1).broadcast(1, H)
        xt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:sz, 0:F], src=x[nl.ds(r0, sz), 0:F])
        y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=xt[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
        tn = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=tn[0:sz, 0:F], op=nl.copy, data=y[0:sz, 0:F], scale=rr[0:sz, 0:1])
        tw = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tw[0:sz, 0:F], data1=tn[0:sz, 0:F], data2=w_sb[0:sz, 0:F], op=nl.multiply)
        t4 = tw[0:sz, 0:F].reshape((sz, H, C, 2))
        ev, od = t4.select(3, 0), t4.select(3, 1)
        a = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        bb = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        o = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        o4 = o[0:sz, 0:F].reshape((sz, H, C, 2))
        nisa.tensor_tensor(dst=a[0:sz], data1=ev, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=bb[0:sz], data1=od, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=o4.select(3, 0), data1=a[0:sz], data2=bb[0:sz], op=nl.subtract)
        c2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        d2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c2[0:sz], data1=ev, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d2[0:sz], data1=od, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=o4.select(3, 1), data1=c2[0:sz], data2=d2[0:sz], op=nl.add)
        nisa.dma_copy(dst=out[nl.ds(r0, sz), 0:F], src=o[0:sz, 0:F])
    return out


@nki.jit
def _qk_norm_rope_t(q, k, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps, q_scale):
    return _qk_norm_rope_t_body((q, 0), (k, 0), bq, bk, ss, wq, wk, cos, sin, inv_dim, eps, q_scale)


@nki.jit
def _qk_norm_rope_t3(qkv, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps, q_scale):
    """LINGBOT_MX_QKV_FUSED + QK_POST_T: _qk_norm_rope_t on the fused projection's output [L, 3F]."""
    F = bq.shape[1]
    return _qk_norm_rope_t_body((qkv, 0), (qkv, F), bq, bk, ss, wq, wk, cos, sin, inv_dim, eps, q_scale)


def _qk_norm_rope_t_body(qs, ks, bq, bk, ss, wq, wk, cos, sin, inv_dim, eps, q_scale):
    """LINGBOT_QK_POST_T: _qk_norm_rope with head-major, d-major outputs (q_t, k_t) [H, D, L] bf16, q_t scaled by
    q_scale (1 / sqrt(D), applied through q's cos / sin): the layouts the constant-offset attention reads, so neither
    cmax_prep_q's scale + transpose nor the K cache-layout transpose runs between the kernels. Each 128-row tile's
    heads are transposed on the Tensor engine (idle here) into PSUM, copied out on the Scalar engine into one
    [D, H, rows] buffer per tensor, written once per program (runs of rows x 2 bytes)."""
    q, k = qs[0], ks[0]
    L = q.shape[0]
    F = bq.shape[1]
    C = cos.shape[1]
    D = 2 * C
    H = F // D
    oq = nl.ndarray((H, D, L), dtype=q.dtype, buffer=nl.shared_hbm)
    ok = nl.ndarray((H, D, L), dtype=k.dtype, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = (_bcast_rows(bq, F, bq.dtype), _bcast_rows(bk, F, bk.dtype))
    w_sb = (_bcast_rows(wq, F, wq.dtype), _bcast_rows(wk, F, wk.dtype))
    ob = (nl.ndarray((D, H, rows), dtype=q.dtype, buffer=nl.sbuf), nl.ndarray((D, H, rows), dtype=k.dtype, buffer=nl.sbuf))
    hg = (H + 1) // 2                     # heads per PSUM transpose tile (two groups: 5 + 5 at H = 10)
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
        # q's scale rides on its cos / sin (fp32), so the rotated q is rounded to bf16 once, already scaled
        csq = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        snq = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=csq[0:sz, 0:C], data=cs[0:sz, 0:C], op0=nl.multiply, operand0=q_scale)
        nisa.tensor_scalar(dst=snq[0:sz, 0:C], data=sn[0:sz, 0:C], op0=nl.multiply, operand0=q_scale)
        cbs = (csq[0:sz, 0:C].expand_dim(1).broadcast(1, H), cs[0:sz, 0:C].expand_dim(1).broadcast(1, H))
        sbs = (snq[0:sz, 0:C].expand_dim(1).broadcast(1, H), sn[0:sz, 0:C].expand_dim(1).broadcast(1, H))
        for j in range(2):
            src, c0 = qs if j == 0 else ks
            cb, sb = cbs[j], sbs[j]                              # [sz, H, C]
            x = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x[0:sz, 0:F], src=src[nl.ds(r0, sz), c0:c0 + F])
            y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=x[0:sz, 0:F], data2=b_sb[j][0:sz, 0:F], op=nl.add)
            tn = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.activation(dst=tn[0:sz, 0:F], op=nl.copy, data=y[0:sz, 0:F], scale=rr[0:sz, j:j + 1])
            tw = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tw[0:sz, 0:F], data1=tn[0:sz, 0:F], data2=w_sb[j][0:sz, 0:F], op=nl.multiply)
            t4 = tw[0:sz, 0:F].reshape((sz, H, C, 2))
            ev, od = t4.select(3, 0), t4.select(3, 1)
            a = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            bb = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            o = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            o4 = o[0:sz, 0:F].reshape((sz, H, C, 2))
            nisa.tensor_tensor(dst=a[0:sz], data1=ev, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=bb[0:sz], data1=od, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=o4.select(3, 0), data1=a[0:sz], data2=bb[0:sz], op=nl.subtract)
            c2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            d2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=c2[0:sz], data1=ev, data2=sb, op=nl.multiply)
            nisa.tensor_tensor(dst=d2[0:sz], data1=od, data2=cb, op=nl.multiply)
            nisa.tensor_tensor(dst=o4.select(3, 1), data1=c2[0:sz], data2=d2[0:sz], op=nl.add)
            for g0 in range(0, H, hg):
                ng = min(hg, H - g0)
                pt = nl.ndarray((D, hg * P), dtype=src.dtype, buffer=nl.psum)
                for i in range(ng):
                    h = g0 + i
                    nisa.nc_transpose(dst=pt[0:D, i * P:i * P + sz], data=o[0:sz, h * D:(h + 1) * D],
                                      engine=nisa.engine.tensor)
                pv = pt[0:D, 0:ng * P].reshape((D, ng, P))
                nisa.activation(dst=ob[j][0:D, g0:g0 + ng, nl.ds(t * P, sz)], op=nl.copy, data=pv[0:D, 0:ng, 0:sz])
    for j in range(2):
        dst = oq if j == 0 else ok
        nisa.dma_copy(dst=dst.ap(pattern=[[L, D], [D * L, H], [1, rows]], offset=r_base), src=ob[j][0:D, 0:H, 0:rows])
    return oq, ok


@nki.jit
def _x_ss3(qkv, c0, b):
    """LINGBOT_QK_POST_TSPLIT: one tensor's (x + b)^2 row sums [L, 1] from the fused projection's columns c0 .. c0 + F."""
    L = qkv.shape[0]
    F = b.shape[1]
    ss = nl.ndarray((L, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = _bcast_rows(b, F, b.dtype)
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        acc = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        xt = nl.ndarray((P, F), dtype=qkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:sz, 0:F], src=qkv[nl.ds(r_base + t * P, sz), c0:c0 + F])
        y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=xt[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
        sq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:sz, 0:F], op=nl.square, data=y[0:sz, 0:F], reduce_op=nl.add,
                        reduce_res=acc[0:sz, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.dma_copy(dst=ss[nl.ds(r_base + t * P, sz), 0:1], src=acc[0:sz, 0:1])
    return ss


@nki.jit
def _x_norm_rope_t3(qkv, c0, b, ss, w, cos, sin, inv_dim, eps, scale):
    """LINGBOT_QK_POST_TSPLIT: _qk_norm_rope_t for ONE tensor (the fused projection's columns c0 .. c0 + F, its own
    [L, 1] sums) -> [H, D, L] bf16 scaled by `scale` through cos / sin, so q's chain can finish (and the cached
    chunks' attention start) before k's."""
    L = qkv.shape[0]
    F = b.shape[1]
    C = cos.shape[1]
    D = 2 * C
    H = F // D
    out = nl.ndarray((H, D, L), dtype=qkv.dtype, buffer=nl.shared_hbm)
    rows = L // nl.num_programs(axes=0)
    r_base = nl.program_id(axis=0) * rows
    b_sb = _bcast_rows(b, F, b.dtype)
    w_sb = _bcast_rows(w, F, w.dtype)
    ob = nl.ndarray((D, H, rows), dtype=qkv.dtype, buffer=nl.sbuf)
    hg = (H + 1) // 2
    for t in nl.affine_range((rows + P - 1) // P):
        sz = min(P, rows - t * P)
        r0 = r_base + t * P
        cs = nl.ndarray((P, C), dtype=cos.dtype, buffer=nl.sbuf)
        sn = nl.ndarray((P, C), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=cs[0:sz, 0:C], src=cos[nl.ds(r0, sz), 0:C])
        nisa.dma_copy(dst=sn[0:sz, 0:C], src=sin[nl.ds(r0, sz), 0:C])
        s1 = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=s1[0:sz, 0:1], src=ss[nl.ds(r0, sz), 0:1])
        rr = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rr[0:sz, 0:1], op=nl.rsqrt, data=s1[0:sz, 0:1], scale=inv_dim, bias=eps)
        if scale != 1.0:
            csx = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            snx = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=csx[0:sz, 0:C], data=cs[0:sz, 0:C], op0=nl.multiply, operand0=scale)
            nisa.tensor_scalar(dst=snx[0:sz, 0:C], data=sn[0:sz, 0:C], op0=nl.multiply, operand0=scale)
            cb = csx[0:sz, 0:C].expand_dim(1).broadcast(1, H)
            sb = snx[0:sz, 0:C].expand_dim(1).broadcast(1, H)
        else:
            cb = cs[0:sz, 0:C].expand_dim(1).broadcast(1, H)
            sb = sn[0:sz, 0:C].expand_dim(1).broadcast(1, H)
        x = nl.ndarray((P, F), dtype=qkv.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x[0:sz, 0:F], src=qkv[nl.ds(r0, sz), c0:c0 + F])
        y = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y[0:sz, 0:F], data1=x[0:sz, 0:F], data2=b_sb[0:sz, 0:F], op=nl.add)
        tn = nl.ndarray((P, F), dtype=qkv.dtype, buffer=nl.sbuf)
        nisa.activation(dst=tn[0:sz, 0:F], op=nl.copy, data=y[0:sz, 0:F], scale=rr[0:sz, 0:1])
        tw = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tw[0:sz, 0:F], data1=tn[0:sz, 0:F], data2=w_sb[0:sz, 0:F], op=nl.multiply)
        t4 = tw[0:sz, 0:F].reshape((sz, H, C, 2))
        ev, od = t4.select(3, 0), t4.select(3, 1)
        a = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        bb = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        o = nl.ndarray((P, F), dtype=qkv.dtype, buffer=nl.sbuf)
        o4 = o[0:sz, 0:F].reshape((sz, H, C, 2))
        nisa.tensor_tensor(dst=a[0:sz], data1=ev, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=bb[0:sz], data1=od, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=o4.select(3, 0), data1=a[0:sz], data2=bb[0:sz], op=nl.subtract)
        c2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        d2 = nl.ndarray((P, H, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c2[0:sz], data1=ev, data2=sb, op=nl.multiply)
        nisa.tensor_tensor(dst=d2[0:sz], data1=od, data2=cb, op=nl.multiply)
        nisa.tensor_tensor(dst=o4.select(3, 1), data1=c2[0:sz], data2=d2[0:sz], op=nl.add)
        for g0 in range(0, H, hg):
            ng = min(hg, H - g0)
            pt = nl.ndarray((D, hg * P), dtype=qkv.dtype, buffer=nl.psum)
            for i in range(ng):
                h = g0 + i
                nisa.nc_transpose(dst=pt[0:D, i * P:i * P + sz], data=o[0:sz, h * D:(h + 1) * D],
                                  engine=nisa.engine.tensor)
            pv = pt[0:D, 0:ng * P].reshape((D, ng, P))
            nisa.activation(dst=ob[0:D, g0:g0 + ng, nl.ds(t * P, sz)], op=nl.copy, data=pv[0:D, 0:ng, 0:sz])
    nisa.dma_copy(dst=out.ap(pattern=[[L, D], [D * L, H], [1, rows]], offset=r_base), src=ob[0:D, 0:H, 0:rows])
    return out


qk_ss = wrap_nki(_qk_ss)
qk_norm_rope = wrap_nki(_qk_norm_rope)
qk_norm_rope3 = wrap_nki(_qk_norm_rope3)
qk_norm_rope_t = wrap_nki(_qk_norm_rope_t)
qk_ss3 = wrap_nki(_qk_ss3)
qk_norm_rope_t3 = wrap_nki(_qk_norm_rope_t3)
x_ss = wrap_nki(_x_ss)
x_norm_rope = wrap_nki(_x_norm_rope)
x_ss3 = wrap_nki(_x_ss3)
x_norm_rope_t3 = wrap_nki(_x_norm_rope_t3)
