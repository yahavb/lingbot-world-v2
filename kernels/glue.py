"""The DiT block's residual / modulation / LayerNorm steps between the matmul kernels, as NKI kernels on this rank's
token shard [S, C] (LINGBOT_GLUE=1). The 2-chip block profile had each of these chains as ~10 dependent compiler
ops (LayerNorm statistics as ones-matmuls on the Tensor engine, rsqrt on GpSimd), 110-190 us each between the
collectives, with no engine above 40% busy.

Tokens on partitions, whole rows of C features: bn_stats / bn_aggr (Vector) for the row mean and variance, one
Scalar pass for (x - mean) * rstd, Vector passes for the per-feature modulation; fp32 math, LNC2 splits the rows.
  ln_mod(x, s1, b0)                     h = LN(x) * s1 + b0                          -> h bf16
  res_film_ln(xs, y, g, cs, csh)        x' = (1 + cs) * (xs + y * g) + csh; LN(x')   -> (x', h) (no LN affine)
  res_ln_mod(xs, yc, s1, b0)            x' = xs + yc; LN(x') * s1 + b0              -> (x', h)
  res_gate(xs, y, g)                    xs + y * g                                    -> x'
Per-feature vectors (s1 = 1 + e, b0, g) come as [1, C] fp32 rows; cs / csh / y / yc as [S, C].
"""
import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_BN = 512   # nl.tile_size.bn_stats_fmax
_EPS = 1e-6


def _rows(S):
    nprog = nl.num_programs(axes=0)
    rows = S // nprog
    return rows, nl.program_id(axis=0) * rows


def _bcast(vec, C):
    t = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=vec.ap(pattern=[[0, _P], [1, C]], offset=0))
    return t


def _load(src, r0, sz, C):
    t = nl.ndarray((_P, C), dtype=src.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t[0:sz, 0:C], src=src[nl.ds(r0, sz), 0:C])
    return t


def _layernorm(x32, sz, C, eps, out_dtype=nl.float32):
    """x32 [P, C] fp32 -> (x - mean) * rstd in out_dtype (Vector statistics, one Scalar pass)."""
    nb = C // _BN
    st = nl.ndarray((_P, nb, 6), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(nb):
        nisa.bn_stats(dst=st[0:sz, j, :], data=x32[0:sz, nl.ds(j * _BN, _BN)])
    mv = nl.ndarray((_P, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.bn_aggr(dst=mv[0:sz, :], data=st[0:sz, :, :])
    rstd = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd[0:sz], data=mv[0:sz, 1:2], op0=nl.add, operand0=eps)
    nisa.activation(dst=rstd[0:sz], op=nl.rsqrt, data=rstd[0:sz])
    nb_ = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)     # -mean * rstd
    nisa.scalar_tensor_tensor(dst=nb_[0:sz], data=mv[0:sz, 0:1], op0=nl.multiply, operand0=-1.0,
                              op1=nl.multiply, operand1=rstd[0:sz])
    n = nl.ndarray((_P, C), dtype=out_dtype, buffer=nl.sbuf)
    nisa.activation(dst=n[0:sz, 0:C], op=nl.copy, data=x32[0:sz, 0:C], scale=rstd[0:sz], bias=nb_[0:sz])
    return n


def _modulate(n, s1_t, b0_t, sz, C, out_dtype):
    h = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=h[0:sz, 0:C], data1=n[0:sz, 0:C], data2=s1_t[0:sz, 0:C], op=nl.multiply)
    hb = nl.ndarray((_P, C), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=hb[0:sz, 0:C], data1=h[0:sz, 0:C], data2=b0_t[0:sz, 0:C], op=nl.add)
    return hb


@nki.jit
def _ln_mod(x, s1, b0):
    S, C = x.shape
    h_out = nl.ndarray((S, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    rows, r_base = _rows(S)
    s1_t, b0_t = _bcast(s1, C), _bcast(b0, C)
    for t in range((rows + _P - 1) // _P):
        sz = min(_P, rows - t * _P)
        r0 = r_base + t * _P
        xt = _load(x, r0, sz, C)
        if x.dtype != nl.float32:
            x32 = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x32[0:sz, 0:C], src=xt[0:sz, 0:C], engine=nisa.engine.scalar)
        else:
            x32 = xt
        n = _layernorm(x32, sz, C, _EPS)
        hb = _modulate(n, s1_t, b0_t, sz, C, nl.bfloat16)
        nisa.dma_copy(dst=h_out[nl.ds(r0, sz), 0:C], src=hb[0:sz, 0:C])
    return h_out


@nki.jit
def _res_film_ln(xs, y, g, cs, csh):
    S, C = xs.shape
    x_out = nl.ndarray((S, C), dtype=xs.dtype, buffer=nl.shared_hbm)
    h_out = nl.ndarray((S, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    rows, r_base = _rows(S)
    g_t = _bcast(g, C)
    for t in range((rows + _P - 1) // _P):
        sz = min(_P, rows - t * _P)
        r0 = r_base + t * _P
        xt, yt, ct, dt_ = _load(xs, r0, sz, C), _load(y, r0, sz, C), _load(cs, r0, sz, C), _load(csh, r0, sz, C)
        a = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a[0:sz, 0:C], data1=yt[0:sz, 0:C], data2=g_t[0:sz, 0:C], op=nl.multiply)
        nisa.tensor_tensor(dst=a[0:sz, 0:C], data1=a[0:sz, 0:C], data2=xt[0:sz, 0:C], op=nl.add)
        b = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)       # (cs + 1) * a
        nisa.scalar_tensor_tensor(dst=b[0:sz, 0:C], data=ct[0:sz, 0:C], op0=nl.add, operand0=1.0,
                                  op1=nl.multiply, operand1=a[0:sz, 0:C])
        xo = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xo[0:sz, 0:C], data1=b[0:sz, 0:C], data2=dt_[0:sz, 0:C], op=nl.add)
        if xs.dtype != nl.float32:
            xob = nl.ndarray((_P, C), dtype=xs.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xob[0:sz, 0:C], src=xo[0:sz, 0:C], engine=nisa.engine.gpsimd)
            nisa.dma_copy(dst=x_out[nl.ds(r0, sz), 0:C], src=xob[0:sz, 0:C])
        else:
            nisa.dma_copy(dst=x_out[nl.ds(r0, sz), 0:C], src=xo[0:sz, 0:C])
        hb = _layernorm(xo, sz, C, _EPS, nl.bfloat16)
        nisa.dma_copy(dst=h_out[nl.ds(r0, sz), 0:C], src=hb[0:sz, 0:C])
    return x_out, h_out


@nki.jit
def _res_ln_mod(xs, yc, s1, b0):
    S, C = xs.shape
    x_out = nl.ndarray((S, C), dtype=xs.dtype, buffer=nl.shared_hbm)
    h_out = nl.ndarray((S, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    rows, r_base = _rows(S)
    s1_t, b0_t = _bcast(s1, C), _bcast(b0, C)
    for t in range((rows + _P - 1) // _P):
        sz = min(_P, rows - t * _P)
        r0 = r_base + t * _P
        xt, yt = _load(xs, r0, sz, C), _load(yc, r0, sz, C)
        xo = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xo[0:sz, 0:C], data1=xt[0:sz, 0:C], data2=yt[0:sz, 0:C], op=nl.add)
        if xs.dtype != nl.float32:
            xob = nl.ndarray((_P, C), dtype=xs.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xob[0:sz, 0:C], src=xo[0:sz, 0:C], engine=nisa.engine.gpsimd)
            nisa.dma_copy(dst=x_out[nl.ds(r0, sz), 0:C], src=xob[0:sz, 0:C])
        else:
            nisa.dma_copy(dst=x_out[nl.ds(r0, sz), 0:C], src=xo[0:sz, 0:C])
        n = _layernorm(xo, sz, C, _EPS)
        hb = _modulate(n, s1_t, b0_t, sz, C, nl.bfloat16)
        nisa.dma_copy(dst=h_out[nl.ds(r0, sz), 0:C], src=hb[0:sz, 0:C])
    return x_out, h_out


@nki.jit
def _res_gate(xs, y, g):
    S, C = xs.shape
    x_out = nl.ndarray((S, C), dtype=xs.dtype, buffer=nl.shared_hbm)
    rows, r_base = _rows(S)
    g_t = _bcast(g, C)
    for t in range((rows + _P - 1) // _P):
        sz = min(_P, rows - t * _P)
        r0 = r_base + t * _P
        xt, yt = _load(xs, r0, sz, C), _load(y, r0, sz, C)
        a = nl.ndarray((_P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=a[0:sz, 0:C], data1=yt[0:sz, 0:C], data2=g_t[0:sz, 0:C], op=nl.multiply)
        xo = nl.ndarray((_P, C), dtype=xs.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=xo[0:sz, 0:C], data1=a[0:sz, 0:C], data2=xt[0:sz, 0:C], op=nl.add)
        nisa.dma_copy(dst=x_out[nl.ds(r0, sz), 0:C], src=xo[0:sz, 0:C])
    return x_out


ln_mod = wrap_nki(_ln_mod)
res_film_ln = wrap_nki(_res_film_ln)
res_ln_mod = wrap_nki(_res_ln_mod)
res_gate = wrap_nki(_res_gate)
