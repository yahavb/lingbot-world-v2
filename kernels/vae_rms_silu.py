"""Fused RMS_norm + SiLU of the VAE decoder (LINGBOT_VAE_NKI_NORM=1, wan/modules/vae2_1_fused.py).

  rms_silu(x, g) -> y: x [C, N] bf16 (channels major: a [1, C, T, H, W] activation viewed as [C, T*H*W]),
                       g [C, 1] fp32 (RMS_norm's scale * gamma),
                       y [C, N] bf16 = silu(x * rsqrt(sum_c x^2 + 1e-24) * g)

F.silu(RMS_norm(x)).to(bf16) in one pass over x. The compiler's form (LINGBOT_VAE_NORM=mm) puts the channel sum of
squares on one partition (a [C -> 1] fp32 matmul), its rsqrt on GpSimd (385 us per stage-3 ResidualBlock call in
the 2-chip profile) and the broadcast back plus the elementwise chain in separate passes. Here every tile of
positions is: square (Vector), the channel sum as a matmul against a ones [C, C] matrix, so the sum lands on every
channel's partition (Tensor, fp32 PSUM), rsqrt straight out of PSUM (Scalar), x * rsqrt (Vector, fp32), and
silu(t * g) with the per-channel g as the activation's scale, written as bf16 (Scalar). C > 128 is split into
partition tiles of 128 (the matmul accumulates over them). 1e-24 is F.normalize's eps^2 (eps = 1e-12): the
reference divides by max(||x||, eps), this adds eps^2 under the root; the two differ only for ||x|| ~ eps.

LINGBOT_VAE_NKI_NORM_SQ=bf16 (default): the squares in bf16 and a bf16 matmul (one PE pass; each square rounds to
2^-9 relative, independently per channel, and the sum accumulates in fp32 PSUM, so the sum is good to ~1e-4
relative); fp32: fp32 squares and an fp32 matmul (four PE passes). LINGBOT_VAE_NKI_NORM_F: positions per DMA tile
(default 2048; the matmul and the rsqrt work in 512-column PSUM banks).

Measured (perf/vae_norm_bench.py, one core, [96, 4 x 120 x 192]): torch mm form 608 us, this kernel 276 us (bf16
squares, F=2048), 331 us (fp32, F=2048), 300 us (fp32, F=3072), 671 us at F=512. In the 16-rank decode it did not
help (813 vs 815 ms at T=6; +18 ms on top of LINGBOT_VAE_NKI_HALO at T=1): see kernels/vae_norm_halo.py.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_B = 512          # one PSUM bank of fp32 per partition
_F = int(os.environ.get("LINGBOT_VAE_NKI_NORM_F", "2048"))
_SQ32 = os.environ.get("LINGBOT_VAE_NKI_NORM_SQ", "bf16") == "fp32"


def _tile(x, out, ones, gs, C, n0, fs):
    nct = (C + _P - 1) // _P
    sdt = nl.float32 if _SQ32 else nl.bfloat16
    xs, sqs = [], []
    for ct in range(nct):
        cs = min(_P, C - ct * _P)
        xt = nl.ndarray((_P, _F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:cs, 0:fs], src=x[ct * _P:ct * _P + cs, nl.ds(n0, fs)])
        sq = nl.ndarray((_P, _F), dtype=sdt, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq[0:cs, 0:fs], data1=xt[0:cs, 0:fs], data2=xt[0:cs, 0:fs], op=nl.multiply)
        xs.append(xt)
        sqs.append(sq)
    nb = (fs + _B - 1) // _B
    for co in range(nct):
        cso = min(_P, C - co * _P)
        inv = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.sbuf)
        for k in range(nb):
            k0 = k * _B
            ks = min(_B, fs - k0)
            ps = nl.ndarray((_P, _B), dtype=nl.float32, buffer=nl.psum)
            for ct in range(nct):
                cs = min(_P, C - ct * _P)
                nisa.nc_matmul(dst=ps[0:cso, 0:ks], stationary=ones[0:cs, 0:cso], moving=sqs[ct][0:cs, k0:k0 + ks],
                               accumulate=(ct > 0))
            nisa.activation(dst=inv[0:cso, k0:k0 + ks], op=nl.rsqrt, data=ps[0:cso, 0:ks], scale=1.0, bias=1e-24)
        t = nl.ndarray((_P, _F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t[0:cso, 0:fs], data1=xs[co][0:cso, 0:fs], data2=inv[0:cso, 0:fs], op=nl.multiply)
        y = nl.ndarray((_P, _F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.activation(dst=y[0:cso, 0:fs], op=nl.silu, data=t[0:cso, 0:fs], scale=gs[co][0:cso, 0:1])
        nisa.dma_copy(dst=out[co * _P:co * _P + cso, nl.ds(n0, fs)], src=y[0:cso, 0:fs])


@nki.jit
def _rms_silu(x, g):
    C, N = x.shape
    out = nl.ndarray((C, N), dtype=x.dtype, buffer=nl.shared_hbm)
    nct = (C + _P - 1) // _P
    ones = nl.ndarray((_P, _P), dtype=nl.float32 if _SQ32 else nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    gs = []
    for ct in range(nct):
        cs = min(_P, C - ct * _P)
        gt = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gt[0:cs, 0:1], src=g[ct * _P:ct * _P + cs, 0:1])
        gs.append(gt)
    nfull = N // _F
    for i in nl.affine_range(nfull):
        _tile(x, out, ones, gs, C, i * _F, _F)
    if N % _F:
        _tile(x, out, ones, gs, C, nfull * _F, N % _F)
    return out


rms_silu = wrap_nki(_rms_silu)        # traceable inside a fullgraph torch.compile region
