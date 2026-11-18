"""The VAE decoder's causal-conv input in one pass (LINGBOT_VAE_NKI_HALO=1, wan/modules/vae2_1_fused.py).

  norm_halo(x, left, right, up, down, cache, g) -> buf [C, T + 2, H + 2, W + 2] bf16:
      x [C, T, H, W] bf16       this rank's raw (pre-norm) tile
      left, right [C, T, H, 1]  the raw neighbour columns (zeros at the image edge)
      up, down [C, T, 1, W + 2] the raw neighbour rows of the column-extended tile (corners included)
      cache [C, 2, H + 2, W + 2] the previous step's normalized, haloed last two frames
      g [C, 1] fp32             RMS_norm's scale * gamma
  buf[:, :2] = cache, buf[:, 2:] = silu(rmsnorm(halo(x)) * g): the input of the next causal 3x3x3 conv.

The decoder computes halo(silu(rmsnorm(x))) with the norm first, then two torch.cat copies of the whole tile for the
column and row halos and a third for the cache frames (in the stage-3 ResidualBlock micro-bench the concatenates and
the norms together cost 3.5 of 5.8 ms). The norm is per position, so normalizing the neighbours' raw edges gives
the values the neighbours' own norm gives; here the raw edges are exchanged and the kernel normalizes every position
of the padded frame (interior, edges, corners) the same way, writing each output row once. Per position: square
(Vector, fp32), channel sum as an fp32 matmul against ones (the sum lands on every channel's partition; C > 128 in
partition tiles of 128, accumulated in PSUM), rsqrt out of PSUM (Scalar, + 1e-24 = F.normalize's eps^2),
x * rsqrt (Vector), silu(. * g) to bf16 (Scalar). Zero edges stay zero (0 * rsqrt(1e-24) = 0, silu(0) = 0), which
is the conv's zero padding. Same arithmetic per position as kernels/vae_rms_silu.py with LINGBOT_VAE_NKI_NORM_SQ=fp32.
LINGBOT_VAE_NKI_HALO_F: positions per tile (default 3072, divided by the number of channel tiles).
LINGBOT_VAE_NKI_HALO_EDGE=copy: load each tile's left / right halo columns with a contiguous DMA (R values per
partition) and place them with a GpSimd copy into the strided tile columns, instead of one strided DMA that
writes R single 2-byte elements per partition. Bit-identical; stage-3 ResidualBlock 3497 -> 3342 us in
perf/vae_rb_bench.py, but the decode did not move (645 vs 644-646 ms), so it is not in the best flag set.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_B = 512          # one PSUM bank of fp32 per partition
_F0 = int(os.environ.get("LINGBOT_VAE_NKI_HALO_F", "3072"))
_EDGE_COPY = os.environ.get("LINGBOT_VAE_NKI_HALO_EDGE", "") == "copy"


def _norm(xts, yts, n, cs, ones, gts, F):
    """yts[k][0:cs[k], 0:n] = silu(x * rsqrt(sum_c x^2 + 1e-24) * g) over the channel tiles xts[k] (flat SBUF)."""
    nct = len(cs)
    sqs = []
    for k in range(nct):
        sq = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq[0:cs[k], 0:n], data1=xts[k][0:cs[k], 0:n], data2=xts[k][0:cs[k], 0:n],
                           op=nl.multiply)
        sqs.append(sq)
    for co in range(nct):
        inv = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        for b in range((n + _B - 1) // _B):
            b0 = b * _B
            bs = min(_B, n - b0)
            ps = nl.ndarray((_P, _B), dtype=nl.float32, buffer=nl.psum)
            for k in range(nct):
                nisa.nc_matmul(dst=ps[0:cs[co], 0:bs], stationary=ones[0:cs[k], 0:cs[co]],
                               moving=sqs[k][0:cs[k], b0:b0 + bs], accumulate=(k > 0))
            nisa.activation(dst=inv[0:cs[co], b0:b0 + bs], op=nl.rsqrt, data=ps[0:cs[co], 0:bs], scale=1.0,
                            bias=1e-24)
        t = nl.ndarray((_P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t[0:cs[co], 0:n], data1=xts[co][0:cs[co], 0:n], data2=inv[0:cs[co], 0:n],
                           op=nl.multiply)
        nisa.activation(dst=yts[co][0:cs[co], 0:n], op=nl.silu, data=t[0:cs[co], 0:n], scale=gts[co][0:cs[co], 0:1])


@nki.jit
def _norm_halo(x, left, right, up, down, cache, g):
    C, T, H, W = x.shape
    Hp, Wp = H + 2, W + 2
    nct = (C + _P - 1) // _P
    cs = [min(_P, C - k * _P) for k in range(nct)]
    F = _F0 // nct
    assert Wp <= F, (Wp, F)
    TG = min(T, F // Wp)                                                   # frames per edge-row tile
    R = max(r for r in range(1, H + 1) if H % r == 0 and r * Wp <= F)      # interior rows per tile
    out = nl.ndarray((C, T + 2, Hp, Wp), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out[0:C, 0:2, 0:Hp, 0:Wp], src=cache[0:C, 0:2, 0:Hp, 0:Wp])
    ones = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones, 1.0)
    gts = []
    for k in range(nct):
        gt = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gt[0:cs[k], 0:1], src=g[k * _P:k * _P + cs[k], 0:1])
        gts.append(gt)
    # padded rows 0 and H + 1 of every frame: the up / down rows, TG frames per tile
    for src, r in ((up, 0), (down, Hp - 1)):
        for f0 in range(0, T, TG):
            tg = min(TG, T - f0)
            ets, eys = [], []
            for k in range(nct):
                et = nl.ndarray((_P, F), dtype=x.dtype, buffer=nl.sbuf)
                for t in range(tg):
                    nisa.dma_copy(dst=et[0:cs[k], t * Wp:(t + 1) * Wp], src=src[k * _P:k * _P + cs[k], f0 + t, 0, 0:Wp])
                ets.append(et)
                eys.append(nl.ndarray((_P, F), dtype=x.dtype, buffer=nl.sbuf))
            _norm(ets, eys, tg * Wp, cs, ones, gts, F)
            for k in range(nct):
                for t in range(tg):
                    nisa.dma_copy(dst=out[k * _P:k * _P + cs[k], 2 + f0 + t, r, 0:Wp],
                                  src=eys[k][0:cs[k], t * Wp:(t + 1) * Wp])
    # interior rows 1..H in tiles of R rows (with their left / right halo columns)
    for t in nl.affine_range(T):
        for j in nl.affine_range(H // R):
            xts, yts = [], []
            for k in range(nct):
                c0, c1 = k * _P, k * _P + cs[k]
                xt = nl.ndarray((_P, R, Wp), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt[0:cs[k], 0:R, 1:W + 1], src=x[c0:c1, t, nl.ds(j * R, R), 0:W])
                if _EDGE_COPY:
                    for src, col in ((left, 0), (right, W + 1)):
                        et = nl.ndarray((_P, R, 1), dtype=x.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=et[0:cs[k], 0:R, 0:1], src=src[c0:c1, t, nl.ds(j * R, R), 0:1])
                        nisa.tensor_copy(dst=xt[0:cs[k], 0:R, col:col + 1], src=et[0:cs[k], 0:R, 0:1],
                                         engine=nisa.gpsimd_engine)
                else:
                    nisa.dma_copy(dst=xt[0:cs[k], 0:R, 0:1], src=left[c0:c1, t, nl.ds(j * R, R), 0:1])
                    nisa.dma_copy(dst=xt[0:cs[k], 0:R, W + 1:W + 2], src=right[c0:c1, t, nl.ds(j * R, R), 0:1])
                xts.append(xt)
                yts.append(nl.ndarray((_P, R, Wp), dtype=x.dtype, buffer=nl.sbuf))
            _norm([v.reshape((_P, R * Wp)) for v in xts], [v.reshape((_P, R * Wp)) for v in yts], R * Wp, cs,
                  ones, gts, F)
            for k in range(nct):
                nisa.dma_copy(dst=out[k * _P:k * _P + cs[k], 2 + t, nl.ds(1 + j * R, R), 0:Wp],
                              src=yts[k][0:cs[k], 0:R, 0:Wp])
    return out


norm_halo = wrap_nki(_norm_halo)       # traceable inside a fullgraph torch.compile region
