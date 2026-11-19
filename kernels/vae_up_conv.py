"""The VAE decoder's spatial upsample (nearest-exact x2 + 3x3 Conv2d, Resample's resample[1]) as one NKI kernel on the
low-resolution haloed tile (LINGBOT_VAE_NKI_UP, wan/modules/vae2_1_fused.py _up_core_nki): the 4x upsampled input
is never built.

  up_conv(x, left, right, up, down, wt, bias) -> y [CO, T, 2H, 2W] bf16
      x [C, T, H, W] bf16       this rank's low-resolution tile (raw: the upsample has no norm)
      left, right [C, T, H, 1]  the neighbour columns, up, down [C, T, 1, W + 2] the neighbour rows (with corners),
                                zeros at the image edge (wan/modules/vae2_1_fused.py _raw_edges_2d)
      wt   fold=0: [C, 9 * CO] bf16, wt[c, (3 kh + kw) CO + co] = w[co, c, kh, kw]
           fold=1: [C, 16 * CO] bf16, the folded 2x2 kernel of each output phase (up_conv_weights)
      bias [CO, 1] fp32
  = F.conv2d(crop(up2(halo(x))), w, bias) per frame, crop = [1:-1, 1:-1] of the x2-upsampled haloed tile.

Output row 2i + py of the upsampled frame reads padded low-res rows i + a(py, kh), a(0, .) = (0, 1, 1),
a(1, .) = (1, 1, 2) (the same for columns), so each output phase (py, px) is a 3x3 conv over the low-resolution
padded tile with tap offsets a(py, kh) * Wp + a(px, kw): kernels/vae_head_conv.py's flat-shift matmuls (tap weight
[C, CO] stationary, offset slice of the SBUF band moving, fp32 PSUM over whole low-res rows), the PSUM evacuation
(Scalar, + bias, bf16) writing the phase's rows / columns of the output band with stride 2. fold=0 keeps the 9 taps
(two taps that read the same input row are two matmuls into the same fp32 PSUM, the same products the compiler's conv
on the upsampled input sums); fold=1 sums them in the weights (4 taps per phase, 2.25x fewer matmuls; the folded
weights are summed in fp32 from the fp32 module weights and rounded to bf16 once).
"""
import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_B = 512          # one PSUM bank of fp32 per partition
_A = ((0, 1, 1), (1, 1, 2))


def up_conv_weights(w, fold=False):
    """[CO, C, 3, 3] -> [C, 9 * CO] (tap = 3 kh + kw), or fold: [C, 16 * CO] (index (4 (2 py + px) + 2 u + v) CO + co,
    the folded tap (u, v) of output phase (py, px), summed in fp32). Traceable; call it on the fp32 weight for fold."""
    co, c = int(w.shape[0]), int(w.shape[1])
    if not fold:
        return w.permute(1, 2, 3, 0).reshape(c, 9 * co).contiguous()
    wf = w.float()
    grp = (((0,), (1, 2)), ((0, 1), (2,)))           # kh of folded tap u for phase py
    taps = []
    for py in (0, 1):
        for px in (0, 1):
            for u in (0, 1):
                for v in (0, 1):
                    taps.append(sum(wf[:, :, kh, kw] for kh in grp[py][u] for kw in grp[px][v]))
    wt = torch.stack(taps, 0)                         # [16, CO, C]
    return wt.permute(2, 0, 1).reshape(c, 16 * co).contiguous().to(torch.bfloat16)


@nki.jit
def _up_conv(x, left, right, up, down, wt, bias, rows=10, fold=0):
    C, T, H, W = x.shape
    Hp, Wp = H + 2, W + 2
    CO = bias.shape[0]
    nct, ncot = (C + _P - 1) // _P, (CO + _P - 1) // _P
    cs = [min(_P, C - k * _P) for k in range(nct)]
    cos = [min(_P, CO - k * _P) for k in range(ncot)]
    if H * Wp <= _B + 2:
        R = H
    else:
        R = max(r for r in range(1, min(rows, H) + 1) if H % r == 0)
    NB = H // R
    RB = min(R, (_B + 2) // Wp)                        # low-res rows per PSUM group
    FL = (R + 2) * Wp
    ntap = 16 if fold else 9
    y = nl.ndarray((CO, T, 2 * H, 2 * W), dtype=x.dtype, buffer=nl.shared_hbm)
    wss = []
    for k in range(nct):
        c0 = k * _P
        ws = nl.ndarray((_P, ntap * CO), dtype=wt.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=ws[0:cs[k], 0:ntap * CO], src=wt[c0:c0 + cs[k], 0:ntap * CO])
        wss.append(ws)
    bss = []
    for m in range(ncot):
        bs = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bs[0:cos[m], 0:1], src=bias[m * _P:m * _P + cos[m], 0:1])
        bss.append(bs)
    for t in range(T):
        for j in range(NB):
            h0 = j * R                                     # padded rows h0 .. h0 + R + 1
            p0, p1 = max(h0, 1), min(h0 + R + 2, Hp - 1)  # interior padded rows (raw rows p - 1)
            d0, nr = p0 - h0, p1 - p0
            xts = []
            for k in range(nct):
                c0, c1 = k * _P, k * _P + cs[k]
                xt = nl.ndarray((_P, R + 2, Wp), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, 1:W + 1], src=x[c0:c1, t, p0 - 1:p1 - 1, 0:W])
                nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, 0:1], src=left[c0:c1, t, p0 - 1:p1 - 1, 0:1])
                nisa.dma_copy(dst=xt[0:cs[k], d0:d0 + nr, W + 1:W + 2], src=right[c0:c1, t, p0 - 1:p1 - 1, 0:1])
                if h0 == 0:
                    nisa.dma_copy(dst=xt[0:cs[k], 0, 0:Wp], src=up[c0:c1, t, 0, 0:Wp])
                if h0 + R + 2 == Hp:
                    nisa.dma_copy(dst=xt[0:cs[k], R + 1, 0:Wp], src=down[c0:c1, t, 0, 0:Wp])
                xts.append(xt.reshape((_P, FL)))
            for m in range(ncot):
                m0, ms = m * _P, cos[m]
                ot = nl.ndarray((_P, 2 * R, 2 * W), dtype=x.dtype, buffer=nl.sbuf)
                for py in (0, 1):
                    for px in (0, 1):
                        ph = 2 * py + px
                        for i0 in range(0, R, RB):
                            rb = min(RB, R - i0)
                            n = rb * Wp - (2 if i0 + rb == R else 0)   # the band's last 2 columns read past FL
                            ps = nl.ndarray((_P, RB, Wp), dtype=nl.float32, buffer=nl.psum)
                            psf = ps.reshape((_P, RB * Wp))
                            if fold:
                                taps = [(4 * ph + 2 * u + v, (py + u) * Wp + (px + v)) for u in (0, 1) for v in (0, 1)]
                            else:
                                taps = [(3 * kh + kw, _A[py][kh] * Wp + _A[px][kw]) for kh in range(3) for kw in range(3)]
                            i = 0
                            for ti, off in taps:
                                s = i0 * Wp + off
                                for k in range(nct):
                                    nisa.nc_matmul(dst=psf[0:ms, 0:n],
                                                   stationary=wss[k][0:cs[k], ti * CO + m0:ti * CO + m0 + ms],
                                                   moving=xts[k][0:cs[k], s:s + n], accumulate=(i > 0))
                                    i += 1
                            nisa.activation(dst=ot[0:ms, 2 * i0 + py:2 * (i0 + rb):2, px:2 * W:2], op=nl.copy,
                                            data=ps[0:ms, 0:rb, 0:W], bias=bss[m][0:ms, 0:1], scale=1.0)
                nisa.dma_copy(dst=y[m0:m0 + ms, t, 2 * h0:2 * h0 + 2 * R, 0:2 * W], src=ot[0:ms, 0:2 * R, 0:2 * W])
    return y


_kernel = wrap_nki(_up_conv)


def up_conv(x, left, right, up, down, wt, bias, rows=10, fold=False):
    """Traceable inside a fullgraph torch.compile region; rows / fold go in as constant kernel arguments (part of the
    kernel's cache key)."""
    return _kernel(x, left, right, up, down, wt, bias, rows=int(rows), fold=int(bool(fold)))
