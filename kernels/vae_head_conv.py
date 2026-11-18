"""The VAE decoder's output conv (96 -> 3, causal 3x3x3, bf16) as an NKI kernel (LINGBOT_VAE_NKI_HEAD=1,
wan/modules/vae2_1_fused.py _head_unit, on the conv input kernels/vae_norm_halo.py builds).

  head_conv(buf, wt, bias) -> y [CO, T, H, W] bf16:
      buf [C, T + 2, H + 2, W + 2] bf16   2 cache frames + T new frames, normalized, SiLU'd, haloed (C <= 128)
      wt [C, 27 * CO] bf16                 wt[c, tap * CO + co] = w[co, c, kt, kh, kw], tap = 9 kt + 3 kh + kw
      bias [CO, 1] fp32
  y[co, t, h, w] = bias[co] + sum_{c, kt, kh, kw} w[co, c, kt, kh, kw] buf[c, t + kt, h + kh, w + kw]

Flat-shift form: in a band of R output rows, every row is computed over the padded width Wp = W + 2 (the last two
columns of each row are discarded), so output position q = r * Wp + c of frame t reads buf frame t + kt at the flat
position q + kh * Wp + kw of the band's input rows: each tap is one matmul with the [C, CO] tap weight as the
stationary and a plain offset slice of the band in SBUF as the moving operand (no shifted copies), 27 matmuls
accumulating in one fp32 PSUM bank of 512 positions. Bias add + bf16 cast on the PSUM evacuation (Scalar).
LINGBOT_VAE_NKI_HEAD_R: output rows per band (default 20; must divide H, else the largest divisor below it). Read
inside the kernel, so it is not part of the NKI / NEFF cache key: a process with another value can reuse a cached
kernel (kernels/vae_norm_halo_conv.py passes its settings as constant arguments instead); only the default was
measured reliably.
"""
import os

import nki
import nki.isa as nisa
import nki.language as nl
from torch_neuronx.nki_hop import wrap_nki

_P = 128
_B = 512          # one PSUM bank of fp32 per partition
_R = int(os.environ.get("LINGBOT_VAE_NKI_HEAD_R", "20"))


def head_conv_weights(w):
    """[CO, C, 3, 3, 3] -> [C, 27 * CO] (traceable; a few thousand elements)."""
    co, c = int(w.shape[0]), int(w.shape[1])
    return w.permute(1, 2, 3, 4, 0).reshape(c, 27 * co).contiguous()


@nki.jit
def _head_conv(buf, wt, bias):
    C, T2, Hp, Wp = buf.shape
    T, H, W = T2 - 2, Hp - 2, Wp - 2
    CO = bias.shape[0]
    assert C <= _P and CO <= _P, (C, CO)
    R = max(r for r in range(1, min(_R, H) + 1) if H % r == 0)
    FL = (R + 2) * Wp
    NQ = R * Wp - 2                    # the band's last valid output is at R * Wp - 3; its last input at FL - 1
    nb = (NQ + _B - 1) // _B
    out = nl.ndarray((CO, T, H, W), dtype=buf.dtype, buffer=nl.shared_hbm)
    ws = nl.ndarray((_P, 27 * CO), dtype=wt.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws[0:C, 0:27 * CO], src=wt[0:C, 0:27 * CO])
    bs = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bs[0:CO, 0:1], src=bias[0:CO, 0:1])
    for j in nl.affine_range(H // R):
        xs = []
        for f in range(T2):
            xf = nl.ndarray((_P, R + 2, Wp), dtype=buf.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xf[0:C, 0:R + 2, 0:Wp], src=buf[0:C, f, nl.ds(j * R, R + 2), 0:Wp])
            xs.append(xf.reshape((_P, FL)))
        for t in range(T):
            ot = nl.ndarray((_P, R, Wp), dtype=buf.dtype, buffer=nl.sbuf)
            otf = ot.reshape((_P, R * Wp))
            for b in range(nb):
                q0 = b * _B
                n = min(_B, NQ - q0)
                ps = nl.ndarray((_P, _B), dtype=nl.float32, buffer=nl.psum)
                for tap in range(27):
                    kt, kh, kw = tap // 9, (tap // 3) % 3, tap % 3
                    s = q0 + kh * Wp + kw
                    nisa.nc_matmul(dst=ps[0:CO, 0:n], stationary=ws[0:C, tap * CO:(tap + 1) * CO],
                                   moving=xs[t + kt][0:C, s:s + n], accumulate=(tap > 0))
                nisa.activation(dst=otf[0:CO, q0:q0 + n], op=nl.copy, data=ps[0:CO, 0:n], bias=bs[0:CO, 0:1],
                                scale=1.0)
            nisa.dma_copy(dst=out[0:CO, t, nl.ds(j * R, R), 0:W], src=ot[0:CO, 0:R, 0:W])
    return out


head_conv = wrap_nki(_head_conv)       # traceable inside a fullgraph torch.compile region
