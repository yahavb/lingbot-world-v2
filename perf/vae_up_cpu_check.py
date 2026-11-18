"""CPU check of kernels/vae_up_conv.py's index math (no device): the flat-shift phase formulation with
up_conv_weights' layout, emulated in fp64 on a haloed low-res tile, vs F.conv2d on the cropped x2 upsample.
    python3 perf/vae_up_cpu_check.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

A = ((0, 1, 1), (1, 1, 2))


def weights(w, fold):
    # same as kernels/vae_up_conv.up_conv_weights without the bf16 cast
    co, c = int(w.shape[0]), int(w.shape[1])
    if not fold:
        return w.permute(1, 2, 3, 0).reshape(c, 9 * co)
    grp = (((0,), (1, 2)), ((0, 1), (2,)))
    taps = []
    for py in (0, 1):
        for px in (0, 1):
            for u in (0, 1):
                for v in (0, 1):
                    taps.append(sum(w[:, :, kh, kw] for kh in grp[py][u] for kw in grp[px][v]))
    return torch.stack(taps, 0).permute(2, 0, 1).reshape(c, 16 * co)


def emulate(xh, wt, bias, co, fold):
    C, Hp, Wp = xh.shape
    H, W = Hp - 2, Wp - 2
    xf = torch.cat([xh.reshape(C, Hp * Wp), torch.zeros(C, 4, dtype=xh.dtype)], 1)
    y = torch.zeros(co, 2 * H, 2 * W, dtype=xh.dtype)
    n = H * Wp
    for py in (0, 1):
        for px in (0, 1):
            ph = 2 * py + px
            if fold:
                taps = [(4 * ph + 2 * u + v, (py + u) * Wp + (px + v)) for u in (0, 1) for v in (0, 1)]
            else:
                taps = [(3 * kh + kw, A[py][kh] * Wp + A[px][kw]) for kh in range(3) for kw in range(3)]
            acc = torch.zeros(co, n, dtype=xh.dtype)
            for ti, off in taps:
                acc += wt[:, ti * co:(ti + 1) * co].t() @ xf[:, off:off + n]
            acc = acc.reshape(co, H, Wp)[:, :, :W] + bias.view(co, 1, 1)
            y[:, py::2, px::2] = acc
    return y


torch.manual_seed(0)
C, co, H, W = 8, 5, 6, 7
xh = torch.randn(C, H + 2, W + 2, dtype=torch.float64)
w = torch.randn(co, C, 3, 3, dtype=torch.float64)
b = torch.randn(co, dtype=torch.float64)
up = xh[:, :, None, :, None].expand(C, H + 2, 2, W + 2, 2).reshape(C, 2 * H + 4, 2 * W + 4)[:, 1:-1, 1:-1]
ref = F.conv2d(up[None], w, b)[0]
for fold in (False, True):
    y = emulate(xh, weights(w, fold), b, co, fold)
    print(f"UPCPU fold={fold} out={tuple(y.shape)} ref={tuple(ref.shape)} max_abs={(y - ref).abs().max().item():.3e}")
