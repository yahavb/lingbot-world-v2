"""Gate calibration table: each output dir vs a reference R (first-9-frame PSNR, whole-video PSNR,
sharpness ratio, c0s0 / chunk-0 x0 rel-RMS). A blurred copy of the first dir is added as a synthetic
negative (Gaussian sigma=1.0 and 2.0 px).
    python3 kiro/e2e/calib.py <R dir> <dir> ...   (dirs hold frames.pt [+ forward_*.pt])"""
import math, os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from validate_gate import frames, sharpness, psnr, rel  # noqa: E402
R = sys.argv[1]; dirs = sys.argv[2:]
fr = frames(R); sr = sharpness(fr)
def blur(x, s):
    k = torch.arange(-3, 4).float(); k = torch.exp(-k ** 2 / (2 * s * s)); k = k / k.sum()
    y = x.permute(0, 3, 1, 2).float()
    c = y.shape[1]
    y = F.conv2d(F.pad(y, (3, 3, 0, 0), mode="replicate"), k.view(1, 1, 1, 7).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(F.pad(y, (0, 0, 3, 3), mode="replicate"), k.view(1, 1, 7, 1).repeat(c, 1, 1, 1), groups=c)
    return y.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
rows = [(d, frames(d)) for d in dirs]
rows += [(f"{dirs[0]} BLUR s={s}", blur(rows[0][1], s)) for s in (1.0, 2.0)]
print(f"{'output':70s} {'early9 dB':>9s} {'all dB':>7s} {'sharp':>6s} {'c0s0 rel':>9s} {'x0c0 rel':>9s}")
for d, f in rows:
    e = [float("nan")] * 2
    for i, t in enumerate(("c0s0", "x0c0")):
        p, q = os.path.join(d, f"forward_{t}.pt"), os.path.join(R, f"forward_{t}.pt")
        if os.path.exists(p) and os.path.exists(q):
            e[i] = rel(torch.load(p), torch.load(q))
    print(f"{d[-70:]:70s} {psnr(f[:9], fr[:9]):9.2f} {psnr(f, fr):7.2f} {sharpness(f) / sr:6.3f} {e[0]:9.4f} {e[1]:9.4f}")
