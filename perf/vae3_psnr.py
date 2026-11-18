"""perf/vae3_psnr.py <base tag> <tag> ...: 4-decimal vs_R (as perf/vae_env_check_here.sh scores it) of the frames each
tag saved in /work/out/vaecheck/<tag>/frames.pt, and the max |diff| / number of differing values against the base tag
(0 = bit-identical frames)."""
import math
import sys

import torch

R = "/node-cache/e2e/gate/r480_ex03_s42/frames.pt"


def load(p):
    f = torch.load(p).float()
    return f.permute(1, 2, 3, 0) if f.shape[0] == 3 else f


r = load(R)
base = None
for i, tag in enumerate(sys.argv[1:]):
    f = load(f"/work/out/vaecheck/{tag}/frames.pt")
    m = (f - r).pow(2).mean().item()
    s = f"PSNR {tag} vs_R={10 * math.log10(255 ** 2 / m):.4f} dB"
    if base is None:
        base = f
    else:
        d = (f - base).abs()
        s += f" vs_base max_abs={d.max().item():.1f} n_diff={(d > 0).sum().item()}"
    print(s, flush=True)
