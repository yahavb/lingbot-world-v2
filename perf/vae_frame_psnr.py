"""Per-frame PSNR of a decode's frames.pt against R's (the gate reference), and the 4 x 4 tile map of the error.
    python3 perf/vae_frame_psnr.py <frames.pt> [R frames.pt]
Prints the overall PSNR, the frames below 50 dB (index:dB), and per-tile PSNR over those frames (rows of the 4 x 4 2-D
decode tiling), so a decode error can be placed in time (which decode steps) and space (which tile edges).
"""
import math
import sys

import torch


def psnr(mse):
    return 10 * math.log10(255 ** 2 / max(float(mse), 1e-12))


def main():
    f = torch.load(sys.argv[1]).float()
    r = torch.load(sys.argv[2] if len(sys.argv) > 2 else "/node-cache/e2e/gate/r480_ex03_s42/frames.pt").float()
    f = f.permute(1, 2, 3, 0) if f.shape[0] == 3 else f
    r = r.permute(1, 2, 3, 0) if r.shape[0] == 3 else r
    d = (f - r).pow(2)                                   # [T, H, W, 3]
    per = d.mean((1, 2, 3))
    bad = [i for i, v in enumerate(per.tolist()) if psnr(v) < 50]
    print(f"FRAMES {sys.argv[1]} all {psnr(d.mean()):.2f} dB, frames < 50 dB: "
          + (" ".join(f"{i}:{psnr(per[i]):.1f}" for i in bad) or "none"), flush=True)
    if bad:
        m = d[bad].mean((0, 3))
        H, W = m.shape
        t = m.reshape(4, H // 4, 4, W // 4).mean((1, 3))
        print("TILES " + " | ".join(" ".join(f"{psnr(v):.1f}" for v in row.tolist()) for row in t), flush=True)


if __name__ == "__main__":
    main()
