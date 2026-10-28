"""Compare dumped final latents of two runs: python3 kiro/cmp_latents.py <ref.pt> <test.pt> [more.pt ...]
Prints rel-RMS and max-abs per latent frame chunk and overall."""
import sys

import torch

ref = torch.load(sys.argv[1]).float()
for p in sys.argv[2:]:
    t = torch.load(p).float()
    assert t.shape == ref.shape, (t.shape, ref.shape)
    d = t - ref
    rel = (d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    per_f = [(d[:, f].pow(2).mean().sqrt() / ref[:, f].pow(2).mean().sqrt()).item() for f in range(ref.shape[1])]
    print(f"{p}: shape {tuple(t.shape)} rel_rms {rel:.5f} max_abs {d.abs().max().item():.4f} "
          f"per-frame rel_rms first/mid/last {per_f[0]:.4f}/{per_f[len(per_f) // 2]:.4f}/{per_f[-1]:.4f} "
          f"max {max(per_f):.4f}")
