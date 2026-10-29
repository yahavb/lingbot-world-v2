"""Compare per-forward outputs dumped with LINGBOT_DUMP_PRED by two runs.
    python3 kiro/cmp_dump.py <dir_a> <dir_b>
Prints, per tag present in both: relative RMS difference, max |diff|, and RMS of the reference.
"""
import os
import sys

import torch

a_dir, b_dir = sys.argv[1], sys.argv[2]
tags = sorted(set(f[:-3] for f in os.listdir(a_dir) if f.endswith(".pt"))
              & set(f[:-3] for f in os.listdir(b_dir) if f.endswith(".pt")),
              key=lambda t: (int(t[1:].split("s")[0]), t))
for t in tags:
    a = torch.load(os.path.join(a_dir, t + ".pt")).double()
    b = torch.load(os.path.join(b_dir, t + ".pt")).double()
    d = a - b
    rms = a.pow(2).mean().sqrt().item()
    print(f"{t:8s} rel_rms {(d.pow(2).mean().sqrt().item() / max(rms, 1e-30)):.3e}  "
          f"max|d| {d.abs().max().item():.3e}  ref_rms {rms:.3e}  equal {torch.equal(a, b)}")
