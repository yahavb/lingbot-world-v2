"""Compare per-block outputs dumped with LINGBOT_DUMP_BLOCKS by two runs.
    python3 kiro/cmp_blocks.py <dir_a> <dir_b>
Prints, per block: equal, relative RMS difference, max |diff|."""
import os
import sys

import torch

a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sorted(set(os.listdir(a_dir)) & set(os.listdir(b_dir)))
for nm in names:
    a = torch.load(os.path.join(a_dir, nm)).double()
    b = torch.load(os.path.join(b_dir, nm)).double()
    d = a - b
    print(f"{nm:8s} equal {torch.equal(a, b)!s:5s} rel_rms {(d.pow(2).mean().sqrt() / a.pow(2).mean().sqrt()).item():.3e}"
          f"  max|d| {d.abs().max().item():.3e}  ndiff {int((d != 0).sum())}/{d.numel()}")
