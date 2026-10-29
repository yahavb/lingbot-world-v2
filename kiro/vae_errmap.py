"""Where a VAE dump differs from a reference: per frame, per tile row/column, and by distance to the
2-D tile boundaries. python3 kiro/vae_errmap.py <ref_dir> <test_dir> [P=4] [D=16]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_cmp import full_video  # noqa: E402  (module code runs only for its own argv)

ref, test = sys.argv[1], sys.argv[2]
P = int(sys.argv[3]) if len(sys.argv) > 3 else 4
D = int(sys.argv[4]) if len(sys.argv) > 4 else 16
a = full_video(ref, 48).double()
b = full_video(test, 48).double()
W = min(a.shape[-1], b.shape[-1])
d = (a[..., :W] - b[..., :W]).abs().amax(0)          # [T, H, W]
T, H, Wd = d.shape
th, tw = H // P, Wd // D
print(f"video {tuple(a.shape)}; tile {th}x{tw}; max {d.max():.3f}")
print("max per frame:", " ".join(f"{v:.2f}" for v in d.amax((1, 2)).tolist()))
print("max per tile row:", " ".join(f"{v:.2f}" for v in d.view(T, P, th, Wd).amax((0, 2, 3)).tolist()))
print("max per tile col:", " ".join(f"{v:.2f}" for v in d.view(T, H, D, tw).amax((0, 1, 3)).tolist()))
r = torch.arange(H) % th
c = torch.arange(Wd) % tw
dr = torch.minimum(r, th - 1 - r)
dc = torch.minimum(c, tw - 1 - c)
for k in range(4):
    m_r = (dr == k)
    m_c = (dc == k)
    print(f"dist {k} from a row boundary: max {d[:, m_r, :].max():.3f} | col boundary: {d[:, :, m_c].max():.3f}")
inner = d[:, dr >= 8][:, :, dc >= 8]
print(f">=8 px from any boundary: max {inner.max():.3f} mean {inner.mean():.2e}; overall mean {d.mean():.2e}")
