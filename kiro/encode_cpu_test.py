"""Time the conditioning-image VAE encode on the host CPU (fp32, exact): python3 kiro/encode_cpu_test.py"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "64")))
CKPT = "/work/ckpt/lingbot-world-v2-14b-causal-fast"
F, H, W = 81, 480, 768
vae = Wan2_1_VAE(vae_pth=os.path.join(CKPT, "Wan2.1_VAE.pth"), device="cpu")
print("vae weight dtype", next(vae.model.parameters()).dtype, flush=True)
x = torch.cat([torch.randn(3, 1, H, W).clamp(-1, 1), torch.zeros(3, F - 1, H, W)], dim=1)
with torch.no_grad():
    t0 = time.perf_counter()
    y = vae.encode([x])[0]
    print(f"cpu encode {F}x{H}x{W}: {time.perf_counter() - t0:.1f}s -> {tuple(y.shape)} {y.dtype}", flush=True)
