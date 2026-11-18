"""kernels/cross_dedup.py vs wan.modules.attention.cross_attention_dedup's fp32 graph path, one logical core, the 2-chip
TP-free cross-attention shape (40 heads, 544 queries, the first 128 of 512 cached keys, a dedup bias on the last rows).
  NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/cross_kernel_test.py
"""
import math
import os
import time

import torch
import torch_neuronx  # noqa: F401

from kernels.cross_dedup import cross_dedup_kernel

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
N, D, LQ, L, KC = int(os.environ.get("N", "40")), 128, int(os.environ.get("LQ", "544")), 512, 128
dev = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
q = (torch.randn(1, LQ, N, D, generator=g) * 2.0).to(torch.bfloat16)
k = (torch.randn(N, D, L, generator=g)).to(torch.bfloat16)
v = (torch.randn(N, L, D, generator=g)).to(torch.bfloat16)
bias = torch.zeros(KC)
bias[100:] = math.log(412.0 / 28.0)          # 28 padding rows inside the slice standing for 412


def ref(q, k, v, b):
    qh = (q[0].float() * (1.0 / math.sqrt(D))).permute(1, 0, 2)
    p_ = torch.softmax(torch.matmul(qh, k[:, :, :KC].float()) + b, dim=-1)
    return torch.matmul(p_, v[:, :KC].float()).permute(1, 0, 2).unsqueeze(0)


r = ref(q, k, v, bias)
kern = cross_dedup_kernel[LNC] if LNC > 1 else cross_dedup_kernel
args = (q[0].reshape(LQ, N * D).contiguous().to(dev), k.to(dev), v.to(dev), bias.to(dev), N)
o = kern(*args)
torch.neuron.synchronize()
o = o.cpu().float().view(1, LQ, N, D)
rel = ((o - r).norm() / r.norm()).item()
mx = (o - r).abs().max().item()
for _ in range(3):
    kern(*args)
torch.neuron.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    kern(*args)
torch.neuron.synchronize()
us = (time.perf_counter() - t0) / 20 * 1e6
print(f"[xk] cross_dedup N={N} LQ={LQ}: rel vs fp32 {rel:.3e} max|d| {mx:.3e}, {us:.1f} us/call (incl. launch)", flush=True)
