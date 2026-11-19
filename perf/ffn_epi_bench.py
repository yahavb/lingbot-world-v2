"""kernels/ffn_epi.py bias_gelu_pad vs the compiled torch ops it replaces, at the 2-chip FFN shape (one logical core).
  NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/ffn_epi_bench.py
Env: M (2176), N (3456), NOUT (3584), REPS (30).
"""
import os
import time
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401
from kernels import ffn_epi

M, N, NOUT = int(os.environ.get("M", "2176")), int(os.environ.get("N", "3456")), int(os.environ.get("NOUT", "3584"))
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")


def bench(fn, *a):
    for _ in range(3):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


g = torch.Generator().manual_seed(0)
u = (torch.randn(M, N, generator=g) * 2.0).to(torch.bfloat16)
b = (torch.randn(N, generator=g) * 0.3).to(torch.bfloat16)
ref = F.pad(F.gelu(u.float() + b.float(), approximate="tanh"), (0, NOUT - N))
ud, bd = u.to(DEV), b.to(DEV)


def torch_ops(u_, b_):
    return F.pad(F.gelu(u_ + b_, approximate="tanh"), (0, NOUT - N))


def rel(y):
    y = y.float().cpu()
    return ((y - ref).norm() / ref.norm()).item(), (y[:, N:] != 0).sum().item()


tc = torch.compile(torch_ops, backend="neuron", fullgraph=True, dynamic=False)
us_t, y_t = bench(tc, ud, bd)
print(f"[epi] torch compiled  {us_t:7.1f} us  rel {rel(y_t)[0]:.2e}", flush=True)
kc = torch.compile(lambda u_, b_: ffn_epi.bias_gelu_pad(u_, b_.view(1, -1), NOUT), backend="neuron",
                   fullgraph=True, dynamic=False)
us_k, y_k = bench(kc, ud, bd)
r, nz = rel(y_k)
print(f"[epi] nki in-graph    {us_k:7.1f} us  rel {r:.2e}  nonzero pad {nz}", flush=True)
# both inside one graph with a matmul-sized neighbour would be the real test; the block A/B does that
print("[epi] done", flush=True)
