"""kernels/glue.py vs CPU fp32 references and vs the compiled torch ops, at the 2-chip shard shape [544, 5120], one
logical core.  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/glue_bench.py"""
import os
import time
import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401
from kernels import glue

S, C = int(os.environ.get("S", "544")), 5120
REPS = int(os.environ.get("REPS", "20"))
DEV = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
xs = torch.randn(S, C, generator=g) * 2.0
y = (torch.randn(S, C, generator=g) * 0.5).to(torch.bfloat16)
e = [torch.randn(1, C, generator=g) * 0.2 for _ in range(6)]
cs = (torch.randn(S, C, generator=g) * 0.1).to(torch.bfloat16)
csh = (torch.randn(S, C, generator=g) * 0.1).to(torch.bfloat16)


def ln(x):
    return F.layer_norm(x.float(), (C,), eps=1e-6)


def bench(fn, *a):
    for _ in range(3):
        o = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, o


def rel(a, b):
    a = a.float().cpu()
    return ((a - b).norm() / b.norm()).item()


d = lambda t: t.to(DEV)  # noqa: E731
# ln_mod
ref = (ln(xs) * (1 + e[1]) + e[0])
us, h = bench(glue.ln_mod, d(xs), d(1 + e[1]), d(e[0]))
tc = torch.compile(lambda x_, a_, b_: (F.layer_norm(x_, (C,), eps=1e-6) * a_ + b_).to(torch.bfloat16),
                   backend="neuron", fullgraph=True, dynamic=False)
us_t, _ = bench(tc, d(xs), d(1 + e[1]), d(e[0]))
print(f"[glue] ln_mod       nki {us:7.1f} us  torch {us_t:7.1f} us  rel {rel(h, ref):.2e}", flush=True)
# res_film_ln
x1 = (1 + cs.float()) * (xs + y.float() * e[2]) + csh.float()
us, (xo, ho) = bench(glue.res_film_ln, d(xs), d(y), d(e[2]), d(cs), d(csh))
print(f"[glue] res_film_ln  nki {us:7.1f} us  rel x {rel(xo, x1):.2e}  rel h {rel(ho, ln(x1)):.2e}", flush=True)
# res_ln_mod
x2 = xs + y.float()
us, (xo, ho) = bench(glue.res_ln_mod, d(xs), d(y), d(1 + e[4]), d(e[3]))
print(f"[glue] res_ln_mod   nki {us:7.1f} us  rel x {rel(xo, x2):.2e}  rel h {rel(ho, ln(x2) * (1 + e[4]) + e[3]):.2e}",
      flush=True)
# res_gate
us, xo = bench(glue.res_gate, d(xs), d(y), d(e[5]))
print(f"[glue] res_gate     nki {us:7.1f} us  rel {rel(xo, xs + y.float() * e[5]):.2e}", flush=True)
print("[glue] done", flush=True)
