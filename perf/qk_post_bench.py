"""kernels/qk_post.py vs a CPU fp32 reference and vs the compiled torch chain, at the 2-chip shape on one logical
core (no TP all-reduce: the local sum of squares stands in for the reduced one).
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/qk_post_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
from kernels import qk_post

L, F, C = int(os.environ.get("L", "2176")), 1280, 64
SEQ = int(os.environ.get("SEQ", "2160"))
DIM, EPS = 5120, 1e-6
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")
g = torch.Generator().manual_seed(0)
q = (torch.randn(L, F, generator=g) * 1.5).to(torch.bfloat16)
k = (torch.randn(L, F, generator=g) * 1.5).to(torch.bfloat16)
bq = (torch.randn(1, F, generator=g) * 0.2).to(torch.bfloat16)
bk = (torch.randn(1, F, generator=g) * 0.2).to(torch.bfloat16)
wq = (1 + torch.randn(1, F, generator=g) * 0.1).to(torch.bfloat16)
wk = (1 + torch.randn(1, F, generator=g) * 0.1).to(torch.bfloat16)
ang = torch.rand(SEQ, C, generator=g) * 6.28
cos = torch.cat([ang.cos(), torch.ones(L - SEQ, C)]).float()
sin = torch.cat([ang.sin(), torch.zeros(L - SEQ, C)]).float()


def ref_one(x, b, w):
    y = x.float() + b.float()
    ss = (y * y).sum(-1)
    r = torch.rsqrt(ss / DIM + EPS)
    t = (y * r[:, None]).to(torch.bfloat16).float() * w.float()
    t4 = t.view(L, F // (2 * C), C, 2)
    e, o = t4[..., 0], t4[..., 1]
    cb, sb = cos[:, None, :], sin[:, None, :]
    out = torch.stack([e * cb - o * sb, e * sb + o * cb], -1).reshape(L, F)
    return ss, out


ss_q, ref_q = ref_one(q, bq, wq)
ss_k, ref_k = ref_one(k, bk, wk)
dv = [t.to(DEV) for t in (q, k, bq, bk, wq, wk, cos, sin)]


def bench(fn, *a):
    for _ in range(3):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


def nki_chain(q_, k_, bq_, bk_, wq_, wk_, cos_, sin_):
    ss = qk_post.qk_ss(q_, k_, bq_, bk_)
    oq, ok = qk_post.qk_norm_rope(q_, k_, bq_, bk_, ss, wq_, wk_, cos_, sin_, 1.0 / DIM, EPS)
    return ss, oq, ok


def torch_chain(q_, k_, bq_, bk_, wq_, wk_, cos_, sin_):
    outs = []
    for x, b, w in ((q_, bq_, wq_), (k_, bk_, wk_)):
        y = (x + b).float()
        ss = y.pow(2).sum(-1, keepdim=True)
        t = (y * torch.rsqrt(ss / DIM + EPS)).to(torch.bfloat16) * w
        t4 = t.float().reshape(L, F // (2 * C), C, 2)
        e, o = t4[..., 0:1].reshape(L, -1, C), t4[..., 1:2].reshape(L, -1, C)
        cb, sb = cos_[:, None, :], sin_[:, None, :]
        out = torch.cat([(e * cb - o * sb).unsqueeze(-1), (e * sb + o * cb).unsqueeze(-1)], -1).reshape(L, F)
        outs.append(out.to(torch.bfloat16))
    return outs


tc = torch.compile(torch_chain, backend="neuron", fullgraph=True, dynamic=False)
us_t, _ = bench(tc, *dv)
print(f"[qkp] torch compiled chain  {us_t:7.1f} us", flush=True)
kc = torch.compile(nki_chain, backend="neuron", fullgraph=True, dynamic=False)
us_k, (ss, oq, ok) = bench(kc, *dv)


def rel(a, b):
    a, b = a.float().cpu(), b.float()
    return ((a - b).norm() / b.norm()).item()


print(f"[qkp] nki qk_ss + qk_norm_rope {us_k:7.1f} us  ss rel q {rel(ss[:, 0], ss_q):.2e} k {rel(ss[:, 1], ss_k):.2e}  "
      f"out rel q {rel(oq, ref_q):.2e} k {rel(ok, ref_k):.2e}", flush=True)
print("[qkp] done", flush=True)
