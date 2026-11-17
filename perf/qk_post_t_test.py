"""kernels/qk_post.py qk_norm_rope_t (LINGBOT_QK_POST_T) vs a CPU fp32 reference and vs qk_norm_rope, at the 2-chip
shape on one logical core (no TP all-reduce: the local sum of squares stands in for the reduced one).
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 NKI_ENABLE_TRACE_CACHE=0 python3 perf/qk_post_t_test.py
"""
import math
import os
import time
import torch
import torch_neuronx  # noqa: F401
from kernels import qk_post
L, F, C = int(os.environ.get("L", "2176")), 1280, 64
SEQ = int(os.environ.get("SEQ", "2160"))
DIM, EPS = 5120, 1e-6
H, D = F // (2 * C), 2 * C
QS = 1.0 / math.sqrt(D)
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
    t4 = t.view(L, H, C, 2)
    e, o = t4[..., 0], t4[..., 1]
    cb, sb = cos[:, None, :], sin[:, None, :]
    return torch.stack([e * cb - o * sb, e * sb + o * cb], -1).reshape(L, F)


ref_q = (ref_one(q, bq, wq) * QS).view(L, H, D).permute(1, 2, 0)        # [H, D, L]
ref_k = ref_one(k, bk, wk).view(L, H, D).permute(1, 2, 0)
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


def chain_t(q_, k_, bq_, bk_, wq_, wk_, cos_, sin_):
    ss = qk_post.qk_ss(q_, k_, bq_, bk_)
    return qk_post.qk_norm_rope_t(q_, k_, bq_, bk_, ss, wq_, wk_, cos_, sin_, 1.0 / DIM, EPS, QS)


def chain_old(q_, k_, bq_, bk_, wq_, wk_, cos_, sin_):
    ss = qk_post.qk_ss(q_, k_, bq_, bk_)
    oq, ok = qk_post.qk_norm_rope(q_, k_, bq_, bk_, ss, wq_, wk_, cos_, sin_, 1.0 / DIM, EPS)
    qt = (oq.float() * QS).view(L, H, D).permute(1, 2, 0).contiguous().to(torch.bfloat16)
    kt = ok.view(L, H, D).permute(1, 2, 0).contiguous()
    return qt, kt


def rel(a, b):
    a = a.float().cpu()
    return ((a - b).norm() / b.norm()).item()


qkv_cpu = torch.cat([q, k, k], dim=1)                                  # the fused projection's [L, 3F] layout
dv3 = [qkv_cpu.to(DEV)] + dv[2:]


def chain_t3(qkv_, bq_, bk_, wq_, wk_, cos_, sin_):
    ss = qk_post.qk_ss3(qkv_, bq_, bk_)
    return qk_post.qk_norm_rope_t3(qkv_, bq_, bk_, ss, wq_, wk_, cos_, sin_, 1.0 / DIM, EPS, QS)


def chain_split3(qkv_, bq_, bk_, wq_, wk_, cos_, sin_):
    qt = qk_post.x_norm_rope_t3(qkv_, 0, bq_, qk_post.x_ss3(qkv_, 0, bq_), wq_, cos_, sin_, 1.0 / DIM, EPS, QS)
    kt = qk_post.x_norm_rope_t3(qkv_, F, bk_, qk_post.x_ss3(qkv_, F, bk_), wk_, cos_, sin_, 1.0 / DIM, EPS, 1.0)
    return qt, kt


def chain_n3(qkv_, bq_, bk_, wq_, wk_, cos_, sin_):
    # the exact layers' fused path: token-major outputs, compared here after the same scale + transposes
    ss = qk_post.qk_ss3(qkv_, bq_, bk_)
    oq, ok = qk_post.qk_norm_rope3(qkv_, bq_, bk_, ss, wq_, wk_, cos_, sin_, 1.0 / DIM, EPS)
    qt = (oq.float() * QS).view(L, H, D).permute(1, 2, 0).contiguous().to(torch.bfloat16)
    return qt, ok.view(L, H, D).permute(1, 2, 0).contiguous()


for name, fn, args in (("qk_norm_rope3 (fused)", chain_n3, dv3), ("qk_norm_rope_t3 (fused)", chain_t3, dv3),
                       ("x_norm_rope_t3 split", chain_split3, dv3)):
    c = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    us, (qt, kt) = bench(c, *args)
    print(f"[qkpt] {name:22s} {us:7.1f} us  q rel {rel(qt, ref_q):.2e}  k rel {rel(kt, ref_k):.2e}", flush=True)
for name, fn in (("old+torch transposes", chain_old), ("qk_norm_rope_t", chain_t)):
    c = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    us, (qt, kt) = bench(c, *dv)
    print(f"[qkpt] {name:22s} {us:7.1f} us  q rel {rel(qt, ref_q):.2e}  k rel {rel(kt, ref_k):.2e}", flush=True)
print("[qkpt] done", flush=True)
