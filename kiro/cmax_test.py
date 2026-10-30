"""Constant-offset attention (kernels/attention_cmax_lb.py) against the production chained attention_cte calls
(LINGBOT_CTE_CHAIN, vendored kernel) and an fp32 reference, on one logical core at the DiT's per-rank shapes
(10 heads, d 128, 4320-key chunks, 288 padded queries): accuracy and ms per call for 1, 2 and 4 chunks.
    NEURON_RT_VISIBLE_CORES=0 LINGBOT_CTE_LOCAL=1 LINGBOT_CTE_VGROUP=8 python3 kiro/cmax_test.py
CMAX=15,40 sets the constants tried (the result must not depend on it beyond rounding)."""
import math
import os
import sys
import time

os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402

from kernels import attention_cmax_lb as CM  # noqa: E402
from wan.modules import attention as A  # noqa: E402

REPS = int(os.environ.get("REPS", "50"))
CMAX = [float(c) for c in os.environ.get("CMAX", "15,40").split(",")]
n, d, L, LQ, LQP = 10, 128, 4320, 270, 288
dev = torch.device("neuron")
g = torch.Generator().manual_seed(11)
KERN = {1: wrap_nki(CM.attention_cmax_1), 2: wrap_nki(CM.attention_cmax_2), 4: wrap_nki(CM.attention_cmax_4)}
from kernels import attention_cmax2_lb as C2  # noqa: E402
K2 = {"full1": wrap_nki(C2.cmax2_full_1), "part1": wrap_nki(C2.cmax2_part_1), "part3": wrap_nki(C2.cmax2_part_3),
      "final1": wrap_nki(C2.cmax2_final_1)}
NB = 16  # rank blocks of the CP all-gather (k_blocks)


def blocked(k):
    """[n, d, L] K stored as rank blocks [NB, n, d, L / NB] under the same shape."""
    n_, d_, l_ = k.shape
    return k.view(n_, d_, NB, l_ // NB).permute(2, 0, 1, 3).contiguous().view(n_, d_, l_)


def token_major(v):
    """[n, L, d] V stored token-major [L, n, d] under the same shape."""
    return v.permute(1, 0, 2).contiguous().view(v.shape)


def cmax2(q, ks, vs, c, kb, tm):
    """The in-block structure: one call over the cached chunks (fp32 accumulator), one over the new chunk."""
    q_t = q.permute(0, 2, 1).contiguous()
    if len(ks) == 1:
        o = K2["full1"](q_t, ks[0], vs[0], c, kb, tm)
    else:
        acc = K2[f"part{len(ks) - 1}"](q_t, *ks[:-1], *vs[:-1], c, kb, tm)
        o = K2["final1"](q_t, ks[-1], vs[-1], acc, c, kb, tm)
    return o[:, :LQ].permute(1, 0, 2)                                    # [LQ, n, d]


def rnd(*s, scale=1.0):
    return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)


def chained(q, ks, vs):
    """The production path: pairs of chunks per call, chained, the last call normalizes ([n, d, Lq_p])."""
    state = None
    if len(ks) == 4:
        state = A.cte_call(q, ks[1], vs[1], ks[0], vs[0])
    if len(ks) == 1:
        o = A.cte_call(q, ks[0], vs[0], chain=state, normalize=True)
    else:
        o = A.cte_call(q, ks[-1], vs[-1], ks[-2], vs[-2], chain=state, normalize=True)
    return o[:, :, :LQ].permute(2, 0, 1)                                 # [LQ, n, d]


def cmax(q, ks, vs, c):
    """One constant-offset call over all chunks; q [n, Lq_p, d] -> [n, d, Lq_p] for the kernel."""
    o = KERN[len(ks)](q.permute(0, 2, 1).contiguous(), *ks, *vs, c)
    return o[:, :LQ].permute(1, 0, 2)                                    # [LQ, n, d]


def reference(q, ks, vs):
    K = torch.cat([k.float() for k in ks], dim=2)
    V = torch.cat([v.float() for v in vs], dim=1)
    sc = torch.matmul(q[:, :LQ].float(), K)
    return (torch.softmax(sc, -1) @ V).permute(1, 0, 2)                  # [LQ, n, d]


def timed(f, args):
    with torch.no_grad():
        for _ in range(3):
            f(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f(*args)
        torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3


def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def main():
    q = torch.nn.functional.pad(rnd(n, LQ, d, scale=3.0 / math.sqrt(d)), (0, 0, 0, LQP - LQ))
    ks = [rnd(n, d, L) for _ in range(4)]
    vs = [rnd(n, L, d) for _ in range(4)]
    for m in (1, 2, 4):
        kk, vv = ks[:m], vs[:m]
        args = [q.to(dev)] + [t.to(dev) for t in kk] + [t.to(dev) for t in vv]
        args_l = [q.to(dev)] + [blocked(t).to(dev) for t in kk] + [token_major(t).to(dev) for t in vv]
        ref = reference(q, kk, vv)
        torch._dynamo.reset()
        fc = torch.compile(lambda q_, *t, m=m: chained(q_, list(t[:m]), list(t[m:])), backend="neuron",
                           fullgraph=True, dynamic=False)
        with torch.no_grad():
            oc = fc(*args).float().cpu()
        tc = timed(fc, args)
        line = f"[cmax] chunks={m} chained {tc:.3f} ms rel(chained, fp32)={rel(oc, ref):.3e}"
        for c in CMAX:
            torch._dynamo.reset()
            fm = torch.compile(lambda q_, *t, m=m, c=c: cmax(q_, list(t[:m]), list(t[m:]), c), backend="neuron",
                               fullgraph=True, dynamic=False)
            with torch.no_grad():
                om = fm(*args).float().cpu()
            tm = timed(fm, args)
            line += (f" | c={c:g} {tm:.3f} ms rel(fp32)={rel(om, ref):.3e} rel(chained)={rel(om, oc):.3e} "
                     f"finite={bool(torch.isfinite(om).all())}")
        if m != 2:
            c = CMAX[0]
            for kb, tm, a_ in ((1, False, args), (NB, True, args_l)):
                torch._dynamo.reset()
                f2 = torch.compile(lambda q_, *t, m=m, c=c, kb=kb, tm=tm: cmax2(q_, list(t[:m]), list(t[m:]), c, kb, tm),
                                   backend="neuron", fullgraph=True, dynamic=False)
                with torch.no_grad():
                    o2 = f2(*a_).float().cpu()
                t2 = timed(f2, a_)
                line += (f" | cmax2 kb={kb} tm={int(tm)} {t2:.3f} ms rel(fp32)={rel(o2, ref):.3e} "
                         f"rel(chained)={rel(o2, oc):.3e} finite={bool(torch.isfinite(o2).all())}")
        print(line, flush=True)


if __name__ == "__main__":
    main()
