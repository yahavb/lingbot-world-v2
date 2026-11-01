"""Is the CP8 cmax2 corruption a race between the kernel and the graph ops that produce its queries? One core (LNC2):
a graph that projects Q (matmul), applies a RoPE-like elementwise step, prepares it as the block does (cmax_prep_q)
and runs the cached-chunk call + the final call (part3 + final1), checked against fp32 on EVERY call. The plain
kernel test (cmax_test.py) feeds the kernel a query permuted inside the graph and is exact; the block (32 ranks) is
corrupted on random ranks.
    NEURON_RT_VISIBLE_CORES=0 LINGBOT_CMAX=2 LQ=540 LQP=544 NB=8 CALLS=60 python3 kiro/cmax_race_test.py
MODE=chain also runs the exact chained attention_cte path on the same queries as the control."""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from wan.modules import attention as A  # noqa: E402

n, d, L, DIM = 10, 128, 4320, 5120
LQ, LQP = int(os.environ.get("LQ", "540")), int(os.environ.get("LQP", "544"))
NB = int(os.environ.get("NB", "8"))
CALLS = int(os.environ.get("CALLS", "60"))
LEAD = int(os.environ.get("LEAD", "3"))
dev = torch.device("neuron")
g = torch.Generator().manual_seed(7)


def rnd(*s, scale=1.0):
    return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)


def blocked(k):
    n_, d_, l_ = k.shape
    return k.view(n_, d_, NB, l_ // NB).permute(2, 0, 1, 3).contiguous().view(n_, d_, l_)


def token_major(v):
    return v.permute(1, 0, 2).contiguous().view(v.shape)


def make_q(x, w):
    """[LQ, DIM] @ [DIM, n * d] -> [1, LQ, n, d] with a RoPE-like elementwise step (as the block feeds the kernel)."""
    q = torch.matmul(x, w).view(1, LQ, n, d).float()
    q = q * 0.9 + torch.roll(q, 1, dims=-1) * 0.1
    return q.to(torch.bfloat16)


def cmax_graph(x, w, *kv):
    ks, vs = list(kv[:LEAD + 1]), list(kv[LEAD + 1:])
    q_t, lq = A.cmax_prep_q(make_q(x, w), pad_to=32)
    acc = A.cmax_partial(q_t, ks[:-1], vs[:-1], 30.0, NB, True) if LEAD else None
    o = A.cmax_final(q_t, ks[-1], vs[-1], acc, 30.0, NB, True)
    return o[:, :lq]                                                        # [n, LQ, d]


def reference(q, ks, vs):
    """fp32 softmax attention of q [1, LQ, n, d] over the plain-layout chunks."""
    K = torch.cat([k.float() for k in ks], dim=2)                           # [n, d, S]
    V = torch.cat([v.float() for v in vs], dim=1)                           # [n, S, d]
    sc = torch.matmul(q[0].float().permute(1, 0, 2) / math.sqrt(d), K)     # [n, LQ, S]
    return torch.softmax(sc, -1) @ V                                        # [n, LQ, d]


def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


def main():
    x = rnd(LQ, DIM, scale=1.0)
    w = rnd(DIM, n * d, scale=3.0 / math.sqrt(DIM * d))
    ks = [rnd(n, d, L) for _ in range(LEAD + 1)]
    vs = [rnd(n, L, d) for _ in range(LEAD + 1)]
    xd, wd = x.to(dev), w.to(dev)
    kd = [blocked(t).to(dev) for t in ks]
    vd = [token_major(t).to(dev) for t in vs]
    with torch.no_grad():
        fq = torch.compile(make_q, backend="neuron", fullgraph=True, dynamic=False)
        q_dev = fq(xd, wd).cpu()                                            # the exact queries the graph computes
    ref = reference(q_dev, ks, vs)
    f = torch.compile(cmax_graph, backend="neuron", fullgraph=True, dynamic=False)
    # SCRIBBLE=1: between calls, another compiled graph with large intermediates of huge values, so a read of an
    # intermediate before its producer writes it sees garbage instead of the previous call's identical values.
    scribble = None
    if os.environ.get("SCRIBBLE", "0") == "1":
        def _scr(a, b):
            h = torch.matmul(a, b) * 1e30
            return (h * h).sum(dim=-1, keepdim=True)
        scribble = torch.compile(_scr, backend="neuron", fullgraph=True, dynamic=False)
        sa, sb = rnd(4096, 4096).to(dev), rnd(4096, 4096).to(dev)
    bad, worst, rels = 0, 0.0, []
    with torch.no_grad():
        for i in range(CALLS):
            if scribble is not None:
                scribble(sa, sb)
            o = f(xd, wd, *kd, *vd).float().cpu()
            r_ = rel(o, ref)
            rels.append(r_)
            worst = max(worst, r_ if math.isfinite(r_) else float("inf"))
            bad += int(not (r_ < 1e-2))
    print(f"[cmax_race] LQ={LQ}/{LQP} lead={LEAD} k_blocks={NB} scribble={int(scribble is not None)} "
          f"calls={CALLS} bad={bad} worst_rel={worst:.3e} "
          f"median_rel={sorted(rels)[len(rels) // 2]:.3e}", flush=True)


if __name__ == "__main__":
    main()
