"""Run-to-run determinism of the constant-offset kernels (kernels/attention_cmax2_lb.py) on one logical core: the
in-block structure (one call over the cached chunks, one over the new chunk) at 2-4 chunks, in the production
layouts (rank-blocked K, token-major V), REPS calls on the same inputs, each output compared bitwise with the first.
    NEURON_RT_VISIBLE_CORES=0 LINGBOT_CTE_LOCAL=1 python3 kiro/cmax_det.py        (LINGBOT_CMAX_DGE=... to vary)
"""
import os
import sys

os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402
from kernels import attention_cmax2_lb as C2  # noqa: E402

REPS = int(os.environ.get("REPS", "20"))
n, d, L, LQ, LQP, NB, C = 10, 128, 4320, 270, 288, 16, 30.0
dev = torch.device("neuron")
g = torch.Generator().manual_seed(7)
K2 = {"part1": wrap_nki(C2.cmax2_part_1), "part2": wrap_nki(C2.cmax2_part_2), "part3": wrap_nki(C2.cmax2_part_3),
      "final1": wrap_nki(C2.cmax2_final_1)}


def blocked(k):
    n_, d_, l_ = k.shape
    return k.view(n_, d_, NB, l_ // NB).permute(2, 0, 1, 3).contiguous().view(n_, d_, l_)


def token_major(v):
    return v.permute(1, 0, 2).contiguous().view(v.shape)


def rnd(*s, scale=1.0):
    return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)


def run(q_t, ks, vs):
    acc = K2[f"part{len(ks) - 1}"](q_t, *ks[:-1], *vs[:-1], C, NB, True)
    return acc, K2["final1"](q_t, ks[-1], vs[-1], acc, C, NB, True)


def main():
    print(f"[cmax_det] LINGBOT_CMAX_DGE={os.environ.get('LINGBOT_CMAX_DGE', '')!r} REPS={REPS}", flush=True)
    q = torch.nn.functional.pad(rnd(n, LQ, d, scale=3.0 / d ** 0.5), (0, 0, 0, LQP - LQ))
    q_t = q.permute(0, 2, 1).contiguous().to(dev)
    ks = [blocked(rnd(n, d, L)).to(dev) for _ in range(4)]
    vs = [token_major(rnd(n, L, d)).to(dev) for _ in range(4)]
    for m in (2, 3, 4):
        torch._dynamo.reset()
        f = torch.compile(lambda q_, *t, m=m: run(q_, list(t[:m]), list(t[m:])), backend="neuron",
                          fullgraph=True, dynamic=False)
        args = [q_t] + ks[:m] + vs[:m]
        with torch.no_grad():
            a0, o0 = (t.float().cpu() for t in f(*args))
            bad_a = bad_o = 0
            worst_a = worst_o = 0.0
            for _ in range(REPS):
                a, o = (t.float().cpu() for t in f(*args))
                da, do = (a - a0).abs(), (o - o0).abs()
                bad_a += int((da != 0).sum())
                bad_o += int((do != 0).sum())
                worst_a, worst_o = max(worst_a, float(da.max())), max(worst_o, float(do.max()))
        print(f"[cmax_det] chunks={m}: accumulator differs in {bad_a} elements over {REPS} calls (max |d| "
              f"{worst_a:.3e}), output in {bad_o} (max |d| {worst_o:.3e}); output finite={bool(torch.isfinite(o0).all())}",
              flush=True)


if __name__ == "__main__":
    main()
