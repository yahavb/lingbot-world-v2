"""LINGBOT_CTE_CHAIN on one core: the chained self-attention calls (vendored kernel, tp_out output)
against the merged per-call results (cte_partial + cte_merge) and an fp32 reference, at the DiT's
per-rank shapes (10 heads, d 128, 4320-key chunks); then kernel + merge time of both paths.
    NEURON_RT_VISIBLE_CORES=0 LINGBOT_CTE_LOCAL=1 LINGBOT_CTE_VGROUP=8 python3 kiro/attn_chain_test.py
QPAD=32 pads the queries to a multiple of 32 rows (288) instead of 128 (384)."""
import math
import os
import sys
import time
os.environ.setdefault("ATTENTION_BACKEND", "cte")
os.environ.setdefault("LINGBOT_HOT_ATTN", "cte")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from wan.modules import attention as A  # noqa: E402
REPS = int(os.environ.get("REPS", "50"))
QPAD = int(os.environ.get("QPAD", "128"))
TM = os.environ.get("TM", "0") == "1"   # chained path reads V token-major (v_token_major)
n, d, L, LQ = 10, 128, 4320, 288
dev = torch.device("neuron")
g = torch.Generator().manual_seed(11)
def rnd(*s, scale=1.0):
    return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)
def merged(q, ks, vs):
    """The current path: (lead pair) + (new chunk with the last cached chunk as prior), merged."""
    if len(ks) == 1:
        parts = [A.cte_partial(q, ks[0], vs[0])]
    elif len(ks) == 2:
        parts = [A.cte_partial(q, ks[1], vs[1], ks[0], vs[0])]
    elif len(ks) == 3:
        parts = [A.cte_partial(q, ks[0], vs[0]), A.cte_partial(q, ks[2], vs[2], ks[1], vs[1])]
    else:
        parts = [A.cte_partial(q, ks[1], vs[1], ks[0], vs[0]), A.cte_partial(q, ks[3], vs[3], ks[2], vs[2])]
    return A.cte_merge(parts, LQ)                                  # [1, LQ, n, d]
def chained(q, ks, vs):
    """LINGBOT_CTE_CHAIN: the same calls, chained; the last one normalizes ([n, d, Lq_p])."""
    state = None
    if len(ks) == 3:
        state = A.cte_call(q, ks[0], vs[0], v_token_major=TM)
    elif len(ks) == 4:
        state = A.cte_call(q, ks[1], vs[1], ks[0], vs[0], v_token_major=TM)
    if len(ks) == 1:
        o = A.cte_call(q, ks[0], vs[0], chain=state, normalize=True, v_token_major=TM)
    else:
        o = A.cte_call(q, ks[-1], vs[-1], ks[-2], vs[-2], chain=state, normalize=True, v_token_major=TM)
    return o[:, :, :LQ].permute(2, 0, 1).unsqueeze(0)               # [1, LQ, n, d]
def reference(q, ks, vs):
    K = torch.cat([k.float() for k in ks], dim=2)                      # [n, d, keys]
    V = torch.cat([v.float() for v in vs], dim=1)                      # [n, keys, d]
    sc = torch.matmul(q[:, :LQ].float(), K)
    return (torch.softmax(sc, -1) @ V).permute(1, 0, 2).unsqueeze(0)  # [1, LQ, n, d]
def timed(cf, args):
    with torch.no_grad():
        for _ in range(3):
            cf(*args)
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            cf(*args)
        torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3
def rel(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()
def main():
    lq_p = -(-LQ // QPAD) * QPAD
    q = torch.nn.functional.pad(rnd(n, LQ, d, scale=1.0 / math.sqrt(d) * 3.0), (0, 0, 0, lq_p - LQ))
    ks = [rnd(n, d, L) for _ in range(4)]
    vs = [rnd(n, L, d) for _ in range(4)]
    for lead in (0, 1, 2, 3):
        kk, vv = ks[:lead + 1], vs[:lead + 1]
        # token-major memory [L, n, d] under the [n, L, d] shape, for the chained path's v_token_major
        vv_c = [v_.permute(1, 0, 2).contiguous().view(n, L, d) for v_ in vv] if TM else vv
        args = [q.to(dev)] + [t.to(dev) for t in kk] + [t.to(dev) for t in vv]
        args_c = [q.to(dev)] + [t.to(dev) for t in kk] + [t.to(dev) for t in vv_c]
        m = lead + 1
        torch._dynamo.reset()
        fm = torch.compile(lambda q_, *t, m=m: merged(q_, list(t[:m]), list(t[m:])), backend="neuron", fullgraph=True, dynamic=False)
        fc = torch.compile(lambda q_, *t, m=m: chained(q_, list(t[:m]), list(t[m:])), backend="neuron", fullgraph=True, dynamic=False)
        with torch.no_grad():
            om = fm(*args).float().cpu()
            oc = fc(*args_c).float().cpu()
        ref = reference(q, kk, vv)
        tm, tc = timed(fm, args), timed(fc, args_c)
        print(f"[attn_chain] lead={lead} qpad={QPAD} tm={int(TM)} merged {tm:.3f} ms  chained {tc:.3f} ms  "
              f"rel(merged, fp32)={rel(om, ref):.2e} rel(chained, fp32)={rel(oc, ref):.2e} "
              f"rel(chained, merged)={rel(oc, om):.2e} max|chained-merged|={(oc - om).abs().max().item():.2e} "
              f"finite={bool(torch.isfinite(oc).all())}", flush=True)
if __name__ == "__main__":
    main()
