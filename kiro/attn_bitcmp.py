"""Bitwise comparison of the vendored attention_cte (kernels/attention_cte_lb.py, with whatever
LINGBOT_CTE_* knobs are set) against the installed nkilib attention_cte, on identical inputs, for
the call shapes the DiT uses (self-attention chunks with and without a prior, cross-attention).
    NEURON_RT_VISIBLE_CORES=0 NKI_ENABLE_TRACE_CACHE=0 LINGBOT_CTE_VGROUP=8 python3 kiro/attn_bitcmp.py
Prints, per case and output (out, row max, row sum): equal, number of differing elements, max |d|."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: E402,F401
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402
from nkilib.core.attention.attention_cte import attention_cte as k_ref  # noqa: E402
from kernels.attention_cte_lb import attention_cte as k_loc  # noqa: E402

A_REF, A_LOC = wrap_nki(k_ref), wrap_nki(k_loc)
dev = torch.device("neuron")
n, d = 10, 128
g = torch.Generator().manual_seed(11)


def r(*s):
    return torch.randn(*s, generator=g).to(torch.bfloat16)


def call(kern, q, k, v, kp=None, vp=None):
    kw = {}
    if kp is not None:
        kw = dict(k_prior=kp, v_prior=vp,
                  prior_used_len=torch.full((1,), int(vp.shape[1]), dtype=torch.int32, device=q.device))
    return kern(q, k, v, scale=1.0, causal_mask=False, tp_q=True, tp_k=False, tp_out=False,
                cache_softmax=True, skip_output_normalization=True, **kw)


CASES = [  # (name, lq, prior keys, active keys)
    ("self lead0 (4320)", 384, 0, 4320),
    ("self pair (prior 4320 + 4320)", 384, 4320, 4320),
    ("self lead2 tail (prior 4320 + 4320)", 384, 4320, 4320),
    ("cross (512)", 384, 0, 512),
    ("split rem (1080)", 256, 0, 1080),
    ("prior 12960 + 4320", 384, 12960, 4320),
]

KB = int(os.environ.get("K_BLOCKS", "1"))
if KB > 1:
    # vendored kernel with K in rank-blocked memory (k_blocks) vs nkilib on the plain layout
    def blocked(k):
        n_, d_, L_ = k.shape
        return k.view(n_, d_, KB, L_ // KB).permute(2, 0, 1, 3).contiguous().view(n_, d_, L_)

    def call_loc(q, k, v, kp=None, vp=None):
        kw = {}
        if kp is not None:
            kw = dict(k_prior=kp, v_prior=vp,
                      prior_used_len=torch.full((1,), int(vp.shape[1]), dtype=torch.int32, device=q.device))
        return A_LOC(q, k, v, scale=1.0, causal_mask=False, tp_q=True, tp_k=False, tp_out=False,
                     cache_softmax=True, skip_output_normalization=True, k_blocks=KB, **kw)
    for name, lq, lp, la in [c for c in CASES if c[3] % KB == 0 and c[2] % KB == 0]:
        q = (r(n, lq, d) * 0.088).to(torch.bfloat16)
        k, v = r(n, d, la), r(n, la, d)
        kp, vp = (r(n, d, lp), r(n, lp, d)) if lp else (None, None)
        a_ref = [t.to(dev) if t is not None else None for t in (q, k, v, kp, vp)]
        a_loc = [t.to(dev) if t is not None else None
                 for t in (q, blocked(k), v, blocked(kp) if kp is not None else None, vp)]
        f_ref = torch.compile(lambda *a: call(A_REF, *a), backend="neuron", fullgraph=True, dynamic=False)
        f_loc = torch.compile(call_loc, backend="neuron", fullgraph=True, dynamic=False)
        with torch.no_grad():
            o_ref = [t.float().cpu() for t in f_ref(*a_ref)]
            o_loc = [t.float().cpu() for t in f_loc(*a_loc)]
        torch.neuron.synchronize()
        msg = []
        for lbl, a, b in zip(("out", "max", "sum"), o_ref, o_loc):
            dd = (a - b).abs()
            msg.append(f"{lbl}: equal={torch.equal(a, b)} ndiff={int((dd > 0).sum())}/{dd.numel()} "
                       f"max|d|={dd.max().item():.3e}")
        print(f"[bitcmp] k_blocks={KB} {name:30s} " + " | ".join(msg), flush=True)
    sys.exit(0)

for name, lq, lp, la in CASES:
    q = (r(n, lq, d) * 0.088).to(torch.bfloat16)
    k, v = r(n, d, la), r(n, la, d)
    kp, vp = (r(n, d, lp), r(n, lp, d)) if lp else (None, None)
    args = [t.to(dev) if t is not None else None for t in (q, k, v, kp, vp)]
    outs = []
    for kern in (A_REF, A_LOC):
        f = torch.compile(lambda *a, kern=kern: call(kern, *a), backend="neuron", fullgraph=True,
                          dynamic=False)
        with torch.no_grad():
            o = f(*args)
        torch.neuron.synchronize()
        outs.append([t.float().cpu() for t in o])
    msg = []
    for lbl, a, b in zip(("out", "max", "sum"), *outs):
        dd = (a - b).abs()
        msg.append(f"{lbl}: equal={torch.equal(a, b)} ndiff={int((dd > 0).sum())}/{dd.numel()} "
                   f"max|d|={dd.max().item():.3e}")
    print(f"[bitcmp] {name:36s} " + " | ".join(msg), flush=True)
