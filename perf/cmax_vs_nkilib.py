"""A/B of constant-offset attention at the 2-chip DiT self-attention shape: kernels/attention_cmax3.py (LINGBOT_CMAX=3,
cmax3_full_1: 512-query P@V on a DMA-transposed P, row sums by ones-matmul with LINGBOT_C3_SUMMODE=pef) against
nkilib's experimental attention_const_max (no MM1 -> MM2 transpose, V augmented with a ones column), one logical core.
    NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES=0 LINGBOT_CMAX=3 LINGBOT_C3_SUMMODE=pef LINGBOT_C3_SUMGROUP=16 \
        LINGBOT_C3_DEPTH=3 SHAPE=10,128,2176,4320 python3 perf/cmax_vs_nkilib.py
SHAPE: heads N, head dim d, queries Sq, keys Sk. Prints CMAXAB lines: us/call and rel_l2 vs fp32 CPU attention of
head 0 (softmax(Q K^T / sqrt(d)) V).
"""
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from torch_neuronx.nki_hop import wrap_nki  # noqa: E402


def timeit(fn, reps):
    out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.neuron.synchronize()
    return out, (time.perf_counter() - t0) / reps * 1e6


def main():
    N, d, Sq, Sk = (int(v) for v in os.environ.get("SHAPE", "10,128,2176,4320").split(","))
    reps = int(os.environ.get("REPS", "20"))
    c = float(os.environ.get("OFFSET", "5"))
    dev = torch.device("neuron:0")
    torch.manual_seed(0)
    q = torch.randn(N, Sq, d).bfloat16()
    k = torch.randn(N, Sk, d).bfloat16()
    v = torch.randn(N, Sk, d).bfloat16()
    s = 1.0 / math.sqrt(d)
    p = torch.softmax(q[0].float() @ k[0].float().T * s, -1)
    ref0 = p @ v[0].float()                                                     # [Sq, d]

    def rel(o0):
        o0 = o0.float().cpu()
        return ((o0 - ref0).norm() / ref0.norm()).item()

    q_hm = (q.float() * s).bfloat16().permute(0, 2, 1).contiguous().to(dev)  # [N, d, Sq] pre-scaled
    q_hm1 = q.permute(0, 2, 1).contiguous().to(dev)                            # [N, d, Sq] unscaled
    k_hm = k.permute(0, 2, 1).contiguous().to(dev)                             # [N, d, Sk]
    vd = v.to(dev)                                                             # [N, Sk, d]
    rows = []
    try:
        from kernels import attention_cmax3 as cm3
        f3 = wrap_nki(cm3.cmax3_full_1)
        g3 = torch.compile(lambda a, b, cc: f3(a, b, cc, c, 1, 0), backend="neuron", fullgraph=True, dynamic=False)
        o, us = timeit(lambda: g3(q_hm, k_hm, vd), reps)                       # [N, d, Sq]
        rows.append(("cmax3 (this repo)", us, rel(o[0].T)))
    except Exception as e:
        rows.append(("cmax3 (this repo)", float("nan"), f"ERROR {type(e).__name__}: {str(e)[:300]}"))
    try:
        from nkilib.experimental.attention.attention_const_max import attention_const_max
        fn = wrap_nki(attention_const_max)
        gn = torch.compile(lambda a, b, cc: fn(a, b, cc, c, s), backend="neuron", fullgraph=True, dynamic=False)
        o, us = timeit(lambda: gn(q_hm1, k_hm, vd), reps)                      # [N, Sq, d]
        rows.append(("nkilib attention_const_max", us, rel(o[0])))
    except Exception as e:
        rows.append(("nkilib attention_const_max", float("nan"), f"ERROR {type(e).__name__}: {str(e)[:300]}"))
    for name, us, r in rows:
        print(f"CMAXAB N={N} d={d} Sq={Sq} Sk={Sk} c={c} {name:28s} us/call={us:9.1f} rel_l2={r}", flush=True)


if __name__ == "__main__":
    main()
