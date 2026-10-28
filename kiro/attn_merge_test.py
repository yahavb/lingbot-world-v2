"""Exactness + speed of the merged per-chunk attention_cte path vs the padded-assembly path.

Single rank:  NEURON_RT_VISIBLE_CORES=<c> python3 kiro/attn_merge_test.py
Env: NCH (KV chunks, default 4), LQ (queries, default 2176), L (tokens per chunk, 4320), REPS.
Reference is fp32 softmax attention on CPU over the bf16-rounded inputs. Heads get different
query gains so both flat (small row max) and peaked heads are covered: the zero-key padding of
the assembly path biases flat heads most.
"""
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

NCH = int(os.environ.get("NCH", "4"))
LQ = int(os.environ.get("LQ", "2176"))
L = int(os.environ.get("L", "4320"))
REPS = int(os.environ.get("REPS", "10"))
n, d = 10, 128
GAINS = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1.0, 0.5, 0.25])


def main():
    dev = torch.device("neuron")
    g = torch.Generator().manual_seed(0)
    q = (torch.randn(1, LQ, n, d, generator=g) * GAINS.view(1, 1, n, 1)).to(torch.bfloat16)
    ks = [torch.randn(1, L, n, d, generator=g).to(torch.bfloat16) for _ in range(NCH)]
    vs = [torch.randn(1, L, n, d, generator=g).to(torch.bfloat16) for _ in range(NCH)]

    # fp32 reference, per head
    K = torch.cat(ks, 1)[0].float().permute(1, 0, 2)          # [n, Lkv, d]
    V = torch.cat(vs, 1)[0].float().permute(1, 0, 2)
    Q = q[0].float().permute(1, 0, 2) / math.sqrt(d)            # [n, LQ, d]
    ref = torch.empty(n, LQ, d)
    rowmax = torch.empty(n)
    for h in range(n):
        s = Q[h] @ K[h].T
        rowmax[h] = s.max(-1).values.mean()
        ref[h] = torch.softmax(s, -1) @ V[h]
    ref = ref.permute(1, 0, 2).unsqueeze(0)                     # [1, LQ, n, d]

    qd = q.to(dev)
    k4 = [t.to(dev) for t in ks]
    v4 = [t.to(dev) for t in vs]
    k_ndl = [t[0].permute(1, 2, 0).contiguous().to(dev) for t in ks]   # [n, d, L]
    k_nld = [t[0].permute(1, 0, 2).contiguous().to(dev) for t in ks]   # [n, L, d]
    v_nld = [t[0].permute(1, 0, 2).contiguous().to(dev) for t in vs]

    variants = {
        "assembly(padded)": (A._cte_attention_chunks, k4, v4),
        "merged K[n,d,L]": (A._cte_attention_chunks_merged, k_ndl, v_nld),
        "merged K[n,L,d]": (A._cte_attention_chunks_merged, k_nld, v_nld),
    }
    print(f"NCH={NCH} LQ={LQ} L={L} Lkv={NCH * L}; mean row max per head: "
          + " ".join(f"{x:.1f}" for x in rowmax.tolist()))
    for name, (fn, kk, vv) in variants.items():
        cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        with torch.no_grad():
            t0 = time.perf_counter()
            out = cf(qd, kk, vv)
            torch.neuron.synchronize()
            t_first = time.perf_counter() - t0
            for _ in range(2):
                out = cf(qd, kk, vv)
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS):
                out = cf(qd, kk, vv)
            torch.neuron.synchronize()
            ms = (time.perf_counter() - t0) / REPS * 1e3
        o = out.float().cpu()
        err = (o - ref)
        per_head = (err.pow(2).mean((0, 1, 3)).sqrt() / ref.pow(2).mean((0, 1, 3)).sqrt())
        scale = (o * ref).sum((0, 1, 3)) / ref.pow(2).sum((0, 1, 3))
        print(f"{name:18s} {ms:7.3f} ms (first {t_first:.1f}s)  rel_rms {err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt():.5f}"
              f"  per-head rel_rms " + " ".join(f"{x:.4f}" for x in per_head.tolist())
              + "  per-head gain " + " ".join(f"{x:.4f}" for x in scale.tolist()))


if __name__ == "__main__":
    main()
