#!/usr/bin/env python3
"""LingBot-World v2 self-attention on one NeuronCore, two ways (talk slides S5b and S5h).

  --mode sdpa : torch.nn.functional.scaled_dot_product_attention, compiled with
                torch.compile(backend="neuron"). The script sets TORCH_NEURONX_ENABLE_NKI_SDPA=0
                before torch_neuronx loads, so the compiler lowers SDPA to matmul -> softmax ->
                matmul (S5b: Tensor waits while Vector/Scalar run the softmax). Without that
                variable, torch-neuronx (SDK 2.33) swaps in an NKI flash-attention kernel by itself.
  --mode nki  : nkilib attention_cte, the NKI flash-attention kernel (its Q x K and P x V are
                nisa.nc_matmul inside a K-tile loop, S5h), called the way LingBot calls it
                (wan/modules/attention.py, _cte_attention), inside the same torch.compile graph.

Shapes default to one LingBot rank at TP4 x CP8, 480x768, steady state (derived from the config,
not read from a run): heads 40 / TP4 = 10; head dim 5120 / 40 = 128; queries = one chunk
(3 latent frames x 30 x 48 patches = 4320 tokens) / CP8 = 540, padded to 640 (multiple of 128);
keys = a full window of 12 latent frames x 1440 = 17280 tokens.

Prints per-call time (median of --iters), and the error against an fp32 CPU reference.
The compiled NEFF lands in TORCH_NEURONX_NEFF_CACHE_DIR; demo_job.yaml captures its NTFF.

  NEURON_LOGICAL_NC_CONFIG=2 TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_sdpa python3 sdpa_vs_nki.py --mode sdpa
  NEURON_LOGICAL_NC_CONFIG=2 TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_nki  python3 sdpa_vs_nki.py --mode nki
"""
import argparse
import math
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["sdpa", "nki"], required=True)
ap.add_argument("--heads", type=int, default=10)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--lq", type=int, default=640)
ap.add_argument("--lk", type=int, default=17280)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

if args.mode == "sdpa":
    # must be set before torch_neuronx is imported: the compiler reads it when it lowers SDPA
    os.environ["TORCH_NEURONX_ENABLE_NKI_SDPA"] = "0"

import torch                      # noqa: E402
import torch.nn.functional as F   # noqa: E402
import torch_neuronx              # noqa: E402,F401

dev = torch.device("neuron")
n, d, lq, lk = args.heads, args.dim, args.lq, args.lk
scale = 1.0 / math.sqrt(d)

if args.mode == "sdpa":
    def attn(q, k, v):            # q, k, v: [1, n, L, d] (the SDPA layout)
        return F.scaled_dot_product_attention(q, k, v)
else:
    from torch_neuronx.nki_hop import wrap_nki
    from nkilib.core.attention.attention_cte import attention_cte as _cte_kernel
    _cte = wrap_nki(_cte_kernel)

    def attn(q, k, v):            # same layout in and out as the sdpa mode
        # heads folded into the batch dim, q pre-scaled in fp32, as LingBot's _cte_attention does
        qc = (q[0].float() * scale).to(torch.bfloat16)   # [n, Lq, d]
        # K/V padded to a multiple of 512 and the pad masked with bound_max, as LingBot does
        pad = (512 - lk % 512) % 512
        kc, vc = F.pad(k[0], (0, 0, 0, pad)), F.pad(v[0], (0, 0, 0, pad))
        lo = torch.zeros(n, lq, 1, dtype=torch.long, device=q.device)
        hi = torch.full((n, lq, 1), lk, dtype=torch.long, device=q.device)
        out = _cte(qc, kc, vc, scale=1.0, causal_mask=False,
                   tp_q=True, tp_k=True, tp_out=False, bound_min=lo, bound_max=hi)
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(0)                         # [1, n, Lq, d]

fn = torch.compile(attn, backend="neuron", fullgraph=True, dynamic=False)

g = torch.Generator().manual_seed(args.seed)
q = torch.randn(1, n, lq, d, generator=g)
k = torch.randn(1, n, lk, d, generator=g)
v = torch.randn(1, n, lk, d, generator=g)
qd, kd, vd = (t.to(torch.bfloat16).to(dev) for t in (q, k, v))

t0 = time.perf_counter()
out = fn(qd, kd, vd)
torch.neuron.synchronize()
print(f"[{args.mode}] first call (compile + load): {time.perf_counter() - t0:.1f} s")

times = []
for _ in range(args.iters):
    t = time.perf_counter()
    out = fn(qd, kd, vd)
    torch.neuron.synchronize()
    times.append((time.perf_counter() - t) * 1e3)

ref = F.scaled_dot_product_attention(*(t.to(torch.bfloat16).float() for t in (q, k, v)))
o = out.float().cpu()
rel = ((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
print(f"[{args.mode}] shapes q [1,{n},{lq},{d}] k/v [1,{n},{lk},{d}] bf16")
print(f"[{args.mode}] per call: median {statistics.median(times):.3f} ms, "
      f"min {min(times):.3f} ms over {args.iters} calls (host-timed, includes launch)")
print(f"[{args.mode}] rel-RMS vs fp32 CPU: {rel:.5f}")
sys.exit(0 if rel < 0.02 else 1)
