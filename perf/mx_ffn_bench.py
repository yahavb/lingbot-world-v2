"""MXFP8 vs BF16 for the DiT FFN at the live per-rank shapes: time and error vs fp32.

One LNC2 rank of the TP4 x CP8 layout (lbw-v2-dit-t5-deploy.yaml). Per block per forward:
  up    x [M, 5120]  @ W1[3456, 5120]^T   (column-parallel, + b1)
  GELU(tanh)
  down  h [M, 3456]  @ W2[5120, 3456]^T   (row-parallel partial, b2 on TP rank 0)
M = 4320 tokens / CP8 = 540, padded to 544 (LINGBOT_TOKEN_PAD=32).

Paths timed per matmul:
  bf16   torch.compile matmul (what the pipeline runs today)
  mx_qq  nkilib matmul_mxfp8, both operands bf16, quantized inside the kernel every call
  mx_pq  weight quantized ONCE (quantize_block_mxfp8_kernel), activation quantized per call
Errors: relative RMS vs an fp32 CPU matmul, overall and worst output channel, for two inputs
(Gaussian, and Gaussian with 1% outlier channels x30). Then the whole FFN chained.

  NEURON_LOGICAL_NC_CONFIG=2 python3 perf/mx_ffn_bench.py
Env: CKPT, BLOCK (10), TP (4), M (544), REPS (50), FP8 (float8_e4m3fn | float8_e5m2)
"""
import math
import os
import time
import traceback

import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl

from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel

CKPT = os.environ.get("CKPT", "/node-cache/lingbot-world-v2-14b-causal-fast")
K_BLK = int(os.environ.get("BLOCK", "10"))
TP = int(os.environ.get("TP", "4"))
M = int(os.environ.get("M", "544"))
REPS = int(os.environ.get("REPS", "50"))
FP8 = os.environ.get("FP8", "float8_e4m3fn")
DEV = torch.device("neuron:0")


@nki.jit
def mx_qq(lhs, rhs):
    """lhs [M, K] bf16, rhs [N, K] bf16 -> [M, N] bf16; both quantized in-kernel."""
    return matmul_mxfp8(lhs, rhs, float8_dtype=FP8, output_dtype=nl.bfloat16, run_with_lnc2=True,
                        lhs_is_swizzled=False, rhs_is_swizzled=False)


@nki.jit
def mx_quant(w):
    """w [F, K] bf16 -> (scales [K/4, F], data [K/4, 4F]), packed scales."""
    return quantize_block_mxfp8_kernel(w, FP8, True, True)


@nki.jit
def mx_pq(lhs, w_data, w_scales):
    """lhs [M, K] bf16 (quantized in-kernel) x pre-quantized weight -> [M, N] bf16."""
    return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                        output_dtype=nl.bfloat16, run_with_lnc2=True, use_scale_packing=True,
                        lhs_is_swizzled=False)


def bench(fn, *a):
    for _ in range(3):
        out = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3, out


def err(y, ref):
    y, ref = y.float().cpu(), ref.float()
    y = y[:, :ref.shape[1]]
    d = y - ref
    rel = (d.norm() / ref.norm()).item()
    ch = (d.norm(dim=0) / ref.norm(dim=0).clamp_min(1e-12)).max().item()
    return f"rel {rel:.2e}  worst-ch {ch:.2e}  max|d| {d.abs().max().item():.2e}"


def pad_k(t, mult=512):
    k = t.shape[1]
    kp = math.ceil(k / mult) * mult
    return F.pad(t, (0, kp - k)) if kp != k else t


def main():
    print(f"[mx] block {K_BLK} TP{TP} rank0 shard, M={M}, {FP8}, reps {REPS}", flush=True)
    sd = torch.load(os.path.join(CKPT, "transformers-bf16.pth"), map_location="cpu",
                    weights_only=True, mmap=True)
    p = f"blocks.{K_BLK}.ffn."
    W1, b1 = sd[p + "0.weight"], sd[p + "0.bias"]
    W2 = sd[p + "2.weight"]
    n1 = W1.shape[0] // TP
    W1 = W1[:n1].contiguous().to(torch.bfloat16)          # [3456, 5120]
    b1 = b1[:n1].float()
    W2 = W2[:, :n1].contiguous().to(torch.bfloat16)       # [5120, 3456]
    del sd
    print(f"[mx] W1 {tuple(W1.shape)} W2 {tuple(W2.shape)}", flush=True)

    g = torch.Generator().manual_seed(0)
    x_g = torch.randn(M, W1.shape[1], generator=g)
    x_o = x_g.clone()
    idx = torch.randperm(x_o.shape[1], generator=g)[: x_o.shape[1] // 100]
    x_o[:, idx] *= 30.0
    inputs = {"gauss": x_g.to(torch.bfloat16), "outlier": x_o.to(torch.bfloat16)}

    mm = torch.compile(lambda a, w: a @ w.t(), backend="neuron", fullgraph=True, dynamic=False)
    W1d, W2d = W1.to(DEV), W2.to(DEV)
    W2p = pad_k(W2)                                      # K 3456 -> 3584 (zeros) for MX
    W2pd = W2p.to(DEV)
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    qq, quant, pq = mx_qq[lnc], mx_quant[lnc], mx_pq[lnc]

    wq = {}
    for name, w in (("W1", W1d), ("W2", W2pd)):
        try:
            t0 = time.perf_counter()
            s, dta = quant(w)
            torch.neuron.synchronize()
            wq[name] = (dta, s)
            print(f"[mx] quantized {name} once: data {tuple(dta.shape)} {dta.dtype}, scales "
                  f"{tuple(s.shape)} {s.dtype} ({(time.perf_counter() - t0) * 1e3:.0f} ms incl. compile)",
                  flush=True)
        except Exception:
            print(f"[mx] weight pre-quantize {name} FAILED:\n{traceback.format_exc()}", flush=True)

    for iname, x in inputs.items():
        ref_u = x.float() @ W1.float().t()
        h = F.gelu(ref_u + b1, approximate="tanh").to(torch.bfloat16)
        ref_d = h.float() @ W2.float().t()
        xd, hd, hpd = x.to(DEV), h.to(DEV), pad_k(h).to(DEV)
        print(f"\n[mx] input {iname}", flush=True)
        for op, a, ad, apd, w, wd, wpd, wname, ref in (
                ("up", x, xd, xd, W1, W1d, W1d, "W1", ref_u),
                ("down", h, hd, hpd, W2, W2d, W2pd, "W2", ref_d)):
            fl = 2 * a.shape[0] * a.shape[1] * w.shape[0]
            rows = [("bf16", lambda: bench(mm, ad, wd)),
                    ("mx_qq", lambda: bench(qq, apd, wpd))]
            if wname in wq:
                rows.append(("mx_pq", lambda: bench(pq, apd, *wq[wname])))
            for path, run in rows:
                try:
                    ms, y = run()
                    print(f"[mx] {op:4s} {path:5s} {ms * 1e3:7.1f} us  {fl / ms / 1e9:6.1f} TFLOPS  "
                          f"{err(y, ref)}", flush=True)
                except Exception:
                    print(f"[mx] {op} {path} FAILED:\n{traceback.format_exc()}", flush=True)

        # whole FFN chained (MX for both matmuls, bias/GELU in torch), vs fp32 chain
        try:
            ffn_ref = (F.gelu(x.float() @ W1.float().t() + b1, approximate="tanh")
                       @ W2.float().t())
            b1d = b1.to(DEV)

            def ffn_bf16():
                u = mm(xd, W1d)
                return mm(F.gelu(u.float() + b1d, approximate="tanh").to(torch.bfloat16), W2d)

            def ffn_mx():
                u = pq(xd, *wq["W1"]) if "W1" in wq else qq(xd, W1d)
                hh = pad_k(F.gelu(u.float() + b1d, approximate="tanh").to(torch.bfloat16))
                return pq(hh, *wq["W2"]) if "W2" in wq else qq(hh, W2pd)

            for path, fn in (("bf16", ffn_bf16), ("mx", ffn_mx)):
                ms, y = bench(fn)
                print(f"[mx] ffn  {path:5s} {ms * 1e3:7.1f} us (eager glue)  {err(y, ffn_ref)}",
                      flush=True)
        except Exception:
            print(f"[mx] ffn chain FAILED:\n{traceback.format_exc()}", flush=True)
    print("[mx] done", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
