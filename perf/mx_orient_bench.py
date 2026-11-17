"""Candidate 5 (principle 2): does matmul_mxfp8 run faster when the activation arrives K-major?

The step-3 pair showed the MX kernel transposing its [M, K] activation on the PE (load_apis.py:1118
MATMUL(transpose)) for as long as the MX matmul itself (FFN up: 0.14 vs 0.14 ms). Here, at the LNC1
per-rank shapes with the tuned tiles of kernels/ffn_mx.py, the current call (x [M, K],
lhs_is_swizzled=False) is timed against x^T [K, M] passed with every lhs_orientation the installed
nkilib offers. Kill: no orientation faster than the current call by more than its self-noise.

  NEURON_LOGICAL_NC_CONFIG=1 python3 perf/mx_orient_bench.py
"""
import inspect
import os
import time
import traceback

import torch
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl

from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.mxfp_utils.mxfp8_utils import common_dataclasses as cd
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel

FP8 = "float8_e4m3fn"
M = int(os.environ.get("M", "544"))
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")
TUNED = {  # kernels/ffn_mx.py _TUNED (LNC1)
    "qkv": (5120, 1280, dict(tn=1280, bm=4, bn=1, bk=10, lm=4, ln=1, sr=False)),
    "ffn_up": (5120, 3456, dict(tn=512, bm=4, bn=4, bk=10, lm=4, ln=2, sr=True)),
    "ffn_down": (3584, 5120, dict(tn=512, bm=4, bn=4, bk=7, lm=4, ln=4, sr=True)),
}
print("[orient] TensorOrientation members:", list(cd.TensorOrientation.__members__), flush=True)
print("[orient] matmul_mxfp8 doc on orientation:",
      [l.strip() for l in (inspect.getdoc(matmul_mxfp8) or "").splitlines() if "orient" in l.lower()][:6], flush=True)


@nki.jit
def _quant(w):
    return quantize_block_mxfp8_kernel(w, FP8, False, True)


def make(t, orient):
    kw = dict(TILES_IN_BLOCK_M=t["bm"], TILES_IN_BLOCK_N=t["bn"], TILES_IN_BLOCK_K=t["bk"],
              TILES_IN_LOAD_M=t["lm"], TILES_IN_LOAD_N=t["ln"],
              lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, t["tn"]),
              spill_reload=t["sr"])
    if orient is not None:
        kw["lhs_orientation"] = orient

    @nki.jit
    def k(lhs, d, s):
        return matmul_mxfp8(lhs, d, rhs_scales=s, float8_dtype=FP8, output_dtype=nl.bfloat16,
                            run_with_lnc2=False, use_scale_packing=True, lhs_is_swizzled=False, **kw)
    return k[1]


def bench(fn, *a):
    for _ in range(3):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


for name, (K, N, t) in TUNED.items():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    xd, xtd, wd = x.to(DEV), x.t().contiguous().to(DEV), w.to(DEV)
    s, d = _quant[1](wd)
    torch.neuron.synchronize()
    base = make(t, None)
    for rep in (1, 2):
        us, y = bench(base, xd, d, s)
        r = ((y.float().cpu() - ref).norm() / ref.norm()).item()
        print(f"[orient] {name} current x[M,K]          run{rep} {us:7.1f} us  rel {r:.2e}", flush=True)
    for o in cd.TensorOrientation:
        for lhs, tag in ((xtd, "x^T[K,M]"), (xd, "x[M,K]  ")):
            try:
                us, y = bench(make(t, o), lhs, d, s)
                yy = y.float().cpu()
                r = ((yy - ref).norm() / ref.norm()).item() if yy.shape == ref.shape else float("nan")
                print(f"[orient] {name} {o.name:10s} {tag} {us:7.1f} us  rel {r:.2e}  out {tuple(yy.shape)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[orient] {name} {o.name:10s} {tag} FAILED: {str(e).splitlines()[0][:140]}", flush=True)
print("[orient] done", flush=True)
