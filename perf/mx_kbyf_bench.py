"""LNC2, M=2176 (2-chip): matmul_mxfp8 with the activation token-major [M, K] (deployed: DMA transpose in the kernel,
fast_dma_transpose) vs feature-major [K, M] (lhs_orientation=K_BY_F, no activation transpose), default and tuned
tiles. One logical core, eager.
  PYTHONPATH=. NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/mx_kbyf_bench.py
"""
import os
import time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.mxfp_utils.mxfp8_utils import common_dataclasses as cd
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel
FP8 = "float8_e4m3fn"
M = int(os.environ.get("M", "2176"))
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")
SHAPES = {"qkv": (5120, 1280), "ffn_up": (5120, 3584), "ffn_down": (3584, 5120), "o": (1536, 5120)}
TUNED = {"qkv": (8, 1, 10, 4, 1, 1280, True), "o": (16, 4, 3, 4, 4, 512, True)}
print("[kbyf] orientations:", list(cd.TensorOrientation.__members__), flush=True)


@nki.jit
def _quant(w):
    return quantize_block_mxfp8_kernel(w, FP8, True, True)


def make(orient, tiles, fast):
    kw = {}
    if tiles:
        bm, bn, bk, lm, ln, tn, sr = tiles
        kw = dict(TILES_IN_BLOCK_M=bm, TILES_IN_BLOCK_N=bn, TILES_IN_BLOCK_K=bk, TILES_IN_LOAD_M=lm,
                  TILES_IN_LOAD_N=ln, lhs_matmul_tile_shape_logical=(512, 128),
                  rhs_matmul_tile_shape_logical=(512, tn), spill_reload=sr)
    if orient is not None:
        kw["lhs_orientation"] = orient
    if fast:
        kw["fast_dma_transpose"] = True

    @nki.jit
    def k(lhs, d, s):
        return matmul_mxfp8(lhs, d, rhs_scales=s, float8_dtype=FP8, output_dtype=nl.bfloat16, run_with_lnc2=True,
                            use_scale_packing=True, lhs_is_swizzled=False, **kw)
    return k[2]


def bench(fn, *a):
    for _ in range(3):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


for name, (K, N) in SHAPES.items():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    s, d = _quant[2](w.to(DEV))
    xd, xtd = x.to(DEV), x.t().contiguous().to(DEV)
    res = []
    for tag, inp, orient, tiles, fast in (
            ("FbyK fast default", xd, None, None, True),
            ("FbyK fast tuned", xd, None, TUNED.get(name), True),
            ("KbyF default", xtd, cd.TensorOrientation.K_BY_F, None, False),
            ("KbyF tuned", xtd, cd.TensorOrientation.K_BY_F, TUNED.get(name), False)):
        if "tuned" in tag and TUNED.get(name) is None:
            continue
        try:
            us, y = bench(make(orient, tiles, fast), inp, d, s)
            rel = ((y.float().cpu() - ref).norm() / ref.norm()).item()
            res.append(f"{tag} {us:6.1f}us (rel {rel:.1e})")
        except Exception as e:  # noqa: BLE001
            res.append(f"{tag} FAILED({str(e).splitlines()[0][:100]})")
    print(f"[kbyf] M{M} {name:8s} K{K} N{N}: " + " | ".join(res), flush=True)
print("[kbyf] done", flush=True)
