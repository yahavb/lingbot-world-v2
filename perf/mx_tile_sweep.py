"""Tile-config sweep of nkilib matmul_mxfp8 at one block's per-rank Linear shapes (pre-quantized
weight, activation quantized in-kernel), against bf16 torch.matmul at the same shape.

e2e-latency-optimization candidate "MX tiles": the block profile (perf/mx_block_prof.sh) showed each
MX Linear re-reading 262-524 MB per call (spill reload) against 4-18 MB of weights, with the default
tiles. Kill criterion: no candidate beats bf16 at the N=1280 shapes by more than the self-noise.

  NEURON_LOGICAL_NC_CONFIG=1 SHAPE=qkv python3 perf/mx_tile_sweep.py
Env: SHAPE (qkv | o | ffn_up | ffn_down | cross_kv | all), M (544), MAXC (24 candidates per shape,
     evenly spaced over nkilib's generate_autotune_candidates), REPS (30), FP8 (float8_e4m3fn).
Prints one line per candidate: us/call, TFLOPS, rel error vs fp32; then the best per shape as a
_AUTOTUNE_CACHE-style dict.
"""
import json
import math
import os
import time
import traceback

import torch
import torch.nn.functional as F
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl

from nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_config as _cfgmod
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import (MatmulMxfp8KernelConfig,
                                                                  generate_autotune_candidates)
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel

LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
FP8 = os.environ.get("FP8", "float8_e4m3fn")
M = int(os.environ.get("M", "544"))
MAXC = int(os.environ.get("MAXC", "24"))
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")
# (name, K, N): K already padded to 512 as kernels/ffn_mx.py pads it
SHAPES = {"qkv": (5120, 1280), "o": (1536, 5120), "ffn_up": (5120, 3456),
          "ffn_down": (3584, 5120), "cross_kv": (5120, 1280), "qkv3": (5120, 3840), "ffn_up_pad": (5120, 3584),
          # full (TP-free) weights on one TP rank's token shard (M = 544 at TP4 x CP2)
          "ffn_up_full": (5120, 13824), "ffn_down_full": (13824, 5120), "sq_full": (5120, 5120)}
# FAST=1: every candidate with fast_dma_transpose (the deployed LINGBOT_MX_FAST_DMA=1 condition)
FAST = os.environ.get("FAST", "0") == "1"


@nki.jit
def _quant(w):
    return quantize_block_mxfp8_kernel(w, FP8, LNC == 2, True)


def make_kernel(c, extra=None):
    kw = dict(extra or {})
    if FAST and not kw.get("disable_dma_transpose") and not kw.get("load_with_PE_swizzle"):
        kw["fast_dma_transpose"] = True
    if c is not None:
        # (an earlier version replaced kw here, silently dropping `extra` and the transpose variants)
        kw.update(TILES_IN_BLOCK_M=c.TILES_IN_BLOCK_M, TILES_IN_BLOCK_N=c.TILES_IN_BLOCK_N,
                  TILES_IN_BLOCK_K=c.TILES_IN_BLOCK_K, TILES_IN_LOAD_M=c.TILES_IN_LOAD_M,
                  TILES_IN_LOAD_N=c.TILES_IN_LOAD_N,
                  lhs_matmul_tile_shape_logical=(c.tile_k, c.tile_m),
                  rhs_matmul_tile_shape_logical=(c.tile_k, c.tile_n),
                  spill_reload=bool(getattr(c, "spill_reload", False)))

    @nki.jit
    def k(lhs, w_data, w_scales):
        return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                            output_dtype=nl.bfloat16, run_with_lnc2=(LNC == 2),
                            use_scale_packing=True, lhs_is_swizzled=False, **kw)
    return k[LNC]


def bench(fn, *a):
    for _ in range(3):
        y = fn(*a)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        y = fn(*a)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y


def rel(y, ref):
    y = y.float().cpu()[:, :ref.shape[1]]
    return ((y - ref).norm() / ref.norm()).item()


def sweep(name, K, N):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    xd, wd = x.to(DEV), w.to(DEV)
    fl = 2 * M * K * N
    mm = torch.compile(lambda a, b: a @ b.t(), backend="neuron", fullgraph=True, dynamic=False)
    us, y = bench(mm, xd, wd)
    us2, _ = bench(mm, xd, wd)
    print(f"[tile] {name} M{M} K{K} N{N}  bf16 {us:7.1f} us (repeat {us2:7.1f})  "
          f"{fl / us / 1e6:6.1f} TFLOPS  rel {rel(y, ref):.2e}", flush=True)
    s, d = _quant[LNC](wd)
    torch.neuron.synchronize()
    # this nkilib: lnc_sharding enum instead of run_with_lnc2 (OFF at LNC1)
    _kw = {} if LNC == 2 else {"lnc_sharding": _cfgmod.LncShardingMode.OFF}
    cands = generate_autotune_candidates(MatmulMxfp8KernelConfig(M=M, K=K, N=N, **_kw))
    step = max(1, len(cands) // MAXC)
    pick = [None] + cands[::step][:MAXC]
    print(f"[tile] {name}: {len(cands)} autotune candidates, timing {len(pick)} (default first)", flush=True)
    best = None
    for c in pick:
        tag = "default" if c is None else (f"tn{c.tile_n} B{c.TILES_IN_BLOCK_M}/{c.TILES_IN_BLOCK_N}/"
                                           f"{c.TILES_IN_BLOCK_K} L{c.TILES_IN_LOAD_M}/{c.TILES_IN_LOAD_N}"
                                           f"{' sr' if getattr(c, 'spill_reload', False) else ''}")
        try:
            t0 = time.perf_counter()
            k = make_kernel(c)
            us_c, yc = bench(k, xd, d, s)
            print(f"[tile] {name} {tag:28s} {us_c:7.1f} us  {fl / us_c / 1e6:6.1f} TFLOPS  "
                  f"rel {rel(yc, ref):.2e}  ({time.perf_counter() - t0:.0f} s incl compile)", flush=True)
            if best is None or us_c < best[0]:
                best = (us_c, tag, c)
        except Exception as e:  # noqa: BLE001
            print(f"[tile] {name} {tag:28s} FAILED: {str(e).splitlines()[0][:150]}", flush=True)
    # transpose placement on the best tiles: DMA transpose variants / PE swizzle loads
    if best:
        for extra in ({"fast_dma_transpose": True}, {"disable_dma_transpose": True},
                      {"load_with_PE_swizzle": True}):
            tag = best[1] + " " + ",".join(extra)
            try:
                us_c, yc = bench(make_kernel(best[2], extra), xd, d, s)
                print(f"[tile] {name} {tag:40s} {us_c:7.1f} us  {fl / us_c / 1e6:6.1f} TFLOPS  "
                      f"rel {rel(yc, ref):.2e}", flush=True)
                if us_c < best[0]:
                    best = (us_c, tag, best[2])
            except Exception as e:  # noqa: BLE001
                print(f"[tile] {name} {tag:40s} FAILED: {str(e).splitlines()[0][:150]}", flush=True)
    if best:
        c = best[2]
        cfg = None if c is None else {k_: getattr(c, k_) for k_ in (
            "tile_m", "tile_k", "tile_n", "TILES_IN_BLOCK_M", "TILES_IN_BLOCK_N", "TILES_IN_BLOCK_K",
            "TILES_IN_LOAD_M", "TILES_IN_LOAD_N", "spill_reload")}
        print(f"[tile] BEST {name}: {best[0]:.1f} us ({best[1]}) vs bf16 {min(us, us2):.1f} us  "
              f"config {json.dumps(cfg)}", flush=True)


def main():
    names = list(SHAPES) if os.environ.get("SHAPE", "all") == "all" else os.environ["SHAPE"].split(",")
    for n in names:
        try:
            sweep(n, *SHAPES[n])
        except Exception:
            print(f"[tile] {n} sweep FAILED:\n{traceback.format_exc()}", flush=True)
    print("[tile] done", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
