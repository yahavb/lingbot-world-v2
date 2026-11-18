"""kiro_mx4: does matmul_mxfp8 at LNC2 split the work so that each core preps (loads, DMA-transposes, quantizes) the
whole activation? lnc_2_shard_rhs=True (default) shards N (each core needs all M rows of the lhs), False shards M.
Shapes: the block's per-rank Linears at the 2-chip (M=2176) and 4-chip (M=1088) layouts, one logical core.
  NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 python3 perf/mx_shard_bench.py
"""
import os, time
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel
FP8 = "float8_e4m3fn"; REPS = int(os.environ.get("REPS", "30")); DEV = torch.device("neuron:0")
SHAPES = {"qkv": (5120, 1280), "ffn_up": (5120, 3456), "ffn_down": (3584, 5120), "o": (1536, 5120),
          "sq": (5120, 5120), "ffn_up_pad": (5120, 3584), "qkv3": (5120, 3840)}
# MS="544,3808" ONLY="sq": the M values / shapes to run (run with NKI_ENABLE_TRACE_CACHE=0: the trace cache keys on
# the kernel arguments, so the N- and M-shard kernels of one shape would otherwise share one compiled kernel)
MS = [int(v) for v in os.environ.get("MS", "2176,1088").split(",")]
ONLY = [v for v in os.environ.get("ONLY", "").split(",") if v]
TUNED = {  # kernels/ffn_mx.py _TUNED_LNC2_M2176 / _TUNED_LNC2 (bm, bn, bk, lm, ln, tn, sr)
    2176: {"qkv": (8, 1, 10, 4, 1, 1280, True), "ffn_up": (16, 13, 2, 8, 13, 128, True),
           "ffn_down": (8, 5, 7, 4, 5, 512, True), "o": (16, 4, 3, 4, 4, 512, True)},
    1088: {"ffn_up": (8, 1, 4, 8, 1, 1728, True), "ffn_down": (8, 5, 7, 8, 5, 512, False), "o": (8, 5, 3, 8, 5, 512, False)},
}
@nki.jit
def quant(w):
    return quantize_block_mxfp8_kernel(w, FP8, True, True)
def mk(shard_rhs, tiles):
    kw = {}
    if tiles:
        bm, bn, bk, lm, ln, tn, sr = tiles
        kw = dict(TILES_IN_BLOCK_M=bm, TILES_IN_BLOCK_N=bn, TILES_IN_BLOCK_K=bk, TILES_IN_LOAD_M=lm, TILES_IN_LOAD_N=ln,
                  lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, tn), spill_reload=sr)
    @nki.jit
    def k(x, wd, ws):
        return matmul_mxfp8(x, wd, rhs_scales=ws, float8_dtype=FP8, output_dtype=nl.bfloat16, run_with_lnc2=True,
                            lnc_2_shard_rhs=shard_rhs, use_scale_packing=True, lhs_is_swizzled=False,
                            fast_dma_transpose=True, **kw)
    return k[2]
def bench(fn):
    for _ in range(3): y = fn()
    torch.neuron.synchronize(); t0 = time.perf_counter()
    for _ in range(REPS): y = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y
for M in MS:
    for name, (K, N) in SHAPES.items():
        if ONLY and name not in ONLY:
            continue
        g = torch.Generator().manual_seed(0)
        x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(DEV)
        s, d = quant[2]((torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).to(DEV))
        torch.neuron.synchronize()
        res, ref = [], None
        for tiles_tag, tiles in (("default", None), ("tuned", TUNED.get(M, {}).get(name))):
            if tiles_tag == "tuned" and tiles is None:
                continue
            for shard_rhs in (True, False):
                try:
                    kern = mk(shard_rhs, tiles)
                    us, y = bench(lambda: kern(x, d, s))
                    ref = y if ref is None else ref
                    rel = ((y.float() - ref.float()).norm() / ref.float().norm()).item()
                    res.append(f"{tiles_tag}/{'N' if shard_rhs else 'M'}-shard {us:6.1f}us(rel {rel:.0e})")
                except Exception as e:  # noqa: BLE001
                    res.append(f"{tiles_tag}/{'N' if shard_rhs else 'M'}-shard FAILED({str(e).splitlines()[0][:80]})")
        print(f"[shard] M{M} {name:8s} K{K} N{N}: " + " | ".join(res), flush=True)
print("[shard] done", flush=True)
