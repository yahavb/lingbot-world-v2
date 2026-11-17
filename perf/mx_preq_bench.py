"""Step 1 microbench (kiro_mx4): q/k/v share one input. Today each MX Linear loads, transposes and quantizes that
input itself (3x). Candidate: quantize it once (quantize_block_mxfp8_kernel) and run the three matmuls on the
pre-quantized lhs. One logical core, the 2-chip per-rank shape (M=2176, K=5120, N=1280 per projection at TP4).
  NEURON_RT_VISIBLE_CORES=<c> NEURON_LOGICAL_NC_CONFIG=2 M=2176 python3 perf/mx_preq_bench.py
Kill: quant + 3 x pre-quantized matmul not faster than 3 x the current (tuned, bf16-lhs) call.
"""
import os, time, traceback
import torch
import torch_neuronx  # noqa: F401
import nki
import nki.language as nl
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
FP8 = "float8_e4m3fn"
M, K, N = int(os.environ.get("M", "2176")), int(os.environ.get("K", "5120")), int(os.environ.get("N", "1280"))
REPS = int(os.environ.get("REPS", "30"))
DEV = torch.device("neuron:0")
@nki.jit
def quant(x):
    return quantize_block_mxfp8_kernel(x, FP8, LNC == 2, True)
TUNED = dict(TILES_IN_BLOCK_M=8, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=10, TILES_IN_LOAD_M=4, TILES_IN_LOAD_N=1,
             lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, 1280), spill_reload=True)
def mk_bf16(tiles):
    @nki.jit
    def k(lhs, wd, ws):
        return matmul_mxfp8(lhs, wd, rhs_scales=ws, float8_dtype=FP8, output_dtype=nl.bfloat16,
                            run_with_lnc2=(LNC == 2), use_scale_packing=True, lhs_is_swizzled=False,
                            fast_dma_transpose=True, **tiles)
    return k[LNC]
def mk_preq(tiles):
    @nki.jit
    def k(xd, xs, wd, ws):
        return matmul_mxfp8(xd, wd, lhs_scales=xs, rhs_scales=ws, float8_dtype=FP8, output_dtype=nl.bfloat16,
                            run_with_lnc2=(LNC == 2), use_scale_packing=True, **tiles)
    return k[LNC]
def bench(fn):
    for _ in range(3): y = fn()
    torch.neuron.synchronize(); t0 = time.perf_counter()
    for _ in range(REPS): y = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6, y
def main():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16).to(DEV)
    ws = [quant[LNC]((torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16).to(DEV)) for _ in range(3)]  # (scales, data)
    torch.neuron.synchronize()
    ref_k = mk_bf16(TUNED)
    us_cur, y_cur = bench(lambda: [ref_k(x, w[1], w[0]) for w in ws])
    print(f"[preq] current 3 x (bf16 lhs, tuned): {us_cur:8.1f} us", flush=True)
    us_q, xq = bench(lambda: quant[LNC](x))
    print(f"[preq] quantize x once: {us_q:8.1f} us; scales {tuple(xq[0].shape)} {xq[0].dtype}, data {tuple(xq[1].shape)} {xq[1].dtype}", flush=True)
    for name, tiles in (("default", {}), ("tuned", {k_: v for k_, v in TUNED.items() if k_ != "spill_reload"})):
        try:
            kk = mk_preq(tiles)
            us_m, y_m = bench(lambda: [kk(xq[1], xq[0], w[1], w[0]) for w in ws])
            us_all, _ = bench(lambda: (lambda q_: [kk(q_[1], q_[0], w[1], w[0]) for w in ws])(quant[LNC](x)))
            rel = max(((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(y_m, y_cur))
            print(f"[preq] pre-quantized {name:8s}: 3 matmuls {us_m:8.1f} us, quant + 3 matmuls {us_all:8.1f} us "
                  f"(vs current {us_cur:.1f}), rel vs current {rel:.2e}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[preq] pre-quantized {name}: FAILED {str(e).splitlines()[0][:300]}", flush=True)
            traceback.print_exc()
    print("[preq] done", flush=True)
if __name__ == "__main__":
    with torch.no_grad():
        main()
