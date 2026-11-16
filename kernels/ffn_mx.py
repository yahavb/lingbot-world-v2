"""MXFP8 FFN matmuls (LINGBOT_FFN_MX=1) on nkilib's matmul_mxfp8 / quantize_block_mxfp8_kernel.

Weights are quantized ONCE (prequantize_weight) to MXFP8 data + packed e8m0 scales; the activation
is quantized inside the matmul kernel on every call. perf/mx_ffn_bench.py measured, at the
TP4 x CP8 per-rank shapes, up 232 -> 173 us and down 216 -> 147 us, error vs fp32 3.7% per matmul
(bf16: 0.17%).

LINGBOT_MX_FP8: float8_e4m3fn (default) | float8_e5m2.
"""
import os

import nki
import nki.language as nl
from nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8
from nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import quantize_block_mxfp8_kernel
from torch_neuronx.nki_hop import wrap_nki

FP8 = os.environ.get("LINGBOT_MX_FP8", "float8_e4m3fn")
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
K_ALIGN = 512   # the quantize kernel needs K % 512 == 0; pad K with zeros


@nki.jit
def _quant(w):
    return quantize_block_mxfp8_kernel(w, FP8, LNC == 2, True)


@nki.jit
def _mm_pq(lhs, w_data, w_scales):
    return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                        output_dtype=nl.bfloat16, run_with_lnc2=(LNC == 2),
                        use_scale_packing=True, lhs_is_swizzled=False)


def prequantize_weight(w):
    """w [N, K] bf16 on device, K % 512 == 0 -> (data, scales) on device. Eager, once."""
    scales, data = _quant[LNC](w)
    return data, scales


mm_eager = _mm_pq[LNC]       # x [M, K] bf16, (data, scales) -> [M, N] bf16
mm = wrap_nki(_mm_pq)        # the same, traceable inside a fullgraph torch.compile region


# LINGBOT_MX_TUNED=1 (LNC1): per-shape tiles from perf/mx_tile_sweep.py at the LNC1 TP4 x CP8
# per-rank shapes (M=544), one Neuron core, us/call vs bf16 torch.matmul at the same shape:
#   (K, N) = (5120, 1280) q/k/v, cross q/k/v, cam layer1/scale/shift: 99.9 vs 172.5 (default 130.8)
#            (1536, 5120) o, cross o, cam layer2:                    107.1 vs 201.8 (default 183.5)
#            (5120, 3456) ffn.0:                                       192.9 vs 407.7 (default 251.9)
#            (3584, 5120) ffn.2:                                       173.0 vs 425.9 (default 192.0)
_TUNED = {
    (5120, 1280): dict(tn=1280, bm=4, bn=1, bk=10, lm=4, ln=1, sr=False, extra={"load_with_PE_swizzle": True}),
    (1536, 5120): dict(tn=512, bm=4, bn=4, bk=3, lm=4, ln=2, sr=True, extra={}),
    (5120, 3456): dict(tn=512, bm=4, bn=4, bk=10, lm=4, ln=2, sr=True, extra={"load_with_PE_swizzle": True}),
    (3584, 5120): dict(tn=512, bm=4, bn=4, bk=7, lm=4, ln=4, sr=True, extra={"load_with_PE_swizzle": True}),
}
_TUNED_ON = os.environ.get("LINGBOT_MX_TUNED", "0") == "1" and LNC == 1
# LINGBOT_MX_TUNED_KEYS="5120x1280,1536x5120": tune only these (K, N) (bisecting in-graph failures)
_TUNED_KEYS = os.environ.get("LINGBOT_MX_TUNED_KEYS", "")
if _TUNED_KEYS:
    _TUNED = {k: v for k, v in _TUNED.items() if f"{k[0]}x{k[1]}" in _TUNED_KEYS.split(",")}
_TUNED_MM = {}


def _tuned_kernel(key):
    t = _TUNED[key]
    kw = dict(TILES_IN_BLOCK_M=t["bm"], TILES_IN_BLOCK_N=t["bn"], TILES_IN_BLOCK_K=t["bk"],
              TILES_IN_LOAD_M=t["lm"], TILES_IN_LOAD_N=t["ln"],
              lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, t["tn"]),
              spill_reload=t["sr"],
              **({} if os.environ.get("LINGBOT_MX_TUNED_NOEXTRA", "0") == "1" else t["extra"]))

    @nki.jit
    def k(lhs, w_data, w_scales):
        return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                            output_dtype=nl.bfloat16, run_with_lnc2=False,
                            use_scale_packing=True, lhs_is_swizzled=False, **kw)
    return wrap_nki(k)


_mm_default = mm


def mm(x, w_data, w_scales):  # noqa: F811 -- tuned dispatch over the default kernel
    if _TUNED_ON:
        key = (int(w_data.shape[0]) * 4, int(w_data.shape[1]) // 4)
        if key in _TUNED:
            if key not in _TUNED_MM:
                _TUNED_MM[key] = _tuned_kernel(key)
            return _TUNED_MM[key](x, w_data, w_scales)
    return _mm_default(x, w_data, w_scales)
