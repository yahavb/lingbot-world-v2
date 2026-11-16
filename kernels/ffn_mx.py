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
