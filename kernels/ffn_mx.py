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


# LINGBOT_MX_FAST_DMA=1: the activation's DMA gather-transpose uses the direct 4-D access pattern
# (nkilib fast_dma_transpose / SwizzleMode.DGT_FAST) instead of vector offsets generated on GpSimd.
# perf/mx_tile_sweep.py at the LNC2 TP4 x CP4 per-rank shapes (M=1088), us/call: qkv 137.1 -> 105.9.
_FAST_DMA = os.environ.get("LINGBOT_MX_FAST_DMA", "0") == "1"
# LINGBOT_MX_PE_SWIZZLE=1 (experiment): the activation's layout change runs on the tensor engine (nkilib
# load_with_PE_swizzle) instead of the DMA queues. In the 4-chip DiT block the DMA queues are the binding resource
# (82% busy, 37% of it these transposes) while the tensor engine has headroom (68%): slower alone, maybe faster in the block.
_PE_SWIZZLE = os.environ.get("LINGBOT_MX_PE_SWIZZLE", "0") == "1"
_MX_LOAD = ({"load_with_PE_swizzle": True} if _PE_SWIZZLE else {"fast_dma_transpose": True} if _FAST_DMA else {})
@nki.jit
def _mm_pq(lhs, w_data, w_scales):
    return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                        output_dtype=nl.bfloat16, run_with_lnc2=(LNC == 2),
                        use_scale_packing=True, lhs_is_swizzled=False, **_MX_LOAD)


def prequantize_weight(w):
    """w [N, K] bf16 on device, K % 512 == 0 -> (data, scales) on device. Eager, once."""
    scales, data = _quant[LNC](w)
    return data, scales


mm_eager = _mm_pq[LNC]       # x [M, K] bf16, (data, scales) -> [M, N] bf16
mm = wrap_nki(_mm_pq)        # the same, traceable inside a fullgraph torch.compile region
# LINGBOT_MX_NSHARD="KxN[,KxN]" (LNC2): these (K, N) take the default tiles with the two cores splitting N
# (lnc_2_shard_rhs=True: each core all M rows, half the output columns) instead of nkilib's automatic M split
# (every core then quantizes half the rows but loads all of the weight), ahead of any tuned table.
# perf/mx_shard_bench.py with the NKI trace cache off, M=2176, (5120, 1280) q / k / v: default N-split 121.3 us,
# default M-split 143.1, the M2176 table's tiles (M split, what LINGBOT_MX_TUNED_M2176 ran) 277.9.
# An entry "KxN@M" applies only to calls with M rows (e.g. 5120x5120@544: the full-weight cross q / o, not the FiLM).
_NSHARD, _NSHARD_M = (), ()
for _k in os.environ.get("LINGBOT_MX_NSHARD", "").replace(":", ",").split(","):
    if not _k:
        continue
    _kn, _, _m = _k.partition("@")
    _key = tuple(int(v) for v in _kn.split("x"))
    if _m:
        _NSHARD_M += (_key + (int(_m),),)
    else:
        _NSHARD += (_key,)
@nki.jit
def _mm_pq_n(lhs, w_data, w_scales):
    return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                        output_dtype=nl.bfloat16, run_with_lnc2=(LNC == 2), lnc_2_shard_rhs=True,
                        use_scale_packing=True, lhs_is_swizzled=False, **_MX_LOAD)
_mm_nshard = wrap_nki(_mm_pq_n)


# LINGBOT_MX_TUNED=1 (LNC1): per-shape tiles from perf/mx_tile_sweep.py at the LNC1 TP4 x CP8
# per-rank shapes (M=544), one Neuron core, us/call vs bf16 torch.matmul at the same shape:
#   (K, N) = (5120, 1280) q/k/v, cross q/k/v, cam layer1/scale/shift: 99.9 vs 172.5 (default 130.8)
#            (1536, 5120) o, cross o, cam layer2:                    107.1 vs 201.8 (default 183.5)
#            (5120, 3456) ffn.0:                                       192.9 vs 407.7 (default 251.9)
#            (3584, 5120) ffn.2:                                       173.0 vs 425.9 (default 192.0)
_TUNED = {
    (5120, 1280): dict(tn=1280, bm=4, bn=1, bk=10, lm=4, ln=1, sr=False, extra={}),  # PE swizzle: NCC_ILMM003 in-graph
    (1536, 5120): dict(tn=512, bm=4, bn=4, bk=3, lm=4, ln=2, sr=True, extra={}),
    (5120, 3456): dict(tn=512, bm=4, bn=4, bk=10, lm=4, ln=2, sr=True, extra={"load_with_PE_swizzle": True}),
    (3584, 5120): dict(tn=512, bm=4, bn=4, bk=7, lm=4, ln=4, sr=True, extra={"load_with_PE_swizzle": True}),
}
# LNC2 (TP4 x CP4 on 4 devices, M=1088 per rank), perf/mx_tile_sweep.py with fast_dma_transpose,
# us/call vs bf16: qkv default tiles 105.9 vs 187.8 | ffn.0 194.7 vs 367.7 | ffn.2 192.5 vs 370.7 |
# o 99.1 vs 170.4 (kiro_mx4, 2026-11-17). Selected by LINGBOT_MX_TUNED=1 at LNC2.
_TUNED_LNC2 = {
    (5120, 3456): dict(tn=1728, bm=8, bn=1, bk=4, lm=8, ln=1, sr=True, extra={}),
    (3584, 5120): dict(tn=512, bm=8, bn=5, bk=7, lm=8, ln=5, sr=False, extra={}),
    (1536, 5120): dict(tn=512, bm=8, bn=5, bk=3, lm=8, ln=5, sr=False, extra={}),
}
# LNC2 at M=2176 rows (TP4 x CP2 on 2 devices), perf/mx_tile_sweep.py M=2176, best vs default vs bf16 us/call:
#   qkv 161.7 / 341.8 / 276.1 | ffn.0 283.6 / 781.5 / 647.9 | ffn.2 269.7 / 410.2 / 667.3 | o 125.4 / 149.4 / 301.9
# (the default tiles lose to bf16 on q/k/v and ffn.0 at this M). LINGBOT_MX_TUNED_M2176=1, selected by the call's rows.
_TUNED_LNC2_M2176 = {
    (5120, 1280): dict(tn=1280, bm=8, bn=1, bk=10, lm=4, ln=1, sr=True, extra={}),
    # fused q/k/v (LINGBOT_MX_QKV_FUSED), perf/mx_tile_sweep.py SHAPE=qkv3 FAST=1 M=2176: 244.4 us vs 3 x 106.6 separate
    (5120, 3840): dict(tn=128, bm=16, bn=15, bk=2, lm=8, ln=15, sr=False, extra={}),
    (5120, 3456): dict(tn=128, bm=16, bn=13, bk=2, lm=8, ln=13, sr=True, extra={}),
    (3584, 5120): dict(tn=512, bm=8, bn=5, bk=7, lm=4, ln=5, sr=True, extra={}),
    (1536, 5120): dict(tn=512, bm=16, bn=4, bk=3, lm=4, ln=4, sr=True, extra={"load_with_PE_swizzle": True}),
}
# LNC2 at M=544 rows (one TP rank's token shard at TP4 x CP2: LINGBOT_CROSS_TPFREE's full-weight cross q / o),
# perf/mx_tile_sweep.py SHAPE=sq_full M=544 FAST=1 with the NKI trace cache off: tn512 B4/4/10 L4/4 (all of K in one
# block, like the default) 144.7 us, B4/4/5 134.8 (K in two blocks), default ~160. LINGBOT_MX_TUNED_M544=1.
_TUNED_LNC2_M544 = {
    (5120, 5120): dict(tn=512, bm=4, bn=4, bk=10, lm=4, ln=4, sr=False, extra={}),
}
_TUNED_BY_M = {}
if LNC == 2:
    _TUNED = _TUNED_LNC2
    if os.environ.get("LINGBOT_MX_TUNED_M2176", "0") == "1":
        _TUNED_BY_M[2176] = _TUNED_LNC2_M2176
    if os.environ.get("LINGBOT_MX_TUNED_M544", "0") == "1":
        _TUNED_BY_M[544] = _TUNED_LNC2_M544
_TUNED_ON = os.environ.get("LINGBOT_MX_TUNED", "0") == "1"
# LINGBOT_MX_TUNED_KEYS="5120x1280,1536x5120": tune only these (K, N) (bisecting in-graph failures)
_TUNED_KEYS = os.environ.get("LINGBOT_MX_TUNED_KEYS", "")
if _TUNED_KEYS:
    _TUNED = {k: v for k, v in _TUNED.items() if f"{k[0]}x{k[1]}" in _TUNED_KEYS.split(",")}
_TUNED_MM = {}


def _tuned_kernel(key, table=None):
    t = (table or _TUNED)[key]
    kw = dict(TILES_IN_BLOCK_M=t["bm"], TILES_IN_BLOCK_N=t["bn"], TILES_IN_BLOCK_K=t["bk"],
              TILES_IN_LOAD_M=t["lm"], TILES_IN_LOAD_N=t["ln"],
              lhs_matmul_tile_shape_logical=(512, 128), rhs_matmul_tile_shape_logical=(512, t["tn"]),
              spill_reload=t["sr"],
              **({} if os.environ.get("LINGBOT_MX_TUNED_NOEXTRA", "0") == "1" else t["extra"]))

    # disable_dma_transpose (the 1x32 loader without a DMA transpose) excludes the DGT fast path
    if _PE_SWIZZLE and not kw.get("disable_dma_transpose"):
        kw.pop("fast_dma_transpose", None)
        kw["load_with_PE_swizzle"] = True         # LINGBOT_MX_PE_SWIZZLE: every MX Linear, tuned tiles too
    elif _FAST_DMA and not kw.get("disable_dma_transpose") and not kw.get("load_with_PE_swizzle"):
        kw["fast_dma_transpose"] = True
    @nki.jit
    def k(lhs, w_data, w_scales):
        return matmul_mxfp8(lhs, w_data, rhs_scales=w_scales, float8_dtype=FP8,
                            output_dtype=nl.bfloat16, run_with_lnc2=(LNC == 2),
                            use_scale_packing=True, lhs_is_swizzled=False, **kw)
    return wrap_nki(k)


_mm_default = mm


# LINGBOT_MX_M2176_DEFAULT="3584x5120[:1536x5120]": these (K, N) take the default tiles at M=2176 rows even under
# LINGBOT_MX_TUNED (perf/mx_tile_sweep.py FAST=1 M=2176: ffn.2 default 208-213 us vs the M2176 table's 243 us).
_M2176_DEFAULT = tuple(tuple(int(v) for v in k.split("x"))
                       for k in os.environ.get("LINGBOT_MX_M2176_DEFAULT", "").replace(":", ",").split(",") if k)


def mm(x, w_data, w_scales):  # noqa: F811 -- tuned dispatch over the default kernel
    if (_NSHARD or _NSHARD_M) and LNC == 2:
        _kk = (int(w_data.shape[0]) * 4, int(w_data.shape[1]) // 4)
        if _kk in _NSHARD or _kk + (int(x.shape[0]),) in _NSHARD_M:
            return _mm_nshard(x, w_data, w_scales)      # LINGBOT_MX_NSHARD
    if _TUNED_ON and int(x.shape[0]) in _TUNED_BY_M:
        tbl = _TUNED_BY_M[int(x.shape[0])]
        key = (int(w_data.shape[0]) * 4, int(w_data.shape[1]) // 4)
        if int(x.shape[0]) == 2176 and key in _M2176_DEFAULT:
            return _mm_default(x, w_data, w_scales)
        if key in tbl:
            mk = (int(x.shape[0]),) + key
            if mk not in _TUNED_MM:
                _TUNED_MM[mk] = _tuned_kernel(key, tbl)
            return _TUNED_MM[mk](x, w_data, w_scales)
    if _TUNED_ON:
        key = (int(w_data.shape[0]) * 4, int(w_data.shape[1]) // 4)
        # tuned M blocks (bm x 128 rows) need at least that many rows: smaller calls (the load-time
        # self-check runs M=256) take the default kernel
        if key in _TUNED and int(x.shape[0]) >= _TUNED[key]["bm"] * 128:
            if key not in _TUNED_MM:
                _TUNED_MM[key] = _tuned_kernel(key)
            return _TUNED_MM[key](x, w_data, w_scales)
    return _mm_default(x, w_data, w_scales)
