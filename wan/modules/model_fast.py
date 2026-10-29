"""Some of the functions are borrowed from SelfForcing (https://github.com/guandeh17/Self-Forcing)."""
import math
import os
import re
from einops import rearrange

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn as nn
import torch.nn.functional as torch_F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from wan.modules.attention import attention
from wan.modules.model import (
    MAX_LATENT_FRAMES,
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    rope_params,
    sinusoidal_embedding_1d
)
from ..distributed import tensor_parallel as tp
from ..distributed import util as dt_util
from wan.modules.attention import attention, flash_attention
from wan.modules.attention import (
    const_max_ag_attention,
    const_max_ag_available,
    cte_cross_attention_nmajor,
    cte_merge,
    cte_merge_rows,
    cte_rows,
    xla_attention_rows,
    cte_partial,
    cte_prep_q,
    exact_attention,
    exact_attention_chunks,
    fa2_ag_attention,
    fa2_ag_available,
)
from wan.modules.chunked_kv_cache import chunk_routing, ordered_chunks_by_block

# HYBRID PER-LAYER ATTENTION -- the reference implementation PR #90, the half of const-max we never ported.
#
# const_max_ag subtracts an ASSUMED constant instead of the true row max. That is exact
# for a "cold" head, whose scores are small and spread out. It is catastrophic for a
# "hot" head, which selects one token: its real row max can exceed 500 while our assumed
# max is 15, and exp(500 - 15) overflows to inf, so that head's output is garbage.
#
# The const-max design note measured the reference model: of 1600 heads (40 layers x 40 heads) only 9
# are hot, all inside layers 33, 34, 38, 39, with a clean gap (9th hottest row max 124.8,
# next 59.7). Their fix is per-LAYER, not per-head, because under TP every rank must run
# the same kernel: if any head in a layer is hot, the whole layer falls back to exact
# attention. That is the FLASH_ATTN_LAYERS=33,34,38,39 in their launcher.
#
# We ran const-max on all 40 layers, which is why the video is coherent but degraded --
# 36 layers are fine and a handful corrupt their output.
#
# 33,34,38,39 IS THE REFERENCE IMPLEMENTATION'S SET, measured on the reference implementation's weights. lingbot is a different
# model, so there is no default here: absent FLASH_ATTN_LAYERS, every layer uses
# const-max, which reproduces today's behaviour rather than silently applying someone
# else's layer indices. scripts/analysis/qk_gamma_temperature.py predicts lingbot's own
# set from its QK-RMSNorm gammas.
# NON-NUMERIC TOKENS ARE IGNORED, not fatal. The manifest sets FLASH_ATTN_LAYERS=auto
# and replaces it with the discovered list before launching; if that discovery step is
# skipped or fails, the literal "auto" would otherwise reach int() and take the whole
# model down at import for what should degrade to "no hot layers".
_FLASH_ATTN_LAYERS = frozenset(
    int(t) for t in os.environ.get("FLASH_ATTN_LAYERS", "").replace(" ", "").split(",")
    if t and t.lstrip("-").isdigit()
)


def flash_attn_layers():
    """Layer indices forced onto exact attention instead of const-max."""
    return _FLASH_ATTN_LAYERS


def layer_uses_const_max(layer_index):
    """False for a hot layer, which must not use the constant-max kernel."""
    return int(layer_index) not in _FLASH_ATTN_LAYERS


# PER-RANK TOKEN PADDING. Measured (scripts/profiling/bench_prologue_variants.py, one core, compiled):
# the block's fp32 norm/modulate + q/k/v + RMSNorm prologue costs 1.01 ms at 540 tokens (cp8) but
# 0.20 ms at 640 -- any fp32 elementwise work on a token count that is not a multiple of 128 falls
# off a lowering cliff (the device trace shows LNC2 cross-core DMA-accumulate of fp32 partials).
# So the residual stream is padded to a multiple of LINGBOT_TOKEN_PAD inside the block stack:
# Q / cross-attn / FFN / camera run on the padded rows, K/V are sliced back to the real tokens
# before the cache write and the attention kernels, and the head sees the real tokens only.
# Padded rows never reach a real row (attention keys exclude them; everything else is per-row).
# 0 disables.
_TOKEN_PAD = int(os.environ.get("LINGBOT_TOKEN_PAD", "128"))
# LINGBOT_CAM_HOIST=1 (default): compute each block's camera FiLM params once per chunk instead of
# inside every block call (see WanModelFast._cam_films). 0 restores the in-block computation.
_CAM_HOIST = os.environ.get("LINGBOT_CAM_HOIST", "1") == "1"
# LINGBOT_FILM_FULL=1 (with LINGBOT_CAM_HOIST): all-gather each block's column-parallel camera
# FiLM scale/shift once per chunk instead of all-gathering the fp32 residual x in every block of
# every forward. Exact (the same elementwise update); costs 2 x [L, dim] bf16 per block of HBM.
_FILM_FULL = os.environ.get("LINGBOT_FILM_FULL", "0") == "1"
# LINGBOT_SP=1: Megatron sequence parallelism inside the TP group (chunked-cache path only): the
# residual stream is token-sharded across the TP ranks between the blocks; see
# CausalWanAttentionBlock.forward. The camera FiLM params become [1, S, C] token shards.
_SP_ON = os.environ.get("LINGBOT_SP", "0") == "1"


def sp_token_shard(x, tp_size, tp_rank):
    """[1, Lp, C] -> this TP rank's contiguous token shard [1, Lp/tp, C]."""
    s = x.shape[1] // tp_size
    assert s * tp_size == x.shape[1], f"padded tokens {x.shape[1]} not divisible by tp {tp_size}"
    return x[:, tp_rank * s:(tp_rank + 1) * s].contiguous()


def film_full_width(scale_local, shift_local):
    """Column-parallel FiLM params -> full feature width (rank order = feature order)."""
    return (tp.all_gather_last_dim(scale_local.contiguous(), _TP_GROUP),
            tp.all_gather_last_dim(shift_local.contiguous(), _TP_GROUP))
# LINGBOT_KV_WRITE_LATE=1 (with LINGBOT_CTE_MERGE): see the chunked-cache branch of
# CausalWanSelfAttention.forward.
_KV_WRITE_LATE = os.environ.get("LINGBOT_KV_WRITE_LATE", "0") == "1"
# LINGBOT_ATTN_EARLY=1 (with LINGBOT_CTE_MERGE, CP over a DeviceMesh): in the chunked-cache branch,
# run the attention calls on the LEADING (cached) chunks first -- they need only Q -- then build the
# new chunk's cache entry from the local shard: transpose the 270-token shard into the cache layout
# and all-gather that, so the gathered chunk needs only large contiguous copies instead of an
# element-wise L<->d transpose of the whole 4320-token chunk. Data movement only and the same merge
# order, but the compiled graph fuses differently and the result is NOT bit-exact: block output rel
# RMS 1.6e-4 vs the default path (kiro/block_bench.py, seeded cache).
# =2 issues the new chunk's all-gathers BEFORE the leading calls (so the collectives can run under
# them), =1 after them.
_ATTN_EARLY = int(os.environ.get("LINGBOT_ATTN_EARLY", "0"))
# LINGBOT_ATTN_PAIR=1 (with LINGBOT_ATTN_EARLY): two KV chunks per attention_cte call, the first as
# the kernel's prefix-caching prior -- 2 calls per block at steady state instead of 4. Exact
# attention either way (no approximation), but the kernel rounds the probabilities to bf16 against
# a different running max: block output rel RMS 1.3e-3 vs the per-chunk path.
_ATTN_PAIR = os.environ.get("LINGBOT_ATTN_PAIR", "0") == "1"
# LINGBOT_KV_CLONE=1 (A/B, with LINGBOT_ATTN_EARLY): pass attention_cte in-graph copies of the cached
# chunks instead of the cache buffers themselves. Profiled, the calls that read cache buffers (graph
# inputs) spent ~160 us of GpSimd on DMA descriptors, the call on the in-graph new chunk 13 us.
_KV_CLONE = os.environ.get("LINGBOT_KV_CLONE", "0") == "1"
# LINGBOT_ATTN_REM=1 (with LINGBOT_ATTN_EARLY): the attention kernel works in 128-row query tiles and
# its cost is per tile, so at 270 real (288 padded) tokens per rank the third tile does 14 rows of
# work at the price of 128. Run the first floor(Lq / 128) * 128 queries through the kernel and the
# remaining rows as fp32 matmuls + softmax over all KV chunks (xla_attention_rows).
_ATTN_REM = os.environ.get("LINGBOT_ATTN_REM", "0") == "1"
# LINGBOT_FILM_BATCH=1 (with LINGBOT_FILM_A2A): all 40 blocks' camera FiLM params in ONE compiled
# graph per chunk instead of 40 (22 ms per chunk measured, mostly per-graph launch cost).
_FILM_BATCH = os.environ.get("LINGBOT_FILM_BATCH", "0") == "1"
# LINGBOT_HEAD_COMPILE=1 (SP + flat CP over a mesh): the head, its TP token gather, the real-token
# slice and the CP gather as ONE compiled graph that returns the [tokens, 64] patch rows (2-D, rank
# order = token order, so no permute after the gather); unpatchify stays eager. The eager tail was
# ~15 small ops per forward (5.5 ms synced). A version that also unpatchified inside the graph failed
# in neuronx-cc (NCC_IBIR158 on the gather's output access pattern).
_HEAD_COMPILE = os.environ.get("LINGBOT_HEAD_COMPILE", "0") == "1"
# LINGBOT_KV_GATHER1=1 (with LINGBOT_ATTN_EARLY): gather the new chunk's K and V over CP in ONE
# all-gather of the stacked local shards instead of two. Profiled, the two 11 MB Ring all-gathers ran
# back to back with a ~110 us gap between them. Data movement only.
_KV_GATHER1 = os.environ.get("LINGBOT_KV_GATHER1", "0") == "1"


def _films_multi(raw_blocks, c_cat, K, P):
    """Every block's camera FiLM (scale, shift) for K chunks at once. c_cat is [1, K * P, C]: each
    chunk's P (padded) CP-local camera tokens in a row. One all-to-all per tensor gives every TP rank
    its token shard of EVERY chunk ([K, P / tp, C]); token shards and column order are those of
    cols_to_token_shard applied to each chunk alone."""
    m = tp.mesh_submesh("tp")
    n = m.size()
    out = []
    for b in raw_blocks:
        pair = []
        for t in b.cam_film(c_cat):                                # [1, K * P, C / n]
            c = int(t.shape[-1])
            x = t[0].view(K, n, P // n, c).permute(1, 0, 2, 3).reshape(n * K * (P // n), c)
            y = fc.all_to_all_single(x.contiguous(), None, None, m)    # [n * K * P/n, c]
            y = y.view(n, K, P // n, c).permute(1, 2, 0, 3).reshape(K, P // n, n * c)
            pair.append(y)
        out.append(tuple(pair))
    return out


def _film_check_log(used, ref, k):
    """LINGBOT_FILM_CHECK=1 (diagnostic, first generation only): chunk k's precomputed FiLM params
    (LINGBOT_FILM_ALL) against the ones the per-chunk path (camera branch + _cam_films) builds,
    over all 40 blocks and all ranks. Rank 0 prints elements compared / differing, max |diff| and
    rel. RMS. COLLECTIVE (all_gather_object): every rank reaches it in the same forward."""
    n_el = n_diff = 0
    mx = ss_d = ss_r = 0.0
    shp = None
    for (su, hu), (sr, hr) in zip(used, ref):
        for a, b in ((su, sr), (hu, hr)):
            shp = shp or (tuple(a.shape), tuple(b.shape))
            a = a.detach().cpu().double().reshape(-1)
            b = b.detach().cpu().double().reshape(-1)
            assert a.numel() == b.numel(), (shp, tuple(a.shape), tuple(b.shape))
            d = a - b
            n_el += d.numel()
            n_diff += int((d != 0).sum())
            mx = max(mx, float(d.abs().max()))
            ss_d += float(d.pow(2).sum())
            ss_r += float(b.pow(2).sum())
    st = [n_el, n_diff, mx, ss_d, ss_r, len(used), len(ref)]
    allst = [None] * dist.get_world_size()
    dist.all_gather_object(allst, st)
    if dist.get_rank() == 0:
        n_el = sum(s[0] for s in allst)
        n_diff = sum(s[1] for s in allst)
        mx = max(s[2] for s in allst)
        rel = math.sqrt(sum(s[3] for s in allst) / max(sum(s[4] for s in allst), 1e-30))
        worst = max(range(len(allst)), key=lambda i: allst[i][2])
        print(f"[film check] chunk {k}: blocks {allst[0][5]}/{allst[0][6]} shapes {shp[0]} vs "
              f"{shp[1]}, {n_diff}/{n_el} elements differ, max|d| {mx:.3e} (rank {worst}), "
              f"rel RMS {rel:.3e}", flush=True)


def _head_fused(head, x, e, P, R):
    """_head_sp_cp for a fused forward: only group b's real tokens [P, P + R) of each CP rank."""
    y = tp.all_gather_tokens(head(x, e), "tp")[0][P:P + R]
    return fc.all_gather_tensor(y.contiguous(), 0, tp.mesh_submesh("cp"))


def _head_sp_cp(head, x, e, real):
    y = tp.all_gather_tokens(head(x, e), "tp")[0]          # [L_pad, 64]
    if real is not None:
        y = y[:real]
    return fc.all_gather_tensor(y.contiguous(), 0, tp.mesh_submesh("cp"))

# LINGBOT_FWD_PHASES=1: DIAGNOSTIC. Sync the device at phase boundaries of WanModelFast.forward and
# record per-phase wall time in model._fwd_phase_log (image2video logs it). The syncs serialise
# host and device, so per-forward totals under this flag are not the production numbers.
_FWD_PHASES = os.environ.get("LINGBOT_FWD_PHASES", "0") == "1"


def _phase_mark(model, name):
    if _FWD_PHASES:
        import time as _time
        torch.neuron.synchronize()
        model._ph_cur.append((name, _time.perf_counter()))
# LINGBOT_FILM_A2A=1 (with LINGBOT_SP): build the per-chunk FiLM token shards with an all-to-all
# inside each block's compiled cam_film graph (cam_film_sp) instead of an eager all-gather + slice.
_FILM_A2A = os.environ.get("LINGBOT_FILM_A2A", "0") == "1"
_RESIDUAL_BF16 = os.environ.get("LINGBOT_RESIDUAL_BF16", "0") == "1"
_REAL_TOKENS = [None]   # this forward's real per-rank token count when padded, else None
_ROWMAX_STATS = os.environ.get("LINGBOT_ROWMAX_STATS")   # dir; eager runs only
_ROWMAX = {}   # (layer, global_head) -> dict of running stats
_ROWMAX_ON = [False]   # set per forward: only REAL generation forwards (not prewarm's zero inputs)


def _record_rowmax(attn, q, k_new, kv_chunks, cp_enabled, sp_group, grid):
    """Runtime row-max statistics of the SCALED self-attention scores, per (layer, head),
    over the full KV window this forward attends to (leading chunks + the new chunk,
    CP-gathered). With across-heads QK-RMSNorm (d=5120) the weight-only Cauchy-Schwarz
    bound is too loose, so classify hot/cold from measured row-max. Eager only (it materialises the scores)."""
    import atexit, json
    if cp_enabled:
        cp = sp_group_dist_size(sp_group)
        k_new = _gather_w(k_new, grid[0], grid[1], grid[2] // cp, sp_group, cp)
    k_chunks = kv_chunks[0]
    lead = []
    for kc in k_chunks[:-1]:
        lead.append(kc.permute(2, 0, 1).unsqueeze(0) if kc.dim() == 3 else kc)  # N-major [N,d,S] -> [1,S,N,d]
    k = torch.cat(lead + [k_new], dim=1).float()            # [1, Sk, n, d]
    qf = q.float()                                          # [1, Sq, n, d]
    n, d = qf.shape[2], qf.shape[3]
    s = torch.einsum("bqnd,bknd->bnqk", qf, k) * (d ** -0.5)
    rm = s.amax(-1)[0].cpu()                                # [n, Sq]
    tp_r = tp.tp_rank(_TP_GROUP) if getattr(attn, "tp_enabled", False) else 0
    layer = int(getattr(attn, "layer_index", -1))
    for h in range(n):
        r = rm[h]
        key = (layer, tp_r * n + h)
        st = _ROWMAX.setdefault(key, {"max": -1e30, "min": 1e30, "sum": 0.0, "cnt": 0,
                                      "gt60": 0, "gt88": 0, "lt_m65": 0, "sk": 0})
        st["max"] = max(st["max"], float(r.max())); st["min"] = min(st["min"], float(r.min()))
        st["sum"] += float(r.sum()); st["cnt"] += r.numel(); st["sk"] = max(st["sk"], int(k.shape[1]))
        st["gt60"] += int((r > 60).sum()); st["gt88"] += int((r > 88).sum())
        st["lt_m65"] += int((r < -65).sum())
    if not getattr(_record_rowmax, "_registered", False):
        _record_rowmax._registered = True
        def _dump():
            os.makedirs(_ROWMAX_STATS, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            out = {f"{l}:{h}": v for (l, h), v in _ROWMAX.items()}
            json.dump(out, open(os.path.join(_ROWMAX_STATS, f"rowmax_rank{rank}.json"), "w"))
        atexit.register(_dump)


def assign_layer_attention(blocks):
    """Stamp each block's self-attention with its index and its kernel choice.

    Called after the block list is built, because the layer index is a property of
    position in the stack and the constructors do not receive it.
    """
    for i, blk in enumerate(blocks):
        sa = getattr(blk, "self_attn", None)
        if sa is None:
            continue
        sa.layer_index = i
        sa.use_const_max = layer_uses_const_max(i)
    return blocks

def causal_rope_grids(freqs, grid_sizes, device, start_frame=0, dtype=torch.float32,
                      w_shard=None):
    """Memoized _causal_rope_grids_build. The grids depend only on (freqs, grid, start_frame,
    shard, dtype), which repeat for all 5 forwards of a chunk and across GEN passes, while a
    rebuild costs a ~10 ms freqs.cpu() D2H sync plus CPU work and an H2D copy every forward."""
    grid_key = tuple(tuple(int(v) for v in row) for row in
                     (grid_sizes.tolist() if torch.is_tensor(grid_sizes) else grid_sizes))
    key = (id(freqs), tuple(freqs.shape), grid_key, str(device), int(start_frame), dtype,
           None if w_shard is None else tuple(w_shard))
    hit = _ROPE_GRID_CACHE.get(key)
    if hit is not None and hit[0] is freqs:
        return hit[1]
    out = _causal_rope_grids_build(freqs, grid_sizes, device, start_frame, dtype, w_shard)
    _ROPE_GRID_CACHE[key] = (freqs, out)
    return out


_ROPE_GRID_CACHE = {}


def _causal_rope_grids_build(freqs, grid_sizes, device, start_frame=0, dtype=torch.float32,
                             w_shard=None):
    """Precompute per-batch cos/sin rotary grids. Independent of x.

    `grid_sizes` is always the FULL patch grid. `w_shard=(cp, rank)` then slices
    the built grids to this rank's WIDTH range -- build-full-then-slice, never
    build-from-a-sliced-grid, because the width term below is freqs_split[2][:w]:
    handing it a pre-sliced w would give every rank positions 0..w_local and
    silently wrong RoPE. (Ported from the reference implementation neuron_cp.py,
    whose docstring calls out exactly this trap.)
    `dtype` is the grid dtype; angles are computed fp32 on CPU for stability,
    then cast to `dtype` before transfer (bf16 halves the HBM footprint)."""
    c = freqs.size(1)  # half head dim

    # Split frequency dims: (c - 2*(c//3), c//3, c//3) for (frame, height, width)
    s0 = c - 2 * (c // 3)
    s1 = c // 3

    # Move freqs to CPU for grid construction (Neuron can't handle cat of expanded views)
    freqs_cpu = freqs.cpu().float()
    freqs_split = freqs_cpu.split([s0, s1, s1], dim=1)

    cos_grids, sin_grids = [], []
    for (f, h, w) in grid_sizes.tolist():
        # Build on the FULL grid so every positional term is absolute.
        angles_grid = torch.cat([
            freqs_split[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1)
        if w_shard is not None and _CP_FLAT_SEQ:
            # Flat mode: flatten to the frame-major sequence FIRST, then take this
            # rank's contiguous run. Same build-full-then-slice discipline as the
            # width branch -- the positional terms above are absolute, so slicing
            # after the cat keeps every rank's angles correct.
            cp, r = w_shard
            seq_len = f * h * w
            loc = seq_len // cp
            angles_grid = angles_grid.reshape(seq_len, 1, c)[r * loc:(r + 1) * loc]
            seq_len = loc
        else:
            if w_shard is not None:
                cp, r = w_shard
                w_loc = w // cp
                angles_grid = angles_grid[:, :, r * w_loc:(r + 1) * w_loc]
                w = w_loc
            seq_len = f * h * w
            angles_grid = angles_grid.reshape(seq_len, 1, c)

        # Compute cos/sin on CPU then transfer to device in the target dtype
        cos_grids.append(torch.cos(angles_grid).to(dtype).to(device))
        sin_grids.append(torch.sin(angles_grid).to(dtype).to(device))

    return cos_grids, sin_grids

def causal_rope_apply_rotation(x, grid_sizes, cos_grids, sin_grids, seq_lens=None):
    """Apply precomputed rotary grids to x.

    `grid_sizes` may be a TENSOR or a list of (f, h, w) int tuples. Pass ints
    inside a compiled region: .tolist() on a tensor yields UNBACKED symints, so
    seq_len = f*h*w and the x_rest remainder become symbolic and fullgraph=True
    fails on the cache write downstream with
      Could not guard on data-dependent expression Eq(1508, u10 + u6*u7*u8)
    where u6*u7*u8 is this f*h*w and u10 the remainder. With ints, seq_len is a
    compile-time constant and the produced width is provably 1508.
    """
    n, c = x.size(2), x.size(3) // 2  # num_heads, half_head_dim

    rope_dtype = _rope_dtype(x)
    output = []
    _grids = grid_sizes.tolist() if torch.is_tensor(grid_sizes) else grid_sizes
    for i, (f, h, w) in enumerate(_grids):
        # seq_lens overrides f*h*w for flat CP sharding, where this rank's token
        # count is (f*h*w)/cp and need not factor back into a (f, h, w) grid.
        # Still a Python int, so the fullgraph=True constant-folding above holds.
        seq_len = f * h * w if seq_lens is None else seq_lens[i]
        cos_grid, sin_grid = cos_grids[i], sin_grids[i]

        # Extract x for this sample using slice (not select — unsupported on Neuron)
        x_seq = x[i:i+1, :seq_len].to(rope_dtype)  # [1, seq_len, N, D]

        # Reshape to pairs: [1, seq_len, N, c, 2]
        x_pairs = x_seq.reshape(1, seq_len, n, c, 2)
        # Use slice 0:1/1:2 instead of [..., 0] select (unsupported on Neuron)
        x_re = x_pairs[:, :, :, :, 0:1].reshape(1, seq_len, n, c)
        x_im = x_pairs[:, :, :, :, 1:2].reshape(1, seq_len, n, c)

        # Apply rotation: (x_re + i*x_im) * (cos + i*sin)
        out_re = x_re * cos_grid - x_im * sin_grid
        out_im = x_re * sin_grid + x_im * cos_grid

        # Interleave with unsqueeze+cat (not stack — unsupported on Neuron)
        x_out = torch.cat([out_re.unsqueeze(-1), out_im.unsqueeze(-1)], dim=-1)
        x_out = x_out.reshape(1, seq_len, n, c * 2)  # [1, seq_len, N, D]

        # Concat with remaining tokens beyond seq_len
        x_rest = x[i:i+1, seq_len:].to(rope_dtype)
        x_i = torch.cat([x_out, x_rest], dim=1)  # [1, L, N, D]
        output.append(x_i)

    result = torch.cat(output, dim=0)  # [B, L, N, D]
    return result.to(x.dtype)


def _causal_rope_dtype(attn):
    """bf16 RoPE at large grids to cut the fp32 transient peak that OOMs the
    DiT; fp32 elsewhere to preserve the existing numeric path."""
    return getattr(attn, "_bf16_rope_flag", torch.float32)

# Default RoPE temporary dtype. bf16 at large grids cuts the fp32 transient
# peak that OOMs the DiT (960x1664, ~3 GB over the 38 GB logical-core HBM at
# 6240+ tokens/chunk); fp32 elsewhere preserves the existing numeric path.
# overrides for A/B. Set per-run once per process from the model.
ROPE_DTYPE = torch.float32


def _rope_dtype(x):
    """RoPE temporary dtype: bf16 when the caller flagged a large frame
    (set via mark_bf16_rope), else the process default."""
    return ROPE_DTYPE


def mark_bf16_rope():
    """Enable bf16 RoPE temporaries for the rest of this process (called by the
    model when seq_len implies a large grid)."""
    global ROPE_DTYPE
    ROPE_DTYPE = torch.bfloat16


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, w_shard=None):
    """`grid_sizes` is the FULL grid; w_shard=(cp, rank) selects this rank's width
    slice. The rotation then runs against the LOCAL grid, since x is already
    width-sharded."""
    rd = _rope_dtype(x)
    cos_grids, sin_grids = causal_rope_grids(freqs, grid_sizes, x.device, start_frame,
                                             dtype=rd, w_shard=w_shard)
    local_grid, seq_lens = grid_sizes, None
    if w_shard is not None:
        cp, _ = w_shard
        if _CP_FLAT_SEQ:
            # Grid stays FULL: the local length is (f*h*w)/cp, which generally has
            # no (f, h, w/cp) form. Pass it explicitly instead of faking a grid.
            seq_lens = [(f * h * w) // cp
                        for (f, h, w) in (grid_sizes.tolist()
                                          if torch.is_tensor(grid_sizes)
                                          else grid_sizes)]
        else:
            local_grid = grid_sizes.clone()
            local_grid[:, 2] = local_grid[:, 2] // cp
    return causal_rope_apply_rotation(x, local_grid, cos_grids, sin_grids,
                                      seq_lens=seq_lens)


# ── frame-major width-axis shard / gather (the reference implementation neuron_cp.py) ──
# Tokens are frame-major L = f*h*w. Slicing the WIDTH axis inside a reshaped
# view keeps every frame and every row on every rank, so per-frame modulation
# and the [f, h, w] RoPE structure stay valid with w_local. A flat contiguous
# cut of L would split frames across ranks instead.
#
# Width, not height: divisibility is what decides feasibility, and grid_w
# factorises where grid_h often does not. For examples/03/image.jpg at 480*832
# the grid is 29 x 52 -- height 29 is PRIME (no CP degree legal), width 52 =
# 2^2 * 13 (cp in {2, 4}).

# FLAT-SEQUENCE CP SHARDING (the reference implementation item 4), opt-in via
# LINGBOT_CP_FLAT_SEQ=1. Width sharding requires grid_w % cp == 0; flat sharding
# requires only (f*h*w) % cp == 0, which is a far weaker condition -- the reference implementation:
# "Removes the ppw divisibility constraint -- any total token count divisible by
# CP works. Gather/scatter simplified to concatenation along the sequence axis."
#
# Safe for the KV path either way: attention sums over key positions, so any
# permutation of K/V along the sequence axis is invisible in the output provided K
# and V are permuted identically and RoPE is already applied (it is -- the shard
# feeds the kernel as `roped_key`). The fused const_max_ag kernel gathers into a
# FLAT (n_heads, kv_chunk, d) buffer, so it needs no change for either layout.
# What DOES depend on the layout is the hidden-state gather before norm_out, which
# must restore frame-major order -- handled by _gather_w's matching branch.
_CP_FLAT_SEQ = os.environ.get("LINGBOT_CP_FLAT_SEQ", "0") == "1"


def _shard_w(x, f, h, w, cp, r):
    """[B, f*h*w, *tail] -> this rank's shard.

    Width mode: [B, f*h*(w/cp), *tail], rank r owning width columns
    [r*w/cp, (r+1)*w/cp). Flat mode: [B, (f*h*w)/cp, *tail], rank r owning a
    contiguous run of the frame-major sequence.
    """
    if cp == 1:
        return x
    if _CP_FLAT_SEQ:
        seq = f * h * w
        assert seq % cp == 0, (
            f"token count {seq} (= {f}*{h}*{w}) not divisible by cp {cp}")
        loc = seq // cp
        return x[:, r * loc:(r + 1) * loc].contiguous()
    assert w % cp == 0, f"patch-grid width {w} not divisible by cp {cp}"
    w_loc = w // cp
    tail = x.shape[2:]
    x = x.view(x.shape[0], f, h, w, *tail)
    x = x[:, :, :, r * w_loc:(r + 1) * w_loc]
    return x.reshape(x.shape[0], f * h * w_loc, *tail).contiguous()


def _gather_w(x, f, h, w_loc, group, cp):
    """Inverse of _shard_w: all_gather the width axis, restore frame-major.

    Uses the traceable functional gather when a DeviceMesh exists, so this can run
    inside a fullgraph=True compiled block: dist.all_gather fills a python list of
    output tensors in place and Dynamo cannot trace it -- the same class of failure
    as async_op=True. fc.all_gather_tensor concatenates along the width axis
    directly, which is also one op instead of a list plus a cat.
    """
    if cp == 1:
        return x
    if _CP_FLAT_SEQ:
        # Rank order IS token order, so concatenating along the sequence axis
        # restores frame-major directly. f/h/w_loc are unused here -- the local
        # length comes from x itself, which is the point: it need not factor as
        # (f, h, w/cp). Callers still pass w // cp, which is meaningless in this
        # mode and deliberately ignored.
        xl = x.contiguous()
        if tp.has_mesh():
            return tp.all_gather_fn(xl, 1, "cp").contiguous()
        parts = [torch.empty_like(xl) for _ in range(cp)]
        dist.all_gather(parts, xl, group=group)
        return torch.cat(parts, dim=1).contiguous()
    tail = x.shape[2:]
    xl = x.view(x.shape[0], f, h, w_loc, *tail).contiguous()
    if tp.has_mesh():
        full = tp.all_gather_fn(xl, 3, "cp")
        return full.reshape(full.shape[0], f * h * (w_loc * cp), *tail).contiguous()
    parts = [torch.empty_like(xl) for _ in range(cp)]
    dist.all_gather(parts, xl, group=group)
    full = torch.cat(parts, dim=3)
    return full.reshape(full.shape[0], f * h * (w_loc * cp), *tail).contiguous()


def set_bf16_rope_for_forward(seq_len):
    """Per-forward: flag bf16 RoPE for large grids so the peak transients use
    half the HBM. Called once per model.forward from WanModelFast.forward."""
    if seq_len >= 4096:
        mark_bf16_rope()


def _tp_row_parallel(linear, x, scatter=False):
    """Row-parallel Linear + all-reduce.

    The matmul dtype is an A/B knob. Default (fp32, historical behaviour) wraps
    the Linear at the weight dtype rather than in an fp32 autocast (an A/B that ran it in the
    activation's own dtype instead, leaving the cross-rank sum in fp32 --
    tp.all_reduce promotes bf16 inputs to fp32 for the reduction either way, so
    the accumulation-error argument for fp32 (stacked per-layer all-reduces) is
    unaffected by this knob; only the matmul's input precision changes.

    Measured at 960*1664: the row-parallel Linears are a large share of DiT device time
    (self_attn.o / cross_attn.o / ffn.2 / cam_injector_layer2) and run at MFU 18.9-36.1%
    with the tensor engine 94-99% busy. That was originally read as paying fp32 matmul
    rates on bf16-stored weights, with a per-forward weight upcast on top. That reading
    is WRONG -- see the no-op note in the body -- so low MFU at a busy tensor engine
    still needs an explanation, and dtype is not it.

    This costs bit-exactness. Note the previous wording here cited "CLAUDE.md's
    max_diff < 0.05 bound", which does not exist -- CLAUDE.md says the opposite, that no
    tolerance has been agreed and one must be agreed before any comparison is relied on.
    """
    orig_dtype = x.dtype
    # THIS AUTOCAST IS A NO-OP, and the docstring above is wrong about why it matters.
    #
    # torch.amp.autocast(dtype=torch.float32) is DISABLED by PyTorch itself -- it warns
    # "In CPU autocast, but the target dtype is not supported. Disabling autocast" --
    # because autocast exists to DOWNCAST and fp32 is not a valid target. So linear(x)
    # has always run at x's dtype, never fp32. And it cannot have been fp32 x against a
    # bf16 weight, because that raises "expected m1 and m2 to have the same dtype". The
    # matmul was already bf16.
    #
    # Proof, from trying to "fix" it: replacing this with linear(x.to(weight.dtype))
    # changed the output by NOTHING (mean_entropy 7.28, mean_frame_diff 29.24,
    # max_frame_diff 56.81 vs 56.80) and the time by nothing (GEN 32.41 -> 32.46 s,
    # per-forward 179.7 -> 179.8 ms). A real dtype change to the largest GEMMs cannot be
    # invisible in both.
    #
    # Left as-is only because removing it is a no-op either way. The docstring's claim
    # about "paying fp32 matmul rates on bf16-stored weights" and the per-forward weight
    # upcast does not survive this: there was no upcast.
    with torch.amp.autocast('neuron', dtype=torch.float32):
        y = linear(x)
    if scatter:
        # Sequence parallel (LINGBOT_SP): reduce-scatter the partial sums to this rank's
        # token shard instead of all-reducing the full sequence.
        y = tp.reduce_scatter_tokens(y, "tp")
    elif tp.has_mesh():
        # Traceable, OUT-OF-PLACE reduce over the DeviceMesh submesh, so this
        # Linear can sit inside a fullgraph=True compiled block. dist.all_reduce
        # below is in-place and untraceable, which is what forced the DiT eager.
        y = tp.all_reduce_fn(y, "tp")
    else:
        y = tp.all_reduce(y, _TP_GROUP)
    return y.to(orig_dtype)


@torch.compile(backend="neuron", dynamic=False, fullgraph=True)
def _tp_rms_norm(x, weight, eps, full_dim):
    """RMSNorm that's correct when x is a column-parallel (head-sharded)
    slice: the local sum-of-squares is all-reduced across ranks before
    finishing the formula, so every rank divides by the true full-width
    element count (full_dim) instead of just its own local shard's width."""
    x_f = x.float()
    local_ss = x_f.pow(2).sum(dim=-1, keepdim=True)
    local_ss = (tp.all_reduce_fn(local_ss, "tp") if tp.has_mesh()
                else tp.all_reduce(local_ss, _TP_GROUP))
    rms = torch.rsqrt(local_ss / full_dim + eps)
    return (x_f * rms).type_as(x) * weight


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Overwritten by assign_layer_attention() once the stack exists. Defaulting to
        # True keeps a module built outside a stack (tests, the profiling harness)
        # behaving as it did before per-layer dispatch.
        self.layer_index = None
        self.use_const_max = True

    def _const_max_here(self):
        """Does THIS layer use the constant-max kernel?

        Both conditions: the kernel has to be loaded at all, and this layer must not be
        one of the hot ones. Read via getattr so a module that predates
        assign_layer_attention() still works.
        """
        return const_max_ag_available() and getattr(self, "use_const_max", True)

    def _fa2_here(self):
        """Hot layer served by the fused AG FlashAttention-2 kernel (LINGBOT_HOT_ATTN=fa2_ag)."""
        return (not self._const_max_here()) and fa2_ag_available()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        kv_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        seq_lens_int=None,
        _async_out=False,
        sp_start_row=0,
        sp_rows=None,
        rope_grids=None,
        rope_local_grid=None,
        kv_write_start=None,
        kv_write_end=None,
        kv_roll_dst=0,
        kv_roll_src=0,
        kv_roll_len=0,
        kv_current_end=None,
        kv_chunks=None,
        return_cache=False,
        grid_ints=None,
        _sp_scatter=False,
        fused=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            frame_seqlen(int, optional): Pre-computed H*W/(patch_h*patch_w). If
                provided, skips a `.item()` sync on `grid_sizes`.
            seq_lens_int(int, optional): Accepted for signature parity with the
                sequence_parallel module; the in-model SP path here doesn't use
                it directly (it re-derives everything from grid_sizes/KV cache).
        """
        sp_group = _get_dit_sp_group()
        sp_enabled = sp_group is not None  # SP via group registration (sp x tp mesh)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            x = x.to(self.q.weight.dtype)
            if getattr(self, 'tp_enabled', False) and self.qk_norm:
                q = _tp_rms_norm(self.q(x), self.norm_q.weight, self.eps, self.dim).view(b, s, n, d)
                k = _tp_rms_norm(self.k(x), self.norm_k.weight, self.eps, self.dim).view(b, s, n, d)
            else:
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if frame_seqlen is None:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        # current_start_frame is computed LAZILY, in the fallback RoPE branch only.
        # Evaluating it here specialised the compiled block on the chunk position:
        #   2/16: current_start == 24128
        #   # current_start_frame = current_start // frame_seqlen  ...:357
        # The compiled path uses the grids precomputed in WanModelFast.forward and
        # never reads it, so the division was pure specialisation for nothing.

        cp_enabled = sp_enabled and _SEQ_MODE == "cp"
        cp_w_shard = None
        if cp_enabled:
            # Token-axis CP on the WIDTH axis: no all_to_all. q/k/v keep
            # num_heads/tp heads and this rank's width slice. RoPE is built on
            # the FULL grid and then sliced to that range (w_shard), never built
            # from a pre-sliced grid.
            cp_w_shard = (sp_group_dist_size(sp_group), _sp_local_r())
            rope_grid = grid_sizes
        elif sp_enabled:
            # Spatial (height-row) Ulysses: x was height-row-sharded, so after
            # the all_to_all this rank holds the full height-row slice
            # [g_f, n_rows, w] (n_rows = h/sp). Frame and width RoPE coords are
            # exact on every rank; height uses rank-local rows [0, n_rows)
            # (the same offset in q and K-cache across chunks cancels in
            # attention, keeping relative positions exact -- see module note).
            q = dt_util.all_to_all(q, 2, 1, group=sp_group)
            k = dt_util.all_to_all(k, 2, 1, group=sp_group)
            v = dt_util.all_to_all(v, 2, 1, group=sp_group)
            n = q.shape[2]
            _g = grid_sizes[0].tolist()
            sp_sz = sp_group_dist_size(sp_group)
            rope_grid = _sp_rope_grid(grid_sizes[0], sp_sz, grid_sizes.dtype,
                                      grid_sizes.device)
        else:
            rope_grid = grid_sizes

        if rope_grids is not None and rope_grids[0] is not None:
            # Precomputed by WanModelFast.forward. Pure tensor math from here, so
            # this traces under fullgraph=True; building the grids in-block does
            # not (data-dependent start_frame -> unbacked symint).
            _cos, _sin = rope_grids
            _lg = rope_local_grid if rope_local_grid is not None else rope_grid
            roped_query = causal_rope_apply_rotation(q, _lg, _cos, _sin).type_as(v)
            roped_key = causal_rope_apply_rotation(k, _lg, _cos, _sin).type_as(v)
        else:
            _csf = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, rope_grid, freqs, start_frame=_csf,
                                            w_shard=cp_w_shard).type_as(v)
            roped_key = causal_rope_apply(k, rope_grid, freqs, start_frame=_csf,
                                          w_shard=cp_w_shard).type_as(v)

        if _REAL_TOKENS[0] is not None and roped_key.shape[1] > _REAL_TOKENS[0]:
            roped_key = roped_key[:, :_REAL_TOKENS[0]]
            v = v[:, :_REAL_TOKENS[0]]
        if cp_enabled:
            # Gather K/V to the FULL chunk across CP, on the WIDTH axis inside a
            # reshaped [B, f, h, w_loc, ...] view. A flat cat on dim 1 would
            # splice frames/rows out of order -- which is why chunk_size>1 no
            # longer needs a guard: the reshape keeps every frame intact.
            # INT grid, not grid_sizes.tolist(): a tensor read yields unbacked
            # symints and the gather's reshape width becomes unguardable inside a
            # compiled block. grid_ints comes from WanModelFast.forward.
            _cp, _ = cp_w_shard
            if grid_ints is not None:
                _g = list(grid_ints[0])
            else:
                _g = [int(v_) for v_ in grid_sizes[0].tolist()]
            # Keep this rank's pre-gather shards: the fused const-max AG kernel
            # all-gathers internally and needs the shard, not the gathered chunk. The
            # eager path below still uses the gathered tensors.
            _cp_shard_k, _cp_shard_v = roped_key, v
            # PER-LAYER, not global: a hot layer takes the exact-attention path and so
            # still needs the CP gather done here, even though const_max_ag is loaded and
            # the other 36 layers skip it.
            _early = fused is not None or (
                _ATTN_EARLY and kv_chunks is not None and kv_chunks[0][-1].dim() == 3
                and _CP_FLAT_SEQ and tp.has_mesh())
            if not self._const_max_here() and not self._fa2_here() and not _early:
                roped_key = _gather_w(roped_key, _g[0], _g[1], _g[2] // _cp, sp_group, _cp)
                v = _gather_w(v, _g[0], _g[1], _g[2] // _cp, sp_group, _cp)

        if not cp_enabled:
            _cp_shard_k, _cp_shard_v = roped_key, v
        if kv_chunks is not None and _ROWMAX_STATS and _ROWMAX_ON[0] and not torch.compiler.is_compiling():
            _record_rowmax(self, roped_query, _cp_shard_k, kv_chunks, cp_enabled, sp_group,
                           _g if cp_enabled else None)
        # CHUNKED CACHE (the reference implementation (kv_cache.py)). No offset of any kind: the
        # destination is a whole tensor and the write lands at its own index 0, so
        # nothing position-dependent enters the graph. wan/modules/chunked_kv_cache.py
        # documents why that matters -- the flat path below reuses one NEFF across a
        # moving write offset and pays an I/O rebind on essentially every call.
        #
        # This branch also sidesteps the SP/CP refusal further down, because there is
        # no rolling window left to gather disjointly: eviction IS the round-robin
        # choice of destination chunk, made outside the graph by chunk_routing().
        if kv_chunks is not None and fused is not None:
            # LINGBOT_FUSE_COMMIT: this forward is the PREVIOUS chunk's KV-cache commit (group a,
            # t=0, the first P tokens of this CP rank) and the CURRENT chunk's first denoise step
            # (group b, the next P), see WanModelFast.forward_fused. kv_chunks is
            # (common cached chunks..., dest_a, dest_b). Group a attends to the common chunks,
            # its own fresh chunk and -- if a_reads_b_old -- the chunk that dest_b still holds and
            # group b now evicts; group b attends to the common chunks, fresh a and fresh b.
            # Written: fresh a into dest_a before any read of it, fresh b into dest_b after the
            # last read of its old content.
            k_chunks, v_chunks = kv_chunks
            P, R, n_common, a_reads_b_old = fused
            _cp = cp_w_shard[0]
            _n, _d = int(roped_key.shape[2]), int(roped_key.shape[3])
            kd_a, vd_a, kd_b, vd_b = k_chunks[-2], v_chunks[-2], k_chunks[-1], v_chunks[-1]
            _k_ndl = int(kd_a.shape[1]) == _d
            ka, kb = roped_key[0, :R], roped_key[0, P:P + R]            # [R, n, d] real tokens
            va, vb = v[0, :R], v[0, P:P + R]
            if _k_ndl:
                kl = torch.stack([ka.permute(1, 2, 0), kb.permute(1, 2, 0)])     # [2, n, d, R]
            else:
                kl = torch.stack([ka.permute(1, 0, 2), kb.permute(1, 0, 2)])     # [2, n, R, d]
            vl = torch.stack([va.permute(1, 0, 2), vb.permute(1, 0, 2)])         # [2, n, R, d]
            kg = tp.all_gather_fn(kl.contiguous(), 0, "cp").view((_cp, 2) + tuple(kl.shape[1:]))
            vg = tp.all_gather_fn(vl.contiguous(), 0, "cp").view((_cp, 2) + tuple(vl.shape[1:]))
            if _k_ndl:
                kga = kg[:, 0].permute(1, 2, 0, 3).reshape(kd_a.shape)
                kgb = kg[:, 1].permute(1, 2, 0, 3).reshape(kd_b.shape)
            else:
                kga = kg[:, 0].permute(1, 0, 2, 3).reshape(kd_a.shape)
                kgb = kg[:, 1].permute(1, 0, 2, 3).reshape(kd_b.shape)
            vga = vg[:, 0].permute(1, 0, 2, 3).reshape(vd_a.shape)
            vgb = vg[:, 1].permute(1, 0, 2, 3).reshape(vd_b.shape)
            kd_a.copy_(kga)
            vd_a.copy_(vga)
            q_all, _ = cte_prep_q(roped_query)                          # both groups' queries
            common = list(zip(k_chunks[:n_common], v_chunks[:n_common]))
            parts_all = []
            while _ATTN_PAIR and len(common) >= 2:
                (k0, v0), (k1, v1) = common.pop(0), common.pop(0)
                parts_all.append(cte_partial(q_all, k1, v1, k0, v0))
            parts_all += [cte_partial(q_all, kc, vc) for kc, vc in common]
            parts_all.append(cte_partial(q_all, kd_a, vd_a))
            parts_a = [cte_rows(p_, 0, P) for p_ in parts_all]
            parts_b = [cte_rows(p_, P, P) for p_ in parts_all]
            if a_reads_b_old:
                q_a, _ = cte_prep_q(roped_query[:, :P])
                parts_a.append(cte_rows(cte_partial(q_a, kd_b, vd_b), 0, P))
            kd_b.copy_(kgb)
            vd_b.copy_(vgb)
            q_b, _ = cte_prep_q(roped_query[:, P:2 * P])
            parts_b.append(cte_rows(cte_partial(q_b, kd_b, vd_b), 0, P))
            x = torch.cat([cte_merge_rows(parts_a), cte_merge_rows(parts_b)], dim=1)
            return self._out_proj(x, _async_out, scatter=_sp_scatter)
        if kv_chunks is not None:
            k_chunks, v_chunks = kv_chunks
            # FUSED all-gather + const-max attention when that backend is loaded. The
            # kernel does the CP gather itself and writes the gathered chunk into
            # k_chunks[-1]/v_chunks[-1] as must-alias outputs, so it replaces the
            # _gather_w above, this copy_, the cat below AND the attention call. It takes
            # THIS RANK'S SHARD, so it must be fed the pre-gather K/V.
            if self._const_max_here() and cp_enabled:
                x, _ka, _va = const_max_ag_attention(
                    roped_query, _cp_shard_k, _cp_shard_v, k_chunks, v_chunks)
                out = self._out_proj(x, _async_out, scatter=_sp_scatter)
                if return_cache and not getattr(self, 'tp_enabled', False):
                    return out, _ka, _va
                return out
            if self._fa2_here() and cp_enabled:
                # Hot layer, exact attention: same fused-AG contract as const-max above, but
                # online softmax (FA2 kernel), so no constant-max assumption.
                x, _ka, _va = fa2_ag_attention(
                    roped_query, _cp_shard_k, _cp_shard_v, k_chunks, v_chunks)
                out = self._out_proj(x, _async_out, scatter=_sp_scatter)
                if return_cache and not getattr(self, 'tp_enabled', False):
                    return out, _ka, _va
                return out
            if (cp_enabled and k_chunks[-1].dim() == 3 and _ATTN_EARLY and _CP_FLAT_SEQ
                    and tp.has_mesh()):
                # LINGBOT_ATTN_EARLY=1: leading chunks first (they need only Q), then the new
                # chunk: local shard -> cache layout -> all-gather on a new leading axis -> one
                # contiguous-run permute into the destination buffer.
                _lq_all = int(roped_query.shape[1])
                _lq_main = (_lq_all // 128) * 128 if (_ATTN_REM and _lq_all % 128) else _lq_all
                q_c, lq = cte_prep_q(roped_query[:, :_lq_main])
                kd, vd = k_chunks[-1], v_chunks[-1]
                _cp = cp_w_shard[0]
                _l, _n, _d = (int(t) for t in roped_key.shape[1:])
                _k_ndl = int(kd.shape[1]) == _d                 # K [n, d, L] (else [n, L, d])

                def _gather_new():
                    if _KV_GATHER1:
                        kl = (roped_key[0].permute(1, 2, 0) if _k_ndl
                              else roped_key[0].permute(1, 0, 2))
                        vl = v[0].permute(1, 0, 2)
                        g = tp.all_gather_fn(torch.stack([kl.reshape(-1), vl.reshape(-1)]), 0,
                                             "cp").view(_cp, 2, -1)
                        if _k_ndl:
                            kg = g[:, 0].view(_cp, _n, _d, _l).permute(1, 2, 0, 3)
                        else:
                            kg = g[:, 0].view(_cp, _n, _l, _d).permute(1, 0, 2, 3)
                        return kg, g[:, 1].view(_cp, _n, _l, _d).permute(1, 0, 2, 3)
                    if _k_ndl:
                        kg = tp.all_gather_fn(roped_key[0].permute(1, 2, 0).contiguous(), 0, "cp")
                        kg = kg.view(_cp, _n, _d, _l).permute(1, 2, 0, 3)
                    else:
                        kg = tp.all_gather_fn(roped_key[0].permute(1, 0, 2).contiguous(), 0, "cp")
                        kg = kg.view(_cp, _n, _l, _d).permute(1, 0, 2, 3)
                    vg = tp.all_gather_fn(v[0].permute(1, 0, 2).contiguous(), 0, "cp")
                    return kg, vg.view(_cp, _n, _l, _d).permute(1, 0, 2, 3)
                if _ATTN_EARLY == 2:
                    kg, vg = _gather_new()
                lead = list(zip(k_chunks[:-1], v_chunks[:-1]))
                if _KV_CLONE:
                    lead = [(kc_ * 1, vc_ * 1) for kc_, vc_ in lead]
                last = lead.pop() if (_ATTN_PAIR and lead) else None
                parts = []
                while _ATTN_PAIR and len(lead) >= 2:
                    (ka, va), (kb, vb) = lead.pop(0), lead.pop(0)
                    parts.append(cte_partial(q_c, kb, vb, ka, va))
                parts += [cte_partial(q_c, kc, vc) for kc, vc in lead]
                if _ATTN_EARLY != 2:
                    kg, vg = _gather_new()
                kd.copy_(kg.reshape(kd.shape))
                vd.copy_(vg.reshape(vd.shape))
                parts.append(cte_partial(q_c, kd, vd, *(last if last is not None else ())))
                x = cte_merge(parts, lq)
                if _lq_main < _lq_all:
                    x_rem = xla_attention_rows(roped_query[:, _lq_main:],
                                               list(k_chunks[:-1]) + [kd], list(v_chunks[:-1]) + [vd])
                    x = torch.cat([x, x_rem], dim=1)
                out = self._out_proj(x, _async_out, scatter=_sp_scatter)
                if return_cache and not getattr(self, 'tp_enabled', False):
                    return out, kd, vd
                return out
            # Destination is LAST in the ordering. A whole-tensor copy_ into a base
            # tensor is what "direct must-alias output binding" needs (kv_cache.py:100);
            # writing a slice of a larger parent is the case that comment rules out.
            if k_chunks[-1].dim() == 3 and _KV_WRITE_LATE:
                # LINGBOT_KV_WRITE_LATE=1 (merged path): attend the new chunk from this
                # forward's own K/V ([n, L, d], tp_k=True) and write the cache AFTER the
                # attention. Before, the K transpose into the cache and the V permute sat on the
                # critical path in front of the first attention call (0.35-0.63 ms of a 3.15 ms
                # block at TP4 x CP16), although only the last chunk's call needs them.
                v_new = v[0].permute(1, 0, 2).contiguous()
                x = exact_attention_chunks(
                    roped_query, list(k_chunks[:-1]) + [roped_key[0].permute(1, 0, 2).contiguous()],
                    list(v_chunks[:-1]) + [v_new])
                if int(k_chunks[-1].shape[1]) == int(roped_key.shape[-1]):
                    k_chunks[-1].copy_(roped_key[0].permute(1, 2, 0))
                else:
                    k_chunks[-1].copy_(roped_key[0].permute(1, 0, 2))
                v_chunks[-1].copy_(v_new)
                if sp_enabled and not cp_enabled:
                    x = dt_util.all_to_all(x, 1, 2, group=sp_group)
                out = self._out_proj(x, _async_out, scatter=_sp_scatter)
                if return_cache and not getattr(self, 'tp_enabled', False):
                    return out, k_chunks[-1], v_chunks[-1]
                return out
            if k_chunks[-1].dim() == 3:
                # N-major chunks (LINGBOT_CTE_MERGE=1, the const-max cache layout: K [n, d, L],
                # V [n, L, d]). The merged cte path reads them in place, so the layout change
                # happens once here, on the new chunk only.
                if int(k_chunks[-1].shape[1]) == int(roped_key.shape[-1]):
                    k_chunks[-1].copy_(roped_key[0].permute(1, 2, 0))
                else:
                    k_chunks[-1].copy_(roped_key[0].permute(1, 0, 2))
                v_chunks[-1].copy_(v[0].permute(1, 0, 2))
            else:
                k_chunks[-1].copy_(roped_key)
                v_chunks[-1].copy_(v)
            # Every token in the concatenation is written, so no attn_mask and no
            # bound_min/bound_max -- the flat cache needed those only to hide its
            # unwritten tail. Length is len(order) x chunk_tokens: it grows while the
            # window fills and is then constant, giving n_chunks graphs in total.
            # EXACT attention: this is the hot-layer path (and the whole path when
            # const_max_ag is not loaded). exact_attention prefers attention_cte over
            # SDPA because SDPA materialises the [B, N, Sq, Sk] scores -- 1.270 GB at
            # 768*1152, which OOM'd on the first denoise forward once 7 layers took it.
            x = exact_attention_chunks(roped_query, k_chunks, v_chunks)
            # PRECEDENCE, mirroring the flat path's "if cp_enabled / elif sp_enabled".
            # cp_enabled is `sp_enabled and _SEQ_MODE == "cp"`, so BOTH are true under
            # CP and a bare `if sp_enabled` fires there -- wrong twice over. Token-axis
            # CP must stay token-sharded (x arrived that way and the rest of the block
            # expects it), and dt_util.all_to_all cannot be traced at all: it calls
            # dist.all_to_all, which builds AllToAllOptions() through pybind, so
            # fullgraph dies with "Attempted to call function marked as skipped ...
            # pybind11_object.__new__". This branch is therefore Ulysses-only and
            # eager-only; scope=block_stack under Ulysses would need a functional
            # all_to_all first.
            if sp_enabled and not cp_enabled:
                x = dt_util.all_to_all(x, 1, 2, group=sp_group)
            out = self._out_proj(x, _async_out, scatter=_sp_scatter)
            if return_cache and not getattr(self, 'tp_enabled', False):
                return out, k_chunks[-1], v_chunks[-1]
            return out

        # kv_current_end when the caller supplies it, so the compiled block never
        # reads current_start. The census showed current_start is the ONLY thing
        # that varies across the whole denoise loop -- x is always
        # (1, 1508, 5120), e0 (1,1,6,5120), ctx (1,512,5120), grid (1,29,52) -- so
        # every extra graph came from this one int. Derived quantities
        # (kv_write_start/end, roll) are already passed precomputed; this was the
        # last use.
        current_end = kv_current_end if kv_current_end is not None else \
            current_start + (roped_key.shape[1] if cp_enabled
                                       else roped_query.shape[1])
        # Spatial SP does not change the per-frame temporal layout (frame=1
        # per rank), so sink stays in units of whole frames.
        sink_tokens = self.sink_size * frame_seqlen
        # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        kv_cache_size = kv_cache["k"].shape[1]
        # Same reason as current_end above: the cache grows by the gathered
        # chunk, not by this rank's Q slice.
        num_new_tokens = roped_key.shape[1] if cp_enabled else roped_query.shape[1]
        if sp_enabled:
            assert self.local_attn_size == -1, (
                "SP path currently supports only the global-KV fast path "
                "(local_attn_size == -1). Rolling/evicting KV under SP needs a "
                "disjoint-window gather that isn't implemented yet; run with "
                "--local_attn_size -1 or disable SP.")
        if self.local_attn_size == -1:
            # Fast path (no eviction possible — cache is global). Both
            # indices start at 0 and advance identically every forward, so
            # local_end_index == current_end and local_start_index ==
            # current_start. All Python ints — no .item() syncs.
            # PLAIN PYTHON INTS when the caller supplies them. Deriving these
            # from current_end/current_start inside a compiled graph makes them
            # symbolic, and fullgraph=True then cannot prove the destination
            # slice matches the source width:
            #   Could not guard on data-dependent expression
            #   Eq(Max(0, Min(9048, u0*u1*u2 + u4)), u10 + u6*u7*u8)
            # 9048 is kv_cache_size (1508 tokens x 6 latent frames), u0*u1*u2 the
            # grid product from the grid_sizes TENSOR, u4 the write offset.
            # Passing ints instead specialises one graph per write position --
            # the reference implementation's the reference flat block stack takes kv_write_start/kv_write_end as
            # ints for exactly this reason (block.py:250-251).
            if kv_write_start is not None:
                local_start_index = int(kv_write_start)
                local_end_index = int(kv_write_end)
            else:
                local_end_index = current_end
                local_start_index = current_start
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        elif kv_write_start is not None:
            # WINDOWED cache, indices supplied as PLAIN INTS by the model forward.
            #
            # The eager branches below read kv_cache["local_end_index"].item() and
            # ["global_end_index"].item() -- four .item() calls that make every
            # index symbolic inside a compiled block. The model forward now does
            # those reads OUTSIDE the graph (it already has current_start as an
            # int) and hands down a complete write plan, so the block does pure
            # tensor ops at constant offsets.
            #
            # This is what bounds the graph count. Until the window fills the write
            # position advances, but ONCE FULL it is constant -- always the last
            # num_new_tokens slots, with a constant roll. So the number of distinct
            # graphs is local_attn_size/chunk_size + 1, independent of clip length,
            # which is the reference implementation's window/chunk bound (setup.py:123
            # "n_chunks = kv.window_size // kv.chunk_size"). With
            # local_attn_size=-1 it was one graph per chunk and grew with the clip:
            # 21 at 81 frames, and the measured cause of GEN 1102.8 s vs 705.4 s.
            if kv_roll_len:
                kv_cache["k"][:, kv_roll_dst:kv_roll_dst + kv_roll_len] = \
                    kv_cache["k"][:, kv_roll_src:kv_roll_src + kv_roll_len].clone()
                kv_cache["v"][:, kv_roll_dst:kv_roll_dst + kv_roll_len] = \
                    kv_cache["v"][:, kv_roll_src:kv_roll_src + kv_roll_len].clone()
            local_start_index = int(kv_write_start)
            local_end_index = int(kv_write_end)
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        elif (current_end > kv_cache["global_end_index"].item()) and (
                num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
            # Calculate the number of new tokens added in this step
            # Shift existing cache content left to discard oldest tokens
            # Clone the source slice to avoid overlapping memory error
            num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            # Insert the new keys/values at the end
            local_end_index = kv_cache["local_end_index"].item() + current_end - \
                kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = roped_key
            kv_cache["v"][:, local_start_index:local_end_index] = v

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        # Fixed-shape read: mask instead of slicing to
        # [local_end_index-max_attention_size:local_end_index] -- that slice
        # grew every chunk and forced a fresh NEFF per shape per block.
        attn_hi = kv_cache["local_end_index"]
        attn_lo = torch.clamp(attn_hi - max_attention_size, min=0)

        positions = torch.arange(kv_cache_size, device=kv_cache["k"].device)
        attn_mask = ((positions >= attn_lo) & (positions < attn_hi)).view(1, 1, 1, kv_cache_size)
        # Same window, handed over as scalars too. The SDPA path consumes
        # attn_mask; attention_cte consumes bound_min/bound_max. The repo's NST
        # kernel consumes NEITHER -- it ignores the mask entirely and attends the
        # whole cache including the unwritten tail.
        _kv_bounds = dict(kv_lo=attn_lo, kv_hi=attn_hi)
        if cp_enabled:
            # Token-axis CP needs NO kernel CP support, because this attention
            # is not token-causal: attn_mask above is a function of KV position
            # ONLY ([1,1,1,kv_cache_size]) and broadcasts over Q, so causality
            # is carried by WHAT IS IN the cache, not by a triangular mask.
            #
            # A shorter Q against the same full K/V is therefore exactly
            # correct and embarrassingly parallel -- no cp_offset, no cross-rank
            # softmax reduction, and no all_to_all in either direction. x comes
            # back still token-sharded, which is the layout the rest of the
            # block already expects, since x arrived that way.
            #
            # attention_cte's cp_offset/global_cp_deg path is for a genuinely
            # causal 1-D sequence and would impose the wrong mask here; it stays
            # available for a later perf step (sliding_window under
            # --local_attn_size), not for correctness.
            x = attention(roped_query, kv_cache["k"], kv_cache["v"], attn_mask=attn_mask,
                          **_kv_bounds)
        elif sp_enabled:
            # SP (height-row sharding): after the Ulysses all_to_all, this rank
            # holds the full THIS-CHUNK height-row slice (frame/width coords
            # exact, height=rank-local rows). Only the CURRENT chunk's K/V are
            # read here and they're RoPE'd with this rank's slice coords and
            # written to this rank's local KV slice, so a per-rank local KV
            # (NO cross-rank KV gather) is exact. Flash-attend over the full
            # local window, then all_to_all back to re-shard the sequence dim
            # for the rest of the block.
            x = attention(roped_query, kv_cache["k"], kv_cache["v"], attn_mask=attn_mask,
                          **_kv_bounds)
            x = dt_util.all_to_all(x, 1, 2, group=sp_group)
        else:
            x = attention(roped_query, kv_cache["k"], kv_cache["v"], attn_mask=attn_mask,
                          **_kv_bounds)

        # output
        out = self._out_proj(x, _async_out)
        if return_cache and not getattr(self, 'tp_enabled', False):
            # the reference implementation, block.py:290-297: with a FLAT cache the writes are ordinary
            # torch scatters, so "every mutated K/V tensor must cross the stack
            # boundary as an explicit output (the framework needs to see the write
            # to alias correctly)". Returning them is not cosmetic -- without it
            # the compiled graph may drop or re-allocate the write.
            return out, kv_cache["k"], kv_cache["v"]
        return out

    def _out_proj(self, x, _async_out=False, scatter=False):
        """Shared o-projection tail for the flat- and chunked-cache paths.

        Lifted out of forward() unchanged so the chunked branch reaches the same
        row-parallel reduce instead of duplicating it. The tp_enabled case returns from
        inside here, which is why the caller's return_cache check excludes it -- that
        was already the behaviour and stays that way.
        """
        x = x.flatten(2)
        if scatter and getattr(self, 'tp_enabled', False):
            # Sequence parallel (LINGBOT_SP): reduce-scatter to this rank's token shard.
            orig_dtype = x.dtype
            with torch.amp.autocast('neuron', dtype=torch.float32):
                y = self.o(x)
            return tp.reduce_scatter_tokens(y, "tp").to(orig_dtype)
        if getattr(self, 'tp_enabled', False):
            # fp32 Linear -> all-reduce -> downcast to the pre-Linear activation
            # dtype, matching _tp_row_parallel exactly. With _async_out (the
            # block's overlap path), return the pending reduce so the block can
            # run the x-independent cam MLP while it's in flight, then wait.
            orig_dtype = x.dtype
            with torch.amp.autocast('neuron', dtype=torch.float32):
                y = self.o(x)
            # async_op=True is not traceable: Dynamo rejects it with
            #   Unsupported: async_op=True for distributed collectives
            # The overlap it buys (running the cam MLP while the reduce is in
            # flight) is an EAGER optimisation and cannot survive inside a graph
            # anyway -- a compiled region has no place to hide the latency. So a
            # compiled block uses the synchronous, traceable mesh reduce; the
            # eager path keeps the async overlap unchanged.
            if tp.has_mesh() and torch.compiler.is_compiling():
                y = tp.all_reduce_fn(y, "tp")
                if _async_out:
                    return tp.AsyncRowOut.already_done(y, orig_dtype)
                return y.to(orig_dtype)
            pend = tp.all_reduce_async(y, _TP_GROUP)
            if _async_out:
                return tp.AsyncRowOut(pend, orig_dtype)
            return pend.wait().to(orig_dtype)
        return self.o(x)


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None,
                cross_attn_first_call=None, _sp_scatter=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            cross_attn_first_call(bool, optional): If provided, used as the
                "first call this generation" gate instead of reading
                crossattn_cache["is_init"].item() (which forces a CPU↔GPU
                sync). Caller (pipeline) tracks this as a Python bool.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        x = x.to(self.q.weight.dtype)
        context = context.to(self.k.weight.dtype)
        use_tp_norm = getattr(self, 'tp_enabled', False) and self.qk_norm
        if use_tp_norm:
            q = _tp_rms_norm(self.q(x), self.norm_q.weight, self.eps, self.dim).view(b, -1, n, d)
        else:
            q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if cross_attn_first_call is None:
                is_first = crossattn_cache["is_init"].item() == 0
            else:
                is_first = cross_attn_first_call
            _nmaj = crossattn_cache["k"].dim() == 3      # LINGBOT_CROSS_NMAJOR layout
            if is_first:
                crossattn_cache["is_init"].fill_(1)
                if use_tp_norm:
                    k = _tp_rms_norm(self.k(context), self.norm_k.weight, self.eps, self.dim).view(b, -1, n, d)
                else:
                    k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                if _nmaj:
                    crossattn_cache["k"].copy_(k[0].permute(1, 2, 0))
                    crossattn_cache["v"].copy_(v[0].permute(1, 0, 2))
                else:
                    crossattn_cache["k"].copy_(k)
                    crossattn_cache["v"].copy_(v)
            if _nmaj:
                x = cte_cross_attention_nmajor(q, crossattn_cache["k"], crossattn_cache["v"])
                x = x.flatten(2)
                if getattr(self, 'tp_enabled', False):
                    return _tp_row_parallel(self.o, x, scatter=_sp_scatter)
                return self.o(x)
            if not is_first:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            if use_tp_norm:
                k = _tp_rms_norm(self.k(context), self.norm_k.weight, self.eps, self.dim).view(b, -1, n, d)
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v, k_lens=context_lens, is_cross_attn=True)

        # output
        x = x.flatten(2)
        if getattr(self, 'tp_enabled', False):
            x = _tp_row_parallel(self.o, x, scatter=_sp_scatter)
        else:
            x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim=dim,
                                                num_heads=num_heads,
                                                local_attn_size=local_attn_size,
                                                sink_size=sink_size,
                                                qk_norm=qk_norm,
                                                eps=eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def cam_film(self, c2ws_plucker_emb):
        """Camera FiLM parameters (cam_scale, cam_shift) of this block. A function of the chunk's
        camera condition and the weights only: no x, no timestep."""
        cam_hidden = torch_F.silu(self.cam_injector_layer1(c2ws_plucker_emb))
        if getattr(self, 'tp_enabled', False):
            c2ws_hidden_states = _tp_row_parallel(self.cam_injector_layer2, cam_hidden)
        else:
            c2ws_hidden_states = self.cam_injector_layer2(cam_hidden)
        c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
        return self.cam_scale_layer(c2ws_hidden_states), self.cam_shift_layer(c2ws_hidden_states)

    def cam_film_sp(self, c2ws_plucker_emb):
        """cam_film() already in the sequence-parallel form the SP block consumes: this rank's
        token shard at full width, via one all-to-all per tensor inside the same graph
        (LINGBOT_FILM_A2A=1) instead of an eager full-width all-gather + slice + clone."""
        scale, shift = self.cam_film(c2ws_plucker_emb)
        return tp.cols_to_token_shard(scale, "tp"), tp.cols_to_token_shard(shift, "tp")

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        seq_lens_int=None,
        sp_start_row=0,
        sp_rows=None,
        rope_grids=None,
        rope_local_grid=None,
        kv_write_start=None,
        kv_write_end=None,
        kv_roll_dst=0,
        kv_roll_src=0,
        kv_roll_len=0,
        kv_current_end=None,
        kv_chunks=None,
        grid_ints=None,
        fused=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        _rdt = x.dtype   # residual-stream dtype (fp32, or bf16 with LINGBOT_RESIDUAL_BF16=1)
        with torch.amp.autocast('neuron', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        if _SP_ON and getattr(self, 'tp_enabled', False) and kv_chunks is not None:
            # MEGATRON SEQUENCE PARALLELISM inside the TP group (LINGBOT_SP=1; Wan2.2 FAST_SP).
            # x is this rank's TOKEN SHARD [1, S, C] of the padded chunk. Each column-parallel
            # projection gets the all-gathered normed input; each row-parallel projection
            # reduce-scatters its partial sums back to the shard (the bytes of the all-reduce it
            # replaces). Norms, modulation, gates, camera FiLM and residual adds run on S tokens
            # instead of all of them, and the per-block fp32 FiLM all-gather of x disappears.
            # Attention, QK-norm, RoPE and the KV write still see all tokens of this rank's heads.
            xs = x
            h = self.norm1(xs).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
            hf = tp.all_gather_tokens(h.to(torch.bfloat16), "tp")
            y = self.self_attn(
                hf, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int, _async_out=False,
                sp_start_row=sp_start_row, sp_rows=sp_rows,
                rope_grids=rope_grids, rope_local_grid=rope_local_grid,
                kv_write_start=kv_write_start, kv_write_end=kv_write_end,
                kv_roll_dst=kv_roll_dst, kv_roll_src=kv_roll_src, kv_roll_len=kv_roll_len,
                kv_current_end=kv_current_end, kv_chunks=kv_chunks,
                grid_ints=grid_ints, _sp_scatter=True, fused=fused)
            with torch.amp.autocast('neuron', dtype=torch.float32):
                xs = (xs + y * e[2].squeeze(2)).to(_rdt)
            if dit_cond_dict is not None and "cam_film" in dit_cond_dict:
                # [1, S, C]: this rank's tokens, full width (built once per chunk by _cam_films).
                cam_scale, cam_shift = dit_cond_dict["cam_film"]
                xs = (1.0 + cam_scale) * xs + cam_shift
            hf = tp.all_gather_tokens(self.norm3(xs).to(torch.bfloat16), "tp")
            xs = xs + self.cross_attn(hf, context, context_lens, crossattn_cache=crossattn_cache,
                                      cross_attn_first_call=cross_attn_first_call,
                                      _sp_scatter=True)
            h = (self.norm2(xs).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(
                self.ffn[0].weight.dtype)
            hf = tp.all_gather_tokens(h, "tp")
            y = _tp_row_parallel(self.ffn[2], self.ffn[1](self.ffn[0](hf)), scatter=True)
            with torch.amp.autocast('neuron', dtype=torch.float32):
                xs = (xs + y * e[5].squeeze(2)).to(_rdt)
            return xs
        # self-attention o-projection row-parallel all-reduce can be issued
        # ASYNC (overlap the x-independent cam-injector MLP while the reduce is
        # in flight). MEASURED on TRN3: net-zero to ~4s SLOWER --
        # NeuronLink-v4's 16 CC-cores already keep the per-block all_reduce
        # cheap, so there is no collective latency left to hide, and the handle/
        # wait lands on the critical path. Bit-exact (latent parity max_diff=0
        # vs eager). OFF: measured net-zero to ~4 s SLOWER on TRN3, and async_op=True
        # does not trace, so it could never survive inside the compiled stack anyway.
        no_ov = True
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
            frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int, _async_out=(not no_ov),
            sp_start_row=sp_start_row, sp_rows=sp_rows,
            rope_grids=rope_grids, rope_local_grid=rope_local_grid,
            kv_write_start=kv_write_start, kv_write_end=kv_write_end,
            kv_roll_dst=kv_roll_dst, kv_roll_src=kv_roll_src, kv_roll_len=kv_roll_len,
            kv_current_end=kv_current_end, kv_chunks=kv_chunks,
            grid_ints=grid_ints)

        # cam injection (only if dit_cond_dict is provided and contains c2ws_plucker_emb)
        if dit_cond_dict is not None and ("cam_film" in dit_cond_dict or "c2ws_plucker_emb" in dit_cond_dict):
            if "cam_film" in dit_cond_dict:
                # Hoisted (LINGBOT_CAM_HOIST): the FiLM params depend only on the chunk's camera,
                # so they were computed once per chunk by cam_film() outside this graph.
                cam_scale, cam_shift = dit_cond_dict["cam_film"]
            else:
                cam_scale, cam_shift = self.cam_film(dit_cond_dict["c2ws_plucker_emb"])
            y = y.wait() if hasattr(y, 'wait') else y
            with torch.amp.autocast('neuron', dtype=torch.float32):
                x = (x + y * e[2].squeeze(2)).to(_rdt)
            if getattr(self, 'tp_enabled', False) and cam_scale.shape[-1] == x.shape[-1]:
                # Full-width FiLM params (LINGBOT_FILM_FULL=1, gathered once per chunk by
                # _cam_films): the same per-feature update on every rank, no collective.
                x = (1.0 + cam_scale) * x + cam_shift
            elif getattr(self, 'tp_enabled', False):
                # cam_scale/cam_shift are column-parallel, so each rank holds
                # only its own output-feature slice. The update is elementwise
                # per feature, so apply it to the matching slice of x (x is
                # full and identical on every rank here -- cam_injector_layer2
                # all-reduced it) and all_gather to restore full width for
                # norm3 / cross-attn, which need the full feature dim.
                #
                # Exact, not approximate: rank r owns features
                # [r*local, (r+1)*local) -- the same offset
                # shard_column_parallel used -- each output element is computed
                # from the full input row, and all_gather concatenates in rank
                # order. Same arithmetic per element as the replicated form.
                local_dim = cam_scale.shape[-1]
                start = tp.tp_rank(_TP_GROUP) * local_dim
                x_local = x[..., start:start + local_dim]
                x_local = (1.0 + cam_scale) * x_local + cam_shift
                x = tp.all_gather_last_dim(x_local, _TP_GROUP)
            else:
                x = (1.0 + cam_scale) * x + cam_shift
        else:
            y = y.wait() if hasattr(y, 'wait') else y
            with torch.amp.autocast('neuron', dtype=torch.float32):
                x = (x + y * e[2].squeeze(2)).to(_rdt)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None,
                           cross_attn_first_call=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens,
                                    crossattn_cache=crossattn_cache,
                                    cross_attn_first_call=cross_attn_first_call)
            ffn_input = (self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(self.ffn[0].weight.dtype)
            if getattr(self, 'tp_enabled', False):
                # Plain fc1 + GELU. The whole block lives inside one compiled stack,
                # so a separately compiled fc1+GELU region bought nothing; it went with
                # the per-leaf compile scopes.
                y = self.ffn[1](self.ffn[0](ffn_input))
                y = _tp_row_parallel(self.ffn[2], y)
            else:
                y = self.ffn(ffn_input)
            with torch.amp.autocast('neuron', dtype=torch.float32):
                x = (x + y * e[5].squeeze(2)).to(_rdt)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache,
                           cross_attn_first_call=cross_attn_first_call)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('neuron', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModelFast(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        control_dim = 6

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.patch_embedding_wancamctrl = nn.Linear(
            control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2], dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        # Per-layer attention choice. Must happen here rather than in the block
        # constructor: which kernel a layer uses depends on its POSITION in the stack.
        assign_layer_attention(self.blocks)

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(MAX_LATENT_FRAMES, d - 4 * (d // 6)),
            rope_params(MAX_LATENT_FRAMES, 2 * (d // 6)),
            rope_params(MAX_LATENT_FRAMES, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        dit_cond_dict=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        max_attention_size=1_000_000,
        frame_seqlen=None,
        cross_attn_first_call=None,
        skip_head=False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            dit_cond_dict (`dict`, *optional*, defaults to None):
                Dictionary of conditioning signals. May contain key ``c2ws_plucker_emb``
                with camera Plucker embeddings of shape [B, C, F, H, W] for camera control.
            kv_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer self-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, kv_size, num_heads, head_dim]), ``global_end_index``,
                and ``local_end_index`` (scalar Tensors tracking cache position).
            crossattn_cache (`list[dict]`, *optional*, defaults to None):
                Per-layer cross-attention KV cache. Each dict contains keys ``k``, ``v``
                (Tensor of shape [B, text_len, num_heads, head_dim]) and ``is_init`` (bool).
            current_start (`int`, *optional*, defaults to 0):
                Token offset of the current chunk in the full sequence. Used to index
                into the KV cache and compute positional embeddings correctly.
            max_attention_size (`int`, *optional*, defaults to 1_000_000):
                Maximum number of KV tokens each query can attend to. Limits the
                effective context window of self-attention to control memory usage.

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert y is not None
        if _FWD_PHASES:
            self._ph_cur = []
            _phase_mark(self, "start")

        _ROWMAX_ON[0] = getattr(self, "_t_host_key", None) is not None
        # Flag bf16 RoPE temporaries when the chunk is large (960x1664),
        # halving the fp32 transient peak that OOMs the DiT. Seed-parity
        set_bf16_rope_for_forward(seq_len)

        # params
        device = getattr(self, '_onload_device', self.patch_embedding.weight.device)
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        # Non-block children are moved once per device, not on every forward (each .to walks
        # every parameter on the host).
        if getattr(self, "_children_on", None) != str(device):
            for _name, _mod in self.named_children():
                if _name != 'blocks':
                    _mod.to(device)
            self._children_on = str(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        sp_group = _get_dit_sp_group()
        sp_size = tp.tp_degree(_SP_GROUP)
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0).to(self.patch_embedding.weight.dtype)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        # time embeddings
        #
        # A 1-D `t` is ONE timestep shared by every token -- t.expand below
        # just replicates it -- so e and e0 are constant along the sequence
        # axis. Materialising that replication is very expensive at high
        # resolution: at 960*1664 (seq_len 6195) e0 is [1, 6195, 6, 5120] fp32
        # = 761 MB carrying 122 KB of distinct values, time_projection computes
        # the same 30720-wide row 6195 times, and each of the 40 blocks then
        # broadcast-adds its modulation across the whole 761 MB
        # (`(self.modulation.unsqueeze(0) + e).chunk(6, dim=2)`). Measured, that
        # per-block graph was the single largest consumer of DiT device time --
        # 24.6% of it, 716 ms per forward, 9.5 GB of HBM traffic per execution
        # of which 7.8 GB was spill, at 0% MFU (all its matmul FLOPs were
        # transposes).
        #
        # So compute e/e0 at sequence length 1 and let broadcasting do the
        # rest. Every consumer is elementwise against x -- the blocks use
        # e[i].squeeze(2) ([1, 1, dim] vs x's [1, seq, dim]) and CausalHead
        # does the same -- so this is numerically identical, not an
        # approximation.
        #
        # A 2-D `t` (genuinely per-token timesteps) keeps the full path.
        if t.dim() == 1:
            t = t.expand(t.size(0), 1)
        e_seq = t.size(1)
        # Host-side cache of e/e0 keyed by the timestep VALUE the caller knows on the host
        # (image2video sets self._t_host_key). Only 5 values occur (4 denoise steps + the
        # cache-commit t=0), and the fp64 sinusoid runs as CPU-fallback ops with D2H/H2D
        # copies every forward. Same ops, same inputs -> identical result.
        _tkey = getattr(self, "_t_host_key", None)
        _tkey = None if _tkey is None else (_tkey, tuple(t.shape), str(device))
        _thit = getattr(self, "_t_emb_cache", {}).get(_tkey) if _tkey is not None else None
        if _thit is not None:
            e, e0 = _thit
        else:
          with torch.amp.autocast('neuron', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, e_seq)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
            e = e.to(device)
            e0 = e0.to(device)
          if _tkey is not None:
            if not hasattr(self, "_t_emb_cache"):
                self._t_emb_cache = {}
            self._t_emb_cache[_tkey] = (e, e0)

        # context: constant for the whole generation, so embed it once per context tensor.
        _ckey = (id(context[0]), tuple(context[0].shape)) if len(context) == 1 else None
        _chit = getattr(self, "_ctx_emb_cache", None)
        if _ckey is not None and _chit is not None and _chit[0] == _ckey and _chit[1] is context[0]:
            context = _chit[2]
        else:
            _ctx_in = context[0] if len(context) == 1 else None
            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            if _ckey is not None:
                self._ctx_emb_cache = (_ckey, _ctx_in, context)
        context_lens = None

        _phase_mark(self, "embed")
        # cam: the condition is per CHUNK (same tuple object for its 5 forwards), so the
        # embedded + sharded result is reused within the chunk.
        _cam_in = None
        self._films_cur = None
        _film_chk = None
        _pre = getattr(self, "_films_pre", None)
        if (_pre and dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict
                and id(dit_cond_dict["c2ws_plucker_emb"]) in _pre
                and _pre[id(dit_cond_dict["c2ws_plucker_emb"])][0] is dit_cond_dict["c2ws_plucker_emb"]):
            # LINGBOT_FILM_ALL: this chunk's FiLM params were computed before the loop.
            self._films_cur = _pre[id(dit_cond_dict["c2ws_plucker_emb"])][1]
            _chk_ids = getattr(self, "_film_check_ids", None)
            if _chk_ids and id(dit_cond_dict["c2ws_plucker_emb"]) in _chk_ids:
                # LINGBOT_FILM_CHECK: keep the condition, so this forward also builds the chunk's
                # FiLM params the per-chunk way (camera branch + _cam_films) to compare.
                _film_chk = _chk_ids.pop(id(dit_cond_dict["c2ws_plucker_emb"]))
            else:
                dit_cond_dict = None
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            _cam_in = dit_cond_dict["c2ws_plucker_emb"]
            _camhit = getattr(self, "_cam_emb_cache", None)
            if _camhit is not None and _camhit[0] is _cam_in:
                dit_cond_dict = dict(dit_cond_dict)
                dit_cond_dict["c2ws_plucker_emb"] = _camhit[1]
                _cam_in = None

        # cam
        if _cam_in is not None:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            c2ws_plucker_emb = [
                rearrange(
                    i,
                    '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                    c1=self.patch_size[0],
                    c2=self.patch_size[1],
                    c3=self.patch_size[2],
                ) for i in c2ws_plucker_emb
            ]
            c2ws_plucker_emb = torch.cat(
                c2ws_plucker_emb, dim=1)  # [1, (L1+...+Ln), C]
            c2ws_plucker_emb = self.patch_embedding_wancamctrl(c2ws_plucker_emb)
            c2ws_hidden_states = self.c2ws_hidden_states_layer2(
                torch_F.silu(self.c2ws_hidden_states_layer1(c2ws_plucker_emb)))
            c2ws_plucker_emb = c2ws_plucker_emb + c2ws_hidden_states
            if sp_size > 1:
                _g = grid_sizes[0].tolist()
                if _SEQ_MODE == "cp":
                    # Shard the camera condition on the SAME axis as x (width),
                    # frame-major. Its spatial size is DERIVED, not assumed: the
                    # old code hardcoded rows = 2*H_patch and width = W_patch,
                    # which cannot both be right (that is 2x the rows against 1x
                    # the width). Deriving it from the actual token count keeps
                    # this correct whether patch_embedding_wancamctrl leaves the
                    # tokens on the image grid (2H x 2W) or the patch grid.
                    _lc = c2ws_plucker_emb.shape[1] // _g[0]
                    assert _lc * _g[0] == c2ws_plucker_emb.shape[1], (
                        f"c2ws tokens {c2ws_plucker_emb.shape[1]} not divisible by frames {_g[0]}")
                    _scale_sq = _lc // (_g[1] * _g[2])
                    assert _scale_sq in (1, 4) and _lc == _scale_sq * _g[1] * _g[2], (
                        f"c2ws spatial {_lc} is neither the patch grid "
                        f"{_g[1]}x{_g[2]} nor 2x it; cannot infer the width axis")
                    _sc = 1 if _scale_sq == 1 else 2
                    _h8, _w8 = _g[1] * _sc, _g[2] * _sc
                    if _CP_FLAT_SEQ:
                        assert (_g[0] * _h8 * _w8) % sp_size == 0, (
                            f"c2ws tokens {_g[0] * _h8 * _w8} not divisible by "
                            f"cp {sp_size}")
                    else:
                        assert _w8 % sp_size == 0, (
                            f"c2ws width {_w8} not divisible by cp {sp_size}")
                    c2ws_plucker_emb = _shard_w(
                        c2ws_plucker_emb, _g[0], _h8, _w8, sp_size, _sp_local_r())
                else:
                    # SP: take this rank's height-row block of the camera condition
                    # (c2ws is [1, F*(H/8)*(W/8), C]; same frame-major rows as x,
                    # but with (H/8) rows per frame rather than the patch grid's).
                    _pw8 = _g[2]  # W/8 == patch-grid width for patch_size=(1,2,2)
                    n_rows8, start_row8 = _sp_row_shard(2 * _g[1])  # c2ws rows = image rows (H/8 = 2*H_patch)
                    _s0, _s1 = start_row8 * _pw8, (start_row8 + n_rows8) * _pw8
                    c2ws_plucker_emb = c2ws_plucker_emb[:, _s0:_s1].contiguous()
            dit_cond_dict = dict(dit_cond_dict)
            dit_cond_dict["c2ws_plucker_emb"] = c2ws_plucker_emb
            self._cam_emb_cache = (_cam_in, c2ws_plucker_emb)

        _phase_mark(self, "cam")
        # seq_lens_int for signature parity with the legacy sequence_parallel
        # module; not consumed by the in-model SP path below.
        seq_lens_int = int(seq_lens.max().item())
        # SP (height-row) slicing: each rank takes a contiguous block of
        # HEIGHT rows (whole rows, so frame/width RoPE coords stay exact; only
        # the height axis is sharded). This is a contiguous token block in the
        # frame-major [f,h,w] flattened layout.
        start_row = 0
        n_rows = None
        if sp_size > 1:
            _g = grid_sizes[0].tolist()  # [F, H_patch, W_patch] (patch grid)
            assert sp_size in (1, 2, 4, 8, 16) or sp_size % 32 == 0, (
                f"SP degree {sp_size} unsupported by Neuron all_to_all "
                "(allowed: 2, 4, 8, 16, or multiples of 32)")
            if _SEQ_MODE == "cp" and _CP_FLAT_SEQ:
                _tok = _g[0] * _g[1] * _g[2]
                assert _tok % sp_size == 0, (
                    f"token count {_tok} (= {_g[0]}*{_g[1]}*{_g[2]}) not divisible "
                    f"by cp_size {sp_size}. Flat CP shards the SEQUENCE, so only "
                    f"the product must divide -- width {_g[2]} need not.")
            elif _SEQ_MODE == "cp":
                assert _g[2] % sp_size == 0, (
                    f"patch-grid WIDTH {_g[2]} not divisible by cp_size {sp_size}. "
                    f"CP shards width, not height (height is {_g[1]}). "
                    f"LINGBOT_CP_FLAT_SEQ=1 relaxes this to (f*h*w) % cp.")
            assert _SEQ_MODE == "cp" or _g[1] % sp_size == 0, (
                f"latent patch-grid height {_g[1]} not divisible by sp_size "
                f"{sp_size}; choose sp_size that divides the patch-grid "
                "height (e.g. h/2 for this task)")
            _pr, _pw = _g[1], _g[2]
            _nf = _g[0]
            if _SEQ_MODE == "cp":
                # WIDTH-axis shard, frame-major, one reshape -- keeps every frame
                # and row on every rank so per-frame modulation stays valid, and
                # divides where height often cannot (29 prime vs 52 = 2^2*13 for
                # examples/03 at 480*832). x is sharded AFTER RoPE/patch-embed and
                # gathered back before norm_out, per the reference implementation neuron_cp.install_cp.
                _r = _sp_local_r()
                x = _shard_w(x, _nf, _pr, _pw, sp_size, _r)
                if e0.size(1) > 1:
                    e = _shard_w(e, _nf, _pr, _pw, sp_size, _r)
                    e0 = _shard_w(e0, _nf, _pr, _pw, sp_size, _r)
                n_rows, start_row = _pr, 0
            else:
                n_rows, start_row = _sp_row_shard(_pr)
                _s0, _s1 = start_row * _pw, (start_row + n_rows) * _pw
                if _nf > 1:
                    # chunk_size>1: repeat the row-shard for each frame's block.
                    _frame = _pr * _pw
                    x = torch.cat([x[:, f*_frame + _s0:f*_frame + _s1] for f in range(_nf)], dim=1).contiguous()
                    if e0.size(1) > 1:
                        e = torch.cat([e[:, f*_frame + _s0:f*_frame + _s1] for f in range(_nf)], dim=1).contiguous()
                        e0 = torch.cat([e0[:, f*_frame + _s0:f*_frame + _s1] for f in range(_nf)], dim=1).contiguous()
                else:
                    x = x[:, _s0:_s1].contiguous()
                    if e0.size(1) > 1:
                        e = e[:, _s0:_s1].contiguous()
                        e0 = e0[:, _s0:_s1].contiguous()
            # A length-1 e/e0 (the shared-timestep case above) needs no row
            # sharding -- it broadcasts across every rank's rows already.
            # Slicing it by row range would yield an empty tensor.

        # HOIST THE ROPE GRIDS OUT OF THE BLOCKS.
        #
        # causal_rope_grids slices freqs on the CPU with a start_frame derived
        # from current_start // frame_seqlen, i.e. from a .item(). Inside a
        # compiled block that is an UNBACKED symint and fullgraph=True fails:
        #   Could not guard on data-dependent expression 22*u3 < 2
        #   Caused by: freqs_split[0][start_frame:start_frame + f] ... :51
        # So the grids are built here, once, and passed in as plain tensors; the
        # blocks then run only causal_rope_apply_rotation, which is pure tensor
        # math and traces cleanly.
        #
        # This also removes work I had measured and wrongly dismissed as not
        # worth fixing: every block rebuilt these for BOTH q and k, so 80 builds
        # per forward where exactly ONE is needed -- all 40 blocks share the same
        # (grid_sizes, start_frame, w_shard). ~12 ms of CPU and 62 MB of H2D per
        # forward, and the reason the block would not compile.
        _rope_cos = _rope_sin = _rope_local_grid = None
        _cp_w_shard = None
        if sp_size > 1 and _SEQ_MODE == "cp":
            _cp_w_shard = (sp_size, _sp_local_r())
        _fs = frame_seqlen if frame_seqlen is not None else int(math.prod(grid_sizes[0][1:]).item())
        _start_frame = int(current_start) // int(_fs)
        _rope_cos, _rope_sin = causal_rope_grids(
            self.freqs, grid_sizes, x.device, _start_frame,
            dtype=_rope_dtype(x if x.dim() == 4 else x.unsqueeze(-1)),
            w_shard=_cp_w_shard)
        # INT tuples, not a tensor -- see causal_rope_apply_rotation's docstring.
        _rope_local_grid = [tuple(int(v) for v in row) for row in grid_sizes.tolist()]
        if _cp_w_shard is not None:
            _rope_local_grid = [(f, h, w // sp_size) for (f, h, w) in _rope_local_grid]

        # WRITE PLAN AS PLAIN INTS, computed here so the compiled block never has
        # to read an index tensor. Mirrors the eager branches in
        # CausalWanSelfAttention.forward exactly, but with the .item() reads on
        # this side of the graph boundary.
        _num_new = int(_fs) * int(grid_sizes[0][0].item())
        # CHUNKED vs FLAT cache. Chunked is the reference implementation's design and needs no write plan
        # at all: routing picks WHICH buffer to write, and the write lands at that
        # buffer's index 0. Detected from the cache's own shape -- a chunked cache
        # holds a LIST of per-chunk tensors per block where the flat one holds a single
        # windowed tensor.
        _chunked = bool(kv_cache) and isinstance(kv_cache[0].get("k"), (list, tuple))
        _dest = _leading = None
        if _chunked:
            _n_chunks = len(kv_cache[0]["k"])
            # chunk index straight from current_start: the caller advances it by exactly
            # one chunk of tokens per step (image2video passes
            # chunk_id * chunk_size * frame_seqlen).
            _chunk_idx = int(current_start) // max(1, _num_new)
            _dest, _leading = chunk_routing(_chunk_idx, _n_chunks)
            _kv_plan = {}
        else:
            _kv_plan = {"kv_write_start": int(current_start),
                        "kv_write_end": int(current_start) + _num_new,
                        "kv_current_end": int(current_start) + _num_new}
            if self.blocks and getattr(self.blocks[0].self_attn, "local_attn_size", -1) != -1:
                _kv_size = int(kv_cache[0]["k"].shape[1])
                _sink = int(getattr(self.blocks[0].self_attn, "sink_size", 0)) * int(_fs)
                _cur_end = int(current_start) + _num_new
                _g_end = int(kv_cache[0]["global_end_index"].item())
                _l_end = int(kv_cache[0]["local_end_index"].item())
                _roll_dst = _roll_src = _roll_len = 0
                if _cur_end > _g_end and _num_new + _l_end > _kv_size:
                    _evicted = _num_new + _l_end - _kv_size
                    _roll_len = _l_end - _evicted - _sink
                    _roll_dst = _sink
                    _roll_src = _sink + _evicted
                    _end = _l_end + _cur_end - _g_end - _evicted
                else:
                    _end = _l_end + _cur_end - _g_end
                _kv_plan = {"kv_write_start": _end - _num_new, "kv_write_end": _end,
                            "kv_roll_dst": _roll_dst, "kv_roll_src": _roll_src,
                            "kv_roll_len": max(0, _roll_len),
                            "kv_current_end": _cur_end}

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            rope_grids=(_rope_cos, _rope_sin),
            rope_local_grid=_rope_local_grid,
            # Write window as PLAIN INTS, so a compiled block specialises on the
            # position instead of carrying a symbolic slice bound. Under CP the
            # cache holds the GATHERED chunk, so the width is the full frame.
            grid_ints=[tuple(int(v) for v in row) for row in grid_sizes.tolist()],
            **_kv_plan,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            dit_cond_dict=dit_cond_dict,
            max_attention_size=max_attention_size,
            frame_seqlen=frame_seqlen,
            cross_attn_first_call=cross_attn_first_call,
            seq_lens_int=seq_lens_int,
            # Where this rank's height-row slice sits in the GLOBAL patch grid.
            # Ulysses ignores both (rank-local RoPE coords cancel); CP needs
            # them to build absolute RoPE positions, because its Q slice is
            # paired against the full gathered K/V.
            sp_start_row=start_row,
            sp_rows=n_rows)

        # At large grids the per-block fp32 RoPE/attention transients fragment
        # the per-logical-core HBM across the 40-block loop; a fresh ~2 GB
        # contiguous peak then fails (Neuron OOM at largest_cached_free ~1.2
        # GB despite ~14 GB total free). One empty_cache before the loop is the
        # minimum-risk reclaim (single sync, measurable), gated to large grids
        # so 480p pays nothing. seq_len ~6240 at 960x1664 vs ~1560 at 480p.
        # The threshold meant 960x1664 (the comment's ~6240 tokens), but seq_len is per CHUNK, so at
        # 480x768 with chunk 3 (4320 tokens) it fired on every forward. LINGBOT_EMPTY_CACHE_SEQ sets it.
        if seq_len >= int(os.environ.get("LINGBOT_EMPTY_CACHE_SEQ", "4096")):
            torch.neuron.empty_cache()

        # (A duplicate of the block loop below used to sit here with no body: it ran
        # block.to(device) for all 40 blocks and never moved them back, which silently
        # defeated the tp_degree<=1 CPU-offload path that CLAUDE.md describes as the
        # safety net for weights that do not fit. No-op under TP, so it went unnoticed.)
        # UNIFORM x DTYPE ACROSS THE BLOCK LOOP.
        #
        # patch_embedding casts x to bf16, but every block ENDS with
        # "x = x + y * e[5]" inside an fp32 autocast, so block 0 received bf16 and
        # blocks 1..39 received fp32. With compiled blocks that is a second
        # specialisation axis on top of the write position, and the recompile log
        # showed the cost exactly:
        #   [2/42] Recompiling function forward ... model_fast.py:670
        #     - 2/40: kv_write_start == 27144
        #     - 2/41: tensor 'x' dtype mismatch. expected BFloat16, actual Float
        # 42 graphs = 21 chunk positions x 2 dtypes. Promoting x to fp32 once here
        # makes every block see the same dtype and halves the graph count, without
        # losing precision: the residual stream was already fp32 for 39 of 40
        # blocks, and each block's first op is self.norm1(x).float() regardless.
        # LINGBOT_RESIDUAL_BF16=1 keeps the residual stream in bf16 between blocks (norm and
        # modulation math still run in fp32 inside each block). Default fp32.
        _phase_mark(self, "plan")
        x = x.to(torch.bfloat16) if _RESIDUAL_BF16 else x.float()
        _REAL_TOKENS[0] = None
        _head_done = False
        _L = x.shape[1]
        if _TOKEN_PAD and _chunked and _L % _TOKEN_PAD:
            _Lp = -(-_L // _TOKEN_PAD) * _TOKEN_PAD
            x = torch_F.pad(x, (0, 0, 0, _Lp - _L))
            _cd = kwargs.get("dit_cond_dict")
            if _cd is not None and "c2ws_plucker_emb" in _cd and _cd["c2ws_plucker_emb"].shape[1] == _L:
                _cd = dict(_cd)
                _cd["c2ws_plucker_emb"] = torch_F.pad(_cd["c2ws_plucker_emb"], (0, 0, 0, _Lp - _L))
                kwargs["dit_cond_dict"] = _cd
            _REAL_TOKENS[0] = _L
        if _chunked:
            # Chunked cache: hand each block its (leading..., destination) buffers.
            # current_start is passed as 0 ON PURPOSE -- the census showed it was the
            # only input that varied across the loop, and it is dead once rope_grids is
            # supplied, so letting it in would reintroduce one graph per chunk position.
            assert _rope_cos is not None, "chunked cache requires prebuilt rope_grids"
            _ks, _vs = ordered_chunks_by_block(kv_cache, _dest, _leading)
            _stacks = self._block_stacks(_n_chunks)
            if _stacks is not None:
                x = _stacks[_dest](x, _ks, _vs, tuple(crossattn_cache),
                                   self._stack_tokens[_dest], kwargs)
            else:
                _films = (self._films_cur if getattr(self, "_films_cur", None) is not None
                          else self._cam_films(kwargs.get("dit_cond_dict")))
                if _film_chk is not None:
                    _film_check_log(_films, self._cam_films(kwargs.get("dit_cond_dict")), _film_chk)
                _phase_mark(self, "films")
                _tpd = tp.tp_degree(_TP_GROUP)
                _sp = _SP_ON and _tpd > 1
                if _sp:
                    assert _films is not None, "LINGBOT_SP needs the hoisted camera FiLM (LINGBOT_CAM_HOIST=1)"
                    x = sp_token_shard(x, _tpd, tp.tp_rank(_TP_GROUP))
                for block_index, block in enumerate(self.blocks):
                    _kw = kwargs
                    if _films is not None:
                        _kw = dict(kwargs)
                        _kw["dit_cond_dict"] = {"cam_film": _films[block_index]}
                    x = block(x, kv_cache=None,
                              kv_chunks=(list(_ks[block_index]), list(_vs[block_index])),
                              crossattn_cache=crossattn_cache[block_index],
                              current_start=0, **_kw)
                _phase_mark(self, "blocks")
                if skip_head:
                    # The caller discards the output (KV-cache commit forward): the blocks have
                    # already written the cache, so the head, its gathers and unpatchify are skipped.
                    _REAL_TOKENS[0] = None
                    if _FWD_PHASES:
                        _phase_mark(self, "head")
                        _p = self._ph_cur
                        if not hasattr(self, "_fwd_phase_log"):
                            self._fwd_phase_log = []
                        self._fwd_phase_log.append(
                            [(n, (t - _p[i][1]) * 1000.0) for i, (n, t) in enumerate(_p[1:])])
                    return None
                if (_sp and _HEAD_COMPILE and sp_size > 1 and _SEQ_MODE == "cp" and _CP_FLAT_SEQ
                        and tp.has_mesh()):
                    hfn = getattr(self, "_head_sp_fn", None)
                    if hfn is None:
                        hfn = torch.compile(_head_sp_cp, backend="neuron", fullgraph=True,
                                            dynamic=False)
                        self._head_sp_fn = hfn
                    _real = _REAL_TOKENS[0]
                    _REAL_TOKENS[0] = None
                    x = self.unpatchify(hfn(self.head, x, e, _real).unsqueeze(0), grid_sizes)
                    if _FWD_PHASES:
                        _phase_mark(self, "head")
                        _p = self._ph_cur
                        if not hasattr(self, "_fwd_phase_log"):
                            self._fwd_phase_log = []
                        self._fwd_phase_log.append(
                            [(n, (t - _p[i][1]) * 1000.0) for i, (n, t) in enumerate(_p[1:])])
                    return [u.float() for u in x]
                if _sp:
                    # Head on the token shard, then gather the small projected tokens (64 wide).
                    x = tp.all_gather_tokens(self.head(x, e), "tp")
                    if _REAL_TOKENS[0] is not None:
                        x = x[:, :_REAL_TOKENS[0]]
                        _REAL_TOKENS[0] = None
                    _head_done = True
        else:
            for block_index, block in enumerate(self.blocks):
                block_is_sharded = getattr(block, 'tp_enabled', False)
                if not block_is_sharded:
                    block.to(device)
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                    }
                )
                x = block(x, **kwargs)
                if not block_is_sharded:
                    block.to('cpu')

        if not _head_done:
            if _REAL_TOKENS[0] is not None:
                x = x[:, :_REAL_TOKENS[0]]
                _REAL_TOKENS[0] = None
            # head
            x = self.head(x, e)

        # Gather the full sequence before unpatchify. The axis must match the one
        # that was sharded:
        #   ulysses -- contiguous blocks of whole HEIGHT rows, so a flat cat on
        #              dim 1 in rank order reconstructs [f,h,w].
        #   cp      -- a WIDTH slice per rank, so the flat cat would interleave
        #              wrongly; regather inside the [B,f,h,w_loc,C] view instead.
        if sp_size > 1:
            if _SEQ_MODE == "cp":
                _g = grid_sizes[0].tolist()
                x = _gather_w(x, _g[0], _g[1], _g[2] // sp_size, sp_group, sp_size)
            else:
                x = dt_util.gather_forward(x, dim=1, group=sp_group)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if _FWD_PHASES:
            _phase_mark(self, "head")
            _p = self._ph_cur
            if not hasattr(self, "_fwd_phase_log"):
                self._fwd_phase_log = []
            self._fwd_phase_log.append(
                [(n, (t - _p[i][1]) * 1000.0) for i, (n, t) in enumerate(_p[1:])])
        return [u.float() for u in x]


    # ------------------------------------------------------------------------------------------
    # LINGBOT_FUSE_COMMIT: one forward = chunk a's KV-cache commit + chunk b's first denoise step.
    # ------------------------------------------------------------------------------------------
    def _fused_time_emb(self, t, host_key, device):
        """(e, e0) for a 1-D timestep: the same ops and the same cache as forward()."""
        if t.dim() == 1:
            t = t.expand(t.size(0), 1)
        e_seq = t.size(1)
        key = (host_key, tuple(t.shape), str(device))
        cache = getattr(self, "_t_emb_cache", None)
        if cache is None:
            cache = self._t_emb_cache = {}
        hit = cache.get(key)
        if hit is not None:
            return hit
        with torch.amp.autocast('neuron', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, e_seq)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            e, e0 = e.to(device), e0.to(device)
        cache[key] = (e, e0)
        return e, e0

    def _fused_cam_tokens(self, cam_in, g, sp_size):
        """This CP rank's camera tokens [1, L/cp, C] for one chunk: forward()'s camera branch
        (flat CP) with its single-entry cache keyed by the condition tuple."""
        hit = getattr(self, "_cam_emb_cache", None)
        if hit is not None and hit[0] is cam_in:
            return hit[1]
        c = [rearrange(i, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                       c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2])
             for i in cam_in]
        c = self.patch_embedding_wancamctrl(torch.cat(c, dim=1))
        c = c + self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(c)))
        _lc = c.shape[1] // g[0]
        _sc = 1 if _lc == g[1] * g[2] else 2
        c = _shard_w(c, g[0], g[1] * _sc, g[2] * _sc, sp_size, _sp_local_r())
        self._cam_emb_cache = (cam_in, c)
        return c

    def forward_fused(self, xa, ya, cam_a, ta, ta_key, chunk_a,
                      xb, yb, cam_b, tb, tb_key, chunk_b,
                      context, seq_len, kv_cache, crossattn_cache, frame_seqlen,
                      max_attention_size=1_000_000):
        """Chunk a's KV-cache commit (its clean latent xa at t = ta = 0) and chunk b's first
        denoise step (xb at tb) in ONE forward: every block sees both chunks' tokens, so the
        weights, collectives and per-block fixed costs are paid once for the two.

        Token layout on each CP rank: [a's R real tokens, pad to P, b's R real tokens, pad to P].
        Each TP shard of that holds tokens of ONE chunk (TP ranks 0..tp/2-1: a, the rest: b), so
        the timestep modulation is per TP rank. Attention (see the `fused` branch of
        CausalWanSelfAttention.forward): a attends to its window -- the cached chunks it shares
        with b, the chunk b evicts, and itself; b to the shared chunks, fresh a and itself -- the
        same key sets as the separate commit and step-0 forwards. Only b's output is returned.

        Supported configuration only: chunked KV cache, LINGBOT_SP, flat CP over a DeviceMesh,
        LINGBOT_TOKEN_PAD, hoisted camera FiLM with LINGBOT_FILM_A2A + LINGBOT_FILM_BATCH, and
        per-block compiled blocks (COMPILE_SCOPE=blocks)."""
        device = getattr(self, '_onload_device', self.patch_embedding.weight.device)
        sp_size, tpd = tp.tp_degree(_SP_GROUP), tp.tp_degree(_TP_GROUP)
        tpr = tp.tp_rank(_TP_GROUP)
        assert (_SP_ON and tpd > 1 and sp_size > 1 and _SEQ_MODE == "cp" and _CP_FLAT_SEQ
                and tp.has_mesh() and _TOKEN_PAD and _CAM_HOIST and _FILM_A2A and _FILM_BATCH), \
            "LINGBOT_FUSE_COMMIT needs SP, flat CP on a mesh, TOKEN_PAD, FILM_A2A and FILM_BATCH"
        assert getattr(self, "_stacks", None) is None, "LINGBOT_FUSE_COMMIT needs COMPILE_SCOPE=blocks"
        set_bf16_rope_for_forward(seq_len)
        if _FWD_PHASES:
            self._ph_cur = []
            _phase_mark(self, "start")

        def embed(x, y):
            u = self.patch_embedding(torch.cat([x, y], dim=0).unsqueeze(0).to(
                self.patch_embedding.weight.dtype))
            return u.flatten(2).transpose(1, 2), tuple(int(v) for v in u.shape[2:])
        xa_t, g = embed(xa, ya)
        xb_t, gb = embed(xb, yb)
        assert g == gb
        f, h, w = g
        L = f * h * w
        assert L % sp_size == 0
        R = L // sp_size
        P = -(-R // _TOKEN_PAD) * _TOKEN_PAD
        S = (2 * P) // tpd
        assert (2 * P) % tpd == 0 and P % S == 0, f"TP shards of {2 * P} tokens must not straddle the chunks"
        r = _sp_local_r()
        pad = lambda u: torch_F.pad(u, (0, 0, 0, P - R))
        x = torch.cat([pad(_shard_w(xa_t, f, h, w, sp_size, r)),
                       pad(_shard_w(xb_t, f, h, w, sp_size, r))], dim=1)
        x = x.to(torch.bfloat16) if _RESIDUAL_BF16 else x.float()

        ea, e0a = self._fused_time_emb(ta, ta_key, device)
        eb, e0b = self._fused_time_emb(tb, tb_key, device)
        grp_b = tpr >= tpd // 2
        e_r, e0_r = (eb, e0b) if grp_b else (ea, e0a)

        _ckey = (id(context[0]), tuple(context[0].shape))
        _chit = getattr(self, "_ctx_emb_cache", None)
        assert _chit is not None and _chit[0] == _ckey and _chit[1] is context[0], \
            "fused forward expects the context embedded by an earlier forward"
        ctx = _chit[2]

        ca = self._fused_cam_tokens(cam_a, g, sp_size)
        cb = self._fused_cam_tokens(cam_b, g, sp_size)
        films = list(self._cam_film_batch_fn(torch.cat([pad(ca), pad(cb)], dim=1)))
        _phase_mark(self, "embed")

        grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
        rdt = _rope_dtype(x.unsqueeze(-1))
        cos_a, sin_a = causal_rope_grids(self.freqs, grid_sizes, x.device, int(chunk_a) * f,
                                         dtype=rdt, w_shard=(sp_size, r))
        cos_b, sin_b = causal_rope_grids(self.freqs, grid_sizes, x.device, int(chunk_b) * f,
                                         dtype=rdt, w_shard=(sp_size, r))
        cos_a, sin_a, cos_b, sin_b = cos_a[0], sin_a[0], cos_b[0], sin_b[0]
        one = torch.ones((P - R,) + tuple(cos_a.shape[1:]), dtype=cos_a.dtype, device=x.device)
        zero = torch.zeros_like(one)
        rope = ([torch.cat([cos_a, one, cos_b, one], 0)], [torch.cat([sin_a, zero, sin_b, zero], 0)])

        n_chunks = len(kv_cache[0]["k"])
        dest_a, lead_a = chunk_routing(int(chunk_a), n_chunks)
        dest_b, lead_b = chunk_routing(int(chunk_b), n_chunks)
        common = [c_ for c_ in lead_a if c_ in lead_b]
        a_reads_b_old = dest_b in lead_a
        assert set(lead_b) == set(common) | {dest_a}, (lead_a, lead_b, dest_a, dest_b)
        assert set(lead_a) == set(common) | ({dest_b} if a_reads_b_old else set())
        order = common + [dest_a, dest_b]
        fz = (P, R, len(common), bool(a_reads_b_old))

        kw = dict(e=e0_r, seq_lens=torch.tensor([L], dtype=torch.long), grid_sizes=grid_sizes,
                  rope_grids=rope, rope_local_grid=[(1, 1, 2 * P)], grid_ints=[(f, h, w)],
                  freqs=self.freqs, context=ctx, context_lens=None,
                  max_attention_size=max_attention_size, frame_seqlen=frame_seqlen,
                  cross_attn_first_call=False, seq_lens_int=L, sp_start_row=0, sp_rows=h)
        _REAL_TOKENS[0] = None
        x = sp_token_shard(x, tpd, tpr)
        for i, block in enumerate(self.blocks):
            ks = [kv_cache[i]["k"][j] for j in order]
            vs = [kv_cache[i]["v"][j] for j in order]
            x = block(x, kv_cache=None, kv_chunks=(ks, vs), crossattn_cache=crossattn_cache[i],
                      current_start=0, dit_cond_dict={"cam_film": films[i]}, fused=fz, **kw)
        _phase_mark(self, "blocks")
        hfn = getattr(self, "_head_fused_fn", None)
        if hfn is None:
            hfn = self._head_fused_fn = torch.compile(_head_fused, backend="neuron",
                                                      fullgraph=True, dynamic=False)
        out = self.unpatchify(hfn(self.head, x, e_r, P, R).unsqueeze(0), grid_sizes)
        if _FWD_PHASES:
            _phase_mark(self, "head")
            _p = self._ph_cur
            if not hasattr(self, "_fwd_phase_log"):
                self._fwd_phase_log = []
            self._fwd_phase_log.append(
                [(n, (t_ - _p[i][1]) * 1000.0) for i, (n, t_) in enumerate(_p[1:])])
        return [u.float() for u in out]

    def precompute_films(self, cam_tuples, c2ws_all, grid):
        """LINGBOT_FILM_ALL: camera FiLM params of every chunk in one compiled graph before the
        chunk loop (each chunk's first forward otherwise spent 19-26 ms on its own). cam_tuples are
        the per-chunk condition tuples the loop will pass (looked up by identity), c2ws_all the
        whole [1, C, F, H, W] Plucker condition, grid the per-chunk patch grid (f, h, w). Same ops
        as forward()'s camera branch and _cam_films, on all chunks' tokens at once."""
        sp_size, tpd = tp.tp_degree(_SP_GROUP), tp.tp_degree(_TP_GROUP)
        assert (_SP_ON and tpd > 1 and sp_size > 1 and _SEQ_MODE == "cp" and _CP_FLAT_SEQ
                and tp.has_mesh() and _TOKEN_PAD and _CAM_HOIST and _FILM_A2A), \
            "LINGBOT_FILM_ALL needs SP, flat CP on a mesh, TOKEN_PAD and FILM_A2A"
        device = getattr(self, '_onload_device', self.patch_embedding.weight.device)
        if getattr(self, "_children_on", None) != str(device):
            for _name, _mod in self.named_children():
                if _name != 'blocks':
                    _mod.to(device)
            self._children_on = str(device)
        f, h, w = grid
        c = rearrange(c2ws_all, '1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)',
                      c1=self.patch_size[0], c2=self.patch_size[1], c3=self.patch_size[2])
        c = self.patch_embedding_wancamctrl(c)
        c = c + self.c2ws_hidden_states_layer2(torch_F.silu(self.c2ws_hidden_states_layer1(c)))
        L = f * h * w
        K = len(cam_tuples)
        assert c.shape[1] >= K * L or c.shape[1] % L == 0, (c.shape, L, K)
        R = L // sp_size
        P = -(-R // _TOKEN_PAD) * _TOKEN_PAD
        r = _sp_local_r()
        n_avail = c.shape[1] // L
        toks = []
        for k in range(K):
            ck = c[:, min(k, n_avail - 1) * L:(min(k, n_avail - 1) + 1) * L]
            ck = _shard_w(ck, f, h, w, sp_size, r)
            toks.append(torch_F.pad(ck, (0, 0, 0, P - R)))
        fn = getattr(self, "_films_multi_fn", None)
        if fn is None:
            raw = [getattr(b, "_orig_mod", b) for b in self.blocks]

            def _all(c_cat, K_, P_):
                return _films_multi(raw, c_cat, K_, P_)
            fn = self._films_multi_fn = torch.compile(_all, backend="neuron", fullgraph=True,
                                                      dynamic=False)
        res = fn(torch.cat(toks, dim=1), K, P)
        self._films_pre = {}
        for k, t in enumerate(cam_tuples):
            self._films_pre[id(t)] = (t, [(sc[k:k + 1], sh[k:k + 1]) for sc, sh in res])
        if os.environ.get("LINGBOT_FILM_CHECK", "") and not getattr(self, "_film_check_done", False):
            # First generation only, so a benchmark's timed pass is unaffected.
            self._film_check_ids = {id(t): k for k, t in enumerate(cam_tuples)}
            self._film_check_done = True

    def _cam_films(self, dit_cond_dict):
        """Per-block camera FiLM params, computed once per chunk (LINGBOT_CAM_HOIST=1, default).

        cam_film() reads only the chunk's camera condition, which is the same object for all
        forwards of a chunk (see _cam_emb_cache), so its 4 matmuls and 1 all-reduce per block
        need not run in every denoise step. Returns a list (one (scale, shift) per block) or
        None to keep the in-block computation."""
        if not _CAM_HOIST or dit_cond_dict is None or "c2ws_plucker_emb" not in dit_cond_dict:
            return None
        c2ws = dit_cond_dict["c2ws_plucker_emb"]
        src = getattr(self, "_cam_emb_cache", None)
        key = (id(src[1]) if src is not None else id(c2ws), tuple(c2ws.shape))
        hit = getattr(self, "_cam_film_cache", None)
        if hit is not None and hit[0] == key and hit[2] is (src[1] if src is not None else c2ws):
            return hit[1]
        _tpd = tp.tp_degree(_TP_GROUP)
        # SP + LINGBOT_FILM_A2A: each block's FiLM params come out of ONE compiled graph already
        # token-sharded (cam_film_sp), so the eager gather/slice/clone below is skipped.
        _a2a = _SP_ON and _FILM_A2A and _tpd > 1 and tp.has_mesh()
        if _a2a and _FILM_BATCH and os.environ.get("LINGBOT_COMPILE_DIT", "0") == "1":
            bfn = getattr(self, "_cam_film_batch_fn", None)
            if bfn is None:
                raw = [getattr(b, "_orig_mod", b) for b in self.blocks]

                def _all_films(c):
                    return tuple(b.cam_film_sp(c) for b in raw)
                bfn = torch.compile(_all_films, backend="neuron", fullgraph=True, dynamic=False)
                self._cam_film_batch_fn = bfn
            films = list(bfn(c2ws))
            self._cam_film_cache = (key, films, src[1] if src is not None else c2ws)
            return films
        fns = getattr(self, "_cam_film_fns", None)
        if fns is None:
            raw = [getattr(b, "_orig_mod", b) for b in self.blocks]
            meth = "cam_film_sp" if _a2a else "cam_film"
            if os.environ.get("LINGBOT_COMPILE_DIT", "0") == "1":
                fns = [torch.compile(getattr(b, meth), backend="neuron", fullgraph=True,
                                     dynamic=False) for b in raw]
            else:
                fns = [getattr(b, meth) for b in raw]
            self._cam_film_fns = fns
        _full = (_FILM_FULL or _SP_ON) and _tpd > 1 and not _a2a
        _r = tp.tp_rank(_TP_GROUP) if (_full and _SP_ON) else None
        # One block at a time, so at most one block's full-width pair is alive (the old
        # list-at-once form held 40 x 2 full-width [1, L, C] tensors per rank).
        films = []
        for fn in fns:
            s, sh = fn(c2ws)
            if _full:
                # Gather the column-parallel FiLM params to full width ONCE PER CHUNK, so each
                # of the chunk's 5 forwards applies them to the full x with no per-block
                # all-gather.
                s, sh = film_full_width(s, sh)
                if _r is not None:
                    # Sequence parallel: keep only this rank's tokens. clone(), not
                    # sp_token_shard's .contiguous(): a dim-1 slice of [1, L, C] is already
                    # contiguous, so the view would pin the whole full-width tensor in the cache.
                    s = sp_token_shard(s, _tpd, _r).clone()
                    sh = sp_token_shard(sh, _tpd, _r).clone()
            films.append((s, sh))
        # Hold a reference to the key object so its id cannot be reused while cached.
        self._cam_film_cache = (key, films, src[1] if src is not None else c2ws)
        return films

    def _block_stacks(self, n_chunks):
        """Compiled block stacks (one shared by default), or None if compile is off.

        ONE shared stack for all `n_chunks` destinations, tracking the reference implementation
        which flipped this default. `NEURON_SPECIALIZE_NEFF=1` restores the old
        per-destination path: one compiled callable PER cache-write destination,
        "to avoid an expensive IO rebind when reusing a single NEFF"
        (the reference implementation setup.py:225) -- overhead the reference implementation now reports as resolved.

        In the specialised path each stack gets a token of a DIFFERENT shape
        (`torch.zeros(dest + 1)`), which is what stops Dynamo collapsing them into one
        cache entry: at steady state the real inputs have identical shapes and differ
        only in WHICH pinned buffers they point at. Shared path uses one shape, so one
        entry. Built lazily because n_chunks comes from the cache the caller allocated.
        """
        if os.environ.get("LINGBOT_COMPILE_DIT", "0") != "1":
            return None
        if getattr(self, "_stacks_built", None) == n_chunks:
            return self._stacks
        if not tp.has_mesh():
            _q("[model_fast] scope=block_stack needs a DeviceMesh "
               "collectives would not lower. Not compiling.")
            self._stacks, self._stacks_built = None, n_chunks
            return None
        # the reference implementation raises BOTH limits before compiling (reference setup.py:215).
        # Without it, 40 block graphs plus their guards blow the default cache and
        # Dynamo thrashes -- 64 recompiles of one shared wrapper, measured here.
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        if hasattr(torch._dynamo.config, "do_not_emit_runtime_asserts"):
            # Unconditional: cp>1 will not lower without it ("failed to legalize
            # operation 'torch.runtime.assert'"), and the constraint it drops is
            # guaranteed twice OUTSIDE the graph -- the job preflight rejects a
            # patch-grid width not divisible by cp, and _shard_w asserts w % cp == 0
            # eagerly before any block runs.
            torch._dynamo.config.do_not_emit_runtime_asserts = True
        # COMPILE_SCOPE, mirroring the reference implementation's NeuronConfig.compile_scope exactly.
        # 'blocks' is the reference implementation's _compile_blocks tail (reference setup.py):
        #     for i, block in enumerate(transformer.blocks):
        #         transformer.blocks[i] = torch.compile(block, backend="neuron",
        #                                               fullgraph=True, dynamic=False)
        # i.e. compile each block module; the stack wrapper is not used at all and the
        # caller's existing per-block loop runs them. That is the scope
        # the reference launcher sets (COMPILE_SCOPE=block, normalized to 'blocks' by
        # run_gen.sh) and therefore the one the reference implementation's working 720p runs measured.
        #
        # 'block_stack' is the other branch: all 40 blocks as ONE region, specialised per
        # cache-write destination. It costs one compile per destination -- the 21-position
        # prewarm and the 181 s straggler forward seen at 480p -- and is the combination
        # recorded as aborting 2/2 at TP8xCP8 with const-max on the reference implementation.
        #
        # Default 'blocks' to match the reference implementation. COMPILE_SCOPE=block_stack restores the
        # previous behaviour.
        _scope = os.environ.get("COMPILE_SCOPE", "blocks").strip().lower()
        if _scope in ("block", "blocks"):
            _n = _PROFILE_BLOCK_CAP or len(self.blocks)
            for _i in range(_n):
                self.blocks[_i] = torch.compile(
                    self.blocks[_i], backend="neuron", fullgraph=True, dynamic=False)
            # None -> the caller's per-block loop runs, which is what 'blocks' means.
            self._stacks, self._stack_tokens, self._stacks_built = None, None, n_chunks
            _q(f"[model_fast] scope=blocks: compiled {_n} shared block calls, "
               f"fullgraph=True (the reference implementation COMPILE_SCOPE=blocks)")
            return None
        if _scope not in ("stack", "block_stack"):
            raise ValueError(f"COMPILE_SCOPE must be blocks or block_stack, got {_scope!r}")

        dev = next(self.parameters()).device
        _blocks = (self.blocks[:_PROFILE_BLOCK_CAP] if _PROFILE_BLOCK_CAP
                   else self.blocks)
        stacks, tokens = [], []
        single = os.environ.get("NEURON_SPECIALIZE_NEFF", "0") != "1"
        if single:
            # ONE shared stack for every destination. the reference implementation flipped this default in
            #: "IO rebind overhead that originally motivated specialization is
            # no longer significant." The specialised path costs n_chunks copies of the
            # whole 40-block stack resident at once, which is what the OOM note in
            # TRN3_HYBRID_ATTENTION_STATE.md suspected. One token of one shape, so
            # Dynamo keeps a single cache entry instead of n_chunks of them.
            tok = torch.zeros(1, dtype=torch.float32, device=dev)
            try:
                torch._dynamo.mark_static_address(tok)
            except Exception:
                pass
            shared = _compile_leaf(_BlockStack(_blocks))
            for _ in range(n_chunks):
                stacks.append(shared)
                tokens.append(tok)
        else:
            for dest in range(n_chunks):
                tok = torch.zeros(dest + 1, dtype=torch.float32, device=dev)
                try:
                    torch._dynamo.mark_static_address(tok)
                except Exception:
                    pass
                tokens.append(tok)
                stacks.append(_compile_leaf(_BlockStack(_blocks)))
        self._stacks, self._stack_tokens, self._stacks_built = stacks, tokens, n_chunks
        _q(f"[model_fast] compiled "
           f"{'1 shared' if single else f'{n_chunks} destination-specialised'} x "
           f"{_PROFILE_BLOCK_CAP or len(self.blocks)}-block stacks"
           + ("  *** PROFILING DEPTH, NOT A VALID VIDEO ***"
              if _PROFILE_BLOCK_CAP else ""))
        return stacks

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


# =============================================================================
# Tensor parallelism (DiT-TP)
#

# ── selective torch.compile for the DiT (LINGBOT_COMPILE_DIT=1) ───────────────
# Ported from ~/wan2-i2v-14b inference_neuron_i2v.py:249-254, which compiles the
# DiT PIECEWISE -- patch_embedding, text_embedding, head and every block.ffn --
# and leaves attention eager. That is the workaround for the constraint this file
# documents at init_dit_mesh but never acted on: torch.compile(backend="neuron")'s
# collective lowering rejects carved sub-groups, so LingBot kept EVERYTHING eager
# and the last run left 944 per-op NEFFs.
#
# Two deliberate differences from that repo's call:
#   - no fullgraph=True. They rely on a graph BREAK at the collective
#     (@torch.compiler.disable on all_reduce_sum, tp_utils.py:96); fullgraph=True
#     would forbid exactly that. _tp_rms_norm above keeps fullgraph=True because
#     it is a leaf whose all_reduce is the whole point.
#   - LingBot never calls self.ffn(x) under TP. model_fast.py splits it into
#     ffn[1](ffn[0](x)) then _tp_row_parallel(ffn[2], y), and _tp_row_parallel
#     already does linear(x) BEFORE the all_reduce -- so compiling the Linears
#     themselves puts the matmul in a NEFF and leaves the collective in eager
#     python with no graph break required.
#
# Attention is left eager on purpose: it holds the KV-cache index arithmetic,
# the .item() reads and the CP all_gather, none of which belong in a static graph.
_COMPILED_DIT = False




_PROFILE_BLOCK_CAP = None


def set_profile_block_cap(n):
    """Build the stack over the first `n` blocks only. PROFILING ONLY.

    the reference implementation's pick_dit_neff.py records the constraint that makes this necessary: at depth
    3 their trace was 4.6 GB and neuron-explorer could not ingest it even at a 60 minute
    timeout, while depth 1 gave 647 MB. Our stack is 40 blocks, so a full-depth trace is
    not ingestible and no NEFF+NTFF pair can be obtained from it.

    The shallow stack is still the COMPLETE compiled region -- same attention, same
    collectives, same chunked-cache plumbing -- just fewer blocks, which is exactly the
    property that makes their depth-1 graph the one they profile.

    Never set this in a generation run: it would silently produce a video from a
    truncated network.
    """
    global _PROFILE_BLOCK_CAP
    _PROFILE_BLOCK_CAP = n


class _BlockStack(nn.Module):
    """All 40 transformer blocks as ONE compiled region. the reference implementation's `block_stack` scope
    (reference setup.py:218), `the reference block stack`).

    Two reasons it beats compiling blocks individually:

    - One graph entry per forward instead of 40, so Dynamo guard evaluation and the
      per-region launch overhead are paid once.
    - It is the unit the reference implementation specialises PER CACHE-WRITE DESTINATION, which is the
      point of the exercise: "Compiling a block-stack per cache write destination to
      avoid an expensive IO rebind when reusing a single NEFF" (setup.py:225).

    Everything crossing the boundary is a tensor, a tuple of tensors, or a plain int.
    `current_start` deliberately does NOT: the shape census showed it was the only
    input that varied across the denoise loop (21 distinct values at 81 frames), and
    it is dead here because RoPE arrives pre-built as `rope_grids`.
    """

    def __init__(self, blocks):
        super().__init__()
        # The model's own ModuleList, not a copy: same parameters, same storage.
        self.blocks = blocks

    def forward(self, x, k_by_block, v_by_block, ca_by_block, token, static):
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                kv_cache=None,
                kv_chunks=(list(k_by_block[i]), list(v_by_block[i])),
                crossattn_cache=ca_by_block[i],
                current_start=0,
                **static,
            )
        return x


def _compile_leaf(mod):
    """fullgraph=True by DEFAULT, and deliberately so.

    I first copied ~/wan2-i2v-14b's call, which omits fullgraph. They omit it
    because their compiled region (block.ffn) CONTAINS a collective and they need
    the graph break at all_reduce_sum. This region contains no collective, so the
    break is neither needed nor wanted -- and fullgraph=True makes a silent
    sub-graph split IMPOSSIBLE instead of something inferred from a
    "graph breaks=0" log line.

    It is also this repo's own convention, which I diverged from without reason:
    _tp_rms_norm above and the T5 encoder via utils/neuron_compile.py both use
    fullgraph=True.
    """
    return torch.compile(mod, backend="neuron", dynamic=False, fullgraph=True)



# Megatron-style tensor-parallel sharding for WanModelFast (causal_fast DiT).
# Registers a named "dit-tp" process group in the shared tensor-parallel
# registry (wan/distributed/tensor_parallel.py) -- the same registry T5-TP
# (wan/modules/t5.py) registers its own "t5-tp" group in -- so DiT's
# and T5's TP groups can't collide even if a future pass runs both stages in
# one torchrun process. Weight-slicing (shard_column_parallel/
# shard_row_parallel/shard_norm_vector) is shared with T5-TP, in
# tensor_parallel.py -- bias is optional there since T5's Linears are
# bias=False and DiT's aren't.
#
# Scope: CausalWanSelfAttention / WanCrossAttention / FFN / camera-injector
# Linears only. model_causal.py and model.py are untouched.
# =============================================================================

_SP_GROUP = "dit-sp"
_TP_GROUP = "dit-tp"

# How the sequence-parallel group ("dit-sp") is USED. The group geometry is the
# same either way -- init_dit_mesh() carves it identically -- but the attention
# math differs, and that difference is what caps the world size:
#
#   "ulysses" (default, unchanged): all_to_all converts the token shard into a
#       HEAD shard inside attention, so heads divide by sp*tp. With 40 heads
#       that forces sp*tp into {1,2,4,5,8,10,20,40}, and generate.py rejects
#       any world size that is not a divisor of 40. tp8 x sp4 = 32 is illegal
#       (40 % 32 = 8), which is why 8 ranks was the ceiling.
#
#   "cp": the token shard STAYS a token shard. Each rank keeps its own Q slice
#       and gets the full K/V, exactly nkilib attention_cte's context-parallel
#       mode (cp_offset / global_cp_deg). Heads then divide by tp ONLY -- 40/8
#       = 5 per rank -- and cp multiplies the rank count without touching the
#       head count. tp8 x cp4 = 32 ranks becomes legal.
_SEQ_MODE = "ulysses"


def _q(msg):
    """Print once, from rank 0 only. These lines were emitted by every rank, so at
    32 ranks each one appeared 32 times and buried the markers that matter."""
    if int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0) == 0:
        print(msg, flush=True)


def set_seq_parallel_mode(mode):
    """Select "ulysses" or "cp" for the dit-sp group. Call once after
    init_dit_mesh(), before the first forward."""
    global _SEQ_MODE
    assert mode in ("ulysses", "cp"), f"unknown sequence-parallel mode {mode!r}"
    _SEQ_MODE = mode


def get_seq_parallel_mode():
    return _SEQ_MODE


def init_dit_parallel_group(tp_degree, ranks=None):
    """Register the "dit-tp" tensor-parallel process group. Call after
    wan.distributed.util.init_distributed_group().

    Two forms:
      init_dit_parallel_group(tp_degree=k)   -- group over ranks [0, k).
      init_dit_parallel_group(ranks=[...])   -- explicit sub-mesh rank list, so
        the DiT TP group can be a true sub-group of a larger torchrun world
        (TRN3 full-parallelism: TP is one axis of an sp x tp rank mesh, no
        longer required to equal world_size).

    IMPORTANT (SDK constraint): torch.compile(backend="neuron")'s collective
    lowering rejects non-world process sub-groups ("replica id #N not seen in
    replica groups"). So every DiT TP/SP collective below stays in EAGER python
    between compiled leaves -- only the collective-free Linear/RMSNorm math is
    compiled. This is unchanged from the world-group TP path; it just works
    for a carved sub-group now.
    """
    if (ranks is not None and len(ranks) > 1) or (ranks is None and tp_degree > 1):
        assert dist.is_initialized(), (
            "init_distributed_group() must run before init_dit_parallel_group()")
    tp.init_tp_group(_TP_GROUP, tp_degree=tp_degree, ranks=ranks)
    # Build the DeviceMesh HERE TOO, not only in init_dit_mesh. This is the
    # TP-only path (cp/sp == 1), and it is the one the cp1 runs take -- the first
    # block-compile attempt refused to compile because init_mesh_groups was only
    # reached via init_dit_mesh, so at cp1 there was no mesh and nothing lowered.
    # A (1, tp) mesh is still a mesh: fc.all_reduce over its "tp" submesh
    # traces, where a bare ProcessGroup does not.
    if (tp_degree > 1 and ranks is None
            ):
        try:
            tp.init_mesh_groups(1, tp_degree)
            _q(f"[model_fast] mesh (cp=1, tp={tp_degree})")
        except Exception as e:
            print(f"[model_fast] NO MESH ({e}) -- compiled collectives will not lower",
                  flush=True)


_MESH_CP_FAST = False


def set_mesh_cp_fast(v=True):
    """Put CP on the FAST mesh axis, so CP ranks are contiguous.

    Required when a collective inside an NKI kernel routes over the CP group: with CP
    strided the runtime cannot build a comm for it at all ("Failed to find device to
    device paths", "no_hier no_mesh"), and a plain framework all_gather over the same
    group fails the same way. Costs TP all-reduce bandwidth, which becomes strided.

    Call BEFORE init_dit_mesh.
    """
    global _MESH_CP_FAST
    _MESH_CP_FAST = v


def init_dit_mesh(sp_size, tp_size):
    """Carve the DiT sp_size x tp_size rank mesh (TRN3 full-parallelism).

    Registers the calling rank's "dit-tp" (strided) and "dit-sp" (contiguous
    tp-block) sub-groups over world ranks [0, sp_size*tp). MUST be called by
    every rank in the torchrun world (dist.new_group is collective across the
    whole world, including ranks outside [0, sp*tp) who pass through).

    Mesh geometry (validated): global rank r -> (sp_local = r // tp_size,
    tp_local = r % tp_size). TP group for tp_local=t = strided {t, t+tp, ...};
    SP group for sp_local=c = contiguous [c*tp, (c+1)*tp). Collectives stay
    EAGER -- the SDK's compiled collective lowering rejects carved sub-groups,
    so all_to_all / all_reduce run in eager python between compiled Linear /
    RMSNorm leaves (the same pattern the world-group TP path already used).
    """
    assert dist.is_initialized(), "init_distributed_group() must run before init_dit_mesh()"
    global _TP_DEGREE_MESH, _SP_DEGREE_MESH
    # ONE SOURCE OF TRUTH: the DeviceMesh. "dit-sp" and "dit-tp" are now the mesh's
    # own cp/tp submesh groups, so the eager dist.* sites and the compiled functional
    # collectives cannot disagree about who is a peer or what index this rank holds.
    #
    # This replaces init_product_mesh, which built "-sp" contiguous and "-tp" strided
    # while the DeviceMesh (tp on the fast axis) makes "cp" strided and "tp"
    # contiguous -- exact transposes. See init_mesh_groups for the rank arithmetic.
    # An earlier fix here swapped the arguments, which corrected the group SIZES and
    # left the LAYOUT inverted; that is what made every cp4 video wrong while every
    # cp1 video was right.
    _TP_DEGREE_MESH, _SP_DEGREE_MESH = tp_size, sp_size
    # The mesh is now LOAD-BEARING, so a failure here must be fatal. It used to be
    # caught and warned about, which was survivable while init_product_mesh registered
    # dit-sp/dit-tp independently. It no longer does: swallowing this would leave both
    # unregistered, tp_degree() would return 1, TP sharding would silently vanish and
    # the run would produce wrong output while reporting success -- the same failure
    # mode that just cost five cp4 runs.
    #
    # A DeviceMesh submesh is also the only form that lowers inside a compiled region;
    # a bare dist.new_group gives "replica id #N not seen in replica groups". tp is the
    # FAST axis: TP contiguous, CP strided.
    tp.init_mesh_groups(sp_size, tp_size, names=(_SP_GROUP, _TP_GROUP),
                       cp_fast=_MESH_CP_FAST)
    _q(f"[model_fast] mesh (cp={sp_size}, tp={tp_size}) -- dit-sp/dit-tp derived from it")


def init_dit_sp_group(sp_size):
    """Register "dit-sp" alone, spanning world ranks [0, sp_size) (legacy
    single-axis SP, tp=1). Prefer init_dit_mesh for the TRN3 sp x tp mesh."""
    if sp_size > 1:
        assert dist.is_initialized(), (
            "init_distributed_group() must run before init_dit_sp_group()")
    tp.init_tp_group(_SP_GROUP, sp_size)


def get_dit_sp_group():
    """Return the registered dit-sp group, or None when SP is disabled."""
    return tp._get(_SP_GROUP) if tp.is_registered(_SP_GROUP) else None


_get_dit_sp_group = get_dit_sp_group  # internal alias (call sites use this name)


def sp_group_dist_size(group):
    """Size of the SP (Ulysses) group from the registered dit-sp group."""
    return dist.get_world_size(group) if group is not None else 1


_TP_DEGREE_MESH = 1
_SP_DEGREE_MESH = 1


def get_sp_size():
    return tp.tp_degree(_SP_GROUP)


def get_sp_rank():
    return tp.tp_rank(_SP_GROUP)


def _sp_rope_grid(g, sp_sz, dtype, device):
    """(Unused: RoPE grid for an SP height-row shard.)"""
    return torch.tensor([[g[0].item(), g[1].item() // sp_sz, g[2].item()]],
                        dtype=dtype, device=device)


def _sp_row_shard(rows):
    """This rank's (n_rows, start_row) height-row shard for spatial SP."""
    sp_sz = sp_group_dist_size(_get_dit_sp_group())
    if sp_sz == 1:
        return rows, 0
    assert rows % sp_sz == 0, f"height rows {rows} not divisible by sp {sp_sz}"
    r = _sp_local_r()
    n = rows // sp_sz
    return n, r * n


def _sp_local_r():
    """This rank's index within its SP (Ulysses) group = tp.get_rank("dit-sp")
    (position among the contiguous tp-sized block that forms one SP cell).
    Used to pick this rank's frame block along the temporal axis."""
    if not tp.is_registered(_SP_GROUP):
        return 0
    return tp.get_rank(_SP_GROUP)


def destroy_dit_parallel_group():
    if tp.is_registered(_TP_GROUP):
        tp.destroy_group(_TP_GROUP)


def get_tp_degree():
    return tp.tp_degree(_TP_GROUP)


def get_tp_rank():
    return tp.tp_rank(_TP_GROUP)


# ---- weight-slicing helpers -------------------------------------------------
# shard_column_parallel / shard_row_parallel / shard_norm_vector live in
# wan/distributed/tensor_parallel.py, shared with T5-TP (wan/modules/
# t5.py) -- see call sites below (tp.shard_*).


# ---- module rebuild: post-construction surgery ------------------------------
#
# WanModelFast.__init__ is @register_to_config'd, so a tp_degree constructor
# kwarg would leak into the saved checkpoint config. Instead, these functions
# reshape an already-built model's Linear/WanRMSNorm submodules in place and
# set plain `tp_enabled` runtime attributes -- invisible to register_to_config,
# which only intercepts __init__ argument binding, once, at construction time.

def _rebuild_attn_for_tp(attn, tp_degree):
    """Rebuild q/k/v/o + norm_q/norm_k in place to TP shapes. Shared by both
    CausalWanSelfAttention and WanCrossAttention -- identical __init__ shapes,
    since WanCrossAttention inherits WanSelfAttention.__init__ unchanged."""
    dim = attn.dim
    num_heads = attn.num_heads
    assert dim % tp_degree == 0 and num_heads % tp_degree == 0
    local_dim = dim // tp_degree
    dtype = attn.q.weight.dtype
    device = attn.q.weight.device

    attn.q = nn.Linear(dim, local_dim, dtype=dtype, device=device)
    attn.k = nn.Linear(dim, local_dim, dtype=dtype, device=device)
    attn.v = nn.Linear(dim, local_dim, dtype=dtype, device=device)
    attn.o = nn.Linear(local_dim, dim, dtype=dtype, device=device)
    if attn.qk_norm:
        attn.norm_q = WanRMSNorm(local_dim, eps=attn.eps).to(dtype=dtype, device=device)
        attn.norm_k = WanRMSNorm(local_dim, eps=attn.eps).to(dtype=dtype, device=device)
    # attn.dim is intentionally left at the ORIGINAL full width (5120, not
    # 1280) -- the QK-norm all-reduce needs it to divide by the true global
    # element count when finishing the RMS formula. attn.head_dim (dim //
    # num_heads) is numerically unchanged by TP (local_dim // local_heads ==
    # dim // num_heads), so it's left as-is too.
    attn.num_heads = num_heads // tp_degree
    attn.tp_enabled = True


def _rebuild_ffn_for_tp(ffn, tp_degree):
    """Rebuild the 2-linear FFN Sequential in place. ffn[1] (GELU) is untouched."""
    dim = ffn[0].in_features
    ffn_dim = ffn[0].out_features
    assert ffn_dim % tp_degree == 0
    local_ffn_dim = ffn_dim // tp_degree
    dtype = ffn[0].weight.dtype
    device = ffn[0].weight.device

    ffn[0] = nn.Linear(dim, local_ffn_dim, dtype=dtype, device=device)
    ffn[2] = nn.Linear(local_ffn_dim, dim, dtype=dtype, device=device)


def _rebuild_cam_injector_for_tp(block, tp_degree):
    """Shard cam_injector_layer1/2 (a clean column->row MLP, same shape as the
    FFN) and column-shard cam_scale_layer/cam_shift_layer.

    cam_scale/cam_shift were previously left replicated on the grounds that
    they consume a full c2ws_hidden_states and must produce a full-width
    scale/shift. The first half is true -- their input stays full -- but the
    consumer is `x = (1 + cam_scale) * x + cam_shift`, which is elementwise per
    feature. So each rank can own an output-feature slice, apply it to the
    matching slice of x, and one all_gather restores full x: one collective per
    block replacing two replicated dim x dim matmuls per block.

    That replication was expensive. At dim=5120 these are two nn.Linear(5120,
    5120) per block -- 50 MB each in bf16, 4.0 GB across 40 blocks, on EVERY
    rank. Measured at tp=8 it was 3.91 GiB of the 4.46 GiB replicated (54.3%)
    of each rank's 8.22 GiB, and it is what put 960*1664 over per-core HBM."""
    dim = block.cam_injector_layer1.in_features
    assert dim % tp_degree == 0
    local_dim = dim // tp_degree
    dtype = block.cam_injector_layer1.weight.dtype
    device = block.cam_injector_layer1.weight.device

    block.cam_injector_layer1 = nn.Linear(dim, local_dim, dtype=dtype, device=device)
    block.cam_injector_layer2 = nn.Linear(local_dim, dim, dtype=dtype, device=device)
    # Column-parallel: full input, this rank's slice of the output features.
    block.cam_scale_layer = nn.Linear(dim, local_dim, dtype=dtype, device=device)
    block.cam_shift_layer = nn.Linear(dim, local_dim, dtype=dtype, device=device)


def rebuild_wan_model_fast_for_tp(model, tp_degree):
    """Reshape every block's self_attn, cross_attn, ffn, and cam-injector
    Linears to their TP-sharded sizes, and flip on the tp_enabled flags the
    forward-pass code checks. Call BEFORE loading a sharded state dict (see
    load_tp_wan_model_fast) -- if the model is still on the meta device at
    this point, to_empty() afterward allocates storage at the correct,
    already-shrunk sizes."""
    if tp_degree <= 1:
        return
    for block in model.blocks:
        _rebuild_attn_for_tp(block.self_attn, tp_degree)
        _rebuild_attn_for_tp(block.cross_attn, tp_degree)
        _rebuild_ffn_for_tp(block.ffn, tp_degree)
        _rebuild_cam_injector_for_tp(block, tp_degree)
        block.tp_enabled = True


# ---- state-dict sharding -----------------------------------------------------

_COLUMN_WEIGHT_RE = re.compile(
    r"\.(self_attn|cross_attn)\.(q|k|v)\.weight$"
    r"|\.ffn\.0\.weight$"
    r"|\.cam_injector_layer1\.weight$"
    r"|\.cam_(scale|shift)_layer\.weight$"
)
_ROW_WEIGHT_RE = re.compile(
    r"\.(self_attn|cross_attn)\.o\.weight$"
    r"|\.ffn\.2\.weight$"
    r"|\.cam_injector_layer2\.weight$"
)
_NORM_WEIGHT_RE = re.compile(
    r"\.(self_attn|cross_attn)\.(norm_q|norm_k)\.weight$"
)


def shard_wan_model_fast_state_dict(full_state_dict, tp_rank, tp_degree):
    """Classify every key in an unsharded WanModelFast state dict by name and
    slice it for this rank. Non-matching keys (patch/text/time embeddings,
    per-block norm1/2/3, modulation, head, ...) pass through unsliced --
    they're replicated on every rank.

    Bias keys are resolved together with their paired weight key (relies on
    nn.Linear registering 'weight' before 'bias', so the weight key is always
    visited first in a real state dict's iteration order).
    """
    if tp_degree <= 1:
        return dict(full_state_dict)

    sharded = {}
    consumed = set()
    for name, tensor in full_state_dict.items():
        if name in consumed:
            continue
        if name.endswith(".weight") and _COLUMN_WEIGHT_RE.search(name):
            bias_name = name[:-len("weight")] + "bias"
            bias = full_state_dict.get(bias_name)
            w, b = tp.shard_column_parallel(tensor, bias, tp_rank, tp_degree)
            sharded[name] = w
            if bias is not None:
                sharded[bias_name] = b
                consumed.add(bias_name)
        elif name.endswith(".weight") and _ROW_WEIGHT_RE.search(name):
            bias_name = name[:-len("weight")] + "bias"
            bias = full_state_dict.get(bias_name)
            w, b = tp.shard_row_parallel(tensor, bias, tp_rank, tp_degree)
            sharded[name] = w
            if bias is not None:
                sharded[bias_name] = b
                consumed.add(bias_name)
        elif _NORM_WEIGHT_RE.search(name):
            sharded[name] = tp.shard_norm_vector(tensor, tp_rank, tp_degree)
        else:
            sharded[name] = tensor.clone()
    return sharded


# ---- weight loading: meta-skeleton + mmap ----------------------------------

def load_tp_wan_model_fast(checkpoint_dir, subfolder, tp_degree, dtype,
                            local_attn_size, sink_size, device, pth_path=None):
    """Build a TP-sharded WanModelFast for this rank.

    Prefers an mmap'd, TP-degree-independent bf16 .pth (pth_path); falls back
    to plain from_pretrained + state_dict() if not yet converted.

    Ranks take turns (barrier between each) rather than all loading at once --
    4 concurrent reads of the same multi-GB file OOM'd on real hardware even
    though each rank's own shard is small.
    """
    tp_rank = get_tp_rank()
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    model = None
    # One WORLD rank loads at a time (barrier between each): under an sp x tp
    # mesh the TP-group-local rank collides across SP cells, so guard on the
    # global rank to keep a single concurrent multi-GB checkpoint load.
    #
    # LINGBOT_LOAD_CONCURRENCY=N lets N ranks load per turn instead (default 1, the
    # behaviour above). At 64 ranks one-at-a-time is ~10 s per rank, 10.5 min in all; the
    # OOM it guards against is HOST memory on a memory-limited pod, so raise it only where
    # host RAM allows (each rank holds ~10-13 GB of CPU tensors while it loads). A turn
    # takes ranks with the same (rank % turns), i.e. one rank per device at LNC2, so the
    # host-to-device copies of a turn spread over every device's link.
    conc = max(1, min(world_size, int(os.environ.get("LINGBOT_LOAD_CONCURRENCY", "1"))))
    n_turns = -(-world_size // conc)
    for turn in range(n_turns):
        if global_rank % n_turns == turn:
            if pth_path is not None:
                config = WanModelFast.load_config(checkpoint_dir, subfolder=subfolder)
                with torch.device("meta"):
                    model = WanModelFast.from_config(
                        config, local_attn_size=local_attn_size, sink_size=sink_size)
                full_sd = torch.load(
                    pth_path, map_location="cpu", weights_only=True, mmap=True)
            else:
                cpu_model = WanModelFast.from_pretrained(
                    checkpoint_dir, subfolder=subfolder, torch_dtype=dtype,
                    local_attn_size=local_attn_size, sink_size=sink_size)
                full_sd = cpu_model.state_dict()
                with torch.device("meta"):
                    model = WanModelFast.from_config(
                        cpu_model.config, local_attn_size=local_attn_size,
                        sink_size=sink_size)
                del cpu_model

            model = model.to(dtype)

            rebuild_wan_model_fast_for_tp(model, tp_degree)
            sharded_sd = shard_wan_model_fast_state_dict(full_sd, tp_rank, tp_degree)
            del full_sd

            model = model.to_empty(device="cpu")

            # self.freqs isn't a registered buffer, so to_empty() leaves it
            # uninitialized -- recompute before load_state_dict or every
            # attention call's rotary embedding reads garbage.
            d = model.dim // model.num_heads
            model.freqs = torch.cat([
                rope_params(MAX_LATENT_FRAMES, d - 4 * (d // 6)),
                rope_params(MAX_LATENT_FRAMES, 2 * (d // 6)),
                rope_params(MAX_LATENT_FRAMES, 2 * (d // 6)),
            ], dim=1)

            model.load_state_dict(sharded_sd, strict=True)
            # Free this rank's CPU shard now rather than when the function returns: with
            # turns, every finished rank used to keep its copy alive until the last turn.
            del sharded_sd
            # Eager device placement only safe when actually sharded small;
            # at tp_degree<=1 forward()'s per-block loop streams it instead.
            model = model.to(dtype=dtype)
            if tp_degree > 1:
                model = model.to(device=device)
            model.eval()
            model.requires_grad_(False)
        if dist.is_initialized():
            dist.barrier()
    return model

