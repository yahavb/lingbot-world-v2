"""Attention module with NKI kernel support for Neuron.

When running on Neuron, uses NKI kernels from kernels/ directory.
Falls back to torch.nn.functional.scaled_dot_product_attention otherwise.
"""
import math
import warnings
import os

import torch
import torch.nn.functional as F

# Flash attention flags — disabled for Neuron
FLASH_ATTN_3_AVAILABLE = False
FLASH_ATTN_2_AVAILABLE = False


# ─── NKI Kernel Loading ─────────────────────────────────────────────────────
# ATTENTION_BACKEND selects which kernels are imported AT ALL:
#   "repo" (default) -- kernels/cross_attention.py + kernels/self_attention_nst.py
#   "cte"            -- ONLY nkilib.core.attention.attention_cte; the repo
#                       kernels are not imported, so nothing else is loaded
#   "sdpa"           -- no NKI at all, F.scaled_dot_product_attention only
#
# Why "cte" is not merely a swap: the NST self-attention path IGNORES attn_mask
# (see _nki_self_attention_nst -- it takes no mask and passes
# actual_seqlen_k = the full cache length), so the [attn_lo, attn_hi) KV window
# that model_fast.py builds is silently discarded whenever that kernel is
# active. attention_cte takes bound_min/bound_max, a per-query KV range, which
# is exactly that window -- so this backend restores masking the repo path drops.
ATTENTION_BACKEND = os.environ.get("ATTENTION_BACKEND", "repo").lower()
USE_NKI_KERNELS = os.environ.get("USE_NKI_KERNELS", "1") == "1"
if ATTENTION_BACKEND == "sdpa":
    USE_NKI_KERNELS = False

_attention_cte = None
_CTE_AVAILABLE = False
_CONST_MAX_AG_AVAILABLE = False
# attention_cte loaded ONLY to serve the hot layers under the const_max_ag backend.
# Separate from _CTE_AVAILABLE, which would reroute all attention including cross.
_CTE_FOR_EXACT_AVAILABLE = False

_nki_cross_attn = None
_nki_self_attn_nst = None
_NKI_CROSS_AVAILABLE = False
_NKI_SELF_NST_AVAILABLE = False

# This module is imported once per rank, so an unconditional success print
# produced one line per rank per kernel -- 64 interleaved lines at 32 ranks,
# which buries everything else in the log. Successes are identical on every
# rank, so rank 0 speaks for all of them. FAILURES stay unconditional: a
# kernel that loads on some ranks and not others is exactly the asymmetry
# worth seeing per-rank.
_RANK = int(os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or 0)


def _log_kernel_ok(msg):
    if _RANK == 0:
        print(msg)


if ATTENTION_BACKEND == "cte":
    # Deliberately does NOT import kernels/cross_attention.py or
    # kernels/self_attention_nst.py, so this is the only kernel loaded.
    try:
        from torch_neuronx.nki_hop import wrap_nki
        # nkilib.core.attention.attention_cte is a MODULE that contains a
        # function of the same name, so `from nkilib.core.attention import
        # attention_cte` binds the module and wrap_nki then fails with
        # "module ... has no attribute 'func'". Import from the submodule and
        # assert callable, so a future layout change fails on this line with a
        # clear message instead of inside the kernel wrapper.
        from nkilib.core.attention.attention_cte import attention_cte as _cte_kernel
        assert callable(_cte_kernel), (
            f"expected a callable attention_cte, got {type(_cte_kernel)}")
        _attention_cte = wrap_nki(_cte_kernel)
        _CTE_AVAILABLE = True
        _log_kernel_ok("[attention.py] kernel: attention_cte (only)")
    except Exception as e:
        print(f"[attention.py] attention_cte: FAILED rank={_RANK} ({e})")
        raise RuntimeError(
            "ATTENTION_BACKEND=cte was requested but attention_cte could not be "
            "loaded. Failing loudly rather than silently falling back to SDPA, "
            "which would look like a working run at a very different speed."
        ) from e
elif ATTENTION_BACKEND == "const_max_ag":
    # Fused all-gather + constant-max attention, ported from the reference implementation. It does the CP
    # all-gather INSIDE the kernel and writes the gathered chunk straight into the cache
    # buffers as must-alias outputs, replacing _gather_w + copy_ + cat + attention with
    # one call. Their measurement: 9.8% faster per DiT forward than attention_cte at
    # TP8xCP4, res 768.
    #
    # Loud on failure, like the cte branch: a silent fall back to SDPA would look like a
    # working run at a completely different speed.
    try:
        from kernels.const_max_ag import (
            ATTENTION_TILE_SIZE,
            ag_const_max_attention,
            build_const_max_ag_kernels,
        )
        _CONST_MAX_AG_AVAILABLE = True
        _log_kernel_ok("[attention.py] kernel: const_max_ag (fused AG, only)")
        # ALSO load attention_cte, for the hot layers that must not use const-max.
        #
        # SDPA is not a usable fallback at this resolution: it materialises the whole
        # [B, N, Sq, Sk] score matrix, and with 7 hot layers the first denoise forward
        # died on a single 1.270 GB tensor allocation, 52 times across ranks
        #   TDRV:log_dev_mem  Failed to allocate 1.270GB (usage: tensors) on ND 7:NC 2
        # const_max_ag never materialises scores, so the memory only appeared when the
        # exact-attention layers did. This is why the reference implementation's knob is FLASH_ATTN_LAYERS:
        # flash/cte is exact AND streams the KV, so its activation does not scale with
        # Sq*Sk.
        #
        # Kept in its OWN flag rather than setting _CTE_AVAILABLE: that flag makes
        # attention() route EVERYTHING through cte, including every cross-attention call
        # on all 40 layers, which is a much broader change than routing the hot layers.
        try:
            from torch_neuronx.nki_hop import wrap_nki as _wrap_nki_hot
            from nkilib.core.attention.attention_cte import attention_cte as _cte_hot
            assert callable(_cte_hot), f"expected a callable, got {type(_cte_hot)}"
            # Bind the SAME module global _cte_attention() already calls, so the
            # wrapper (pre-scaled q, 128/512 sequence padding, bound_min/bound_max)
            # is reused rather than duplicated. It is None on this backend otherwise.
            _attention_cte = _wrap_nki_hot(_cte_hot)
            _CTE_FOR_EXACT_AVAILABLE = True
            _log_kernel_ok("[attention.py] kernel: attention_cte loaded for exact "
                           "(hot) layers alongside const_max_ag")
        except Exception as e:
            print(f"[attention.py] attention_cte for hot layers: FAILED rank={_RANK} "
                  f"({e}) -- hot layers fall back to SDPA, which OOMs at 768*1152")
    except Exception as e:
        print(f"[attention.py] const_max_ag: FAILED rank={_RANK} ({e})")
        raise RuntimeError(
            "ATTENTION_BACKEND=const_max_ag was requested but the kernel could not be "
            "loaded. Check the container's NKI surface: it needs nki.jit, kernel[LNC] "
            "indexing, nl.shared_hbm and nisa.dma_copy(dge_mode=swdge)."
        ) from e
elif USE_NKI_KERNELS:
    try:
        from torch_neuronx.nki_hop import wrap_nki
        from kernels.cross_attention import wan_cross_attn
        _nki_cross_attn = wrap_nki(wan_cross_attn)
        _NKI_CROSS_AVAILABLE = True
        _log_kernel_ok("[attention.py] NKI cross_attention kernel: LOADED")
    except Exception as e:
        print(f"[attention.py] NKI cross_attention kernel: FAILED rank={_RANK} ({e})")

    # NST self-attention (no mask/identity needed, handles actual_seqlen_k)
    try:
        from kernels.self_attention_nst import wan_flash_self_attn
        _nki_self_attn_nst = wan_flash_self_attn  # already @wrap_nki decorated
        _NKI_SELF_NST_AVAILABLE = True
        _log_kernel_ok("[attention.py] NKI self_attention NST kernel: LOADED")
    except Exception as e:
        print(f"[attention.py] NKI self_attention NST kernel: FAILED rank={_RANK} ({e})")

if ATTENTION_BACKEND != "cte":
    _log_kernel_ok(f"[attention.py] backend={ATTENTION_BACKEND} "
                   f"cross={_NKI_CROSS_AVAILABLE} self_nst={_NKI_SELF_NST_AVAILABLE}")



_AG_KERNELS = None


def const_max_ag_available():
    return _CONST_MAX_AG_AVAILABLE


# ── HOT-LAYER ATTENTION: FA2 all-gather ─────────────────────────────────────
# LINGBOT_HOT_ATTN=fa2_ag (default with const_max_ag) runs the FLASH_ATTN_LAYERS through the fused
# all-gather FlashAttention-2 kernel (kernels/fa2_ag.py): exact online softmax, the
# CP gather of the new chunk inside the kernel, and the same N-major chunk cache as the const-max
# layers -- so no _gather_w, no torch.cat, no layout transposes in the graph.
# LINGBOT_HOT_ATTN=cte keeps the previous path (CP gather + exact_attention_chunks/attention_cte).
_HOT_ATTN = os.environ.get("LINGBOT_HOT_ATTN", "fa2_ag").strip().lower()
_FA2_AG_AVAILABLE = False
_FA2_KERNELS = None
if ATTENTION_BACKEND == "const_max_ag" and _HOT_ATTN == "fa2_ag":
    try:
        from kernels.fa2_ag import ag_fa2_attention, build_fa2_ag_kernels
        _FA2_AG_AVAILABLE = True
        _log_kernel_ok("[attention.py] kernel: fa2_ag (fused AG FlashAttention-2) for hot layers")
    except Exception as e:
        print(f"[attention.py] fa2_ag: FAILED rank={_RANK} ({e}) -- hot layers use attention_cte")


def fa2_ag_available():
    return _FA2_AG_AVAILABLE


def init_fa2_ag(cp_replica_groups, kv_chunk_tokens):
    """Build the four fixed-arity FA2 AG callables (same contract as init_const_max_ag)."""
    global _FA2_KERNELS
    import math as _math
    import nki.collectives as ncc
    rg = ncc.ReplicaGroup([list(g) for g in cp_replica_groups])
    _FA2_KERNELS = build_fa2_ag_kernels(rg, 1.0 / _math.sqrt(ATTENTION_TILE_SIZE), kv_chunk_tokens)
    return _FA2_KERNELS


def fa2_ag_attention(q, k_shard, v_shard, k_chunks, v_chunks):
    """Fused AG FA2 over `(leading..., destination)` N-major chunks; mirrors const_max_ag_attention."""
    if _FA2_KERNELS is None:
        raise RuntimeError("call init_fa2_ag() before fa2_ag_attention()")
    lead = len(k_chunks) - 1
    return ag_fa2_attention(
        q, k_shard, v_shard, tuple(k_chunks[:-1]), tuple(v_chunks[:-1]),
        _FA2_KERNELS[lead], k_chunks[-1], v_chunks[-1])


def init_const_max_ag(cp_replica_groups, kv_chunk_tokens, softmax_max=15.0):
    """Build the four fixed-arity kernel callables. Call once, after the mesh exists.

    the reference implementation setup.py:167-178. The replica group, softmax_max, scale and chunk size are
    opaque constants closed over eagerly, because they cannot cross the Torch->NKI
    boundary per call.

    softmax_max is an ASSUMED bound, not measured -- see kernels/const_max_ag.py.
    """
    global _AG_KERNELS
    if not _CONST_MAX_AG_AVAILABLE:
        raise RuntimeError("init_const_max_ag called without the kernel loaded")
    # Alignment is NO LONGER REQUIRED. the updated kernel made the kernel mask a
    # sub-128 KV tail tile via last_tile_extent, so a non-aligned chunk width works and 720p
    # became their default. This used to raise, which is what forced SIZE=768*1152 here.
    # Kept as a one-line notice rather than deleted: if a non-aligned width ever produces
    # wrong pixels, this is the assumption to re-test first.
    if kv_chunk_tokens % ATTENTION_TILE_SIZE != 0 and _RANK == 0:
        print(f"[attention.py] const_max_ag: kv_chunk={kv_chunk_tokens} is not "
              f"{ATTENTION_TILE_SIZE}-aligned (remainder "
              f"{kv_chunk_tokens % ATTENTION_TILE_SIZE}) -- relying on the tail-tile mask "
              f"from the updated kernel")
    import math as _math

    import nki.collectives as ncc
    rg = ncc.ReplicaGroup([list(g) for g in cp_replica_groups])
    _AG_KERNELS = build_const_max_ag_kernels(
        rg, softmax_max, 1.0 / _math.sqrt(ATTENTION_TILE_SIZE), kv_chunk_tokens)
    return _AG_KERNELS


def const_max_ag_attention(q, k_shard, v_shard, k_chunks, v_chunks):
    """Fused AG attention over `(leading..., destination)` chunks.

    `k_chunks`/`v_chunks` are the ordered chunk buffers from chunked_kv_cache; the LAST
    is the destination the kernel gathers into, the rest are leading full chunks. Returns
    attention output plus the must-alias destination buffers.
    """
    if _AG_KERNELS is None:
        raise RuntimeError("call init_const_max_ag() before const_max_ag_attention()")
    lead = len(k_chunks) - 1
    if lead not in _AG_KERNELS:
        raise ValueError(
            f"const_max_ag has fixed wrappers for 0..3 leading chunks; got {lead}. "
            f"Lists cannot cross the Torch->NKI boundary, so total chunks must be <= 4: "
            f"set local_attn_size / chunk_size <= 4 (currently gives {lead + 1}).")
    return ag_const_max_attention(
        q, k_shard, v_shard, tuple(k_chunks[:-1]), tuple(v_chunks[:-1]),
        _AG_KERNELS[lead], k_chunks[-1], v_chunks[-1])


# Identity matrix buffer (created once, reused)
_identity_matrix = None


def _get_identity(device, dtype):
    """Get or create 128x128 identity matrix for NKI transpose trick."""
    global _identity_matrix
    if _identity_matrix is None or _identity_matrix.device != device:
        _identity_matrix = torch.eye(128, dtype=dtype, device=device)
    return _identity_matrix


def _nki_cross_attention(q, k, v, dtype=torch.bfloat16):
    """Run cross-attention using NKI kernel.
    
    Input shapes: q [B, L1, n, d], k [B, L2, n, d], v [B, L2, n, d]
    Output shape: [B, L1, n, d]
    """
    b, l1, n, d = q.shape
    l2 = k.shape[1]
    
    # Kernel expects: q(bs, d, seq_q), k(bs, d, seq_k), v(bs, seq_k, d)
    # where bs = num_heads (we process B=1, heads as batch)
    assert b == 1, "NKI kernels only support batch_size=1"
    
    q_nki = q[0].permute(1, 2, 0).contiguous()   # [n, d, L1]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [n, d, L2]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [n, L2, d]
    
    # Pad seqlen_q to multiple of 128
    P = 128
    pad_q = (P - l1 % P) % P
    if pad_q > 0:
        q_nki = F.pad(q_nki, (0, pad_q))
    
    identity = _get_identity(q.device, dtype)
    softmax_scale = 1.0 / math.sqrt(d)
    
    # Call NKI kernel
    out_nki = _nki_cross_attn(q_nki, k_nki, v_nki, identity, softmax_scale=softmax_scale)
    
    # Output: [seqlen_q_padded, n, d] → slice → [1, L1, n, d]
    out = out_nki[:l1].unsqueeze(0)
    return out


def _nki_self_attention_nst(q, k, v, dtype=torch.bfloat16):
    """Run self-attention using NST kernel (no mask/identity needed).

    Input shapes: q [B, L1, n, d], k [B, L2, n, d], v [B, L2, n, d]
    Output shape: [B, L1, n, d]
    """
    b, l1, n, d = q.shape
    assert b == 1, "NKI kernels only support batch_size=1"

    q_nki = q[0].permute(1, 2, 0).contiguous()   # [n, d, L1]
    k_nki = k[0].permute(1, 2, 0).contiguous()   # [n, d, L2]
    v_nki = v[0].permute(1, 0, 2).contiguous()   # [n, L2, d]
    
    l2 = k_nki.shape[2] 
    P = 128
    SEQLEN_MULT = 512  # NST kernel uses 512 alignment for K
    pad_q = (P - l1 % P) % P
    pad_k = (SEQLEN_MULT - l2 % SEQLEN_MULT) % SEQLEN_MULT    
    if pad_q > 0:
        q_nki = F.pad(q_nki, (0, pad_q))
    if pad_k > 0:
        k_nki = F.pad(k_nki, (0, pad_k))
        v_nki = F.pad(v_nki, (0, 0, 0, pad_k))

    softmax_scale = 1.0 / math.sqrt(d)
    out_nki = _nki_self_attn_nst(
        q_nki, k_nki, v_nki,
        softmax_scale=softmax_scale,
        actual_seqlen_k=l2,    
        use_dynamic_loop=False)

    out = out_nki[:l1].unsqueeze(0)
    return out



def _cte_attention(q, k, v, kv_lo=None, kv_hi=None, dtype=torch.bfloat16):
    """Self/cross attention via nkilib attention_cte.

    Input  [B, Lq, n, d] / [B, Lkv, n, d]; output [B, Lq, n, d].

    attention_cte folds heads into the batch dim ("For MHA attention, the heads
    can be included as part of the batch dimension"), so [1, L, n, d] -> [n, L, d]
    and batch_size = n. tp_q/tp_k True means (batch, seqlen, d); tp_out False
    returns (batch, seqlen, d).

    kv_lo/kv_hi carry LingBot's [attn_lo, attn_hi) KV-cache window as
    bound_min/bound_max -- a per-query KV range, "Query position i attends only
    to KV positions j where bound_min <= j < bound_max". causal_mask is False
    because this attention is NOT token-causal: causality lives in what has been
    written to the cache, and within the window attention is full (model_fast.py
    builds a range mask, not a triangular one).
    """
    b, lq, n, d = q.shape
    assert b == 1, "attention_cte path assumes batch_size=1 (heads occupy batch)"
    lkv = k.shape[1]

    # PRE-SCALE Q, and pass scale=1.0. The kernel asserts
    #   "range_select path doesn't support scale != 1.0"
    # ([NCC_INKI016]) whenever bound_min/bound_max, sliding_window or prefix
    # caching are in play -- its docstring says "q can be scaled before calling
    # the kernel". Passing both a scale and bounds fails to COMPILE, on every
    # rank, which is exactly how the first working-CP run died. Scale in fp32
    # before the cast so the multiply does not round in bf16.
    softmax_scale = 1.0 / math.sqrt(d)
    q_c = (q[0].float() * softmax_scale).permute(1, 0, 2).contiguous().to(dtype)  # [n, Lq, d]
    k_c = k[0].permute(1, 0, 2).contiguous().to(dtype)   # [n, Lkv, d]
    v_c = v[0].permute(1, 0, 2).contiguous().to(dtype)   # [n, Lkv, d]

    # PAD THE SEQUENCE AXES. The kernel tiles queries in 128s and the
    # range_select path walks the full padded extent, so a ragged Lq reads past
    # the end: at Lq=377 it emitted
    #   [NCC_IBIR243] Access pattern out of bounds. Pattern: [[1,128],[128,3]]
    # i.e. 3x128 = 384 against 377 rows. Its own docs give "seqlen_q must be a
    # multiple of 128" for position_bias; range_select wants the same. K/V get
    # 512 alignment, matching what this repo's own NST wrapper already does.
    #
    # Padding is safe in both directions: padded QUERY rows are sliced off the
    # output, and padded KV columns sit outside [bound_min, bound_max) so the
    # kernel never reads them. 377 -> 384 (29 rows x 13 width at cp4).
    Q_ALIGN, KV_ALIGN = 128, 512
    pad_q = (Q_ALIGN - lq % Q_ALIGN) % Q_ALIGN
    pad_kv = (KV_ALIGN - lkv % KV_ALIGN) % KV_ALIGN
    if pad_q:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))               # pad the seqlen axis
    if pad_kv:
        k_c = F.pad(k_c, (0, 0, 0, pad_kv))
        v_c = F.pad(v_c, (0, 0, 0, pad_kv))

    kwargs = {}
    if kv_lo is not None and kv_hi is not None:
        # (batch, seqlen_q, 1), same range for every query in this chunk, and
        # sized to the PADDED Lq so the kernel's per-tile walk stays in bounds.
        lq_p = lq + pad_q
        lo = torch.as_tensor(kv_lo, device=q.device).reshape(1, 1, 1).expand(n, lq_p, 1)
        hi = torch.as_tensor(kv_hi, device=q.device).reshape(1, 1, 1).expand(n, lq_p, 1)
        kwargs.update(bound_min=lo.contiguous(), bound_max=hi.contiguous())

    out = _attention_cte(
        q_c, k_c, v_c,
        scale=1.0,          # q is pre-scaled above; see the note there
        causal_mask=False,
        tp_q=True, tp_k=True, tp_out=False,
        **kwargs)
    if isinstance(out, tuple):      # cache_softmax not requested, but be safe
        out = out[0]
    out = out[:, :lq]                                    # drop padded queries
    return out.permute(1, 0, 2).unsqueeze(0)             # [1, Lq, n, d]


def _cte_attention_chunks(q, k_chunks, v_chunks, dtype=torch.bfloat16):
    """_cte_attention over a LIST of KV chunks, without torch.cat or F.pad.

    Same math and the same zero padding as _cte_attention(q, cat(k), cat(v)). The difference
    is only in how K/V are built: each chunk is permuted straight into its slice of one
    preallocated, already-padded [n, Lkv_padded, d] buffer. At cp8 the cat + permute + pad form
    lowered (inside the compiled hot block) to spill/reload DMAs at ~1.7 GB/s and cost ~23 ms
    per KV chunk; the same form at cp4 ran at ~40 GB/s. LINGBOT_HOT_KV_CAT=1 restores the cat.
    """
    b, lq, n, d = q.shape
    assert b == 1, "attention_cte path assumes batch_size=1 (heads occupy batch)"
    lkv = sum(int(c.shape[1]) for c in k_chunks)
    Q_ALIGN, KV_ALIGN = 128, 512
    pad_q = (Q_ALIGN - lq % Q_ALIGN) % Q_ALIGN
    lkv_p = lkv + (KV_ALIGN - lkv % KV_ALIGN) % KV_ALIGN
    softmax_scale = 1.0 / math.sqrt(d)
    q_c = (q[0].float() * softmax_scale).permute(1, 0, 2).contiguous().to(dtype)
    if pad_q:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    k_c = q.new_zeros((n, lkv_p, d), dtype=dtype)
    v_c = q.new_zeros((n, lkv_p, d), dtype=dtype)
    off = 0
    for kc, vc in zip(k_chunks, v_chunks):
        L = int(kc.shape[1])
        k_c[:, off:off + L] = kc[0].permute(1, 0, 2).to(dtype)
        v_c[:, off:off + L] = vc[0].permute(1, 0, 2).to(dtype)
        off += L
    out = _attention_cte(q_c, k_c, v_c, scale=1.0, causal_mask=False,
                         tp_q=True, tp_k=True, tp_out=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, :lq]
    return out.permute(1, 0, 2).unsqueeze(0)


# LINGBOT_CROSS_XLA=1: cross-attention (queries x 512 context tokens) as plain matmul / softmax /
# matmul in fp32 inside the block graph instead of one attention_cte call. At TP4 x CP16 a rank
# has 270 queries and the cte call is 0.35 ms of a 3.15 ms block for ~0.5 GFLOP of work: the
# kernel's fixed cost. Same math, full softmax, no padding; fp32 scores and probabilities.
_CROSS_XLA = os.environ.get("LINGBOT_CROSS_XLA", "0") == "1"


def _xla_cross_attention(q, k, v, dtype=torch.bfloat16):
    """[1, Lq, n, d] x [1, Lk, n, d] -> [1, Lq, n, d]; heads as batch, fp32 throughout."""
    d = q.shape[-1]
    qh = q[0].permute(1, 0, 2).float() * (1.0 / math.sqrt(d))     # [n, Lq, d]
    kh = k[0].permute(1, 2, 0).float()                             # [n, d, Lk]
    vh = v[0].permute(1, 0, 2).float()                             # [n, Lk, d]
    p = torch.softmax(torch.matmul(qh, kh), dim=-1)                # [n, Lq, Lk]
    return torch.matmul(p, vh).permute(1, 0, 2).unsqueeze(0).to(dtype)


def cte_merge_enabled():
    """LINGBOT_CTE_MERGE=1: N-major KV chunks + one attention_cte call per chunk, merged (see
    _cte_attention_chunks_merged). Read by the cache allocators and by the KV write."""
    return os.environ.get("LINGBOT_CTE_MERGE", "0") == "1"


# LINGBOT_CROSS_NMAJOR=1: the cross-attention K/V cache is stored in attention_cte's own layout,
# K [n, d, text_len] and V [n, text_len, d], written once per generation (the first cross-attention
# call). Otherwise every call permutes the [1, text_len, n, d] cache into that layout: two strided
# 1.3 MB transposes per block, ~250 us on the critical path of a 3.2 ms block at TP4 x CP16.
# The same values reach the kernel (data movement only).
_CROSS_NMAJOR = os.environ.get("LINGBOT_CROSS_NMAJOR", "0") == "1"


def cross_nmajor_enabled():
    return _CROSS_NMAJOR and (_CTE_AVAILABLE or _CTE_FOR_EXACT_AVAILABLE)


def cross_cache_shapes(batch, text_len, heads, head_dim):
    """(K shape, V shape) of one layer's cross-attention cache."""
    if cross_nmajor_enabled():
        assert batch == 1, "N-major cross cache assumes batch 1"
        return (heads, head_dim, text_len), (heads, text_len, head_dim)
    return (batch, text_len, heads, head_dim), (batch, text_len, heads, head_dim)


def cte_cross_attention_nmajor(q, k_ndl, v_nld, dtype=torch.bfloat16):
    """[1, Lq, n, d] queries against the N-major cross cache -> [1, Lq, n, d]. Same kernel call as
    _cte_attention (pre-scaled q padded to 128 rows, no bounds), with K already transposed
    (tp_k=False) instead of permuted from [1, L, n, d] on every call."""
    q_c, lq = cte_prep_q(q, dtype)
    out = _attention_cte(q_c, k_ndl.to(dtype), v_nld.to(dtype), scale=1.0, causal_mask=False,
                         tp_q=True, tp_k=False, tp_out=False)
    if isinstance(out, tuple):
        out = out[0]
    return out[:, :lq].permute(1, 0, 2).unsqueeze(0)


def cte_prep_q(q, dtype=torch.bfloat16):
    """[1, Lq, n, d] -> (pre-scaled N-major q padded to 128 rows [n, Lq_p, d], Lq)."""
    b, lq, n, d = q.shape
    assert b == 1, "attention_cte path assumes batch_size=1 (heads occupy batch)"
    q_c = (q[0].float() * (1.0 / math.sqrt(d))).permute(1, 0, 2).contiguous().to(dtype)
    pad_q = (128 - lq % 128) % 128
    if pad_q:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    return q_c, lq


def cte_partial(q_c, kc, vc, k_prior=None, v_prior=None):
    """Unnormalized attention (out, row max, row sum) of one KV chunk -- or of two, with the
    second given as the kernel's prefix-caching prior (k_prior/v_prior, all of it used) -- for
    cte_merge. Two chunks in one call is the same softmax over their union, computed by the
    kernel's own online softmax instead of by cte_merge."""
    d = int(q_c.shape[-1])
    tp_k = int(kc.shape[-1]) == d and int(kc.shape[1]) != d
    kw = {}
    if k_prior is not None:
        lp = int(v_prior.shape[1])
        kw = dict(k_prior=k_prior, v_prior=v_prior,
                  prior_used_len=torch.full((1,), lp, dtype=torch.int32, device=q_c.device))
    o, neg_max, s = _attention_cte(q_c, kc, vc, scale=1.0, causal_mask=False,
                                   tp_q=True, tp_k=tp_k, tp_out=False,
                                   cache_softmax=True, skip_output_normalization=True, **kw)
    return o, -neg_max.float(), s.float()


def cte_rows(part, start, length):
    """Rows [start, start + length) of a cte_partial result, with the softmax stats converted from
    the kernel's [n, 128, Lq_p / 128] layout (query = group * 128 + row) to per-query [n, length]."""
    o, m, s = part
    n, g = int(m.shape[0]), int(m.shape[2])
    m = m.transpose(1, 2).reshape(n, g * 128)[:, start:start + length]
    s = s.transpose(1, 2).reshape(n, g * 128)[:, start:start + length]
    return o[:, start:start + length], m, s


def cte_merge_rows(parts, dtype=torch.bfloat16):
    """cte_merge over per-query partials (cte_rows): the same fp32 ops in the same order, so the
    same values element for element. [n, R, d] parts -> [1, R, n, d]."""
    outs = [p[0] for p in parts]
    maxs = [p[1] for p in parts]
    sums = [p[2] for p in parts]
    if len(outs) == 1:
        lam = [1.0 / sums[0]]
    else:
        m_all = maxs[0]
        for m in maxs[1:]:
            m_all = torch.maximum(m_all, m)
        w = [torch.exp(m - m_all) for m in maxs]
        den = w[0] * sums[0]
        for wc, sc in zip(w[1:], sums[1:]):
            den = den + wc * sc
        inv = 1.0 / den
        lam = [wc * inv for wc in w]
    out = lam[0].unsqueeze(-1) * outs[0].float()
    for l_, o in zip(lam[1:], outs[1:]):
        out = out + l_.unsqueeze(-1) * o.float()
    return out.permute(1, 0, 2).unsqueeze(0).to(dtype)


def cte_merge(parts, lq, dtype=torch.bfloat16):
    """Merge cte_partial results exactly as _cte_attention_chunks_merged does."""
    outs = [p[0] for p in parts]
    maxs = [p[1] for p in parts]
    sums = [p[2] for p in parts]
    n, lq_p = int(outs[0].shape[0]), int(outs[0].shape[1])
    if len(outs) == 1:
        lam = [1.0 / sums[0]]
    else:
        m_all = maxs[0]
        for m in maxs[1:]:
            m_all = torch.maximum(m_all, m)
        w = [torch.exp(m - m_all) for m in maxs]
        den = w[0] * sums[0]
        for wc, sc in zip(w[1:], sums[1:]):
            den = den + wc * sc
        inv = 1.0 / den
        lam = [wc * inv for wc in w]
    lam = [l_.transpose(1, 2).reshape(n, lq_p, 1) for l_ in lam]
    out = lam[0] * outs[0].float()
    for l_, o in zip(lam[1:], outs[1:]):
        out = out + l_ * o.float()
    return out[:, :lq].permute(1, 0, 2).unsqueeze(0).to(dtype)


def _cte_attention_chunks_merged(q, k_chunks, v_chunks, dtype=torch.bfloat16):
    """Exact attention over N-MAJOR KV chunks [n, L_c, d], without assembling them.

    One attention_cte call per chunk returns the UNNORMALIZED output exp(s - m_c) @ V_c with its
    row max m_c and denominator S_c (cache_softmax + skip_output_normalization, the ring-attention
    contract); the chunks are merged in fp32 with the usual running-max rescale:
        out = sum_c exp(m_c - M) O_c / sum_c exp(m_c - M) S_c.
    Versus _cte_attention_chunks this drops (a) the per-block copy of every chunk into a zero-filled
    buffer -- on the critical path before the kernel, 1.7 ms of a 10.4 ms block at TP4xCP2, and
    the source of the chunk-1 slowdown -- and (b) the 512-alignment KV padding, whose zero keys
    were attended (no bounds were passed) and added n_pad * exp(-m) to every softmax denominator.
    The kernel handles a ragged last K tile itself; Q is still padded to 128 (cache_softmax tiles
    Q in 128s), and the padded rows are dropped.
    """
    b, lq, n, d = q.shape
    assert b == 1, "attention_cte path assumes batch_size=1 (heads occupy batch)"
    softmax_scale = 1.0 / math.sqrt(d)
    q_c = (q[0].float() * softmax_scale).permute(1, 0, 2).contiguous().to(dtype)  # [n, Lq, d]
    pad_q = (128 - lq % 128) % 128
    if pad_q:
        q_c = F.pad(q_c, (0, 0, 0, pad_q))
    lq_p = lq + pad_q
    outs, maxs, sums = [], [], []
    for kc, vc in zip(k_chunks, v_chunks):
        # K is [n, d, L] (pre-transposed, the const-max cache layout: tp_k=False, no transpose
        # inside the kernel) or [n, L, d] (tp_k=True). V is always [n, L, d].
        tp_k = int(kc.shape[-1]) == d and int(kc.shape[1]) != d
        o, neg_max, s = _attention_cte(q_c, kc, vc, scale=1.0, causal_mask=False,
                                       tp_q=True, tp_k=tp_k, tp_out=False,
                                       cache_softmax=True, skip_output_normalization=True)
        outs.append(o)
        maxs.append(-neg_max.float())     # [n, 128, Lq/128], query = group * 128 + row
        sums.append(s.float())
    if len(outs) == 1:
        lam = [1.0 / sums[0]]
    else:
        # All the softmax bookkeeping on the SMALL stats tensors; the full-size work is then a
        # single weighted sum of the chunk outputs. (A running merge with full-size fp32
        # rescales and a full-size divide lowered to 1.5 ms of a 10.5 ms block.)
        m_all = maxs[0]
        for m in maxs[1:]:
            m_all = torch.maximum(m_all, m)
        w = [torch.exp(m - m_all) for m in maxs]
        den = w[0] * sums[0]
        for wc, sc in zip(w[1:], sums[1:]):
            den = den + wc * sc
        inv = 1.0 / den
        lam = [wc * inv for wc in w]
    lam = [l_.transpose(1, 2).reshape(n, lq_p, 1) for l_ in lam]
    out = lam[0] * outs[0].float()
    for l_, o in zip(lam[1:], outs[1:]):
        out = out + l_ * o.float()
    return out[:, :lq].permute(1, 0, 2).unsqueeze(0).to(dtype)


def exact_attention_chunks(q, k_chunks, v_chunks, dtype=torch.bfloat16):
    """exact_attention over a list of KV chunks (see _cte_attention_chunks). N-major (3-D) chunks
    come from LINGBOT_CTE_MERGE=1 and take the copy-free merged path."""
    if (q.device.type == "neuron" and (_CTE_FOR_EXACT_AVAILABLE or _CTE_AVAILABLE)
            and k_chunks[0].dim() == 3):
        return _cte_attention_chunks_merged(q, list(k_chunks), list(v_chunks), dtype=dtype)
    if (q.device.type == "neuron" and (_CTE_FOR_EXACT_AVAILABLE or _CTE_AVAILABLE)
            and os.environ.get("LINGBOT_HOT_KV_CAT", "0") != "1"):
        return _cte_attention_chunks(q, list(k_chunks), list(v_chunks), dtype=dtype)
    return exact_attention(q, torch.cat(list(k_chunks), dim=1), torch.cat(list(v_chunks), dim=1),
                           dtype=dtype)


def exact_attention(q, k, v, dtype=torch.bfloat16):
    """Numerically EXACT attention, for layers const-max cannot serve.

    A hot head's true row max can exceed 500 while const_max_ag assumes 15, and
    exp(500 - 15) overflows -- so those layers need the real max. Both paths below
    compute it; they differ only in memory.

    attention_cte when it is loaded, SDPA otherwise. That preference is not about
    speed: SDPA materialises the full [B, N, Sq, Sk] score matrix, and at 768*1152
    with 7 hot layers that was a 1.270 GB allocation which failed on the first
    denoise forward. cte streams the KV, so its activation does not scale with
    Sq*Sk.

    No kv_lo/kv_hi: the chunked cache writes every token it presents, so there is no
    unwritten tail to mask -- the same reason the SDPA call here passes no attn_mask.
    """
    if q.device.type == "neuron" and (_CTE_FOR_EXACT_AVAILABLE or _CTE_AVAILABLE):
        return _cte_attention(q, k, v, kv_lo=None, kv_hi=None, dtype=dtype)
    return attention(q, k, v, dtype=dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
    attn_mask=None,
    is_cross_attn=False,
    kv_lo=None,
    kv_hi=None,
):
    """Unified attention function with NKI kernel support.
    
    Args:
        q, k, v: [B, seq, num_heads, head_dim]
        is_cross_attn: If True,  use cross-attention kernel (small seq_k).
                       If False, use self-attention kernel (large seq_k).
        attn_mask: bool mask broadcastable to (B, H, Lq, Lk), True = attend.
        Only wired through the scaled_dot_product_attention path (Neuron).
    """
    # Try NKI kernels first (Neuron device)
    if q.device.type == "neuron":
        if is_cross_attn and _CROSS_XLA:
            return _xla_cross_attention(q, k, v, dtype=dtype)
        if _CTE_AVAILABLE:
            # One kernel for both self and cross attention. Cross-attention has
            # no cache window, so it passes no bounds.
            return _cte_attention(
                q, k, v,
                kv_lo=None if is_cross_attn else kv_lo,
                kv_hi=None if is_cross_attn else kv_hi,
                dtype=dtype)
        if is_cross_attn and _NKI_CROSS_AVAILABLE:
            return _nki_cross_attention(q, k, v, dtype=dtype)
        elif not is_cross_attn and _NKI_SELF_NST_AVAILABLE:
            return _nki_self_attention_nst(q, k, v, dtype=dtype)
    
    # Fallback: scaled_dot_product_attention
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. '
            'It can have a significant impact on performance.'
        )
    
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    
    # set attn_mask=None as before
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
    
    out = out.transpose(1, 2).contiguous()
    return out

# Alias for backward compatibility (wan/modules/__init__.py imports flash_attention)
flash_attention = attention
