"""The nki-library branch's attention_cte (grouped K/V loads, k_blocks) against origin/main's, on one
logical core: bitwise comparison of every output and the time per call.
    NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=/work/nkl_attn/src python3 kiro/nkl_cte_cmp.py
/work/nkl_attn/src holds the branch tree plus origin/main's kernel copied in as
nkilib_src/nkilib/core/attention/attention_cte_ref.py (same utils, so both load in one process).
CASES=perf|kblocks|all: the branch kernel vs main on CASES (default all).
CASES=sweep: every shape of SWEEP against main for each VTAGS kernel, a copy of the branch module with
  other load-grouping constants (attention_cte_<tag>.py next to attention_cte.py, made by the job: the
  NKI tracer does not see module attributes patched after import).
CASES=one ONE_CASE=<name> ONE_TAG=<main|tag>: one kernel on one shape, REPS calls (for a profile).
CASES=table: every shape of SWEEP, the branch kernel against main: ms per call (TRIALS alternating blocks of
  REPS calls, the fastest block, and main's spread over its blocks), K + V copies per head, and for every
  output the elements that differ, their largest difference in absolute terms and in units in the last
  place, plus both kernels' error against an fp32 reference on the host; then k_blocks against the plain
  layout on CASES. One JSON object per line ([nkl_table] / [nkl_table_kb])."""
import importlib
import json
import math
import os
import time

import torch
import torch_neuronx  # noqa: F401
from torch_neuronx.nki_hop import wrap_nki

from nkilib_src.nkilib.core.attention.attention_cte import attention_cte as k_new
from nkilib_src.nkilib.core.attention.attention_cte_ref import attention_cte as k_ref

A_NEW, A_REF = wrap_nki(k_new), wrap_nki(k_ref)
dev = torch.device("neuron")
REPS = int(os.environ.get("REPS", "30"))
TRIALS = int(os.environ.get("TRIALS", "3"))
g = torch.Generator().manual_seed(5)


def r(*s, scale=1.0):
    return (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)


# name, bs, bs_kv, lq, d, prior, prior_used, keys, causal, ring (cache_softmax + skip_output_normalization)
CASES = [
    ("dit self 4320 keys", 10, 10, 384, 128, 0, 0, 4320, False, True),
    ("dit self prior 4320 + 4320", 10, 10, 384, 128, 4320, 4320, 4320, False, True),
    ("dit cross 512 keys", 10, 10, 384, 128, 0, 0, 512, False, False),
    ("prefill causal 4096", 8, 8, 4096, 128, 0, 0, 4096, True, False),
    ("prefill causal 8192 gqa4", 8, 2, 8192, 128, 0, 0, 8192, True, False),
    ("long 2048 x 16384 keys", 4, 4, 2048, 128, 0, 0, 16384, False, False),
    ("d64 1024 x 8192 keys", 8, 8, 1024, 64, 0, 0, 8192, False, False),
    ("prior 4096 (used 3000) + 2048", 4, 4, 512, 128, 4096, 3000, 2048, False, True),
]
SWEEP = CASES + [
    ("short q 128 x 8192 bs16", 16, 16, 128, 128, 0, 0, 8192, False, False),
    ("nc 4096 x 4096", 8, 8, 4096, 128, 0, 0, 4096, False, False),
    ("nc 8192 x 8192 bs2", 2, 2, 8192, 128, 0, 0, 8192, False, False),
    ("causal 1024", 8, 8, 1024, 128, 0, 0, 1024, True, False),
    ("causal 2048", 8, 8, 2048, 128, 0, 0, 2048, True, False),
    ("causal 2560", 8, 8, 2560, 128, 0, 0, 2560, True, False),
    ("causal 3072", 8, 8, 3072, 128, 0, 0, 3072, True, False),
    ("causal 3584", 8, 8, 3584, 128, 0, 0, 3584, True, False),
    ("causal 4096 bs1", 1, 1, 4096, 128, 0, 0, 4096, True, False),
    ("causal 4096 bs2", 2, 2, 4096, 128, 0, 0, 4096, True, False),
    ("causal 4096 d64", 8, 8, 4096, 64, 0, 0, 4096, True, False),
    ("causal 16384 bs2", 2, 2, 16384, 128, 0, 0, 16384, True, False),
    ("causal prior 2048 + 2048", 4, 4, 2048, 128, 2048, 2048, 2048, True, False),
]


def shape_info(case):
    """Q groups per head, flash sections, and K / V tile loads per head of the whole call."""
    name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
    prior_padded = -(-lp // 512) * 512 if lp else 0
    total = prior_padded + lk
    sections = 1 if total <= 10240 else math.ceil(total / 8192)
    k_tiles = math.ceil(lp / 512) + math.ceil(lk / 512)
    v_tiles = math.ceil(lp / 128) + math.ceil(lk / 128)
    return f"qgrp={math.ceil(lq / 128)} sec={sections} kt={k_tiles} vt={v_tiles}"


def blocked(k, nb):
    bs_kv, d, seqlen = k.shape
    return k.view(bs_kv, d, nb, seqlen // nb).permute(2, 0, 1, 3).contiguous().view(bs_kv, d, seqlen)


def make(case):
    name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
    q = r(bs, lq, d, scale=1.0 / d ** 0.5)
    k, v = r(bs_kv, d, lk), r(bs_kv, lk, d)
    kp, vp = (r(bs_kv, d, lp), r(bs_kv, lp, d)) if lp else (None, None)
    return q, k, v, kp, vp


def caller(kern, case, k_blocks=1):
    name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case

    def f(q, k, v, kp=None, vp=None):
        kw = {}
        if kp is not None:
            kw = dict(k_prior=kp, v_prior=vp, prior_used_len=torch.full((1,), pul, dtype=torch.int32, device=q.device))
        if ring:
            kw.update(cache_softmax=True, skip_output_normalization=True)
        if k_blocks > 1:
            kw["k_blocks"] = k_blocks
        out = kern(q, k, v, scale=1.0, causal_mask=causal, tp_q=True, tp_k=False, tp_out=False, **kw)
        return out if isinstance(out, tuple) else (out,)

    torch._dynamo.reset()  # every case compiles the same inner function: no recompile limit
    return torch.compile(f, backend="neuron", fullgraph=True, dynamic=False)


def first(f, args):
    """Outputs of one call, after the compile and 5 warmup calls."""
    with torch.no_grad():
        out = f(*args)
        for _ in range(5):
            f(*args)
        torch.neuron.synchronize()
    return [t.float().cpu() for t in out]


def timed(f, args):
    with torch.no_grad():
        torch.neuron.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            f(*args)
        torch.neuron.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3


def ab(fa, args_a, fb, args_b):
    """Outputs of both, then ms per call of both: TRIALS alternating blocks of REPS calls, the fastest block."""
    oa, ob = first(fa, args_a), first(fb, args_b)
    ta, tb = [], []
    for _ in range(TRIALS):
        ta.append(timed(fa, args_a))
        tb.append(timed(fb, args_b))
    return oa, min(ta), ob, min(tb)


def cmp(a_list, b_list):
    msg = []
    for lbl, a, b in zip(("out", "max", "sum"), a_list, b_list):
        dd = (a - b).abs()
        msg.append(f"{lbl}: {'equal' if torch.equal(a, b) else 'DIFF'} n={int((dd > 0).sum())}/{dd.numel()} "
                   f"max|d|={dd.max().item():.2e}")
    return " | ".join(msg)


def tag_kernel(tag):
    if tag == "main":
        return A_REF
    mod = importlib.import_module(f"nkilib_src.nkilib.core.attention.attention_cte_{tag}")
    return wrap_nki(mod.attention_cte)


def device_args(case):
    q, k, v, kp, vp = make(case)
    return [t.to(dev) for t in (q, k, v) + ((kp, vp) if kp is not None else ())]


mode = os.environ.get("CASES", "all")
if mode in ("perf", "kblocks", "all"):
    for case in CASES:
        args = device_args(case)
        if mode in ("perf", "all"):
            o_ref, t_ref, o_new, t_new = ab(caller(A_REF, case), args, caller(A_NEW, case), args)
            print(f"[nkl_cmp] {case[0]:32s} main {t_ref:7.3f} ms  branch {t_new:7.3f} ms  "
                  f"({(1 - t_new / t_ref) * 100:+5.1f}%)  {cmp(o_ref, o_new)}", flush=True)
        if mode in ("kblocks", "all"):
            nb = 16 if case[7] % 16 == 0 and (case[5] % 16 == 0) else 8
            if case[7] % nb or case[5] % nb:
                continue
            bargs = [args[0], blocked(args[1], nb), args[2]] + ([blocked(args[3], nb), args[4]] if case[5] else [])
            o_new, t_new, o_kb, t_kb = ab(caller(A_NEW, case), args, caller(A_NEW, case, k_blocks=nb), bargs)
            print(f"[nkl_cmp] {case[0]:32s} k_blocks={nb:2d} {t_kb:7.3f} ms vs plain {t_new:7.3f} ms  "
                  f"{cmp(o_new, o_kb)}", flush=True)

if mode == "sweep":
    tags = os.environ.get("VTAGS", "K1V1,K4V8,K2V4f,K1V8").split(",")
    kerns = {t: tag_kernel(t) for t in tags}
    only = os.environ.get("VCASES", "")
    for case in [c for c in SWEEP if not only or c[0] in only.split(",")]:
        args = device_args(case)
        res, mains = [], []
        for t in tags:
            o_ref, t_ref, o_new, t_new = ab(caller(A_REF, case), args, caller(kerns[t], case), args)
            mains.append(t_ref)
            eq = all(torch.equal(a, b) for a, b in zip(o_ref, o_new))
            res.append(f"{t} {t_new:.3f} {(t_new / t_ref - 1) * 100:+.1f}%{'' if eq else '~'}")
        print(f"[nkl_sweep] {case[0]:28s} {shape_info(case):28s} main {min(mains):.3f} "
              f"(spread {(max(mains) / min(mains) - 1) * 100:.1f}%) | " + " | ".join(res), flush=True)

def copies_per_head(case, grouped):
    """K + V dma_copy count per head of the whole call (all flash sections): one per tile, or the runs of
    the branch's load plan (_kv_load_runs over each section's tiles, as the kernel builds them)."""
    from nkilib_src.nkilib.core.attention.attention_cte import (
        _K_LOAD_GROUP_TILES, _K_TILE_SZ, _V_LOAD_GROUP_TILES, _V_TILE_SZ, _kv_load_runs)
    name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
    prior_padded = -(-lp // _K_TILE_SZ) * _K_TILE_SZ if lp else 0
    total = prior_padded + lk
    sec_len = total if total <= 10240 else 8192
    n = 0
    for tile_sz, group in ((_K_TILE_SZ, _K_LOAD_GROUP_TILES), (_V_TILE_SZ, _V_LOAD_GROUP_TILES)):
        for sec0 in range(0, total, sec_len):
            tiles = []
            for start in range(sec0, min(sec0 + sec_len, total), tile_sz):
                src, seqlen, off = (("a", lk, start - prior_padded) if start >= prior_padded else ("p", lp, start))
                valid = min(seqlen - off, tile_sz)
                tiles.append((src, off, valid, tile_sz) if valid > 0 else None)
            if grouped:
                n += len(_kv_load_runs(tiles, group))
            else:
                n += sum(t is not None for t in tiles)
    return n


def ulp(x, dtype):
    """Unit in the last place of each |x| in dtype (bf16 keeps 8 significant bits, fp32 24)."""
    bits = 8 if dtype == torch.bfloat16 else 24
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - (bits - 1))


def diff_stats(a, b, dtype):
    dd = (a - b).abs()
    nd = int((dd > 0).sum())
    out = {"equal": bool(torch.equal(a, b)), "n_diff": nd, "numel": dd.numel(),
           "max_abs": float(dd.max()), "max_ulp": 0}
    if nd:
        mag = torch.maximum(a.abs(), b.abs())
        out["max_ulp"] = int(round(float((dd / ulp(mag, dtype))[dd > 0].max())))
    return out


def reference(case, q, k, v, kp, vp):
    """fp32 attention on the host, [bs, lq, d], with the kernel's masking: the used prior keys are all
    visible, active key j to query i iff j <= i when causal (lq == keys in every causal case)."""
    name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
    rep = bs // bs_kv
    outs = []
    for b in range(bs):
        kb = b // rep
        K, V = k[kb].float(), v[kb].float()
        if lp:
            K = torch.cat([kp[kb].float()[:, :pul], K], dim=1)
            V = torch.cat([vp[kb].float()[:pul], V], dim=0)
        s = q[b].float() @ K
        if causal:
            p0 = pul if lp else 0
            s[:, p0:] = s[:, p0:].masked_fill(torch.ones(lq, lk, dtype=torch.bool).triu(1), float("-inf"))
        outs.append(torch.softmax(s, dim=-1) @ V)
    return torch.stack(outs)


def normalized(case, outs):
    """The attention output [bs, lq, d] of a call: ring calls return it unnormalized with the row sum
    [bs, 128, groups] (query = group * 128 + row)."""
    if not case[9]:
        return outs[0]
    o, s = outs[0], outs[2]
    bs, lq = o.shape[0], o.shape[1]
    s_q = s.transpose(1, 2).reshape(bs, -1)[:, :lq]
    return o / s_q[..., None]


def err(o, ref):
    return {"rel_rms": float((o - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
            "max_abs": float((o - ref).abs().max())}


def ab_trials(fa, args_a, fb, args_b):
    """Outputs (with dtypes) of both, then every block time of both: TRIALS alternating blocks of REPS calls."""
    with torch.no_grad():
        ra, rb = fa(*args_a), fb(*args_b)
        torch.neuron.synchronize()
        for _ in range(5):
            fa(*args_a)
            fb(*args_b)
        torch.neuron.synchronize()
    oa = [(t.float().cpu(), t.dtype) for t in ra]
    ob = [(t.float().cpu(), t.dtype) for t in rb]
    ta, tb = [], []
    for _ in range(TRIALS):
        ta.append(timed(fa, args_a))
        tb.append(timed(fb, args_b))
    return oa, ta, ob, tb


if mode == "table":
    only = os.environ.get("VCASES", "")
    for case in [c for c in SWEEP if not only or c[0] in only.split(",")]:
        name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
        host = make(case)
        args = [t.to(dev) for t in host[:3] + ((host[3], host[4]) if host[3] is not None else ())]
        o_ref, t_ref, o_new, t_new = ab_trials(caller(A_REF, case), args, caller(A_NEW, case), args)
        qgrp = math.ceil(lq / 128)
        routed = causal and qgrp > 24
        row = {"name": name, "bs": bs, "bs_kv": bs_kv, "lq": lq, "d": d, "prior": lp, "prior_used": pul,
               "keys": lk, "causal": causal, "ring": ring, "qgrp": qgrp, "routed": routed,
               "copies_main": copies_per_head(case, False),
               "copies_branch": copies_per_head(case, not routed),
               "main_ms": min(t_ref), "branch_ms": min(t_new),
               "main_spread_pct": (max(t_ref) / min(t_ref) - 1) * 100,
               "branch_spread_pct": (max(t_new) / min(t_new) - 1) * 100,
               "change_pct": (min(t_new) / min(t_ref) - 1) * 100,
               "outputs": [diff_stats(a, b, dt) for (a, dt), (b, _) in zip(o_ref, o_new)]}
        ref = reference(case, *host)
        row["err_main"] = err(normalized(case, [t for t, _ in o_ref]), ref)
        row["err_branch"] = err(normalized(case, [t for t, _ in o_new]), ref)
        print("[nkl_table] " + json.dumps(row), flush=True)
    for case in CASES:
        name, bs, bs_kv, lq, d, lp, pul, lk, causal, ring = case
        nb = 16 if lk % 16 == 0 and lp % 16 == 0 else 8
        if lk % nb or lp % nb:
            continue
        host = make(case)
        args = [t.to(dev) for t in host[:3] + ((host[3], host[4]) if host[3] is not None else ())]
        bargs = [args[0], blocked(args[1], nb), args[2]] + ([blocked(args[3], nb), args[4]] if lp else [])
        o_pl, t_pl, o_kb, t_kb = ab_trials(caller(A_NEW, case), args, caller(A_NEW, case, k_blocks=nb), bargs)
        row = {"name": name, "k_blocks": nb, "plain_ms": min(t_pl), "kb_ms": min(t_kb),
               "plain_spread_pct": (max(t_pl) / min(t_pl) - 1) * 100,
               "change_pct": (min(t_kb) / min(t_pl) - 1) * 100,
               "outputs": [diff_stats(a, b, dt) for (a, dt), (b, _) in zip(o_pl, o_kb)]}
        print("[nkl_table_kb] " + json.dumps(row), flush=True)

if mode == "one":
    case = [c for c in SWEEP if c[0] == os.environ["ONE_CASE"]][0]
    f = caller(tag_kernel(os.environ.get("ONE_TAG", "main")), case)
    args = device_args(case)
    first(f, args)
    print(f"[nkl_one] {case[0]} {os.environ.get('ONE_TAG')} {timed(f, args):.3f} ms", flush=True)
