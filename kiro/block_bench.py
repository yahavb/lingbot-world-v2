"""One compiled DiT block of lingbot causal_fast at the real per-rank shapes (Wan2 block_bench idea).

torchrun --nproc_per_node=<world> kiro/block_bench.py
Env:
  TP (world), CP (1)          parallel layout, same mesh helpers as the pipeline
  BLOCK (10)                  which layer's real weights to load (hot layers: 33,34,38,39)
  LEAD (3)                    leading KV chunks in the window (3 = steady state, 0..3)
  REPS (20)                   timed calls
  SIZE (480*768) CHUNK (3) WINDOW (12)
  SAVE_REF=<path> / REF=<path>  save / compare the block output (rank 0, fp32 rel-RMS)
  plus every LINGBOT_* / ATTENTION_BACKEND switch the pipeline reads (they are module-level).
Prints ms/block (device time for back-to-back calls, synced at the end) and per-rank memory.
"""
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wan.distributed import tensor_parallel as tp  # noqa: E402
from wan.modules import model_fast as mf  # noqa: E402
from wan.modules.chunked_kv_cache import allocate_self_kv_chunks  # noqa: E402

rope_params = mf.rope_params

CKPT = os.environ.get("CKPT", "/work/ckpt/lingbot-world-v2-14b-causal-fast")
H, W = (int(v) for v in os.environ.get("SIZE", "480*768").split("*"))
CHUNK = int(os.environ.get("CHUNK", "3"))
WINDOW = int(os.environ.get("WINDOW", "12"))
LEAD = int(os.environ.get("LEAD", "3"))
REPS = int(os.environ.get("REPS", "20"))
K = int(os.environ.get("BLOCK", "10"))


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.neuron.set_device(local)
    dist.init_process_group(backend="neuron", init_method="env://", rank=rank, world_size=world)
    dev = torch.device(f"neuron:{local}")
    cp = int(os.environ.get("CP", "1"))
    tpd = int(os.environ.get("TP", str(world // cp)))
    assert cp * tpd == world
    if cp > 1:
        mf.init_dit_mesh(cp, tpd)
        mf.set_seq_parallel_mode("cp")
    else:
        mf.init_dit_parallel_group(tpd)
    tp_rank = mf.get_tp_rank()

    # ---- one block with the real weights of layer K, TP-sharded like load_tp_wan_model_fast
    config = mf.WanModelFast.load_config(CKPT, subfolder="transformers")
    with torch.device("meta"):
        model = mf.WanModelFast.from_config(config, local_attn_size=WINDOW, sink_size=1)
    blk = model.blocks[K]
    mf._rebuild_attn_for_tp(blk.self_attn, tpd)
    mf._rebuild_attn_for_tp(blk.cross_attn, tpd)
    mf._rebuild_ffn_for_tp(blk.ffn, tpd)
    mf._rebuild_cam_injector_for_tp(blk, tpd)
    blk.tp_enabled = True
    blk = blk.to(torch.bfloat16).to_empty(device="cpu")
    sd = torch.load(os.path.join(CKPT, "transformers-bf16.pth"), map_location="cpu", weights_only=True, mmap=True)
    pre = f"blocks.{K}."
    sub = {k: v for k, v in sd.items() if k.startswith(pre)}
    sharded = mf.shard_wan_model_fast_state_dict(sub, tp_rank, tpd)
    blk.load_state_dict({k[len(pre):]: v for k, v in sharded.items()}, strict=True)
    del sd, sub, sharded
    blk = blk.to(torch.bfloat16).to(dev).eval().requires_grad_(False)
    blk.self_attn.layer_index = K
    blk.self_attn.use_const_max = mf.layer_uses_const_max(K)
    blk.self_attn.use_cmax = mf.layer_uses_cmax(K)
    d = config["dim"] // config["num_heads"]
    freqs = torch.cat([rope_params(mf.MAX_LATENT_FRAMES, d - 4 * (d // 6)),
                       rope_params(mf.MAX_LATENT_FRAMES, 2 * (d // 6)),
                       rope_params(mf.MAX_LATENT_FRAMES, 2 * (d // 6))], dim=1).to(dev)

    # ---- per-rank shapes exactly as WanModelFast.forward builds them (chunked cache path)
    fh, fw = H // 16, W // 16
    fs = fh * fw                       # tokens per latent frame (patch grid)
    L = CHUNK * fs                     # tokens per chunk
    grid = torch.tensor([[CHUNK, fh, fw]], dtype=torch.long)
    mf.set_bf16_rope_for_forward(L)
    w_shard = (cp, mf._sp_local_r()) if cp > 1 else None
    L_loc = L // cp
    pad = mf._TOKEN_PAD
    Lp = -(-L_loc // pad) * pad if pad else L_loc
    g = torch.Generator().manual_seed(1234 + rank // tpd)  # same x on every TP rank of a CP cell
    x = torch.randn(1, L_loc, config["dim"], generator=g) * 0.5
    x = torch.nn.functional.pad(x, (0, 0, 0, Lp - L_loc)).to(dev).float()
    e0 = (torch.randn(1, 1, 6, config["dim"], generator=g) * 0.1).to(dev).float()
    ctx = torch.randn(1, 512, config["dim"], generator=g) * 0.5
    ctx_real = int(os.environ.get("CTX_REAL", "0") or 0)
    if ctx_real:
        # CTX_REAL=N: the pipeline's context, N prompt rows then 512 - N identical padding rows (its zero padding,
        # embedded); what LINGBOT_CROSS_DEDUP relies on
        ctx[:, ctx_real:] = ctx[:, ctx_real:ctx_real + 1]
    ctx = ctx.to(dev).to(torch.bfloat16)
    c2ws = (torch.randn(1, L_loc, config["dim"], generator=g) * 0.5)
    c2ws = torch.nn.functional.pad(c2ws, (0, 0, 0, Lp - L_loc)).to(dev).to(torch.bfloat16)
    cos, sin = mf.causal_rope_grids(freqs, grid, dev, 2 * CHUNK, dtype=mf._rope_dtype(x), w_shard=w_shard)
    lg = [tuple(int(v) for v in row) for row in grid.tolist()]
    if w_shard is not None:
        lg = [(f, h, w_ // cp) for (f, h, w_) in lg]
    heads = config["num_heads"] // tpd
    n_chunks = WINDOW // CHUNK
    from wan.modules.attention import const_max_ag_available, fa2_ag_available
    from wan.modules.attention import cte_merge_enabled
    n_major = (const_max_ag_available() and (mf.layer_uses_const_max(K) or fa2_ag_available())) \
        or cte_merge_enabled()
    split = os.environ.get("LINGBOT_ATTN_SPLIT", "0") == "1" and cp > 1
    kv = allocate_self_kv_chunks(1, 1, L, heads, d, n_chunks, torch.bfloat16, dev, n_major=n_major,
                                 own_tokens=L_loc if split else 0)[0]
    if const_max_ag_available():
        from wan.modules.attention import init_const_max_ag
        init_const_max_ag(tp.replica_groups("cp"), L)
    # From a seeded CPU generator, so two runs see the same cache and REF comparisons are valid
    # (t.normal_() on the device draws from an unseeded device generator). The cache holds the
    # GATHERED chunk, so every rank of a TP group sees the same values: seed by TP rank only.
    gk = torch.Generator().manual_seed(4321 + tp_rank)
    for t in kv["k"] + kv["v"]:
        t.copy_(torch.randn(t.shape, generator=gk).to(t.dtype))
    if os.environ.get("LINGBOT_K_BLOCKED", "0") == "1" and cp > 1:
        # the same logical K, stored as CP rank blocks [cp, n, d, l] (what the pipeline caches)
        for t in kv["k"]:
            n_, d_, L_ = t.shape
            t.copy_(t.view(n_, d_, cp, L_ // cp).permute(2, 0, 1, 3).contiguous().view(n_, d_, L_))
    if os.environ.get("LINGBOT_V_TOKEN_MAJOR", "0") == "1":
        # the same logical V, stored token-major: [L, n, d] memory under the [n, L, d] shape
        for t in kv["v"]:
            t.copy_(t.permute(1, 0, 2).contiguous().view(t.shape))
    if os.environ.get("KV_STATS", "0") == "1":
        # diagnostic: per cache slot, the largest |K| and |V| and whether all are finite (randn: ~5)
        torch.neuron.synchronize()
        st = [(i, float(k_.float().abs().amax()), float(v_.float().abs().amax()),
               bool(torch.isfinite(k_.float()).all() and torch.isfinite(v_.float()).all()))
              for i, (k_, v_) in enumerate(zip(kv["k"], kv["v"]))]
        print(f"[block_bench] rank {rank} KV_STATS (slot, max|K|, max|V|, finite): {st}", flush=True)
    if split:
        # each CP rank's own shard of every cached chunk = its token slice of the gathered chunk
        r_cp, l_ = mf._sp_local_r(), L_loc
        for kc_, vc_, ko_, vo_ in zip(kv["k"], kv["v"], kv["ko"], kv["vo"]):
            if int(kc_.shape[1]) == d:            # K [n, d, L]
                ko_.copy_(kc_[:, :, r_cp * l_:(r_cp + 1) * l_])
            else:
                ko_.copy_(kc_[:, r_cp * l_:(r_cp + 1) * l_])
            vo_.copy_(vc_[:, r_cp * l_:(r_cp + 1) * l_])
    # ORDER=i,j,...: diagnostic, the cache slots used as the LEAD cached chunks (default 0 .. LEAD-1)
    order = ([int(s) for s in os.environ["ORDER"].split(",")] if os.environ.get("ORDER")
             else list(range(LEAD))) + [n_chunks - 1]
    assert len(order) == LEAD + 1, (order, LEAD)
    ks, vs = [kv["k"][i] for i in order], [kv["v"][i] for i in order]
    own = (([kv["ko"][i] for i in order], [kv["vo"][i] for i in order]),) if split else ()
    if os.environ.get("WINDOW_KV", "0") == "1" and LEAD >= 2:
        # The leading chunks as ONE contiguous buffer (what LINGBOT_KV_WINDOW builds once per
        # chunk), so LINGBOT_ATTN_PAIR passes it as the kernel's prior: one call per block.
        ks = [torch.cat(ks[:-1], dim=-1).contiguous(), ks[-1]]
        vs = [torch.cat(vs[:-1], dim=1).contiguous(), vs[-1]]
    from wan.modules.attention import cross_cache_shapes
    _cks, _cvs = cross_cache_shapes(1, 512, heads, d)
    cross = {"k": torch.zeros(_cks, dtype=torch.bfloat16, device=dev),
             "v": torch.zeros(_cvs, dtype=torch.bfloat16, device=dev),
             "is_init": torch.tensor(0, dtype=torch.int32, device=dev)}
    for t in (cross["k"], cross["v"]):
        torch._dynamo.mark_static_address(t)
    if mf._CROSS_DEDUP:
        # LINGBOT_CROSS_DEDUP: its per-key bias buffer, and the forward's prompt-length check (needs CTX_REAL)
        cross["mult"] = torch.zeros(mf._CROSS_DEDUP, dtype=torch.float32, device=dev)
        torch._dynamo.mark_static_address(cross["mult"])
        mf._CROSS_DEDUP_ON[0] = 0 < ctx_real < mf._CROSS_DEDUP

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    if hasattr(torch._dynamo.config, "do_not_emit_runtime_asserts"):
        torch._dynamo.config.do_not_emit_runtime_asserts = True
    film_fn = torch.compile(blk.cam_film, backend="neuron", fullgraph=True, dynamic=False)
    cblk = torch.compile(blk, backend="neuron", fullgraph=True, dynamic=False)
    kw = dict(e=e0, seq_lens=torch.tensor([L_loc], dtype=torch.long), grid_sizes=grid,
              rope_grids=(cos, sin), rope_local_grid=lg, grid_ints=lg, freqs=freqs, context=ctx,
              context_lens=None, max_attention_size=WINDOW * fs, frame_seqlen=fs,
              seq_lens_int=L_loc, sp_start_row=0, sp_rows=None)

    # Camera FiLM params are hoisted per chunk in the pipeline (_cam_films), so they are built once
    # here, outside the timed calls; LINGBOT_FILM_FULL / LINGBOT_SP prepare them as there.
    sp = mf._SP_ON and tpd > 1
    a2a = sp and mf._FILM_A2A and tp.has_mesh()
    if a2a:
        film_fn = torch.compile(blk.cam_film_sp, backend="neuron", fullgraph=True, dynamic=False)

    def prep_film():
        film = film_fn(c2ws)
        if a2a:
            return film
        if (mf._FILM_FULL or sp) and tpd > 1:
            film = mf.film_full_width(*film)
        if sp:
            film = tuple(mf.sp_token_shard(t, tpd, tp_rank).clone() for t in film)
        return film

    with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
        film = prep_film()
        torch.neuron.synchronize()
        if a2a and os.environ.get("FILM_CHECK", "0") == "1":
            # The A2A path must reproduce gather + slice exactly: compare the tensors themselves
            # (the block output barely moves under a wrong FiLM, so it cannot prove this).
            g_fn = torch.compile(blk.cam_film, backend="neuron", fullgraph=True, dynamic=False)
            raw = g_fn(c2ws)
            ref_g = tuple(mf.sp_token_shard(t, tpd, tp_rank).clone()
                          for t in mf.film_full_width(*raw))
            torch.neuron.synchronize()
            for nm, a_, r_ in zip(("scale", "shift"), film, ref_g):
                a_, r_ = a_.float().cpu(), r_.float().cpu()
                log(f"[block_bench] FILM_CHECK rank {rank} {nm}: shape {tuple(a_.shape)} vs "
                    f"{tuple(r_.shape)} max|diff| {(a_ - r_).abs().max():.3e} "
                    f"rms {r_.pow(2).mean().sqrt():.3e} bitwise {torch.equal(a_, r_)}")
        if int(os.environ.get("FILM_REPS", "0")) > 0:
            for _ in range(2):
                prep_film()
            torch.neuron.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(int(os.environ["FILM_REPS"])):
                prep_film()
            torch.neuron.synchronize()
            dist.barrier()
            log(f"[block_bench] film prep {'a2a' if a2a else 'gather'}: "
                f"{(time.perf_counter() - t0) / int(os.environ['FILM_REPS']) * 1e3:.2f} ms/block")
        for fr in [int(s) for s in os.environ.get("FILM_ROWS", "").split(",") if s]:
            # FILM_ROWS=512,544,...: this block's per-chunk FiLM graph (cam_film_sp) on fr camera tokens (the chunk's
            # tokens truncated or zero-padded), to see how its cost depends on the row count
            assert a2a and fr % tpd == 0, "FILM_ROWS needs LINGBOT_SP=1 LINGBOT_FILM_A2A=1 and rows divisible by TP"
            c_r = c2ws[:, :fr] if fr <= c2ws.shape[1] else torch.nn.functional.pad(c2ws, (0, 0, 0, fr - c2ws.shape[1]))
            fn_r = torch.compile(blk.cam_film_sp, backend="neuron", fullgraph=True, dynamic=False)
            for _ in range(2):
                fn_r(c_r)
            torch.neuron.synchronize()
            dist.barrier()
            nr = int(os.environ.get("FILM_REPS", "0") or 20)
            t0 = time.perf_counter()
            for _ in range(nr):
                fn_r(c_r)
            torch.neuron.synchronize()
            dist.barrier()
            log(f"[block_bench] film rows={fr}: {(time.perf_counter() - t0) / nr * 1e3:.2f} ms/block")
        if os.environ.get("FILM_PADCHECK"):
            # FILM_PADCHECK=N: the per-chunk FiLM with LINGBOT_FILM_ROWS=N against the unpadded one, every rank, + time
            mf._FILM_ROWS = int(os.environ["FILM_PADCHECK"])
            fn_p = torch.compile(blk.cam_film_sp, backend="neuron", fullgraph=True, dynamic=False)
            with torch.no_grad():
                got = fn_p(c2ws)
            for nm, a_, r_ in zip(("scale", "shift"), got, film):
                a_, r_ = a_.float().cpu(), r_.float().cpu()
                print(f"[block_bench] rank {rank} FILM_PADCHECK {nm}: shape {tuple(a_.shape)} vs {tuple(r_.shape)} "
                      f"max|diff| {(a_ - r_).abs().max():.3e} bitwise {torch.equal(a_, r_)}", flush=True)
            for _ in range(2):
                fn_p(c2ws)
            torch.neuron.synchronize()
            dist.barrier()
            nr = int(os.environ.get("FILM_REPS", "0") or 20)
            t0 = time.perf_counter()
            for _ in range(nr):
                fn_p(c2ws)
            torch.neuron.synchronize()
            dist.barrier()
            log(f"[block_bench] film padded to {mf._FILM_ROWS} rows: {(time.perf_counter() - t0) / nr * 1e3:.2f} ms/block")
            mf._FILM_ROWS = 0
        if os.environ.get("FILM_ARCHECK"):
            # FILM_ARCHECK=N: the per-chunk FiLM on N rows (LINGBOT_FILM_ROWS=N) with cam_injector_layer2's all-reduce
            # in fp32 and in bf16 (LINGBOT_FILM_AR_BF16): the params compared on every rank, both timed
            mf._FILM_ROWS = int(os.environ["FILM_ARCHECK"])
            outs, tms = [], []
            for flag in (False, True):
                mf._FILM_AR_BF16 = flag
                fn_a = torch.compile(blk.cam_film_sp, backend="neuron", fullgraph=True, dynamic=False)
                with torch.no_grad():
                    outs.append([t.float().cpu() for t in fn_a(c2ws)])
                for _ in range(2):
                    fn_a(c2ws)
                torch.neuron.synchronize()
                dist.barrier()
                nr = int(os.environ.get("FILM_REPS", "0") or 20)
                t0 = time.perf_counter()
                for _ in range(nr):
                    fn_a(c2ws)
                torch.neuron.synchronize()
                dist.barrier()
                tms.append((time.perf_counter() - t0) / nr * 1e3)
            for nm, a_, r_ in zip(("scale", "shift"), outs[1], outs[0]):
                print(f"[block_bench] rank {rank} FILM_ARCHECK {nm}: max|diff| {(a_ - r_).abs().max():.3e} rel "
                      f"{((a_ - r_).pow(2).mean().sqrt() / r_.pow(2).mean().sqrt()).item():.3e} "
                      f"rms {r_.pow(2).mean().sqrt():.3e}", flush=True)
            log(f"[block_bench] film on {mf._FILM_ROWS} rows, all-reduce fp32 / bf16: {tms[0]:.2f} / {tms[1]:.2f} ms/block")
            mf._FILM_ROWS = 0
            mf._FILM_AR_BF16 = False
        for fk in [int(s) for s in os.environ.get("FILM_K", "").split(",") if s]:
            # FILM_K=1,2,4: this block's FiLM for K chunks in one graph (LINGBOT_FILM_ALL's _films_multi), so the
            # cost per chunk can be compared with the per-chunk graph's
            assert a2a, "FILM_K needs LINGBOT_SP=1 LINGBOT_FILM_A2A=1"
            c_cat = c2ws.repeat(1, fk, 1)
            fn_k = torch.compile(lambda c_, k_=fk: mf._films_multi([blk], c_, k_, Lp), backend="neuron",
                                 fullgraph=True, dynamic=False)
            for _ in range(2):
                fn_k(c_cat)
            torch.neuron.synchronize()
            dist.barrier()
            nr = int(os.environ.get("FILM_REPS", "0") or 20)
            t0 = time.perf_counter()
            for _ in range(nr):
                fn_k(c_cat)
            torch.neuron.synchronize()
            dist.barrier()
            ms_k = (time.perf_counter() - t0) / nr * 1e3
            # every chunk of c_cat is this chunk's condition, so each chunk's params must match the per-chunk graph's
            with torch.no_grad():
                res = fn_k(c_cat)
            worst = 0.0
            for (sc, sh), ref_t in ((res[0], film), ):
                for j in range(fk):
                    for a_, r_ in ((sc[j:j + 1], ref_t[0]), (sh[j:j + 1], ref_t[1])):
                        a_, r_ = a_.float().cpu(), r_.float().cpu()
                        worst = max(worst, ((a_ - r_).pow(2).mean().sqrt() / r_.pow(2).mean().sqrt()).item())
            print(f"[block_bench] rank {rank} film K={fk}: worst rel vs per-chunk {worst:.3e}", flush=True)
            log(f"[block_bench] film K={fk}: {ms_k:.2f} ms/block for {fk} chunks, {ms_k / fk:.2f} ms/block per chunk")
    x_in = mf.sp_token_shard(x, tpd, tp_rank) if sp else x

    # KV_WRITE=0: the denoise-step variant of LINGBOT_STEP_NOWRITE (no write into the cache slot)
    kv_extra = {"kv_write": False} if os.environ.get("KV_WRITE", "1") == "0" else {}
    dump_dir = os.environ.get("DUMP_CMAX")
    if dump_dir:
        # diagnostic: the constant-offset path copies its kernel queries / accumulator / output here every call
        mf._DIAG_BUFS["q_t"] = torch.zeros(heads, d, Lp, dtype=torch.bfloat16, device=dev)
        mf._DIAG_BUFS["acc"] = torch.zeros(heads, Lp, d + 1, dtype=torch.float32, device=dev)
        mf._DIAG_BUFS["o_n"] = torch.zeros(heads, Lp, d, dtype=torch.bfloat16, device=dev)

    def run(first):
        mf._REAL_TOKENS[0] = L_loc if Lp != L_loc else None
        return cblk(x_in, kv_cache=None, kv_chunks=(list(ks), list(vs)) + own, crossattn_cache=cross,
                    current_start=0, dit_cond_dict={"cam_film": film}, cross_attn_first_call=first,
                    **kw, **kv_extra)

    with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
        t0 = time.perf_counter()
        run(True)
        torch.neuron.synchronize()
        t_first = time.perf_counter() - t0
        out = run(False)
        torch.neuron.synchronize()
        out = run(False)
        torch.neuron.synchronize()
        if int(os.environ.get("FIRST_REPS", "0")) > 0:
            # The cross_attn_first_call variant (chunk 0, step 0 of every video).
            fr = int(os.environ["FIRST_REPS"])
            for _ in range(2):
                run(True)
            torch.neuron.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(fr):
                run(True)
            torch.neuron.synchronize()
            dist.barrier()
            log(f"[block_bench] first-call variant: {(time.perf_counter() - t0) / fr * 1e3:.2f} ms/block")
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = run(False)
        t_host = time.perf_counter() - t0
        torch.neuron.synchronize()
        dist.barrier()
        ms = (time.perf_counter() - t0) / REPS * 1e3
        if sp:  # reassemble the token shards for the output check (outside the timed loop)
            out = tp.all_gather_tokens(out.contiguous(), "tp")
            torch.neuron.synchronize()
    nblk = int(os.environ.get("NBLK", "1"))
    if nblk > 1:
        # NBLK=N: N copies of the block (cloned weights, own KV and cross-attention caches), called in turn, as the
        # pipeline's 40 blocks are: the same compiled graph executed on other buffers every call.
        import copy
        assert not own, "NBLK: no LINGBOT_ATTN_SPLIT"
        cops = [(cblk, ks, vs, cross)]
        for _ in range(nblk - 1):
            b2 = copy.deepcopy(blk)
            ks2, vs2 = [t.clone() for t in ks], [t.clone() for t in vs]
            cr2 = {k_: t.clone() for k_, t in cross.items()}
            for t in ks2 + vs2 + [cr2["k"], cr2["v"]]:
                torch._dynamo.mark_static_address(t)
            cops.append((torch.compile(b2, backend="neuron", fullgraph=True, dynamic=False), ks2, vs2, cr2))

        def run_copy(c_):
            mf._REAL_TOKENS[0] = L_loc if Lp != L_loc else None
            cb_, ks_, vs_, cr_ = c_
            return cb_(x_in, kv_cache=None, kv_chunks=(list(ks_), list(vs_)), crossattn_cache=cr_, current_start=0,
                       dit_cond_dict={"cam_film": film}, cross_attn_first_call=False, **kw, **kv_extra)
        with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
            for c_ in cops[1:]:
                run_copy(c_)  # cross-attention cache: copied from the initialized one
                run_copy(c_)
            torch.neuron.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(REPS):
                for c_ in cops:
                    run_copy(c_)
            torch.neuron.synchronize()
            dist.barrier()
            ms_n = (time.perf_counter() - t0) / (REPS * nblk) * 1e3
        log(f"[block_bench] NBLK={nblk}: {ms_n:.2f} ms/block cycling {nblk} copies (one copy: {ms:.2f} ms/block)")
    if os.environ.get("ALT", "0") == "1":
        # Run once on DIFFERENT data in the same buffers (perturbed cache contents, scaled input),
        # restore the original contents, then do the compared call. Every timed rep sees identical
        # inputs, so a read of state left by the previous execution would match there; in the
        # pipeline each block follows a block with other data.
        with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
            bufs = list(ks) + list(vs) + ([t for pair in own for lst in pair for t in lst] if own else [])
            saved = [t.clone() for t in bufs]
            for t in bufs:
                t.mul_(-1.0)
            torch.neuron.synchronize()
            mf._REAL_TOKENS[0] = L_loc if Lp != L_loc else None
            cblk(x_in * 1.5, kv_cache=None, kv_chunks=(list(ks), list(vs)) + own, crossattn_cache=cross,
                 current_start=0, dit_cond_dict={"cam_film": film}, cross_attn_first_call=False, **kw)
            torch.neuron.synchronize()
            for t, s_ in zip(bufs, saved):
                t.copy_(s_)
            torch.neuron.synchronize()
    zero_dest = os.environ.get("ZERO_DEST", "0") == "1" or os.environ.get("ALT", "0") == "1"
    if zero_dest:
        # Every timed rep writes the same new chunk into the destination buffers, so a read that
        # raced ahead of that write would still see the right values. Clear what this forward
        # writes before the compared call, so a premature read shows up as a difference.
        with torch.no_grad():
            ks[-1].zero_()
            vs[-1].zero_()
            for t in (own[0][0][-1], own[0][1][-1]) if own else ():
                t.zero_()
            if os.environ.get("CMP_FIRST", "0") == "1":
                cross["k"].zero_()
                cross["v"].zero_()
        torch.neuron.synchronize()
    if os.environ.get("CMP_FIRST", "0") == "1" or zero_dest:
        # compare the cross_attn_first_call=True variant (chunk 0, step 0 of every video) instead:
        # it recomputes the cross-attention K/V from the context, then runs the block
        with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
            out = run(os.environ.get("CMP_FIRST", "0") == "1")
            torch.neuron.synchronize()
            if sp:
                out = tp.all_gather_tokens(out.contiguous(), "tp")
                torch.neuron.synchronize()
    if dump_dir:
        # the last call's kernel tensors plus the cache chunks it read (memory layouts as the kernel reads them)
        torch.neuron.synchronize()
        os.makedirs(dump_dir, exist_ok=True)
        torch.save({"q_t": mf._DIAG_BUFS["q_t"].cpu(), "acc": mf._DIAG_BUFS["acc"].cpu(),
                    "o_n": mf._DIAG_BUFS["o_n"].cpu(), "ks": [t.cpu() for t in ks], "vs": [t.cpu() for t in vs],
                    "lead": LEAD, "cp": cp, "L_loc": L_loc}, os.path.join(dump_dir, f"r{rank:02d}.pt"))
    if os.environ.get("KV_STATS", "0") == "1":
        # diagnostic: the cache after the timed calls (they must not have written it with KV_WRITE=0)
        torch.neuron.synchronize()
        st = [(i, float(k_.float().abs().amax()), float(v_.float().abs().amax()),
               bool(torch.isfinite(k_.float()).all() and torch.isfinite(v_.float()).all()))
              for i, (k_, v_) in enumerate(zip(kv["k"], kv["v"]))]
        print(f"[block_bench] rank {rank} KV_STATS after (slot, max|K|, max|V|, finite): {st}", flush=True)
    o = out.float().cpu()
    msg = (f"[block_bench] tp={tpd} cp={cp} block={K} lead={LEAD} L_loc={L_loc}->{Lp} heads/rank={heads} "
           f"{ms:.2f} ms/block (host {t_host / REPS * 1e3:.2f} ms, first call {t_first:.1f}s) "
           f"out_rms={o[0, :L_loc].pow(2).mean().sqrt():.4f} finite={bool(torch.isfinite(o).all())} "
           f"mem={torch.neuron.memory_allocated(dev) / 1e9:.2f}GB")
    ref, save = os.environ.get("REF"), os.environ.get("SAVE_REF")
    if rank == 0 and save:
        torch.save(o, save)
        msg += f" saved {save}"
    if rank == 0 and ref and os.path.exists(ref):
        r = torch.load(ref)
        rel = ((o - r)[0, :L_loc].pow(2).mean().sqrt() / r[0, :L_loc].pow(2).mean().sqrt()).item()
        msg += (f" rel_rms_vs_ref={rel:.5f} ref_equal={torch.equal(o[0, :L_loc], r[0, :L_loc])}"
                f" ref_max_abs={(o - r)[0, :L_loc].abs().max().item():.3e}")
        t0_ = (L_loc // 128) * 128
        if t0_ < L_loc:
            # the rows past the last full 128-row query tile (what LINGBOT_ATTN_SPLIT/REM change)
            relr = ((o - r)[0, t0_:L_loc].pow(2).mean().sqrt()
                    / r[0, t0_:L_loc].pow(2).mean().sqrt()).item()
            msg += f" rem_rows_rel={relr:.5f}"
    # SAVE_REFDIR=<dir> / REFDIR=<dir>: every rank saves / compares its OWN output. Rank 0's output
    # alone cannot show an error on another CP rank: it only reaches rank 0 through the next
    # block's K/V gather, which this bench does not run.
    rdir_save, rdir = os.environ.get("SAVE_REFDIR"), os.environ.get("REFDIR")
    if rdir_save:
        os.makedirs(rdir_save, exist_ok=True)
        torch.save(o, os.path.join(rdir_save, f"r{rank:02d}.pt"))
        msg += f" saved_all {rdir_save}"
    if rdir and os.path.exists(os.path.join(rdir, f"r{rank:02d}.pt")):
        r = torch.load(os.path.join(rdir, f"r{rank:02d}.pt"))
        d_ = (o - r)[0, :L_loc]
        st = (bool(torch.equal(o[0, :L_loc], r[0, :L_loc])),
              (d_.pow(2).mean().sqrt() / r[0, :L_loc].pow(2).mean().sqrt()).item(),
              d_.abs().max().item())
    else:
        st = None
    if rdir:
        allst = [None] * world
        dist.all_gather_object(allst, st)
        if rank == 0:
            got = [(i, s_) for i, s_ in enumerate(allst) if s_ is not None]
            bad = [i for i, s_ in got if not s_[0]]
            worst = max(got, key=lambda t: t[1][1]) if got else None
            msg += (f" ranks_equal={len(got) - len(bad)}/{len(got)} bad_ranks={bad[:16]}"
                    + (f" worst_rank={worst[0]} worst_rel={worst[1][1]:.3e} worst_max={worst[1][2]:.3e}"
                       if worst else ""))
    log(msg)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
