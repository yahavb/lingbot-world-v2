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
    ctx = (torch.randn(1, 512, config["dim"], generator=g) * 0.5).to(dev).to(torch.bfloat16)
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
    kv = allocate_self_kv_chunks(1, 1, L, heads, d, n_chunks, torch.bfloat16, dev, n_major=n_major)[0]
    if const_max_ag_available():
        from wan.modules.attention import init_const_max_ag
        init_const_max_ag(tp.replica_groups("cp"), L)
    for t in kv["k"] + kv["v"]:
        t.normal_(0, 1.0)
    order = list(range(LEAD)) + [n_chunks - 1]
    ks, vs = [kv["k"][i] for i in order], [kv["v"][i] for i in order]
    cross = {"k": torch.zeros(1, 512, heads, d, dtype=torch.bfloat16, device=dev),
             "v": torch.zeros(1, 512, heads, d, dtype=torch.bfloat16, device=dev),
             "is_init": torch.tensor(0, dtype=torch.int32, device=dev)}
    for t in (cross["k"], cross["v"]):
        torch._dynamo.mark_static_address(t)

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
    x_in = mf.sp_token_shard(x, tpd, tp_rank) if sp else x

    def run(first):
        mf._REAL_TOKENS[0] = L_loc if Lp != L_loc else None
        return cblk(x_in, kv_cache=None, kv_chunks=(list(ks), list(vs)), crossattn_cache=cross,
                    current_start=0, dit_cond_dict={"cam_film": film}, cross_attn_first_call=first, **kw)

    with torch.no_grad(), torch.amp.autocast("neuron", dtype=torch.bfloat16):
        t0 = time.perf_counter()
        run(True)
        torch.neuron.synchronize()
        t_first = time.perf_counter() - t0
        out = run(False)
        torch.neuron.synchronize()
        out = run(False)
        torch.neuron.synchronize()
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
        msg += f" rel_rms_vs_ref={rel:.5f}"
    log(msg)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
