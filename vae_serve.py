"""Width-sharded VAE decode server for serve.py's remote-decode mode (VAE_URL).

    torchrun --nproc_per_node=<N> vae_serve.py --ckpt_dir <dir> --lat_h 60 --lat_w 96

    POST /decode?first=0|1   body: torch.save'd latent chunk [C, T, H, W]
        -> {"frames": [base64 JPEG, ...]}
    POST /decode?first=0|1&stream=1
        -> application/x-ndjson, one {"frames": [...]} line per decode step as it finishes
    GET  /health
    GET  /readiness

The chunks of one video must arrive in order, with first=1 on the first; the
causal decode cache carries over between calls. Rank 0 serves HTTP from a
background thread; every rank decodes its shard on its main thread.

LINGBOT_VAE_FUSED=1 decodes with wan/modules/vae2_1_fused.py (compiled decoder blocks,
halos inside the graphs; LINGBOT_VAE_2D=1 for height x width tiles), the decoder
docs/perf/LOOP_ENGINEERING.md measured; otherwise the per-conv width-sharded decoder.
"""

import argparse
import asyncio
import io
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist

import generate as G
from serve import b64_jpeg, to_pil
from wan.configs import WAN_CONFIGS
from wan.modules.vae2_1_sharded import gather_video_width, width_shard_latent

PORT = int(os.environ.get("PORT", "8000"))
IDLE_S = 60


def run_server(jobs, state):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    import uvicorn

    app = FastAPI(title="LingBot-World v2 VAE decode")

    @app.post("/link")
    def link(gen: int, dit_rank: int):
        """A DiT starts a session here: answer this process's transport rank, then the decode loop
        builds the RDMA link (wan/rdma_link.py) while the DiT builds its end."""
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        if state.get("nixl") is not None:        # NIXL: the DiT writes into our slots; nothing to build here
            # LINGBOT_VAE_SESSIONS=K: each new session takes the next of K slot blocks / decoder caches (round robin;
            # the K+1-th session reuses the oldest block, so K must cover the sessions served at once)
            from wan.nixl_link import SESSIONS
            sid = state.get("next_sid", 0)
            state["next_sid"] = (sid + 1) % SESSIONS
            state.setdefault("seen", {})[sid] = time.time()
            return state["nixl"].info(sid)
        if state.get("rank") is None:
            raise HTTPException(status_code=503, detail="warming up")
        jobs.put(("__link__", gen, dit_rank))
        return {"vae_rank": state["rank"]}

    @app.post("/decode")
    async def decode(request: Request, first: int = 0, stream: int = 0, rdma: int = 0, seq: int = -1, sid: int = 0):
        """The decode loop puts lists of JPEG futures, one per step, then None or an exception."""
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        result = queue.Queue()
        body = await request.body()
        state.setdefault("seen", {})[sid] = time.time()
        jobs.put((None if rdma else body, bool(first), result, seq, sid))

        def parts():
            while True:
                part = result.get()
                if part is None or isinstance(part, Exception):
                    yield part
                    return
                yield [f.result() for f in part]

        if stream:
            async def lines():
                it = parts()
                while True:
                    part = await asyncio.to_thread(next, it)
                    if part is None:
                        return
                    if isinstance(part, Exception):
                        yield json.dumps({"error": str(part)}) + "\n"
                        return
                    yield json.dumps({"frames": part}) + "\n"
            return StreamingResponse(lines(), media_type="application/x-ndjson")
        frames = []
        for part in await asyncio.to_thread(lambda: list(parts())):
            if isinstance(part, Exception):
                raise HTTPException(status_code=500, detail=str(part))
            if part is not None:
                frames += part
        return {"frames": frames}

    @app.get("/load")
    def load():
        """Sessions linked or decoding in the last LOAD_WINDOW_S seconds (the DiT's least-loaded VAE choice)."""
        now, win = time.time(), float(os.environ.get("LOAD_WINDOW_S", "10"))
        return {"sessions": sum(1 for t in state.get("seen", {}).values() if now - t < win)}

    @app.get("/health")
    def health():
        logging.info("probe /health: healthy")
        return {"status": "healthy"}

    @app.get("/readiness")
    def readiness():
        if not state["ready"]:
            logging.info("probe /readiness: warming up (503)")
            raise HTTPException(status_code=503, detail="warming up")
        logging.info("probe /readiness: ready")
        return {"status": "ready"}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--task", default="i2v-A14B")
    p.add_argument("--lat_h", type=int, required=True)
    p.add_argument("--lat_w", type=int, required=True)
    p.add_argument("--chunk_size", type=int, default=3)
    args = p.parse_args()

    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    G._init_logging(rank)
    torch.neuron.set_device(local_rank)
    dist.init_process_group(backend="neuron")
    dev = torch.device(f"neuron:{local_rank}")
    torch.set_grad_enabled(False)
    cfg = WAN_CONFIGS[args.task]
    if world > 1 and os.environ.get("LINGBOT_VAE_FUSED", "0") == "1":
        return serve_fused(args, cfg, dev, rank)
    vae, vae_world, pad_w = G._width_sharded_vae(args, cfg, dev, rank, world, args.lat_w)

    def decode(z, first, emit, sid=0):
        """Decode one chunk on every rank; rank 0 emits JPEG futures."""
        lat_w = z.shape[-1]
        assert lat_w == args.lat_w, f"latent width {lat_w}, server built for {args.lat_w}"
        if vae is not None:
            if pad_w != lat_w:
                z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], pad_w - lat_w)], dim=-1)
            video = gather_video_width(vae.decode_chunk(
                width_shard_latent(z.to(dev), rank=rank, world=vae_world), first=first))
            if rank == 0:
                video = video[..., :lat_w * cfg.vae_stride[2]]
                emit([POOL.submit(jpeg, video[:, t]) for t in range(video.shape[1])])
        dist.barrier()

    serve_loop(decode, args, rank)


def serve_fused(args, cfg, dev, rank):
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.modules import vae2_1_fused as F
    extra = F.apply_vae_cc_flags()
    vae = Wan2_1_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=dev,
                     dtype=torch.bfloat16)
    fdec = F.FusedWidthDecoder(vae.model)
    two_d = F._TWO_D
    if rank == 0:
        logging.info(f"fused VAE decoder, {'2-D tiles' if two_d else 'width slices'}"
                     + (f", compiler flags + {extra}" if extra else ""))

    caches = {}     # session id -> the fused decoder's causal caches (LINGBOT_VAE_SESSIONS)

    def decode(z, first, emit, sid=0):
        """Decode one chunk on every rank; rank 0 emits each step's JPEG futures as it lands. Each session keeps its
        own causal decoder cache, swapped in before its chunk and saved after (sessions interleave chunk by chunk)."""
        assert z.shape[-1] == args.lat_w, f"latent width {z.shape[-1]}, server built for {args.lat_w}"
        if not first:
            fdec._caches = caches.get(sid)
        local = F.shard_latent_2d(z) if two_d else width_shard_latent(z, rank=rank, world=dist.get_world_size())

        def step(out):
            out = out.float().clamp_(-1, 1).squeeze(0)
            video = F.gather_video_2d(out) if two_d else gather_video_width(out)
            if rank == 0:
                emit([POOL.submit(jpeg, video[:, t]) for t in range(video.shape[1])])
        fdec.decode_chunk(local.to(dev).unsqueeze(0), vae.scale, first, on_step=step)
        # a compiled graph may write its next outputs into the same device buffers, so a session's saved cache is
        # copied unless it is the only session (LINGBOT_VAE_SESSIONS=1 keeps the old, copy-free path)
        from wan.nixl_link import SESSIONS
        caches[sid] = _clone(fdec._caches) if SESSIONS > 1 else fdec._caches
        if rank == 0 and first and SESSIONS > 1:
            nb = _nbytes(caches[sid])
            logging.info(f"[SESSIONS] session {sid}: causal cache {nb / 2**20:.1f} MiB on this rank; "
                         f"{len(caches)} sessions hold {sum(_nbytes(c) for c in caches.values()) / 2**20:.1f} MiB; "
                         f"device allocated {torch.neuron.memory_allocated() / 2**30:.2f} GiB" if hasattr(torch.neuron, "memory_allocated") else "")
        dist.barrier()

    serve_loop(decode, args, rank)


def _nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_nbytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_nbytes(v) for v in x.values())
    return 0


def _clone(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_clone(v) for v in x)
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


def jpeg(frame):
    return b64_jpeg(to_pil(frame))


# JPEG encoding off the decode loop's thread, so the next step's device work is issued at once.
POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("JPEG_THREADS", "6")))


def serve_loop(decode, args, rank):
    z = torch.zeros(16, args.chunk_size, args.lat_h, args.lat_w)
    for first in (True, False):
        parts = []
        decode(z, first, parts.append)
        [f.result() for p in parts for f in p]
    if rank == 0:
        logging.info("VAE warmup done")

    # LINGBOT_VAE_RDMA=1: the DiT's rank 0 sends each chunk's latents to this server's rank 0 over EFA
    # (wan/rdma_link.py; this side hosts the rendezvous store); rank 0 then broadcasts them on the device.
    use_rdma = os.environ.get("LINGBOT_VAE_RDMA", "0") == "1"
    link, zbuf, store, R = None, None, None, None
    jobs = queue.Queue()
    state = {"ready": False}
    if use_rdma:
        dev = torch.device("neuron", torch.neuron.current_device())
        zbuf = torch.empty(16, args.chunk_size, args.lat_h, args.lat_w, dtype=torch.float32, device=dev)
        if rank == 0 and os.environ.get("LINGBOT_VAE_TRANSPORT", "nixl") == "nixl":
            from wan.nixl_link import NixlVae
            state["nixl"] = NixlVae((16, args.chunk_size, args.lat_h, args.lat_w), dev)
        elif rank == 0:
            import random
            from wan import rdma_link as R
            store = R.shared_store()                          # the cluster's rendezvous store; a link per DiT session
            state["rank"] = R.new_transport_rank(store)       # this process's transport rank, never reused
    if rank == 0:
        threading.Thread(target=run_server, args=(jobs, state), daemon=True).start()
    state["ready"] = True

    while True:
        job = None
        if rank == 0:
            try:
                job = jobs.get(timeout=IDLE_S)
            except queue.Empty:
                pass
        if job is not None and job[0] == "__link__":
            # rank 0 only: a new session's link (the other ranks never see it)
            if link is not None:
                link.close()
            try:
                # short receive timeout: a DiT that gave up on this link relinks within seconds, and the
                # decode loop must be free to take its new /link (the DiT sends right after its request)
                link = R.RdmaLink("vae", store, job[1], job[2], state["rank"],
                                  timeout_s=float(os.environ.get("VAE_RECV_TIMEOUT_S", "5")))
            except Exception:
                logging.exception("rdma link failed")
                link = None
            job = None
        msg = [None if job is None else {"z": job[0], "first": job[1], "rdma": job[0] is None,
                                         "seq": job[3] if len(job) > 3 else -1, "sid": job[4] if len(job) > 4 else 0}]
        dist.broadcast_object_list(msg, src=0)
        if msg[0] is None:
            continue
        try:
            t0 = time.perf_counter()
            if msg[0]["rdma"]:
                # A failed receive (the DiT died or gave up) drops the link and fails only this chunk: every
                # rank learns it from `ok` and skips the decode together, instead of rank 0 raising and taking
                # the whole server down (the other ranks then failed in their next barrier).
                ok = torch.ones(1, dtype=torch.int32, device=zbuf.device)
                err = None
                if rank == 0:
                    try:
                        if state.get("nixl") is not None:
                            zbuf.copy_(state["nixl"].wait(msg[0]["seq"], float(os.environ.get("VAE_RECV_TIMEOUT_S", "5")),
                                                                     sid=msg[0]["sid"]))
                            torch.neuron.synchronize()
                        elif link is None:
                            raise RuntimeError("no RDMA link (the DiT must POST /link first)")
                        else:
                            link.recv(zbuf).wait()
                            torch.neuron.synchronize()
                    except Exception as e:
                        err = e
                        ok.zero_()
                        logging.warning(f"rdma recv failed, link dropped: {type(e).__name__}: {e}")
                        if link is not None:
                            link.close()
                        link = None
                dist.broadcast(ok, src=0)
                if int(ok.item()) == 0:
                    if rank == 0:
                        job[2].put(RuntimeError(f"rdma recv failed: {err}"))
                    continue
                dist.broadcast(zbuf, src=0)
                z = zbuf
            else:
                z = torch.load(io.BytesIO(msg[0]["z"]), map_location="cpu")
            n = [0]

            def emit(part):
                n[0] += len(part)
                job[2].put(part)
            decode(z, msg[0]["first"], emit, msg[0]["sid"])
            if rank == 0:
                logging.info(f"decode first={int(msg[0]['first'])} latent={tuple(z.shape)} "
                             f"-> {n[0]} frames in {(time.perf_counter() - t0) * 1000:.0f} ms")
                job[2].put(None)
        except Exception as e:
            logging.exception("decode failed")
            if rank == 0:
                job[2].put(e)
            raise


if __name__ == "__main__":
    main()
