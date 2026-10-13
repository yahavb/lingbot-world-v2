"""Width-sharded VAE decode server for serve.py's remote-decode mode (VAE_URL).

    torchrun --nproc_per_node=<N> vae_serve.py --ckpt_dir <dir> --lat_h 60 --lat_w 96

    POST /decode?first=0|1   body: torch.save'd latent chunk [C, T, H, W]
        -> {"frames": [base64 JPEG, ...]}
    GET  /health
    GET  /readiness

The chunks of one video must arrive in order, with first=1 on the first; the
causal decode cache carries over between calls. Rank 0 serves HTTP from a
background thread; every rank decodes its width shard on its main thread.
"""

import argparse
import asyncio
import io
import logging
import os
import queue
import threading
import time

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
    import uvicorn

    app = FastAPI(title="LingBot-World v2 VAE decode")

    @app.post("/decode")
    async def decode(request: Request, first: int = 0):
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="warming up")
        result = queue.Queue()
        jobs.put((await request.body(), bool(first), result))
        out = await asyncio.to_thread(result.get)
        if isinstance(out, Exception):
            raise HTTPException(status_code=500, detail=str(out))
        return {"frames": out}

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
    vae, vae_world, pad_w = G._width_sharded_vae(args, cfg, dev, rank, world, args.lat_w)

    def decode(z, first):
        """Decode one chunk on every rank; rank 0 returns base64 JPEG frames."""
        lat_w = z.shape[-1]
        assert lat_w == args.lat_w, f"latent width {lat_w}, server built for {args.lat_w}"
        frames = None
        if vae is not None:
            if pad_w != lat_w:
                z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], pad_w - lat_w)], dim=-1)
            video = gather_video_width(vae.decode_chunk(
                width_shard_latent(z.to(dev), rank=rank, world=vae_world), first=first))
            if rank == 0:
                video = video[..., :lat_w * cfg.vae_stride[2]]
                frames = [b64_jpeg(to_pil(video[:, t])) for t in range(video.shape[1])]
        dist.barrier()
        return frames

    z = torch.zeros(16, args.chunk_size, args.lat_h, args.lat_w)
    decode(z, True)
    decode(z, False)
    if rank == 0:
        logging.info("VAE warmup done")

    jobs = queue.Queue()
    state = {"ready": False}
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
        msg = [None if job is None else {"z": job[0], "first": job[1]}]
        dist.broadcast_object_list(msg, src=0)
        if msg[0] is None:
            continue
        try:
            t0 = time.perf_counter()
            z = torch.load(io.BytesIO(msg[0]["z"]), map_location="cpu")
            frames = decode(z, msg[0]["first"])
            if rank == 0:
                logging.info(f"decode first={int(msg[0]['first'])} latent={tuple(z.shape)} "
                             f"-> {len(frames)} frames in {(time.perf_counter() - t0) * 1000:.0f} ms")
                job[2].put(frames)
        except Exception as e:
            logging.exception("decode failed")
            if rank == 0:
                job[2].put(e)
            raise


if __name__ == "__main__":
    main()
