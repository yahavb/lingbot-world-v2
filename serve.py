"""Streaming inference server for LingBot-World v2 causal_fast i2v on Trainium.

    torchrun --nproc_per_node=<N> serve.py <generate.py flags>

Every rank runs T5 and the DiT in lockstep. Rank 0 also serves HTTP from a
background thread and hands jobs to the main loop, which broadcasts each job to
the other ranks, so all device work stays on each rank's main thread.

Decode runs in one of two places:
  - VAE_URL unset: on the same ranks, between chunks (width-sharded).
  - VAE_URL set:   on a separate vae_serve.py. Rank 0 posts each latent chunk
    from a sender thread, so the DiT moves on to the next chunk while the
    previous one decodes.

One segment generates at a time. A request that arrives while a segment is
generating waits for it to finish; if several arrive, only the newest waits and
older ones are told they were superseded. With "continue": true the new
segment is conditioned on the last frame of the previous one, so successive
prompts extend the same scene.

    POST /generate/stream  {"prompt": str, "continue": bool, "seed": int}
        -> text/event-stream of
           {"frame_index", "frame"(base64 JPEG)}, ... {"done", "frames", "seconds"}

Live mode is one continuous generation: memory, camera motion and scene carry
on, and a new prompt takes effect at the next chunk.

    POST /live             {"prompt": str, "seed": int, "scene": "02"} -> same event stream,
                           until stopped; scene picks examples/<scene>/ (start image + camera path)
    POST /live/prompt      {"prompt": str}  switch the running stream's prompt
    POST /live/stop
    GET  /health
    GET  /readiness
"""

import base64
import io
import json
import logging
import os
import queue
import threading
import time

import torch
import torch.distributed as dist
from PIL import Image

import generate as G
import wan
from wan.configs import EXACT_GRID_SIZES, MAX_AREA_CONFIGS
from wan.modules.t5 import T5EncoderModel, init_t5_parallel_group
from wan.modules.vae2_1_sharded import gather_video_width, width_shard_latent

PORT = int(os.environ.get("PORT", "8000"))
VAE_URL = os.environ.get("VAE_URL")
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "90"))
IDLE_S = 60
LIVE_MAX_S = int(os.environ.get("LIVE_MAX_S", "600"))
LIVE_MAX_CHUNKS = 900


class Job:

    def __init__(self, prompt, cont, seed, live=False, scene=None):
        self.prompt = prompt
        self.cont = cont
        self.seed = seed
        self.live = live
        self.scene = scene
        self.frames = queue.Queue()


class Scheduler:
    """One running job, at most one pending; a newer pending job replaces the older."""

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = None
        self.busy = False
        self.ready = False
        self.live = None          # the running live job
        self.live_prompt = None   # prompt to switch to at the next chunk
        self.live_stop = False

    def submit(self, job):
        with self.cv:
            if self.live is not None:
                self.live_stop = True
            if self.pending is not None:
                self.pending.frames.put({"superseded": True})
                self.pending.frames.put(None)
            self.pending = job
            self.cv.notify()

    def take(self, timeout):
        with self.cv:
            if self.pending is None:
                self.cv.wait(timeout)
            if self.pending is None:
                return None
            job, self.pending = self.pending, None
            self.busy = True
            if job.live:
                self.live, self.live_prompt, self.live_stop = job, None, False
            return job

    def finish(self):
        with self.cv:
            self.busy = False
            self.live = None

    def control(self):
        """Rank 0, between chunks: (prompt to switch to or None, stop?)."""
        with self.cv:
            prompt, self.live_prompt = self.live_prompt, None
            return prompt, self.live_stop


class RemoteDecoder:
    """Rank 0: decodes chunks on the VAE server, in order, off the DiT's path."""

    def __init__(self, url):
        import requests
        self.session = requests.Session()
        self.url = url
        self.q = queue.Queue()
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, chunk_id, x0, emit):
        self.q.put((chunk_id, x0, emit))

    def drain(self):
        """Block until every submitted chunk is decoded; return and clear the first error."""
        self.q.join()
        err, self.error = self.error, None
        return err

    def _run(self):
        while True:
            chunk_id, x0, emit = self.q.get()
            try:
                if self.error is None:
                    buf = io.BytesIO()
                    torch.save(x0, buf)
                    r = self.session.post(f"{self.url}/decode",
                                          params={"first": int(chunk_id == 0)},
                                          data=buf.getvalue(), timeout=600)
                    r.raise_for_status()
                    for frame in r.json()["frames"]:
                        emit(frame)
            except Exception as e:
                logging.exception("remote decode failed")
                self.error = str(e)
            finally:
                self.q.task_done()


def to_pil(frame):
    """[C, H, W] in [-1, 1] -> PIL."""
    x = ((frame.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def b64_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_server(sched):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn

    app = FastAPI(title="LingBot-World v2 streaming")

    class StreamRequest(BaseModel):
        prompt: str
        seed: int = 42
        cont: bool = Field(False, alias="continue")

    @app.post("/generate/stream")
    def generate_stream(req: StreamRequest):
        if not sched.ready:
            raise HTTPException(status_code=503, detail="warming up")
        job = Job(req.prompt, req.cont, req.seed)
        sched.submit(job)

        def events():
            while True:
                item = job.frames.get()
                if item is None:
                    return
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    class LiveRequest(BaseModel):
        prompt: str
        seed: int = 42
        scene: str = ""

    @app.post("/live")
    def live(req: LiveRequest):
        if not sched.ready:
            raise HTTPException(status_code=503, detail="warming up")
        if req.scene and not (req.scene.isdigit() and os.path.isfile(f"examples/{req.scene}/poses.npy")):
            raise HTTPException(status_code=400, detail=f"unknown scene {req.scene!r}")
        job = Job(req.prompt, False, req.seed, live=True, scene=req.scene or None)
        sched.submit(job)

        def events():
            while True:
                item = job.frames.get()
                if item is None:
                    return
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/live/prompt")
    def live_prompt(req: LiveRequest):
        with sched.cv:
            if sched.live is None:
                raise HTTPException(status_code=409, detail="no live stream running")
            sched.live_prompt = req.prompt
        logging.info(f"live prompt queued: {req.prompt[:80]!r}")
        return {"status": "queued for the next chunk"}

    @app.post("/live/stop")
    def live_stop():
        with sched.cv:
            sched.live_stop = True
        return {"status": "stopping"}

    @app.get("/health")
    def health():
        logging.info("probe /health: healthy")
        return {"status": "healthy"}

    @app.get("/readiness")
    def readiness():
        if not sched.ready:
            logging.info("probe /readiness: warming up (503)")
            raise HTTPException(status_code=503, detail="warming up")
        logging.info(f"probe /readiness: ready, busy={sched.busy}")
        return {"status": "ready", "busy": sched.busy}

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def main():
    args = G._parse_args()
    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    G._init_logging(rank)
    cfg = G._init_distributed(args, rank, world, local_rank)
    dev = torch.device(f"neuron:{local_rank}")
    torch.set_grad_enabled(False)

    init_t5_parallel_group(world)
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer))
    t5.model.to(dev)

    pipe = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode="causal_fast",
        assets_dir=args.assets_dir,
        skip_text_encoder=True,
        tp_degree=args.tp_degree,
        sp_size=args.sp_size,
        seq_mode=args.seq_mode)

    first_img = Image.open(args.image).convert("RGB")
    max_area = MAX_AREA_CONFIGS[args.size]
    target_hw = EXACT_GRID_SIZES.get(args.size)
    pipe.prewarm(first_img, max_area=max_area, target_hw=target_hw,
                 frame_num=args.frame_num, chunk_size=args.chunk_size,
                 text_seq_len=cfg.text_len)

    remote = RemoteDecoder(VAE_URL) if (VAE_URL and rank == 0) else None

    def decode_local(chunk_id, x0, emit):
        lat_w = x0.shape[-1]
        vae, vae_world, pad_w = G._width_sharded_vae(args, cfg, dev, rank, world, lat_w)
        if vae is not None:
            z = x0
            if pad_w != lat_w:
                z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], pad_w - lat_w)], dim=-1)
            video = gather_video_width(vae.decode_chunk(
                width_shard_latent(z.to(dev), rank=rank, world=vae_world),
                first=(chunk_id == 0)))
            if rank == 0:
                video = video[..., :lat_w * cfg.vae_stride[2]]
                for t in range(video.shape[1]):
                    emit(b64_jpeg(to_pil(video[:, t])))
        dist.barrier()

    def run_segment(prompt, img, seed, emit):
        """Generate one segment on every rank; rank 0 passes each frame (base64 JPEG) to emit.

        Returns an error string on rank 0 if remote decode failed, else None.
        """
        embeds = [t.cpu() for t in t5([prompt], dev)]

        def on_chunk(chunk_id, x0):
            if VAE_URL is None:
                decode_local(chunk_id, x0, emit)
            elif remote is not None:
                remote.submit(chunk_id, x0.cpu(), emit)

        pipe.generate(
            prompt, img,
            action_path=args.action_path,
            chunk_size=args.chunk_size,
            max_area=max_area,
            target_hw=target_hw,
            denoise_steps=args.denoise_steps,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=seed,
            offload_model=False,
            max_attention_size=args.max_attention_size,
            prompt_embeds=embeds,
            return_latents=True,
            chunk_callback=on_chunk)
        return remote.drain() if remote is not None else None

    def run_live(prompt, img, seed, emit, control, action_path=None):
        """One continuous generation on every rank until control() says stop.

        control(chunk_id) runs on rank 0 and returns (new prompt or None, stop?);
        its answer is broadcast so every rank switches or stops at the same chunk.
        """
        embeds = [t.cpu() for t in t5([prompt], dev)]

        def on_chunk(chunk_id, x0):
            if VAE_URL is None:
                decode_local(chunk_id, x0, emit)
            elif remote is not None:
                remote.submit(chunk_id, x0.cpu(), emit)

        def chunk_control(chunk_id):
            c = [control(chunk_id) if rank == 0 else None]
            dist.broadcast_object_list(c, src=0)
            new_prompt, stop = c[0]
            if stop:
                return {"stop": True}
            if new_prompt is not None:
                if rank == 0:
                    logging.info(f"live: chunk {chunk_id} switches to {new_prompt[:80]!r}")
                return {"context": [t.cpu() for t in t5([new_prompt], dev)]}
            return None

        pipe.generate(
            prompt, img,
            action_path=action_path or args.action_path,
            chunk_size=args.chunk_size,
            max_area=max_area,
            target_hw=target_hw,
            denoise_steps=args.denoise_steps,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=seed,
            offload_model=False,
            max_attention_size=args.max_attention_size,
            prompt_embeds=embeds,
            return_latents=True,
            chunk_callback=on_chunk,
            chunk_control=chunk_control,
            max_chunks=LIVE_MAX_CHUNKS)
        return remote.drain() if remote is not None else None

    t0 = time.perf_counter()
    err = run_segment(args.prompt, first_img, args.base_seed, lambda f: None)
    if rank == 0:
        logging.info(f"warmup segment: {time.perf_counter() - t0:.1f} s"
                     + (f" (remote decode failed: {err})" if err else ""))

    # Live warmup: switch prompts at chunks 4, 5 and 6 so the first-call cross-attention
    # graph is compiled for every rotating KV slot, not inside a user's stream.
    t0 = time.perf_counter()
    err = run_live(args.prompt, first_img, args.base_seed, lambda f: None,
                   lambda i: (args.prompt if i in (4, 5, 6) else None, i >= 9))
    if rank == 0:
        logging.info(f"live warmup: {time.perf_counter() - t0:.1f} s"
                     + (f" (remote decode failed: {err})" if err else ""))

    sched = Scheduler()
    if rank == 0:
        threading.Thread(target=run_server, args=(sched,), daemon=True).start()
    sched.ready = True

    last_frame = None
    while True:
        # Idle ranks wait inside a collective; a heartbeat every IDLE_S keeps that
        # wait under the process-group timeout.
        job = sched.take(IDLE_S) if rank == 0 else None
        msg = [None]
        if job is not None:
            cont = job.cont and last_frame is not None
            img = last_frame if cont else first_img
            if job.scene:
                img = Image.open(f"examples/{job.scene}/image.jpg").convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            msg = [{"prompt": job.prompt, "seed": job.seed, "png": buf.getvalue(),
                    "skip_first": cont, "live": job.live,
                    "action_path": f"examples/{job.scene}" if job.scene else None}]
        dist.broadcast_object_list(msg, src=0)
        m = msg[0]
        if m is None:
            continue
        img = Image.open(io.BytesIO(m["png"])).convert("RGB")
        seen = {"n": 0, "last": None}
        t0 = time.perf_counter()

        def emit(frame):
            seen["n"] += 1
            seen["last"] = frame
            if m["skip_first"] and seen["n"] == 1:
                return
            job.frames.put({"frame_index": seen["n"] - 1, "frame": frame})

        try:
            if m["live"]:
                t_live = time.perf_counter()

                def control(chunk_id):
                    prompt, stop = sched.control()
                    return prompt, stop or time.perf_counter() - t_live > LIVE_MAX_S

                err = run_live(m["prompt"], img, m["seed"], emit, control, m["action_path"])
            else:
                err = run_segment(m["prompt"], img, m["seed"], emit)
            if rank == 0:
                if err:
                    job.frames.put({"error": err})
                else:
                    if seen["last"] is not None:
                        last_frame = Image.open(io.BytesIO(base64.b64decode(seen["last"]))).convert("RGB")
                    job.frames.put({"done": True, "frames": seen["n"],
                                    "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as e:
            logging.exception("segment failed")
            if rank == 0:
                job.frames.put({"error": str(e)})
            raise
        finally:
            if rank == 0:
                job.frames.put(None)
                sched.finish()


if __name__ == "__main__":
    main()
