"""Streaming inference server for LingBot-World v2 causal_fast i2v on Trainium.

    torchrun --nproc_per_node=<N> serve.py <generate.py flags>

Every rank runs T5 and the DiT in lockstep. Rank 0 also serves HTTP from a
background thread and hands work to the main loop, which broadcasts it to the
other ranks, so all device work stays on each rank's main thread.

A session is one continuous generation, delivered in segments of SEG_CHUNKS
chunks (81 frames for the first). After a segment the generation pauses with its
state intact -- KV cache, decode cache, camera motion -- and the next /continue
resumes it, optionally with a new prompt. A /continue sent while a segment is
still generating is queued and starts the moment that segment's last chunk is
denoised. Only the newest queued /continue is kept.

Decode runs in one of two places:
  - VAE_URL unset: on the same ranks, between chunks (width-sharded).
  - VAE_URL set:   on a separate vae_serve.py. Rank 0 posts each latent chunk
    from a sender thread, so the DiT moves on to the next chunk while the
    previous one decodes.

    POST /session   {"prompt": str, "scene": "02", "seed": int}  -> first segment
    POST /continue  {"prompt": str}                              -> next segment
        both -> text/event-stream of
           {"frame_index", "frame"(base64 JPEG)}, ... {"done", "segment", "frames", "seconds"}
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
SEG_CHUNKS = int(os.environ.get("SEG_CHUNKS", "7"))
SESSION_IDLE_S = int(os.environ.get("SESSION_IDLE_S", "300"))
MAX_CHUNKS = 900    # RoPE table: 2880 latent frames / 3 per chunk, with margin
IDLE_S = 60


class Job:
    """One segment: its frames go to its own event stream."""

    def __init__(self, prompt, seed=42, scene=None):
        self.prompt = prompt
        self.seed = seed
        self.scene = scene
        self.frames = queue.Queue()
        self.count = 0
        self.t0 = None

    def emit(self, frame):
        self.count += 1
        self.frames.put({"frame_index": self.count - 1, "frame": frame})

    def close(self, segment, err=None):
        if err:
            self.frames.put({"error": err})
        else:
            self.frames.put({"done": True, "segment": segment, "frames": self.count,
                             "seconds": round(time.perf_counter() - self.t0, 2)})
        self.frames.put(None)


class Scheduler:
    """Rank 0: a pending new session, and the queued continue for the running one."""

    def __init__(self):
        self.cv = threading.Condition()
        self.ready = False
        self.pending = None     # new session waiting to start
        self.running = False    # a session is generating or paused
        self.cont = None        # newest queued continue
        self.stop = False       # end the running session

    def _drop(self, job, why):
        if job is not None:
            job.frames.put({"superseded": why})
            job.frames.put(None)

    def start(self, job):
        with self.cv:
            self._drop(self.pending, "a newer session started")
            self._drop(self.cont, "a newer session started")
            self.pending, self.cont = job, None
            self.stop = self.running
            self.cv.notify_all()

    def queue_continue(self, job):
        with self.cv:
            if not self.running:
                return False
            self._drop(self.cont, "a newer continue arrived")
            self.cont = job
            self.cv.notify_all()
            return True

    def take_session(self, timeout):
        with self.cv:
            if self.pending is None:
                self.cv.wait(timeout)
            job, self.pending = self.pending, None
            if job is not None:
                self.running, self.stop = True, False
            return job

    def take_continue(self, timeout):
        """(job or None, stop?) -- waits up to timeout for a continue."""
        with self.cv:
            if self.cont is None and not self.stop:
                self.cv.wait(timeout)
            job, self.cont = self.cont, None
            return job, self.stop

    def end_session(self):
        with self.cv:
            self.running = False
            self._drop(self.cont, "the session ended")
            self.cont = None


class RemoteDecoder:
    """Rank 0: decodes chunks on the VAE server, in order, off the DiT's path."""

    def __init__(self, url):
        import requests
        self.session = requests.Session()
        self.url = url
        self.q = queue.Queue()
        self.error = None
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, first, x0, emit):
        self.q.put((first, x0, emit))

    def drain(self):
        """Block until every submitted chunk is decoded; return and clear the first error."""
        self.q.join()
        err, self.error = self.error, None
        return err

    def _run(self):
        while True:
            first, x0, emit = self.q.get()
            try:
                if self.error is None:
                    buf = io.BytesIO()
                    torch.save(x0, buf)
                    r = self.session.post(f"{self.url}/decode", params={"first": int(first)},
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
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="LingBot-World v2 streaming")

    class SessionRequest(BaseModel):
        prompt: str
        seed: int = 42
        scene: str = ""

    class ContinueRequest(BaseModel):
        prompt: str

    def stream(job):
        def events():
            while True:
                item = job.frames.get()
                if item is None:
                    return
                yield f"data: {json.dumps(item)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/session")
    def session(req: SessionRequest):
        if not sched.ready:
            raise HTTPException(status_code=503, detail="warming up")
        if req.scene and not (req.scene.isdigit() and os.path.isfile(f"examples/{req.scene}/poses.npy")):
            raise HTTPException(status_code=400, detail=f"unknown scene {req.scene!r}")
        job = Job(req.prompt, req.seed, req.scene or None)
        sched.start(job)
        return stream(job)

    @app.post("/continue")
    def cont(req: ContinueRequest):
        job = Job(req.prompt)
        if not sched.queue_continue(job):
            raise HTTPException(status_code=409, detail="no session - start one first")
        logging.info(f"continue queued: {req.prompt[:80]!r}")
        return stream(job)

    @app.get("/health")
    def health():
        logging.info("probe /health: healthy")
        return {"status": "healthy"}

    @app.get("/readiness")
    def readiness():
        if not sched.ready:
            logging.info("probe /readiness: warming up (503)")
            raise HTTPException(status_code=503, detail="warming up")
        logging.info(f"probe /readiness: ready, session={sched.running}")
        return {"status": "ready", "session": sched.running}

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

    def run_session(prompt, img, seed, action_path, control):
        """One continuous generation on every rank.

        control(chunk_id), rank 0 only, returns ("go", new prompt or None),
        ("wait", None) or ("stop", None). Its answer is broadcast, so every rank
        switches, pauses or stops at the same chunk. emit_of(chunk_id) says where
        that chunk's frames go.
        """
        embeds = [t.cpu() for t in t5([prompt], dev)]

        def on_chunk(chunk_id, x0):
            emit = state["emit"]
            if VAE_URL is None:
                decode_local(chunk_id, x0, emit)
            elif remote is not None:
                remote.submit(chunk_id == 0, x0.cpu(), emit)

        def chunk_control(chunk_id):
            while True:
                c = [control(chunk_id) if rank == 0 else None]
                dist.broadcast_object_list(c, src=0)
                action, new_prompt = c[0]
                if action == "wait":
                    continue
                if action == "stop":
                    return {"stop": True}
                if new_prompt is not None:
                    return {"context": [t.cpu() for t in t5([new_prompt], dev)]}
                return None

        pipe.generate(
            prompt, img,
            action_path=action_path,
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
            max_chunks=MAX_CHUNKS)

    state = {"emit": lambda f: None}

    # Warmup: a 3-segment session, switching prompts at both boundaries, so every
    # graph a real session uses -- including the prompt-switch ones -- compiles here.
    t0 = time.perf_counter()
    run_session(args.prompt, first_img, args.base_seed, args.action_path,
                lambda i: ("stop", None) if i >= 3 * SEG_CHUNKS else
                ("go", args.prompt if i in (SEG_CHUNKS, 2 * SEG_CHUNKS) else None))
    if remote is not None:
        err = remote.drain()
        if err:
            logging.info(f"warmup: remote decode failed: {err}")
    if rank == 0:
        logging.info(f"warmup session: {time.perf_counter() - t0:.1f} s")

    sched = Scheduler()
    if rank == 0:
        threading.Thread(target=run_server, args=(sched,), daemon=True).start()
    sched.ready = True

    while True:
        job = sched.take_session(IDLE_S) if rank == 0 else None
        msg = [None]
        if job is not None:
            scene = job.scene or None
            img = Image.open(f"examples/{scene}/image.jpg").convert("RGB") if scene else first_img
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            msg = [{"prompt": job.prompt, "seed": job.seed, "png": buf.getvalue(),
                    "action_path": f"examples/{scene}" if scene else args.action_path}]
        dist.broadcast_object_list(msg, src=0)
        m = msg[0]
        if m is None:
            continue
        img = Image.open(io.BytesIO(m["png"])).convert("RGB")

        seg = {"job": job, "n": 1, "end": SEG_CHUNKS, "open": True, "idle": None}
        if rank == 0:
            job.t0 = time.perf_counter()
            state["emit"] = job.emit

        def close_segment():
            err = remote.drain() if remote is not None else None
            seg["job"].close(seg["n"], err)
            seg["open"] = False
            seg["idle"] = time.perf_counter()
            logging.info(f"segment {seg['n']} done: {seg['job'].count} frames"
                         + (f", error {err}" if err else ""))

        def control(chunk_id):
            if chunk_id < seg["end"]:
                return ("stop", None) if sched.stop else ("go", None)
            if seg["open"]:
                close_segment()
            nxt, stop = sched.take_continue(IDLE_S)
            if stop:
                return ("stop", None)
            if nxt is None:
                over = time.perf_counter() - seg["idle"] > SESSION_IDLE_S
                return ("stop", None) if over else ("wait", None)
            seg.update(job=nxt, n=seg["n"] + 1, end=chunk_id + SEG_CHUNKS, open=True)
            nxt.t0 = time.perf_counter()
            state["emit"] = nxt.emit
            logging.info(f"segment {seg['n']}: {nxt.prompt[:80]!r}")
            return ("go", nxt.prompt)

        try:
            run_session(m["prompt"], img, m["seed"], m["action_path"], control)
            if rank == 0 and seg["open"]:
                close_segment()
        except Exception as e:
            logging.exception("session failed")
            if rank == 0 and seg["open"]:
                seg["job"].close(seg["n"], str(e))
            raise
        finally:
            if rank == 0:
                state["emit"] = lambda f: None
                sched.end_session()
                logging.info("session ended")


if __name__ == "__main__":
    main()
