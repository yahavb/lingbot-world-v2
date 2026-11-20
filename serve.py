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

A live session ({"live": true}) never pauses: it streams until the client
disconnects, /stop, a new session, or MAX_CHUNKS. /control steers it: the action
("auto" = the scene's camera path, or one of ACTIONS) drives the camera of every
chunk from the next one on, and a prompt switches the text from the next chunk.

Decode runs in one of two places:
  - VAE_URL unset: on the same ranks, between chunks (width-sharded).
  - VAE_URL set:   on a separate vae_serve.py. Rank 0 posts each latent chunk
    from a sender thread, so the DiT moves on to the next chunk while the
    previous one decodes.

    POST /session   {"prompt": str, "scene": "02", "seed": int, "live": bool}
    POST /continue  {"prompt": str}                              -> next segment
        both -> text/event-stream of {"frame_index", "frame"(base64 JPEG), "chunk",
           "action", "t"}, ... {"done", "segment", "frames", "seconds"}
           t: server wall times of that chunk -- ctl (the /control it follows was
           received), start (its denoise began), x0 (denoised), dec (frame decoded)
    POST /control   {"action": str, "prompt": str}  (both optional) -> {"action", "t"}
    POST /stop      ends the running session
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

import math

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
SAVE_LATENTS = os.environ.get("LINGBOT_SERVE_SAVE_LATENTS")
NULL_DECODE = os.environ.get("LINGBOT_NULL_DECODE", "0") == "1"
PHASE_SYNC = os.environ.get("LINGBOT_PHASE_SYNC", "0") == "1"
NULL_FRAMES = lambda c: 4 * int(os.environ.get("CHUNK_SIZE", "3")) - (3 if c == 0 else 0)   # 9, then 12 per chunk


@__import__('functools').lru_cache(None)
def _null_jpeg():
    from PIL import Image
    b = io.BytesIO()
    Image.new("RGB", (64, 64), (40, 40, 40)).save(b, "JPEG")
    return base64.b64encode(b.getvalue()).decode()
VAE_POOL = {}       # rank 0, LINGBOT_VAE_RDMA=1: {"pool": wan.vae_pool.VaePool} (the routes read it)
# LINGBOT_REANCHOR_FRAMES=N (live sessions): every ~N frames, end the generation and start a new one from the
# last frame on screen, same prompt and camera. A long rollout drifts (positions keep growing, the memory
# window loses the start image); a fresh run from the current frame continues the scene with a clean state.
REANCHOR_CHUNKS = -(-int(os.environ.get("LINGBOT_REANCHOR_FRAMES", "0")) // 12)
MAX_CHUNKS = 900    # RoPE table: 2880 latent frames / 3 per chunk, with margin
IDLE_S = 60
BACKLOG = int(os.environ.get("LIVE_BACKLOG_FRAMES", "480"))  # unread frames that end a live session

# Camera motion per latent frame (4 video frames), OpenCV axes (x right, y down, z forward):
# (yaw deg, + = right; pitch deg, + = up; translation). Sized like the example paths'
# framewise relative poses (forward ~0.75 per latent frame, turns of a few degrees).
ACTIONS = {
    "forward": (0.0, 0.0, (0.0, 0.0, 0.75)),
    "back": (0.0, 0.0, (0.0, 0.0, -0.6)),
    "left": (-5.0, 0.0, (0.0, 0.0, 0.4)),
    "right": (5.0, 0.0, (0.0, 0.0, 0.4)),
    "up": (0.0, 4.0, (0.0, 0.0, 0.4)),
    "down": (0.0, -4.0, (0.0, 0.0, 0.4)),
    "stop": (0.0, 0.0, (0.0, 0.0, 0.0)),
}



def _mem(tag, rank):
    """[MEM] rank 0 device memory after a startup phase (LINGBOT_SERVE_MEM_LOG=1)."""
    if rank == 0 and os.environ.get("LINGBOT_SERVE_MEM_LOG", "0") == "1":
        g = 1 << 30
        logging.info(f"[MEM] {tag}: allocated {torch.neuron.memory_allocated() / g:.2f} GiB, "
                     f"reserved {torch.neuron.memory_reserved() / g:.2f} GiB")

def action_poses(name, n):
    """[n, 4, 4] framewise relative poses for ACTIONS[name], or None for "auto"."""
    if name not in ACTIONS:
        return None
    yaw, pitch, t = ACTIONS[name]
    a, b = math.radians(yaw), math.radians(pitch)
    ry = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rx = torch.tensor([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    m = torch.eye(4)
    m[:3, :3] = ry @ rx
    m[:3, 3] = torch.tensor(t)
    return m.expand(n, 4, 4).clone()


class Job:
    """One segment: its frames go to its own event stream."""

    def __init__(self, prompt, seed=42, scene=None, live=False):
        self.prompt = prompt
        self.seed = seed
        self.scene = scene
        self.live = live
        self.on_backlog = None  # called once when a live client stops reading
        self.frames = queue.Queue()
        self.count = 0
        self.t0 = None

    def emit(self, frame, chunk=None, action=None, t=None):
        self.count += 1
        self.last_frame = frame          # base64 JPEG: the start image of the next re-anchored run
        self.frames.put({"frame_index": self.count - 1, "frame": frame,
                         "chunk": chunk, "action": action, "t": t})
        # A dead connection behind a proxy (kubectl port-forward) may never look
        # disconnected; a live client that stops reading ends the session instead.
        if self.live and self.on_backlog is not None and self.frames.qsize() > BACKLOG:
            logging.info(f"live client not reading ({self.frames.qsize()} frames queued): stopping")
            self.on_backlog(self)
            self.on_backlog = None

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
        self.current = None     # job whose stream is open
        self.action = "auto"    # live camera control
        self.action_t = None    # when it was received
        self.prompt = None      # live prompt switch, taken once

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
            self.action, self.action_t, self.prompt = "auto", None, None
            self.cv.notify_all()

    def control(self, action=None, prompt=None):
        with self.cv:
            if action is not None:
                self.action, self.action_t = action, time.time()
            if prompt:
                self.prompt = prompt
            return self.action

    def take_control(self):
        """(action, prompt or None, when the action arrived) for the next chunk."""
        with self.cv:
            p, self.prompt = self.prompt, None
            return self.action, p, self.action_t

    def stop_job(self, job):
        """End the session if `job` is still the one streaming (its client went away)."""
        with self.cv:
            if job is self.pending:
                self.pending = None
            elif job is self.current and self.running:
                self.stop = True
            self.cv.notify_all()

    def queue_continue(self, job):
        with self.cv:
            if not self.running:
                return False
            self._drop(self.cont, "a newer continue arrived")
            self.cont = job
            self.current = job
            self.cv.notify_all()
            return True

    def take_session(self, timeout):
        with self.cv:
            if self.pending is None:
                self.cv.wait(timeout)
            job, self.pending = self.pending, None
            if job is not None:
                self.running, self.stop = True, False
                self.current = job
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
            self.current = None
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

    def submit(self, first, x0, emit, url=None, seq=None):
        self.q.put((first, x0, emit, url or self.url, seq))

    def drain(self):
        """Block until every submitted chunk is decoded; return and clear the first error."""
        self.q.join()
        err, self.error = self.error, None
        return err

    def _run(self):
        while True:
            first, x0, emit, url, seq = self.q.get()
            try:
                if self.error is None:
                    buf = io.BytesIO()
                    if x0 is not None:              # None: the latents travel over RDMA
                        torch.save(x0, buf)
                    # stream=1: the VAE server sends each decode step's frames (NDJSON) as
                    # soon as they exist, not the whole chunk at the end.
                    with self.session.post(f"{url}/decode", params={"first": int(first), "stream": 1, "rdma": int(x0 is None),
                                                   **({"seq": seq} if seq is not None else {})},
                                           data=buf.getvalue(), timeout=600, stream=True) as r:
                        r.raise_for_status()
                        # Big reads: iter_lines' default 512 B pieces keep this thread busy in
                        # Python, and rank 0's main thread, which issues the DiT, waits on the GIL.
                        for line in r.iter_lines(chunk_size=1 << 16):
                            if not line:
                                continue
                            part = json.loads(line)
                            if "error" in part:
                                raise RuntimeError(part["error"])
                            for frame in part["frames"]:
                                emit(frame)
            except Exception as e:
                logging.exception("remote decode failed")
                self.error = str(e)
            finally:
                self.q.task_done()


def load_image(path):
    """A start image as every encode sees it: one memory block under LINGBOT_IMG_BLOCK=1, so its
    encode-cache key (which records the pixel layout) is the same wherever it was loaded."""
    img = Image.open(path).convert("RGB")
    if os.environ.get("LINGBOT_IMG_BLOCK", "0") == "1":
        from wan.image2video import single_block_image
        img = single_block_image(img)
    return img


def to_pil(frame):
    """[C, H, W] in [-1, 1] -> PIL."""
    x = ((frame.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def b64_jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_server(sched):
    import asyncio
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="LingBot-World v2 streaming")

    class SessionRequest(BaseModel):
        prompt: str
        seed: int = 42
        scene: str = ""
        live: bool = False

    class ContinueRequest(BaseModel):
        prompt: str

    class ControlRequest(BaseModel):
        action: str | None = None
        prompt: str | None = None

    def stream(job, request):
        def get():
            try:
                return job.frames.get(timeout=1)
            except queue.Empty:
                return "idle"

        async def events():
            try:
                while True:
                    item = await asyncio.to_thread(get)
                    if item is None:
                        return
                    if item == "idle":
                        if await request.is_disconnected():
                            return
                        continue
                    yield f"data: {json.dumps(item)}\n\n"
            finally:
                if job.live:
                    sched.stop_job(job)
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/vaes")
    def vaes():
        """The VAEs this DiT can use now, and the session's current one."""
        pool = VAE_POOL.get("pool")
        if pool is None:
            return {"rdma": False}
        cur = pool.current
        return {"rdma": True, "available": pool.available(), "listed": dict(pool.endpoints),
                "set_aside": {k: round(time.time() - v, 1) for k, v in pool.bad.items()},
                "current": None if cur is None else {"name": cur[0], "ip": cur[1]}}

    @app.post("/session")
    def session(req: SessionRequest, request: Request):
        if not sched.ready:
            raise HTTPException(status_code=503, detail="warming up")
        pool = VAE_POOL.get("pool")
        if pool is not None and not pool.available():
            raise HTTPException(status_code=503, detail="VAE_SERVICE_UNAVAILABLE: no lbw-vae pod is ready")
        if req.scene and not (req.scene.isdigit() and os.path.isfile(f"examples/{req.scene}/poses.npy")):
            raise HTTPException(status_code=400, detail=f"unknown scene {req.scene!r}")
        job = Job(req.prompt, req.seed, req.scene or None, req.live)
        job.on_backlog = sched.stop_job
        sched.start(job)
        return stream(job, request)

    @app.post("/continue")
    def cont(req: ContinueRequest, request: Request):
        job = Job(req.prompt)
        if not sched.queue_continue(job):
            raise HTTPException(status_code=409, detail="no session - start one first")
        logging.info(f"continue queued: {req.prompt[:80]!r}")
        return stream(job, request)

    @app.post("/control")
    def control(req: ControlRequest):
        if req.action is not None and req.action != "auto" and req.action not in ACTIONS:
            raise HTTPException(status_code=400, detail=f"unknown action {req.action!r}")
        action = sched.control(req.action, req.prompt)
        logging.info(f"control: action={action}" + (f" prompt={req.prompt[:60]!r}" if req.prompt else ""))
        return {"action": action, "t": time.time()}

    @app.post("/stop")
    def stop():
        with sched.cv:
            sched.stop = sched.running
            sched.cv.notify_all()
        return {"stopped": sched.running}

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

    # Every scene's start image, VAE-encoded once now, before T5 and the DiT take the device
    # memory: a session's first chunk then needs no encode (11-15 s per new image).
    from wan.image2video import pre_encode_condition
    scene_imgs = {}
    t0 = time.perf_counter()
    # LINGBOT_SCENES=03,00: pre-encode only these scenes (each 720p encode is ~2 GB of device memory per rank).
    # LINGBOT_PRE_ENCODE_RANK0=1: rank 0 encodes, every other rank receives the latents: at 720p all four ranks
    # of a chip encoding at once ran out of device memory.
    only = [x for x in os.environ.get("LINGBOT_SCENES", "").split(",") if x]
    rank0_only = os.environ.get("LINGBOT_PRE_ENCODE_RANK0", "0") == "1"
    from wan.image2video import _PRE_ENCODED
    for sid in sorted(os.listdir("examples")):
        d = f"examples/{sid}"
        if not (sid.isdigit() and os.path.isfile(f"{d}/image.jpg") and os.path.isfile(f"{d}/poses.npy")):
            continue
        if only and sid not in only:
            continue
        scene_imgs[sid] = load_image(f"{d}/image.jpg")
        if rank0_only and rank != 0:
            continue
        pre_encode_condition(os.path.join(args.ckpt_dir, cfg.vae_checkpoint), dev, scene_imgs[sid], d,
                             args.frame_num, args.chunk_size, MAX_AREA_CONFIGS[args.size],
                             EXACT_GRID_SIZES.get(args.size), cfg.vae_stride, cfg.patch_size)
    if rank0_only:
        # through a host file (every rank is in this pod): broadcast_object_list stages the pickle in device
        # memory, which the encode had just filled (720p: 1.4 GB allocation failed)
        shared = "/dev/shm/lbw_pre_encoded.pt"
        if rank == 0:
            torch.save({k: v.cpu() for k, v in _PRE_ENCODED.items()}, shared)
        dist.barrier()
        if rank != 0:
            _PRE_ENCODED.update(torch.load(shared))
    dist.barrier()
    if rank == 0:
        logging.info(f"pre-encoded scenes {sorted(scene_imgs)} in {time.perf_counter() - t0:.1f} s")
    _mem("after scene pre-encode", rank)

    init_t5_parallel_group(world)
    t5 = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer))
    t5.model.to(dev)
    _mem("after T5 load", rank)

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
    _mem("after DiT pipeline load", rank)

    first_img = load_image(args.image)

    # T5 embeddings by prompt. Every rank runs the same prompts in the same order, so every
    # rank's cache holds the same keys.
    t5_cache = {}

    def encode_prompt(p):
        if p not in t5_cache:
            if len(t5_cache) >= 64:
                t5_cache.pop(next(iter(t5_cache)))
            # Zero-padded to text_len here, as the model pads it: on the device that pad is an
            # eager op shaped by the prompt's token count, a new graph (~3.7 s) per new length.
            t5_cache[p] = [torch.cat([u, u.new_zeros(cfg.text_len - u.size(0), u.size(1))])
                           for u in (t.cpu() for t in t5([p], dev))]
        return t5_cache[p]

    t0 = time.perf_counter()
    for sid in sorted(scene_imgs):
        f = f"examples/{sid}/prompt.txt"
        if os.path.isfile(f):
            encode_prompt(open(f).read().strip())
    if rank == 0:
        logging.info(f"encoded {len(t5_cache)} scene prompts in {time.perf_counter() - t0:.1f} s")
    max_area = MAX_AREA_CONFIGS[args.size]
    target_hw = EXACT_GRID_SIZES.get(args.size)
    pipe.prewarm(first_img, max_area=max_area, target_hw=target_hw,
                 frame_num=args.frame_num, chunk_size=args.chunk_size,
                 text_seq_len=cfg.text_len)

    remote = RemoteDecoder(VAE_URL) if (VAE_URL and rank == 0) else None
    # LINGBOT_VAE_RDMA=1 (with VAE_URL): each chunk's latents go device to device over EFA (wan/rdma_link.py);
    # the HTTP request carries only the chunk's metadata and brings the frames back.
    rdma = None
    if remote is not None and os.environ.get("LINGBOT_VAE_RDMA", "0") == "1":
        # the VAE is chosen per session from the Ready lbw-vae pods, with failover (wan/vae_pool.py)
        from wan.vae_pool import VaePool, VaeUnavailable
        rdma = VaePool()
        VAE_POOL["pool"] = rdma

    # LINGBOT_VAE_FUSED=1 (world > 1): the inline decode uses the fused decoder in decode_chunk mode,
    # the same path vae_serve.py serve_fused streams with (2-D tiles with LINGBOT_VAE_2D=1). The plain
    # width-sharded path below failed to compile its per-frame gather at 480x768 on 16 LNC2 ranks
    # (neuronx-cc NCC_ITIN902 in vae2_1_sharded.gather_video_width).
    _fused_vae = {}

    def decode_local_fused(chunk_id, x0, emit):
        from wan.modules import vae2_1_fused as F
        if not _fused_vae:
            from wan.modules.vae2_1 import Wan2_1_VAE
            # the VAE's compiler options apply only inside decode_local_fused (F.vae_cc_scope below); the old
            # process-wide F.apply_vae_cc_flags() leaked them into every later DiT compile
            extra = os.environ.get("LINGBOT_VAE_CC_FLAGS", "").replace("+", " ").strip()
            with F.vae_cc_scope():
                v = Wan2_1_VAE(vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint), device=dev,
                               dtype=torch.bfloat16)
                _fused_vae.update(vae=v, fdec=F.FusedWidthDecoder(v.model))
            if rank == 0:
                logging.info(f"inline fused VAE decoder, {'2-D tiles' if F._TWO_D else 'width slices'}"
                             + (f", compiler flags + {extra}" if extra else ""))
        v, fdec = _fused_vae["vae"], _fused_vae["fdec"]
        z = x0.to(dev)
        local = F.shard_latent_2d(z) if F._TWO_D else width_shard_latent(z, rank=rank, world=world)

        def step(out):
            out = out.float().clamp_(-1, 1).squeeze(0)
            video = F.gather_video_2d(out) if F._TWO_D else gather_video_width(out)
            if rank == 0:
                for t in range(video.shape[1]):
                    emit(b64_jpeg(to_pil(video[:, t])))
        with F.vae_cc_scope():      # VAE compiler options for the decoder's graphs only, never the DiT's
            fdec.decode_chunk(local.unsqueeze(0), v.scale, chunk_id == 0, on_step=step)
        dist.barrier()

    def decode_local(chunk_id, x0, emit):
        if world > 1 and os.environ.get("LINGBOT_VAE_FUSED", "0") == "1":
            return decode_local_fused(chunk_id, x0, emit)
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

        control(chunk_id), rank 0 only, returns ("go", new prompt or None, action),
        ("wait", None, None) or ("stop", None, None). Its answer is broadcast, so every
        rank switches, steers, pauses or stops at the same chunk. state["emit"] says
        where that chunk's frames go.
        """
        embeds = encode_prompt(prompt)
        actions, starts = {}, {}

        def on_chunk(chunk_id, x0):
            job_emit, action = state["emit"], actions.pop(chunk_id, None)
            x0_dev = x0
            x0 = x0.cpu() if (rank == 0 and (rdma is None or SAVE_LATENTS)) else x0
            if rank == 0 and SAVE_LATENTS:
                # LINGBOT_SERVE_SAVE_LATENTS=<dir>: every chunk's latents, to decode a live session
                # offline (decode_latents_width_sharded.py) and compare with the inline VAE's frames
                os.makedirs(SAVE_LATENTS, exist_ok=True)
                torch.save(x0.clone(), f"{SAVE_LATENTS}/c{chunk_id:04d}.pt")
            times = {"ctl": ctl_times.pop(chunk_id, None), "start": starts.pop(chunk_id, None),
                     "x0": time.time()}

            def emit(frame, _c=chunk_id + state.get("chunk_base", 0)):
                job_emit(frame, _c, action, dict(times, dec=time.time()))
            if NULL_DECODE:
                # LINGBOT_NULL_DECODE=1: DiT only (phase characterization): no VAE; rank 0 emits one constant
                # frame per video frame once the latent exists, so the client measures the DiT's pace
                if rank == 0:
                    x0_dev.sum().item()               # the chunk's latents are really done
                    for _ in range(NULL_FRAMES(chunk_id)):
                        emit(_null_jpeg())
            elif VAE_URL is None and PHASE_SYNC:
                # LINGBOT_PHASE_SYNC=1 (measurement): wait for the device before and after the inline decode, so
                # [PHASE] gives each chunk's true DiT time (start -> device done) and VAE time on the same ranks;
                # with async execution a decode's device work otherwise lands inside the next chunk's DiT time
                torch.neuron.synchronize()
                t_dit = time.time()
                decode_local(chunk_id, x0, emit)
                torch.neuron.synchronize()
                if rank == 0:
                    st = times.get("start") or t_dit
                    logging.info(f"[PHASE] chunk {chunk_id} dit_ms {1000 * (t_dit - st):.1f} "
                                 f"vae_ms {1000 * (time.time() - t_dit):.1f}")
            elif VAE_URL is None:
                decode_local(chunk_id, x0, emit)
            elif remote is not None and rdma is not None:
                try:
                    rdma.send(remote, chunk_id == 0, x0_dev.float().contiguous(), emit)
                except VaeUnavailable as e:
                    logging.warning(f"chunk {chunk_id}: no VAE, frames dropped ({e})")
            elif remote is not None:
                remote.submit(chunk_id == 0, x0, emit)

        def chunk_control(chunk_id):
            while True:
                c = [control(chunk_id) if rank == 0 else None]
                dist.broadcast_object_list(c, src=0)
                verb, new_prompt, action = c[0]
                if verb == "wait":
                    continue
                if verb == "stop":
                    return {"stop": True}
                actions[chunk_id] = action or "auto"
                starts[chunk_id] = time.time()
                out = {"camera": action_poses(action, args.chunk_size)}
                if new_prompt is not None:
                    out["context"] = encode_prompt(new_prompt)
                return out

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

    state = {"emit": lambda f, c=None, a=None, t=None: None}
    ctl_times = {}  # rank 0: chunk -> when the /control it follows arrived

    # Warmup: a 3-segment session, switching prompts at both boundaries and steering the
    # camera in between, so every graph a real session uses -- including the prompt-switch
    # and camera-control ones -- compiles here.
    warm_actions = ["auto"] + list(ACTIONS)
    t0 = time.perf_counter()
    if os.environ.get("LINGBOT_SERVE_EMPTY_CACHE", "0") == "1":
        # startup transients (scene pre-encode, T5/DiT load) stay reserved in the caching allocator;
        # hand the free blocks back before the first session allocates the KV caches
        torch.neuron.synchronize()
        torch.neuron.empty_cache()
        _mem("after empty_cache", rank)
    run_session(args.prompt, first_img, args.base_seed, args.action_path,
                lambda i: ("stop", None, None) if i >= 3 * SEG_CHUNKS else
                ("go", args.prompt if i in (SEG_CHUNKS, 2 * SEG_CHUNKS) else None,
                 warm_actions[i % len(warm_actions)]))
    if remote is not None:
        err = remote.drain()
        if err:
            logging.info(f"warmup: remote decode failed: {err}")
    if rank == 0:
        logging.info(f"warmup session: {time.perf_counter() - t0:.1f} s")
    # The first session after the warmup ran ~4 s per chunk for its first ~6 chunks (then 0.42 s),
    # on any scene; a second short one, steered, on another scene's pre-encoded image takes that.
    t0 = time.perf_counter()
    sid = next((k for k in sorted(scene_imgs) if f"examples/{k}" != os.path.normpath(args.action_path)), None)
    if sid is not None:
        run_session(args.prompt, scene_imgs[sid], args.base_seed, f"examples/{sid}",
                    lambda i: ("stop", None, None) if i >= 12 else
                    ("go", None, warm_actions[i % len(warm_actions)]))
        if remote is not None:
            remote.drain()
        if rank == 0:
            logging.info(f"second warmup session (scene {sid}): {time.perf_counter() - t0:.1f} s")

    sched = Scheduler()
    if rank == 0:
        threading.Thread(target=run_server, args=(sched,), daemon=True).start()
    sched.ready = True

    while True:
        job = sched.take_session(IDLE_S) if rank == 0 else None
        msg = [None]
        if job is not None:
            scene = job.scene or None
            msg = [{"prompt": job.prompt, "seed": job.seed, "scene": scene}]
        dist.broadcast_object_list(msg, src=0)
        m = msg[0]
        if m is None:
            continue
        # Every rank has the repo's scenes loaded; only the scene id is broadcast.
        img = scene_imgs.get(m["scene"], first_img)
        m["action_path"] = f"examples/{m['scene']}" if m["scene"] else args.action_path

        live = rank == 0 and job.live
        seg = {"job": job, "n": 1, "end": MAX_CHUNKS if live else SEG_CHUNKS,
               "open": True, "idle": None}
        ctl_times.clear()
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
                if sched.stop:
                    return ("stop", None, None)
                action, new_prompt, t_ctl = sched.take_control()
                ctl_times[chunk_id] = t_ctl
                return ("go", new_prompt, action)
            if seg["open"]:
                close_segment()
            nxt, stop = sched.take_continue(IDLE_S)
            if stop:
                return ("stop", None, None)
            if nxt is None:
                over = time.perf_counter() - seg["idle"] > SESSION_IDLE_S
                return ("stop", None, None) if over else ("wait", None, None)
            seg.update(job=nxt, n=seg["n"] + 1, end=chunk_id + SEG_CHUNKS, open=True)
            nxt.t0 = time.perf_counter()
            state["emit"] = nxt.emit
            logging.info(f"segment {seg['n']}: {nxt.prompt[:80]!r}")
            action, _, ctl_times[chunk_id] = sched.take_control()
            return ("go", nxt.prompt, action)

        try:
            cur_img, state["chunk_base"] = img, 0
            while True:
                rean = {"hit": False}

                def control_r(chunk_id):
                    if live and REANCHOR_CHUNKS and chunk_id >= REANCHOR_CHUNKS and not sched.stop:
                        rean["hit"] = True
                        return ("stop", None, None)
                    r = control(chunk_id)
                    if r[0] == "go" and r[1]:
                        m["prompt"] = r[1]          # a live prompt switch carries over to the next run
                    return r
                run_session(m["prompt"], cur_img, m["seed"], m["action_path"], control_r)
                # re-anchor: every rank learns it, then starts a new run from rank 0's last frame
                flag = [bool(rank == 0 and rean["hit"] and seg["open"] and not sched.stop)]
                dist.broadcast_object_list(flag, src=0)
                if not flag[0]:
                    break
                if rank == 0 and remote is not None:
                    remote.drain()                  # the last chunk's frames have arrived
                data = [(seg["job"].last_frame, m["prompt"]) if rank == 0 else None]
                dist.broadcast_object_list(data, src=0)
                path = f"/tmp/reanchor_{rank}.jpg"
                with open(path, "wb") as f:
                    f.write(base64.b64decode(data[0][0]))
                cur_img, m["prompt"] = load_image(path), data[0][1]
                state["chunk_base"] += REANCHOR_CHUNKS
                if rank == 0:
                    logging.info(f"re-anchor: new run from the last frame (chunk {state['chunk_base']})")
            if rank == 0 and seg["open"]:
                close_segment()
        except Exception as e:
            logging.exception("session failed")
            if rank == 0 and seg["open"]:
                seg["job"].close(seg["n"], str(e))
            raise
        finally:
            if rank == 0:
                state["emit"] = lambda f, c=None, a=None, t=None: None
                sched.end_session()
                logging.info("session ended")


if __name__ == "__main__":
    main()
