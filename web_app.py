"""Browser front end for serve.py's live sessions: a steered, never-pausing stream.

Serves web/index.html and proxies /api/* to serve.py. The page starts a live
session, plays its frames from a small jitter buffer at the measured generation
rate, and steers the camera with /api/control (buttons, arrow keys or WASD).

    LBW_URL=http://lbw-v2-dit-t5:8000 VLM_URL=http://muse-glimmer-30b-vllm:8000/v1 python web_app.py

/api/describe (content eval): the last frames the relay passed through, as a short mp4, go to
Muse-Glimmer-30B on vLLM (VLM_URL, the cluster-internal Service), which says what it sees.

Binds HOST (default 127.0.0.1); the k8s deployment sets HOST=0.0.0.0 behind a
ClusterIP Service, reached with kubectl port-forward.
"""

import asyncio
import base64
import collections
import io
import json
import os
import tempfile

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
VLM_URL = os.environ.get("VLM_URL", "http://muse-glimmer-30b-vllm.default.svc.cluster.local:8000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "meta-models/Muse-Glimmer-30B")
VLM_MAX_TOKENS = int(os.environ.get("VLM_MAX_TOKENS", "384"))
DESCRIBE_FRAMES = int(os.environ.get("DESCRIBE_FRAMES", "48"))      # ~3 s of the stream
DEFAULT_QUESTION = "Describe what you see in this video in one or two sentences."
ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(ROOT, "examples")
SCENES = [
    ("05", "Dragon flight over the jungle"),
    ("02", "City alley walk"),
    ("00", "Fantasy jungle castle"),
    ("01", "Stonehenge"),
]

app = FastAPI(title="LingBot-World v2 live")
client = httpx.AsyncClient(base_url=LBW_URL, timeout=httpx.Timeout(10, read=None))
vlm = httpx.AsyncClient(timeout=httpx.Timeout(10, read=600))
# The last DESCRIBE_FRAMES frames (base64 JPEG) the session relay passed to the browser.
recent = collections.deque(maxlen=DESCRIBE_FRAMES)


@app.exception_handler(httpx.HTTPError)
async def backend_down(request, exc):
    return JSONResponse({"detail": f"backend unreachable: {exc}"}, status_code=502)


def scene_prompt(sid):
    p = os.path.join(EXAMPLES, sid, "prompt.txt")
    return open(p).read().strip() if os.path.isfile(p) else ""


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


@app.get("/api/scenes")
def scenes():
    return [{"id": s, "name": n, "prompt": scene_prompt(s)} for s, n in SCENES]


@app.get("/api/scenes/{sid}/image.jpg")
def scene_image(sid: str):
    if sid not in dict(SCENES):
        raise HTTPException(status_code=404)
    return FileResponse(os.path.join(EXAMPLES, sid, "image.jpg"))


@app.get("/api/status")
async def status():
    r = await client.get("/readiness")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/session")
async def session(request: Request):
    body = await request.json()
    if body.get("scene") not in dict(SCENES):
        raise HTTPException(status_code=400, detail="unknown scene")
    body["live"] = True
    req = client.build_request("POST", "/session", json=body)
    r = await client.send(req, stream=True)
    if r.status_code != 200:
        detail = (await r.aread()).decode(errors="replace")
        await r.aclose()
        return JSONResponse({"detail": detail}, status_code=r.status_code)

    async def relay():
        # Closing the upstream response when the browser goes away lets serve.py see the
        # disconnect and end the session.
        recent.clear()
        text = ""
        try:
            async for chunk in r.aiter_raw():
                yield chunk
                # Keep the newest frames for /api/describe (SSE "data: {...}" events).
                text += chunk.decode(errors="replace")
                while "\n\n" in text:
                    line, text = text.split("\n\n", 1)
                    if line.startswith("data: ") and '"frame"' in line:
                        try:
                            recent.append(json.loads(line[6:])["frame"])
                        except (ValueError, KeyError):
                            pass
        finally:
            await r.aclose()
    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/control")
async def control(request: Request):
    r = await client.post("/control", json=await request.json())
    return JSONResponse(r.json(), status_code=r.status_code)


def frames_to_mp4(frames_b64, fps=16):
    """Base64 JPEG frames -> mp4 bytes (H.264 via imageio-ffmpeg)."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        w = imageio.get_writer(f.name, fps=fps, codec="libx264", quality=7, macro_block_size=16)
        for b in frames_b64:
            w.append_data(np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")))
        w.close()
        return open(f.name, "rb").read()


def vlm_answer(msg):
    """Final answer of a Muse-Glimmer reply (as wan2-i2v-14b's gradio_app.describe_video)."""
    reply = msg.get("content") or ""
    if "to=user" in reply:            # no reasoning parser: content holds both channels
        return reply.rsplit("to=user", 1)[-1].strip()
    if not reply.strip() or (msg.get("reasoning_content") is None and "to=self" in reply):
        raise HTTPException(status_code=502,
                            detail=f"no final answer within {VLM_MAX_TOKENS} tokens; raise VLM_MAX_TOKENS")
    return reply.strip()


@app.post("/api/describe")
async def describe(request: Request):
    """What Muse-Glimmer sees in the last ~3 s of the stream."""
    body = await request.json() if (await request.body()) else {}
    question = (body.get("question") or DEFAULT_QUESTION)[:500]
    frames = list(recent)
    if len(frames) < 8:
        raise HTTPException(status_code=409, detail="not enough frames yet; start a session")
    mp4 = await asyncio.to_thread(frames_to_mp4, frames)
    req = {"model": VLM_MODEL, "temperature": 0, "max_tokens": VLM_MAX_TOKENS,
           "messages": [{"role": "system", "content": "Reasoning strength: low"},
                        {"role": "user", "content": [
                            {"type": "video_url", "video_url": {
                                "url": "data:video/mp4;base64," + base64.b64encode(mp4).decode("ascii")}},
                            {"type": "text", "text": question}]}]}
    try:
        r = await vlm.post(f"{VLM_URL}/chat/completions", json=req)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return JSONResponse({"detail": f"Muse-Glimmer unreachable: {exc}"}, status_code=502)
    return {"text": vlm_answer(r.json()["choices"][0]["message"]), "frames": len(frames)}


@app.post("/api/stop")
async def stop():
    r = await client.post("/stop")
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")),
                log_level="warning")
