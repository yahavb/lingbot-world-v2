"""Browser front end for serve.py's live sessions: a steered, never-pausing stream.

Serves web/index.html and proxies /api/* to serve.py. The page starts a live
session, plays its frames from a small jitter buffer at the measured generation
rate, and steers the camera with /api/control (buttons, arrow keys or WASD).

    LBW_URL=http://lbw-v2-dit-t5:8000 python web_app.py

Binds HOST (default 127.0.0.1); the k8s deployment sets HOST=0.0.0.0 behind a
ClusterIP Service, reached with kubectl port-forward.
"""

import os

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
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
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
    return StreamingResponse(relay(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/control")
async def control(request: Request):
    r = await client.post("/control", json=await request.json())
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/api/stop")
async def stop():
    r = await client.post("/stop")
    return JSONResponse(r.json(), status_code=r.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "7860")),
                log_level="warning")
