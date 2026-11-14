#!/usr/bin/env python3
"""S9d: key press -> new frames, and stream fps, against serve.py (talk-live-deploy.yaml).

  kubectl port-forward --address 127.0.0.1 svc/lbw-talk-live 8000:8000 &
  python3 live_latency.py --url http://127.0.0.1:8000 --presses 8

Starts a live session (scene 03), lets it run --settle seconds, then presses alternating actions.
Each press is POST /control; its latency is the client time from sending the press to receiving
the first streamed frame tagged with that action (what a browser user waits for, through the same
port-forward). fps = frames received per second after --settle. Standard library only.
"""
import argparse
import http.client
import json
import statistics
import threading
import time
from urllib.parse import urlparse

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--scene", default="03")
ap.add_argument("--presses", type=int, default=8)
ap.add_argument("--settle", type=float, default=15.0)
ap.add_argument("--gap", type=float, default=4.0, help="seconds after a press is seen before the next")
args = ap.parse_args()
u = urlparse(args.url)


def post(path, body, timeout=30):
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return c


frames = []          # (client receive time, action, chunk)
lock = threading.Lock()
done = threading.Event()


def reader():
    c = post("/session", {"prompt": "", "scene": args.scene, "seed": 42, "live": True}, timeout=600)
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:300])
    buf = b""
    while not done.is_set():
        line = r.readline()
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        ev = json.loads(line[6:])
        if "frame_index" in ev:
            with lock:
                frames.append((time.time(), ev.get("action"), ev.get("chunk")))
        elif "error" in ev or "done" in ev or "superseded" in ev:
            print("stream:", {k: v for k, v in ev.items() if k != "frame"})
            break
    done.set()


th = threading.Thread(target=reader, daemon=True)
th.start()
t_start = time.time()
while not frames and not done.is_set():
    time.sleep(0.05)
print(f"first frame after {time.time() - t_start:.2f} s")
time.sleep(args.settle)

lat = []
acts = ["left", "right"]
for i in range(args.presses):
    a = acts[i % 2]
    with lock:
        n0 = len(frames)
    t_press = time.time()
    c = post("/control", {"action": a})
    resp = json.loads(c.getresponse().read())
    seen = None
    while seen is None and not done.is_set() and time.time() - t_press < 30:
        with lock:
            for t, act, _ in frames[n0:]:
                if act == a and t >= t_press:
                    seen = t
                    break
        time.sleep(0.01)
    if seen is None:
        print(f"press {i} ({a}): no frame with that action within 30 s")
        continue
    lat.append(seen - t_press)
    print(f"press {i} ({a}): {lat[-1]:.3f} s to the first new frame (server action={resp.get('action')})")
    time.sleep(args.gap)

done.set()
post("/stop", {}).getresponse().read()
with lock:
    fr = [f for f in frames if f[0] >= t_start + args.settle]
if len(fr) > 1:
    print(f"stream: {len(fr)} frames in {fr[-1][0] - fr[0][0]:.2f} s = {(len(fr) - 1) / (fr[-1][0] - fr[0][0]):.2f} fps")
if lat:
    print(f"key press -> new frames: median {statistics.median(lat):.3f} s, min {min(lat):.3f}, "
          f"max {max(lat):.3f} over {len(lat)} presses")
