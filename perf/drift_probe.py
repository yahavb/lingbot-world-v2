"""Drive one long live session on serve.py and measure per-chunk frame quality.

Finds the chunk where a long live session starts to degrade. Run next to the server
(e.g. kubectl exec into the serving pod):

    python3 perf/drift_probe.py --url http://127.0.0.1:8000 --chunks 150 --out /logs/drift

Per chunk it records, over that chunk's frames:
  sharp  variance of the Laplacian of the luminance (blur -> down, noise -> up)
  sat    mean HSV saturation (washed out -> down, oversaturated -> up)
  lum    mean luminance
  hf     mean |pixel - 3x3 box blur| (high-frequency energy: speckle / noise)
  d1     mean |frame - chunk 1's frame| (how far the picture has moved from the start)
It writes drift.csv and one JPEG every --every chunks. The server's own [STATS] lines
(LINGBOT_SERVE_STATS=1) give the latent statistics for the same chunk ids.
"""
import argparse
import base64
import csv
import io
import json
import os
import threading
import time
import urllib.request

import numpy as np
from PIL import Image


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)


def metrics(img, first):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    lap = (-4 * y[1:-1, 1:-1] + y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:])
    box = sum(y[1 + i:y.shape[0] - 1 + i, 1 + j:y.shape[1] - 1 + j]
              for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9.0
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d1 = float(np.abs(a - first).mean()) if first is not None else 0.0
    return {"sharp": float(lap.var()), "sat": float(sat.mean()), "lum": float(y.mean()),
            "hf": float(np.abs(y[1:-1, 1:-1] - box).mean()), "d1": d1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--chunks", type=int, default=150)
    ap.add_argument("--scene", default="03")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=None, help="default: the scene's prompt.txt, else $PROMPT (the pod's)")
    ap.add_argument("--actions", default="forward",
                    help="comma list, one per --hold chunks, cycled (e.g. forward,left,forward,right)")
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", default="/logs/drift")
    ap.add_argument("--video", default=None, help="also write every frame to this mp4 (16 fps)")
    ap.add_argument("--no-steer", action="store_true", help="never POST /control: the scene's own camera path")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pf = f"examples/{a.scene}/prompt.txt"
    prompt = a.prompt or (open(pf).read().strip() if os.path.isfile(pf) else os.environ["PROMPT"])
    acts = a.actions.split(",")

    rows, cur, frames, first = [], None, [], None
    sent = {"chunk": -1}

    def steer(chunk):
        if a.no_steer:
            return
        act = acts[(chunk // a.hold) % len(acts)]
        if act != sent.get("act"):
            post(f"{a.url}/control", {"action": act}).read()
            sent["act"] = act

    def flush(chunk, fr):
        nonlocal first
        if not fr:
            return
        mid = fr[len(fr) // 2]
        if first is None:
            first = np.asarray(mid.convert("RGB"), dtype=np.float32) / 255.0
        ms = [metrics(f, first) for f in fr]
        row = {"chunk": chunk, "frames": len(fr), "t": round(time.time() - t0, 2)}
        for k in ms[0]:
            row[k] = round(float(np.mean([m[k] for m in ms])), 5)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if chunk % a.every == 0:
            mid.save(f"{a.out}/chunk_{chunk:04d}.jpg", quality=92)

    writer = None
    if a.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(a.video, fps=16, quality=8, macro_block_size=1)
    t0 = time.time()
    steer(0)
    r = post(f"{a.url}/session", {"prompt": prompt, "scene": a.scene, "seed": a.seed, "live": True})
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        item = json.loads(line[6:])
        if "frame" not in item:
            if item.get("done"):
                break
            continue
        c = item.get("chunk")
        if c != cur:
            flush(cur, frames)
            frames, cur = [], c
            if c is not None and c >= a.chunks:
                threading.Thread(target=lambda: post(f"{a.url}/stop", {}).read(), daemon=True).start()
                break
            if c is not None:
                steer(c + 1)
        im = Image.open(io.BytesIO(base64.b64decode(item["frame"])))
        frames.append(im)
        if writer is not None:
            writer.append_data(np.asarray(im.convert("RGB")))
    if writer is not None:
        writer.close()
        print(f"wrote {a.video}")
    with open(f"{a.out}/drift.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}/drift.csv ({len(rows)} chunks)")


if __name__ == "__main__":
    main()
