"""Press-to-photon latency of a live session (serve.py): every --period seconds send a camera action (POST /control),
alternating --actions, and time how long until the first frame generated under that action arrives at this client.

    python3 latency_probe.py --url http://127.0.0.1:8000 --presses 100 --out /logs/lat

Per press: send time, arrival of the first frame whose "action" is the new one, and the server-side stamps carried
by that frame (t.ctl = /control arrived, t.start = its chunk's DiT started, t.x0 = latents done, t.dec = decoded),
so the latency splits into wait (ctl -> start), DiT (start -> x0), decode + hand-off (x0 -> dec) and delivery
(dec -> arrival). Writes latency.csv and prints the median and p90.
"""
import argparse
import csv
import json
import os
import statistics as st
import threading
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--scene", default="03")
    ap.add_argument("--presses", type=int, default=100)
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument("--actions", default="left,right")
    ap.add_argument("--warm", type=float, default=15.0, help="seconds of stream before the first press")
    ap.add_argument("--out", default="/logs/lat")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acts = a.actions.split(",")
    pending, rows, lock = {}, [], threading.Lock()   # pending: action -> send time (one in flight at a time)
    done = threading.Event()

    def presser():
        time.sleep(a.warm)
        for i in range(a.presses):
            act = acts[i % len(acts)]
            with lock:
                pending.clear()
                pending[act] = time.time()
            post(f"{a.url}/control", {"action": act}).read()
            time.sleep(a.period)
        done.set()

    pf = f"examples/{a.scene}/prompt.txt"
    prompt = open(pf).read().strip() if os.path.isfile(pf) else os.environ.get("PROMPT", "")
    r = post(f"{a.url}/session", {"prompt": prompt, "scene": a.scene, "seed": 42, "live": True})
    threading.Thread(target=presser, daemon=True).start()
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        item = json.loads(line[6:])
        if "frame" not in item:
            continue
        now = time.time()
        act = item.get("action")
        with lock:
            t_send = pending.pop(act, None) if act in pending else None
        if t_send is not None:
            t = item.get("t") or {}
            row = {"action": act, "latency_ms": round(1000 * (now - t_send), 1)}
            for k0, k1, name in (("ctl", "start", "wait_ms"), ("start", "x0", "dit_ms"), ("x0", "dec", "decode_ms")):
                if t.get(k0) and t.get(k1):
                    row[name] = round(1000 * (t[k1] - t[k0]), 1)
            if t.get("dec"):
                row["delivery_ms"] = round(1000 * (now - t["dec"]), 1)
            rows.append(row)
            print(json.dumps(row), flush=True)
        if done.is_set():
            threading.Thread(target=lambda: post(f"{a.url}/stop", {}).read(), daemon=True).start()
            break
    with open(f"{a.out}/latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r_ in rows for k in r_}))
        w.writeheader()
        w.writerows(rows)
    lat = sorted(r_["latency_ms"] for r_ in rows)
    if lat:
        p90 = lat[int(0.9 * (len(lat) - 1))]
        parts = {k: round(st.median(r_[k] for r_ in rows if k in r_), 1) for k in ("wait_ms", "dit_ms", "decode_ms", "delivery_ms")
                 if any(k in r_ for r_ in rows)}
        print(f"LATENCY presses {len(lat)} median {st.median(lat):.0f} ms p90 {p90:.0f} ms parts {parts}")


if __name__ == "__main__":
    main()
