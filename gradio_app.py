"""Gradio front end for serve.py: a session of ~10 s segments that continue one generation.

Start generates the first segment; Continue (any time, also while one is still
generating) extends the same generation with the prompt in the box. Each segment
arrives as one mp4 and plays at a steady 16 fps; the whole session is rebuilt
after every segment.

    LBW_URL=http://lbw-v2-dit-t5:8000 python gradio_app.py
"""

import base64
import glob
import json
import os
import shutil
import tempfile
import threading
import time

import gradio as gr
import imageio.v2 as imageio
import requests

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
PLAYBACK_FPS = 16
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
SCENES = {
    "02 · city alley, game-style walk": "02",
    "01 · Stonehenge": "01",
    "00 · fantasy jungle castle": "00",
    "05 · dragon flight": "05",
    "04": "04",
    "03 · lake": "03",
}
LAKE_PROMPT = (
    "gentle ripples spreading across the calm lake water, tree leaves swaying softly "
    "in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera "
    "push-in, serene natural motion")
PLAYBACK = {}   # session dir -> wall time the segment now playing finishes
LOCK = threading.Lock()


def scene_info(label):
    d = os.path.join(EXAMPLES, SCENES[label])
    p = os.path.join(d, "prompt.txt")
    prompt = open(p).read().strip() if os.path.isfile(p) else LAKE_PROMPT
    return os.path.join(d, "image.jpg"), prompt


def to_video(frames, out):
    with imageio.get_writer(out, fps=PLAYBACK_FPS, codec="libx264", macro_block_size=8,
                            output_params=["-pix_fmt", "yuv420p"]) as w:
        for p in frames:
            w.append_data(imageio.imread(p))
    return out


def segment(path, body, sdir):
    """Stream one segment from the server; yield (now playing, whole session, status)."""
    tmp = tempfile.mkdtemp(dir=sdir)
    n, t0, seg = 0, time.time(), None
    yield gr.update(), gr.update(), "generating..."
    with requests.post(f"{LBW_URL}{path}", json=body, stream=True, timeout=3600) as r:
        if r.status_code == 409:
            yield gr.update(), gr.update(), "no session - press Start first"
            return
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                yield gr.update(), gr.update(), f"error: {ev['error']}"
                return
            if ev.get("superseded"):
                yield gr.update(), gr.update(), f"skipped: {ev['superseded']}"
                return
            if ev.get("done"):
                seg = ev["segment"]
                break
            with open(os.path.join(tmp, f"{n:06d}.jpg"), "wb") as f:
                f.write(base64.b64decode(ev["frame"]))
            n += 1
            if n % 12 == 0:
                yield gr.update(), gr.update(), \
                    f"generating: {n} frames ({n / (time.time() - t0):.1f} fps)"
    if seg is None:
        return
    sd = os.path.join(sdir, f"seg_{seg:03d}")
    os.rename(tmp, sd)
    clip = to_video(sorted(glob.glob(os.path.join(sd, "*.jpg"))),
                    os.path.join(sdir, f"seg_{seg:03d}.mp4"))
    whole = to_video(sorted(glob.glob(os.path.join(sdir, "seg_*", "*.jpg"))),
                     os.path.join(sdir, f"session_{seg:03d}.mp4"))
    # Don't cut the segment now playing off: wait until it has finished.
    with LOCK:
        until = PLAYBACK.get(sdir, 0)
    if until > time.time():
        yield gr.update(), gr.update(), f"segment {seg} ready, plays after the current one"
        time.sleep(until - time.time())
    with LOCK:
        PLAYBACK[sdir] = time.time() + n / PLAYBACK_FPS
    yield clip, whole, f"segment {seg}: {n} frames generated in {time.time() - t0:.1f} s"


def start(prompt, scene, sdir):
    if sdir:
        shutil.rmtree(sdir, ignore_errors=True)
    sdir = tempfile.mkdtemp(prefix="lbw_session_")
    for clip, whole, status in segment("/session", {"prompt": prompt, "scene": SCENES[scene]}, sdir):
        yield clip, whole, status, sdir


def cont(prompt, sdir):
    if not sdir:
        yield gr.update(), gr.update(), "press Start first", sdir
        return
    for clip, whole, status in segment("/continue", {"prompt": prompt}, sdir):
        yield clip, whole, status, sdir


with gr.Blocks(title="LingBot-World v2 on AWS Trainium") as demo:
    gr.Markdown("# LingBot-World v2 on AWS Trainium\n**Start** generates ~10 s. "
                "**Continue** extends the same scene with the prompt in the box - press it any "
                "time, the next segment is generated while this one plays.")
    sdir = gr.State("")
    first_scene = next(iter(SCENES))
    first_img, first_prompt = scene_info(first_scene)
    with gr.Row():
        scene = gr.Dropdown(list(SCENES), value=first_scene, label="Scene (start image + camera path)")
        start_img = gr.Image(value=first_img, label="Start image", type="filepath", height=160,
                             interactive=False)
    prompt = gr.Textbox(label="Prompt", value=first_prompt, lines=4)
    scene.change(scene_info, [scene], [start_img, prompt])
    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        cont_btn = gr.Button("Continue")
    status = gr.Textbox(label="Status", interactive=False)
    with gr.Row():
        now = gr.Video(label="Now playing", autoplay=True, height=420)
        whole = gr.Video(label="Whole session", height=420)
    start_btn.click(start, [prompt, scene, sdir], [now, whole, status, sdir], concurrency_limit=None)
    cont_btn.click(cont, [prompt, sdir], [now, whole, status, sdir], concurrency_limit=None)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
