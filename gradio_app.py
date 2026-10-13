"""Gradio front end for serve.py's live mode: one continuous stream, prompts change on the fly.

Frames are kept as the server's JPEGs on disk: the live view shows the newest, the
gallery accumulates them, and the whole stream becomes a playable mp4 when it ends.

    LBW_URL=http://lbw-v2-dit-t5:8000 python gradio_app.py
"""

import base64
import json
import os
import tempfile
import time

import gradio as gr
import imageio.v2 as imageio
import requests

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
PLAYBACK_FPS = 16
GALLERY_MAX = int(os.environ.get("GALLERY_MAX", "600"))
DEFAULT_PROMPT = (
    "gentle ripples spreading across the calm lake water, tree leaves swaying softly "
    "in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera "
    "push-in, serene natural motion")


def to_video(paths):
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    with imageio.get_writer(out, fps=PLAYBACK_FPS, codec="libx264", macro_block_size=8,
                            output_params=["-pix_fmt", "yuv420p"]) as w:
        for p in paths:
            w.append_data(imageio.imread(p))
    return out


def start(prompt):
    outdir = tempfile.mkdtemp(prefix="lbw_")
    paths, t0, next_show = [], time.time(), 0.0
    yield None, [], None, "waiting for the model..."
    with requests.post(f"{LBW_URL}/live", stream=True, timeout=3600,
                       json={"prompt": prompt}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                yield gr.update(), gr.update(), gr.update(), f"error: {ev['error']}"
                break
            if ev.get("superseded"):
                yield gr.update(), gr.update(), gr.update(), "replaced by a newer stream"
                break
            if ev.get("done"):
                break
            path = os.path.join(outdir, f"{len(paths):06d}.jpg")
            with open(path, "wb") as f:
                f.write(base64.b64decode(ev["frame"]))
            paths.append(path)
            wait = next_show - time.time()
            if wait > 0:
                time.sleep(wait)
            next_show = time.time() + 1.0 / PLAYBACK_FPS
            n = len(paths)
            status = f"live: {n} frames, {n / (time.time() - t0):.1f} fps"
            # The gallery re-sends its whole list, so refresh it once per chunk, not per frame.
            gallery = paths[-GALLERY_MAX:] if n % 12 == 0 else gr.update()
            yield path, gallery, gr.update(), status
    if paths:
        yield paths[-1], paths[-GALLERY_MAX:], to_video(paths), \
            f"stopped after {len(paths)} frames ({time.time() - t0:.0f} s) - press play on Whole stream"


def change_prompt(prompt):
    r = requests.post(f"{LBW_URL}/live/prompt", json={"prompt": prompt}, timeout=30)
    if r.status_code == 409:
        return "no stream running - press Start"
    r.raise_for_status()
    return f"next chunk uses: {prompt[:80]}"


def stop():
    requests.post(f"{LBW_URL}/live/stop", timeout=30).raise_for_status()
    return "stopping at the next chunk"


with gr.Blocks(title="LingBot-World v2 on AWS Trainium") as demo:
    gr.Markdown("# LingBot-World v2 on AWS Trainium\nOne continuous stream. "
                "**Change prompt** steers the scene from the next chunk on.")
    prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        change_btn = gr.Button("Change prompt")
        stop_btn = gr.Button("Stop", variant="stop")
    with gr.Row():
        live = gr.Image(label="Live", type="filepath", height=480)
        video = gr.Video(label="Whole stream (ready when it stops)", height=480)
    status = gr.Textbox(label="Stream", interactive=False)
    control = gr.Textbox(label="Control", interactive=False)
    gallery = gr.Gallery(label="Frames so far", columns=8, height=360, object_fit="contain")
    start_btn.click(start, [prompt], [live, gallery, video, status], concurrency_limit=None)
    change_btn.click(change_prompt, [prompt], [control], concurrency_limit=None)
    stop_btn.click(stop, [], [control], concurrency_limit=None)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
