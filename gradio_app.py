"""Gradio front end for serve.py: streams frames into one live view.

    LBW_URL=http://lbw-v2-dit-t5:8000 python gradio_app.py
"""

import base64
import io
import json
import os
import tempfile
import time

import gradio as gr
import imageio
import numpy as np
import requests
from PIL import Image

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
PLAYBACK_FPS = 16
DEFAULT_PROMPT = (
    "gentle ripples spreading across the calm lake water, tree leaves swaying softly "
    "in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera "
    "push-in, serene natural motion")


def stream(prompt, cont):
    yield gr.update(), "waiting for the model...", None
    frames, t0, next_show = [], time.time(), 0.0
    with requests.post(f"{LBW_URL}/generate/stream", stream=True, timeout=3600,
                       json={"prompt": prompt, "continue": cont}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                yield gr.update(), f"error: {ev['error']}", None
                return
            if ev.get("superseded"):
                yield gr.update(), "replaced by a newer prompt", None
                return
            if ev.get("done"):
                break
            frame = Image.open(io.BytesIO(base64.b64decode(ev["frame"]))).convert("RGB")
            frames.append(np.asarray(frame))
            wait = next_show - time.time()
            if wait > 0:
                time.sleep(wait)
            next_show = time.time() + 1.0 / PLAYBACK_FPS
            yield frame, f"{len(frames)} frames, {len(frames) / (time.time() - t0):.1f} fps", None
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    imageio.mimwrite(path, frames, fps=PLAYBACK_FPS, codec="libx264", macro_block_size=8)
    yield gr.update(), f"done: {len(frames)} frames in {time.time() - t0:.1f} s", path


def new_scene(prompt):
    yield from stream(prompt, False)


def continue_scene(prompt):
    yield from stream(prompt, True)


with gr.Blocks(title="LingBot-World v2 on AWS Trainium") as demo:
    gr.Markdown("# LingBot-World v2 on AWS Trainium\nFrames stream as they are generated. "
                "**Continue** extends the current scene with a new prompt.")
    prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
    with gr.Row():
        new_btn = gr.Button("Generate", variant="primary")
        cont_btn = gr.Button("Continue")
    live = gr.Image(label="Live", type="pil", height=480)
    status = gr.Textbox(label="Status", interactive=False)
    video = gr.Video(label="Last segment")
    outputs = [live, status, video]
    new_btn.click(new_scene, [prompt], outputs, concurrency_limit=None)
    cont_btn.click(continue_scene, [prompt], outputs, concurrency_limit=None)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
