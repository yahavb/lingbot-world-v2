"""Gradio front end for serve.py's live mode: one continuous stream, prompts change on the fly.

    LBW_URL=http://lbw-v2-dit-t5:8000 python gradio_app.py
"""

import base64
import io
import json
import os
import time

import gradio as gr
import requests
from PIL import Image

LBW_URL = os.environ.get("LBW_URL", "http://lbw-v2-dit-t5.default.svc.cluster.local:8000")
PLAYBACK_FPS = 16
DEFAULT_PROMPT = (
    "gentle ripples spreading across the calm lake water, tree leaves swaying softly "
    "in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera "
    "push-in, serene natural motion")


def start(prompt):
    yield gr.update(), "waiting for the model..."
    n, t0, next_show = 0, time.time(), 0.0
    with requests.post(f"{LBW_URL}/live", stream=True, timeout=3600,
                       json={"prompt": prompt}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                yield gr.update(), f"error: {ev['error']}"
                return
            if ev.get("superseded"):
                yield gr.update(), "replaced by a newer stream"
                return
            if ev.get("done"):
                break
            frame = Image.open(io.BytesIO(base64.b64decode(ev["frame"]))).convert("RGB")
            n += 1
            wait = next_show - time.time()
            if wait > 0:
                time.sleep(wait)
            next_show = time.time() + 1.0 / PLAYBACK_FPS
            yield frame, f"live: {n} frames, {n / (time.time() - t0):.1f} fps"
    yield gr.update(), f"stopped after {n} frames ({time.time() - t0:.0f} s)"


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
    live = gr.Image(label="Live", type="pil", height=480)
    status = gr.Textbox(label="Stream", interactive=False)
    control = gr.Textbox(label="Control", interactive=False)
    start_btn.click(start, [prompt], [live, status], concurrency_limit=None)
    change_btn.click(change_prompt, [prompt], [control], concurrency_limit=None)
    stop_btn.click(stop, [], [control], concurrency_limit=None)

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
                    root_path=os.environ.get("ROOT_PATH", ""))
