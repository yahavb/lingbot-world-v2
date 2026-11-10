"""fp32 CPU T5 (umt5-xxl) encoding of the prompt vs the context each run fed the DiT.

    python3 kiro/e2e/t5_ref.py <ckpt_dir> <upstream_repo> <out.pt> <rollout_inputs.pt> ...

Uses the upstream (CUDA) T5EncoderModel on the CPU in fp32 (weights from the bf16 checkpoint,
upcast). Prints rel-RMS and max |d| of every run's context against it.
"""
import os
import sys

import torch

ckpt, upstream, out = sys.argv[1:4]
runs = sys.argv[4:]
PROMPT = ("gentle ripples spreading across the calm lake water, tree leaves swaying softly in the "
          "breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, "
          "serene natural motion")
torch.set_grad_enabled(False)
if not os.path.exists(out):
    sys.path.insert(0, upstream)
    torch.cuda.current_device = lambda: 0
    import types  # stub packages: wan/__init__ pulls in torchvision and the DiT stack
    for name, sub in (("wan", "wan"), ("wan.modules", "wan/modules")):
        m = types.ModuleType(name); m.__path__ = [os.path.join(upstream, sub)]; sys.modules[name] = m
    from wan.modules.t5 import T5EncoderModel  # noqa: E402
    enc = T5EncoderModel(text_len=512, dtype=torch.float32, device=torch.device("cpu"),
                         checkpoint_path=os.path.join(ckpt, "models_t5_umt5-xxl-enc-bf16.pth"),
                         tokenizer_path=os.path.join(ckpt, "google/umt5-xxl"))
    enc.model.float()
    ctx = enc([PROMPT], torch.device("cpu"))[0].float()
    torch.save(ctx, out)
ref = torch.load(out)
print(f"[t5] fp32 CPU context {tuple(ref.shape)}")
for p in runs:
    c = torch.load(p, map_location="cpu")["context"].float()
    d = c - ref
    print(f"[t5] {p}: rel-RMS {float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()):.4f} "
          f"max|d| {float(d.abs().max()):.3f}")
