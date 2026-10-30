"""fp32 CPU reference of the chunk-0, step-0 DiT forward (the c0s0 noise prediction).

    python3 kiro/ref_fp32_c0s0.py <c0s0_inputs.pt> <ckpt_dir> <upstream_repo> <out.pt> [<c0s0.pt> ...]

<c0s0_inputs.pt> is written by wan/image2video.py with LINGBOT_DUMP_PRED=<dir>:c0s0 (x, t, context,
y, cam, seq_len, frame_seqlen, max_attention_size). <upstream_repo> is a checkout of the original
(CUDA) lingbot-world-v2: its WanModelFast is plain PyTorch and runs on CPU once the two attention
entry points are pointed at an fp32 SDPA (self-attention defaults to bf16, cross-attention calls
flash_attn). Chunk 0, step 0 runs on zeroed KV caches, so the dumped inputs define it completely.
Weights: <ckpt_dir>/transformers-bf16.pth (the converted checkpoint the Neuron runs load),
upcast to fp32. Prints the rel-RMS of every given c0s0.pt against the reference.
Peak memory ~60-70 GB (14B fp32 weights + activations).
"""
import inspect
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

inp_path, ckpt, upstream, out_path = sys.argv[1:5]
others = sys.argv[5:]
LOCAL_ATTN_SIZE, SINK_SIZE = int(os.environ.get("LOCAL_ATTN_SIZE", "12")), int(os.environ.get("SINK_SIZE", "1"))


def f32_attention(q, k, v, *args, **kwargs):  # [B, L, N, D] -> [B, L, N, D], fp32, no masks
    o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                       v.transpose(1, 2).float())
    return o.transpose(1, 2).contiguous()


if not os.path.exists(out_path):
    torch.set_num_threads(int(os.environ.get("THREADS", str(os.cpu_count()))))
    torch.set_grad_enabled(False)
    sys.path.insert(0, upstream)
    import wan.modules.attention as A  # noqa: E402  (the upstream module)
    A.attention, A.flash_attention = f32_attention, f32_attention
    import wan.modules.model_fast as MF  # noqa: E402
    MF.attention = MF.flash_attention = f32_attention
    assert os.path.realpath(MF.__file__).startswith(os.path.realpath(upstream)), MF.__file__

    cfg = json.load(open(os.path.join(ckpt, "transformers", "config.json")))
    params = inspect.signature(MF.WanModelFast.__init__).parameters
    kw = {k: v for k, v in cfg.items() if k in params}
    kw.update(local_attn_size=LOCAL_ATTN_SIZE, sink_size=SINK_SIZE)
    t0 = time.time()
    model = MF.WanModelFast(**kw).eval().float()
    sd = torch.load(os.path.join(ckpt, "transformers-bf16.pth"), map_location="cpu", mmap=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ref] loaded in {time.time() - t0:.0f}s; missing {len(missing)} {missing[:5]} "
          f"unexpected {len(unexpected)} {unexpected[:5]}", flush=True)
    assert not missing, "weights missing: the reference would use random init"
    del sd

    inp = torch.load(inp_path, map_location="cpu")
    x, t, ctx, y = inp["x"].float(), inp["t"].float(), inp["context"].float(), inp["y"].float()
    seq_len, frame_seqlen = int(inp["seq_len"]), int(inp["frame_seqlen"])
    n, d = model.num_heads, model.dim // model.num_heads
    kv = seq_len
    kv_cache = [{"k": torch.zeros(1, kv, n, d), "v": torch.zeros(1, kv, n, d),
                 "global_end_index": torch.tensor([0], dtype=torch.long),
                 "local_end_index": torch.tensor([0], dtype=torch.long)} for _ in model.blocks]
    cross_cache = [{"k": torch.zeros(1, model.text_len, n, d), "v": torch.zeros(1, model.text_len, n, d),
                    "is_init": False} for _ in model.blocks]
    t0 = time.time()
    out = model(x=[x], t=t, context=[ctx], seq_len=seq_len, y=[y],
                dit_cond_dict={"c2ws_plucker_emb": [c.float() for c in inp["cam"]]},
                kv_cache=kv_cache, crossattn_cache=cross_cache, current_start=int(inp["current_start"]),
                max_attention_size=int(inp["max_attention_size"]), frame_seqlen=frame_seqlen)
    out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"[ref] fp32 CPU forward {time.time() - t0:.0f}s, output {tuple(out.shape)}", flush=True)
    torch.save(out.float(), out_path)

ref = torch.load(out_path)
for p in others:
    if not os.path.exists(p):
        print(f"[ref] {p}: missing")
        continue
    a = torch.load(p, map_location="cpu").float().reshape(ref.shape)
    e = ((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    print(f"[ref] {p}: c0s0 rel-RMS vs fp32 CPU = {e:.5f}", flush=True)
