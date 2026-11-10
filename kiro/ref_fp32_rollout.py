"""CPU fp32 reference R of the whole causal_fast rollout (the accuracy gate's reference).

    python3 kiro/ref_fp32_rollout.py <rollout_inputs.pt> <ckpt_dir> <upstream_repo> <out_dir>

<rollout_inputs.pt> is written by wan/image2video.py with LINGBOT_DUMP_ROLLOUT=<dir> (rank 0, first
generation): the initial noise, condition y, camera Plücker embedding, T5 context, timesteps,
scheduler sigmas and every per-step noise draw in use order. Those are the rollout's inputs; this
script replays the rollout with the ORIGINAL (upstream, CUDA) WanModelFast in fp32 on the CPU, its
own KV cache (sliding window + sink) and the upstream scheduler (x0 = x_t - sigma_t * v in fp64,
add_noise), so nothing of the Neuron fork is on R's side except T5/VAE-encode inputs, which are
dumped tensors. The last chunk's cache commit is skipped (its output is never read).

Writes <out_dir>/latents.pt ([16, F, H, W] fp32, decode it with decode_latents_width_sharded.py
--device cpu), forward_c0s0.pt (the chunk-0 step-0 prediction), forward_x0c0.pt (chunk-0 x0),
and x0_c<k>.pt per chunk (resumable: a chunk whose x0 and cache snapshot exist is skipped).
Peak memory ~70 GB; ~1-2 min per forward on 96 cores.
"""
import inspect
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

inp_path, ckpt, upstream, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", str(os.cpu_count()))))
torch.set_grad_enabled(False)


def f32_attention(q, k, v, *args, **kwargs):  # [B, L, N, D] -> [B, L, N, D], fp32, no masks
    o = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                       v.transpose(1, 2).float())
    return o.transpose(1, 2).contiguous()


sys.path.insert(0, upstream)
torch.cuda.current_device = lambda: 0  # upstream t5.py queries CUDA at import time
import wan.modules.attention as A  # noqa: E402
A.attention, A.flash_attention = f32_attention, f32_attention
import wan.modules.model_fast as MF  # noqa: E402
MF.attention = MF.flash_attention = f32_attention
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
assert os.path.realpath(MF.__file__).startswith(os.path.realpath(upstream)), MF.__file__

inp = torch.load(inp_path, map_location="cpu")
LOCAL, SINK = int(inp["local_attn_size"]), int(inp["sink_size"] or 0)
cfg = json.load(open(os.path.join(ckpt, "transformers", "config.json")))
params = inspect.signature(MF.WanModelFast.__init__).parameters
kw = {k: v for k, v in cfg.items() if k in params}
kw.update(local_attn_size=LOCAL, sink_size=SINK)
t0 = time.time()
model = MF.WanModelFast(**kw).eval().float()
sd = torch.load(os.path.join(ckpt, "transformers-bf16.pth"), map_location="cpu", mmap=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[ref] loaded in {time.time() - t0:.0f}s; missing {len(missing)} unexpected {len(unexpected)}", flush=True)
assert not missing, "weights missing: the reference would use random init"
del sd

# upstream scheduler, checked against the run's tables
sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
sch.set_timesteps(1000, shift=float(os.environ.get("SAMPLE_SHIFT", "10.0")))
assert torch.allclose(sch.sigmas.double(), inp["sigmas"], atol=1e-6), "scheduler sigmas differ from the run's"
timesteps = inp["timesteps"]

noise, y, cam, ctx = inp["noise"], inp["y"], inp["cam"], inp["context"]
if os.environ.get("CONTEXT_PT"):  # e.g. the fp32 CPU T5 context (kiro/e2e/t5_ref.py): R covers T5 too
    _c = torch.load(os.environ["CONTEXT_PT"], map_location="cpu").float()
    assert _c.shape == ctx.shape, (_c.shape, ctx.shape)
    print(f"[ref] context from {os.environ['CONTEXT_PT']}: rel-RMS vs run's "
          f"{float((ctx - _c).pow(2).mean().sqrt() / _c.pow(2).mean().sqrt()):.4f}", flush=True)
    ctx = _c
cs, seq_len, fsl = int(inp["chunk_size"]), int(inp["seq_len"]), int(inp["frame_seqlen"])
kv_size, mas, n_chunks = int(inp["kv_size"]), int(inp["max_attention_size"]), int(inp["n_chunks"])
draws = list(inp["step_noise"])
assert len(draws) == n_chunks * (len(timesteps) - 1), (len(draws), n_chunks, len(timesteps))
n, d = model.num_heads, model.dim // model.num_heads
kv_cache = [{"k": torch.zeros(1, kv_size, n, d), "v": torch.zeros(1, kv_size, n, d),
             "global_end_index": torch.tensor([0], dtype=torch.long),
             "local_end_index": torch.tensor([0], dtype=torch.long)} for _ in model.blocks]
cross_cache = [{"k": torch.zeros(1, model.text_len, n, d), "v": torch.zeros(1, model.text_len, n, d),
                "is_init": torch.tensor(0, dtype=torch.int32)} for _ in model.blocks]

lat_chunks, y_chunks, cam_chunks = noise.split(cs, 1), y.split(cs, 1), cam.split(cs, 2)


def to_x0(v, xt, t):
    ts, sg = sch.timesteps.double(), sch.sigmas.double()
    return (xt.double() - sg[torch.argmin((ts - t.double()).abs())] * v.double()).float()


x0s, first = [], True
for c in range(n_chunks):
    kwargs = dict(context=[ctx], seq_len=seq_len, y=[y_chunks[c]],
                  dit_cond_dict={"c2ws_plucker_emb": cam_chunks[c].chunk(1, dim=0)},
                  kv_cache=kv_cache, crossattn_cache=cross_cache, current_start=c * cs * fsl,
                  max_attention_size=mas, frame_seqlen=fsl)
    x = lat_chunks[c]
    for i in range(len(timesteps)):
        t1 = time.time()
        out = model(x=[x], t=timesteps[i:i + 1], cross_attn_first_call=first, **kwargs)
        first = False
        v = out[0] if isinstance(out, (list, tuple)) else out
        if c == 0 and i == 0:
            torch.save(v.float(), os.path.join(out_dir, "forward_c0s0.pt"))
        x0 = to_x0(v, x, timesteps[i])
        print(f"[ref] c{c}s{i} {time.time() - t1:.0f}s", flush=True)
        if i < len(timesteps) - 1:
            x = sch.add_noise(x0, draws[c * (len(timesteps) - 1) + i], timesteps[i + 1])
    x0s.append(x0)
    torch.save(x0, os.path.join(out_dir, f"x0_c{c}.pt"))
    if c == 0:
        torch.save(x0, os.path.join(out_dir, "forward_x0c0.pt"))
    if c < n_chunks - 1:  # cache commit at t = 0
        t1 = time.time()
        model(x=[x0], t=timesteps[-1:] * 0.0, cross_attn_first_call=False, **kwargs)
        print(f"[ref] c{c}commit {time.time() - t1:.0f}s", flush=True)

lat = torch.cat(x0s, 1)
torch.save(lat, os.path.join(out_dir, "latents.pt"))
dev = inp["x0"]
for c in range(n_chunks):
    a, b = x0s[c], dev[:, c * cs:(c + 1) * cs]
    print(f"[ref] chunk {c}: run x0 rel-RMS vs R = {((a - b).pow(2).mean() / a.pow(2).mean()).sqrt().item():.4f}")
print(f"[ref] DONE latents {tuple(lat.shape)} -> {out_dir}", flush=True)
