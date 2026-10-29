"""Decode-only bench of the width-sharded VAE, set up exactly like decode_latents_width_sharded.py.
    torchrun --nproc_per_node=64 kiro/vae_bench.py <latents.pt>
Env:
  REPS=2            timed decodes after one warm-up
  LINGBOT_VAE_FUSED=1  block-compiled decoder with in-graph halos (wan/modules/vae2_1_fused.py)
  VAE_CHECK_DIR=dir    each rank writes the sha1 of its decoded slice to dir/<slice>.txt
  VAE_SAVE_DIR=dir     each rank saves its decoded slice to dir/<slice>.pt (fp32)
  VAE_DIAG=halo     DIAGNOSTIC ONLY (wrong pixels): skip the width-halo all-gathers
  VAE_DIAG=halo,attn  ... and replace the attention block's full-width gather by a local repeat
The difference between a normal and a diagnostic run is what those collectives cost.
"""
import math
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules import vae2_1_sharded as vs  # noqa: E402
from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402
CKPT = "/work/ckpt/lingbot-world-v2-14b-causal-fast"
def main(latents_path):
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.neuron.set_device(local)
    dist.init_process_group(backend="neuron", init_method="env://", rank=rank, world_size=world)
    cfg = WAN_CONFIGS["i2v-A14B"]
    dev = torch.device(f"neuron:{local}")
    diag = set(filter(None, os.environ.get("VAE_DIAG", "").split(",")))
    fused = os.environ.get("LINGBOT_VAE_FUSED", "0") == "1"
    if "halo" in diag:
        vs._halo_exchange_w = lambda x, radius: (None, None)
    if "attn" in diag:
        vs._all_gather_w = lambda x: x.repeat(*([1] * (x.dim() - 1)), vs._world())
    vae = Wan2_1_VAE(vae_pth=os.path.join(CKPT, cfg.vae_checkpoint), device=dev,
                     dtype=torch.bfloat16)
    if fused:
        from wan.modules.vae2_1_fused import FusedWidthDecoder, apply_vae_cc_flags
        apply_vae_cc_flags()
        fdec = FusedWidthDecoder(vae.model)
        decode = lambda z: fdec.decode(z.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0)
    else:
        vs.shard_vae_decoder_width(vae.model.decoder)
        decode = lambda z: vae.decode([z])[0]
    latent = torch.load(latents_path, map_location="cpu").to(dev)
    two_d = fused and os.environ.get("LINGBOT_VAE_2D", "0") == "1"
    if two_d:
        from wan.modules.vae2_1_fused import shard_latent_2d
        local_latent = shard_latent_2d(latent)
    else:
        lat_w = latent.shape[-1]
        pad_w = ((lat_w + world - 1) // world) * world
        if pad_w != lat_w and world > math.gcd(lat_w, world):
            latent = torch.cat([latent, latent[..., -1:].expand(*latent.shape[:-1], pad_w - lat_w)],
                               dim=-1)
        local_latent = vs.width_shard_latent(latent, world=world)
    times = []
    for i in range(1 + int(os.environ.get("REPS", "2"))):
        if i == 1:
            vs.reset_vae_collective_stats()
        dist.barrier()
        t0 = time.perf_counter()
        out = decode(local_latent)
        issue_ms = (time.perf_counter() - t0) * 1000   # host issue time (launches are async)
        torch.neuron.synchronize()
        dist.barrier()
        times.append((time.perf_counter() - t0) * 1000)
        if rank == 0 and i > 0:
            print(f"[vae_bench] rep {i}: host issue {issue_ms:.0f} ms of {times[-1]:.0f} ms",
                  flush=True)
    st = vs.vae_collective_stats()
    n = len(times) - 1
    digest = ""
    check_dir = os.environ.get("VAE_CHECK_DIR")
    if check_dir and not diag:
        # Each rank hashes the width slice it decoded, keyed by SLICE index, so two halo modes
        # that hand slices to ranks in different orders can be compared slice by slice.
        import hashlib
        os.makedirs(check_dir, exist_ok=True)
        h = hashlib.sha1(out.float().cpu().numpy().tobytes()).hexdigest()
        with open(os.path.join(check_dir, f"{vs._rank():03d}.txt"), "w") as f:
            f.write(h + "\n")
        digest = f" slice hashes -> {check_dir}"
    save_dir = os.environ.get("VAE_SAVE_DIR")
    if save_dir and not diag:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(out.float().cpu(), os.path.join(save_dir, f"{vs._rank():03d}.pt"))
        if two_d and rank == 0:
            with open(os.path.join(save_dir, "LAYOUT"), "w") as f:
                f.write("2d\n")
    if rank == 0:
        print(f"[vae_bench] world={world} fused={int(fused)} diag={sorted(diag) or 'none'} "
              f"latent={tuple(latent.shape)} "
              f"local={tuple(local_latent.shape)} out_local={tuple(out.shape)} "
              f"warmup {times[0]:.0f} ms, steady " + ", ".join(f"{t:.0f}" for t in times[1:])
              + f" ms; per decode: halo {st['halo'] // max(n, 1)} calls ({st['halo_s'] / n:.2f} s host), "
              f"gather {st['gather'] // max(n, 1)} ({st['gather_s'] / n:.2f} s host), "
              f"halo mode {os.environ.get('LINGBOT_VAE_HALO', 'allgather')}{digest}", flush=True)
    dist.destroy_process_group()
if __name__ == "__main__":
    main(sys.argv[1])
