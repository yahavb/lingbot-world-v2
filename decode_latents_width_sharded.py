"""Stage 2 of the width-sharded VAE decode split.

Loads a pre-decode latent dumped by `generate.py --save_latents_path` (stage
1, single-process T5+DiT) and decodes it with the width-sharded Wan2_1_VAE
decoder (see wan/modules/vae2_1_sharded.py). Every rank holds a full unsharded
copy of the decoder; the latent is split along width, each rank decodes its
own slice, results are all-gathered back into the full video.

Always launched via torchrun, even for a single-process world=1 sanity
check (reproducing the original unsharded CPU-decode path exactly):

    torchrun --nproc_per_node=1 decode_latents_width_sharded.py \\
        --latents_path latents.pt \\
        --ckpt_dir /shared/huggingface/lingbot-world-v2-14b-causal-fast \\
        --save_file output/wanaka_test.mp4 --device cpu

    torchrun --nproc_per_node=4 decode_latents_width_sharded.py \\
        --latents_path latents.pt \\
        --ckpt_dir /shared/huggingface/lingbot-world-v2-14b-causal-fast \\
        --save_file output/wanaka_test.mp4
"""

import argparse
import logging
import math
import os
import sys
import time

# --profile's env vars must be set before `import torch` (see
# scripts/profiling/profile_utils.setup_env), so we peek sys.argv here
# instead of waiting for argparse. Each rank writes to its own
# `rank{RANK}` subdirectory, always under this fixed, timestamped path.
# Cores visibility is left unrestricted, because restricting it per-rank 
# would collide with torch.neuron.set_device() below.
# `capture_core` in main() scopes capture to this rank's core instead.
_PROFILE_BASE_DIR = os.path.join(
    "scripts/profiling/results/vae", time.strftime("%Y%m%d_%H%M%S"))

if "--profile" in sys.argv:
    _profile_output_dir = os.path.join(
        _PROFILE_BASE_DIR, f"rank{os.environ.get('RANK', '0')}")
    from scripts.profiling.profile_utils import setup_env, profile_call
    setup_env(_profile_output_dir)

import torch
import torch.distributed as dist

from wan.configs import WAN_CONFIGS
from wan.modules.vae2_1 import Wan2_1_VAE
from wan.modules.vae2_1_sharded import (
    gather_video_width,
    shard_vae_decoder_width,
    width_shard_latent,
)
from wan.utils.utils import save_video


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Decode a saved latent with the width-sharded VAE decoder.")
    parser.add_argument("--latents_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="i2v-A14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="neuron",
                        choices=["neuron", "cpu"],
                        help="'cpu' is for a world=1 correctness sanity check "
                             "against the original CPU-decode baseline.")
    parser.add_argument("--warmup", action="store_true",
                        help="Only meaningful with --timing. Times an extra "
                             "first ('warmup') call to compare against "
                             "steady-state speed -- useful for estimating a "
                             "persistent-service setup.")
    measurement_mode = parser.add_mutually_exclusive_group()
    measurement_mode.add_argument("--timing", action="store_true",
                        help="Time the decode call. Off by default so a normal "
                             "run doesn't pay the synchronization/timing overhead.")
    measurement_mode.add_argument("--profile", action="store_true",
                        help="Profile the decode with torch.profiler + "
                             "NeuronProfiler (see scripts/profiling/"
                             "profile_utils.py). Each rank writes its own "
                             "scripts/profiling/results/vae/<launch "
                             "timestamp>/rank<N>/.")
    return parser.parse_args()


def main():
    args = _parse_args()

    assert "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ, \
        "launch via torchrun"
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format=f"[rank {rank}] %(asctime)s %(levelname)s: %(message)s")

    if world_size > 1:
        torch.neuron.set_device(local_rank)
        dist.init_process_group(
            backend="neuron",
            init_method="env://",
            rank=rank,
            world_size=world_size)

    cfg = WAN_CONFIGS[args.task]
    device = (torch.device(f"neuron:{local_rank}") if args.device == "neuron"
              else torch.device("cpu"))

    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        device=device, dtype=torch.bfloat16)

    # LINGBOT_VAE_FUSED=1: whole decoder blocks compiled with the width halos inside the graphs
    # (wan/modules/vae2_1_fused.py) instead of per-conv graphs and eager halo all-gathers.
    _fused = world_size > 1 and os.environ.get("LINGBOT_VAE_FUSED", "0") == "1"
    if _fused:
        from wan.modules.vae2_1_fused import FusedWidthDecoder, apply_vae_cc_flags
        _ccx = apply_vae_cc_flags()
        if _ccx:
            logging.info(f"VAE compiler flags: + {_ccx}")
        _fdec = FusedWidthDecoder(vae.model)

        def _decode(z):
            return _fdec.decode(z.unsqueeze(0), vae.scale).float().clamp_(-1, 1).squeeze(0)
        logging.info(f"Width-sharded FUSED decoder across {world_size} ranks.")
    else:
        if world_size > 1:
            shard_vae_decoder_width(vae.model.decoder)
            logging.info(f"Width-sharded decoder across {world_size} ranks.")

        def _decode(z):
            return vae.decode([z])[0]

    def _device_sync():
        # Launches are asynchronous; time the device work, not the host issue.
        if device.type == "neuron":
            torch.neuron.synchronize()

    latent = torch.load(args.latents_path, map_location="cpu").to(device)

    # SAME width padding as the inline path in generate.py. This script sharded on
    # world_size directly, which asserts the moment lat_w does not divide it: at
    # 768*1152 lat_w is 144 and 144 % 32 = 16. Pad up to the next multiple so every rank
    # participates, then crop after the gather.
    #
    # NOT numerically neutral -- the decoder's right-hand columns see replicated latent
    # content where they previously saw the tensor edge. Replicate keeps that minimal.
    _lat_w = latent.shape[-1]
    _pad_w = ((_lat_w + world_size - 1) // world_size) * world_size
    if _pad_w != _lat_w and world_size > math.gcd(_lat_w, world_size):
        latent = torch.cat(
            [latent, latent[..., -1:].expand(*latent.shape[:-1], _pad_w - _lat_w)],
            dim=-1)
        if rank == 0:
            logging.info(
                f"latent width {_lat_w} padded to {_pad_w} (replicate) so all "
                f"{world_size} ranks participate, {_pad_w // world_size} px each")
    else:
        _pad_w = _lat_w
    _two_d = _fused and os.environ.get("LINGBOT_VAE_2D", "0") == "1"
    if _two_d:
        # LINGBOT_VAE_2D: height x width tiles of the UNPADDED latent (see vae2_1_fused).
        from wan.modules.vae2_1_fused import shard_latent_2d
        latent = latent[..., :_lat_w]
        _pad_w = _lat_w
        latent_local = shard_latent_2d(latent)
    else:
        latent_local = width_shard_latent(latent, rank=rank, world=world_size)

    if args.profile:
        def _decode_once():
            return _decode(latent_local)

        # Every rank runs the same call sequence, so the width-sharded
        # collective ops (halo exchange / all-gather) stay in lockstep.
        video_local = profile_call(
            _decode_once, _profile_output_dir,
            label=f"[rank {rank}] VAE DECODE steady-state (width={world_size})",
            warmup=3, capture_core=str(local_rank))

        if dist.is_initialized():
            dist.barrier()
    elif args.timing:
        if args.warmup:
            if dist.is_initialized():
                dist.barrier()
            t0 = time.perf_counter()
            _decode(latent_local)
            _device_sync()
            if dist.is_initialized():
                dist.barrier()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"[TIMING] vae.decode warmup (width={world_size}): {dt_ms:.0f} ms")

        if dist.is_initialized():
            dist.barrier()
        t0 = time.perf_counter()
        video_local = _decode(latent_local)
        _device_sync()
        if dist.is_initialized():
            dist.barrier()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        label = "steady-state" if args.warmup else "single-call"
        logging.info(f"[TIMING] vae.decode {label} (width={world_size}): {dt_ms:.0f} ms")
    else:
        video_local = _decode(latent_local)

    if _two_d:
        from wan.modules.vae2_1_fused import gather_video_2d
        video_full = gather_video_2d(video_local)
    else:
        video_full = gather_video_width(video_local)

    if rank == 0 and _pad_w != _lat_w:

        video_full = video_full[..., :_lat_w * cfg.vae_stride[2]].contiguous()

    if rank == 0:
        os.makedirs(os.path.dirname(args.save_file) or ".", exist_ok=True)
        logging.info(f"Saving decoded video to {args.save_file}")
        save_video(
            tensor=video_full[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
