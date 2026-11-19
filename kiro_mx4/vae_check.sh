#!/usr/bin/env bash
# vae_check.sh <tag> <nproc> <lnc> [K=V ...]: decode R's latents (CPU fp32 rollout) on this pod and score the frames
# against R's fp32 CPU decode (PSNR); twice, the faster steady-state time reported.
set -uo pipefail
source kiro_mx4/common.sh
tag=$1; np=$2; lnc=$3; shift 3
for kv in "$@"; do export "$kv"; done
R=/node-cache/e2e/gate/r480_ex03_s42; o="/work/out/vaecheck/$tag"; mkdir -p "$o"
for i in 1 2; do
  LINGBOT_SAVE_FRAMES="$o/frames.pt" NEURON_LOGICAL_NC_CONFIG=$lnc torchrun --nproc_per_node=$np --master_port=29743 \
    decode_latents_width_sharded.py --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$R/latents.pt" --save_file "$o/out.mp4" \
    --timing --warmup 2>&1 | grep -avE "AliasingInfo|Mutated inputs" > "$o/dec$i.log"
done
v=$(cat "$o"/dec*.log | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
p=$(python3 -c "
import torch, math
f = torch.load('$o/frames.pt').float(); r = torch.load('$R/frames.pt').float()
f = f.permute(1, 2, 3, 0) if f.shape[0] == 3 else f
m = (f - r).pow(2).mean().item(); print(f'{10 * math.log10(255 ** 2 / m):.2f} dB, mean {f.mean().item():.1f}' if m > 0 else 'inf')" 2>&1 | tail -1)
echo "VAECHECK $tag np=$np lnc=$lnc $* VAE_ms=${v:-NA} vs_R=$p" | tee -a /work/out/e2e/results.txt
