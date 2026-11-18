#!/usr/bin/env bash
# vae_env_check.sh <tag> <visible cores> <port> [K=V ...]: kiro_mx4/vae_check.sh at LNC1 16 ranks on the given cores
# (two decodes side by side on a 4-device pod), its own compile cache dir per tag.
source kiro_mx4/common.sh
tag=$1; vis=$2; port=$3; shift 3
for kv in "$@"; do export "$kv"; done
export NEURON_RT_VISIBLE_CORES=$vis NEURON_LOGICAL_NC_CONFIG=1 LINGBOT_VAE_T=${LINGBOT_VAE_T:-1}
R=/node-cache/e2e/gate/r480_ex03_s42; o=/work/out/vaecheck/$tag; rm -rf $o; mkdir -p $o
for i in 1 2; do
  LINGBOT_SAVE_FRAMES=$o/frames.pt torchrun --nproc_per_node=16 --master_port=$port decode_latents_width_sharded.py \
    --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$R/latents.pt" --save_file "$o/out.mp4" --timing --warmup 2>&1 \
    | grep -avE "AliasingInfo|Mutated inputs" > $o/dec$i.log
done
v=$(cat $o/dec*.log | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
p=$(python3 -c "
import torch, math
f = torch.load('$o/frames.pt').float(); r = torch.load('$R/frames.pt').float()
f = f.permute(1, 2, 3, 0) if f.shape[0] == 3 else f
m = (f - r).pow(2).mean().item(); print(f'{10 * math.log10(255 ** 2 / m):.2f} dB')" 2>&1 | tail -1)
echo "VAEENV $tag $* VAE_ms=${v:-NA} vs_R=$p $(grep -m1 -aoE 'Error.{0,120}' $o/dec1.log)"
