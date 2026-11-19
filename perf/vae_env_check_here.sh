#!/usr/bin/env bash
# perf/vae_env_check_here.sh <tag> <visible cores> <port> [K=V ...]: kiro_mx4/vae_env_check.sh, but the decode runs from
# the directory this is started in (kiro_mx4/common.sh cds to /work/lingbot), so a helper lane's own code dir
# (e.g. /work/lingbot_vae) is what gets measured. Same decode, same scoring, same output line (+ the code dir).
here=$(pwd)
source kiro_mx4/common.sh
cd "$here" || exit 1
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
all=$(cat $o/dec*.log | grep -aoE "vae\.decode (steady-state|warmup) \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | tr '\n' ',')
p=$(python3 -c "
import torch, math
f = torch.load('$o/frames.pt').float(); r = torch.load('$R/frames.pt').float()
f = f.permute(1, 2, 3, 0) if f.shape[0] == 3 else f
m = (f - r).pow(2).mean().item(); print(f'{10 * math.log10(255 ** 2 / m):.2f} dB')" 2>&1 | tail -1)
echo "VAEENV $tag $* VAE_ms=${v:-NA} vs_R=$p all=$all dir=$here $(grep -m1 -aoE 'Error.{0,120}' $o/dec1.log)"
