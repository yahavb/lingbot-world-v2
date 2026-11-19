#!/usr/bin/env bash
# vae_sweep.sh <e2e name> <nproc> <lnc> <T...>: decode-only A/B of LINGBOT_VAE_T on that run's latents (2 decodes each)
set -uo pipefail
source kiro_mx4/common.sh
name=$1; np=$2; lnc=$3; shift 3
d="/work/out/e2e/$name"
for T in "$@"; do
  o="$d/vae_sweep_lnc${lnc}_T$T"; mkdir -p "$o"
  for i in 1 2; do
    LINGBOT_VAE_T=$T NEURON_LOGICAL_NC_CONFIG=$lnc torchrun --nproc_per_node=$np --master_port=29742 decode_latents_width_sharded.py \
      --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$o/out.mp4" --timing --warmup 2>&1 \
      | grep -avE "AliasingInfo|Mutated inputs" > "$o/dec$i.log"
  done
  v=$(cat "$o"/dec*.log | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
  echo "VAESWEEP $name LNC$lnc x$np T=$T VAE_ms=${v:-NA}" | tee -a /work/out/e2e/results.txt
done
