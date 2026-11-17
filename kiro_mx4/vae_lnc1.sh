#!/usr/bin/env bash
# vae_lnc1.sh <e2e name>: decode that run's latents twice at LNC1 (32 ranks on the same 4 devices)
set -uo pipefail
source kiro_mx4/common.sh
d="/work/out/e2e/$1"; o="$d/vae_lnc1"; mkdir -p "$o"
for i in 1 2; do
  NEURON_LOGICAL_NC_CONFIG=1 torchrun --nproc_per_node=32 --master_port=29741 decode_latents_width_sharded.py \
    --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$o/out.mp4" \
    --timing --warmup 2>&1 | grep -avE "AliasingInfo|Mutated inputs" > "$o/dec$i.log"
done
v=$(cat "$o"/dec*.log | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
echo "VAERESULT $1 LNC1x32 VAE_ms=${v:-NA}" | tee -a /work/out/e2e/results.txt
grep -m3 -ahE "Traceback|Error|error:" "$o/dec1.log" | cut -c1-200
