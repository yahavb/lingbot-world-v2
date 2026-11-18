#!/usr/bin/env bash
# perf/vae3_unit_timing.sh <tag> <visible cores> <port> [K=V ...]: perf/vae_unit_timing.sh with the cores and the torchrun
# port as arguments (helper lanes share a pod): one 16-rank LNC1 decode process (warmup + steady decode) of the gate
# latents with LINGBOT_VAE_UNIT_TIMING=1 (a device sync around every block call; DIAGNOSTIC), run from the current
# directory; prints rank 0's per-(block, shape) synced times of the steady decode.
here=$(pwd)
source kiro_mx4/common.sh
cd "$here" || exit 1
tag=$1; vis=$2; port=$3; shift 3
for kv in "$@"; do export "$kv"; done
export NEURON_RT_VISIBLE_CORES=$vis NEURON_LOGICAL_NC_CONFIG=1 LINGBOT_VAE_T=${LINGBOT_VAE_T:-1} LINGBOT_VAE_UNIT_TIMING=1
R=/node-cache/e2e/gate/r480_ex03_s42; o=/work/out/vaeunit/$tag; rm -rf $o; mkdir -p $o
torchrun --nproc_per_node=16 --master_port=$port decode_latents_width_sharded.py \
  --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$R/latents.pt" --save_file "$o/out.mp4" --timing --warmup 2>&1 \
  | grep -avE "AliasingInfo|Mutated inputs" > $o/dec.log
# the second VAEUNIT block is the steady-state decode
awk '/VAEUNIT total/{n++} {if (n >= 1 && /VAEUNIT/) print}' $o/dec.log | tail -n +2 | sed "s/^/$tag /"
grep -a "TIMING" $o/dec.log | sed "s/^/$tag /"
