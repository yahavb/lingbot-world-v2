#!/usr/bin/env bash
# perf/vae_repeat_check.sh <tag> <visible cores> <port> <n> [K=V ...]: one 16-rank LNC1 decode process of the gate
# latents (perf/vae_env_check_here.sh's environment, run from the current directory) that decodes the same latents
# n more times after the warm decode (LINGBOT_VAE_REPEAT) and scores every decode against R: prints rank 0's REPEAT
# lines (ms, PSNR, frames below 50 dB) and the summary, to measure how often an intermittent decode error occurs.
here=$(pwd)
source kiro_mx4/common.sh
cd "$here" || exit 1
tag=$1; vis=$2; port=$3; n=$4; shift 4
for kv in "$@"; do export "$kv"; done
export NEURON_RT_VISIBLE_CORES=$vis NEURON_LOGICAL_NC_CONFIG=1 LINGBOT_VAE_T=${LINGBOT_VAE_T:-1} LINGBOT_VAE_REPEAT=$n
R=/node-cache/e2e/gate/r480_ex03_s42; o=/work/out/vaerep/$tag; rm -rf $o; mkdir -p $o
torchrun --nproc_per_node=16 --master_port=$port decode_latents_width_sharded.py \
  --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$R/latents.pt" --save_file "$o/out.mp4" --timing --warmup 2>&1 \
  | grep -avE "AliasingInfo|Mutated inputs" > $o/dec.log
grep -aoE "REPEAT .*" $o/dec.log | sed "s/^/$tag /"
grep -m1 -aoE 'Error.{0,160}' $o/dec.log
