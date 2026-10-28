#!/usr/bin/env bash
# One generate.py e2e run (T5 + DiT + width-sharded VAE in one torchrun) inside the lbw container.
#   bash kiro/run_lbw.sh <name> <nproc> <lnc> <visible_cores> [KEY=VALUE ...] [-- extra generate.py args]
# Defaults mirror lbw-v2-i2v-job.yaml (480*768, 81 frames, chunk 3, window 12, sink 1, 4 steps) at
# --tp_degree <nproc> --cp_size 1; override with TP=, CP=, SIZE=, FRAME_NUM=, ... as KEY=VALUE.
# Log: /work/logs/<name>.log   video: /work/out/<name>.mp4
set -uo pipefail
NAME="$1"; NPROC="$2"; LNC="$3"; CORES="$4"; shift 4
EXTRA=()
KV=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
  KV+=("$1"); shift
done
export PYTHONUNBUFFERED=1
export NEURON_LOGICAL_NC_CONFIG="$LNC" NEURON_RT_VISIBLE_CORES="$CORES"
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache NKI_ENABLE_TRACE_CACHE=1
export LINGBOT_BENCH_RUNS=2 LINGBOT_CHUNKED_KV=1 LINGBOT_COMPILE_DIT=1 COMPILE_SCOPE=blocks LINGBOT_CP_FLAT_SEQ=1
export ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte FLASH_ATTN_LAYERS=33,34,38,39
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc"
export NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728 NEURON_LOAD_CONCURRENCY=8
export NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR NEURON_LAUNCH_BLOCKING=1
TP="$NPROC"; CP=1; SIZE="480*768"; FRAME_NUM=81; CHUNK_SIZE=3; LOCAL_ATTN_SIZE=12; SINK_SIZE=1
DENOISE_STEPS=4; MASTER_PORT=29700
for kv in "${KV[@]}"; do export "$kv"; eval "$kv"; done
CKPT=/work/ckpt/lingbot-world-v2-14b-causal-fast
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"
mkdir -p /work/logs /work/out
LOG=/work/logs/${NAME}.log
cd /work/repo
{
  echo "### $NAME start $(date -Is) nproc=$NPROC lnc=$LNC cores=$CORES tp=$TP cp=$CP size=$SIZE frames=$FRAME_NUM chunk=$CHUNK_SIZE extra=${EXTRA[*]:-}"
  env | grep -E "^(LINGBOT_|ATTENTION_|COMPILE_|FLASH_|NEURON_|NKI_|TORCH_NEURONX)" | sort
} > "$LOG"
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" generate.py \
  --task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT" \
  --size "$SIZE" --frame_num "$FRAME_NUM" --chunk_size "$CHUNK_SIZE" \
  --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT" \
  --tp_degree "$TP" --cp_size "$CP" --denoise_steps "$DENOISE_STEPS" \
  --local_attn_size "$LOCAL_ATTN_SIZE" --sink_size "$SINK_SIZE" --base_seed 42 \
  --save_file "/work/out/${NAME}.mp4" "${EXTRA[@]}" >> "$LOG" 2>&1
echo "### exit=$? end $(date -Is)" >> "$LOG"
