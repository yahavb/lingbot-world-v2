#!/usr/bin/env bash
# The upstream 16-device job (lbw-v2-i2v-job.yaml) on this host, inside the lbw container:
# generate.py (GEN, steady-state pass 2) then decode_latents_width_sharded.py (VAE steady state),
# with the job's args and env, run from REPO. steady fps = FRAME_NUM / ((GEN + VAE) / 1000),
# exactly as the job computes it.
#   bash kiro/run_job16.sh <repo_dir> <name> [KEY=VALUE ...]     (KEY=VALUE overrides the job env)
# Logs: /work/logs/<PFX>_<name>_{gen,dec}.log, summary line in /work/logs/<PFX>_<name>.log
# (PFX=j16 by default). NEURON_RT_VISIBLE_CORES=<comma list> puts local rank i on the i-th core
# listed (kiro/core_order.py gives the order the job's TP8 x CP8 needs on this host); two runs
# sharing the host need distinct MASTER_PORT=... values.
set -uo pipefail
REPO="$1"; NAME="$2"; shift 2
export PYTHONUNBUFFERED=1
# --- job yaml env ---
NPROC=64; LNC=2; CP=8; TP=8; SIZE="480*768"; FRAME_NUM=81; CHUNK_SIZE=3; LOCAL_ATTN_SIZE=12
SINK_SIZE=1; DENOISE_STEPS=4; MASTER_PORT=29740; PFX=j16
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"
export LINGBOT_BENCH_RUNS=2 LINGBOT_CHUNKED_KV=1 LINGBOT_COMPILE_DIT=1 COMPILE_SCOPE=blocks LINGBOT_CP_FLAT_SEQ=1
export ATTENTION_BACKEND=const_max_ag
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc"
export NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728 NEURON_LOAD_CONCURRENCY=8
export NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR
# the job computes this with scripts/analysis/qk_gamma_temperature.py; same checkpoint, same answer
export FLASH_ATTN_LAYERS=33,34,38,39
# --- host-local paths (the job's /var/tmp caches) ---
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache NKI_ENABLE_TRACE_CACHE=1 HF_HOME=/work/hf
for kv in "$@"; do export "$kv"; eval "$kv"; done
export NEURON_LOGICAL_NC_CONFIG="$LNC"
CKPT=/work/ckpt/lingbot-world-v2-14b-causal-fast
mkdir -p /work/logs /work/out
SAVE=/work/out/${PFX}_${NAME}.mp4; LATENTS=/work/out/${PFX}_${NAME}_latents.pt
GL=/work/logs/${PFX}_${NAME}_gen.log; DL=/work/logs/${PFX}_${NAME}_dec.log; SL=/work/logs/${PFX}_${NAME}.log
cd "$REPO"
{
  echo "### $PFX $NAME start $(date -Is) repo=$REPO nproc=$NPROC lnc=$LNC tp=$TP cp=$CP" \
       "cores=${NEURON_RT_VISIBLE_CORES:-all} overrides: $*"
  env | grep -E "^(LINGBOT_|ATTENTION_|COMPILE_|FLASH_|NEURON_|NKI_|TORCH_NEURONX)" | sort
} > "$SL"
ARGS=(--task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT"
      --size "$SIZE" --frame_num "$FRAME_NUM" --chunk_size "$CHUNK_SIZE"
      --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT"
      --tp_degree "$TP" --cp_size "$CP" --denoise_steps "$DENOISE_STEPS"
      --local_attn_size "$LOCAL_ATTN_SIZE" --sink_size "$SINK_SIZE" --base_seed 42)
cp "$SL" "$GL"
# PROMPT_EMBEDS=<file from kiro/encode_prompt.py>: skip generate.py's inline T5 (it does not
# enter GEN; needed when the core order breaks world-wide collectives, see encode_prompt.py).
if [ -n "${PROMPT_EMBEDS:-}" ]; then ARGS+=(--prompt_embeds_path "$PROMPT_EMBEDS"); fi
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" generate.py \
  "${ARGS[@]}" --save_file "$SAVE" --save_latents_path "$LATENTS" >> "$GL" 2>&1
echo "### gen exit=$? end $(date -Is)" | tee -a "$GL" >> "$SL"
# DEC_DEFAULT_CORES=1: decode with the default core order even when generate ran with a custom
# NEURON_RT_VISIBLE_CORES order (the VAE step is the same code in every run we compare).
if [ "${DEC_DEFAULT_CORES:-0}" = "1" ]; then unset NEURON_RT_VISIBLE_CORES; fi
torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" decode_latents_width_sharded.py \
  --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$LATENTS" --save_file "$SAVE" \
  --timing --warmup > "$DL" 2>&1
echo "### dec exit=$? end $(date -Is)" | tee -a "$DL" >> "$SL"
GEN=$(grep -oE "\[BENCH\] GEN time: [0-9.]+" "$GL" | grep -oE "[0-9.]+$" | tail -1)
VAE=$(grep -oE "vae\.decode steady-state \([^)]*\): [0-9.]+" "$DL" | grep -oE "[0-9.]+$" | tail -1)
STEADY=$(python3 -c "print(round($FRAME_NUM / ((${GEN:-nan} + ${VAE:-nan}) / 1000), 3))" 2>/dev/null)
echo "### RESULT $NAME GEN_ms=${GEN:-NA} VAE_ms=${VAE:-NA} steady_fps=${STEADY:-NA}" >> "$SL"
python3 tests/smoke_test_video.py --skip-run --save-file "$SAVE" \
  --expected-frame-num "$FRAME_NUM" --size "$SIZE" --steady-fps "${STEADY:-0}" >> "$SL" 2>&1
echo "### smoke exit=$? end $(date -Is)" >> "$SL"
