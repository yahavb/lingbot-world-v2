#!/usr/bin/env bash
# Single-block bench (kiro/block_bench.py) with the pipeline's env, inside the lbw container.
#   bash kiro/run_bb.sh <name> <nproc> <lnc> <visible_cores> [KEY=VALUE ...]
# Log: /work/logs/bb_<name>.log (one result line starting with [block_bench])
set -uo pipefail
NAME="$1"; NPROC="$2"; LNC="$3"; CORES="$4"; shift 4
export PYTHONUNBUFFERED=1
export NEURON_LOGICAL_NC_CONFIG="$LNC" NEURON_RT_VISIBLE_CORES="$CORES"
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache NKI_ENABLE_TRACE_CACHE=1
export LINGBOT_CHUNKED_KV=1 LINGBOT_COMPILE_DIT=1 COMPILE_SCOPE=blocks LINGBOT_CP_FLAT_SEQ=1
export ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte FLASH_ATTN_LAYERS=33,34,38,39
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc"
export NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728 NEURON_LOAD_CONCURRENCY=8
export NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR NEURON_LAUNCH_BLOCKING=1
export TP="$NPROC" CP=1 MASTER_PORT=29720
for kv in "$@"; do export "$kv"; done
mkdir -p /work/logs /work/out
LOG=/work/logs/bb_${NAME}.log
cd /work/repo
echo "### bb $NAME start $(date -Is) nproc=$NPROC lnc=$LNC cores=$CORES overrides: $*" > "$LOG"
if [ -n "${PROF_DIR:-}" ]; then
  rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" --no-python \
    bash kiro/prof_rank0.sh python3 kiro/block_bench.py >> "$LOG" 2>&1
else
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" kiro/block_bench.py >> "$LOG" 2>&1
fi
echo "### exit=$? end $(date -Is)" >> "$LOG"
