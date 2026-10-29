#!/usr/bin/env bash
# VAE decode bench with run_job16.sh's environment.
#   bash kiro/run_vae.sh <name> [KEY=VALUE ...]     (NPROC=64 MASTER_PORT=29760 LAT=<latents.pt>)
# Log: /work/logs/vae_<name>.log
set -uo pipefail
NAME="$1"; shift
export PYTHONUNBUFFERED=1
NPROC=64; LNC=2; MASTER_PORT=29760; LAT=/work/out/j16_ours16_latents.pt
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc"
export NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728
export NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache NKI_ENABLE_TRACE_CACHE=1 HF_HOME=/work/hf
for kv in "$@"; do export "$kv"; eval "$kv"; done
export NEURON_LOGICAL_NC_CONFIG="$LNC"
cd /work/repo
L=/work/logs/vae_${NAME}.log
{ echo "### vae $NAME start $(date -Is) nproc=$NPROC overrides: $*"; } > "$L"
if [ -n "${PROF_DIR:-}" ]; then
  # Runtime inspector (system + session device trace) on local rank 0 only.
  rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" --no-python \
    bash kiro/prof_rank0.sh python3 kiro/vae_bench.py "$LAT" >> "$L" 2>&1
else
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" kiro/vae_bench.py "$LAT" >> "$L" 2>&1
fi
echo "### exit=$? end $(date -Is)" >> "$L"
