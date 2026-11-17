#!/usr/bin/env bash
# One single-logical-core Python test (kernel tests: kiro/cmax_guard_test.py, kiro/nki_branch_test.py) with the
# pipeline's compiler / runtime env, inside the lbw container.
#   bash kiro/run_core.sh <core> <script.py> [KEY=VALUE ...]      (log: /work/logs/core_<script>_<core>.log)
set -uo pipefail
CORE="$1"; SCRIPT="$2"; shift 2
export PYTHONUNBUFFERED=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_VISIBLE_CORES="$CORE"
export NEURON_COMPILE_CACHE_URL=/tmp/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/tmp/hlo_cache NKI_ENABLE_TRACE_CACHE=0
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc"
export NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
export NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR
for kv in "$@"; do export "$kv"; done
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG=/work/logs/core_$(basename "$SCRIPT" .py)_${CORE}.log
echo "### $SCRIPT core $CORE start $(date -Is) overrides: $*" > "$LOG"
python3 "$SCRIPT" >> "$LOG" 2>&1
echo "### exit=$? end $(date -Is)" >> "$LOG"
grep -E "^\[|^### exit" "$LOG" | tail -40
