#!/usr/bin/env bash
# cc_alg_sweep.sh <out dir> "<NEURON_RT_CC_ALG_TYPES value>" ...: perf/block_bench.py once per collective-algorithm
# list (the runtime knob; "" = the runtime's own choice), the block flags from $C (comma-separated K=V), one line
# each. The compiled NEFFs are cached, so each value costs one load + REPS blocks.
#   C="K=V,..." bash perf/cc_alg_sweep.sh /work/out/cc "" "INTRA_FULL_MESH_ALG,INTER_FULL_MESH_ALG"
set -uo pipefail
OUT="${1:?out dir}"; shift; mkdir -p "$OUT"
export NEURON_LOGICAL_NC_CONFIG="${LNC:-2}" REPS="${REPS:-10}" BLOCK="${BLOCK:-10}"
IFS=',' read -r -a kvs <<< "${C:-}"; for kv in "${kvs[@]}"; do [ -n "$kv" ] && export "$kv"; done
i=0
for v in "$@"; do
  i=$((i + 1))
  (
    if [ -n "$v" ]; then export NEURON_RT_CC_ALG_TYPES="$v"; else unset NEURON_RT_CC_ALG_TYPES; fi
    torchrun --nproc_per_node="$NPROC" --master_port="${PORT:-29733}" perf/block_bench.py 2>&1 \
      | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$OUT/cc_$i.log"
    echo "[cc] ${v:-default}: $(grep -a '^\[block_bench\] tp=' "$OUT/cc_$i.log" | head -1 | grep -oE '[0-9.]+ ms/block|rel_rms_vs_ref=[0-9.e-]+') $(grep -m1 -aoE 'Invalid NEURON_RT_CC_ALG_TYPES[^.]*|Traceback' "$OUT/cc_$i.log")"
  )
done
echo "[cc] done"
