#!/usr/bin/env bash
# env_sweep.sh <out dir> "K=V;K=V" ...: perf/block_bench.py once per set of runtime env assignments (';'-separated,
# values may contain commas; "" = none), on top of the block flags in $C (comma-separated K=V). One line each. The
# compiled NEFFs are cached, so runtime-only knobs (NEURON_RT_*) cost one load + REPS blocks per set.
#   C="K=V,..." bash perf/env_sweep.sh /work/out/es "" "NEURON_RT_DBG_CC_DMA_PACKET_SIZE=65536"
set -uo pipefail
OUT="${1:?out dir}"; shift; mkdir -p "$OUT"
export NEURON_LOGICAL_NC_CONFIG="${LNC:-2}" REPS="${REPS:-10}" BLOCK="${BLOCK:-10}"
IFS=',' read -r -a kvs <<< "${C:-}"; for kv in "${kvs[@]}"; do [ -n "$kv" ] && export "$kv"; done
i=0
for v in "$@"; do
  i=$((i + 1))
  (
    IFS=';' read -r -a es <<< "$v"; for e in "${es[@]}"; do [ -n "$e" ] && export "$e"; done
    torchrun --nproc_per_node="$NPROC" --master_port="${PORT:-29733}" perf/block_bench.py 2>&1 \
      | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$OUT/es_$i.log"
    echo "[es] ${v:-default}: $(grep -a '^\[block_bench\] tp=' "$OUT/es_$i.log" | head -1 | grep -oE '[0-9.]+ ms/block') $(grep -m1 -aoE 'Invalid [A-Z_]+[^.]*|Traceback' "$OUT/es_$i.log")"
  )
done
echo "[es] done"
