#!/usr/bin/env bash
# One DiT block (real weights of block $BLOCK), bf16 vs MXFP8 Linears, at the pod's layout (TP, CP,
# NPROC, LNC and every LINGBOT_* flag from the environment):
#   1. timed: perf/block_bench.py, ms/block; bf16 saves its output, MX compares against it
#   2. profiled: the same under the runtime device profile on rank 0, converted to parquet,
#      perf/prof_block.py -> per-engine busy, top ops per engine, framework nodes with MFU
# Usage (in the pod, repo root): bash perf/mx_block_prof.sh <out dir>
set -uo pipefail
OUT="${1:?out dir}"; mkdir -p "$OUT"
export NEURON_LOGICAL_NC_CONFIG="${LNC:-2}" REPS="${REPS:-20}" BLOCK="${BLOCK:-10}"
pip install -q pandas pyarrow duckdb 2>/dev/null || true
PROF="$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)"

to_parquet() {   # <runtime inspect dir> <out dir>: device + system profile -> parquet
  rm -rf "$2"; mkdir -p "$2"
  "$PROF" view -d "$1" --output-format parquet --output-file "$2" > "$2.log" 2>&1 &
  local pid=$! prev=-1 stable=0 cur
  for _ in $(seq 1 900); do
    sleep 2
    kill -0 "$pid" 2>/dev/null || break
    cur=$(du -sk "$2" | cut -f1)
    if [ -n "$(find "$2" -name Instruction.parquet | head -1)" ] && [ "$cur" = "$prev" ]; then
      stable=$((stable + 1)); else stable=0; fi
    prev=$cur
    [ "$stable" -ge 15 ] && { kill "$pid"; break; }
  done
}

for MX in 0 1; do
  tag=$([ "$MX" = 1 ] && echo mx || echo bf16)
  export LINGBOT_FFN_MX=$MX LINGBOT_LIN_MX=$MX
  if [ "$MX" = 0 ]; then export SAVE_REF="$OUT/ref_bf16.pt"; unset REF; else export REF="$OUT/ref_bf16.pt"; unset SAVE_REF; fi
  echo "=== [$tag] timed"
  torchrun --nproc_per_node="$NPROC" --master_port=29731 perf/block_bench.py 2>&1 \
    | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$OUT/bb_$tag.log"
  grep -aE "^\[block_bench\]|LIN_MX|FFN_MX|Traceback|Error" "$OUT/bb_$tag.log" | head -20
  unset SAVE_REF REF
  echo "=== [$tag] profiled (rank 0 device profile)"
  export PROF_DIR="$OUT/prof_$tag"; rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
  REPS=5 torchrun --nproc_per_node="$NPROC" --master_port=29732 --no-python \
    bash perf/prof_rank0.sh python3 perf/block_bench.py > "$OUT/bbprof_$tag.log" 2>&1
  unset PROF_DIR
  d=$(ls -d "$OUT/prof_$tag"/* 2>/dev/null | head -1)
  echo "  inspect dir: $d"
  to_parquet "$d" "$OUT/pq_$tag"
  s=$(dirname "$(find "$OUT/pq_$tag" -name Instruction.parquet | head -1)")
  echo "  parquet: $s"
  [ -n "$s" ] && python3 perf/prof_block.py "$s" > "$OUT/block_$tag.txt" 2>&1
  head -80 "$OUT/block_$tag.txt"
done
echo "MX_BLOCK_PROF DONE: $OUT"
