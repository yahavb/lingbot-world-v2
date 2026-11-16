#!/usr/bin/env bash
# One DiT block (perf/block_bench.py, real weights of $BLOCK), A/B of named env variants at the pod's
# layout, each timed twice (self-noise); the first variant saves the reference output, the others
# report rel-RMS against it. PROF_VARIANT=<name>: also a rank-0 device profile of that variant,
# converted and summarized with perf/prof_block.py (largest execution).
#   VARIANTS="bf16:LINGBOT_FFN_MX=0,LINGBOT_LIN_MX=0 mx:LINGBOT_LIN_MX=1" bash perf/mx_block_ab.sh <out>
set -uo pipefail
OUT="${1:?out dir}"; mkdir -p "$OUT"
export NEURON_LOGICAL_NC_CONFIG="${LNC:-2}" REPS="${REPS:-20}" BLOCK="${BLOCK:-10}"
pip install -q pandas pyarrow 2>/dev/null || true
PROF="$(command -v neuron-explorer || echo /opt/aws/neuron/bin/neuron-explorer)"
first=""
for v in $VARIANTS; do
  name="${v%%:*}"; kvs="${v#*:}"
  (
    IFS=',' read -r -a a <<< "$kvs"; for kv in "${a[@]}"; do export "$kv"; done
    if [ -z "$first" ]; then export SAVE_REF="$OUT/ref.pt"; else export REF="$OUT/ref.pt"; fi
    for i in 1 2; do
      torchrun --nproc_per_node="$NPROC" --master_port=29733 perf/block_bench.py 2>&1 \
        | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$OUT/bb_${name}_$i.log"
      echo "[ab] $name run$i: $(grep -a '^\[block_bench\] tp=' "$OUT/bb_${name}_$i.log" | head -1 | grep -oE '[0-9.]+ ms/block|rel_rms_vs_ref=[0-9.e-]+')"
      grep -m3 -aE "Traceback|Error|failed" "$OUT/bb_${name}_$i.log" | cut -c1-200
      unset SAVE_REF
    done
    if [ "${PROF_VARIANT:-}" = "$name" ]; then
      export PROF_DIR="$OUT/prof_$name"; rm -rf "$PROF_DIR"; mkdir -p "$PROF_DIR"
      REPS=5 torchrun --nproc_per_node="$NPROC" --master_port=29734 --no-python \
        bash perf/prof_rank0.sh python3 perf/block_bench.py > "$OUT/bbprof_$name.log" 2>&1
      d=$(ls -d "$PROF_DIR"/* | head -1); P="$OUT/pq_$name"; rm -rf "$P"; mkdir -p "$P"
      "$PROF" view -d "$d" --output-format parquet --output-file "$P" > "$P.log" 2>&1 &
      pid=$!; prev=-1; st=0
      for _ in $(seq 1 900); do sleep 2; kill -0 $pid 2>/dev/null || break
        cur=$(du -sk "$P" | cut -f1); [ -n "$(find "$P" -name Instruction.parquet)" ] && [ "$cur" = "$prev" ] && st=$((st+1)) || st=0
        prev=$cur; [ $st -ge 15 ] && { kill $pid; break; }; done
      s=$(dirname "$(find "$P" -name Instruction.parquet | head -1)")
      ex=$(python3 -c "
import pandas as pd,sys; i=pd.read_parquet(sys.argv[1]+'/Instruction.parquet',columns=['execution_index','start_ts','end_ts'])
g=i.groupby('execution_index').agg(a=('start_ts','min'),b=('end_ts','max')); g['ms']=g.b-g.a
print(g.sort_values('ms').index[len(g)//2] if False else g.ms.sort_values().index[-3])" "$s")
      python3 perf/prof_block.py "$s" "$ex" 2>&1 | sed -n '/=== execution/,/^\[GpSimd\]/p' | head -8 | sed "s/^/[ab] prof $name: /"
    fi
  )
  [ -z "$first" ] && first="$name"
done
echo "MX_BLOCK_AB DONE"
