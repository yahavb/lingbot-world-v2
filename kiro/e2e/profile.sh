#!/usr/bin/env bash
# Profiling stage of one config (run in the bench pod, devices held), after kiro/k8s_sweep.sh ran it:
#   1. a third generate.py process (2 passes, warm NEFF cache) under the runtime system trace, all ranks
#   2. ranks 0 and 1 -> parquet; per-graph latency contribution (tools/neuron_attribution.py)
#   3. a NEFF + NTFF pair for every graph above 0.5% of the run, engine overlap per pair
#      (tools/neuron_pairs.sh), summary-json (model_flops, MFU, engine busy) per pair
#   4. component split of rank 0 (tools/component_split.py) once the DiT stack hash is known
# Usage: CFG_DIR=<sweep config dir> TP=4 CP=4 NPROC=16 bash kiro/e2e/profile.sh   (env as the sweep)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
T="$REPO/kiro/e2e/tools"
D="${CFG_DIR:?}"; P="$D/prof"; rm -rf "$P"; mkdir -p "$P"
pip install -q duckdb pyarrow 2>/dev/null || true
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"
echo "=== [prof 1] system trace run ($NPROC ranks) ==="
(
  export LINGBOT_BENCH_RUNS=2 LINGBOT_DUMP_PRED= LINGBOT_DUMP_ROLLOUT= LINGBOT_SAVE_FRAMES=
  export NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=0
  export NEURON_RT_INSPECT_OUTPUT_DIR="$P/systrace"
  torchrun --nproc_per_node="$NPROC" --master_port=29742 generate.py --task i2v-A14B --infer_mode causal_fast \
    --ckpt_dir "$CKPT" --size "${SIZE:-480*768}" --frame_num 81 --chunk_size 3 --image examples/03/image.jpg \
    --action_path examples/03 --prompt "$PROMPT" --tp_degree "$TP" --cp_size "$CP" --denoise_steps 4 \
    --local_attn_size 12 --sink_size 1 --base_seed 42 --save_file "$P/out.mp4" \
    --save_latents_path "$P/latents.pt" > "$P/gen_systrace.log" 2>&1
  echo "  systrace gen exit=$?"
)
grep -E "\[BENCH\] GEN run" "$P/gen_systrace.log"
ls -d "$P"/systrace/i-* 2>/dev/null | sort -t_ -k3 -n > "$P/rankdirs.txt"
echo "  rank dirs: $(wc -l < "$P/rankdirs.txt")"
echo "=== [prof 2] parquet + attribution ==="
i=0; F=""; N=""
for d in $(cat "$P/rankdirs.txt"); do
  case $i in 0|1) bash "$T/neuron_trace_to_parquet.sh" "$d" "$P/pq/r$i" > "$P/pq_r$i.log" 2>&1
     f="$P/pq/r$i/system_profile/SystemProfileEvents.parquet"; [ -f "$f" ] && { F="$F $f"; N="$N,r$i"; };; esac
  i=$((i+1))
done
GEN_MS=$(grep -oE "GEN run2 time: [0-9.]+" "$P/gen_systrace.log" | grep -oE "[0-9.]+$")
FW=""; for f in $F; do w="${f%.parquet}_gen.parquet"; python3 "$REPO/kiro/e2e/trace_window.py" "$f" "$GEN_MS" "$w"; FW="$FW $w"; done
F="$FW"
python3 "$T/neuron_attribution.py" $F --names "${N#,}" --run-gap-s 5 --min-share 0.005 \
  --roles 'dit=model_fast|block|attention|cte|stack;vae=vae|decoder;t5=t5|umt5' \
  --json-out "$P/manifest.json" > "$P/attr.txt" 2>&1
cat "$P/attr.txt" | head -60
echo "=== [prof 3] NEFF + NTFF pairs ==="
ROLES='dit=model_fast|tensor_parallel|attention|kernels;vae=vae;t5=t5' CAPTURE_TIMEOUT=1800 \
  bash "$T/neuron_pairs.sh" "$P/manifest.json" "/tmp/neff_cache:$NEURON_COMPILE_CACHE_URL" "$P/pairs" "$NPROC" \
  > "$P/pairs.log" 2>&1
tail -30 "$P/pairs.log"
PROF="$(command -v neuron-explorer || command -v neuron-profile)"
for p in "$P"/pairs/*/; do
  n=$(ls "$p"/*.neff 2>/dev/null | head -1); t=$(ls "$p"/*.ntff 2>/dev/null | head -1)
  [ -n "$n" ] && [ -n "$t" ] || continue
  timeout 900 "$PROF" view -n "$n" -s "$t" --output-format summary-json > "$p/summary.json" 2>/dev/null
  python3 - "$p" <<'PY'
import json, os, sys
p = sys.argv[1]; d = json.load(open(os.path.join(p, "summary.json"))); s = d[list(d)[0]]
print("%-46s %8.2f ms  model_flops %.3e  mfu %5.1f%%  tensor %3.0f%%" % (
    os.path.basename(p.rstrip("/")), s["total_time"] * 1e3, s.get("model_flops", 0),
    100 * s.get("mfu_estimated_percent", 0), 100 * s.get("tensor_engine_active_time_percent", 0)))
PY
done | tee "$P/pairs_summary.txt"
echo "PROFILE DONE: $P"
