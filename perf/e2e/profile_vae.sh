#!/usr/bin/env bash
# SKILL.md Step 3 for the VAE decode (the counterpart of perf/e2e/profile.sh for generate.py), run in
# the bench pod after perf/k8s_sweep.sh produced <CFG_DIR>/latents.pt, same env as the sweep:
#   1. decode_latents_width_sharded.py (--warmup + timed decode) under the runtime system trace, all ranks
#   2. every rank -> parquet; per-graph latency contribution (tools/neuron_attribution.py, role vae)
#   3. a NEFF + NTFF pair for every graph above 0.5% of the decode, engine overlap per pair
#      (tools/neuron_pairs.sh), summary-json per pair
# Usage: CFG_DIR=<sweep config dir> NPROC=32 bash perf/e2e/profile_vae.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
T="$REPO/perf/e2e/tools"
D="${CFG_DIR:?}"; P="$D/prof_vae"; [ -n "${RESUME:-}" ] || rm -rf "$P"; mkdir -p "$P"
export NEURON_LOGICAL_NC_CONFIG="${LNC:?LNC}"   # outside k8s_sweep.sh run(): set it here
pip install -q duckdb pyarrow 2>/dev/null || true
if [ -z "${RESUME:-}" ]; then
echo "=== [vae 1] system trace decode ($NPROC ranks) ==="
NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=0 \
NEURON_RT_INSPECT_OUTPUT_DIR="$P/systrace" torchrun --nproc_per_node="$NPROC" --master_port=29743 \
  decode_latents_width_sharded.py --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$D/latents.pt" \
  --save_file "$P/out.mp4" --timing --warmup > "$P/dec_systrace.log" 2>&1
echo "  systrace decode exit=$?"
grep -E "vae\.decode steady-state" "$P/dec_systrace.log"
ls -d "$P"/systrace/i-* 2>/dev/null | sort -t_ -k3 -n > "$P/rankdirs.txt"
fi   # RESUME=1: keep the trace and every rank already converted
echo "  rank dirs: $(wc -l < "$P/rankdirs.txt")"
echo "=== [vae 2] parquet (every rank) + attribution ==="
i=0
for d in $(cat "$P/rankdirs.txt"); do
  [ -n "${RESUME:-}" ] && [ -f "$P/pq/r$i/system_profile/SystemProfileEvents.parquet" ] && { i=$((i+1)); continue; }
  bash "$T/neuron_trace_to_parquet.sh" "$d" "$P/pq/r$i" > "$P/pq_r$i.log" 2>&1 &
  i=$((i+1)); [ $((i % 8)) = 0 ] && wait
done; wait
F=""; N=""; i=0
for d in $(cat "$P/rankdirs.txt"); do
  f="$P/pq/r$i/system_profile/SystemProfileEvents.parquet"; [ -f "$f" ] && { F="$F $f"; N="$N,r$i"; }
  i=$((i+1))
done
echo "  ranks converted: $(echo $F | wc -w) of $i"
# the timed decode only: the trace also holds compile, the warm-up decode and the gather/save tail
VAE_MS=$(grep -oE "vae\.decode steady-state \([^)]*\): [0-9.]+" "$P/dec_systrace.log" | grep -oE "[0-9.]+$" | tail -1)
FW=""; for f in $F; do w="${f%.parquet}_dec.parquet"; python3 "$REPO/perf/e2e/trace_window.py" "$f" "$VAE_MS" "$w" busiest > /dev/null; FW="$FW $w"; done
F="$FW"; echo "  window: ${VAE_MS} ms ending at each rank's busiest segment"
python3 "$T/neuron_attribution.py" $F --names "${N#,}" --run-gap-s 2 --min-share 0.005 \
  --roles 'vae=vae|decoder|conv' --json-out "$P/manifest.json" > "$P/attr.txt" 2>&1
head -60 "$P/attr.txt"
echo "=== [vae 3] NEFF + NTFF pairs ==="
ROLES='vae=vae|decoder|conv' CAPTURE_TIMEOUT=1800 \
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
echo "PROFILE_VAE DONE: $P"
