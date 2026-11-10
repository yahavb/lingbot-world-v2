#!/usr/bin/env bash
# NEFF + NTFF pairs for a manifest from tools/neuron_attribution.py (devices held, nothing else running):
#   pairs.sh <manifest.json> <neff dir> <out dir> <world size>
# Then summary-json per pair: time, model FLOPs, MFU, tensor-engine busy.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; T="$REPO/kiro/e2e/tools"
MAN="$1"; NEFFS="$2"; OUT="$3"; WS="$4"
pip install -q duckdb pyarrow 2>/dev/null || true
ROLES='dit=model_fast|tensor_parallel|attention|kernels;vae=vae;t5=t5' CAPTURE_TIMEOUT=1800 \
  bash "$T/neuron_pairs.sh" "$MAN" "$NEFFS" "$OUT" "$WS" > "$OUT.log" 2>&1
tail -30 "$OUT.log"
PROF="$(command -v neuron-explorer || command -v neuron-profile)"
for p in "$OUT"/*/; do
  n=$(ls "$p"/*.neff 2>/dev/null | head -1); t=$(ls "$p"/*.ntff 2>/dev/null | head -1)
  [ -n "$n" ] && [ -n "$t" ] || continue
  timeout 900 "$PROF" view -n "$n" -s "$t" --output-format summary-json > "$p/summary.json" 2>/dev/null
  python3 - "$p" <<'PY'
import json, os, sys
p = sys.argv[1]; d = json.load(open(os.path.join(p, "summary.json"))); s = d[list(d)[0]]
print("%-46s %8.3f ms  model_flops %.3e  mfu %5.1f%%  tensor %3.0f%%" % (
    os.path.basename(p.rstrip("/")), s["total_time"] * 1e3, s.get("model_flops", 0),
    100 * s.get("mfu_estimated_percent", 0), 100 * s.get("tensor_engine_active_time_percent", 0)))
PY
done | tee "$OUT/pairs_summary.txt"
echo "PAIRS DONE"
