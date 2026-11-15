#!/usr/bin/env bash
# Second stage of the Neuron attribution workflow (references/profiling.md, "From system trace to
# code"): for every graph in the manifest from neuron_attribution.py, find its NEFF, capture an
# NTFF device profile, export the instruction table, run engine_overlap.py, set the role from the
# dominant source file, and store the pair under its latency label:
#   <out>/rank<r>-<ms>ms-<role>-<hash8>/{<hash>.neff, <hash>.ntff, pq/, engine_overlap.txt, .json}
# Then it checks coherence: pair time x executions against the system-trace contribution.
#
#   neuron_pairs.sh manifest.json <neff_search_dirs (colon-separated)> <out_dir> [ranks_for_collectives]
# Env: ROLES='dit=fast_dit|model_fast;vae=vae;t5=t5'  (regex on the dominant source path; default
#      role = that file's stem), CAPTURE_TIMEOUT=1800, PROF=<neuron-explorer|neuron-profile path>.
# ranks_for_collectives: world size for NEFFs with collectives (capture -r N -i 0), default 1.
# Needs: a Neuron device (capture runs the NEFF), duckdb (engine_overlap.py), python3.
set -uo pipefail
MAN="${1:?manifest.json}"; SEARCH="${2:?neff dirs}"; OUT="${3:?out dir}"; WS="${4:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF="${PROF:-$(command -v neuron-explorer || command -v neuron-profile || echo /opt/aws/neuron/bin/neuron-explorer)}"
mkdir -p "$OUT"
python3 - "$MAN" <<'PY' > "$OUT/.todo"
import json, sys
for g in json.load(open(sys.argv[1]))["graphs"]:
    print(g["hash"], g["label"], g["executions"], g["contribution_ms"], g["graph"])
PY
while read -r hash label nexec contrib graph; do
  neff=""
  IFS=':' read -r -a dirs <<< "$SEARCH"
  for d in "${dirs[@]}"; do
    neff=$(find "$d" -name "*${hash}*.neff" 2>/dev/null | head -1); [ -n "$neff" ] && break
  done
  if [ -z "$neff" ]; then
    # torch-neuronx's native backend stores NEFFs by cache key (neff_cache/ab/cd/<key>.neff); the
    # trace's model_name hash is in the NEFF header, as raw bytes or hex text. Search the headers.
    neff=$(python3 - "$hash" "$SEARCH" <<'PY'
import glob, os, sys
h, dirs = sys.argv[1], sys.argv[2].split(":")
needles = (bytes.fromhex(h), h.encode())
for d in dirs:
    for f in glob.glob(os.path.join(d, "**", "*.neff"), recursive=True):
        with open(f, "rb") as fh:
            head = fh.read(1 << 16)
        if any(n in head for n in needles):
            print(f); sys.exit(0)
PY
)
  fi
  if [ -z "$neff" ]; then echo "  $label: NEFF not found under $SEARCH"; continue; fi
  W="$OUT/.work_$hash"; rm -rf "$W"; mkdir -p "$W"; cp "$neff" "$W/$hash.neff"
  extra=(); [ "$WS" -gt 1 ] && extra=(-r "$WS" -i 0)
  ( cd "$W" && timeout "${CAPTURE_TIMEOUT:-1800}" "$PROF" capture -n "$hash.neff" -s "$hash.ntff" \
      "${extra[@]}" --num-exec=1 --enable-dge-notifs ) > "$W/capture.log" 2>&1
  [ -s "$W/$hash.ntff" ] || { f=$(ls -S "$W"/*rank_0*.ntff "$W"/*.ntff 2>/dev/null | head -1); [ -n "$f" ] && mv "$f" "$W/$hash.ntff"; }
  if [ ! -s "$W/$hash.ntff" ]; then echo "  $label: capture failed"; tail -5 "$W/capture.log"; continue; fi
  # instruction table (the writer idles after writing). Stop it only once Instruction.parquet is
  # READABLE: a size that stops growing is not enough -- on a 2 GB NTFF the writer paused mid-file
  # and the stable-size rule killed it, leaving a parquet with no footer ("No magic bytes").
  timeout 5400 "$PROF" view -n "$W/$hash.neff" -s "$W/$hash.ntff" --output-format parquet \
      --output-file "$W/pq" --ignore-event-trace > "$W/view.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 1080); do sleep 5; kill -0 $pid 2>/dev/null || break
    python3 -c "import duckdb,sys; duckdb.connect().execute(\"select count(*) from read_parquet('$W/pq/Instruction.parquet')\").fetchall()" \
      2>/dev/null && { sleep 20; break; }
  done
  kill $pid 2>/dev/null
  python3 "$HERE/engine_overlap.py" "$W/pq" --top 15 --json-out "$W/engine_overlap.json" > "$W/engine_overlap.txt" 2>&1
  final=$(python3 - "$W" "$label" "$nexec" "$contrib" "${ROLES:-}" <<'PY'
import collections, json, os, re, sys
w, label, nexec, contrib, roles = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
txt = open(f"{w}/engine_overlap.txt").read()
m = re.search(r"window\s+([0-9.]+) ms", txt)
pair_ms = float(m.group(1)) if m else None
src = collections.Counter()
for line in txt.splitlines():   # "  <ms> ms  <pct>%  <source>  <opcode>  <hlo>" rows of the per-engine lists
    mm = re.match(r"\s+([0-9.]+) ms\s+[0-9.]+%\s+(\S+)", line)
    if mm and not mm.group(2).startswith("("):
        src[mm.group(2).split(":")[0]] += float(mm.group(1))
top = src.most_common(1)[0][0] if src else ""
role = None
for r in filter(None, roles.split(";")):
    k, rx = r.split("=", 1)
    if re.search(rx, top):
        role = k; break
role = role or (re.sub(r"\W+", "", os.path.splitext(os.path.basename(top))[0])[:12] or "unk")
parts = label.split("-")
new = "-".join(parts[:2] + ([role] if parts[2] == "unk" else [parts[2]]) + parts[3:])
coh = f"pair {pair_ms:.3f} ms x {nexec} = {pair_ms * nexec:.1f} ms vs system trace {contrib:.1f} ms " \
      f"({100 * (pair_ms * nexec - contrib) / contrib:+.0f}%)" if pair_ms else "pair time unknown"
json.dump({"label": new, "dominant_source": top, "coherence": coh}, open(f"{w}/pair.json", "w"))
print(new + "\t" + coh + "\t" + top)
PY
)
  new="${final%%$'\t'*}"; rest="${final#*$'\t'}"
  rm -rf "$OUT/$new"; mv "$W" "$OUT/$new"
  echo "  $new  | ${rest//$'\t'/  | code: }"
done < "$OUT/.todo"
rm -f "$OUT/.todo"
