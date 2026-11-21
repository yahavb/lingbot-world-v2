#!/usr/bin/env bash
# run_cell.sh DIT_APP CELL [CHUNKS] [extra drift_probe args]: one steady-state measurement of a matrix cell.
# Waits for the DiT pod (app=DIT_APP) to be Ready, runs perf/drift_probe.py inside it (scene 03, the scene's
# own camera path, no key presses), copies the per-chunk CSV to perf/disagg/results/CELL.csv and prints a
# summary line (perf/disagg/summarize.py).
set -euo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
app=$1; cell=$2; chunks=${3:-60}; shift $(( $# < 3 ? $# : 3 ))
cd "$(dirname "$0")/../.."
until p=$($K get pods -l app=$app --no-headers 2>/dev/null | grep -v Terminating | awk '{split($2,r,"/"); if (r[1]==r[2]) print $1}' | head -1) && [ -n "$p" ]; do
  q=$($K get pods -l app=$app --no-headers 2>/dev/null | awk '{print $1}' | head -1)
  [ -n "$q" ] && $K logs "$q" --tail=300 2>/dev/null | grep -aq "^Traceback" && { echo "FAILED $cell: $q"; $K logs "$q" --tail=400 | grep -aE "Error|error:" | tail -5; exit 1; }
  sleep 30
done
$K cp perf/drift_probe.py "$p:/tmp/drift_probe.py" -c inference
# the probe runs detached inside the pod (a dropped kubectl connection must not lose the measurement); poll for it
$K exec "$p" -c inference -- bash -c "rm -rf /logs/$cell; mkdir -p /logs/$cell; cd /tmp/lingbot && nohup python3 /tmp/drift_probe.py --url http://127.0.0.1:8000 --chunks $chunks --no-steer --every 1000 --out /logs/$cell $* > /logs/$cell.out 2>&1 &"
until $K exec "$p" -c inference -- test -f "/logs/$cell/drift.csv" 2>/dev/null; do
  sleep 20
  $K get pod "$p" >/dev/null 2>&1 || { echo "FAILED $cell: pod gone"; exit 1; }
done
$K exec "$p" -c inference -- cat "/logs/$cell.out" > "perf/disagg/results/$cell.log" 2>&1
$K cp -c inference "$p:/logs/$cell/drift.csv" "perf/disagg/results/$cell.csv"
python3 perf/disagg/summarize.py "perf/disagg/results/$cell.csv" "$cell" | tee -a perf/disagg/results/summary.txt
