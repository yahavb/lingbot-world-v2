#!/usr/bin/env bash
# run_lat.sh APP CELL [PRESSES]: press-to-photon latency (perf/disagg/latency_probe.py) inside the DiT pod.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
app=$1; cell=$2; n=${3:-100}
cd "$(dirname "$0")/../.."
p=$($K get pods -l app=$app --no-headers | grep -v Terminating | awk '{split($2,r,"/"); if (r[1]==r[2]) print $1}' | head -1)
$K cp perf/disagg/latency_probe.py "$p:/tmp/latency_probe.py" -c inference
$K exec "$p" -c inference -- bash -c "rm -rf /logs/$cell; mkdir -p /logs/$cell; cd /tmp/lingbot && nohup python3 /tmp/latency_probe.py --presses $n --out /logs/$cell > /logs/$cell.out 2>&1 &"
until $K exec "$p" -c inference -- test -f "/logs/$cell/latency.csv" 2>/dev/null; do sleep 20; done
$K exec "$p" -c inference -- cat "/logs/$cell.out" > "perf/disagg/results/$cell.log" 2>&1
$K cp -c inference "$p:/logs/$cell/latency.csv" "perf/disagg/results/$cell.csv" >/dev/null 2>&1 || true
echo "$cell: $(grep LATENCY perf/disagg/results/$cell.log)" | tee -a perf/disagg/results/summary.txt
