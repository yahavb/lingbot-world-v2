#!/usr/bin/env bash
# prof_cell.sh APP CELL [CHUNKS]: on a pod made with mk.py --sysprof, run a session (run_cell.sh), stop rank 0 with
# SIGINT so the Neuron runtime writes its system profile, convert it to parquet and print the device timeline of the
# last ~4 chunks (perf/sysprof_gaps.py). Keeps the parquet in the pod (/logs/pq_CELL) and a tgz locally.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
app=$1; cell=$2; chunks=${3:-30}
cd "$(dirname "$0")/../.."
perf/disagg/run_cell.sh "$app" "$cell" "$chunks" >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
p=$($K get pods -l app=$app --no-headers | awk '{print $1}')
$K exec "$p" -- bash -c 'for pid in $(pgrep -f "python3 serve.py"); do grep -qz "^RANK=0$" /proc/$pid/environ 2>/dev/null && kill -INT $pid; done
  last=; for i in $(seq 1 60); do s=$(du -sk /logs/sysprof | cut -f1); [ "$s" -gt 1000 ] && [ "$s" = "$last" ] && break; last=$s; sleep 5; done'
d=$($K exec "$p" -- bash -c 'ls -td /logs/sysprof/*_pid_* | head -1')
$K cp perf/e2e/tools/neuron_trace_to_parquet.sh "$p:/logs/n2p.sh"; $K cp perf/sysprof_gaps.py "$p:/logs/sysprof_gaps.py"
$K exec "$p" -- bash -c "bash /logs/n2p.sh $d /logs/pq_$cell >/dev/null 2>&1; tar czf /logs/$cell.tgz -C /logs sysprof"
mkdir -p perf/disagg/results/prof; $K cp "$p:/logs/$cell.tgz" "perf/disagg/results/prof/$cell.tgz" >/dev/null 2>&1
$K exec "$p" -- bash -c "cd /logs && python3 sysprof_gaps.py /logs/pq_$cell/system_profile 3040 500" > "perf/disagg/results/prof/$cell.gaps.txt" 2>&1
head -12 "perf/disagg/results/prof/$cell.gaps.txt"
