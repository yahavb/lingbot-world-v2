#!/usr/bin/env bash
# demo_check.sh: wait for lbw-vae (2), lbw-dit and lbw-web to be Ready, then measure the demo's fps (40 chunks).
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
for a in lbw-vae lbw-dit lbw-web; do
  while true; do
    n=$($K get pods -l app=$a --no-headers 2>/dev/null | awk '$2 ~ /^1\/1$/ && $3 == "Running"' | wc -l | tr -d ' ')
    [ "$n" -ge 1 ] && break
    sleep 30
  done
  echo "$a ready"
done
perf/disagg/run_cell.sh lbw-dit demo_4p1_check 40 >/dev/null 2>&1
tail -1 perf/disagg/results/summary.txt
