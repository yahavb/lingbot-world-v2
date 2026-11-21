#!/usr/bin/env bash
# queue2.sh: 1) in-pod 8 + 1 (one claim per container) fps; 2) press-to-photon of the 4 + 4 split (bigger VAE: lower decode
# latency?) vs the 4-chip monolith (p12_lat_mono4_480) and 4 + 1 (p12_lat_split4p1_480).
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
$K delete deploy c8p1 c4p1 --wait=true >/dev/null 2>&1
$K apply -f perf/disagg/gen/c8p1.json >/dev/null
$K label pod -l app=nvae-1c pool- >/dev/null 2>&1
$K apply -f perf/disagg/gen/nvae-4c.json >/dev/null; $K apply -f perf/disagg/gen/n4a-split.json >/dev/null
ready1 nvae-4c; $K label pod -l app=nvae-4c pool=nm --overwrite >/dev/null
ready1 c8p1; perf/disagg/run_cell.sh c8p1 p11_combo8p1_480 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
ready1 n4a-split; sleep 30
perf/disagg/run_cell.sh n4a-split p11_split4p4_480 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
perf/disagg/run_lat.sh n4a-split p12_lat_split4p4_480 100
git add -f perf/disagg/results/summary.txt perf/disagg/results/p11_*.csv perf/disagg/results/p12_*.csv perf/disagg/results/p12_*.log; git commit -qm "in-pod 8+1 fps; 4+4 split fps and press-to-photon" -q; git push -q fork main
echo QUEUE2 DONE
