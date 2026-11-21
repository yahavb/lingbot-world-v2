#!/usr/bin/env bash
# queue1.sh: the next experiments back to back. 1) 9 streams x 3 VAEs, per-session least-loaded placement;
# 2) in-pod partitioned baseline c4p1 (4 DiT + 1 VAE chips in one pod): fps; 3) press-to-photon latency of the 4-chip
# monolith, the 4 + 1 split and c4p1; 4) c8p1 fps.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { # exactly one pod for the app, not terminating, all containers ready
  until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
# 2) + 3)
$K delete deploy c4p1 --wait=true >/dev/null 2>&1; $K apply -f perf/disagg/gen/c4p1.json >/dev/null
python3 perf/disagg/mk.py dit m4e-inline 4 inline --env LINGBOT_EMIT_ASYNC=1 >/dev/null
python3 perf/disagg/mk.py dit n4a-split 4 split --env VAE_SELECTOR=pool=nm --env LINGBOT_EMIT_ASYNC=1 >/dev/null
$K apply -f perf/disagg/gen/m4e-inline.json >/dev/null; $K apply -f perf/disagg/gen/n4a-split.json >/dev/null
ready1 c4p1; ready1 m4e-inline; ready1 n4a-split
perf/disagg/run_cell.sh c4p1 p11_combo4p1_480 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
perf/disagg/run_lat.sh m4e-inline p12_lat_mono4_480 100
perf/disagg/run_lat.sh n4a-split p12_lat_split4p1_480 100
perf/disagg/run_lat.sh c4p1 p12_lat_combo4p1_480 100
git add -f perf/disagg/results/summary.txt perf/disagg/results/p11_*.csv perf/disagg/results/p12_*.csv; git commit -qm "in-pod partitioned baseline 4+1 fps; press-to-photon latency: monolith 4, split 4+1, combo 4+1" -q; git push -q fork main
# 4)
$K delete deploy m4e-inline n4a-split n2a-split --wait=true >/dev/null
$K apply -f perf/disagg/gen/c8p1.json >/dev/null; ready1 c8p1
perf/disagg/run_cell.sh c8p1 p11_combo8p1_480 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
git add -f perf/disagg/results/summary.txt perf/disagg/results/p11_*.csv; git commit -qm "in-pod partitioned baseline 8+1 fps" -q; git push -q fork main
echo QUEUE1 DONE
