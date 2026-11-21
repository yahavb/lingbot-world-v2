#!/usr/bin/env bash
# queue8.sh: press-to-photon at equal chips: co-located 8 vs the 8 + 1 split (100 presses each), plus fps.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
for f in m8e-inline lat8-split lat8-vae; do $K apply -f perf/disagg/gen/$f.json >/dev/null; done
ready1 lat8-vae; $K label pod -l app=lat8-vae pool=lat8 --overwrite >/dev/null
ready1 m8e-inline; ready1 lat8-split; sleep 30
perf/disagg/run_cell.sh m8e-inline p16_mono8_fps 60 >/dev/null 2>&1 & perf/disagg/run_cell.sh lat8-split p16_split8p1_fps 60 >/dev/null 2>&1 & wait
perf/disagg/run_lat.sh m8e-inline p16_lat_mono8_480 100 & perf/disagg/run_lat.sh lat8-split p16_lat_split8p1_480 100 & wait
grep "^p16_" perf/disagg/results/summary.txt
git add -f perf/disagg/results/summary.txt perf/disagg/results/p16_*; git commit -qm "latency at equal chips: co-located 8 vs split 8+1" -q; git pull -q --rebase fork main; git push -q fork main
$K delete deploy m8e-inline lat8-split lat8-vae --wait=false >/dev/null
echo QUEUE8 DONE
