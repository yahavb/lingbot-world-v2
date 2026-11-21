#!/usr/bin/env bash
# queue7.sh: 9 LingBot streams (2-chip DiTs) on 3 one-chip VAEs, per-session least-loaded placement: all 9 recorded
# (the probe runs detached in each pod). Results: perf/disagg/results/hs3_9x3_*.csv + summary.txt.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
for v in nvae-1c nvae2-1c nvae3-1c; do python3 perf/disagg/mk.py vae $v 1 $v --env LINGBOT_VAE_SESSIONS=4 >/dev/null; $K apply -f perf/disagg/gen/$v.json >/dev/null; done
for x in a b c d e f g h i; do python3 perf/disagg/mk.py dit n2$x-split 2 split --env VAE_SELECTOR=pool=nm --env LINGBOT_EMIT_ASYNC=1 --env LINGBOT_VAE_LEAST_LOADED=1 >/dev/null; $K apply -f perf/disagg/gen/n2$x-split.json >/dev/null; sleep 5; done
for v in nvae-1c nvae2-1c nvae3-1c; do ready1 $v; $K label pod -l app=$v pool=nm --overwrite >/dev/null; done
for x in a b c d e f g h i; do ready1 n2$x-split; done
sleep 60
for x in a b c d e f g h i; do perf/disagg/run_cell.sh n2$x-split hs3_9x3_$x 60 >/dev/null 2>&1 & sleep 10; done; wait
grep "^hs3_9x3" perf/disagg/results/summary.txt
for x in a b c d e f g h i; do echo "n2$x: $($K logs $($K get pods -l app=n2$x-split --no-headers | awk '{print $1}') --since=15m | grep -ao 'session VAE: [a-z0-9-]*' | tail -1)"; done | tee perf/disagg/results/hs3_9x3_placement.txt
git add -f perf/disagg/results/summary.txt perf/disagg/results/hs3_*; git commit -qm "9 streams x 3 VAEs rerun (all streams recorded, placement logged)" -q; git push -q fork main
for x in a b c d e f g h i; do $K delete deploy n2$x-split --wait=false >/dev/null; done; $K delete deploy nvae-1c nvae2-1c nvae3-1c --wait=false >/dev/null
echo QUEUE7 DONE
