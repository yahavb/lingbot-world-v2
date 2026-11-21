#!/usr/bin/env bash
# queue3.sh: LingBot at 2 denoising steps (DiT about half): DiT alone, monolith and 4 + 1 split on 4 chips;
# then streams per VAE at 8 steps (1..4 streams of 4-chip DiTs on one 1-chip VAE).
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
$K label pod -l app=nvae-1c pool=nm --overwrite >/dev/null
for f in s2null-4 s2inline-4 s2split-4; do $K apply -f perf/disagg/gen/$f.json >/dev/null; sleep 10; done
for f in s2null-4 s2inline-4 s2split-4; do ready1 $f; done; sleep 30
perf/disagg/run_cell.sh s2null-4 p13_dit4_st2_480 60 >/dev/null 2>&1 &
perf/disagg/run_cell.sh s2inline-4 p13_mono4_st2_480 60 >/dev/null 2>&1 &
perf/disagg/run_cell.sh s2split-4 p13_split4p1_st2_480 60 >/dev/null 2>&1 &
wait; grep "_st2_" perf/disagg/results/summary.txt
# 4 + 2: the same split DiT on a 2-chip VAE
$K label pod -l app=nvae-1c pool- >/dev/null; $K apply -f perf/disagg/gen/nvae-2c.json >/dev/null; ready1 nvae-2c
$K label pod -l app=nvae-2c pool=nm --overwrite >/dev/null; sleep 20
perf/disagg/run_cell.sh s2split-4 p13_split4p2_st2_480 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
git add -f perf/disagg/results/summary.txt perf/disagg/results/p13_*.csv; git commit -qm "LingBot 2 steps: DiT alone / monolith / 4+1 split (4 chips)" -q; git push -q fork main
echo QUEUE3 DONE
