#!/usr/bin/env bash
# queue5.sh: 1) elasticity: 3 streams on one VAE, then a second VAE is added at run time and streams 4-6 arrive;
# 2) placement: a 4 + 1 split with the VAE on the DiT's node vs on the other node (fps, press-to-photon).
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do sleep 20; done; }
$K delete deploy s2null-4 s2inline-4 s2split-4 nvae-2c --wait=true >/dev/null 2>&1
# 1) elasticity
$K label pod -l app=nvae-1c pool=nm --overwrite >/dev/null
for x in a b c d e f; do $K apply -f perf/disagg/gen/n2$x-split.json >/dev/null; sleep 5; done
for x in a b c d e f; do ready1 n2$x-split; done; sleep 30
t0=$(date +%s)
for x in a b c; do perf/disagg/run_cell.sh n2$x-split el_$x 240 >/dev/null 2>&1 & sleep 10; done
sleep 30; echo "t=$(( $(date +%s) - t0 )) s: scaling the decoder pool to 2"
$K apply -f perf/disagg/gen/nvae2-1c.json >/dev/null; ready1 nvae2-1c; $K label pod -l app=nvae2-1c pool=nm --overwrite >/dev/null
echo "t=$(( $(date +%s) - t0 )) s: second VAE ready (cold start from apply)"
for x in d e f; do perf/disagg/run_cell.sh n2$x-split el_$x 60 >/dev/null 2>&1 & sleep 10; done
wait; grep "^el_" perf/disagg/results/summary.txt | tail -6
for x in a b c d e f; do echo "n2$x: $($K logs $($K get pods -l app=n2$x-split --no-headers | awk '{print $1}') --since=40m | grep -ao 'session VAE: [a-z0-9-]*' | tail -1)"; done
git add -f perf/disagg/results/summary.txt perf/disagg/results/el_*.csv; git commit -qm "elasticity: decoder pool 1 -> 2 under load" -q; git push -q fork main
# 2) placement
$K delete deploy n2a-split n2b-split n2c-split n2d-split n2e-split n2f-split nvae2-1c --wait=true >/dev/null
for f in pl-vae-same pl-vae-far pl-dit; do $K apply -f perf/disagg/gen/$f.json >/dev/null; done
ready1 pl-vae-same; ready1 pl-vae-far; ready1 pl-dit
for where in same far; do
  other=$([ $where = same ] && echo far || echo same)
  $K label pod -l app=pl-vae-$other pool- >/dev/null 2>&1; $K label pod -l app=pl-vae-$where pool=pl --overwrite >/dev/null; sleep 20
  perf/disagg/run_cell.sh pl-dit pl_${where}_fps 60 >/dev/null 2>&1; tail -1 perf/disagg/results/summary.txt
  perf/disagg/run_lat.sh pl-dit pl_${where}_lat 60
done
git add -f perf/disagg/results/summary.txt perf/disagg/results/pl_*.csv; git commit -qm "placement: VAE on the DiT's node vs the other node" -q; git push -q fork main
echo QUEUE5 DONE
