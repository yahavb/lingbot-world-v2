#!/usr/bin/env bash
# nm_lingbot.sh: LingBot N:M on ONE 1-chip VAE (nvae-1c, LINGBOT_VAE_SESSIONS=4). 1) exactness: a session's frames
# alone vs next to another session (md5 of chunks 10 and 20); 2) N = 1, 2, 3 concurrent 2-chip DiT streams and N = 1,
# 2 concurrent 4-chip DiT streams: per-stream fps (run_cell.sh) and the VAE's [SESSIONS] memory lines.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
until $K get pods -l app=nvae-1c --no-headers | grep -q " 1/1 "; do sleep 20; done
$K label pod -l app=nvae-1c pool=nm --overwrite >/dev/null; sleep 15
pod() { $K get pods -l app=$1 --no-headers | awk '{print $1}'; }
md5s() { for c in 0010 0020; do $K exec $(pod $1) -- md5sum /logs/$2/chunk_$c.jpg 2>/dev/null | cut -c1-12; done | tr '\n' ' '; }
run() { perf/disagg/run_cell.sh $1 $2 ${3:-60} --every 10 >/dev/null 2>&1; }
# 1) exactness
run n2a-split nm_alone_n2a 30
run n2a-split nm_pair_n2a 30 & run n2b-split nm_pair_n2b 30 & wait
echo "EXACT alone [$(md5s n2a-split nm_alone_n2a)] with-another [$(md5s n2a-split nm_pair_n2a)]"
# 2) N:M
for n in 1 2 3; do
  for x in $(echo a b c | cut -d' ' -f1-$n); do run n2$x-split nm2_N${n}_$x & done; wait
done
for n in 1 2; do
  for x in $(echo a b | cut -d' ' -f1-$n); do run n4$x-split nm4_N${n}_$x & done; wait
done
grep -E "^nm" perf/disagg/results/summary.txt | tail -12
$K logs $(pod nvae-1c) | grep -a "\[SESSIONS\]" | tail -4 | cut -c1-220
