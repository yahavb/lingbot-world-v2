#!/usr/bin/env bash
# queue6.sh: MX activation layout change on the tensor engine (LINGBOT_MX_PE_SWIZZLE=1) vs the DMA queues (default),
# 4-chip monolith, 480x768: fps and DiT ms per chunk, 2 runs each, and byte-compare frames (a transpose is exact).
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
cd "$(dirname "$0")/../.."
ready1() { until [ "$($K get pods -l app=$1 --no-headers 2>/dev/null | wc -l | tr -d ' ')" = 1 ] && \
        $K get pods -l app=$1 --no-headers | awk '{split($2,r,"/"); exit !(r[1]==r[2] && $3=="Running")}'; do
        q=$($K get pods -l app=$1 --no-headers 2>/dev/null | awk '{print $1}' | head -1)
        [ -n "$q" ] && [ "$($K get pod $q -o jsonpath='{.status.containerStatuses[0].restartCount}' 2>/dev/null)" -ge 2 ] && { echo "$1 CRASHLOOP"; $K logs $q --previous 2>&1 | grep -aE "Error|error" | tail -5; return 1; }
        sleep 20; done; }
for f in m4e-inline pe4-inline; do $K apply -f perf/disagg/gen/$f.json >/dev/null; done
ready1 m4e-inline && ready1 pe4-inline || exit 1
for r in 1 2; do
  perf/disagg/run_cell.sh m4e-inline p15_dma_r${r}_480 60 --every 10 >/dev/null 2>&1 &
  perf/disagg/run_cell.sh pe4-inline p15_pe_r${r}_480 60 --every 10 >/dev/null 2>&1 &
  wait
done
grep "^p15_" perf/disagg/results/summary.txt
a=$($K get pods -l app=m4e-inline --no-headers | awk '{print $1}'); b=$($K get pods -l app=pe4-inline --no-headers | awk '{print $1}')
for c in 0010 0030 0050; do echo "chunk $c: $($K exec $a -- md5sum /logs/p15_dma_r2_480/chunk_$c.jpg | cut -c1-12) vs $($K exec $b -- md5sum /logs/p15_pe_r2_480/chunk_$c.jpg | cut -c1-12)"; done
git add -f perf/disagg/results/summary.txt perf/disagg/results/p15_*.csv; git commit -qm "MX transpose on PE vs DMA: 4-chip monolith A/B" -q; git push -q fork main
echo QUEUE6 DONE
