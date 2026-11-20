#!/usr/bin/env bash
# split_round.sh SIZE_TAG "DIT:VAE DIT:VAE ...": one round of split cells, in parallel. Each DiT (app=s<N>-split)
# gets the VAE pod app=vae-<C>c by labelling it pool=<dit app> (every other VAE label removed first), then
# run_cell.sh measures it. Cell name p3_split_d<N>v<C>_<SIZE_TAG>.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
tag=$1; pairs=$2
cd "$(dirname "$0")/../.."
$K label pod -l 'app in (vae-1c,vae-2c,vae-4c,vae-8c,hvae-1c,hvae-2c,hvae-4c,hvae-8c)' pool- >/dev/null 2>&1   # every VAE out of every pool first
for pr in $pairs; do
  d=${pr%%:*}; v=${pr##*:}
  until $K get pods -l app=${VPFX:-vae}-${v}c --no-headers | grep -q " 1/1 "; do sleep 20; done
  $K label pod -l app=${VPFX:-vae}-${v}c pool=${PFX:-s}$d-split --overwrite >/dev/null
done
sleep 15   # the DiT's pool re-lists pods every 5 s
for pr in $pairs; do
  d=${pr%%:*}; v=${pr##*:}
  perf/disagg/run_cell.sh ${PFX:-s}$d-split p3_split_d${d}v${v}_$tag 60 > perf/disagg/results/run_s${d}v$v.out 2>&1 &
done
wait
for pr in $pairs; do
  d=${pr%%:*}; v=${pr##*:}
  used=$($K logs "$($K get pods -l app=${PFX:-s}$d-split --no-headers | awk '{print $1}')" --since=10m 2>/dev/null | grep -ao "session VAE: [a-z0-9-]*" | tail -1)
  echo "$(tail -1 perf/disagg/results/run_s${d}v$v.out)  [$used]"
done
