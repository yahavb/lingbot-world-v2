#!/usr/bin/env bash
# vaeprof.sh VAE_APP CELL WS [CACHE]: the VAE side of devprof.sh. On a VAE pod made with mk.py vae --sysprof that
# has decoded at least one session: stop vae_serve.py rank 0 (the container stays), convert its system trace,
# attribute device time per NEFF, capture NEFF+NTFF pairs of the top graphs (WS ranks for collectives) and print
# per pair: ms, MFU, engine busy.
set -uo pipefail
K="kubectl --context yahavb-Isengard@KEEP-trn3-dev1.us-east-2.eksctl.io"
app=$1; cell=$2; ws=$3; cache=${4:-5}
cd "$(dirname "$0")/../.."
p=$($K get pods -l app=$app --no-headers | awk '{print $1}' | head -1)
$K exec "$p" -c vae -- bash -c 'for pid in $(pgrep -f "python3 vae_serve.py"); do grep -qz "^RANK=0$" /proc/$pid/environ 2>/dev/null && kill -INT $pid; done'
until $K logs "$p" -c vae --tail=5 2>/dev/null | grep -aq "SERVER STOPPED"; do sleep 10; done
sleep 20
$K exec "$p" -c vae -- bash -c "
  set -u; cd /tmp/lingbot; T=perf/e2e/tools; d=\$(ls -td /logs/sysprof/*_pid_* | head -1)
  C=/node-cache/cache/v$cache
  bash \$T/neuron_trace_to_parquet.sh \$d /logs/pq_$cell >/dev/null 2>&1
  pip install -q duckdb pyarrow 2>/dev/null
  python3 \$T/neuron_attribution.py /logs/pq_$cell/system_profile/SystemProfileEvents.parquet --min-share 0.02 \
     --roles 'vae=.' --json-out /logs/man_$cell.json | head -30
  ROLES='vae=.' CAPTURE_TIMEOUT=1800 bash \$T/neuron_pairs.sh /logs/man_$cell.json \$C/neff_cache /logs/pairs_$cell $ws > /logs/pairs_$cell.log 2>&1
  tail -15 /logs/pairs_$cell.log
  PROF=\$(command -v neuron-explorer || command -v neuron-profile)
  for q in /logs/pairs_$cell/*/; do
    n=\$(ls \$q/*.neff 2>/dev/null | head -1); t=\$(ls \$q/*.ntff 2>/dev/null | head -1); [ -n \"\$n\" ] && [ -n \"\$t\" ] || continue
    timeout 900 \$PROF view -n \$n -s \$t --output-format summary-json > \$q/summary.json 2>/dev/null
    python3 -c 'import json,os,sys; p=sys.argv[1]; d=json.load(open(p+\"/summary.json\")); s=d[list(d)[0]]; print(\"%-44s %8.3f ms  mfu %5.1f%%  tensor %3.0f%%  vector %3.0f%%  dma %3.0f%%\" % (os.path.basename(p.rstrip(\"/\")), s[\"total_time\"]*1e3, 100*s.get(\"mfu_estimated_percent\",0), 100*s.get(\"tensor_engine_active_time_percent\",0), 100*s.get(\"vector_engine_active_time_percent\",0), 100*s.get(\"dma_active_time_percent\",0)))' \$q
  done | tee /logs/pairs_$cell/summary.txt
" 2>&1 | tee "perf/disagg/results/prof/$cell.vaeprof.txt"
