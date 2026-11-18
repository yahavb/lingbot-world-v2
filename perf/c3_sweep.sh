#!/usr/bin/env bash
# c3_sweep.sh "K=V,K=V" "K=V" ...: perf/cmax3_bench.py once per variant (comma-separated env settings), one line each.
#   NEURON_RT_VISIBLE_CORES=<c> bash perf/c3_sweep.sh "LINGBOT_C3_DEPTH=2" "LINGBOT_C3_DEPTH=4"
# NKI_ENABLE_TRACE_CACHE=0: the persistent NKI trace cache keys on the kernel arguments, not on the LINGBOT_C3_*
# knobs read at import, so without it every variant after the first can reuse the first one's compiled kernel
export PYTHONPATH="${PYTHONPATH:-/work/lingbot}" NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}" CMP2=0 NKI_ENABLE_TRACE_CACHE=0
for v in "$@"; do
  (
    IFS=',' read -r -a kv <<< "$v"
    for x in "${kv[@]}"; do [ -n "$x" ] && export "$x"; done
    r=$(python3 perf/cmax3_bench.py 2>&1 | grep -a "\[c3\] cmax3\|Error" | head -2 | tr '\n' ' ')
    echo "[c3s] ${v:-default}: $r"
  )
done
echo "[c3s] done"
