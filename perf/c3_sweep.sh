#!/usr/bin/env bash
# c3_sweep.sh "K=V,K=V" "K=V" ...: perf/cmax3_bench.py once per variant (comma-separated env settings), one line each.
#   NEURON_RT_VISIBLE_CORES=<c> bash perf/c3_sweep.sh "LINGBOT_C3_DEPTH=2" "LINGBOT_C3_DEPTH=4"
export PYTHONPATH="${PYTHONPATH:-/work/lingbot}" NEURON_LOGICAL_NC_CONFIG="${NEURON_LOGICAL_NC_CONFIG:-2}" CMP2=0
for v in "$@"; do
  (
    IFS=',' read -r -a kv <<< "$v"
    for x in "${kv[@]}"; do [ -n "$x" ] && export "$x"; done
    r=$(python3 perf/cmax3_bench.py 2>&1 | grep -a "\[c3\] cmax3\|Error" | head -2 | tr '\n' ' ')
    echo "[c3s] ${v:-default}: $r"
  )
done
echo "[c3s] done"
