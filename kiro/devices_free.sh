#!/usr/bin/env bash
# Exit 0 if no process (in any container) holds a Neuron device, else print the holders, exit 1.
# Run on the HOST (neuron-ls on the host sees every container's processes). Parses
# `neuron-ls --json-output` (neuron_processes per device): the text table changes its columns
# with the processes present, so parsing it is not reliable.
#   bash kiro/devices_free.sh            check once
#   bash kiro/devices_free.sh wait N     poll every 30 s until free for N consecutive checks
set -u
check() {
  timeout 60 neuron-ls --json-output 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("neuron-ls unreadable:", e); sys.exit(1)
busy = []
for dev in d:
    for p in dev.get("neuron_processes") or []:
        pid = p.get("pid") if isinstance(p, dict) else p
        busy.append("dev%s pid%s" % (dev.get("neuron_device"), pid))
if busy:
    print(";".join(busy)); sys.exit(1)
if len(d) != 16:
    print(f"only {len(d)} devices"); sys.exit(1)
'
}
if [ "${1:-}" = "wait" ]; then
  need=${2:-2}; ok=0
  while [ "$ok" -lt "$need" ]; do
    if check > /tmp/devices_busy.txt; then ok=$((ok + 1)); else ok=0; fi
    [ "$ok" -lt "$need" ] && sleep 30
  done
  echo "free x$need at $(date -Is)"
  exit 0
fi
check
