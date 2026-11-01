#!/usr/bin/env bash
# Host-side job queue for the lbw container: runs $Q/*.sh one at a time, in name order, each only
# after every Neuron device has been free for FREE_CHECKS (default 4, 30 s apart) consecutive checks
# (kiro/devices_free.sh), so a job never shares cores with another workload on this host.
#   setsid nohup bash kiro/queue_runner.sh </dev/null >/dev/null 2>&1 &
# A job script runs inside the container (`dk exec lbw`) from /work/<queue dir>/; its output goes to
# <queue dir>/done/<name>.out and the script moves to <queue dir>/done/. touch <queue dir>/PAUSE to hold it.
# Another workload can still take the devices between the last free check and the launch; then the
# job's logs say "Failed to initialize Neuron Runtime". Such a job goes back into the queue (up to
# MAX_RETRY times) and the runner waits for free devices again.
# Half-host queue (8-device jobs while another workload holds the other half):
#   Q=/mnt/wan/kiro/lbw/q8 DEVS=0-7 CPUS=0-47,96-143 setsid nohup bash kiro/queue_runner.sh ... &
# waits only for devices DEVS and runs the job under `taskset -c CPUS` (devices 0-7 are local to
# NUMA node 0 = CPUs 0-47,96-143 on trn2.48xlarge). Jobs in it must use only those devices' cores.
Q=${Q:-/mnt/wan/kiro/lbw/queue}
REPO=/mnt/wan/kiro/lbw/repo
LOGS=/mnt/wan/kiro/lbw/logs
QIN=/work/${Q#/mnt/wan/kiro/lbw/}      # the queue dir inside the container (/work = /mnt/wan/kiro/lbw)
PIN=()
if [ -n "${CPUS:-}" ]; then PIN=(taskset -c "$CPUS"); fi
export DEVS="${DEVS:-}"
mkdir -p "$Q/done"
echo "$(date -Is) runner start pid $$ devs=${DEVS:-all} cpus=${CPUS:-all}" >> "$Q/runner.log"
echo $$ > "$Q/runner.pid"
while true; do
  if [ -f "$Q/PAUSE" ]; then sleep 30; continue; fi
  next=$(ls "$Q"/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then sleep 20; continue; fi
  name=$(basename "$next" .sh)
  bash "$REPO/kiro/devices_free.sh" wait "${FREE_CHECKS:-4}" >> "$Q/runner.log" 2>&1
  [ -f "$next" ] || continue
  echo "$(date -Is) start $name" >> "$Q/runner.log"
  mv "$next" "$Q/done/$name.sh"
  t0=$(date +%s)
  /mnt/wan/dk exec lbw "${PIN[@]}" bash -l "$QIN/done/$name.sh" > "$Q/done/$name.out" 2>&1
  rc=$?
  echo "$(date -Is) end $name rc=$rc" >> "$Q/runner.log"
  busy=$(find "$LOGS" -type f -newermt "@$t0" -print0 2>/dev/null \
         | xargs -0 grep -l "Failed to initialize Neuron Runtime" 2>/dev/null | head -3 | tr '\n' ' ')
  if [ -n "$busy" ]; then
    n=$(cat "$Q/done/$name.retries" 2>/dev/null || echo 0)
    if [ "$n" -lt "${MAX_RETRY:-3}" ]; then
      echo $((n + 1)) > "$Q/done/$name.retries"
      echo "$(date -Is) RETRY $name ($((n + 1))): runtime init failed in $busy" >> "$Q/runner.log"
      mv "$Q/done/$name.sh" "$Q/$name.sh"
      sleep 60
    else
      echo "$(date -Is) GIVE UP $name after $n retries" >> "$Q/runner.log"
    fi
  fi
done
