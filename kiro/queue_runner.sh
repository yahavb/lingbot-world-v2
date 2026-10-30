#!/usr/bin/env bash
# Host-side job queue for the lbw container: runs /mnt/wan/kiro/lbw/queue/*.sh one at a time, in
# name order, each only after every Neuron device has been free for FREE_CHECKS (default 4, 30 s apart) consecutive checks
# (kiro/devices_free.sh), so a job never shares cores with another workload on this host.
#   setsid nohup bash kiro/queue_runner.sh </dev/null >/dev/null 2>&1 &
# A job script runs inside the container (`dk exec lbw`) from /work/queue/; its output goes to
# queue/done/<name>.out and the script moves to queue/done/. touch queue/PAUSE to hold the queue.
# Another workload can still take the devices between the last free check and the launch; then the
# job's logs say "Failed to initialize Neuron Runtime". Such a job goes back into the queue (up to
# MAX_RETRY times) and the runner waits for free devices again.
Q=/mnt/wan/kiro/lbw/queue
REPO=/mnt/wan/kiro/lbw/repo
LOGS=/mnt/wan/kiro/lbw/logs
mkdir -p "$Q/done"
echo "$(date -Is) runner start pid $$" >> "$Q/runner.log"
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
  /mnt/wan/dk exec lbw bash -l "/work/queue/done/$name.sh" > "$Q/done/$name.out" 2>&1
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
