#!/usr/bin/env bash
# Poll 2x3060 (78.83.187.54:17637) until it comes back online.
# When reachable, relaunch the two dead jobs:
#   GPU 0: Phase-x s5 (Path 9 NS=1024 compromise test)
#   GPU 1: Phase-y s4 (Path 10 default NS)
# Then exit. One-shot resume.

set -u
HOST=78.83.187.54
PORT=17637
ts() { date -u +%FT%TZ; }

echo "[watcher] $(ts) start — polling $HOST:$PORT every 60s"

while true; do
  if ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
       root@$HOST "echo ok" >/dev/null 2>&1; then
    echo "[watcher] $(ts) box ONLINE"
    break
  fi
  sleep 60
done

# Check what's currently running before relaunching
echo "[watcher] $(ts) checking existing processes..."
ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
    "ps -eo pid,etime,cmd --no-headers | grep -E 'run_benchmark' | grep -v grep" 2>/dev/null

# Relaunch Phase-x s5 (NS=1024) on GPU 0 if not running
if ! ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
        "pgrep -f 'run_benchmark.*seed 5.*mppi_n_samples 1024' >/dev/null 2>&1"; then
  echo "[watcher] $(ts) relaunching Phase-x s5 (NS=1024) on GPU 0"
  ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
      "cd /root/helios-rl
       SEED=5 NS=1024 nohup setsid bash scripts/run_phasex_seed5_ns1024.sh \
           > /tmp/phasex_s5_resume.log 2>&1 < /dev/null & disown
       sleep 3
       echo 'phasex s5 relaunched'" 2>&1 | tail -3
else
  echo "[watcher] $(ts) Phase-x s5 already running, skip"
fi

# Relaunch Phase-y s4 on GPU 1 if not running
if ! ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
        "pgrep -f 'run_benchmark.*seed 4.*glass_num_super_clusters' >/dev/null 2>&1"; then
  echo "[watcher] $(ts) relaunching Phase-y s4 on GPU 1"
  ssh -p $PORT -o StrictHostKeyChecking=no root@$HOST \
      "cd /root/helios-rl
       SEED=4 nohup setsid bash scripts/run_phasey_seed4_2x3060.sh \
           > /tmp/phasey_s4_resume.log 2>&1 < /dev/null & disown
       sleep 3
       echo 'phasey s4 relaunched'" 2>&1 | tail -3
else
  echo "[watcher] $(ts) Phase-y s4 already running, skip"
fi

echo "[watcher] $(ts) done — both jobs back up if they weren't"
