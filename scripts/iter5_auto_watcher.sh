#!/usr/bin/env bash
# Iter5 auto-watcher: every 5 min, check each remote slot.
# If the expected process is gone (OOM/recycle/etc), relaunch it.
# Each slot has a unique cmd-pattern to detect dead-vs-alive without false matches.
#
# Slots (box → launcher → cmd-fingerprint):
#   ssh3 3060Ti GPU 0      → run_phasey_3060ti.sh                    pattern: "seed 3.*glass_num_super_clusters"
#   ssh17637 2x3060 GPU 0  → run_phasex_seed5_ns1024.sh (SEED=5)     pattern: "seed 5.*mppi_n_samples 1024"
#   ssh17637 2x3060 GPU 1  → run_phasey_seed4_2x3060.sh (SEED=4)     pattern: "seed 4.*glass_num_super_clusters"
#   ssh6 4060              → run_phasex_seed4_4060.sh (SEED=4)       pattern: "seed 4.*mppi_n_samples 2048"
#
# Retry budget: 3 relaunches per slot per watcher run (prevents infinite loops).
# Cooldown: don't relaunch if last attempt was <10 min ago.

set -u
ts() { date -u +%H:%M:%SZ; }

# Slot definitions:
#   name | port | host | grep_pattern | csv_to_backup | launcher_cmd
# Before relaunch, the watcher cp's csv_to_backup → csv_to_backup_vN_<ts>.csv on the remote.
declare -a SLOTS=(
  # ssh3_phasey_3060ti REMOVED — Phase-y queue completed; Phase-x s7 sleeper handed off the box
  "ssh3_phasex_s7|11271|ssh3.vast.ai|seed 7.*mppi_n_samples 2048|exp/tdmpc_glass/HopperHop_phasex_3060ti/seed_7.csv|nohup setsid bash scripts/sleep_then_phasex_s7_3060ti.sh > /tmp/relaunch_phasex_s7.log 2>&1 < /dev/null & disown"
  "ssh17637_phasex_s5_ns1024|17637|78.83.187.54|seed 5.*mppi_n_samples 1024|exp/tdmpc_glass/HopperHop_phasex_ns1024/seed_5.csv|SEED=5 NS=1024 nohup setsid bash scripts/run_phasex_seed5_ns1024.sh > /tmp/relaunch_phasex_s5.log 2>&1 < /dev/null & disown"
  "ssh17637_phasey_s4|17637|78.83.187.54|seed 4.*glass_num_super_clusters|exp/tdmpc_glass/HopperHop_phasey_2x3060/seed_4.csv|SEED=4 nohup setsid bash scripts/run_phasey_seed4_2x3060.sh > /tmp/relaunch_phasey_s4.log 2>&1 < /dev/null & disown"
  # ssh6_phasex_s4 removed — s4 produced stuck-seed result (peak 15.4), sleeper handles s8 now
  "ssh6_phasex_s8|11115|ssh6.vast.ai|seed 8.*mppi_n_samples 2048|exp/tdmpc_glass/HopperHop_phasex_4060/seed_8.csv|SEED=8 nohup setsid bash scripts/run_phasex_seed4_4060.sh > /tmp/relaunch_phasex_s8.log 2>&1 < /dev/null & disown"
)

declare -A retry_count last_relaunch
for entry in "${SLOTS[@]}"; do
  name=$(echo "$entry" | cut -d'|' -f1)
  retry_count[$name]=0
  last_relaunch[$name]=0
done

echo "[watcher] $(ts) start — ${#SLOTS[@]} slots, poll every 300s"

while true; do
  for entry in "${SLOTS[@]}"; do
    IFS='|' read -r name port host pattern csv launch <<< "$entry"

    # Check process via grep on cmd line (no self-match — pattern doesn't appear in `ps -eo cmd`)
    alive=$(ssh -p "$port" -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
                root@"$host" "ps -eo cmd | grep -E '$pattern' | grep -v grep | wc -l" 2>/dev/null)

    if [[ -z "$alive" ]]; then
      echo "[watcher] $(ts) $name SSH-unreachable, skip"
      continue
    fi

    if [[ "$alive" == "0" ]]; then
      # Dead. Check cooldown + retry budget.
      now=$(date +%s)
      since=$((now - last_relaunch[$name]))
      if (( since < 600 )); then
        echo "[watcher] $(ts) $name dead but in cooldown (${since}s ago), skip"
        continue
      fi
      if (( retry_count[$name] >= 3 )); then
        echo "[watcher] $(ts) $name dead but retry budget exhausted, skip"
        continue
      fi
      retry_count[$name]=$((retry_count[$name] + 1))
      last_relaunch[$name]=$now
      ts_tag=$(date -u +%Y%m%d_%H%M%SZ)
      echo "[watcher] $(ts) $name DEAD — backing up $csv → ${csv%.csv}_v${retry_count[$name]}_${ts_tag}.csv then relaunching (attempt ${retry_count[$name]}/3)"
      ssh -p "$port" -o StrictHostKeyChecking=no root@"$host" \
          "cd /root/helios-rl
           if [[ -f $csv ]]; then cp $csv ${csv%.csv}_v${retry_count[$name]}_${ts_tag}.csv; fi
           eval $launch" 2>&1 | head -3
    else
      # alive — nothing to do, silent
      :
    fi
  done
  sleep 300
done
