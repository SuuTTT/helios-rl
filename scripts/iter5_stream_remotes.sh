#!/usr/bin/env bash
# Pull CSVs + tail-of-logs from all remote boxes into local mirrors.
# Designed to be run as a Monitor — emits one summary line per pass per box.
# Sync cadence: every 2 minutes.

set -u
LOCAL=/root/helios-rl/exp/tdmpc_glass
MIRROR=$LOCAL/remote_mirror
LOGS=$LOCAL/logs_mirror

while true; do
  ts=$(date -u +%H:%M:%S)

  # === 3060Ti (ssh3:11271) ===
  rsync -a -e "ssh -p 11271 -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        --exclude='**/checkpoints/**' --exclude='**/checkpoints' \
        root@ssh3.vast.ai:/root/helios-rl/exp/tdmpc_glass/HopperHop_phasey_3060ti/ \
        $MIRROR/ssh3_3060ti/HopperHop_phasey_3060ti/ >/dev/null 2>&1
  # also grab the last lines of seed_1 log
  ssh -p 11271 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh3.vast.ai \
      "tail -30 /root/helios-rl/exp/tdmpc_glass/logs/phasey_3060ti/HopperHop_seed_*.log 2>/dev/null" \
      > $LOGS/ssh3_3060ti/phasey_tail.log 2>/dev/null
  best=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.1f", m}' \
         $MIRROR/ssh3_3060ti/HopperHop_phasey_3060ti/seed_1.csv 2>/dev/null)
  echo "[$ts][stream] ssh3_3060ti phasey s1 best_MPPI=${best:-—}"

  # === 4060 (ssh6:11115) ===
  rsync -a -e "ssh -p 11115 -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        --include='HopperHop_phasev_4060/' --include='HopperHop_phasev_4060/**/*.csv' --include='HopperHop_phasev_4060/**' \
        --include='HopperHop_phasep_remote_3m/' --include='HopperHop_phasep_remote_3m/**/*.csv' --include='HopperHop_phasep_remote_3m/**' \
        --exclude='**/checkpoints/**' --exclude='**/checkpoints' \
        root@ssh6.vast.ai:/root/helios-rl/exp/tdmpc_glass/ \
        $MIRROR/ssh6_4060/ >/dev/null 2>&1
  ssh -p 11115 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@ssh6.vast.ai \
      "tail -30 /root/helios-rl/exp/tdmpc_glass/logs/phasev_4060/HopperHop_seed_*.log 2>/dev/null" \
      > $LOGS/ssh6_4060/phasev_tail.log 2>/dev/null
  bestp=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.1f", m}' \
          $MIRROR/ssh6_4060/HopperHop_phasep_remote_3m/seed_6.csv 2>/dev/null)
  bestv=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.1f", m}' \
          $MIRROR/ssh6_4060/HopperHop_phasev_4060/seed_3.csv 2>/dev/null)
  echo "[$ts][stream] ssh6_4060   phasep s6 best=${bestp:-—}  phasev s3 best=${bestv:-—}"

  # === 2x3060 (78.83.187.54:17637) ===
  rsync -a -e "ssh -p 17637 -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        --include='HopperHop_phasex_2x3060/' --include='HopperHop_phasex_2x3060/**/*.csv' --include='HopperHop_phasex_2x3060/**' \
        --include='HopperHop_phasev_2x3060/' --include='HopperHop_phasev_2x3060/**/*.csv' --include='HopperHop_phasev_2x3060/**' \
        --exclude='**/checkpoints/**' --exclude='**/checkpoints' \
        root@78.83.187.54:/root/helios-rl/exp/tdmpc_glass/ \
        $MIRROR/ssh17637_2x3060/ >/dev/null 2>&1
  ssh -p 17637 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@78.83.187.54 \
      "tail -30 /root/helios-rl/exp/tdmpc_glass/logs/phasex_2x3060/HopperHop_seed_*.log 2>/dev/null" \
      > $LOGS/ssh17637_2x3060/phasex_tail.log 2>/dev/null
  bestx1=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.1f", m}' \
           $MIRROR/ssh17637_2x3060/HopperHop_phasex_2x3060/seed_1.csv 2>/dev/null)
  bestx2=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.1f", m}' \
           $MIRROR/ssh17637_2x3060/HopperHop_phasex_2x3060/seed_2.csv 2>/dev/null)
  echo "[$ts][stream] ssh17637    phasex s1 best=${bestx1:-—}  phasex s2 best=${bestx2:-—}"

  sleep 120
done
