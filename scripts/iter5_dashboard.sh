#!/usr/bin/env bash
# TD-MPC-Glass live dashboard. Shows for each box in the fleet:
#   - running run_benchmark process (PID, etime, seed, NS)
#   - GPU / CPU utilization
#   - best MPPI / last eval for every active HopperHop_phase*/seed_*.csv
#
# Usage:  bash scripts/iter5_dashboard.sh
#
# Live fleet is read from /root/helios-rl/scripts/queues/*.queue names —
# auto-syncs to whatever boxes the rest of the iter5/6 tooling believes in.

set -u
RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; CYN='\033[0;36m'; BLD='\033[1m'; NC='\033[0m'
REPO=/root/helios-rl

# ─── Box registry. To add/remove a box, edit BOXES below. ────────────────
# Format: tag|port|host|gpu_idx|label
# gpu_idx is the CUDA index to query nvidia-smi for (default 0). For 2x3060 set
# 0 / 1. Empty/0 → query device 0.
BOXES=(
  "local|||0|Local 4070 Ti (12GB)"
  "ssh6_4060|11115|ssh6.vast.ai|0|ssh6:11115 4060 (8GB)"
  "ssh17637_gpu0|17637|78.83.187.54|0|78.83.187.54:17637 3060 Laptop GPU0 (6GB)"
  "ssh17637_gpu1|17637|78.83.187.54|1|78.83.187.54:17637 3060 Laptop GPU1 (6GB)"
  "ssh1_2080ti|34217|ssh1.vast.ai|0|ssh1:34217 2080 Ti (22GB)"
  "ssh3_3070|15229|ssh3.vast.ai|0|ssh3:15229 3070 (8GB)"
  "ssh6_3080|16779|ssh6.vast.ai|0|ssh6:16779 3080 (10GB)"
  "ssh3_3060ti|11271|ssh3.vast.ai|0|ssh3:11271 3060Ti (8GB)"
)

# ─── helpers run remotely via SSH. Single quoted -> sent as-is. ─────────
REMOTE_SUMMARY_SCRIPT='
gpu_idx="$1"
ps -eo pid,etime,cmd --no-headers 2>/dev/null | grep -E "run_benchmark" | grep -v grep |
  awk -v idx="$gpu_idx" '"'"'{ pid=$1; et=$2;
    for(i=3;i<=NF;i++){
      if($i=="--seed") s=$(i+1);
      if($i=="--mppi_n_samples") ns=$(i+1);
      if($i=="--algos") algo=$(i+1);
      if($i=="--knee_penalty_coef") tag="knee";
      if($i=="--use_cluster_obs") tag=tag"+cobs";
      if($i=="--glass_num_super_clusters") tag=tag"+hier";
    }
    printf "  ▶ PID=%s  etime=%s  algo=%s seed=%s NS=%s %s\n", pid, et, algo, s, (ns?ns:"?"), tag
  }'"'"'
gpu_line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader -i "$gpu_idx" 2>/dev/null | head -1)
cpu_line=$(top -bn1 2>/dev/null | grep -E "^%Cpu" | head -1 | awk '"'"'{printf "%.0f%%", 100-$8}'"'"')
echo "  GPU $gpu_line | CPU $cpu_line"
# Print best/last MPPI for every HopperHop_phase*/seed_*.csv modified in the last 2 days.
find /root/helios-rl/exp/tdmpc_glass -path "*/HopperHop_phase*/seed_*.csv" -mtime -2 -size +100c 2>/dev/null | sort | while read -r f; do
  fname=$(basename "$f" .csv)
  case "$fname" in
    *_v[0-9]_*|*_partial_*|*_died_*|*_final_*|*_done_*|*_diag) continue ;;
  esac
  pdir=$(basename "$(dirname "$f")" | sed "s/HopperHop_//")
  seed=$(echo "$fname" | sed "s/seed_//")
  best=$(awk -F, '"'"'NR>1 && $3=="mppi" {if($2+0>m){m=$2+0; ms=$1}} END{if(m>0) printf "%.1f @ %s", m, ms; else printf "—"}'"'"' "$f")
  last=$(awk -F, '"'"'NR>1 && $3=="mppi"'"'"' "$f" | tail -1 | awk -F, '"'"'{printf "step=%s MPPI=%.1f", $1, $2+0}'"'"')
  [[ -z "$last" ]] && last="(no mppi rows yet)"
  printf "  %s s%s: best %-22s   last %s\n" "$pdir" "$seed" "$best" "$last"
done
'

# fmt-green-if >=500 wrapper. Applied to lines on stdin.
hi500() {
  awk -v g="$(printf '\033[0;32m')" -v b="$(printf '\033[1m')" -v n="$(printf '\033[0m')" '
    { if (match($0, /[Bb]est [0-9]+\.[0-9]+/)) {
        v=substr($0, RSTART+5)+0
        if (v>=500) { print b g $0 n; next }
      }
      print
    }'
}

run_box() {
  local tag="$1" port="$2" host="$3" gpu_idx="$4" label="$5"
  echo -e "${CYN}── ${label} [${tag}] ──${NC}"
  if [[ "$tag" == "local" ]]; then
    echo "$REMOTE_SUMMARY_SCRIPT" | bash -s -- "$gpu_idx" 2>/dev/null | hi500
  else
    ssh -p "$port" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
        root@"$host" "bash -s -- $gpu_idx" <<<"$REMOTE_SUMMARY_SCRIPT" 2>/dev/null | hi500 \
      || echo "  (SSH timeout / box unreachable)"
  fi
}

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "    TD-MPC-Glass live dashboard  ($(date -u +%FT%TZ))"
echo "═══════════════════════════════════════════════════════════════════════════════"
for entry in "${BOXES[@]}"; do
  IFS='|' read -r tag port host gpu_idx label <<< "$entry"
  run_box "$tag" "$port" "$host" "$gpu_idx" "$label"
done
echo
echo -e "${YLW}Goals: G1 = 5/5 seeds > MPPI 500  |  G2 = break MPPI 600${NC}"
echo -e "${GRN}${BLD}Bold-green best line = exceeds 500 target.${NC}"
