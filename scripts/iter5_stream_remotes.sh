#!/usr/bin/env bash
# Pull ALL HopperHop_phase* CSVs from every remote box → local mirror.
# Designed to be run as a Monitor — emits one summary line per pass per box.
# Cadence: every 10 min (sleep 600). Excludes checkpoints/ to keep transfers small.

set -u
LOCAL=/root/helios-rl/exp/tdmpc_glass
MIRROR=$LOCAL/remote_mirror
LOGS=$LOCAL/logs_mirror

# Per-box rsync helper: mirrors ALL HopperHop_* dirs (no per-phase include list).
sync_box() {
  local port=$1 host=$2 dest=$3
  mkdir -p "$dest"
  rsync -a -e "ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=10" \
        --include='HopperHop_*/' --include='HopperHop_*/**' \
        --exclude='**/checkpoints/**' --exclude='**/checkpoints' \
        --exclude='**/*.pkl' \
        root@$host:/root/helios-rl/exp/tdmpc_glass/ \
        "$dest/" >/dev/null 2>&1
}

# Emit one summary line listing best MPPI for every seed_*.csv directly under the box mirror.
summarize_box() {
  local label=$1 dest=$2
  local out=""
  shopt -s nullglob
  for csv in "$dest"/HopperHop_*/seed_*.csv; do
    [[ -f $csv ]] || continue
    local fname=$(basename "$csv" .csv)
    [[ "$fname" == *_v1_* || "$fname" == *_v[0-9]_* || "$fname" == *_partial_* || "$fname" == *_died_* || "$fname" == *_final_* ]] && continue
    local phase=$(basename "$(dirname "$csv")" | sed 's/HopperHop_//; s/_remote_3m//; s/_3060ti//; s/_4060//; s/_2x3060//; s/_local//; s/_ns1024/NS1024/; s/_baseline//')
    local seed=$(echo "$fname" | sed 's/seed_//')
    local best=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.0f", m}' "$csv" 2>/dev/null)
    [[ -z "$best" ]] && best="—"
    out+=" ${phase}s${seed}=${best}"
  done
  shopt -u nullglob
  [[ -z "$out" ]] && out=" (no csvs yet)"
  echo "[$(date -u +%H:%M:%S)][stream] ${label}${out}"
}

while true; do
  # Mirror all three remote boxes in parallel for speed.
  sync_box 11271 ssh3.vast.ai          $MIRROR/ssh3_3060ti       &
  sync_box 11115 ssh6.vast.ai          $MIRROR/ssh6_4060         &
  sync_box 17637 78.83.187.54          $MIRROR/ssh17637_2x3060   &
  wait

  summarize_box "ssh3_3060ti " $MIRROR/ssh3_3060ti
  summarize_box "ssh6_4060   " $MIRROR/ssh6_4060
  summarize_box "ssh17637    " $MIRROR/ssh17637_2x3060

  sleep 600
done
