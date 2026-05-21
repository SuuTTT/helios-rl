#!/usr/bin/env bash
# Pull ALL HopperHop_phase* CSVs from every remote box → local mirror.
# Designed to be run as a Monitor — emits one summary line per pass per box.
# Cadence: every 10 min (sleep 600). Excludes checkpoints/ to keep transfers small.

set -u
LOCAL=/root/helios-rl/exp/tdmpc_glass
MIRROR=$LOCAL/remote_mirror
LOGS=$LOCAL/logs_mirror

# Per-box rsync helper. Hard 60s timeout + SSH keepalives so a single dead box
# can't stall the whole stream loop (we hit that bug once already).
sync_box() {
  local port=$1 host=$2 dest=$3
  mkdir -p "$dest"
  # Mirror only the HopperHop_*/ directories' CSVs (eval + diag).
  # Earlier --include rules broke when phase tags grew long; use --filter rules
  # in classic rsync syntax: P (protect) and explicit dir include.
  timeout 60 rsync -av --prune-empty-dirs \
        -e "ssh -i /home/coder/.ssh/id_ed25519 -p $port -o StrictHostKeyChecking=no \
            -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 \
            -o BatchMode=yes" \
        --include='HopperHop_*/' \
        --include='HopperHop_*/seed_*.csv' \
        --include='HopperHop_*/seed_*_diag.csv' \
        --exclude='*' \
        root@$host:/root/helios-rl/exp/tdmpc_glass/ \
        "$dest/" >/dev/null 2>&1
}

# Emit one summary line listing best MPPI for every LIVE seed_*.csv (modified in last 30 min).
# This filters out historical/done phase data and only shows actively-updated runs.
summarize_box() {
  local label=$1 dest=$2
  local out=""
  shopt -s nullglob
  # Two-tier filter:
  # 1. CSV mtime newer than the last fully-completed phase (older than 7 days = "old archived")
  # 2. CSV has at least one eval row (size > 100 bytes; bare header is ~27)
  # 3. Phase prefix is in "active" allowlist (current iter 5-6 phases)
  for csv in $(find "$dest" -path "*/HopperHop_*/seed_*.csv" -mtime -2 -size +30c 2>/dev/null | sort); do
    [[ -f $csv ]] || continue
    local fname=$(basename "$csv" .csv)
    # Skip backup snapshots and diagnostic sidecars
    [[ "$fname" == *_v1_* || "$fname" == *_v[0-9]_* || "$fname" == *_partial_* || "$fname" == *_died_* || "$fname" == *_final_* || "$fname" == *_done_* || "$fname" == *_diag ]] && continue
    local pdir=$(basename "$(dirname "$csv")")
    # active-phase allowlist for iter 5-6
    case "$pdir" in
      HopperHop_phasex_*|HopperHop_phasey_*|HopperHop_phaseq_*|HopperHop_phasez_*|HopperHop_phasev_*|HopperHop_phasex_ns1024) ;;
      *) continue;;
    esac
    local phase=$(echo "$pdir" | sed 's/HopperHop_//; s/_remote_3m//; s/_3060ti//; s/_4060//; s/_2x3060//; s/_local//; s/_ns1024/_NS1024/; s/_baseline//; s/_knee//')
    local seed=$(echo "$fname" | sed 's/seed_//')
    local best=$(awk -F, 'NR>1 && $3=="mppi" {if($2+0>m)m=$2+0} END{printf "%.0f", m}' "$csv" 2>/dev/null)
    [[ -z "$best" ]] && best="—"
    out+=" ${phase}s${seed}=${best}"
  done
  shopt -u nullglob
  [[ -z "$out" ]] && out=" (no active csvs)"
  echo "[$(date -u +%H:%M:%S)][stream] ${label}${out}"
}

while true; do
  # Mirror all remote boxes in parallel for speed.
  # sync_box 11271 ssh3.vast.ai (KILLED)          $MIRROR/ssh3_3060ti       &
  sync_box 11115 ssh6.vast.ai          $MIRROR/ssh6_4060         &
  sync_box 17637 78.83.187.54          $MIRROR/ssh17637_2x3060   &
  sync_box 34217 ssh1.vast.ai          $MIRROR/ssh1_2080ti       &
  sync_box 15229 ssh3.vast.ai          $MIRROR/ssh3_3070         &
  sync_box 16779 ssh6.vast.ai          $MIRROR/ssh6_3080         &
  sync_box 11271 ssh3.vast.ai          $MIRROR/ssh3_3060ti_new   &
  wait

  # summarize_box "ssh3_3060ti " $MIRROR/ssh3_3060ti (KILLED)
  summarize_box "ssh6_4060   " $MIRROR/ssh6_4060
  summarize_box "ssh17637    " $MIRROR/ssh17637_2x3060
  summarize_box "ssh1_2080ti " $MIRROR/ssh1_2080ti
  summarize_box "ssh3_3070   " $MIRROR/ssh3_3070
  summarize_box "ssh6_3080   " $MIRROR/ssh6_3080
  summarize_box "ssh3_3060ti " $MIRROR/ssh3_3060ti_new

  sleep 300
done
