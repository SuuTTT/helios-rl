#!/usr/bin/env bash
# Template launcher for gpu-fleet task queue.
#
# The queue daemon calls this script as:
#   <ENV_VARS> nohup setsid bash <launcher> > /tmp/fleet_<task_id>.log 2>&1 < /dev/null & disown
#
# Key patterns to follow:
#   1. Use set -u (fail on undefined vars) + set +e (don't abort on experiment error).
#   2. Source the project venv and set PYTHONPATH before running experiments.
#   3. Support SEEDS env var to run one seed per task (avoids watcher clobbering).
#   4. Use tee -a (append) not tee to preserve logs across restarts.
#   5. Set TDMPC_GLASS_OUTPUT_TAG (or your own TAG var) for output isolation.
#   6. Print a clear start/done line with timestamps so the log is easy to scan.

set -u
set +e

# ── Environment ───────────────────────────────────────────────────────────────
REPO=${REPO:-/root/my-repo}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}

# ── Experiment config ─────────────────────────────────────────────────────────
# Accept seed(s) from task env — queue one task per seed.
SEEDS=${SEEDS:-1}
# Accept any hyperparam from task env.
LR=${LR:-3e-4}
TAG=${TAG:-"myexperiment_lr${LR}"}

LOG_DIR=$REPO/exp/$TAG
mkdir -p "$LOG_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[launcher] start $(date -u +%FT%TZ) tag=$TAG seeds=$SEEDS lr=$LR" \
  | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/seed_${seed}.log"
  echo "[launcher] === seed=$seed start $(date -u +%FT%TZ) ===" | tee -a "$log"

  python3 -u scripts/train.py \
    --seed "$seed" \
    --lr   "$LR" \
    --output_dir "$LOG_DIR" \
    --no_plot \
    2>&1 | tee -a "$log"

  echo "[launcher] === seed=$seed done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log"
done

echo "[launcher] all seeds done $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
