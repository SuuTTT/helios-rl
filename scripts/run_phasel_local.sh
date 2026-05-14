#!/usr/bin/env bash
# Phase-l on the LOCAL 4070 Ti — the Glass ablation.
#
# Question: did Glass contribute anything beyond what smoothing alone provides?
# Phase-f's seed-1 surge (438 → 571) might have been entirely from smoothing,
# with Glass just adding compute. This run uses `--algos tdmpc2` (NO Glass loss,
# NO prototypes, NO assign_logits) plus `--latent_action_smooth_coef 1e-3`.
#
# Possible outcomes:
#   1. Phase-l seed 1 surges past 500 → Glass was irrelevant. Drop it.
#   2. Phase-l seed 1 falls back to Phase 1b's stuck pattern → Glass IS needed.
#   3. Phase-l does something different (e.g. all seeds at 400) → tells us
#      how Glass biases basin distribution.
#
# 10M cap + 1.5M early-stop.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasel
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasel
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2"}
SMOOTH=${SMOOTH:-0.001}
PATIENCE=${PATIENCE:-1500000}

echo "[phasel] start $(date -u +%FT%TZ) seeds=[$SEEDS] smooth=$SMOOTH patience=$PATIENCE" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasel] config: plain TD-MPC2 (--algos tdmpc2, NO Glass) + latent_action_smooth_coef=$SMOOTH" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasel] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  # Uses tdmpc-glass code path so CSV/checkpoint machinery is intact,
  # but ALL Glass loss weights are 0 → no SE pressure, no balance, no
  # temporal coherence loss. Glass prototypes/assign_logits still exist
  # in params but receive zero gradient. Effectively: TD-MPC2 + smoothing.
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass \
    --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_lambda_se 0.0 \
    --glass_lambda_balance 0.0 \
    --glass_lambda_temporal 0.0 \
    --latent_action_smooth_coef "$SMOOTH" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasel] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasel] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
