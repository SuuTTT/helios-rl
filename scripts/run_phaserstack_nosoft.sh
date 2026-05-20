#!/usr/bin/env bash
# r-stack ablation B: drop soft_stand_bonus (keep everything else, including
# gait_fall_penalty + action_smooth).
# Tests hypothesis: soft+gait both pulling on height creates a conflicting
# standing-vs-falling double-bind that confuses the policy. 3 seeds default.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaserstack_nosoft
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaserstack_nosoft
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3"}

echo "[r-stack-nosoft] start $(date -u +%FT%TZ) — Glass + NS=2048 + gait (NO soft_bonus), seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[r-stack-nosoft] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --gait_fall_penalty 0.1 \
    --gait_fall_height 0.45 \
    --gait_action_smooth 0.005 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[r-stack-nosoft] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
