#!/usr/bin/env bash
# Phase-r2 (iter 6 §7.C): vanilla TD-MPC2 + GAIT PENALTY bundle — 3 seeds default.
# Adds two training-only penalties:
#   • fall_penalty = -0.1   when torso-foot height < 0.45 m
#   • action_smooth = -0.005 * mean((a_t - a_{t-1})**2) per env
# Eval reward is unmodified — we measure against the original task.
# Same base config as Phase-q (NS=2048, EXPL_UNTIL=500k, smoothing curriculum).

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaser2_gait
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaser2_gait
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3"}
FALL_COEF=${FALL_COEF:-0.1}
FALL_H=${FALL_H:-0.45}
SMOOTH=${SMOOTH:-0.005}

echo "[phaser2] start $(date -u +%FT%TZ) — TD-MPC2 + gait penalty (fall=$FALL_COEF h<$FALL_H smooth=$SMOOTH) seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaser2] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --gait_fall_penalty "$FALL_COEF" \
    --gait_fall_height "$FALL_H" \
    --gait_action_smooth "$SMOOTH" \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaser2] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phaser2] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
