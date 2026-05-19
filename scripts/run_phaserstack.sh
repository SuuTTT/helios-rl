#!/usr/bin/env bash
# Phase-r-stack (iter 6 §7.E): the headline experiment.
# Stacks every intervention that's shown ≥1 G1 winner in iter 1-5:
#   • TD-MPC-Glass on (prototype clustering)
#   • NS=2048 MPPI (Path 9)
#   • EXPL_UNTIL=500k (Path 1)
#   • Latent-action smoothing curriculum (Phase-j)
#   • Soft-reward bundle (iter-6 §7.B Phase-r1)  — stand_bonus + anneal
#   • Gait penalty bundle (iter-6 §7.C Phase-r2) — fall_penalty + action_smooth
# Goal: does stacking push through G2 = break MPPI 600?
# Default 5 seeds (per §7.F). Eval reward is the original task.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaserstack
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaserstack
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3 4 5"}
FALL_COEF=${FALL_COEF:-0.1}
FALL_H=${FALL_H:-0.45}
SMOOTH=${SMOOTH:-0.005}
STAND_BONUS=${STAND_BONUS:-0.1}
STAND_FLOOR=${STAND_FLOOR:-0.4}
ANNEAL=${ANNEAL:-3000000}

echo "[r-stack] start $(date -u +%FT%TZ) — TD-MPC-Glass + NS=2048 + soft + gait, seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[r-stack] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --soft_stand_bonus "$STAND_BONUS" \
    --soft_stand_floor "$STAND_FLOOR" \
    --soft_anneal_steps "$ANNEAL" \
    --gait_fall_penalty "$FALL_COEF" \
    --gait_fall_height "$FALL_H" \
    --gait_action_smooth "$SMOOTH" \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[r-stack] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[r-stack] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
