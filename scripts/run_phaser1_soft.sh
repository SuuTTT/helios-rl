#!/usr/bin/env bash
# Phase-r1 (iter 6 §7.B v1): vanilla TD-MPC2 + SOFT-REWARD bundle — 3 seeds default.
# v1 implements the cheap two of §7.B's four components:
#   • soft_stand_bonus = 0.1 * clip((h - 0.4)/0.2, 0, 1)  -- smooth height ramp
#   • shaping_anneal: linear fade weight 1.0 -> 0.0 over [0, 3M] env steps
# Skipped from v1 (deferred to v2): speed_curriculum, last-200-step early bonus.
# Eval reward is unmodified — we measure against the original task.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaser1_soft
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaser1_soft
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3"}
STAND_BONUS=${STAND_BONUS:-0.1}
STAND_FLOOR=${STAND_FLOOR:-0.4}
ANNEAL=${ANNEAL:-3000000}

echo "[phaser1] start $(date -u +%FT%TZ) — TD-MPC2 + soft-reward bundle v1 (stand_bonus=$STAND_BONUS floor=$STAND_FLOOR anneal=$ANNEAL) seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaser1] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --soft_stand_bonus "$STAND_BONUS" \
    --soft_stand_floor "$STAND_FLOOR" \
    --soft_anneal_steps "$ANNEAL" \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaser1] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phaser1] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
