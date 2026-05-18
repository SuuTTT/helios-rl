#!/usr/bin/env bash
# Phase-q (iter 6 Q2): vanilla TD-MPC2 + KNEE PENALTY — 5 seeds
# Tests the physical ceiling for HopperHop. Phase-t s2 hit 612 with knee
# penalty + Glass; this strips Glass to isolate the knee-penalty effect.
# Reward-shaped (NOT benchmark-fair) — for ceiling measurement only.
#
# Knee penalty subtracts a reward proportional to how close non-foot geoms
# (torso/nose/pelvis/thigh/calf) come to the floor — forces foot-hop technique.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaseq_knee
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaseq_knee
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3 4 5"}
KNEE_COEF=${KNEE_COEF:-0.1}
KNEE_THR=${KNEE_THR:-0.15}

echo "[phaseq] start $(date -u +%FT%TZ) — vanilla TD-MPC2 + knee penalty (coef=$KNEE_COEF thr=$KNEE_THR) 5-seed ceiling" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaseq] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --knee_penalty_coef "$KNEE_COEF" \
    --knee_penalty_threshold "$KNEE_THR" \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaseq] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phaseq] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
