#!/usr/bin/env bash
# Phase-z (iter 6 Q1): vanilla TD-MPC2 (NO Glass) — 5 seeds — apples-to-apples
# baseline for Phase-x (Path 9 NS=2048).
#
# Difference vs Phase-x: --algos tdmpc2 (not tdmpc-glass). All other knobs
# match so we can isolate the Glass contribution.
#
# 5 seeds sequentially on local 4070 Ti. ~25h total.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasez_baseline
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasez_baseline
mkdir -p "$LOG_DIR"

SEEDS=${SEEDS:-"1 2 3 4 5"}

echo "[phasez] start $(date -u +%FT%TZ) — vanilla TD-MPC2 NS=2048 EXPL=500k 5-seed baseline" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasez] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasez] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasez] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
