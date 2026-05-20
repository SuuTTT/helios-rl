#!/usr/bin/env bash
# Phase-ac-codex: selected benchmark-fair TD-MPC-Glass 5-seed comparison.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-"1 2 3 4 5"}
K_UPDATE=${K_UPDATE:-128}
export TDMPC_GLASS_OUTPUT_TAG="phaseac_codex_glass_k${K_UPDATE}"
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

echo "[phaseac] start $(date -u +%FT%TZ) TD-MPC-Glass K_UPDATE=$K_UPDATE seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaseac] === seed=$seed start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --mppi_n_samples 2048 \
    --k_update "$K_UPDATE" \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --early_stop_patience 3000000 \
    --save_full_state \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaseac] === seed=$seed done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

