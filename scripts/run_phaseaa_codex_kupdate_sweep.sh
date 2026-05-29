#!/usr/bin/env bash
# Phase-aa-codex: benchmark-fair K_UPDATE sweep for HopperHop.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-"1 2 3"}
K_UPDATES=${K_UPDATES:-"64 128 256"}
SAVE_FULL_STATE=${SAVE_FULL_STATE:-false}

for ku in $K_UPDATES; do
  export TDMPC_GLASS_OUTPUT_TAG="phaseaa_codex_tdmpc2_k${ku}"
  LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
  mkdir -p "$LOG_DIR"
  echo "[phaseaa] start $(date -u +%FT%TZ) K_UPDATE=$ku seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
  for seed in $SEEDS; do
    log="$LOG_DIR/HopperHop_seed_${seed}.log"
    echo "[phaseaa] === k=$ku seed=$seed start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
    cmd=(python3 -u scripts/run_benchmark.py \
      --algos tdmpc2 --tasks HopperHop \
      --total_steps 10000000 --seed "$seed" \
      --mppi_n_samples 2048 \
      --k_update "$ku" \
      --expl_until 500000 \
      --latent_action_smooth_coef 0.001 \
      --latent_smooth_warmup_env_steps 250000 \
      --early_stop_patience 3000000 \
      --no_plot)
    if [[ "$SAVE_FULL_STATE" == "true" ]]; then
      cmd+=(--save_full_state)
    fi
    "${cmd[@]}" 2>&1 | tee -a "$log"
    echo "[phaseaa] === k=$ku seed=$seed done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  done
done
