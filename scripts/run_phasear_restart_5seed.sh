#!/usr/bin/env bash
# Phase-ar (iter-7 §2.1): Auto-Restart on plateau detection.
# Vanilla TD-MPC2 + the proven post-basin levers (NS=2048, K=128, EXPL_UNTIL=500k,
# smoothing curriculum), with the *new* basin-entry lever: at every 1M env-step
# boundary, if best MPPI is still < 100 (basin-locked), re-initialise pi+q+target_q+opt
# and continue training. Up to 3 attempts per seed. Encoder, dynamics, reward
# head, replay buffer, env state, and PRNG advancement are all preserved.
#
# Hypothesis: stuck seeds are a stochastic basin-entry problem. By re-rolling pi+q
# while keeping the world model + replay, each attempt is a fresh shot at the
# hop-basin with the encoder already warm. Per-attempt G1 ≈ 0.30 (iter 6 data)
# → 3 attempts gives expected ~66% per-seed → 5-seed sweep expected 3.3 winners
# with long tail toward 5/5.
#
# Benchmark-fair: no reward shaping, no BC, no env modification, eval reward
# unchanged. Closest prior art: Nikishin 2022 "Primacy Bias in Deep RL" (scheduled
# layer resets every 200k steps); our trigger is reactive (only-if-stuck).
#
# Usage:
#   SEEDS=1 bash scripts/run_phasear_restart_5seed.sh
# Honoured env vars: SEEDS, XLA_PYTHON_CLIENT_MEM_FRACTION, K_UPDATE,
# RESTART_CHECK_AT, RESTART_THRESHOLD, RESTART_MAX_ATTEMPTS.

set -u; set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-"1 2 3 4 5"}
K_UPDATE=${K_UPDATE:-128}
RESTART_CHECK_AT=${RESTART_CHECK_AT:-1000000}
RESTART_THRESHOLD=${RESTART_THRESHOLD:-100}
RESTART_MAX_ATTEMPTS=${RESTART_MAX_ATTEMPTS:-3}

export TDMPC_GLASS_OUTPUT_TAG="phasear_restart_K${K_UPDATE}"
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

echo "[phasear] start $(date -u +%FT%TZ)  K=${K_UPDATE}  restart_check_at=${RESTART_CHECK_AT}  threshold=${RESTART_THRESHOLD}  max=${RESTART_MAX_ATTEMPTS}  seeds=$SEEDS" \
  | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasear] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 \
    --tasks HopperHop \
    --total_steps 10000000 \
    --seed "$seed" \
    --k_update "$K_UPDATE" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --restart_on_plateau \
    --restart_check_at "$RESTART_CHECK_AT" \
    --restart_threshold "$RESTART_THRESHOLD" \
    --restart_max_attempts "$RESTART_MAX_ATTEMPTS" \
    --early_stop_patience 3000000 \
    --save_full_state \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasear] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[phasear] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
