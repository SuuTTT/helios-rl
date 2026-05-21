#!/usr/bin/env bash
# Phase-g2 (iter-8 §2.3): TD-MPC-Glass V2 — per-pair temporal stability loss.
#
# Adds `lambda_temp_stability * mean(1 - cos_sim(s_t, s_{t+1}))` to the Glass
# loss. Penalises within-gait-phase cluster oscillation — the failure mode
# the rollout-video analysis identified for stuck seeds (blog §3 / suuttt.github.io).
#
# Combined with the proven iter-7 fair stack: K=128 + NS=2048 + EXPL_UNTIL=500k +
# latent_action_smooth curriculum. Eval reward unchanged; Phase-eval (§2.0)
# checkpointing saves best_pi.pkl, best_mppi.pkl, best_any.pkl.
#
# Hypothesis: G1 hit-rate rises from baseline 1/5 to >=3/5 by enforcing the
# winner's signature (stable cluster assignment within a gait phase).
#
# Usage:
#   SEEDS=1 bash scripts/run_phaseg2_temp_stability.sh
# Honoured env vars: SEEDS, XLA_PYTHON_CLIENT_MEM_FRACTION, K_UPDATE, TEMP_STABILITY.

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
TEMP_STABILITY=${TEMP_STABILITY:-0.05}

export TDMPC_GLASS_OUTPUT_TAG="phaseg2_tempstab_${TEMP_STABILITY}"
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

echo "[phaseg2] start $(date -u +%FT%TZ)  K=${K_UPDATE}  temp_stability=${TEMP_STABILITY}  seeds=$SEEDS" \
  | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaseg2] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass \
    --tasks HopperHop \
    --total_steps 10000000 \
    --seed "$seed" \
    --k_update "$K_UPDATE" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --latent_action_smooth_coef 0.001 \
    --latent_smooth_warmup_env_steps 250000 \
    --glass_lambda_temp_stability "$TEMP_STABILITY" \
    --early_stop_patience 3000000 \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaseg2] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[phaseg2] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
