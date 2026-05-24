#!/usr/bin/env bash
# Phase-i9b (iter-9): Glass V2 lower temporal-stability coefficient probe.
#
# One-seed sentinel probe for whether Phase-g2's positive signal survives with a
# weaker temporal-stability coefficient. Queue one seed per task.

set -u; set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-1}
K_UPDATE=${K_UPDATE:-128}
TEMP_STABILITY=${TEMP_STABILITY:-0.01}
CODE_SHA=${CODE_SHA:-$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)}
TAG_SHA=${CODE_SHA//[^A-Za-z0-9_.-]/_}

export TDMPC_GLASS_OUTPUT_TAG=${TDMPC_GLASS_OUTPUT_TAG:-phasei9b_temp001_${TAG_SHA}}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

{
  echo "[phasei9b] start $(date -u +%FT%TZ)  seeds=${SEEDS}  K=${K_UPDATE}  temp_stability=${TEMP_STABILITY}"
  echo "[phasei9b] git_sha=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "[phasei9b] code_sha_env=${CODE_SHA:-unset}"
  echo "[phasei9b] output_tag=${TDMPC_GLASS_OUTPUT_TAG}"
} | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasei9b] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
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
  echo "[phasei9b] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[phasei9b] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
