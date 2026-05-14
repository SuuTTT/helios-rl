#!/usr/bin/env bash
# Phase-k on the REMOTE 4060.
#
# Phase-j's curriculum (warmup=250k) + raised λ_temporal=5e-2 (50x default 1e-3).
# Tests the K-flicker hypothesis: penalising rapid cluster changes more
# strongly should push policy toward stable long-cluster gaits (the
# foot-hop signature) and away from flickery knee-walk gaits.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phasek
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasek
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
LAMBDA_TEMPORAL=${LAMBDA_TEMPORAL:-0.05}
PATIENCE=${PATIENCE:-1500000}

echo "[phasek] start $(date -u +%FT%TZ) seeds=[$SEEDS] smooth=$SMOOTH warmup=$WARMUP λ_temporal=$LAMBDA_TEMPORAL" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasek] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --glass_lambda_temporal "$LAMBDA_TEMPORAL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasek] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasek] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
