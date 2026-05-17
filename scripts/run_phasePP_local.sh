#!/usr/bin/env bash
# Phase-P (iteration 4 / Path P) on the LOCAL 4070 Ti.
#
# Cluster-entropy intrinsic reward — benchmark-fair (no env modification).
# At each env step, compute current Glass cluster id; maintain a sliding
# window of last W cluster ids per env; add coef * entropy(window) to
# training reward. Encourages gait diversity (proper hopping = multiple
# distinct phases) without telling the policy WHICH gait is correct.
#
# Knobs vs Phase-p baseline:
#   --cluster_intrinsic_coef       0.1    ← NEW (Path P)
#   --cluster_intrinsic_window     20     (~0.4s at 50Hz)
#   --expl_until                   500000 (Path 1, helps slow-burn)
#
# 3 seeds × 10M cap + 3M patience.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasePP
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasePP
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3"}
INTR_COEF=${INTR_COEF:-0.1}
INTR_W=${INTR_W:-20}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-3000000}

echo "[phasePP] start $(date -u +%FT%TZ) seeds=[$SEEDS] intr_coef=$INTR_COEF intr_window=$INTR_W expl_until=$EXPL_UNTIL" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasePP] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --cluster_intrinsic_coef "$INTR_COEF" \
    --cluster_intrinsic_window "$INTR_W" \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasePP] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasePP] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
