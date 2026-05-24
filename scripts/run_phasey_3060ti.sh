#!/usr/bin/env bash
# Phase-y (iteration 4 §7.4 / Path 10) on the 3060Ti.
#
# Hierarchical Glass smoke test: K_sub=8 fine clusters AND K_super=4 coarse
# super-clusters trained jointly via two 2D-SE losses on the same prototype
# graph A. See docs/tdmpc-glass/iterations/iteration_4_findings.md §7.5b for design.
#
# Knobs vs Phase-p (Path 1, EXPL_UNTIL=500k):
#   --glass_num_super_clusters   4       ← NEW (Path 10)
#   --glass_lambda_super_se      5e-3    ← NEW (same as λ_se)
#   --glass_lambda_super_balance 1e-2    ← NEW (same as λ_balance)
#
# HopperHop has only 4 gait phases so this is mostly a smoke test of the
# implementation. Real test target is QuadrupedRun (16-32 effective
# primitives) once HopperHop confirms the math is sound.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phasey_3060ti
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasey_3060ti
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
PATIENCE=${PATIENCE:-1500000}
NUM_SUPER=${NUM_SUPER:-4}
LAMBDA_SUPER_SE=${LAMBDA_SUPER_SE:-0.005}
LAMBDA_SUPER_BAL=${LAMBDA_SUPER_BAL:-0.01}

echo "[phasey_3060ti] start $(date -u +%FT%TZ) seeds=[$SEEDS] K_super=$NUM_SUPER λ_super_se=$LAMBDA_SUPER_SE λ_super_bal=$LAMBDA_SUPER_BAL" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasey_3060ti] config: Phase-p baseline + hierarchical Glass (K_sub=8, K_super=$NUM_SUPER)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasey_3060ti] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --glass_num_super_clusters "$NUM_SUPER" \
    --glass_lambda_super_se "$LAMBDA_SUPER_SE" \
    --glass_lambda_super_balance "$LAMBDA_SUPER_BAL" \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasey_3060ti] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasey_3060ti] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
