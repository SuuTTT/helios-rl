#!/usr/bin/env bash
# Phase-Pa (iteration 4 / Path P-anneal) on the LOCAL 4070 Ti.
#
# Cluster-entropy intrinsic reward with LINEAR DECAY. Phase-P (static
# coef=0.1) hit MPPI=91 @ 1.25M then collapsed to 2.4 @ 2M because the
# converging policy lost intrinsic reward and abandoned the hop gait
# to maintain cluster diversity. Fix: anneal coef 0.1 -> 0 over
# [expl_until, decay_steps] so intrinsic acts as exploration curriculum
# (analogous to Phase-j curriculum smoothing), then training runs on
# pure extrinsic reward — same regime as the Phase-p seed-4 = 538 winner.
#
# Knobs vs Phase-P:
#   --cluster_intrinsic_decay_steps  3000000  ← NEW (anneal)
#   --cluster_intrinsic_coef         0.1
#   --cluster_intrinsic_window       20
#   --expl_until                     500000
#
# 3 seeds x 10M cap, 3M patience.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasePa
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasePa
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3"}
INTR_COEF=${INTR_COEF:-0.1}
INTR_W=${INTR_W:-20}
INTR_DECAY=${INTR_DECAY:-3000000}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-3000000}

echo "[phasePa] start $(date -u +%FT%TZ) seeds=[$SEEDS] intr=${INTR_COEF}->0 over [${EXPL_UNTIL},${INTR_DECAY}] window=$INTR_W" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasePa] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --cluster_intrinsic_coef "$INTR_COEF" \
    --cluster_intrinsic_window "$INTR_W" \
    --cluster_intrinsic_decay_steps "$INTR_DECAY" \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasePa] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasePa] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
