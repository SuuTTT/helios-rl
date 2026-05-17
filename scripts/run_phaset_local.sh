#!/usr/bin/env bash
# Phase-t (iteration 4 / Path 5) on the LOCAL 4070 Ti.
#
# Reward shaping: penalise non-foot geoms (torso/nose/pelvis/thigh/calf)
# coming close to the floor. Forces foot-hop technique by punishing
# knee-walk gaits directly. Training-only modification — eval reward
# unchanged so we measure against the original HopperHop task.
#
# Knobs vs Phase-p baseline:
#   --knee_penalty_coef        0.1     ← NEW (Path 5)
#   --knee_penalty_threshold   0.15    (geom z threshold in meters)
#   --total_steps              3000000 (reduced from 10M, per user request)
#
# 3 seeds (user request: "later phase seed scale, we run 3 seed for one phase")

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaset
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaset
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3"}
KNEE_COEF=${KNEE_COEF:-0.1}
KNEE_THR=${KNEE_THR:-0.15}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-3000000}

echo "[phaset] start $(date -u +%FT%TZ) seeds=[$SEEDS] knee_coef=$KNEE_COEF knee_thr=$KNEE_THR total=$TOTAL_STEPS" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaset] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --knee_penalty_coef "$KNEE_COEF" \
    --knee_penalty_threshold "$KNEE_THR" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaset] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phaset] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
