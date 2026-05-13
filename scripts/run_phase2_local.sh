#!/usr/bin/env bash
# Phase 2: Glass with stopgrad_graph=False on HopperHop, 5 seeds, local 3090.
#
# Hypothesis: Phase 1b plateaus near ~500 because the world model never
# receives the SE gradient (stopgrad_graph=True). Phase 2 relaxes that and
# keeps all other Phase 1b knobs:
#   --glass_proto_temperature 0.7
#   --glass_assign_logits_init_scale 0.5
#   --glass_stopgrad_graph false   (the only Phase-2 change)
#
# Outputs are isolated from Phase 1 / Phase 1b via TDMPC_GLASS_OUTPUT_TAG=phase2:
#   exp/tdmpc_glass/HopperHop_phase2/seed_<n>.csv
#   exp/tdmpc_glass/HopperHop_phase2/seed_<n>/checkpoints/
#   exp/benchmark/glass_diag/HopperHop_phase2/seed_<n>/
#
# Logs:
#   exp/tdmpc_glass/logs/phase2/HopperHop_seed_<n>.log
#   exp/tdmpc_glass/logs/phase2/queue.log
set -u
set +e

cd /workspace/helios-rl || exit 1

export PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phase2

LOG_DIR=/workspace/helios-rl/exp/tdmpc_glass/logs/phase2
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}

echo "[queue] === Phase 2 start $(date -u +%FT%TZ) ===" | tee -a "$LOG_DIR/queue.log"
echo "[queue] config: stopgrad_graph=false, proto_T=0.7, init_scale=0.5, steps=$TOTAL_STEPS" \
  | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="/workspace/helios-rl/exp/tdmpc_glass/HopperHop_phase2/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[queue] skip seed=${seed}: final checkpoint exists" | tee -a "$LOG_DIR/queue.log"
    continue
  fi
  echo "[queue] === seed=${seed} start $(date -u +%FT%TZ) ===" \
    | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass \
    --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" \
    --seed "$seed" \
    --glass_proto_temperature 0.7 \
    --glass_assign_logits_init_scale 0.5 \
    --glass_stopgrad_graph false \
    --no_plot 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "[queue] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[queue] all phase-2 seeds done at $(date -u +%FT%TZ)" \
  | tee -a "$LOG_DIR/queue.log"
