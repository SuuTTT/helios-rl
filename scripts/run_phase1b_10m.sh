#!/usr/bin/env bash
# Phase 1b Glass baseline rerun — 10M cap + 3M early-stop patience.
#
# Same Glass knobs as the original Phase 1b:
#   proto_temperature=0.7, assign_logits_init_scale=0.5, stopgrad_graph=true,
#   act_noise=0.30 (default), no latent smoothing, no EXPL_UNTIL boost.
# Original ran to 3–4M steps; finals were [438,526,294,187,562] mean=401.
# This reruns to 10M with 3M patience to see if stuck seeds eventually escape.
#
# Output tag: phase1b_10m
# CSV -> exp/tdmpc_glass/HopperHop_phase1b_10m/seed_N/eval.csv
# Checkpoint -> exp/tdmpc_glass/HopperHop_phase1b_10m/seed_N/checkpoints/

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phase1b_10m
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-"1 2 3 4 5"}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phase1b_10m
mkdir -p "$LOG_DIR"

echo "[phase1b_10m] start $(date -u +%FT%TZ) seeds=$SEEDS" | tee -a "$LOG_DIR/queue.log"
echo "[phase1b_10m] config: Glass baseline knobs (T=0.7, scale=0.5, stopgrad=true), 10M cap, 3M patience" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phase1b_10m] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps 10000000 --seed "$seed" \
    --glass_proto_temperature 0.7 \
    --glass_assign_logits_init_scale 0.5 \
    --glass_stopgrad_graph true \
    --early_stop_patience 3000000 \
    --save_full_state \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phase1b_10m] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
