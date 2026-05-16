#!/usr/bin/env bash
# Phase-p on the 3060Ti remote (port 11271) — extra seeds 6, 7, 8.
#
# Fills out statistical sample for Path 1 (larger EXPL_UNTIL=500k). Standard
# seeds 1-5 run on local 4070 Ti + remote 4060. Seeds 6-8 here add three
# more independent RNG paths to confirm Path 1's hit rate isn't a fluke
# of seeds 1-5.

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phasep_3060ti
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasep_3060ti
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"6 7 8"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
EXPL_UNTIL=${EXPL_UNTIL:-500000}
PATIENCE=${PATIENCE:-1500000}

echo "[phasep_3060ti] start $(date -u +%FT%TZ) seeds=[$SEEDS] expl_until=$EXPL_UNTIL" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasep_3060ti] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --expl_until "$EXPL_UNTIL" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasep_3060ti] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phasep_3060ti] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
