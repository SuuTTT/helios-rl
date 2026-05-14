#!/usr/bin/env bash
# Phase-f on the LOCAL 4070 Ti.
#
# Hypothesis: latent action smoothing helps HopperHop policy escape local-maximum
# gaits by penalising rapid action changes across the consistency-loss rollout.
# Blog §9 item 10 specifically calls this out as a "big return gains on
# underactuated DMC tasks" cheap intervention.
#
# Phase 1b setup PLUS:
#   --latent_action_smooth_coef 1e-3    ← NEW
#
# Notably we revert H back to 3 (Phase 1b default) — Phase-d showed H=5 alone
# also produces a stuck plateau (just at a higher floor). H=3 is the v24 baseline.
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phasef):
#   CSV  -> exp/tdmpc_glass/HopperHop_phasef/seed_*.csv
#   ckpt -> exp/tdmpc_glass/HopperHop_phasef/seed_*/checkpoints/
#   logs -> exp/tdmpc_glass/logs/phasef/HopperHop_seed_*.log

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phasef] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasef
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasef
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
PATIENCE=${PATIENCE:-1500000}

echo "[phasef] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS smooth=$SMOOTH patience=$PATIENCE" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasef] config: Phase 1b knobs + latent_action_smooth_coef=$SMOOTH (H=3 default), 10M cap + 1.5M early-stop" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phasef/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phasef] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phasef] === seed=${seed} start $(date -u +%FT%TZ) ===" \
      | tee "$log" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks HopperHop \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --glass_proto_temperature 0.7 \
      --glass_assign_logits_init_scale 0.5 \
      --glass_stopgrad_graph true \
      --latent_action_smooth_coef "$SMOOTH" \
      --early_stop_patience "$PATIENCE" \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phasef] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phasef] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
