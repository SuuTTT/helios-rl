#!/usr/bin/env bash
# Phase-j on the LOCAL 4070 Ti.
#
# Curriculum smoothing — the principled fix for the trade-off we found:
#   • Phase-f (smooth=1e-3 from step 0): strong, surged seed 1 to 571, but
#     perturbed Glass basin → seeds 4,5 flipped to K=3 (capped at 300).
#   • Phase-i (smooth=1e-4 from step 0): basin-neutral, but too weak —
#     seed 1 plateaued at 308 (same shape as Phase 1b stuck seed).
#
# Curriculum: 0 for first 250k env steps (basin lock window per blog §5.4),
# then jump to 1e-3 (full Phase-f strength). Should keep all seeds K=4 AND
# give the policy-regularisation lift.
#
#   --latent_action_smooth_coef       1e-3
#   --latent_smooth_warmup_env_steps  250000
#   (+ 10M cap + 1.5M early-stop)
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phasej):
#   CSV  -> exp/tdmpc_glass/HopperHop_phasej/seed_*.csv

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phasej] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasej
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasej
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
PATIENCE=${PATIENCE:-1500000}

echo "[phasej] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS smooth=$SMOOTH warmup=$WARMUP patience=$PATIENCE" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasej] config: Phase 1b knobs + curriculum smoothing (0 → $SMOOTH at env_steps=$WARMUP), 10M cap" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phasej/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phasej] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phasej] === seed=${seed} start $(date -u +%FT%TZ) ===" \
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
      --latent_smooth_warmup_env_steps "$WARMUP" \
      --early_stop_patience "$PATIENCE" \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phasej] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phasej] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
