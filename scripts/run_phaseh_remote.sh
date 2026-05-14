#!/usr/bin/env bash
# Phase-h on the REMOTE 4060.
#
# Hypothesis: combining Phase-f's latent action smoothing (the strong-but-
# inconsistent winner) with Phase-g's consistency_coef=1.0 (the weak-but-
# present lift) is additive — both interventions act on different parts of
# the loss (policy regularisation vs model regularisation) so they shouldn't
# compete.
#
# Phase 1b setup PLUS BOTH:
#   --latent_action_smooth_coef 1e-3   (Phase-f winner)
#   --consistency_coef 1.0              (Phase-g lift)
#
# Includes 10M cap + 1.5M early-stop (Phase-f methodology).
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phaseh):
#   CSV  -> exp/tdmpc_glass/HopperHop_phaseh/seed_*.csv

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phaseh] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phaseh
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaseh
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
CCOEF=${CCOEF:-1.0}
PATIENCE=${PATIENCE:-1500000}

echo "[phaseh] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS smooth=$SMOOTH ccoef=$CCOEF patience=$PATIENCE" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phaseh] config: Phase 1b knobs + latent_action_smooth_coef=$SMOOTH + consistency_coef=$CCOEF (10M cap, ${PATIENCE}-step early-stop)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phaseh/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phaseh] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phaseh] === seed=${seed} start $(date -u +%FT%TZ) ===" \
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
      --consistency_coef "$CCOEF" \
      --early_stop_patience "$PATIENCE" \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phaseh] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phaseh] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
