#!/usr/bin/env bash
# Phase-i on the LOCAL 4070 Ti.
#
# Hypothesis: Phase-f's latent_action_smooth=1e-3 perturbed Glass basin
# discovery for 2/5 seeds (K=3 ⊂ K=4 collapse). At 10x weaker (1e-4), the
# basin should stay K=4 like Phase 1b had for all seeds, while still
# providing some policy-side regularisation. Test if seed 1 still surges.
#
# Phase 1b setup PLUS:
#   --latent_action_smooth_coef 1e-4   (10x weaker than Phase-f's 1e-3)
#
# 10M cap + 1.5M early-stop (Phase-f methodology).
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phasei):
#   CSV  -> exp/tdmpc_glass/HopperHop_phasei/seed_*.csv

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phasei] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phasei
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasei
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.0001}
PATIENCE=${PATIENCE:-1500000}

echo "[phasei] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS smooth=$SMOOTH patience=$PATIENCE" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasei] config: Phase 1b knobs + latent_action_smooth_coef=$SMOOTH (10x weaker than Phase-f), 10M cap + 1.5M early-stop" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phasei/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phasei] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phasei] === seed=${seed} start $(date -u +%FT%TZ) ===" \
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
    echo "[phasei] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phasei] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
