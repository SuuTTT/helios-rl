#!/usr/bin/env bash
# Phase-g on the REMOTE 4060.
#
# Hypothesis: reducing the consistency loss weight from 2.0 to 1.0 lets the
# encoder fit the actual data distribution better instead of being dominated
# by the latent rollout prediction error. Blog §9 item 3 calls this out:
# "Decouple consistency and TD loss weights. They share one Adam now; on
# dense-reward tasks the consistency term over-regularises."
# HopperHop is medium-density, so a 2x reduction is a conservative test.
#
# Phase 1b setup PLUS:
#   --consistency_coef 1.0    (was hard-coded 2.0)
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phaseg):
#   CSV  -> exp/tdmpc_glass/HopperHop_phaseg/seed_*.csv
#   ckpt -> exp/tdmpc_glass/HopperHop_phaseg/seed_*/checkpoints/
#   logs -> exp/tdmpc_glass/logs/phaseg/HopperHop_seed_*.log

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phaseg] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 4060 has only 8 GiB; 0.55 is safe (Phase-e never exceeded ~1 GiB).
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.55}
export TDMPC_GLASS_OUTPUT_TAG=phaseg
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaseg
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
CCOEF=${CCOEF:-1.0}

echo "[phaseg] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS ccoef=$CCOEF" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phaseg] config: Phase 1b knobs + consistency_coef=$CCOEF (default 2.0)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phaseg/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phaseg] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phaseg] === seed=${seed} start $(date -u +%FT%TZ) ===" \
      | tee "$log" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks HopperHop \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --glass_proto_temperature 0.7 \
      --glass_assign_logits_init_scale 0.5 \
      --glass_stopgrad_graph true \
      --consistency_coef "$CCOEF" \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phaseg] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phaseg] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
