#!/usr/bin/env bash
# Phase-e on the REMOTE 5070 box.
#
# Hypothesis: Phase 1b seed-4 oscillation (60–230, std≈50, no monotone climb)
# is Q-overestimation oscillation — the value head locks onto two regions
# and `RunningScale` keeps re-normalising. REDQ shows that periodically
# re-initialising the online Q (while keeping the target Q for warm-restart)
# breaks the oscillation.
#
# Knobs vs Phase 1b:
#   --glass_proto_temperature        0.7      (same)
#   --glass_assign_logits_init_scale 0.5      (same)
#   --glass_stopgrad_graph           true     (same)
#   --q_reset_steps                  "1000000,2000000,3000000"   ← NEW
#   (act-noise stays at default 0.30 constant — Phase 1b knob)
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phasee):
#   CSV  -> exp/tdmpc_glass/HopperHop_phasee/seed_*.csv
#   ckpt -> exp/tdmpc_glass/HopperHop_phasee/seed_*/checkpoints/
#   logs -> exp/tdmpc_glass/logs/phasee/HopperHop_seed_*.log
#
# Launch on the remote 5070:
#   nohup setsid bash scripts/run_phasee_remote.sh \
#       > exp/tdmpc_glass/logs/phasee/queue.log 2>&1 < /dev/null & disown

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phasee] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 5070 has more VRAM headroom than 3090 — but we keep this conservative;
# raise to 0.85 once smoke run passes.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}
export TDMPC_GLASS_OUTPUT_TAG=phasee
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phasee
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}

echo "[phasee] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phasee] config: proto_T=0.7 init_scale=0.5 stopgrad=true q_reset=1M,2M,3M expl_noise=0.30_constant" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phasee/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phasee] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phasee] === seed=${seed} start $(date -u +%FT%TZ) ===" \
      | tee "$log" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks HopperHop \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --glass_proto_temperature 0.7 \
      --glass_assign_logits_init_scale 0.5 \
      --glass_stopgrad_graph true \
      --q_reset_steps "1000000,2000000,3000000" \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phasee] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phasee] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
