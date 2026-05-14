#!/usr/bin/env bash
# Phase-d on the LOCAL 4070 Ti box.
#
# Hypothesis: Phase 1b seeds 3/4 stuck (294 / 187) because (a) MPPI horizon
# H=3 can't plan past the local-maximum "shuffle" gait and (b) action-noise
# 0.30 isn't wide enough to escape the policy basin.
#
# Knobs vs Phase 1b:
#   --glass_proto_temperature        0.7      (same)
#   --glass_assign_logits_init_scale 0.5      (same)
#   --glass_stopgrad_graph           true     (same)
#   --mppi_horizon                   5        ↑ from 3 — wider planning reach
#   (noise stays at Phase 1b default 0.30 — see phased_v1_noise040: 0.40
#    caused mjx Warp graph-capture error 901 at ~1M env steps when hopper
#    drifted into non-converging solver states; H=5 isolated from noise)
#
# Outputs (TDMPC_GLASS_OUTPUT_TAG=phased):
#   CSV  -> exp/tdmpc_glass/HopperHop_phased/seed_*.csv
#   ckpt -> exp/tdmpc_glass/HopperHop_phased/seed_*/checkpoints/
#   logs -> exp/tdmpc_glass/logs/phased/HopperHop_seed_*.log
#
# Smoke first (SEEDS="1 2", TOTAL_STEPS=1500000) to see if hypothesis is alive
# by 1.5M; if at least one seed > 300 by 1.5M, scale to full 5 seeds × 4M.
#
# Launch:
#   nohup setsid bash scripts/run_phased_local.sh \
#       > exp/tdmpc_glass/logs/phased/queue.log 2>&1 < /dev/null & disown

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phased] repo not found at $REPO"; exit 1; }

if [[ -f /root/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phased
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phased
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}

echo "[phased] start $(date -u +%FT%TZ) seeds=[$SEEDS] total_steps=$TOTAL_STEPS" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phased] config: proto_T=0.7 init_scale=0.5 stopgrad=true H=5 expl_noise=0.30_default (was 0.40 in v1, hit mjx-warp-901)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phased/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phased] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phased] === seed=${seed} start $(date -u +%FT%TZ) ===" \
      | tee "$log" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks HopperHop \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --glass_proto_temperature 0.7 \
      --glass_assign_logits_init_scale 0.5 \
      --glass_stopgrad_graph true \
      --mppi_horizon 5 \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phased] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phased] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
