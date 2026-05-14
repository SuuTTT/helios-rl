#!/usr/bin/env bash
# Phase 1c on remote (vast.ai 4070 Ti, ssh8.vast.ai:20305).
#
# "Fixed-version" experiment: Phase 1b knobs PLUS act-noise anneal
# 0.30 -> 0.10 over the first 1 M env-steps (blog §5.6.1 fix candidate #1).
# We re-run seeds 1-5 so the comparison to Phase 1b is direct
# (Phase 1b finals: 438 / 526 / 294 / 187 / 562 — seeds 3/4 stuck in low
#  basins despite identical Glass diagnostics).
#
# Knobs:
#   --glass_proto_temperature        0.7
#   --glass_assign_logits_init_scale 0.5
#   --glass_stopgrad_graph           true
#   --act_noise_start                0.30
#   --act_noise_end                  0.10
#   --act_noise_anneal_steps         1000000
#
# Outputs (tag=phase1c):
#   CSV  -> exp/tdmpc_glass/HopperHop_phase1c/seed_*.csv
#   ckpt -> exp/tdmpc_glass/HopperHop_phase1c/seed_*/checkpoints/
#   mp4  -> exp/tdmpc_glass/videos/phase1c/seed_*_best_mppi.mp4
#
# Launch on the box:
#   nohup setsid scripts/run_phase1c_remote.sh \
#       > exp/tdmpc_glass/logs/phase1c/queue.log 2>&1 < /dev/null & disown

set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || { echo "[phase1c] repo not found at $REPO"; exit 1; }

# Activate the project venv if present (vast.ai box convention: /root/venv).
if [[ -f /root/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venv/bin/activate
fi

export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phase1c
export MUJOCO_GL=${MUJOCO_GL:-egl}

LOG_DIR=$REPO/exp/tdmpc_glass/logs/phase1c
VID_DIR=$REPO/exp/tdmpc_glass/videos/phase1c
mkdir -p "$LOG_DIR" "$VID_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}

echo "[phase1c] start $(date -u +%FT%TZ) seeds=[$SEEDS]" | tee -a "$LOG_DIR/queue.log"
echo "[phase1c] config: proto_T=0.7 init_scale=0.5 stopgrad=true act_noise=0.30->0.10/1M" \
    | tee -a "$LOG_DIR/queue.log"

# ── Phase 1: train all seeds sequentially (rendering never touches the GPU here)
for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phase1c/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[phase1c] skip seed=${seed}: final.pkl already present" | tee -a "$LOG_DIR/queue.log"
  else
    echo "[phase1c] === seed=${seed} start $(date -u +%FT%TZ) ===" \
      | tee "$log" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks HopperHop \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --glass_proto_temperature 0.7 \
      --glass_assign_logits_init_scale 0.5 \
      --glass_stopgrad_graph true \
      --act_noise_start 0.30 \
      --act_noise_end 0.10 \
      --act_noise_anneal_steps 1000000 \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "[phase1c] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
      | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  fi
done

echo "[phase1c] all training done at $(date -u +%FT%TZ), launching renders in background" \
  | tee -a "$LOG_DIR/queue.log"

# ── Phase 2: render all seeds in parallel background jobs (CPU-bound, non-blocking)
render_pids=()
for seed in $SEEDS; do
  ckpt_dir="$REPO/exp/tdmpc_glass/HopperHop_phase1c/seed_${seed}/checkpoints"
  ckpt="${ckpt_dir}/best_mppi.pkl"
  out="$VID_DIR/seed_${seed}_best_mppi.mp4"
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  if [[ -f "$ckpt" ]]; then
    echo "[phase1c] rendering seed=${seed} -> ${out} (background)" | tee -a "$LOG_DIR/queue.log"
    python3 -u scripts/render_glass_rollout.py \
      --ckpt "$ckpt" \
      --env_id HopperHop \
      --out "$out" \
      --n_episodes 3 \
      --episode_length 1000 \
      --camera cam0 \
      --seed $((100 + seed)) >> "$log" 2>&1 &
    render_pids+=($!)
  else
    echo "[phase1c] no best_mppi.pkl for seed=${seed}, skipping render" \
      | tee -a "$LOG_DIR/queue.log"
  fi
done

# Wait for all renders to finish before declaring all done
if [[ ${#render_pids[@]} -gt 0 ]]; then
  echo "[phase1c] waiting for ${#render_pids[@]} render job(s)..." | tee -a "$LOG_DIR/queue.log"
  for pid in "${render_pids[@]}"; do wait "$pid"; done
fi

echo "[phase1c] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
