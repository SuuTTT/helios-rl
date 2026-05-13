#!/usr/bin/env bash
# Queue: wait for the in-flight seed_3 run, then run the other four seeds
# sequentially on one GPU. Each run takes ~3h on RTX 3090.
set -u
set +e

cd /workspace/helios-rl || exit 1

export PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

LOG_DIR=/workspace/helios-rl/exp/tdmpc_glass/logs/phase1
mkdir -p "$LOG_DIR"

# 1) Wait for the in-flight seed_3 run (PID 801272 or any running run_benchmark).
echo "[queue] waiting for in-flight run_benchmark to exit..." | tee -a "$LOG_DIR/queue.log"
while pgrep -f "run_benchmark.py.*tdmpc-glass.*HopperHop" >/dev/null 2>&1; do
  sleep 60
done
echo "[queue] in-flight run finished at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"

TOTAL_STEPS=${TOTAL_STEPS:-4000000}

for seed in 1 2 4 5; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  ckpt_dir="/workspace/helios-rl/exp/tdmpc_glass/HopperHop/seed_${seed}/checkpoints"
  if [[ -f "${ckpt_dir}/final.pkl" ]]; then
    echo "[queue] skip seed=${seed}: final checkpoint exists" | tee -a "$LOG_DIR/queue.log"
    continue
  fi
  echo "[queue] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass \
    --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" \
    --seed "$seed" \
    --no_plot 2>&1 | tee -a "$log"
  status=${PIPESTATUS[0]}
  echo "[queue] === seed=${seed} done status=${status} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

# 2) When all 5 seeds are present, generate the 95% CI plot.
echo "[queue] generating 5-seed CI plot..." | tee -a "$LOG_DIR/queue.log"
mkdir -p exp/tdmpc_glass/plots/pre_phase1
mv exp/tdmpc_glass/plots/hopperhop_tdmpc_glass_vs_official_95ci*.{png,csv} \
   exp/tdmpc_glass/plots/pre_phase1/ 2>/dev/null || true
python3 -u scripts/plot_tdmpc_glass_ci.py \
  --task HopperHop \
  --glass_dir exp/tdmpc_glass/HopperHop \
  --official_csv /workspace/tdmpc2/results/tdmpc2/hopper-hop.csv \
  --out_dir exp/tdmpc_glass/plots 2>&1 \
  | tee -a "$LOG_DIR/queue.log"

echo "[queue] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
