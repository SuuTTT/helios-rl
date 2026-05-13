#!/usr/bin/env bash
set -u
set +e

cd /workspace/helios-rl || exit 1

export PYTHONPATH=/workspace/helios-rl/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Slightly below the previous full-speed setting to reduce long-run Warp/CUDA
# capture instability while keeping most of the throughput.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}

TASKS=(HopperHop FishSwim AcrobotSwingup)
SEEDS=(1 2 3 4 5)
TOTAL_STEPS=${TOTAL_STEPS:-4000000}
LOG_DIR=/workspace/helios-rl/exp/tdmpc_glass/logs
mkdir -p "$LOG_DIR"

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    log="$LOG_DIR/${task}_seed_${seed}.log"
    ckpt_dir="/workspace/helios-rl/exp/tdmpc_glass/${task}/seed_${seed}/checkpoints"
    if [[ -f "${ckpt_dir}/final.pkl" ]]; then
      echo "=== skip ${task} seed=${seed}: final checkpoint exists ===" | tee "$log"
      continue
    fi
    echo "=== TD-MPC-Glass ${task} seed=${seed} steps=${TOTAL_STEPS} ===" | tee "$log"
    python3 scripts/run_benchmark.py \
      --algos tdmpc-glass \
      --tasks "$task" \
      --total_steps "$TOTAL_STEPS" \
      --seed "$seed" \
      --save_full_state \
      --no_plot 2>&1 | tee -a "$log"
    status=${PIPESTATUS[0]}
    echo "=== done ${task} seed=${seed} status=${status} ===" | tee -a "$log"
    if [[ "$status" -ne 0 ]]; then
      echo "Run failed; continuing to next seed/task." | tee -a "$log"
    else
      rm -f "${ckpt_dir}"/*_full.pkl
      echo "Cleaned full-state recovery checkpoints after successful run." | tee -a "$log"
    fi
  done
done
