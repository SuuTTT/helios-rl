#!/usr/bin/env bash
set -u
set +e

REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.65}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SEEDS=${SEEDS:-"1 2 3 4 5"}
K_UPDATE=${K_UPDATE:-128}
MPC_DISTILL_COEF=${MPC_DISTILL_COEF:-1.0}
MPC_DISTILL_ANNEAL_STEPS=${MPC_DISTILL_ANNEAL_STEPS:-3000000}
MPC_DISTILL_DISABLE_GAP=${MPC_DISTILL_DISABLE_GAP:-100}
MPC_DISTILL_BATCH_SIZE=${MPC_DISTILL_BATCH_SIZE:-16}

export TDMPC_GLASS_OUTPUT_TAG="phasemplite_k${K_UPDATE}_d${MPC_DISTILL_COEF}"
LOG_DIR=$REPO/exp/tdmpc_glass/logs/$TDMPC_GLASS_OUTPUT_TAG
mkdir -p "$LOG_DIR"

echo "[phasemplite] start $(date -u +%FT%TZ)  seeds=$SEEDS  K=${K_UPDATE}  distill=${MPC_DISTILL_COEF}" \
  | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phasemplite] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc2 \
    --tasks HopperHop \
    --total_steps 10000000 \
    --seed "$seed" \
    --k_update "$K_UPDATE" \
    --mppi_n_samples 2048 \
    --expl_until 500000 \
    --early_stop_patience 3000000 \
    --mpc_distill_coef "$MPC_DISTILL_COEF" \
    --mpc_distill_anneal_steps "$MPC_DISTILL_ANNEAL_STEPS" \
    --mpc_distill_disable_gap "$MPC_DISTILL_DISABLE_GAP" \
    --mpc_distill_batch_size "$MPC_DISTILL_BATCH_SIZE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phasemplite] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" \
    | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done

echo "[phasemplite] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
