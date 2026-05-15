#!/usr/bin/env bash
# Phase-o on the LOCAL 4070 Ti — hybrid abstraction (Glass-then-vanilla).
#
# Hypothesis: Glass is helpful as scaffolding early (representation
# learning, basin discovery) but acts as a blocker later (the partition
# is an inductive bias the policy can't unlearn). Empirically: blog
# Phase 1 Glass-on avg 366 < official TD-MPC2 449. And Phase-m showed
# K=4 basin still caps at 262 — abstraction doesn't help peak.
#
# Phase-o: Glass active in window [100k warmup, 2M cutoff], OFF after 2M.
# By 2M the encoder is well-structured from Glass pressure; from then on
# it's pure TD-MPC2 + latent action smoothing.
#
# Knobs:
#   --glass_decay_steps           2000000   ← NEW (Phase-o)
#   --latent_action_smooth_coef   1e-3
#   --latent_smooth_warmup_env_steps 250000   (Phase-m fix retained)
#   --early_stop_patience         1500000

set -u
set +e
REPO=${REPO:-/root/helios-rl}
cd "$REPO" || exit 1
[[ -f /root/venv/bin/activate ]] && source /root/venv/bin/activate
export PYTHONPATH=$REPO/src:/root/mujoco_playground_repo:${PYTHONPATH:-}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}
export TDMPC_GLASS_OUTPUT_TAG=phaseo
export MUJOCO_GL=${MUJOCO_GL:-egl}
LOG_DIR=$REPO/exp/tdmpc_glass/logs/phaseo
mkdir -p "$LOG_DIR"

TOTAL_STEPS=${TOTAL_STEPS:-10000000}
SEEDS=${SEEDS:-"1 2 3 4 5"}
SMOOTH=${SMOOTH:-0.001}
WARMUP=${WARMUP:-250000}
DECAY=${DECAY:-2000000}
PATIENCE=${PATIENCE:-1500000}

echo "[phaseo] start $(date -u +%FT%TZ) seeds=[$SEEDS] smooth=$SMOOTH warmup=$WARMUP glass_decay=$DECAY" \
    | tee -a "$LOG_DIR/queue.log"
echo "[phaseo] config: Phase-m + Glass loss OFF after env_steps=$DECAY (hybrid abstraction)" \
    | tee -a "$LOG_DIR/queue.log"

for seed in $SEEDS; do
  log="$LOG_DIR/HopperHop_seed_${seed}.log"
  echo "[phaseo] === seed=${seed} start $(date -u +%FT%TZ) ===" | tee "$log" | tee -a "$LOG_DIR/queue.log"
  python3 -u scripts/run_benchmark.py \
    --algos tdmpc-glass --tasks HopperHop \
    --total_steps "$TOTAL_STEPS" --seed "$seed" \
    --glass_proto_temperature 0.7 --glass_assign_logits_init_scale 0.5 --glass_stopgrad_graph true \
    --glass_decay_steps "$DECAY" \
    --latent_action_smooth_coef "$SMOOTH" \
    --latent_smooth_warmup_env_steps "$WARMUP" \
    --early_stop_patience "$PATIENCE" \
    --no_plot 2>&1 | tee -a "$log"
  echo "[phaseo] === seed=${seed} done status=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$log" | tee -a "$LOG_DIR/queue.log"
done
echo "[phaseo] all done at $(date -u +%FT%TZ)" | tee -a "$LOG_DIR/queue.log"
