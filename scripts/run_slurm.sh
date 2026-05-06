#!/usr/bin/env bash
# =============================================================================
# SLURM job submission template for helios-rl experiments.
#
# Usage:
#   sbatch scripts/run_slurm.sh [AGENT] [ENV] [EXTRA_HYDRA_ARGS...]
#
# Examples:
#   sbatch scripts/run_slurm.sh ppo mujoco
#   sbatch scripts/run_slurm.sh dreamer_v3 dm_control seed=123
#   sbatch scripts/run_slurm.sh tdmpc2 mujoco agent.lr=1e-4
#
# Adjust the SBATCH directives below to match your cluster configuration.
# At NTU the typical GPU cluster uses nodes with A100/V100 GPUs.
# =============================================================================

#SBATCH --job-name=helios-rl
#SBATCH --output=logs/slurm/%j_%x.out       # stdout/stderr (%j = job id, %x = job name)
#SBATCH --error=logs/slurm/%j_%x.err
#SBATCH --partition=gpu                      # Change to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                   # Adjust based on num_envs
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --mem=32G
#SBATCH --time=48:00:00                     # Maximum wall time

# ---- NTU / HPC-specific module loads (uncomment & adapt as needed) ----
# module load cuda/12.1
# module load python/3.10
# source /path/to/your/virtualenv/bin/activate

# ---- Or use a Conda environment ----
# conda activate helios-rl

# ---- Parse arguments ----
AGENT="${1:-ppo}"
ENV="${2:-mujoco}"
shift 2 2>/dev/null || true
EXTRA_ARGS="$@"

# ---- Ensure log directory exists ----
mkdir -p logs/slurm

# ---- Log environment info ----
echo "========================================================"
echo " Job ID       : $SLURM_JOB_ID"
echo " Node         : $SLURMD_NODENAME"
echo " Agent        : $AGENT"
echo " Environment  : $ENV"
echo " Extra args   : $EXTRA_ARGS"
echo " Start time   : $(date)"
echo "========================================================"

nvidia-smi
python -c "import jax; print('JAX devices:', jax.devices())"

# ---- Run training ----
python -m helios.main \
    agent="${AGENT}" \
    env="${ENV}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?
echo "========================================================"
echo " End time : $(date)"
echo " Exit code: $EXIT_CODE"
echo "========================================================"
exit $EXIT_CODE
