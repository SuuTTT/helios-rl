# helios-rl

**H**igh-performance **E**xtensible **L**atent **I**nference & **O**ptimization **S**ystem

A modular JAX-first reinforcement learning research framework. The architecture cleanly separates *World Model* logic from *Action Selection* logic, enabling easy composition of different dynamics models and planning algorithms.

---

## Features

- **Three algorithms out of the box** — PPO (model-free), DreamerV3 (imagination-based), TD-MPC2 (planning-based)
- **Pluggable dynamics models** — RSSM (DreamerV3), JEPA, RNN-MDN
- **Derivative-free planners** — CEM, MPPI
- **Replay buffers** — ephemeral rollout buffer (PPO) + sequence-based trajectory buffer (world models)
- **JAX-native** — pure-function networks, explicit state passing, `jax.vmap`/`jax.jit` throughout
- **Hydra config system** — compose experiments from YAML overrides
- **W&B logging** and SLURM job templates included

---

## Project Layout

```text
helios-rl/
├── pyproject.toml              # Dependency management (hatchling / uv / poetry)
├── configs/
│   ├── experiment/
│   │   └── default.yaml        # Root experiment config (seed, steps, W&B, …)
│   └── agent/
│       ├── ppo.yaml
│       ├── dreamer_v3.yaml
│       └── tdmpc2.yaml
├── src/
│   └── helios/
│       ├── main.py             # Hydra entry point
│       ├── core/
│       │   ├── networks.py     # Shared encoders / decoders / MLPs
│       │   └── distributions.py# TanhNormal, OneHotCategorical
│       ├── dynamics/
│       │   ├── base.py         # BaseDynamics abstract interface
│       │   ├── rssm.py         # Recurrent State Space Model (DreamerV3)
│       │   ├── jepa.py         # Joint-Embedding Predictive Architecture
│       │   └── rnn_mdn.py      # Classic Ha (2018) World Model
│       ├── algorithms/
│       │   ├── base.py         # BaseAgent abstract interface
│       │   ├── ppo.py          # Proximal Policy Optimisation
│       │   ├── dreamer.py      # DreamerV3
│       │   └── tdmpc.py        # TD-MPC2
│       ├── planners/
│       │   ├── cem.py          # Cross-Entropy Method
│       │   └── mppi.py         # Model Predictive Path Integral
│       └── memory/
│           ├── rollout.py      # Ephemeral buffer for PPO (GAE)
│           └── trajectory.py   # Sequence buffer for world models
└── scripts/
    └── run_slurm.sh            # SLURM job submission template (NTU / A100)
```

---

## Installation

### GPU (CUDA 12)

```bash
pip install -e ".[dev]"
# JAX with CUDA 12 is already listed as a dependency in pyproject.toml.
# Follow https://jax.readthedocs.io/en/latest/installation.html if needed.
```

### CPU-only (testing / development)

```bash
pip install -e ".[dev,cpu]"
```

**Requirements:** Python ≥ 3.10, JAX ≥ 0.4.30, Flax ≥ 0.8.0, Optax ≥ 0.2.0, Gymnasium ≥ 0.29.0, Hydra ≥ 1.3.2, W&B ≥ 0.16.0, einops ≥ 0.7.0.

---

## Quick Start

```bash
# PPO on MuJoCo (default)
python -m helios.main agent=ppo env=mujoco

# DreamerV3 on dm_control
python -m helios.main agent=dreamer_v3 env=dm_control

# TD-MPC2 with a custom learning rate
python -m helios.main agent=tdmpc2 env=mujoco agent.lr=1e-4

# Disable W&B logging
python -m helios.main agent=ppo env=mujoco wandb.mode=disabled
```

The `helios` console script (installed via `pyproject.toml`) is also available:

```bash
helios agent=ppo env=mujoco
```

### SLURM (cluster)

```bash
sbatch scripts/run_slurm.sh ppo mujoco
sbatch scripts/run_slurm.sh dreamer_v3 dm_control seed=123
sbatch scripts/run_slurm.sh tdmpc2 mujoco agent.lr=1e-4
```

Edit the `#SBATCH` directives in `scripts/run_slurm.sh` to match your cluster's partition names and GPU types (the template targets NTU HPC nodes with A100/V100 GPUs).

---

## Configuration

helios-rl uses [Hydra](https://hydra.cc). The root config is `configs/experiment/default.yaml`:

```yaml
defaults:
  - agent: ppo       # swap to dreamer_v3 or tdmpc2
  - env: mujoco
  - _self_

seed: 42
total_steps: 1_000_000
log_interval: 1000
eval_interval: 10_000
eval_episodes: 10
wandb:
  project: helios-rl
  mode: online        # "disabled" to turn off
```

Agent-specific hyperparameters live in `configs/agent/<name>.yaml`. Key knobs:

| Agent | Notable Hyperparameters |
|---|---|
| `ppo` | `num_steps`, `clip_coef`, `vf_coef`, `ent_coef`, `lr`, `anneal_lr` |
| `dreamer_v3` | `rssm.*`, `kl_alpha`, `imagination_horizon`, `train_ratio` |
| `tdmpc2` | `mppi.*`, `latent_dim`, `utd_ratio`, `buffer_size` |

---

## Architecture

### Agent Contract (`algorithms/base.py`)

Every agent implements `BaseAgent`:

```python
class BaseAgent:
    def initial_state(self, key) -> dict:
        """Initialise parameters + optimizer states."""

    def act(self, obs, state, key, deterministic=False) -> (action, hidden):
        """Returns action and next hidden state."""

    def update(self, batch, state) -> (new_state, metrics_dict):
        """Returns updated state and a dict of scalar metrics."""
```

### Dynamics Contract (`dynamics/base.py`)

Every world model implements `BaseDynamics`:

```python
class BaseDynamics:
    def initial_state(self, batch_size) -> dict:
        """Zeroed latent state."""

    def observe(self, obs, prev_state, action, key) -> (new_state, extras):
        """Posterior update: q(s_t | s_{t-1}, a_{t-1}, o_t)"""

    def imagine(self, prev_state, action, key) -> (new_state, extras):
        """Prior prediction: p(s_t | s_{t-1}, a_{t-1})"""
```

### Training Loop (`main.py`)

1. **Initialise** — config (Hydra), PRNG, env, buffer, agent.
2. **Warm-up** — collect *N* random steps to seed the buffer.
3. **Main loop:**
   - **Interact** — `agent.act(obs, state, key)` → `env.step(action)`
   - **Store** — save transition / sequence to buffer
   - **Train** — every `train_freq` steps: `batch = buffer.sample()` → `agent.update(batch, state)`
   - **Log** — ship metrics to stdout and W&B
   - **Evaluate** — periodic deterministic rollouts

---

## Development

```bash
# Lint
ruff check src/
black --check src/

# Type-check
mypy src/

# Tests
pytest
```

---

## Roadmap

- [ ] PPO: validate ≥ 200 reward on `InvertedPendulum-v4`
- [ ] RSSM + DreamerV3: functional Flax implementation
- [ ] TD-MPC2: MPPI planner with `jax.vmap` rollouts
- [ ] `configs/env/` configs for dm_control and Atari
- [ ] Multi-GPU / multi-node support

---

## License

MIT
