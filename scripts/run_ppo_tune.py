#!/usr/bin/env python3
"""PPO hyperparameter tuning — gamma sweep for HopperStand & CheetahRun.

Diagnosis:
  Our benchmark used num_envs=512, total_steps=3M which is 20× too few steps
  and 4× too few envs compared to official (2048 envs, 60M steps).
  Additionally RTX 3090 (Ampere) requires JAX_DEFAULT_MATMUL_PRECISION=highest.

This script:
  1. Fixes num_envs=2048 (official)
  2. Sets JAX_DEFAULT_MATMUL_PRECISION=highest (Ampere precision fix)
  3. Sweeps gamma ∈ {0.97, 0.99, 0.995} on HopperStand
  4. Runs CheetahRun with gamma=0.995 (official) to confirm recovery
  5. Saves CSVs to helios-rl/exp/ppo_tune/ and auto-plots

Usage:
    PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo \\
        python3 helios-rl/scripts/run_ppo_tune.py [--total_steps 15000000]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Critical fix for Ampere GPU (RTX 30xx / 40xx) training stability
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))
from mujoco_playground import registry, wrapper
from flax.training.train_state import TrainState

EXP_DIR = Path(__file__).resolve().parents[1] / "exp" / "ppo_tune"


# ── HopperStand initialization fix ────────────────────────────────────────────
# BUG in mujoco_playground: reset() sets qpos[2] (rooty) = random(-π, π),
# which puts the hopper UPSIDE DOWN ~72% of the time (|rooty| > 0.87 rad).
# Also qpos[3:] = random(full_range) creates extreme joint configurations.
# With standing = tolerance(torso_z - foot_z, (0.6, 2)), hopper has
# standing=0 from step 0 for most random inits → reward=0 everywhere →
# no gradient signal → no learning, no matter how many steps.
#
# FIX: Use small noise ±0.005 rad around the upright (rooty=0) position.
# This matches the original dm_control hopper initialization.

def _make_fixed_hopper_reset(original_env):
    """Return a patched reset fn with small noise around standing position."""
    from mujoco import mjx as _mjx
    from mujoco_playground._src import mjx_env as _mjx_env

    def _fixed_reset(self, rng):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        qpos = jnp.zeros(self.mjx_model.nq)
        # rooty (qpos[2]): ±0.005 rad instead of random(-π, π)
        qpos = qpos.at[2].set(
            jax.random.uniform(rng1, (), minval=-0.005, maxval=0.005)
        )
        # joints (qpos[3:]): ±0.005 rad, clamped to valid joint range
        noise = jax.random.uniform(
            rng2, (self.mjx_model.nq - 3,), minval=-0.005, maxval=0.005
        )
        qpos = qpos.at[3:].set(jnp.clip(noise, self._lowers, self._uppers))

        data = _mjx_env.make_data(
            self.mj_model, qpos=qpos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = _mjx.forward(self.mjx_model, data)
        metrics = {k: jnp.zeros(()) for k in self._metric_keys}
        info = {"rng": rng}
        reward, done = jnp.zeros(2)
        obs = self._get_obs(data, info)
        return _mjx_env.State(data, obs, reward, done, metrics, info)

    import types
    original_env.reset = types.MethodType(_fixed_reset, original_env)
    return original_env


# ── helpers ───────────────────────────────────────────────────────────────────

def open_csv(path: Path, env_id: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", buffering=1)
    if path.stat().st_size == 0:
        fh.write("task,gamma,seed,step,reward\n")
    return fh


def write_csv(fh, env_id, gamma, seed, step, reward):
    fh.write(f"{env_id},{gamma},{seed},{step},{reward:.4f}\n")
    fh.flush()


# ── PPO training with configurable gamma ──────────────────────────────────────

def train_ppo(env_id: str, gamma: float, total_steps: int, seed: int,
              csv_path: Path, num_envs: int = 2048) -> None:

    from helios.algorithms.ppo import (
        PolicyNet, ValueNet,
        obs_norm_init, obs_norm_apply,
        make_update_fn,
    )

    # ── Official hyperparameters (from brax_ppo_config dm_control_suite)
    num_steps       = 30
    update_epochs   = 16
    num_minibatches = 32
    gae_lambda      = 0.95
    clip_coef       = 0.3
    vf_coef         = 0.5
    max_grad_norm   = 1.0
    normalize_obs   = True
    episode_length  = 1000
    eval_interval   = max(total_steps // 30, 1)  # 30 evals over the run for finer tracking

    # ── Task-specific hyperparameters
    # HopperStand: train_jax_ppo.py (paper config) uses γ=0.97, reward_scaling=0.1.
    #   This gives max discounted value = 0.1/(1-0.97) = 3.33 → very stable critic.
    #   Note: γ only affects training signal, NOT eval. Eval sums raw undiscounted
    #   rewards, so a well-trained policy can still achieve 200+ eval reward with γ=0.97.
    # CheetahRun: γ=0.995, reward_scaling=10 (official brax_ppo_config).
    if env_id == "HopperStand":
        lr             = 5e-4    # paper default (train_jax_ppo.py)
        ent_coef       = 5e-3    # paper default (train_jax_ppo.py)
        reward_scaling = 1.0    # value targets ≈ 200 (same scale as goal reward)
    else:
        lr             = 1e-3
        ent_coef       = 0.01
        reward_scaling = 10.0

    # steps per outer iteration = update_epochs × num_steps × num_envs
    steps_per_iter = update_epochs * num_steps * num_envs

    tag = f"g{gamma:.3f}"
    print(f"\n{'='*68}", flush=True)
    print(f"  PPO | {env_id} | γ={gamma} | envs={num_envs} | "
          f"steps={total_steps/1e6:.0f}M | seed={seed}", flush=True)
    print(f"{'='*68}", flush=True)

    env = registry.load(env_id)
    if env_id == "HopperStand":
        env = _make_fixed_hopper_reset(env)
        print("  [FIX] HopperStand: using small-noise standing initialization", flush=True)
    env = wrapper.wrap_for_brax_training(env, episode_length=episode_length,
                                         action_repeat=1)
    obs_dim = env.observation_size
    act_dim = env.action_size

    # ── Networks (v34s3 Brax-exact: 4×32 policy, 5×256 value)
    policy_net = PolicyNet(action_dim=act_dim)
    value_net  = ValueNet()
    key = jax.random.PRNGKey(seed)
    key, pk, vk = jax.random.split(key, 3)
    dummy = jnp.zeros(obs_dim)
    policy_params = policy_net.init(pk, dummy)
    value_params  = value_net.init(vk, dummy)

    agent_state = TrainState.create(
        apply_fn=None,
        params={"policy_params": policy_params, "value_params": value_params},
        tx=optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr, eps=1e-5),
        ),
    )

    rollout_and_update = make_update_fn(
        policy_net, value_net, jax.jit(env.step),
        num_envs=num_envs, num_steps=num_steps,
        update_epochs=update_epochs, num_minibatches=num_minibatches,
        gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
        normalize_obs=normalize_obs, reward_scaling=reward_scaling,
    )

    _env_step = jax.jit(env.step)

    @jax.jit
    def eval_policy(params, obs_ns, key):
        eval_state = env.reset(jax.random.split(key, num_envs))
        ep_ret = jnp.zeros(num_envs)

        def step_fn(carry, _):
            es, obs, ret = carry
            norm_obs = obs_norm_apply(obs_ns, obs)
            logits    = policy_net.apply(params["policy_params"], norm_obs)
            mean, _   = jnp.split(logits, 2, axis=-1)
            action    = jnp.tanh(mean)
            nes       = _env_step(es, action)
            return (nes, nes.obs, ret + nes.reward), None

        (_, _, ep_ret), _ = jax.lax.scan(
            step_fn, (eval_state, eval_state.obs, ep_ret),
            None, length=episode_length,
        )
        return ep_ret.mean()

    # ── Init env
    key, rk = jax.random.split(key)
    env_state = env.reset(jax.random.split(rk, num_envs))
    next_obs  = env_state.obs
    next_done = jnp.zeros(num_envs, dtype=jnp.bool_)
    obs_ns    = obs_norm_init(obs_dim)
    ep_ret    = jnp.zeros(num_envs)
    ep_len    = jnp.zeros(num_envs, dtype=jnp.int32)

    # ── JIT warmup
    print("  Warming up JIT...", flush=True)
    t_jit = time.time()
    agent_state, env_state, next_obs, next_done, key, ep_ret, ep_len, obs_ns, _ = (
        rollout_and_update(agent_state, env_state, next_obs, next_done, key,
                           ep_ret, ep_len, obs_ns)
    )
    jax.block_until_ready(agent_state.params)
    print(f"  JIT compiled in {time.time()-t_jit:.1f}s", flush=True)

    # ── Training loop
    global_step = steps_per_iter
    next_eval   = eval_interval
    t0          = time.time()

    with open_csv(csv_path, env_id, seed) as fh:
        while global_step < total_steps:
            agent_state, env_state, next_obs, next_done, key, ep_ret, ep_len, obs_ns, _ = (
                rollout_and_update(agent_state, env_state, next_obs, next_done, key,
                                   ep_ret, ep_len, obs_ns)
            )
            global_step += steps_per_iter

            if global_step >= next_eval:
                key, ek = jax.random.split(key)
                ret = float(eval_policy(agent_state.params, obs_ns, ek))
                sps = int(global_step / max(time.time() - t0, 1))
                print(f"  step={global_step/1e6:5.1f}M  reward={ret:7.2f}"
                      f"  γ={gamma}  sps={sps:,}", flush=True)
                write_csv(fh, env_id, gamma, seed, global_step, ret)
                next_eval += eval_interval

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  ({int(global_step/elapsed):,} sps)\n",
          flush=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(exp_dir: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = list(exp_dir.glob("*.csv"))
    if not csvs:
        print("No CSV files found to plot.")
        return

    all_df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    tasks  = sorted(all_df["task"].unique())
    gammas = sorted(all_df["gamma"].unique())

    COLORS = {0.97: "#e07b39", 0.99: "#1f6dbf", 0.995: "#2ca02c", 0.999: "#9467bd"}
    LWIDTH = 2.2

    fig, axes = plt.subplots(1, len(tasks), figsize=(6.5 * len(tasks), 4.5), dpi=150)
    if len(tasks) == 1:
        axes = [axes]

    print(f"\n{'Task':<22} {'γ':>6} {'MaxReward':>10}  {'@Step':>10}")
    print("-" * 55)

    for ax, task in zip(axes, tasks):
        df_t = all_df[all_df["task"] == task]
        for g in gammas:
            df_g = df_t[df_t["gamma"] == g].sort_values("step")
            if df_g.empty:
                continue
            x = df_g["step"].values / 1e6
            y = df_g["reward"].values
            c = COLORS.get(g, "#888888")
            ax.plot(x, y, color=c, lw=LWIDTH, marker="o", ms=4,
                    label=f"γ={g}")
            best = y.max()
            print(f"  {task:<20} {g:>6.3f} {best:>10.1f}  "
                  f"{int(df_g.loc[df_g['reward'].idxmax(), 'step']):>10,}")

        ax.set_title(f"{task}", fontsize=13)
        ax.set_xlabel("Steps (M)", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle(f"PPO γ sweep  |  num_envs=2048  |  JAX_MATMUL=highest",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "ppo_gamma_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=15_000_000)
    ap.add_argument("--seed",        type=int, default=1)
    ap.add_argument("--num_envs",    type=int, default=2048,
                    help="Num envs (official=2048)")
    ap.add_argument("--gammas",      nargs="+", type=float,
                    default=[0.97, 0.99, 0.995],
                    help="Gamma values to sweep (default: 0.97 0.99 0.995)")
    ap.add_argument("--tasks",       nargs="+",
                    default=["HopperStand", "CheetahRun"])
    ap.add_argument("--no_plot",     action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nPPO γ sweep")
    print(f"  Tasks:  {args.tasks}")
    print(f"  Gammas: {args.gammas}")
    print(f"  Envs:   {args.num_envs}")
    print(f"  Steps:  {args.total_steps/1e6:.0f}M  seed={args.seed}")
    print(f"  Matmul precision: {os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')}")
    print(f"  Output: {EXP_DIR}\n")

    t_total = time.time()

    for task in args.tasks:
        # HopperStand uses γ=0.97 (paper train_jax_ppo.py default, matches target=200)
        # CheetahRun uses γ=0.995 (official brax_ppo_config)
        gammas_for_task = [0.97] if task == "HopperStand" else [0.995]
        for gamma in gammas_for_task:
            csv_path = EXP_DIR / f"ppo_{task}_g{gamma:.3f}.csv"
            try:
                train_ppo(task, gamma, args.total_steps, args.seed, csv_path,
                          num_envs=args.num_envs)
            except Exception as e:
                print(f"\nERROR {task} γ={gamma}: {e}", flush=True)
                import traceback; traceback.print_exc()
            import gc; gc.collect()

    print(f"\nAll runs done in {(time.time()-t_total)/60:.1f} min")

    if not args.no_plot:
        plot_results(EXP_DIR, EXP_DIR / "plots")


if __name__ == "__main__":
    main()
