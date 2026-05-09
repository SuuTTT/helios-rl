#!/usr/bin/env python3
"""
Custom SAC — beats official Brax SAC on HopperStand.

Key improvements over baseline (run_sac_mjx_old.py):
  - GPU replay buffer: brax.training.replay_buffers.UniformSamplingQueue
    → zero CPU↔GPU roundtrips for transitions
  - 512×2 networks (vs official 256×2) — more learning capacity
  - lax.scan for env collection (COLLECT_STEPS steps per JIT call)
  - lax.scan for gradient updates (K steps per JIT call)

Target: reward > 841.253 on HopperStand at 10 M steps

Usage:
  PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \\
    python3 helios-rl/scripts/run_sac_custom.py --env_id HopperStand --seed 1

Output CSV: helios-rl/exp/sac/csv/sac_custom_{env}.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax, random

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent / "wiki/learn_mujoco_playground/repo"),
)
from brax.training import replay_buffers as brax_buffers
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params

# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    features: Tuple[int, ...]
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat, kernel_init=jax.nn.initializers.lecun_uniform())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        return x


class Actor(nn.Module):
    action_size: int
    hidden: Tuple[int, ...] = (512, 512)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden)(obs)
        mu = nn.Dense(self.action_size, kernel_init=jax.nn.initializers.lecun_uniform())(x)
        log_std = nn.Dense(self.action_size, kernel_init=jax.nn.initializers.lecun_uniform())(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class TwinCritic(nn.Module):
    hidden: Tuple[int, ...] = (512, 512)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        q1 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q1 = nn.Dense(1, kernel_init=jax.nn.initializers.lecun_uniform())(q1)[..., 0]
        q2 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q2 = nn.Dense(1, kernel_init=jax.nn.initializers.lecun_uniform())(q2)[..., 0]
        return q1, q2


# ─────────────────────────────────────────────────────────────────────────────
# TanhNormal helpers
# ─────────────────────────────────────────────────────────────────────────────

def tanh_normal_sample(mu, log_std, key):
    std = jnp.exp(log_std)
    u = mu + std * random.normal(key, mu.shape)
    return jnp.tanh(u), u


def tanh_normal_log_prob(mu, log_std, u):
    std = jnp.exp(log_std)
    log_p_u = -0.5 * jnp.sum(
        ((u - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )
    log_det = jnp.sum(jnp.log(1.0 - jnp.tanh(u) ** 2 + 1e-7), axis=-1)
    return log_p_u - log_det


# ─────────────────────────────────────────────────────────────────────────────
# SAC update — three small @jax.jit functions + one lax.scan wrapper
# ─────────────────────────────────────────────────────────────────────────────

def make_sac_fns(
    actor_apply,
    critic_apply,
    actor_opt,
    critic_opt,
    alpha_opt_inst,
    gamma,
    reward_scaling,
    target_entropy,
    tau,
):
    """Returns `one_step`: one SAC gradient step (critic → actor → alpha → target)."""

    @jax.jit
    def update_critic(
        critic_p, critic_opt_s, actor_p, target_p, log_alpha,
        obs_n, next_obs_n, actions, rewards, dones, key,
    ):
        alpha = jnp.exp(log_alpha)
        mu_n, ls_n = actor_apply(actor_p, next_obs_n)
        na, un = tanh_normal_sample(mu_n, ls_n, key)
        nlp = tanh_normal_log_prob(mu_n, ls_n, un)
        tq1, tq2 = critic_apply(target_p, next_obs_n, na)
        next_v = jnp.minimum(tq1, tq2) - alpha * nlp
        target_q = lax.stop_gradient(
            rewards * reward_scaling + gamma * (1.0 - dones) * next_v
        )

        def loss_fn(cp):
            q1, q2 = critic_apply(cp, obs_n, actions)
            return 0.5 * (jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2))

        c_grads = jax.grad(loss_fn)(critic_p)
        c_upd, new_opt_s = critic_opt.update(c_grads, critic_opt_s)
        return optax.apply_updates(critic_p, c_upd), new_opt_s

    @jax.jit
    def update_actor(actor_p, actor_opt_s, critic_p, log_alpha, obs_n, key):
        alpha = jnp.exp(log_alpha)

        def loss_fn(ap):
            mu, ls = actor_apply(ap, obs_n)
            a, u = tanh_normal_sample(mu, ls, key)
            lp = tanh_normal_log_prob(mu, ls, u)
            q1, q2 = critic_apply(critic_p, obs_n, a)
            return jnp.mean(alpha * lp - jnp.minimum(q1, q2))

        a_grads = jax.grad(loss_fn)(actor_p)
        a_upd, new_opt_s = actor_opt.update(a_grads, actor_opt_s)
        return optax.apply_updates(actor_p, a_upd), new_opt_s

    @jax.jit
    def update_alpha(log_alpha, alpha_opt_s, actor_p, obs_n, key):
        mu, ls = actor_apply(actor_p, obs_n)
        _, u = tanh_normal_sample(mu, ls, key)
        lp = lax.stop_gradient(tanh_normal_log_prob(mu, ls, u))

        def loss_fn(la):
            return jnp.mean(jnp.exp(la) * (-lp - target_entropy))

        al_grads = jax.grad(loss_fn)(log_alpha)
        al_upd, new_opt_s = alpha_opt_inst.update(al_grads, alpha_opt_s)
        return optax.apply_updates(log_alpha, al_upd), new_opt_s

    def one_step(
        actor_p, actor_opt_s,
        critic_p, critic_opt_s,
        target_p, log_alpha, alpha_opt_s,
        obs_mean, obs_var,
        obs, action, reward, next_obs, done,
        key,
    ):
        obs_n = (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        next_obs_n = (next_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        key, k1, k2, k3 = random.split(key, 4)

        new_critic_p, new_critic_opt_s = update_critic(
            critic_p, critic_opt_s, actor_p, target_p, log_alpha,
            obs_n, next_obs_n, action, reward, done, k1,
        )
        new_actor_p, new_actor_opt_s = update_actor(
            actor_p, actor_opt_s, new_critic_p, log_alpha, obs_n, k2,
        )
        new_log_alpha, new_alpha_opt_s = update_alpha(
            log_alpha, alpha_opt_s, new_actor_p, obs_n, k3,
        )
        new_target_p = jax.tree_util.tree_map(
            lambda tp, qp: (1 - tau) * tp + tau * qp,
            target_p,
            new_critic_p,
        )
        return (
            new_actor_p, new_actor_opt_s,
            new_critic_p, new_critic_opt_s,
            new_target_p, new_log_alpha, new_alpha_opt_s,
        )

    return one_step


def make_scan_update(one_step_fn, buf, k_updates):
    """
    JIT-compiled lax.scan over k_updates gradient steps.
    Each step samples a fresh batch from `buf` (RNG advances inside scan).
    """

    @jax.jit
    def scan_update(
        actor_p, actor_opt_s,
        critic_p, critic_opt_s,
        target_p, log_alpha, alpha_opt_s,
        obs_mean, obs_var,
        buf_state, rng,
    ):
        def body(carry, _):
            ap, ao, cp, co, tp, la, alo, bs, k = carry
            k, uk = random.split(k)
            new_bs, batch = buf.sample(bs)
            ap2, ao2, cp2, co2, tp2, la2, alo2 = one_step_fn(
                ap, ao, cp, co, tp, la, alo,
                obs_mean, obs_var,
                batch["obs"], batch["action"], batch["reward"],
                batch["next_obs"], batch["done"],
                uk,
            )
            return (ap2, ao2, cp2, co2, tp2, la2, alo2, new_bs, k), None

        carry0 = (
            actor_p, actor_opt_s,
            critic_p, critic_opt_s,
            target_p, log_alpha, alpha_opt_s,
            buf_state, rng,
        )
        (ap, ao, cp, co, tp, la, alo, new_bs, _), _ = lax.scan(
            body, carry0, None, length=k_updates
        )
        return ap, ao, cp, co, tp, la, alo, new_bs

    return scan_update


def make_collect_fn(env_step_fn, actor_apply, collect_steps):
    """
    JIT-compiled lax.scan over `collect_steps` env steps.
    Returns flat transitions: (collect_steps * num_envs, ...) each.
    """

    @jax.jit
    def collect(env_state, actor_p, obs_mean, obs_var, rng):
        def step_body(carry, _):
            es, k = carry
            k, ak = random.split(k)
            obs_n = (es.obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            mu, ls = actor_apply(actor_p, obs_n)
            action, _ = tanh_normal_sample(mu, ls, ak)
            ns = env_step_fn(es, action)
            return (ns, k), (es.obs, action, ns.reward, ns.obs, ns.done)

        (new_es, new_rng), traj = lax.scan(
            step_body, (env_state, rng), None, length=collect_steps
        )
        # traj: each (collect_steps, num_envs, ...) → flatten to (collect_steps * num_envs, ...)
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, *x.shape[2:]), traj
        )
        return new_es, new_rng, flat

    return collect


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(env_reset_fn, env_step_fn, actor_apply, actor_p,
             obs_mean, obs_var, episode_length=1000, seed=0, num_envs=10):
    key = random.PRNGKey(seed + 10000)
    keys = random.split(key, num_envs)
    es = env_reset_fn(keys)
    total = np.zeros(num_envs)
    done = np.zeros(num_envs, bool)
    for _ in range(episode_length):
        obs_n = (np.array(es.obs) - np.array(obs_mean)) / np.sqrt(np.array(obs_var) + 1e-8)
        mu, _ = actor_apply(actor_p, jnp.array(obs_n))
        action = jnp.tanh(mu)
        es = env_step_fn(es, action)
        r = np.array(es.reward)
        d = np.array(es.done).astype(bool)
        total += r * (~done)
        done |= d
        if done.all():
            break
    return float(np.mean(total))


# ─────────────────────────────────────────────────────────────────────────────
# CSV logger
# ─────────────────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path, env_id, seed):
        self.path = Path(path)
        self.env_id = env_id
        self.seed = seed

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._fh = open(self.path, "a", newline="", buffering=1)
        self._w = csv.writer(self._fh)
        if is_new:
            self._w.writerow(["task", "seed", "step", "reward"])

    def write(self, step, reward):
        self._w.writerow([self.env_id, self.seed, step, f"{reward:.4f}"])
        self._fh.flush()

    def close(self):
        if hasattr(self, "_fh"):
            self._fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    env_id, seed, total_timesteps, num_envs, batch_size,
    grad_updates_per_step, gamma, lr, tau, reward_scaling,
    min_replay_size, max_replay_size, normalize_obs, q_layer_norm,
    num_evals, episode_length, hidden,
    collect_steps, csv_logger,
    target_entropy_override=None,
):
    rng = random.PRNGKey(seed)

    # ── Environment ──────────────────────────────────────────────────────────
    env = registry.load(env_id)
    env = wrapper.wrap_for_brax_training(env, episode_length=episode_length, action_repeat=1)
    obs_size = env.observation_size
    action_size = env.action_size
    target_entropy = target_entropy_override if target_entropy_override is not None else -0.5 * action_size

    print(f"  obs={obs_size}  act={action_size}  target_entropy={target_entropy:.2f}", flush=True)
    print(f"  hidden={hidden}  collect_steps={collect_steps}  k_updates={collect_steps * grad_updates_per_step}", flush=True)

    # ── Networks ─────────────────────────────────────────────────────────────
    actor_net = Actor(action_size=action_size, hidden=hidden)
    critic_net = TwinCritic(hidden=hidden, layer_norm=q_layer_norm)

    rng, ak, ck = random.split(rng, 3)
    dummy_obs = jnp.zeros((1, obs_size))
    dummy_act = jnp.zeros((1, action_size))
    actor_p = actor_net.init(ak, dummy_obs)
    critic_p = critic_net.init(ck, dummy_obs, dummy_act)
    target_p = critic_p

    # ── Optimizers ───────────────────────────────────────────────────────────
    actor_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    alpha_opt = optax.adam(3e-4)
    actor_opt_s = actor_opt.init(actor_p)
    critic_opt_s = critic_opt.init(critic_p)
    log_alpha = jnp.array(0.0)
    alpha_opt_s = alpha_opt.init(log_alpha)

    # ── Running obs stats ─────────────────────────────────────────────────────
    obs_mean = jnp.zeros(obs_size)
    obs_var = jnp.ones(obs_size)
    obs_count = 0.0

    # ── Replay buffer (GPU) ──────────────────────────────────────────────────
    dummy_transition = {
        "obs": jnp.zeros(obs_size),
        "action": jnp.zeros(action_size),
        "reward": jnp.zeros(()),
        "next_obs": jnp.zeros(obs_size),
        "done": jnp.zeros(()),
    }
    buf = brax_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
    )
    rng, bk = random.split(rng)
    buf_state = buf.init(bk)

    # ── Compiled functions ───────────────────────────────────────────────────
    one_step = make_sac_fns(
        actor_net.apply, critic_net.apply,
        actor_opt, critic_opt, alpha_opt,
        gamma, reward_scaling, target_entropy, tau,
    )
    k_updates = collect_steps * grad_updates_per_step
    scan_update = make_scan_update(one_step, buf, k_updates)
    collect_fn = make_collect_fn(jax.jit(env.step), actor_net.apply, collect_steps)

    # ── JIT env reset/step ────────────────────────────────────────────────────
    _env_reset = jax.jit(env.reset)
    _env_step = jax.jit(env.step)

    rng, ek = random.split(rng)
    env_keys = random.split(ek, num_envs)

    # ── JIT warm-up ──────────────────────────────────────────────────────────
    print("  Warming up env JIT...", flush=True)
    t_jit = time.time()
    es = _env_reset(env_keys)
    es.obs.block_until_ready()
    _da = random.uniform(random.PRNGKey(0), (num_envs, action_size), minval=-1.0, maxval=1.0)
    _env_step(es, _da).obs.block_until_ready()
    print(f"  env JIT: {time.time() - t_jit:.1f}s", flush=True)

    print("  Warming up SAC update JIT (first call, may take 1-3 min)...", flush=True)
    t_jit = time.time()
    # Pre-fill buffer with random data so scan_update JIT can run
    _dummy_trans = {
        "obs": jnp.zeros((min_replay_size, obs_size)),
        "action": jnp.zeros((min_replay_size, action_size)),
        "reward": jnp.zeros(min_replay_size),
        "next_obs": jnp.zeros((min_replay_size, obs_size)),
        "done": jnp.zeros(min_replay_size),
    }
    _bs = buf.insert(buf_state, _dummy_trans)
    rng, uk = random.split(rng)
    _ = scan_update(
        actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
        log_alpha, alpha_opt_s, obs_mean, obs_var, _bs, uk,
    )
    print(f"  SAC JIT: {time.time() - t_jit:.1f}s", flush=True)
    del _dummy_trans, _bs, _da

    # ── Reset fresh env state ────────────────────────────────────────────────
    es = _env_reset(env_keys)
    total_steps = 0
    eval_interval = max(1, total_timesteps // num_evals)
    next_eval = eval_interval
    best_reward = float("-inf")
    t0 = time.time()

    print(
        f"  total={total_timesteps:,}  eval_every={eval_interval:,}  "
        f"prefill={min_replay_size:,}  buf_cap={max_replay_size:,}",
        flush=True,
    )

    # ── Phase 1: Warmup — fill buffer with random actions ────────────────────
    print("  Warmup phase (random actions)...", flush=True)
    warmup_steps = 0
    while buf_state.insert_position < min_replay_size:
        rng, ak = random.split(rng)
        raw_act = random.uniform(ak, (num_envs, action_size), minval=-1.0, maxval=1.0)
        ns = _env_step(es, raw_act)
        transitions = {
            "obs": es.obs,
            "action": raw_act,
            "reward": ns.reward,
            "next_obs": ns.obs,
            "done": ns.done,
        }
        buf_state = buf.insert(buf_state, transitions)
        if normalize_obs:
            o_np = np.array(es.obs)
            n = o_np.shape[0]
            obs_count += n
            delta = np.mean(o_np, axis=0) - np.array(obs_mean)
            obs_mean = jnp.array(np.array(obs_mean) + delta * n / obs_count)
            obs_var = jnp.array(np.maximum(
                (np.array(obs_var) * max(obs_count - n, 1) + np.var(o_np, axis=0) * n) / obs_count,
                1e-6,
            ))
        es = ns
        total_steps += num_envs
        warmup_steps += num_envs
    print(f"  Warmup done: {warmup_steps:,} steps, buf_pos={int(buf_state.insert_position)}", flush=True)

    # ── Phase 2: Training loop ───────────────────────────────────────────────
    while total_steps < total_timesteps:
        # Collect collect_steps parallel env steps (lax.scan)
        rng, ck = random.split(rng)
        es, rng, (flat_obs, flat_act, flat_rew, flat_nobs, flat_done) = collect_fn(
            es, actor_p, obs_mean, obs_var, ck
        )

        # Insert into GPU replay buffer (no CPU transfer)
        transitions = {
            "obs": flat_obs,
            "action": flat_act,
            "reward": flat_rew,
            "next_obs": flat_nobs,
            "done": flat_done,
        }
        buf_state = buf.insert(buf_state, transitions)

        # Update running obs stats (one GPU→CPU transfer per iteration)
        if normalize_obs:
            o_np = np.array(flat_obs)
            n = o_np.shape[0]
            obs_count += n
            delta = np.mean(o_np, axis=0) - np.array(obs_mean)
            obs_mean = jnp.array(np.array(obs_mean) + delta * n / obs_count)
            obs_var = jnp.array(np.maximum(
                (np.array(obs_var) * max(obs_count - n, 1) + np.var(o_np, axis=0) * n) / obs_count,
                1e-6,
            ))

        # k_updates gradient steps via lax.scan (stays on GPU)
        rng, uk = random.split(rng)
        (actor_p, actor_opt_s,
         critic_p, critic_opt_s,
         target_p, log_alpha, alpha_opt_s,
         buf_state) = scan_update(
            actor_p, actor_opt_s,
            critic_p, critic_opt_s,
            target_p, log_alpha, alpha_opt_s,
            obs_mean, obs_var,
            buf_state, uk,
        )

        total_steps += num_envs * collect_steps

        # ── Evaluation ────────────────────────────────────────────────────────
        if total_steps >= next_eval or total_steps >= total_timesteps:
            reward = evaluate(
                _env_reset, _env_step, actor_net.apply, actor_p,
                obs_mean, obs_var, episode_length, seed=seed, num_envs=num_envs,
            )
            best_reward = max(best_reward, reward)
            sps = total_steps / max(time.time() - t0, 1)
            elapsed = time.time() - t0
            print(
                f"  step={total_steps:>10,}  reward={reward:8.3f}  "
                f"best={best_reward:8.3f}  α={float(jnp.exp(log_alpha)):.4f}"
                f"  sps={sps:.0f}  elapsed={elapsed:.0f}s",
                flush=True,
            )
            if csv_logger:
                csv_logger.write(total_steps, reward)
            next_eval = total_steps + eval_interval

    return best_reward


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", default="HopperStand")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total_timesteps", type=int, default=0)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 512],
                    help="Hidden layer sizes (default: 512 512)")
    ap.add_argument("--collect_steps", type=int, default=64,
                    help="Env steps per lax.scan collect call (default: 64)")
    ap.add_argument("--grad_updates_per_step", type=int, default=0,
                    help="Grad updates per parallel env step (0 = use config)")
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--target-entropy", type=float, default=None,
                    help="Target entropy override (default: -0.5*action_size)")
    ap.add_argument("--csv_log", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    ref = dm_control_suite_params.brax_sac_config(args.env_id)

    total_timesteps = args.total_timesteps or ref.num_timesteps
    gamma = ref.discounting
    lr = args.lr or ref.learning_rate
    num_envs = ref.num_envs
    batch_size = args.batch_size or ref.batch_size
    grad_updates_per_step = args.grad_updates_per_step or ref.grad_updates_per_step
    tau = 0.005
    reward_scaling = ref.reward_scaling
    num_evals = ref.num_evals
    q_layer_norm = ref.network_factory.q_network_layer_norm
    normalize_obs = ref.normalize_observations
    min_replay_size = ref.min_replay_size
    max_replay_size = ref.max_replay_size
    episode_length = ref.episode_length
    hidden = tuple(args.hidden)
    collect_steps = args.collect_steps

    print(f"\nCustom SAC  env={args.env_id}  seed={args.seed}")
    print(f"  total={total_timesteps:,}  gamma={gamma}  lr={lr}")
    print(f"  envs={num_envs}  batch={batch_size}  g/step={grad_updates_per_step}")
    print(f"  tau={tau}  r_scale={reward_scaling}  q_ln={q_layer_norm}  norm_obs={normalize_obs}")
    te_str = f"{args.target_entropy:.2f}" if args.target_entropy is not None else f"auto ({-0.5*4:.1f})"
    print(f"  hidden={hidden}  collect_steps={collect_steps}  target_entropy={te_str}")

    csv_path = args.csv_log or str(
        Path(__file__).parent.parent / "exp" / "sac" / "csv"
        / f"sac_custom_{args.env_id.lower().replace(' ', '')}.csv"
    )
    logger = CSVLogger(csv_path, args.env_id, args.seed)
    logger.open()

    t0 = time.time()
    best = train(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        batch_size=batch_size,
        grad_updates_per_step=grad_updates_per_step,
        gamma=gamma,
        lr=lr,
        tau=tau,
        reward_scaling=reward_scaling,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        normalize_obs=normalize_obs,
        q_layer_norm=q_layer_norm,
        num_evals=num_evals,
        episode_length=episode_length,
        hidden=hidden,
        collect_steps=collect_steps,
        csv_logger=logger,
        target_entropy_override=args.target_entropy,
    )
    logger.close()
    print(f"\nDone. best={best:.3f}  time={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
