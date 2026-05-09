#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) for MuJoCo Playground DMC Suite envs.
JAX + Flax implementation — no pmap, single-GPU, JIT-compiled rollouts.

Algorithm: SAC with automatic entropy tuning (Haarnoja et al. 2018 v2)
  - Twin critics with LayerNorm (reduces overestimation)
  - TanhNormal actor distribution (correct log-prob for bounded actions)
  - Automatic temperature α via a separate optimizer
  - Off-policy replay buffer (circular, uniform sampling)

Reference config from mujoco_playground.config.dm_control_suite_params.brax_sac_config

Usage:
  PYTHONPATH=/workspace/wiki/learn_mujoco_playground/repo \
    python3 helios-rl/scripts/run_sac_mjx.py --env_id BallInCup --seed 1

Output CSV: helios-rl/exp/sac/csv/sac_{env}.csv  (task,seed,step,reward)
"""

import argparse
import csv
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "wiki/learn_mujoco_playground/repo"))
from mujoco_playground import registry, wrapper
from mujoco_playground.config import dm_control_suite_params

# ── Networks ──────────────────────────────────────────────────────────────────

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
    hidden: Tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden)(obs)
        mu      = nn.Dense(self.action_size, kernel_init=jax.nn.initializers.lecun_uniform())(x)
        log_std = nn.Dense(self.action_size, kernel_init=jax.nn.initializers.lecun_uniform())(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std


class TwinCritic(nn.Module):
    hidden: Tuple[int, ...] = (256, 256)
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs, action):
        x  = jnp.concatenate([obs, action], axis=-1)
        q1 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q1 = nn.Dense(1, kernel_init=jax.nn.initializers.lecun_uniform())(q1)[..., 0]
        q2 = MLP(self.hidden, layer_norm=self.layer_norm)(x)
        q2 = nn.Dense(1, kernel_init=jax.nn.initializers.lecun_uniform())(q2)[..., 0]
        return q1, q2


# ── TanhNormal helpers ────────────────────────────────────────────────────────

def tanh_normal_sample(mu, log_std, key):
    std = jnp.exp(log_std)
    u   = mu + std * jax.random.normal(key, mu.shape)
    return jnp.tanh(u), u


def tanh_normal_log_prob(mu, log_std, u):
    std     = jnp.exp(log_std)
    log_p_u = -0.5 * jnp.sum(((u - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    log_det = jnp.sum(jnp.log(1.0 - jnp.tanh(u) ** 2 + 1e-7), axis=-1)
    return log_p_u - log_det


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, obs_size, action_size):
        self.cap  = capacity
        self.ptr  = 0
        self.size = 0
        self.obs     = np.zeros((capacity, obs_size),    np.float32)
        self.actions = np.zeros((capacity, action_size), np.float32)
        self.rewards = np.zeros(capacity,                np.float32)
        self.nobs    = np.zeros((capacity, obs_size),    np.float32)
        self.dones   = np.zeros(capacity,                np.float32)

    def add(self, obs, act, rew, nobs, done):
        n   = len(obs)
        idx = np.arange(self.ptr, self.ptr + n) % self.cap
        self.obs[idx]     = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.nobs[idx]    = nobs
        self.dones[idx]   = done
        self.ptr  = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (jnp.array(self.obs[idx]), jnp.array(self.actions[idx]),
                jnp.array(self.rewards[idx]), jnp.array(self.nobs[idx]),
                jnp.array(self.dones[idx]))


# ── SAC update — three small JIT functions (fast compilation) ─────────────────
# One monolithic JIT traces critic grads + actor grads in one pass → huge graph.
# Three separate JITs each compile in <10s on first call.

def make_sac_update(actor_apply, critic_apply,
                    actor_opt, critic_opt, alpha_opt_inst,
                    gamma, reward_scaling, target_entropy, tau):
    """Returns a callable that does one SAC gradient step (3 JITs, fast)."""

    @jax.jit
    def update_critic(critic_p, critic_opt_s,
                      actor_p, target_p, log_alpha,
                      obs_n, next_obs_n, actions, rewards, dones, key):
        alpha = jnp.exp(log_alpha)
        mu_n, ls_n = actor_apply(actor_p, next_obs_n)
        na, un     = tanh_normal_sample(mu_n, ls_n, key)
        nlp        = tanh_normal_log_prob(mu_n, ls_n, un)
        tq1, tq2   = critic_apply(target_p, next_obs_n, na)
        next_v     = jnp.minimum(tq1, tq2) - alpha * nlp
        target_q   = jax.lax.stop_gradient(
            rewards * reward_scaling + gamma * (1.0 - dones) * next_v)

        def loss_fn(cp):
            q1, q2 = critic_apply(cp, obs_n, actions)
            return 0.5 * (jnp.mean((q1 - target_q)**2) + jnp.mean((q2 - target_q)**2))

        c_loss, c_grads = jax.value_and_grad(loss_fn)(critic_p)
        c_upd, new_opt_s = critic_opt.update(c_grads, critic_opt_s)
        return optax.apply_updates(critic_p, c_upd), new_opt_s, c_loss

    @jax.jit
    def update_actor(actor_p, actor_opt_s,
                     critic_p, log_alpha, obs_n, key):
        alpha = jnp.exp(log_alpha)

        def loss_fn(ap):
            mu, ls = actor_apply(ap, obs_n)
            a, u   = tanh_normal_sample(mu, ls, key)
            lp     = tanh_normal_log_prob(mu, ls, u)
            q1, q2 = critic_apply(critic_p, obs_n, a)
            return jnp.mean(alpha * lp - jnp.minimum(q1, q2))

        a_loss, a_grads = jax.value_and_grad(loss_fn)(actor_p)
        a_upd, new_opt_s = actor_opt.update(a_grads, actor_opt_s)
        return optax.apply_updates(actor_p, a_upd), new_opt_s, a_loss

    @jax.jit
    def update_alpha(log_alpha, alpha_opt_s, actor_p, obs_n, key):
        mu, ls = actor_apply(actor_p, obs_n)
        _, u   = tanh_normal_sample(mu, ls, key)
        lp     = jax.lax.stop_gradient(tanh_normal_log_prob(mu, ls, u))

        def loss_fn(la):
            return jnp.mean(jnp.exp(la) * (-lp - target_entropy))

        al_loss, al_grads = jax.value_and_grad(loss_fn)(log_alpha)
        al_upd, new_opt_s = alpha_opt_inst.update(al_grads, alpha_opt_s)
        return optax.apply_updates(log_alpha, al_upd), new_opt_s, al_loss

    def sac_update(actor_p, actor_opt_s,
                   critic_p, critic_opt_s,
                   target_p, log_alpha, alpha_opt_s,
                   obs_mean, obs_var,
                   obs, actions, rewards, next_obs, dones,
                   key):
        obs_n      = (obs      - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        next_obs_n = (next_obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
        key, k1, k2, k3 = jax.random.split(key, 4)

        new_critic_p, new_critic_opt_s, c_loss = update_critic(
            critic_p, critic_opt_s, actor_p, target_p, log_alpha,
            obs_n, next_obs_n, actions, rewards, dones, k1)

        new_actor_p, new_actor_opt_s, a_loss = update_actor(
            actor_p, actor_opt_s, new_critic_p, log_alpha, obs_n, k2)

        new_log_alpha, new_alpha_opt_s, al_loss = update_alpha(
            log_alpha, alpha_opt_s, new_actor_p, obs_n, k3)

        new_target_p = jax.tree_util.tree_map(
            lambda tp, qp: (1 - tau) * tp + tau * qp, target_p, new_critic_p)

        return (new_actor_p, new_actor_opt_s,
                new_critic_p, new_critic_opt_s,
                new_target_p, new_log_alpha, new_alpha_opt_s,
                c_loss, a_loss, al_loss)

    return sac_update


def make_batched_sac_update(sac_update_fn, k):
    """k gradient updates in one JIT call via lax.scan — no Python loop overhead."""

    @jax.jit
    def batched(actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                log_alpha, alpha_opt_s, obs_mean, obs_var,
                obs_b, act_b, rew_b, nobs_b, done_b, key):
        # obs_b / act_b / ... have leading axis k

        def body(carry, xs):
            ap, ao, cp, co, tp, la, alo, rng = carry
            obs_s, act_s, rew_s, nobs_s, done_s = xs
            rng, k_sub = jax.random.split(rng)
            out = sac_update_fn(ap, ao, cp, co, tp, la, alo,
                                obs_mean, obs_var,
                                obs_s, act_s, rew_s, nobs_s, done_s, k_sub)
            ap2, ao2, cp2, co2, tp2, la2, alo2 = out[:7]
            return (ap2, ao2, cp2, co2, tp2, la2, alo2, rng), None

        carry0 = (actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                  log_alpha, alpha_opt_s, key)
        xs = (obs_b, act_b, rew_b, nobs_b, done_b)
        (ap, ao, cp, co, tp, la, alo, _), _ = jax.lax.scan(body, carry0, xs)
        return ap, ao, cp, co, tp, la, alo

    return batched


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(env_reset_fn, env_step_fn, actor_net, actor_p, obs_mean, obs_var,
             episode_length=1000, seed=0, num_envs=10):
    """Evaluate using pre-JITted env functions to avoid re-compilation."""
    key  = jax.random.PRNGKey(seed + 10000)
    keys = jax.random.split(key, num_envs)
    es   = env_reset_fn(keys)
    total = np.zeros(num_envs)
    done  = np.zeros(num_envs, bool)
    for _ in range(episode_length):
        obs_n  = (np.array(es.obs) - np.array(obs_mean)) / np.sqrt(np.array(obs_var) + 1e-8)
        mu, _  = actor_net.apply(actor_p, jnp.array(obs_n))
        action = jnp.tanh(mu)
        es     = env_step_fn(es, action)
        r      = np.array(es.reward)
        d      = np.array(es.done).astype(bool)
        total += r * (~done)
        done  |= d
        if done.all():
            break
    return float(np.mean(total))


# ── CSV logger ────────────────────────────────────────────────────────────────

class CSVLogger:
    def __init__(self, path, env_id, seed):
        self.path = Path(path)
        self.env_id = env_id
        self.seed   = seed

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.path.exists()
        self._fh = open(self.path, "a", newline="", buffering=1)
        self._w  = csv.writer(self._fh)
        if is_new:
            self._w.writerow(["task", "seed", "step", "reward"])

    def write(self, step, reward):
        self._w.writerow([self.env_id, self.seed, step, f"{reward:.4f}"])
        self._fh.flush()

    def close(self):
        if hasattr(self, "_fh"):
            self._fh.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def train(env_id, seed, total_timesteps, num_envs, batch_size,
          grad_updates_per_step, gamma, lr, tau, reward_scaling,
          min_replay_size, max_replay_size, normalize_obs, q_layer_norm,
          num_evals, episode_length, csv_logger, progress_fn=None):

    rng    = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)

    env = registry.load(env_id)
    env = wrapper.wrap_for_brax_training(env, episode_length=episode_length, action_repeat=1)
    obs_size    = env.observation_size
    action_size = env.action_size
    target_entropy = -0.5 * action_size

    print(f"  obs={obs_size}  act={action_size}  target_entropy={target_entropy:.2f}")

    actor_net  = Actor(action_size=action_size)
    critic_net = TwinCritic(layer_norm=q_layer_norm)

    rng, ak, ck = jax.random.split(rng, 3)
    dummy_obs   = jnp.zeros((1, obs_size))
    dummy_act   = jnp.zeros((1, action_size))
    actor_p     = actor_net.init(ak, dummy_obs)
    critic_p    = critic_net.init(ck, dummy_obs, dummy_act)
    target_p    = critic_p

    actor_opt  = optax.adam(lr)
    critic_opt = optax.adam(lr)
    alpha_opt  = optax.adam(3e-4)
    actor_opt_s  = actor_opt.init(actor_p)
    critic_opt_s = critic_opt.init(critic_p)
    log_alpha    = jnp.array(0.0)
    alpha_opt_s  = alpha_opt.init(log_alpha)

    obs_mean  = jnp.zeros(obs_size)
    obs_var   = jnp.ones(obs_size)
    obs_count = 0.0

    # Build the JIT-compiled update fn (optimizers closed over)
    _sac_update = make_sac_update(
        actor_net.apply, critic_net.apply,
        actor_opt, critic_opt, alpha_opt,
        gamma, reward_scaling, target_entropy, tau)

    buf = ReplayBuffer(max_replay_size, obs_size, action_size)

    rng, ek = jax.random.split(rng)
    env_keys = jax.random.split(ek, num_envs)

    # ── Pre-warm JIT (env + SAC update) ──────────────────────────────────
    print(f"  Pre-warming env JIT...", flush=True)
    t_jit = time.time()
    _env_reset = jax.jit(env.reset)
    _env_step  = jax.jit(env.step)
    es = _env_reset(env_keys)
    es.obs.block_until_ready()
    _dummy_act = jax.random.uniform(jax.random.PRNGKey(0), (num_envs, action_size), minval=-1., maxval=1.)
    _env_step(es, _dummy_act).obs.block_until_ready()
    print(f"  env JIT: {time.time()-t_jit:.1f}s", flush=True)

    print(f"  Pre-warming SAC update JIT...", flush=True)
    t_jit = time.time()
    _db = jnp.zeros((batch_size, obs_size))
    _db = jnp.zeros((batch_size, obs_size))
    _da = jnp.zeros((batch_size, action_size))
    _dr = jnp.zeros(batch_size)
    _sac_update(actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                log_alpha, alpha_opt_s, obs_mean, obs_var,
                _db, _da, _dr, _db, _dr, jax.random.PRNGKey(1))
    print(f"  SAC JIT: {time.time()-t_jit:.1f}s", flush=True)
    del _db, _da, _dr, _dummy_act

    # Re-init env state (clean)
    es = _env_reset(env_keys)

    total_steps   = 0
    eval_interval = max(1, total_timesteps // num_evals)
    next_eval     = eval_interval
    best_reward   = float("-inf")
    t0            = time.time()

    print(f"  total={total_timesteps:,}  eval_every={eval_interval:,}  prefill={min_replay_size:,}")

    while total_steps < total_timesteps:
        # ── Collect transitions ────────────────────────────────────────────
        rng, ak2 = jax.random.split(rng)
        if buf.size < min_replay_size:
            raw_act = jax.random.uniform(ak2, (num_envs, action_size), minval=-1.0, maxval=1.0)
        else:
            obs_n   = (np.array(es.obs) - np.array(obs_mean)) / np.sqrt(np.array(obs_var) + 1e-8)
            mu, ls  = actor_net.apply(actor_p, jnp.array(obs_n))
            raw_act, _ = tanh_normal_sample(mu, ls, ak2)

        ns = _env_step(es, jnp.array(raw_act))

        o_np  = np.array(es.obs)
        a_np  = np.array(raw_act)
        r_np  = np.array(ns.reward)
        no_np = np.array(ns.obs)
        d_np  = np.array(ns.done)

        buf.add(o_np, a_np, r_np, no_np, d_np)

        if normalize_obs:
            n      = o_np.shape[0]
            obs_count += n
            delta     = np.mean(o_np, axis=0) - np.array(obs_mean)
            obs_mean  = jnp.array(np.array(obs_mean) + delta * n / obs_count)
            obs_var   = jnp.array(np.maximum(
                (np.array(obs_var) * (obs_count - n) + np.var(o_np, axis=0) * n) / obs_count,
                1e-6))

        es          = ns
        total_steps += num_envs

        # ── Gradient updates ───────────────────────────────────────────────
        if buf.size >= min_replay_size:
            for _ in range(grad_updates_per_step):
                batch = buf.sample(batch_size, np_rng)
                rng, uk = jax.random.split(rng)
                (actor_p, actor_opt_s,
                 critic_p, critic_opt_s,
                 target_p, log_alpha, alpha_opt_s,
                 c_loss, a_loss, al_loss) = _sac_update(
                    actor_p, actor_opt_s,
                    critic_p, critic_opt_s,
                    target_p, log_alpha, alpha_opt_s,
                    obs_mean, obs_var,
                    *batch, uk)

        # ── Evaluation ────────────────────────────────────────────────────
        if total_steps >= next_eval or total_steps >= total_timesteps:
            reward = evaluate(_env_reset, _env_step, actor_net, actor_p, obs_mean, obs_var,
                              episode_length, seed=seed, num_envs=num_envs)
            best_reward = max(best_reward, reward)
            sps = total_steps / max(time.time() - t0, 1)
            print(f"  step={total_steps:>10,}  reward={reward:8.3f}  "
                  f"best={best_reward:8.3f}  α={float(jnp.exp(log_alpha)):.4f}  sps={sps:.0f}")
            if csv_logger:
                csv_logger.write(total_steps, reward)
            if progress_fn:
                progress_fn(total_steps, {"eval/episode_reward": reward})
            next_eval = total_steps + eval_interval

    return best_reward


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id",                default="BallInCup")
    ap.add_argument("--seed",                  type=int,   default=1)
    ap.add_argument("--total_timesteps",       type=int,   default=0)
    ap.add_argument("--gamma",                 type=float, default=0.0)
    ap.add_argument("--lr",                    type=float, default=0.0)
    ap.add_argument("--num_envs",              type=int,   default=0)
    ap.add_argument("--batch_size",            type=int,   default=0)
    ap.add_argument("--grad_updates_per_step", type=int,   default=0)
    ap.add_argument("--tau",                   type=float, default=0.0)
    ap.add_argument("--reward_scaling",        type=float, default=0.0)
    ap.add_argument("--num_evals",             type=int,   default=0)
    ap.add_argument("--csv_log",               default="")
    ap.add_argument("--exp_name",              default="")   # ignored, for compat
    return ap.parse_args()


def main():
    args = parse_args()
    ref  = dm_control_suite_params.brax_sac_config(args.env_id)

    total_timesteps     = args.total_timesteps        or ref.num_timesteps
    gamma               = args.gamma                  or ref.discounting
    lr                  = args.lr                     or ref.learning_rate
    num_envs            = args.num_envs               or ref.num_envs
    batch_size          = args.batch_size             or ref.batch_size
    grad_updates_per_step = args.grad_updates_per_step or ref.grad_updates_per_step
    tau                 = args.tau                    or 0.005
    reward_scaling      = args.reward_scaling         or ref.reward_scaling
    num_evals           = args.num_evals              or ref.num_evals
    q_layer_norm        = ref.network_factory.q_network_layer_norm
    normalize_obs       = ref.normalize_observations
    min_replay_size     = ref.min_replay_size
    max_replay_size     = ref.max_replay_size
    episode_length      = ref.episode_length

    print(f"\nSAC  env={args.env_id}  seed={args.seed}")
    print(f"  total={total_timesteps:,}  gamma={gamma}  lr={lr}")
    print(f"  envs={num_envs}  batch={batch_size}  g/step={grad_updates_per_step}")
    print(f"  tau={tau}  r_scale={reward_scaling}  q_ln={q_layer_norm}")

    csv_path = args.csv_log or str(
        Path(__file__).parent.parent / "exp" / "sac" / "csv"
        / f"sac_{args.env_id.lower().replace(' ', '')}.csv")
    logger = CSVLogger(csv_path, args.env_id, args.seed)
    logger.open()

    t0 = time.time()
    best = train(
        env_id=args.env_id, seed=args.seed,
        total_timesteps=total_timesteps, num_envs=num_envs,
        batch_size=batch_size, grad_updates_per_step=grad_updates_per_step,
        gamma=gamma, lr=lr, tau=tau, reward_scaling=reward_scaling,
        min_replay_size=min_replay_size, max_replay_size=max_replay_size,
        normalize_obs=normalize_obs, q_layer_norm=q_layer_norm,
        num_evals=num_evals, episode_length=episode_length,
        csv_logger=logger)
    logger.close()
    print(f"\nDone. Best={best:.3f}  Time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
