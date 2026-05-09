#!/usr/bin/env python3
"""
SAC (Soft Actor-Critic) for MuJoCo Playground DMC Suite envs.
JAX + Flax — single GPU, scan-based rollout + scan-based gradient updates.

Speed design:
  COLLECT_STEPS env steps are gathered in ONE lax.scan JIT call (no Python per step).
  K * COLLECT_STEPS gradient updates are run in ONE lax.scan JIT call.
  Python loop only triggers ~(total / COLLECT_STEPS / num_envs) times total.

Algorithm: SAC with automatic entropy tuning (Haarnoja et al. 2018 v2)
  Twin critics (LayerNorm), TanhNormal actor, auto-alpha.

Output CSV: helios-rl/exp/sac/csv/sac_{env}.csv  (task,seed,step,reward)
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


# ── Numpy replay buffer ───────────────────────────────────────────────────────

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

    def add_batch(self, obs, act, rew, nobs, done):
        n   = len(obs)
        idx = np.arange(self.ptr, self.ptr + n) % self.cap
        self.obs[idx]     = obs
        self.actions[idx] = act
        self.rewards[idx] = rew
        self.nobs[idx]    = nobs
        self.dones[idx]   = done
        self.ptr  = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample_bulk(self, n_batches, batch_size, rng):
        """Sample n_batches x batch_size in one alloc; convert to JAX once."""
        total = n_batches * batch_size
        idx   = rng.integers(0, self.size, size=total)
        return (jnp.array(self.obs[idx].reshape(n_batches, batch_size, -1)),
                jnp.array(self.actions[idx].reshape(n_batches, batch_size, -1)),
                jnp.array(self.rewards[idx].reshape(n_batches, batch_size)),
                jnp.array(self.nobs[idx].reshape(n_batches, batch_size, -1)),
                jnp.array(self.dones[idx].reshape(n_batches, batch_size)))


# ── SAC gradient scan ─────────────────────────────────────────────────────────

def make_sac_fns(actor_apply, critic_apply,
                 actor_opt, critic_opt, alpha_opt_inst,
                 gamma, reward_scaling, target_entropy, tau):

    def _step(carry, xs):
        """One SAC gradient step — lax.scan body (no inner @jax.jit)."""
        actor_p, actor_opt_s, critic_p, critic_opt_s, target_p, log_alpha, alpha_opt_s, key = carry
        obs_n, act_b, rew_b, nobs_n, done_b = xs

        key, k1, k2, k3 = jax.random.split(key, 4)
        alpha = jnp.exp(log_alpha)

        # critic
        mu_n, ls_n = actor_apply(actor_p, nobs_n)
        na, un     = tanh_normal_sample(mu_n, ls_n, k1)
        nlp        = tanh_normal_log_prob(mu_n, ls_n, un)
        tq1, tq2   = critic_apply(target_p, nobs_n, na)
        next_v     = jnp.minimum(tq1, tq2) - alpha * nlp
        target_q   = jax.lax.stop_gradient(
            rew_b * reward_scaling + gamma * (1.0 - done_b) * next_v)

        def closs(cp):
            q1, q2 = critic_apply(cp, obs_n, act_b)
            return 0.5 * (jnp.mean((q1 - target_q)**2) + jnp.mean((q2 - target_q)**2))

        cg = jax.grad(closs)(critic_p)
        cu, new_co = critic_opt.update(cg, critic_opt_s)
        new_cp = optax.apply_updates(critic_p, cu)

        # actor
        def aloss(ap):
            mu, ls = actor_apply(ap, obs_n)
            a, u   = tanh_normal_sample(mu, ls, k2)
            lp     = tanh_normal_log_prob(mu, ls, u)
            q1, q2 = critic_apply(new_cp, obs_n, a)
            return jnp.mean(alpha * lp - jnp.minimum(q1, q2))

        ag = jax.grad(aloss)(actor_p)
        au, new_ao = actor_opt.update(ag, actor_opt_s)
        new_ap = optax.apply_updates(actor_p, au)

        # alpha
        mu2, ls2 = actor_apply(new_ap, obs_n)
        _, u2    = tanh_normal_sample(mu2, ls2, k3)
        lp2      = jax.lax.stop_gradient(tanh_normal_log_prob(mu2, ls2, u2))

        def ealoss(la):
            return jnp.mean(jnp.exp(la) * (-lp2 - target_entropy))

        elg = jax.grad(ealoss)(log_alpha)
        elu, new_alo = alpha_opt_inst.update(elg, alpha_opt_s)
        new_la = optax.apply_updates(log_alpha, elu)

        # soft target
        new_tp = jax.tree_util.tree_map(
            lambda t, q: (1 - tau) * t + tau * q, target_p, new_cp)

        return (new_ap, new_ao, new_cp, new_co, new_tp, new_la, new_alo, key), None

    @jax.jit
    def scan_updates(actor_p, actor_opt_s, critic_p, critic_opt_s,
                     target_p, log_alpha, alpha_opt_s,
                     obs_stack, act_stack, rew_stack, nobs_stack, done_stack,
                     key):
        """Run all gradient updates for one epoch in one JIT call."""
        carry0 = (actor_p, actor_opt_s, critic_p, critic_opt_s,
                  target_p, log_alpha, alpha_opt_s, key)
        (ap, ao, cp, co, tp, la, alo, _), _ = jax.lax.scan(
            _step, carry0, (obs_stack, act_stack, rew_stack, nobs_stack, done_stack))
        return ap, ao, cp, co, tp, la, alo

    return scan_updates


# ── Collection scan ───────────────────────────────────────────────────────────

def make_collect_fn(env_step_fn, actor_apply, collect_steps):

    @jax.jit
    def collect(env_state, actor_p, obs_mean, obs_var, key, use_random):
        def body(carry, _):
            es, key = carry
            key, ak, rk = jax.random.split(key, 3)
            obs_n  = (es.obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)
            mu, ls = actor_apply(actor_p, obs_n)
            act_pol, _ = tanh_normal_sample(mu, ls, ak)
            act_rnd    = jax.random.uniform(rk, act_pol.shape, minval=-1., maxval=1.)
            act = jnp.where(use_random, act_rnd, act_pol)
            ns  = env_step_fn(es, act)
            return (ns, key), (es.obs, act, ns.reward, ns.obs, ns.done)

        (new_es, _), trs = jax.lax.scan(body, (env_state, key), None, length=collect_steps)
        return new_es, trs   # trs: tuple of (collect_steps, num_envs, ...)

    return collect


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(env_reset_fn, env_step_fn, actor_net, actor_p, obs_mean, obs_var,
             episode_length=1000, seed=0, num_envs=10):
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
        self.path   = Path(path)
        self.env_id = env_id
        self.seed   = seed

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new   = not self.path.exists()
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


# ── Train ─────────────────────────────────────────────────────────────────────

COLLECT_STEPS = 128   # env steps collected per Python loop iteration


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
    actor_p  = actor_net.init(ak,  jnp.zeros((1, obs_size)))
    critic_p = critic_net.init(ck, jnp.zeros((1, obs_size)), jnp.zeros((1, action_size)))
    target_p = critic_p

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

    buf = ReplayBuffer(max_replay_size, obs_size, action_size)

    scan_updates = make_sac_fns(
        actor_net.apply, critic_net.apply,
        actor_opt, critic_opt, alpha_opt,
        gamma, reward_scaling, target_entropy, tau)

    rng, ek = jax.random.split(rng)
    env_keys = jax.random.split(ek, num_envs)

    print("  Pre-warming env JIT...", flush=True)
    tj = time.time()
    _env_reset = jax.jit(env.reset)
    _env_step  = jax.jit(env.step)
    es = _env_reset(env_keys)
    es.obs.block_until_ready()
    _env_step(es, jnp.zeros((num_envs, action_size))).obs.block_until_ready()
    print(f"  env JIT: {time.time()-tj:.1f}s", flush=True)

    _collect = make_collect_fn(_env_step, actor_net.apply, COLLECT_STEPS)

    print("  Pre-warming collect JIT...", flush=True)
    tj = time.time()
    _collect(es, actor_p, obs_mean, obs_var, jax.random.PRNGKey(0), True)
    jax.effects_barrier()
    print(f"  collect JIT: {time.time()-tj:.1f}s", flush=True)

    n_updates = COLLECT_STEPS * grad_updates_per_step
    print(f"  Pre-warming grad-scan JIT (n_updates={n_updates})...", flush=True)
    tj = time.time()
    _db = jnp.zeros((n_updates, batch_size, obs_size))
    _da = jnp.zeros((n_updates, batch_size, action_size))
    _dr = jnp.zeros((n_updates, batch_size))
    scan_updates(actor_p, actor_opt_s, critic_p, critic_opt_s,
                 target_p, log_alpha, alpha_opt_s,
                 _db, _da, _dr, _db, _dr, jax.random.PRNGKey(1))
    jax.effects_barrier()
    del _db, _da, _dr
    print(f"  grad-scan JIT: {time.time()-tj:.1f}s", flush=True)

    es = _env_reset(env_keys)

    transitions_per_iter = COLLECT_STEPS * num_envs
    total_steps   = 0
    eval_interval = max(1, total_timesteps // num_evals)
    next_eval     = eval_interval
    best_reward   = float("-inf")
    t0 = time.time()
    print(f"  total={total_timesteps:,}  eval_every={eval_interval:,}  "
          f"prefill={min_replay_size:,}  collect={COLLECT_STEPS}  n_grad={n_updates}",
          flush=True)

    while total_steps < total_timesteps:
        # ── Collect COLLECT_STEPS steps in one JIT ────────────────────────
        rng, ck2 = jax.random.split(rng)
        use_random = bool(buf.size < min_replay_size)

        es, (obs_t, act_t, rew_t, nobs_t, done_t) = _collect(
            es, actor_p, obs_mean, obs_var, ck2, use_random)

        obs_np  = np.array(obs_t).reshape(-1, obs_size)
        act_np  = np.array(act_t).reshape(-1, action_size)
        rew_np  = np.array(rew_t).reshape(-1)
        nobs_np = np.array(nobs_t).reshape(-1, obs_size)
        done_np = np.array(done_t).reshape(-1)
        buf.add_batch(obs_np, act_np, rew_np, nobs_np, done_np)

        if normalize_obs:
            n          = obs_np.shape[0]
            obs_count += n
            delta      = np.mean(obs_np, 0) - np.array(obs_mean)
            obs_mean   = jnp.array(np.array(obs_mean) + delta * n / obs_count)
            obs_var    = jnp.array(np.maximum(
                (np.array(obs_var) * (obs_count - n) + np.var(obs_np, 0) * n) / obs_count, 1e-6))

        total_steps += transitions_per_iter

        # ── Gradient updates: all in one scan JIT ─────────────────────────
        if buf.size >= min_replay_size:
            rng, uk = jax.random.split(rng)
            om_np = np.array(obs_mean)
            os_np = np.sqrt(np.array(obs_var) + 1e-8)
            obs_s, act_s, rew_s, nobs_s, done_s = buf.sample_bulk(
                n_updates, batch_size, np_rng)
            # Normalize on GPU
            obs_s  = (obs_s  - jnp.array(om_np)) / jnp.array(os_np)
            nobs_s = (nobs_s - jnp.array(om_np)) / jnp.array(os_np)

            (actor_p, actor_opt_s, critic_p, critic_opt_s,
             target_p, log_alpha, alpha_opt_s) = scan_updates(
                actor_p, actor_opt_s, critic_p, critic_opt_s,
                target_p, log_alpha, alpha_opt_s,
                obs_s, act_s, rew_s, nobs_s, done_s, uk)

        # ── Evaluation ────────────────────────────────────────────────────
        if total_steps >= next_eval or total_steps >= total_timesteps:
            reward = evaluate(_env_reset, _env_step, actor_net, actor_p,
                              obs_mean, obs_var, episode_length, seed=seed,
                              num_envs=num_envs)
            best_reward = max(best_reward, reward)
            sps = total_steps / max(time.time() - t0, 1)
            print(f"  step={total_steps:>10,}  reward={reward:8.3f}  "
                  f"best={best_reward:8.3f}  α={float(jnp.exp(log_alpha)):.4f}  sps={sps:.0f}",
                  flush=True)
            if csv_logger:
                csv_logger.write(total_steps, reward)
            if progress_fn:
                progress_fn(total_steps, {"eval/episode_reward": reward})
            next_eval = total_steps + eval_interval

    return best_reward


# ── Args ──────────────────────────────────────────────────────────────────────

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
    ap.add_argument("--exp_name",              default="")
    return ap.parse_args()


def main():
    args = parse_args()
    ref  = dm_control_suite_params.brax_sac_config(args.env_id)

    total_timesteps       = args.total_timesteps       or ref.num_timesteps
    gamma                 = args.gamma                 or ref.discounting
    lr                    = args.lr                    or ref.learning_rate
    num_envs              = args.num_envs              or ref.num_envs
    batch_size            = args.batch_size            or ref.batch_size
    grad_updates_per_step = args.grad_updates_per_step or ref.grad_updates_per_step
    tau                   = args.tau                   or 0.005
    reward_scaling        = args.reward_scaling        or ref.reward_scaling
    num_evals             = args.num_evals             or ref.num_evals
    q_layer_norm          = ref.network_factory.q_network_layer_norm
    normalize_obs         = ref.normalize_observations
    min_replay_size       = ref.min_replay_size
    max_replay_size       = ref.max_replay_size
    episode_length        = ref.episode_length

    print(f"\nSAC  env={args.env_id}  seed={args.seed}")
    print(f"  total={total_timesteps:,}  gamma={gamma}  lr={lr}")
    print(f"  envs={num_envs}  batch={batch_size}  g/step={grad_updates_per_step}")
    print(f"  tau={tau}  r_scale={reward_scaling}  q_ln={q_layer_norm}")

    csv_path = args.csv_log or str(
        Path(__file__).parent.parent / "exp" / "sac" / "csv"
        / f"sac_{args.env_id.lower().replace(' ', '')}.csv")
    logger = CSVLogger(csv_path, args.env_id, args.seed)
    logger.open()

    t0   = time.time()
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
