#!/usr/bin/env python3
"""Benchmark PPO, SAC, and TD-MPC2 on 3 diverse MuJoCo Playground tasks.

Uses helios.algorithms library modules throughout.

Outputs:
    helios-rl/exp/benchmark/ppo_<task>.csv
    helios-rl/exp/benchmark/sac_<task>.csv
    helios-rl/exp/benchmark/tdmpc2_<task>.csv
    (CSV format: task,seed,step,reward)

Usage:
    PYTHONPATH=/workspace/helios-rl/src:/workspace/wiki/learn_mujoco_playground/repo \\
        python3 helios-rl/scripts/run_benchmark.py [--total_steps 3000000] [--seed 1]
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "wiki/learn_mujoco_playground/repo"))
from mujoco_playground import registry, wrapper

EXP_DIR = Path(__file__).resolve().parents[1] / "exp" / "benchmark"

TASKS = ["CartpoleBalance", "HopperStand", "CheetahRun"]


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_csv(path: Path, env_id: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fh = open(path, "a", buffering=1)
    if is_new:
        fh.write("task,seed,step,reward\n")
    return fh


def write_csv(fh, env_id: str, seed: int, step: int, reward: float):
    fh.write(f"{env_id},{seed},{step},{reward:.4f}\n")
    fh.flush()


def save_pickle_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def buffer_state(buf) -> dict:
    """Return a replay-buffer snapshot suitable for exact off-policy resume."""
    return {
        "cap": buf.cap,
        "N": buf.N,
        "T": buf.T,
        "obs": buf.obs,
        "acts": buf.acts,
        "rews": buf.rews,
        "done": buf.done,
        "ptr": buf.ptr,
        "size": buf.size,
    }


def restore_buffer_state(buf, state: dict) -> None:
    """Restore a replay-buffer snapshot into an existing buffer object."""
    buf.cap = int(state["cap"])
    buf.N = int(state["N"])
    buf.T = int(state["T"])
    buf.obs = state["obs"]
    buf.acts = state["acts"]
    buf.rews = state["rews"]
    buf.done = state["done"]
    buf.ptr = state["ptr"]
    buf.size = state["size"]


# ─────────────────────────────────────────────────────────────────────────────
# PPO (v34s3 architecture from helios.algorithms.ppo)
# ─────────────────────────────────────────────────────────────────────────────

def train_ppo(env_id: str, total_steps: int, seed: int, csv_path: Path) -> None:
    """Train PPO (v34s3 Brax-exact) using helios.algorithms.ppo."""
    print(f"\n{'='*60}", flush=True)
    print(f"  PPO | {env_id} | seed={seed} | steps={total_steps:,}", flush=True)
    print(f"{'='*60}", flush=True)

    from flax.training.train_state import TrainState
    from helios.algorithms.ppo import (
        PolicyNet, ValueNet,
        obs_norm_init, obs_norm_apply, obs_norm_update,
        make_update_fn,
    )

    # ── Hyperparams (Brax-exact milestone, num_envs reduced for benchmark speed)
    num_envs        = 512
    num_steps       = 30
    update_epochs   = 16
    num_minibatches = 32
    lr              = 1e-3
    gamma           = 0.995
    gae_lambda      = 0.95
    clip_coef       = 0.3
    vf_coef         = 0.5
    ent_coef        = 0.01
    max_grad_norm   = 1.0
    reward_scaling  = 10.0
    normalize_obs   = True
    episode_length  = 1000
    eval_interval   = max(total_steps // 12, 1)

    steps_per_iter = update_epochs * num_steps * num_envs  # ≈ 246K

    # ── Environment
    force_jax_tasks = {
        task.strip()
        for task in os.environ.get("TDMPC_GLASS_FORCE_JAX_TASKS", "FishSwim").split(",")
        if task.strip()
    }
    config_overrides = {"impl": "jax"} if use_glass and env_id in force_jax_tasks else None
    if config_overrides:
        print(f"  using env config overrides: {config_overrides}", flush=True)
    env      = registry.load(env_id, config_overrides=config_overrides)
    env      = wrapper.wrap_for_brax_training(env, episode_length=episode_length,
                                              action_repeat=1)
    obs_dim  = env.observation_size
    act_dim  = env.action_size

    # ── Networks
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

    # ── Compiled update fn from library
    rollout_and_update = make_update_fn(
        policy_net, value_net,
        jax.jit(env.step),
        num_envs=num_envs, num_steps=num_steps,
        update_epochs=update_epochs, num_minibatches=num_minibatches,
        gamma=gamma, gae_lambda=gae_lambda,
        clip_coef=clip_coef, vf_coef=vf_coef, ent_coef=ent_coef,
        normalize_obs=normalize_obs, reward_scaling=reward_scaling,
    )

    # ── Eval (deterministic mean action)
    _env_step = jax.jit(env.step)

    @jax.jit
    def eval_policy(params, obs_ns, key):
        eval_state = env.reset(jax.random.split(key, num_envs))
        ep_ret = jnp.zeros(num_envs)

        def step_fn(carry, _):
            es, obs, ep_ret = carry
            norm_obs = obs_norm_apply(obs_ns, obs)
            logits = policy_net.apply(params["policy_params"], norm_obs)
            mean, _ = jnp.split(logits, 2, axis=-1)
            action = jnp.tanh(mean)
            nes = _env_step(es, action)
            return (nes, nes.obs, ep_ret + nes.reward), None

        (_, _, ep_ret), _ = jax.lax.scan(
            step_fn, (eval_state, eval_state.obs, ep_ret), None, length=episode_length
        )
        return ep_ret.mean()

    # ── Init env state
    key, rk = jax.random.split(key)
    env_state = env.reset(jax.random.split(rk, num_envs))
    next_obs  = env_state.obs
    next_done = jnp.zeros(num_envs, dtype=jnp.bool_)
    obs_ns    = obs_norm_init(obs_dim)
    ep_ret    = jnp.zeros(num_envs)
    ep_len    = jnp.zeros(num_envs, dtype=jnp.int32)

    # ── Warmup JIT
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
    t0 = time.time()

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
                print(f"  step={global_step:>9,}  reward={ret:7.2f}  sps={sps:,}", flush=True)
                write_csv(fh, env_id, seed, global_step, ret)
                next_eval += eval_interval

    print(f"  PPO {env_id} done in {time.time()-t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# SAC (custom v1 from helios.algorithms.sac)
# ─────────────────────────────────────────────────────────────────────────────

def train_sac(env_id: str, total_steps: int, seed: int, csv_path: Path) -> None:
    """Train SAC (custom v1) using helios.algorithms.sac."""
    print(f"\n{'='*60}", flush=True)
    print(f"  SAC | {env_id} | seed={seed} | steps={total_steps:,}", flush=True)
    print(f"{'='*60}", flush=True)

    from brax.training import replay_buffers as brax_buffers
    from helios.algorithms.sac import (
        Actor, TwinCritic,
        make_sac_fns, make_scan_update, make_collect_fn,
        tanh_normal_sample,
    )

    # ── Hyperparams (official brax SAC reference where applicable)
    hidden              = (512, 512)
    lr                  = 3e-4
    alpha_lr            = 3e-4
    gamma               = 0.99
    tau                 = 0.005
    reward_scaling      = 1.0
    normalize_obs       = True
    num_envs            = 32
    collect_steps       = 64
    grad_updates_ratio  = 2          # gradient updates per env step
    k_updates           = collect_steps * grad_updates_ratio
    batch_size          = 256
    min_replay_size     = 10_000
    max_replay_size     = 300_000
    episode_length      = 1000
    eval_interval       = max(total_steps // 12, 1)

    # ── Environment
    env     = registry.load(env_id)
    env     = wrapper.wrap_for_brax_training(env, episode_length=episode_length,
                                             action_repeat=1)
    obs_dim = env.observation_size
    act_dim = env.action_size
    target_entropy = -0.5 * act_dim
    print(f"  obs={obs_dim}  act={act_dim}  target_entropy={target_entropy:.2f}", flush=True)

    rng = jax.random.PRNGKey(seed)

    # ── Networks
    actor_net  = Actor(action_size=act_dim, hidden=hidden)
    critic_net = TwinCritic(hidden=hidden, layer_norm=True)

    rng, ak, ck = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, act_dim))
    actor_p   = actor_net.init(ak, dummy_obs)
    critic_p  = critic_net.init(ck, dummy_obs, dummy_act)
    target_p  = critic_p

    # ── Optimizers
    actor_opt  = optax.adam(lr)
    critic_opt = optax.adam(lr)
    alpha_opt  = optax.adam(alpha_lr)
    actor_opt_s  = actor_opt.init(actor_p)
    critic_opt_s = critic_opt.init(critic_p)
    log_alpha    = jnp.array(0.0)
    alpha_opt_s  = alpha_opt.init(log_alpha)

    # ── Running obs stats
    obs_mean  = jnp.zeros(obs_dim)
    obs_var   = jnp.ones(obs_dim)
    obs_count = 0.0

    # ── Replay buffer (GPU)
    dummy_transition = {
        "obs":      jnp.zeros(obs_dim),
        "action":   jnp.zeros(act_dim),
        "reward":   jnp.zeros(()),
        "next_obs": jnp.zeros(obs_dim),
        "done":     jnp.zeros(()),
    }
    buf = brax_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
    )
    rng, bk = jax.random.split(rng)
    buf_state = buf.init(bk)

    # ── SAC functions from library
    one_step    = make_sac_fns(
        actor_net.apply, critic_net.apply,
        actor_opt, critic_opt, alpha_opt,
        gamma, reward_scaling, target_entropy, tau,
    )
    scan_update = make_scan_update(one_step, buf, k_updates)
    collect_fn  = make_collect_fn(jax.jit(env.step), actor_net.apply, collect_steps)

    # ── Env reset
    _env_reset = jax.jit(env.reset)
    _env_step  = jax.jit(env.step)
    rng, ek = jax.random.split(rng)
    env_keys = jax.random.split(ek, num_envs)

    # ── Warmup: fill buffer with random actions
    print("  Filling replay buffer...", flush=True)
    t_pre = time.time()
    env_state = _env_reset(env_keys)
    total_env_steps = 0

    while buf_state.insert_position < min_replay_size:
        rng, ak2 = jax.random.split(rng)
        raw_act  = jax.random.uniform(ak2, (num_envs, act_dim), minval=-1.0, maxval=1.0)
        ns = _env_step(env_state, raw_act)
        transitions = {
            "obs":      env_state.obs,
            "action":   raw_act,
            "reward":   ns.reward,
            "next_obs": ns.obs,
            "done":     ns.done,
        }
        buf_state = buf.insert(buf_state, transitions)
        if normalize_obs:
            o_np = np.array(env_state.obs)
            n = o_np.shape[0]
            obs_count += n
            delta   = np.mean(o_np, axis=0) - np.array(obs_mean)
            obs_mean = jnp.array(np.array(obs_mean) + delta * n / obs_count)
            obs_var  = jnp.array(np.maximum(
                (np.array(obs_var) * max(obs_count - n, 1) + np.var(o_np, axis=0) * n) / obs_count,
                1e-6,
            ))
        env_state = ns
        total_env_steps += num_envs
    print(f"  Replay filled: {int(buf_state.insert_position):,} transitions in {time.time()-t_pre:.1f}s", flush=True)

    # ── Warmup JIT for scan_update
    print("  Warming up scan_update JIT (may take 1-2 min)...", flush=True)
    t_jit = time.time()
    rng, uk = jax.random.split(rng)
    ap2, ao2, cp2, co2, tp2, la2, alo2, bs2 = scan_update(
        actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
        log_alpha, alpha_opt_s, obs_mean, obs_var, buf_state, uk,
    )
    jax.block_until_ready(ap2)
    actor_p, actor_opt_s, critic_p, critic_opt_s, target_p, log_alpha, alpha_opt_s, buf_state = (
        ap2, ao2, cp2, co2, tp2, la2, alo2, bs2
    )
    print(f"  scan_update JIT in {time.time()-t_jit:.1f}s", flush=True)

    # ── Warmup collect JIT
    rng, ck2 = jax.random.split(rng)
    env_state, rng, _ = collect_fn(env_state, actor_p, obs_mean, obs_var, ck2)
    total_env_steps += num_envs * collect_steps

    # ── Eval
    def evaluate():
        import numpy as _np
        key_e = jax.random.PRNGKey(seed + 99999)
        keys_e = jax.random.split(key_e, num_envs)
        es_e = _env_reset(keys_e)
        total = _np.zeros(num_envs)
        done  = _np.zeros(num_envs, bool)
        for _ in range(episode_length):
            obs_n = (_np.array(es_e.obs) - _np.array(obs_mean)) / _np.sqrt(_np.array(obs_var) + 1e-8)
            mu, _ = actor_net.apply(actor_p, jnp.array(obs_n))
            action = jnp.tanh(mu)
            es_e = _env_step(es_e, action)
            r = _np.array(es_e.reward)
            d = _np.array(es_e.done).astype(bool)
            total += r * (~done)
            done  |= d
            if done.all():
                break
        return float(_np.mean(total))

    # ── Training loop
    next_eval = eval_interval
    t0 = time.time()

    with open_csv(csv_path, env_id, seed) as fh:
        while total_env_steps < total_steps:
            # Collect
            rng, ck3 = jax.random.split(rng)
            env_state, rng, (flat_obs, flat_act, flat_rew, flat_nobs, flat_done) = collect_fn(
                env_state, actor_p, obs_mean, obs_var, ck3
            )
            transitions = {
                "obs":      flat_obs, "action":   flat_act,
                "reward":   flat_rew, "next_obs": flat_nobs,
                "done":     flat_done,
            }
            buf_state = buf.insert(buf_state, transitions)

            # Update obs stats
            if normalize_obs:
                o_np = np.array(flat_obs)
                n = o_np.shape[0]
                obs_count += n
                delta   = np.mean(o_np, axis=0) - np.array(obs_mean)
                obs_mean = jnp.array(np.array(obs_mean) + delta * n / obs_count)
                obs_var  = jnp.array(np.maximum(
                    (np.array(obs_var) * max(obs_count - n, 1) + np.var(o_np, axis=0) * n) / obs_count,
                    1e-6,
                ))

            # Gradient updates (lax.scan on GPU)
            rng, uk2 = jax.random.split(rng)
            (actor_p, actor_opt_s,
             critic_p, critic_opt_s,
             target_p, log_alpha, alpha_opt_s,
             buf_state) = scan_update(
                actor_p, actor_opt_s, critic_p, critic_opt_s, target_p,
                log_alpha, alpha_opt_s, obs_mean, obs_var, buf_state, uk2,
            )

            total_env_steps += num_envs * collect_steps

            if total_env_steps >= next_eval:
                ret = evaluate()
                sps = int(total_env_steps / max(time.time() - t0, 1))
                print(f"  step={total_env_steps:>9,}  reward={ret:7.2f}  "
                      f"α={float(jnp.exp(log_alpha)):.4f}  sps={sps:,}", flush=True)
                write_csv(fh, env_id, seed, total_env_steps, ret)
                next_eval += eval_interval

    print(f"  SAC {env_id} done in {time.time()-t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# TD-MPC2 (v24 from helios.algorithms.tdmpc2)
# ─────────────────────────────────────────────────────────────────────────────

def train_tdmpc2(
    env_id: str,
    total_steps: int,
    seed: int,
    csv_path: Path,
    use_glass: bool = False,
    resume_checkpoint: str | None = None,
    save_full_state: bool = False,
    glass_overrides: dict | None = None,
    act_noise_start: float | None = None,
    act_noise_end: float | None = None,
    act_noise_anneal_steps: int = 1_000_000,
    mppi_horizon: int | None = None,
    mppi_n_samples: int | None = None,
    k_update: int | None = None,
    q_reset_steps: list[int] | None = None,
    latent_action_smooth_coef: float = 0.0,
    consistency_coef: float | None = None,
    early_stop_patience: int = 0,
    latent_smooth_warmup_env_steps: int = 0,
    glass_decay_steps: int = 0,
    expl_until: int | None = None,
    expl_mix_decay_steps: int = 0,
    knee_penalty_coef: float = 0.0,
    knee_penalty_threshold: float = 0.15,
    cluster_intrinsic_coef: float = 0.0,
    cluster_intrinsic_window: int = 20,
    cluster_intrinsic_decay_steps: int = 0,
    # iter-6 §7.C — Phase-r2 gait penalty bundle
    gait_fall_penalty: float = 0.0,
    gait_fall_height: float = 0.45,
    gait_action_smooth: float = 0.0,
    # iter-6 §7.B — Phase-r1 soft-reward bundle
    soft_stand_bonus: float = 0.0,
    soft_stand_floor: float = 0.4,
    soft_anneal_steps: int = 0,
    # iter-7 §2.1 — Phase-ar auto-restart on plateau (basin-lottery escape)
    restart_on_plateau: bool = False,
    restart_check_at: int = 1_000_000,
    restart_threshold: float = 100.0,
    restart_max_attempts: int = 3,
    mpc_distill_coef: float = 0.0,
    mpc_distill_anneal_steps: int = 3_000_000,
    mpc_distill_disable_gap: float = 100.0,
    mpc_distill_batch_size: int = 16,
    controller_arbitration: str = "none",
    arbitration_margin: float = 0.0,
) -> None:
    """Train TD-MPC2 or TD-MPC-Glass."""
    algo_name = "TD-MPC-Glass" if use_glass else "TD-MPC2"
    print(f"\n{'='*60}", flush=True)
    print(f"  {algo_name} | {env_id} | seed={seed} | steps={total_steps:,}", flush=True)
    print(f"{'='*60}", flush=True)

    if use_glass:
        from helios.algorithms.tdmpc_glass import (
            Encoder, Dynamics, RewardHead, QEnsemble, Pi,
            MultiEnvBuffer, make_update_fn, make_mppi_fn, make_glass_diag_fn,
            init_glass_params, DEFAULTS,
        )
    else:
        from helios.algorithms.tdmpc2 import (
            Encoder, Dynamics, RewardHead, QEnsemble, Pi,
            MultiEnvBuffer, make_update_fn, make_mppi_fn, DEFAULTS,
        )

    # ── Hyperparams (v24 milestone)
    d          = dict(DEFAULTS)
    latent_dim = d["latent_dim"]   # 512
    hidden     = d["hidden"]       # (512, 512)
    num_bins   = d["num_bins"]     # 101
    V          = d["V"]            # 8
    lr         = d["lr"]           # 3e-4
    gamma      = d["gamma"]        # 0.99
    tau        = d["tau"]          # 0.01
    rho        = d["rho"]          # 0.5
    rew_scale  = d["rew_scale"]    # 10.0
    K_UPDATE   = int(k_update) if k_update is not None else d["K_UPDATE"]     # 64
    if k_update is not None and K_UPDATE != d["K_UPDATE"]:
        print(f"  K_UPDATE override: {K_UPDATE} gradient updates per batch (default {d['K_UPDATE']})", flush=True)
    BS         = d["BS"]           # 256
    N_ENVS     = d["N_ENVS"]       # 256
    WARMUP     = d["WARMUP_ENV"]   # 25_000
    EXPL_NOISE = d["EXPL_NOISE"]   # 0.3
    EXPL_UNTIL = int(expl_until) if expl_until is not None else d["EXPL_UNTIL"]   # 25_000 default
    if expl_until is not None and EXPL_UNTIL != d["EXPL_UNTIL"]:
        print(f"  EXPL_UNTIL override: random actions for first {EXPL_UNTIL:,} env-steps (default {d['EXPL_UNTIL']:,})", flush=True)
    EXPL_MIX_DECAY_STEPS = max(int(expl_mix_decay_steps), 0)
    if EXPL_MIX_DECAY_STEPS > 0:
        print(
            f"  EXPL_MIX override: random-policy action mixture decays random prob 1.0 -> 0.0 "
            f"over {EXPL_MIX_DECAY_STEPS:,} env-steps",
            flush=True,
        )
    # Optional act-noise anneal: linearly decay from start -> end over
    # `act_noise_anneal_steps` env steps. Defaults reproduce baseline behaviour
    # (constant EXPL_NOISE).
    _noise_start = float(act_noise_start) if act_noise_start is not None else float(EXPL_NOISE)
    _noise_end   = float(act_noise_end)   if act_noise_end   is not None else _noise_start
    _noise_anneal_steps = max(int(act_noise_anneal_steps), 1)
    def _current_noise(es: int) -> float:
        frac = min(max(es / _noise_anneal_steps, 0.0), 1.0)
        return _noise_start + (_noise_end - _noise_start) * frac
    if _noise_start != _noise_end:
        print(f"  act-noise anneal: {_noise_start:.3f} -> {_noise_end:.3f} over {_noise_anneal_steps:,} env-steps", flush=True)
    H          = int(mppi_horizon) if mppi_horizon is not None else d["H"]   # 3 by default
    if mppi_horizon is not None and H != d["H"]:
        print(f"  MPPI horizon override: H={H} (DEFAULTS['H']={d['H']})", flush=True)
    NS         = int(mppi_n_samples) if mppi_n_samples is not None else d["NS"]   # 512 default
    if mppi_n_samples is not None and NS != d["NS"]:
        print(f"  MPPI n_samples override: NS={NS} (DEFAULTS['NS']={d['NS']})", flush=True)
    elites     = d["NUM_ELITES"]   # 64
    pi_trajs   = d["NUM_PI_TRAJS"] # 24
    NI         = d["NI"]           # 6
    MIN_STD    = d["MIN_STD"]      # 0.05
    MAX_STD    = d["MAX_STD"]      # 2.0
    glass_cfg  = dict(d.get("glass", {}))
    if glass_overrides:
        glass_cfg.update({k: v for k, v in glass_overrides.items() if v is not None})
    seq_len    = H + 1             # 4 — trajectory length in buffer
    buf_cap    = max(total_steps // N_ENVS + 1000, 50_000)
    eval_interval = 250_000 if env_id == "HopperHop" else max(total_steps // 12, 1)
    episode_length = 1000

    # ── Environment
    env      = registry.load(env_id)
    env      = wrapper.wrap_for_brax_training(env, episode_length=episode_length,
                                              action_repeat=1)
    obs_dim  = env.observation_size
    act_dim  = env.action_size
    al, ah   = -1.0, 1.0
    print(f"  obs={obs_dim}  act={act_dim}", flush=True)

    # ── Networks
    enc_net = Encoder(latent_dim=latent_dim, hidden=hidden, V=V)
    dyn_net = Dynamics(latent_dim=latent_dim, hidden=hidden, V=V)
    rew_net = RewardHead(hidden=hidden, num_bins=num_bins)
    q_net   = QEnsemble(hidden=hidden, num_bins=num_bins)
    pi_net  = Pi(action_dim=act_dim, hidden=hidden)

    key = jax.random.PRNGKey(seed)
    key, ek, dk, rk, qk, pk, gk = jax.random.split(key, 7)
    dummy_obs  = jnp.zeros((1, obs_dim))
    dummy_z    = jnp.zeros((1, latent_dim))
    dummy_act  = jnp.zeros((1, act_dim))
    # Path 7 / Phase-v: when use_cluster_obs is on, pi/q first layer is sized
    # for (latent_dim + num_clusters). Init with augmented dummy.
    _use_cluster_obs = bool(use_glass and glass_cfg.get("use_cluster_obs", False))
    if _use_cluster_obs:
        _K = int(glass_cfg.get("num_clusters", 8))
        dummy_z_aug = jnp.zeros((1, latent_dim + _K))
        print(f"  Cluster-id policy observation active: pi/q first layer in_dim={latent_dim}+{_K}={latent_dim + _K}", flush=True)
    else:
        dummy_z_aug = dummy_z
    params = {
        "enc": enc_net.init(ek, dummy_obs),
        "dyn": dyn_net.init(dk, dummy_z, dummy_act),
        "rew": rew_net.init(rk, dummy_z, dummy_act),
        "q":   q_net.init(qk, dummy_z_aug, dummy_act),
        "pi":  pi_net.init(pk, dummy_z_aug),
    }
    if use_glass:
        params["glass"] = init_glass_params(
            gk,
            latent_dim=latent_dim,
            num_prototypes=glass_cfg.get("num_prototypes", 32),
            num_clusters=glass_cfg.get("num_clusters", 8),
            assign_logits_init_scale=glass_cfg.get("assign_logits_init_scale", 1.0),
            num_super_clusters=glass_cfg.get("num_super_clusters", 0),
        )
    tp    = params.copy()
    scale = jnp.array(1.0)
    glass_step = jnp.array(0, dtype=jnp.int32)

    # ── Optimizer (single shared chain so clip_by_global_norm sees the same
    #  parameter set as baseline TD-MPC2; with stopgrad on the Glass graph the
    #  glass subtree contributes negligible gradient norm).
    tx = optax.chain(
        optax.clip_by_global_norm(20.0),
        optax.adam(lr),
    )
    opt = tx.init(params)
    resume_env_steps = 0
    resume_best_mppi = -float("inf")
    resume_best_mppi_step = 0
    resume_payload = None
    if resume_checkpoint:
        with open(resume_checkpoint, "rb") as rf:
            resume_payload = pickle.load(rf)
        params = resume_payload["params"]
        tp = resume_payload.get("target_params", params)
        opt = resume_payload.get("opt_state", opt)
        scale = jnp.asarray(resume_payload.get("scale", scale))
        glass_step = jnp.asarray(resume_payload.get("glass_step", glass_step))
        resume_env_steps = int(resume_payload.get("env_steps", 0))
        resume_best_mppi = float(
            resume_payload.get("best_mppi", resume_payload.get("mppi_reward", -float("inf")))
        )
        resume_best_mppi_step = int(resume_payload.get("best_mppi_step", resume_env_steps))
        print(
            f"  Resumed model checkpoint {resume_checkpoint} "
            f"at env_steps={resume_env_steps:,}",
            flush=True,
        )

    # ── Library update functions
    _consistency_coef = float(consistency_coef) if consistency_coef is not None else float(d.get("consistency_coef", 2.0))
    _smooth_target = float(latent_action_smooth_coef)
    _smooth_warmup = int(latent_smooth_warmup_env_steps)
    _curriculum_active = _smooth_warmup > 0 and _smooth_target > 0
    _smooth_curr = 0.0 if _curriculum_active else _smooth_target
    _mpc_distill_target_coef = max(float(mpc_distill_coef), 0.0)
    _mpc_distill_anneal_steps = max(int(mpc_distill_anneal_steps), 1)
    _mpc_distill_disable_gap = float(mpc_distill_disable_gap)
    _mpc_distill_batch_size = max(int(mpc_distill_batch_size), 1)
    _mpc_distill_enabled = (not use_glass) and _mpc_distill_target_coef > 0
    if _smooth_target > 0 or _consistency_coef != 2.0:
        if _curriculum_active:
            print(f"  loss-coef: consistency={_consistency_coef} latent_action_smooth={_smooth_target} CURRICULUM "
                  f"(0 until {_smooth_warmup:,} env-steps, then ramp to target)", flush=True)
        else:
            print(f"  loss-coef overrides: consistency={_consistency_coef} latent_action_smooth={_smooth_target}", flush=True)
    if _mpc_distill_enabled:
        print(
            f"  Phase-mpc-lite active: coef={_mpc_distill_target_coef:.3f} "
            f"anneal_steps={_mpc_distill_anneal_steps:,} disable_gap={_mpc_distill_disable_gap:.1f} "
            f"anchor_batch={_mpc_distill_batch_size}",
            flush=True,
        )

    def _build_multi_step(smooth_coef: float):
        # smoothing_enabled controls whether the vmap-over-pi smoothing forward
        # pass is in the JIT graph at all. When False, the compiled graph
        # matches the pre-smoothing (Phase 1b) version exactly — basin choice
        # is then unperturbed (Phase-m fix for K=3-flip on seeds 4, 5).
        smoothing_enabled = smooth_coef > 0
        if use_glass:
            _, ms = make_update_fn(
                enc_net, dyn_net, rew_net, q_net, pi_net, tx,
                gamma=gamma, rho=rho, tau=tau, rew_scale=rew_scale,
                glass_enabled=glass_cfg.get("enabled", True),
                glass_every_k_updates=glass_cfg.get("every_k_updates", 4),
                glass_proto_temperature=glass_cfg.get("proto_temperature", 1.0),
                glass_assignment_temperature=glass_cfg.get("assignment_temperature", 1.0),
                glass_lambda_se=glass_cfg.get("lambda_se", 5.0e-3),
                glass_lambda_balance=glass_cfg.get("lambda_balance", 1.0e-2),
                glass_lambda_temporal=glass_cfg.get("lambda_temporal", 1.0e-3),
                glass_lambda_temp_stability=float(glass_cfg.get("lambda_temp_stability") or 0.0),
                glass_stopgrad_graph=glass_cfg.get("stopgrad_graph", True),
                glass_use_cosine_assign=glass_cfg.get("use_cosine_assign", True),
                latent_action_smooth_coef=smooth_coef,
                consistency_coef=_consistency_coef,
                smoothing_enabled=smoothing_enabled,
                glass_lambda_super_se=float(glass_cfg.get("lambda_super_se") or 0.0),
                glass_lambda_super_balance=float(glass_cfg.get("lambda_super_balance") or 0.0),
                use_cluster_obs=bool(glass_cfg.get("use_cluster_obs", False)),
                cluster_obs_proto_temperature=float(glass_cfg.get("proto_temperature", 1.0)),
            )
        else:
            _, ms = make_update_fn(
                enc_net, dyn_net, rew_net, q_net, pi_net, tx,
                gamma=gamma, rho=rho, tau=tau, rew_scale=rew_scale,
                latent_action_smooth_coef=smooth_coef,
                consistency_coef=_consistency_coef,
                smoothing_enabled=smoothing_enabled,
                mpc_distill_enabled=_mpc_distill_enabled,
            )
        return ms

    multi_step = _build_multi_step(_smooth_curr)

    # ── MPPI planner (for eval)
    # Build MPPI planner. Path-7 cluster_obs kwargs only valid for Glass make_mppi_fn,
    # so only pass them when use_glass=True (vanilla tdmpc2 make_mppi_fn doesn't accept them).
    _mppi_kw = {}
    if use_glass:
        _mppi_kw["use_cluster_obs"] = bool(glass_cfg.get("use_cluster_obs", False))
        _mppi_kw["cluster_obs_proto_temperature"] = float(glass_cfg.get("proto_temperature", 1.0))
    plan = make_mppi_fn(
        enc_net, dyn_net, rew_net, q_net, pi_net,
        horizon=H, n_samples=NS, num_elites=elites,
        num_pi_trajs=pi_trajs, n_iter=NI,
        min_std=MIN_STD, max_std=MAX_STD,
        act_low=al, act_high=ah, act_dim=act_dim,
        gamma=gamma, rew_scale=rew_scale,
        **_mppi_kw,
    )
    if use_glass:
        glass_diag = make_glass_diag_fn(
            enc_net,
            dyn_net,
            proto_temperature=glass_cfg.get("proto_temperature", 1.0),
            assignment_temperature=glass_cfg.get("assignment_temperature", 1.0),
            stopgrad_graph=glass_cfg.get("stopgrad_graph", True),
            use_cosine_assign=glass_cfg.get("use_cosine_assign", True),
        )
        diag_dir = EXP_DIR / "glass_diag" / f"{env_id}{('_' + os.environ.get('TDMPC_GLASS_OUTPUT_TAG','').strip()) if os.environ.get('TDMPC_GLASS_OUTPUT_TAG','').strip() else ''}" / f"seed_{seed}"
        diag_dir.mkdir(parents=True, exist_ok=True)

    # ── Vectorised action function (pi + noise, used during data collection)
    _proto_T = float(glass_cfg.get("proto_temperature", 1.0))
    @jax.jit
    def act_fn_batch(p, obs):
        from helios.algorithms.tdmpc_glass import augment_z_with_cluster as _aug
        z = enc_net.apply(p["enc"], obs)
        z_in = _aug(z, p["glass"], _proto_T) if _use_cluster_obs else z
        mu, _ = pi_net.apply(p["pi"], z_in)
        return jnp.tanh(mu)
    _mpc_mu0 = jnp.zeros((H, act_dim))
    _mpc_std0 = jnp.full((H, act_dim), MAX_STD)
    @jax.jit
    def batch_mppi_targets(p, obs_batch, plan_key):
        keys = jax.random.split(plan_key, obs_batch.shape[0])
        mu_b = jnp.broadcast_to(_mpc_mu0, (obs_batch.shape[0], H, act_dim))
        std_b = jnp.broadcast_to(_mpc_std0, (obs_batch.shape[0], H, act_dim))
        def _one(obs_i, mu_i, std_i, key_i):
            act_i, _, _ = plan(p, obs_i, mu_i, std_i, key_i, jnp.bool_(True))
            return act_i
        return jax.vmap(_one)(obs_batch, mu_b, std_b, keys)

    # ── Buffer (numpy, per-env ring buffer)
    buf  = MultiEnvBuffer(buf_cap, N_ENVS, obs_dim, act_dim, seq_len)
    rng_np = np.random.default_rng(seed)
    np.random.seed(seed)
    if resume_payload and "replay_buffer" in resume_payload:
        restore_buffer_state(buf, resume_payload["replay_buffer"])
        print(f"  Restored replay buffer. Buffer={buf.total_size()}", flush=True)
    if resume_payload and "rng_np_state" in resume_payload:
        rng_np.bit_generator.state = resume_payload["rng_np_state"]
    if resume_payload and "np_random_state" in resume_payload:
        np.random.set_state(resume_payload["np_random_state"])

    # ── Vectorised env reset/step
    @jax.jit
    def batch_step(state, acts):
        return env.step(state, acts)

    # Path 5 / Phase-t — knee penalty fn. Computes per-env penalty based on
    # how far below threshold any non-foot geom z drops. Geom IDs (HopperHop):
    # 0=floor, 1=torso, 2=nose, 3=pelvis, 4=thigh, 5=calf, 6=foot.
    # Penalty target: sum over geoms 1..5 of max(0, threshold - z).
    # Returns shape (N_ENVS,) — subtracted from training reward only.
    _knee_pen_thr = float(knee_penalty_threshold)
    _knee_pen_coef = float(knee_penalty_coef)
    @jax.jit
    def knee_penalty_fn(state):
        # state.data.geom_xpos shape: (batch, num_geoms, 3)
        non_foot_z = state.data.geom_xpos[..., 1:6, 2]  # (batch, 5)
        per_geom = jnp.maximum(_knee_pen_thr - non_foot_z, 0.0)
        return jnp.sum(per_geom, axis=-1)  # (batch,)
    if _knee_pen_coef > 0:
        print(f"  Knee penalty active: coef={_knee_pen_coef} threshold={_knee_pen_thr}", flush=True)

    # iter-6 §7.C — Phase-r2 gait penalty + iter-6 §7.B — Phase-r1 soft-reward bundle.
    # Both shape the training reward only; eval reward stays the original task reward.
    # Height = torso_z - foot_z, computed off the inner Hopper env's body ids
    # (wrappers forward attribute access to the inner env).
    _gait_fall_coef = float(gait_fall_penalty)
    _gait_fall_h = float(gait_fall_height)
    _gait_smooth_coef = float(gait_action_smooth)
    _soft_bonus = float(soft_stand_bonus)
    _soft_floor = float(soft_stand_floor)
    _soft_anneal = int(soft_anneal_steps)
    _need_height = (_gait_fall_coef > 0 or _soft_bonus > 0)
    _torso_id = getattr(env, "_torso_id", None)
    _foot_id = getattr(env, "_foot_id", None)
    if _need_height and (_torso_id is None or _foot_id is None):
        print(f"  [warn] gait/soft-reward needs _torso_id/_foot_id from env; got "
              f"torso={_torso_id} foot={_foot_id}, disabling", flush=True)
        _gait_fall_coef = 0.0
        _soft_bonus = 0.0
        _need_height = False
    @jax.jit
    def _height_per_env(state):
        if _torso_id is None or _foot_id is None:
            return jnp.zeros(state.data.xipos.shape[0])
        return state.data.xipos[..., _torso_id, 2] - state.data.xipos[..., _foot_id, 2]
    if _gait_fall_coef > 0 or _gait_smooth_coef > 0:
        print(f"  Phase-r2 gait penalty active: fall_coef={_gait_fall_coef} h<{_gait_fall_h} "
              f"action_smooth={_gait_smooth_coef}", flush=True)
    if _soft_bonus > 0:
        print(f"  Phase-r1 soft-reward bundle: stand_bonus={_soft_bonus} floor={_soft_floor} "
              f"anneal_steps={_soft_anneal} (linear fade 1->0)", flush=True)
    _prev_acts = None  # populated on first step

    # Path P / Phase-P — cluster-entropy intrinsic reward (benchmark-fair).
    # Compute current Glass cluster id per env, maintain ring buffer of last W,
    # add coef * entropy(window) to training reward. Encourages gait diversity
    # via Glass partition without modifying the env.
    _cluster_coef_init = float(cluster_intrinsic_coef)
    _cluster_window = int(cluster_intrinsic_window)
    _cluster_decay_steps = int(cluster_intrinsic_decay_steps)
    _cluster_decay_start = int(expl_until) if expl_until is not None else 0
    _cluster_active = use_glass and _cluster_coef_init > 0
    if _cluster_active:
        if _cluster_decay_steps > 0:
            print(f"  Cluster-entropy intrinsic reward active: coef={_cluster_coef_init} window={_cluster_window} "
                  f"linear-decay [{_cluster_decay_start:,} -> {_cluster_decay_steps:,}] env-steps", flush=True)
        else:
            print(f"  Cluster-entropy intrinsic reward active: coef={_cluster_coef_init} window={_cluster_window} (static)", flush=True)
        _glass_K = int(glass_cfg.get("num_clusters", 8))
        # per-env ring buffer of cluster ids (np for cheap host-side updates)
        _cluster_history = np.zeros((N_ENVS, _cluster_window), dtype=np.int32)
        _cluster_history_ptr = 0  # ring buffer write index, shared across envs

        @jax.jit
        def cluster_id_batch(params, obs_batch):
            """Compute argmax(S[argmax(z·μᵀ)]) per env."""
            z = enc_net.apply(params["enc"], obs_batch)  # (N_ENVS, latent_dim)
            mu = params["glass"]["prototypes"]
            zn = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            mn = mu / (jnp.linalg.norm(mu, axis=-1, keepdims=True) + 1e-8)
            sim = (zn @ mn.T) / glass_cfg.get("proto_temperature", 1.0)   # (N_ENVS, N_proto)
            n_star = jnp.argmax(sim, axis=-1)                              # (N_ENVS,)
            S = jax.nn.softmax(params["glass"]["assign_logits"], axis=-1)
            return jnp.argmax(S[n_star], axis=-1)                          # (N_ENVS,)

        def cluster_entropy_per_env(history):
            """Shannon entropy of cluster-id distribution per env. history: (N_ENVS, W)."""
            ent = np.zeros(history.shape[0], dtype=np.float32)
            for env_i in range(history.shape[0]):
                _, counts = np.unique(history[env_i], return_counts=True)
                p = counts / history.shape[1]
                ent[env_i] = -np.sum(p * np.log(np.clip(p, 1e-8, 1.0)))
            return ent

    key, ek2 = jax.random.split(key)
    if resume_payload and "env_state" in resume_payload and "obs_np" in resume_payload:
        env_state = resume_payload["env_state"]
        obs_np = resume_payload["obs_np"]
        print("  Restored vectorized environment state", flush=True)
    else:
        env_state = env.reset(jax.random.split(ek2, N_ENVS))
        obs_np    = np.array(env_state.obs)
    env_steps = resume_env_steps

    # ── Pi-based eval (deterministic, single episode at a time)
    @jax.jit
    def enc_apply(p, obs):
        return enc_net.apply(p["enc"], obs[None])

    @jax.jit
    def pi_apply(p, z):
        from helios.algorithms.tdmpc_glass import augment_z_with_cluster as _aug
        z_in = _aug(z, p["glass"], _proto_T) if _use_cluster_obs else z
        mu, _ = pi_net.apply(p["pi"], z_in)
        return jnp.tanh(mu[0])

    @jax.jit
    def single_env_step(state, act):
        return env.step(state, act[None])

    @jax.jit
    def single_env_reset(key):
        return env.reset(jax.random.split(key, 1))

    # Iter 6 §7.1 — episode diagnostics from per-step reward signal.
    # full_reward (>0.5)   = standing AND fast (~target speed met)
    # standing_proxy(>0.01) = at least upright
    # fall_count            = transitions standing → fallen (proxy via reward crossing 0.01)
    # time_to_first_full    = first step where reward > 0.5 (or episode_length if never)
    # No internal env state access — env-agnostic, no extra device syncs.
    def _episode_diag(rewards_list):
        if not rewards_list:
            return {"full": 0.0, "stand": 0.0, "falls": 0, "ttf": episode_length}
        r = np.asarray(rewards_list, dtype=np.float32)
        full = (r > 0.5)
        stand = (r > 0.01)
        falls = int(np.sum(stand[:-1] & (~stand[1:]))) if len(stand) > 1 else 0
        ttf = int(np.argmax(full)) if full.any() else episode_length
        return {
            "full": float(full.mean()),
            "stand": float(stand.mean()),
            "falls": falls,
            "ttf": ttf,
        }

    def eval_pi(n_eps: int = 5):
        nonlocal key
        rets, diags = [], []
        for _ in range(n_eps):
            key, rk2 = jax.random.split(key)
            state = single_env_reset(rk2)
            obs   = jnp.asarray(state.obs[0])
            er = 0.0
            ep_rew = []
            for _ in range(episode_length):
                z   = enc_apply(params, obs)
                act = pi_apply(params, z)
                state = single_env_step(state, act)
                r = float(state.reward[0])
                er += r
                ep_rew.append(r)
                if bool(state.done[0] > 0.5):
                    break
                obs = jnp.asarray(state.obs[0])
            rets.append(er)
            diags.append(_episode_diag(ep_rew))
        # mean across episodes
        agg = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
        return float(np.mean(rets)), agg

    def eval_mppi(n_eps: int = 3):
        nonlocal key
        rets, diags = [], []
        for _ in range(n_eps):
            key, rk2 = jax.random.split(key)
            state = single_env_reset(rk2)
            obs = jnp.asarray(state.obs[0])
            mu = jnp.zeros((H, act_dim))
            std = jnp.full((H, act_dim), MAX_STD)
            er = 0.0
            ep_rew = []
            t0_mppi = jnp.bool_(True)
            key, pk2 = jax.random.split(key)
            for _ in range(episode_length):
                act, mu, std = plan(params, obs, mu, std, pk2, t0_mppi)
                t0_mppi = jnp.bool_(False)
                key, pk2 = jax.random.split(key)
                state = single_env_step(state, act)
                r = float(state.reward[0])
                er += r
                ep_rew.append(r)
                if bool(state.done[0] > 0.5):
                    break
                obs = jnp.asarray(state.obs[0])
            rets.append(er)
            diags.append(_episode_diag(ep_rew))
        agg = {k: float(np.mean([d[k] for d in diags])) for k in diags[0]}
        return float(np.mean(rets)), agg

    eval_type_csv = None
    ckpt_dir = None
    best_mppi = resume_best_mppi
    best_mppi_step = resume_best_mppi_step
    # iter-8 §2.0 — Phase-eval: best-pi + best-any tracking. MPPI < pi in
    # ~17% of runs (mppi_vs_pi_analysis.md); preserve both.
    best_pi = -float("inf")
    best_pi_step = 0
    best_any = -float("inf")
    best_any_step = 0
    best_any_selector = ""  # "pi" or "mppi" — which evaluator picked best_any

    # iter-7 §2.1 — Phase-ar auto-restart on plateau tracker
    # The plateau check fires every restart_check_at env steps. If best_mppi is
    # still below restart_threshold at the check, re-init pi+q (keep encoder,
    # dynamics, reward, replay buffer, env state). Up to restart_max_attempts.
    _restart_count = 0
    _restart_next_check = int(restart_check_at) if restart_on_plateau else 0
    if restart_on_plateau:
        print(
            f"  Phase-ar plateau detector: check every {restart_check_at:,} env-steps; "
            f"reset pi+q if best MPPI < {restart_threshold:.0f}; max {restart_max_attempts} attempts.",
            flush=True,
        )
    _mpc_gap_allows_distill = True
    early_stop_triggered = False
    _patience = max(int(early_stop_patience), 0)
    if _patience > 0:
        print(f"  Early-stop: will halt after {_patience:,} env-steps with no new best MPPI", flush=True)
    _controller_arbitration = str(controller_arbitration or "none").lower()
    _arbitration_margin = float(arbitration_margin)
    if _controller_arbitration not in {"none", "eval_only"}:
        raise ValueError(f"Unknown controller_arbitration={controller_arbitration!r}")
    if _controller_arbitration == "eval_only":
        print(
            f"  Controller arbitration: eval_only, selecting MPPI only when "
            f"MPPI > pi + {_arbitration_margin:.1f}. Replay collection is unchanged.",
            flush=True,
        )
    # Optional output-tag suffix so we can run multiple experiment phases
    # (e.g. phase1 / phase2) against the same env_id without clobbering files.
    _tag = os.environ.get("TDMPC_GLASS_OUTPUT_TAG", "").strip()
    _env_dir = f"{env_id}{('_' + _tag) if _tag else ''}"
    if use_glass:
        eval_type_csv = (
            EXP_DIR.parent
            / "tdmpc_glass"
            / _env_dir
            / f"seed_{seed}.csv"
        )
        eval_type_csv.parent.mkdir(parents=True, exist_ok=True)
        if not eval_type_csv.exists():
            with open(eval_type_csv, "w") as cf:
                cf.write("step,reward,eval_type,seed\n")
    elif env_id == "HopperHop":
        # iter 6 fix — vanilla tdmpc2 now uses the same per-seed, tag-aware
        # layout as Glass. Old shared-CSV path `tdmpc_dmc/hopper-hop-tdmpc2-rerun.csv`
        # got overwritten on each seed launch (bug discovered with Phase-z).
        eval_type_csv = (
            EXP_DIR.parent
            / "tdmpc_glass"
            / _env_dir
            / f"seed_{seed}.csv"
        )
        eval_type_csv.parent.mkdir(parents=True, exist_ok=True)
        if not eval_type_csv.exists():
            with open(eval_type_csv, "w") as cf:
                cf.write("step,reward,eval_type,seed\n")

    if eval_type_csv is not None:
        ckpt_dir = eval_type_csv.parent / f"seed_{seed}" / "checkpoints"

    # ── Warmup JIT — compile multi_step with dummy data
    print("  Warming up JIT (may take 30-90s)...", flush=True)
    t_jit = time.time()
    _dummy_obs = np.zeros((K_UPDATE, BS, seq_len, obs_dim), np.float32)
    _dummy_act = np.zeros((K_UPDATE, BS, seq_len, act_dim), np.float32)
    _dummy_rew = np.zeros((K_UPDATE, BS, seq_len), np.float32)
    _dummy_don = np.zeros((K_UPDATE, BS, seq_len), np.float32)
    _dummy_mpc_obs = np.zeros((_mpc_distill_batch_size, obs_dim), np.float32)
    _dummy_mpc_act = np.zeros((_mpc_distill_batch_size, act_dim), np.float32)
    _dummy_mpc_coef = jnp.array(0.0, dtype=jnp.float32)
    if use_glass:
        params, tp, opt, key, scale, glass_step, _, _ = multi_step(
            params, tp, opt,
            jnp.asarray(_dummy_obs), jnp.asarray(_dummy_act),
            jnp.asarray(_dummy_rew), jnp.asarray(_dummy_don),
            key, scale, glass_step, False,
        )
    else:
        params, tp, opt, key, scale, _, _ = multi_step(
            params, tp, opt,
            jnp.asarray(_dummy_obs), jnp.asarray(_dummy_act),
            jnp.asarray(_dummy_rew), jnp.asarray(_dummy_don),
            key, scale,
            jnp.asarray(_dummy_mpc_obs), jnp.asarray(_dummy_mpc_act), _dummy_mpc_coef,
        )
    jax.block_until_ready(scale)
    print(f"  JIT compiled in {time.time()-t_jit:.1f}s", flush=True)
    if resume_checkpoint:
        with open(resume_checkpoint, "rb") as rf:
            ckpt = pickle.load(rf)
        params = ckpt["params"]
        tp = ckpt.get("target_params", params)
        opt = ckpt.get("opt_state", opt)
        scale = jnp.asarray(ckpt.get("scale", scale))
        glass_step = jnp.asarray(ckpt.get("glass_step", glass_step))
        key = jnp.asarray(ckpt.get("key", key))
        print("  Restored checkpoint state after JIT warmup", flush=True)

    # ── Training loop (Phase 4: vectorised buffer sample + lax.scan update)
    next_eval     = (
        ((env_steps // eval_interval) + 1) * eval_interval
        if env_steps
        else eval_interval
    )
    log_interval  = N_ENVS * 20   # log every 20 * N_ENVS steps
    t0 = time.time()
    loss_val = 0.0

    # Q-reset (REDQ-style): re-init online Q params + optimizer state when env_steps
    # crosses any threshold in q_reset_steps. Target Q (tp["q"]) is preserved so the
    # critic restarts from the EMA of past good critics rather than from scratch.
    q_reset_pending = sorted(int(s) for s in (q_reset_steps or []))
    if q_reset_pending:
        print(f"  Q-reset scheduled at env_steps: {q_reset_pending}", flush=True)

    with open_csv(csv_path, env_id, seed) as fh:
        while env_steps < total_steps:
            # Collect N_ENVS steps
            if EXPL_MIX_DECAY_STEPS > 0:
                rand_prob = max(0.0, 1.0 - env_steps / EXPL_MIX_DECAY_STEPS)
                rand_acts = rng_np.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
                acts_jax = act_fn_batch(params, jnp.asarray(obs_np))
                _sigma = _current_noise(env_steps)
                noise = rng_np.normal(0, _sigma, (N_ENVS, act_dim))
                policy_acts = np.clip(np.array(acts_jax) + noise, al, ah).astype(np.float32)
                rand_mask = rng_np.random((N_ENVS, 1)) < rand_prob
                acts_np = np.where(rand_mask, rand_acts, policy_acts).astype(np.float32)
            elif env_steps < EXPL_UNTIL:
                acts_np = rng_np.uniform(al, ah, (N_ENVS, act_dim)).astype(np.float32)
            else:
                acts_jax = act_fn_batch(params, jnp.asarray(obs_np))
                _sigma   = _current_noise(env_steps)
                noise    = rng_np.normal(0, _sigma, (N_ENVS, act_dim))
                acts_np  = np.clip(np.array(acts_jax) + noise, al, ah).astype(np.float32)

            env_state = batch_step(env_state, jnp.asarray(acts_np))
            new_obs  = np.array(env_state.obs)
            rews_np  = np.array(env_state.reward)
            done_np  = np.array(env_state.done > 0.5, np.float32)
            # Path 5 / Phase-t — knee penalty (training reward only). Eval
            # reward is unmodified so we measure against the original task.
            if _knee_pen_coef > 0:
                pen_np = np.array(knee_penalty_fn(env_state))
                rews_np = rews_np - _knee_pen_coef * pen_np
            # iter-6 §7.C — Phase-r2 gait penalty (training reward only)
            #     fall_penalty: -coef when height < gait_fall_height
            #     action_smooth: -coef * mean((a_t - a_{t-1})**2) per env
            # iter-6 §7.B — Phase-r1 soft-reward bundle (training reward only)
            #     stand_bonus: +coef * clip((h - floor)/(STAND_H - floor), 0, 1)
            #     weight linearly anneals 1.0 -> 0.0 over [0, soft_anneal_steps]
            _h_np_cache = None
            if _gait_fall_coef > 0:
                _h_np_cache = np.array(_height_per_env(env_state))
                rews_np = rews_np - _gait_fall_coef * (_h_np_cache < _gait_fall_h).astype(np.float32)
            if _gait_smooth_coef > 0:
                if _prev_acts is None:
                    _prev_acts = np.zeros_like(acts_np)
                da_sq = np.mean((acts_np - _prev_acts) ** 2, axis=-1)
                rews_np = rews_np - _gait_smooth_coef * da_sq
            if _soft_bonus > 0:
                _soft_weight = 1.0
                if _soft_anneal > 0:
                    _soft_weight = max(0.0, 1.0 - env_steps / _soft_anneal)
                if _soft_weight > 0:
                    if _h_np_cache is None:
                        _h_np_cache = np.array(_height_per_env(env_state))
                    _STAND_H = 0.6
                    bonus = np.clip((_h_np_cache - _soft_floor) / max(_STAND_H - _soft_floor, 1e-6), 0.0, 1.0)
                    rews_np = rews_np + _soft_bonus * _soft_weight * bonus
            if _gait_smooth_coef > 0:
                _prev_acts = acts_np.copy()
            # Path P / Phase-P — cluster-entropy intrinsic reward (training only).
            # Phase-Pa: linearly decay coef to 0 over [_cluster_decay_start, _cluster_decay_steps]
            # so intrinsic is an exploration curriculum, not a permanent reward distortion.
            if _cluster_active:
                if _cluster_decay_steps > 0:
                    frac = max(
                        0.0,
                        min(
                            1.0,
                            (env_steps - _cluster_decay_start)
                            / max(1, _cluster_decay_steps - _cluster_decay_start),
                        ),
                    )
                    _cluster_coef = _cluster_coef_init * (1.0 - frac)
                else:
                    _cluster_coef = _cluster_coef_init
                if _cluster_coef > 1e-6:
                    cluster_ids = np.array(cluster_id_batch(params, jnp.asarray(new_obs)))  # (N_ENVS,)
                    _cluster_history[:, _cluster_history_ptr] = cluster_ids
                    _cluster_history_ptr = (_cluster_history_ptr + 1) % _cluster_window
                    ent_np = cluster_entropy_per_env(_cluster_history)
                    rews_np = rews_np + _cluster_coef * ent_np
            buf.add_batch(obs_np, acts_np, rews_np, done_np)
            obs_np    = new_obs
            env_steps += N_ENVS

            # Curriculum smoothing: turn on smoothing at the warmup boundary.
            # Rebuilds multi_step with the new coef (one-time ~3 min JIT recompile).
            if _curriculum_active and env_steps >= _smooth_warmup and _smooth_curr != _smooth_target:
                print(
                    f"  [curriculum] env_steps={env_steps:,} crossed warmup={_smooth_warmup:,} — "
                    f"raising latent_action_smooth_coef {_smooth_curr} -> {_smooth_target} "
                    f"(JIT recompile cost ~3 min)", flush=True
                )
                _smooth_curr = _smooth_target
                multi_step = _build_multi_step(_smooth_curr)

            # Q-reset check (must happen before the gradient update so the freshly
            # initialised Q sees the next batch with a clean opt state).
            while q_reset_pending and env_steps >= q_reset_pending[0]:
                threshold = q_reset_pending.pop(0)
                key, qreset_k = jax.random.split(key)
                fresh_q = q_net.init(qreset_k, dummy_z, dummy_act)
                params = {**params, "q": fresh_q}
                opt = tx.init(params)  # opt state shape changes with q tree; safest to re-init
                print(
                    f"  [Q-reset] env_steps={env_steps:,} crossed threshold={threshold:,}"
                    f" — online Q + opt state re-initialised (target Q preserved)",
                    flush=True,
                )

            # Gradient updates (one vectorised sample + one H2D + lax.scan)
            samp_k = buf.sample_k(K_UPDATE, BS, rng_np)
            if samp_k is not None:
                ob_k, ab_k, rb_k, db_k = [jnp.asarray(x) for x in samp_k]
                if use_glass:
                    _glass_warmup = glass_cfg.get("warmup_env_steps", 100_000)
                    _glass_decay = int(glass_decay_steps) if glass_decay_steps > 0 else None
                    # glass active in window [warmup, decay): turn ON after warmup, OFF after decay if set
                    glass_active = (env_steps >= _glass_warmup) and (_glass_decay is None or env_steps < _glass_decay)
                    params, tp, opt, key, scale, glass_step, loss_val, aux = multi_step(
                        params, tp, opt, ob_k, ab_k, rb_k, db_k, key, scale,
                        glass_step, glass_active,
                    )
                else:
                    mpc_obs_anchor = jnp.asarray(_dummy_mpc_obs)
                    mpc_action_target = jnp.asarray(_dummy_mpc_act)
                    mpc_coef_step = jnp.array(0.0, dtype=jnp.float32)
                    if _mpc_distill_enabled and env_steps >= EXPL_UNTIL and _mpc_gap_allows_distill:
                        anneal_frac = min(max((env_steps - EXPL_UNTIL) / _mpc_distill_anneal_steps, 0.0), 1.0)
                        curr_coef = _mpc_distill_target_coef * (1.0 - anneal_frac)
                        if curr_coef > 1e-6:
                            anchor = buf.sample(_mpc_distill_batch_size, rng_np)
                            if anchor is not None:
                                anchor_obs = np.asarray(anchor[0][:, 0, :], dtype=np.float32)
                                mpc_obs_anchor = jnp.asarray(anchor_obs)
                                key, mpc_k = jax.random.split(key)
                                mpc_action_target = batch_mppi_targets(params, mpc_obs_anchor, mpc_k)
                                mpc_coef_step = jnp.array(curr_coef, dtype=jnp.float32)
                    params, tp, opt, key, scale, loss_val, aux = multi_step(
                        params, tp, opt, ob_k, ab_k, rb_k, db_k, key, scale,
                        mpc_obs_anchor, mpc_action_target, mpc_coef_step,
                    )

            if env_steps % log_interval < N_ENVS:
                elapsed = time.time() - t0
                _mpc_log = "" if use_glass else f"  mpc={float(aux.get('mpc', 0.0)):.4f}"
                print(f"  es={env_steps:>9,}  sps={env_steps/max(elapsed,1):.0f}"
                      f"  loss={float(loss_val):.4f}  scale={float(scale):.2f}{_mpc_log}", flush=True)

            if env_steps >= next_eval:
                ret, pi_diag = eval_pi(n_eps=5)
                mppi_ret, mppi_diag = eval_mppi(n_eps=8 if use_glass else 3)
                arb_ret = None
                arb_selector = ""
                arb_gap = float(mppi_ret) - float(ret)
                if _controller_arbitration == "eval_only":
                    if mppi_ret > ret + _arbitration_margin:
                        arb_selector = "mppi"
                        arb_ret = float(mppi_ret)
                    else:
                        arb_selector = "pi"
                        arb_ret = float(ret)
                elapsed = time.time() - t0
                _pi_minus_mppi = float(ret) - float(mppi_ret)
                _gap_marker = "  pi>>MPPI" if _pi_minus_mppi >= 100 else ("  MPPI>>pi" if _pi_minus_mppi <= -100 else "")
                print(f"  step={env_steps:>9,}  pi_reward={ret:7.1f}"
                      f"  MPPI={mppi_ret:7.1f}"
                      f"  sps={env_steps/max(elapsed,1):.0f}{_gap_marker}", flush=True)
                if arb_ret is not None:
                    print(
                        f"    arb eval: selected={arb_selector} reward={arb_ret:.1f} "
                        f"mppi_minus_pi={arb_gap:.1f} margin={_arbitration_margin:.1f}",
                        flush=True,
                    )
                if _mpc_distill_enabled:
                    _prev_gate = _mpc_gap_allows_distill
                    _mpc_gap_allows_distill = (_pi_minus_mppi < _mpc_distill_disable_gap)
                    if _prev_gate != _mpc_gap_allows_distill:
                        state = "ENABLED" if _mpc_gap_allows_distill else "DISABLED"
                        print(
                            f"    [Phase-mpc-lite] planner distill {state}: "
                            f"pi-mppi gap={_pi_minus_mppi:.1f} vs disable_gap={_mpc_distill_disable_gap:.1f}",
                            flush=True,
                        )
                # §7.1 diagnostics — per-step reward signal proxies.
                # full=fraction with reward>0.5 (standing+fast); stand=fraction>0.01 (upright);
                # falls=transitions from stand→fallen; ttf=first step at full reward (or episode_length if never).
                print(f"    diag pi:   full={pi_diag['full']:.2f} stand={pi_diag['stand']:.2f}"
                      f"  falls={pi_diag['falls']}  ttf={pi_diag['ttf']}", flush=True)
                print(f"    diag mppi: full={mppi_diag['full']:.2f} stand={mppi_diag['stand']:.2f}"
                      f"  falls={mppi_diag['falls']}  ttf={mppi_diag['ttf']}", flush=True)
                if use_glass:
                    diag_sample = buf.sample(128, rng_np)
                    if diag_sample is not None:
                        ob_d, ab_d, _, _ = [jnp.asarray(x) for x in diag_sample]
                        gd = jax.device_get(glass_diag(params, ob_d, ab_d))
                        print(
                            "    glass"
                            f" se={float(gd['glass_se']):.4f}"
                            f" ent={float(gd['glass_entropy']):.3f}"
                            f" active={float(gd['glass_active_clusters']):.0f}"
                            f" max_mass={float(gd['glass_max_cluster_mass']):.3f}"
                            f" cut={float(gd['glass_transition_cut_mass']):.3f}",
                            flush=True,
                        )
                        if glass_cfg.get("diag_dump_matrices", True):
                            diag_dir.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(
                                diag_dir / f"step_{env_steps}.npz",
                                P=np.asarray(gd["P"]),
                                A=np.asarray(gd["A"]),
                                S=np.asarray(gd["S"]),
                            )
                write_csv(fh, env_id, seed, env_steps, ret)
                if eval_type_csv is not None:
                    with open(eval_type_csv, "a") as cf:
                        cf.write(f"{env_steps},{ret:.1f},pi,{seed}\n")
                        cf.write(f"{env_steps},{mppi_ret:.1f},mppi,{seed}\n")
                        if arb_ret is not None:
                            cf.write(f"{env_steps},{arb_ret:.1f},arb,{seed}\n")
                    # §7.1 diagnostics CSV — sibling file, doesn't affect main eval CSV.
                    _diag_csv = eval_type_csv.with_name(eval_type_csv.name.replace(".csv", "_diag.csv"))
                    if not _diag_csv.exists():
                        with open(_diag_csv, "w") as df:
                            df.write("step,eval_type,seed,full_reward_rate,standing_rate,fall_count,time_to_first_full\n")
                    with open(_diag_csv, "a") as df:
                        df.write(f"{env_steps},pi,{seed},{pi_diag['full']:.4f},{pi_diag['stand']:.4f},{pi_diag['falls']},{pi_diag['ttf']}\n")
                        df.write(f"{env_steps},mppi,{seed},{mppi_diag['full']:.4f},{mppi_diag['stand']:.4f},{mppi_diag['falls']},{mppi_diag['ttf']}\n")
                    if arb_ret is not None:
                        _arb_csv = eval_type_csv.with_name(eval_type_csv.name.replace(".csv", "_arbitration.csv"))
                        if not _arb_csv.exists():
                            with open(_arb_csv, "w") as af:
                                af.write("step,seed,pi_reward,mppi_reward,mppi_minus_pi,selected,reward,margin\n")
                        with open(_arb_csv, "a") as af:
                            af.write(
                                f"{env_steps},{seed},{ret:.1f},{mppi_ret:.1f},{arb_gap:.1f},"
                                f"{arb_selector},{arb_ret:.1f},{_arbitration_margin:.1f}\n"
                            )
                if ckpt_dir is not None:
                    ckpt_payload = {
                        "algo": "tdmpc-glass" if use_glass else "tdmpc2",
                        "env_id": env_id,
                        "seed": seed,
                        "env_steps": env_steps,
                        "pi_reward": ret,
                        "mppi_reward": mppi_ret,
                        "params": jax.device_get(params),
                        "target_params": jax.device_get(tp),
                        "opt_state": jax.device_get(opt),
                        "scale": jax.device_get(scale),
                        "glass_step": jax.device_get(glass_step),
                        "key": jax.device_get(key),
                        "glass_config": dict(glass_cfg) if use_glass else {},
                        "best_mppi": best_mppi,
                        "best_mppi_step": best_mppi_step,
                    }
                    if arb_ret is not None:
                        ckpt_payload.update(
                            {
                                "arbitration_reward": arb_ret,
                                "arbitration_selector": arb_selector,
                                "arbitration_margin": _arbitration_margin,
                            }
                        )
                    if mppi_ret > best_mppi:
                        best_mppi = mppi_ret
                        best_mppi_step = env_steps
                        ckpt_payload["best_mppi"] = best_mppi
                        ckpt_payload["best_mppi_step"] = best_mppi_step
                        save_pickle_atomic(ckpt_dir / "best_mppi.pkl", ckpt_payload)
                    # iter-8 §2.0 — Phase-eval: track best-pi and best-any.
                    # MPPI is empirically worse than pi in a large minority of
                    # runs; preserving best-pi avoids discarding good actors.
                    if ret > best_pi:
                        best_pi = ret
                        best_pi_step = env_steps
                        ckpt_payload["best_pi"] = best_pi
                        ckpt_payload["best_pi_step"] = best_pi_step
                        save_pickle_atomic(ckpt_dir / "best_pi.pkl", ckpt_payload)
                    _curr_best_pair = max(ret, mppi_ret)
                    _curr_selector = "pi" if ret >= mppi_ret else "mppi"
                    if _curr_best_pair > best_any:
                        best_any = _curr_best_pair
                        best_any_step = env_steps
                        best_any_selector = _curr_selector
                        ckpt_payload["best_any"] = best_any
                        ckpt_payload["best_any_step"] = best_any_step
                        ckpt_payload["best_any_selector"] = best_any_selector
                        save_pickle_atomic(ckpt_dir / "best_any.pkl", ckpt_payload)
                    save_pickle_atomic(ckpt_dir / "latest_eval.pkl", ckpt_payload)
                    if save_full_state:
                        full_payload = {
                            **ckpt_payload,
                            "replay_buffer": buffer_state(buf),
                            "env_state": jax.device_get(env_state),
                            "obs_np": obs_np,
                            "rng_np_state": rng_np.bit_generator.state,
                            "np_random_state": np.random.get_state(),
                        }
                        save_pickle_atomic(ckpt_dir / "latest_full.pkl", full_payload)
                next_eval += eval_interval

                # iter-7 §2.1 — Phase-ar auto-restart on plateau.
                # Check fires once env_steps crosses each restart_check_at boundary.
                # If best_mppi is still below threshold, the seed is basin-locked;
                # re-init pi+q (keep enc/dyn/rew/glass + replay buffer + env state).
                if (restart_on_plateau
                        and env_steps >= _restart_next_check
                        and _restart_count < restart_max_attempts):
                    if best_mppi < restart_threshold:
                        _restart_count += 1
                        pre_restart_best = best_mppi
                        key, pikey, qkey = jax.random.split(key, 3)
                        fresh_pi = pi_net.init(pikey, dummy_z_aug)
                        fresh_q = q_net.init(qkey, dummy_z_aug, dummy_act)
                        params = {**params, "pi": fresh_pi, "q": fresh_q}
                        tp = {**tp, "pi": fresh_pi, "q": fresh_q}  # target Q resync
                        opt = tx.init(params)
                        # Clear best tracking so early-stop patience resets and a future
                        # check fires only if the *next* restart_check_at window also
                        # plateaus.
                        best_mppi = -float("inf")
                        best_mppi_step = env_steps
                        _restart_next_check = env_steps + int(restart_check_at)
                        print(
                            f"  [Phase-ar restart {_restart_count}/{restart_max_attempts}] "
                            f"env_steps={env_steps:,}: pre-restart best MPPI={pre_restart_best:.1f} "
                            f"< threshold={restart_threshold:.1f}. "
                            f"Re-init pi+q+target_q+opt (keep enc/dyn/rew/glass/replay/env).",
                            flush=True,
                        )
                        # Log restart marker row to eval CSV so dashboard / analysis can
                        # see where attempts began.
                        if eval_type_csv is not None:
                            with open(eval_type_csv, "a") as cf:
                                cf.write(f"{env_steps},-1.0,restart_{_restart_count},{seed}\n")
                    else:
                        # Crossed the boundary but already above threshold — schedule
                        # next check without restarting.
                        _restart_next_check = env_steps + int(restart_check_at)
                        print(
                            f"  [Phase-ar] env_steps={env_steps:,} above threshold "
                            f"(best={best_mppi:.1f} >= {restart_threshold:.1f}); no restart.",
                            flush=True,
                        )

                # Early stop: halt if no new best MPPI in the last `_patience` env-steps.
                if _patience > 0 and best_mppi_step > 0 and (env_steps - best_mppi_step) >= _patience:
                    print(
                        f"  Early-stop fired at env_steps={env_steps:,}: "
                        f"no new best MPPI since step={best_mppi_step:,} "
                        f"(best={best_mppi:.1f}, patience={_patience:,} env-steps).",
                        flush=True,
                    )
                    early_stop_triggered = True
                    break

        # Outer while-loop also exits if early-stop fired during the eval block above.
        if early_stop_triggered:
            pass  # fall through to checkpoint code below

    if ckpt_dir is not None:
        final_payload = {
            "algo": "tdmpc-glass" if use_glass else "tdmpc2",
            "env_id": env_id,
            "seed": seed,
            "env_steps": env_steps,
            "params": jax.device_get(params),
            "target_params": jax.device_get(tp),
            "opt_state": jax.device_get(opt),
            "scale": jax.device_get(scale),
            "glass_step": jax.device_get(glass_step),
            "key": jax.device_get(key),
            "glass_config": dict(glass_cfg) if use_glass else {},
            "best_mppi": best_mppi,
            "best_mppi_step": best_mppi_step,
            # iter-8 §2.0 — Phase-eval: persist best-pi + best-any with final
            "best_pi": best_pi,
            "best_pi_step": best_pi_step,
            "best_any": best_any,
            "best_any_step": best_any_step,
            "best_any_selector": best_any_selector,
        }
        save_pickle_atomic(ckpt_dir / "final.pkl", final_payload)
        if save_full_state:
            save_pickle_atomic(
                ckpt_dir / "final_full.pkl",
                {
                    **final_payload,
                    "replay_buffer": buffer_state(buf),
                    "env_state": jax.device_get(env_state),
                    "obs_np": obs_np,
                    "rng_np_state": rng_np.bit_generator.state,
                    "np_random_state": np.random.get_state(),
                },
            )
    print(f"  {algo_name} {env_id} done in {time.time()-t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=3_000_000)
    ap.add_argument("--seed",        type=int, default=1)
    ap.add_argument("--tasks",       nargs="+", default=TASKS,
                    help=f"Tasks to benchmark (default: {TASKS})")
    ap.add_argument("--algos",       nargs="+", default=["ppo", "sac", "tdmpc2"],
                    help="Algorithms to run (default: ppo sac tdmpc2; also supports tdmpc-glass)")
    ap.add_argument("--no_plot",     action="store_true",
                    help="Skip plotting after training")
    ap.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Resume TD-MPC2/TD-MPC-Glass model/optimizer state from a pickle checkpoint")
    ap.add_argument("--save_full_state", action="store_true",
                    help="Save replay buffer, vectorized env state, and RNG state for exact TD-MPC2/TD-MPC-Glass resume")
    ap.add_argument("--glass_warmup_env_steps", type=int, default=None,
                    help="Override TD-MPC-Glass warmup_env_steps")
    ap.add_argument("--glass_every_k_updates", type=int, default=None,
                    help="Override TD-MPC-Glass every_k_updates")
    ap.add_argument("--glass_proto_temperature", type=float, default=None,
                    help="Override TD-MPC-Glass proto_temperature")
    ap.add_argument("--glass_assignment_temperature", type=float, default=None,
                    help="Override TD-MPC-Glass assignment_temperature")
    ap.add_argument("--glass_lambda_se", type=float, default=None,
                    help="Override TD-MPC-Glass lambda_se")
    ap.add_argument("--glass_lambda_balance", type=float, default=None,
                    help="Override TD-MPC-Glass lambda_balance")
    ap.add_argument("--glass_lambda_temporal", type=float, default=None,
                    help="Override TD-MPC-Glass lambda_temporal")
    ap.add_argument("--glass_lambda_temp_stability", type=float, default=None,
                    help="Phase-g2: weight on the per-pair cosine-similarity penalty between consecutive "
                         "soft-cluster distributions. Penalises cluster oscillation within a single gait "
                         "phase (per blog §3). Default 0 = off; try 0.05.")
    ap.add_argument("--glass_stopgrad_graph", choices=["true", "false"], default=None,
                    help="Override TD-MPC-Glass stopgrad_graph")
    ap.add_argument("--glass_num_prototypes", type=int, default=None,
                    help="Override TD-MPC-Glass num_prototypes")
    ap.add_argument("--glass_num_clusters", type=int, default=None,
                    help="Override TD-MPC-Glass num_clusters")
    ap.add_argument("--glass_assign_logits_init_scale", type=float, default=None,
                    help="Override TD-MPC-Glass assign_logits_init_scale")
    ap.add_argument("--act_noise_start", type=float, default=None,
                    help="Initial exploration-noise std (default: 0.3, the EXPL_NOISE constant)")
    ap.add_argument("--act_noise_end", type=float, default=None,
                    help="Final exploration-noise std after annealing (default: same as --act_noise_start, i.e. no anneal)")
    ap.add_argument("--act_noise_anneal_steps", type=int, default=1_000_000,
                    help="Env-steps over which to linearly anneal noise from start to end (default: 1M)")
    ap.add_argument("--mppi_n_samples", type=int, default=None,
                    help="Path 9 / Phase-x: override MPPI sample count NS. Default 512. Try 2048 to test "
                         "whether stuck seeds are search-failure (limited samples) vs learning-failure.")
    ap.add_argument("--mppi_horizon", type=int, default=None,
                    help="Override MPPI horizon H (and training seq_len=H+1). Default: DEFAULTS['H']=3")
    ap.add_argument("--k_update", type=int, default=None,
                    help="Override TD-MPC2 K_UPDATE, the number of gradient updates per env batch. "
                         "Default: DEFAULTS['K_UPDATE']=64. Iteration 7 fair sweep tests 128 and 256.")
    ap.add_argument("--q_reset_steps", type=str, default=None,
                    help="Comma-separated env-step thresholds at which to re-init online Q + optimizer (REDQ-style). "
                         "Target Q (tp['q']) and other params are preserved. Example: '1000000,2000000,3000000'.")
    ap.add_argument("--latent_action_smooth_coef", type=float, default=0.0,
                    help="Latent action smoothing weight (Phase-f). Penalises ||pi(z_t)-pi(z_{t-1})||^2 in the policy "
                         "loss, computed over the dynamics rollout. 0 = off (default). Try 1e-3 for HopperHop.")
    ap.add_argument("--consistency_coef", type=float, default=None,
                    help="Override TD-MPC consistency-loss weight (Phase-g). Default: 2.0 (v13 stable). Try 1.0 to "
                         "let the model focus on TD instead of dynamics regularisation.")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="If > 0, halt training when no new best MPPI has been recorded in the last N env-steps. "
                         "Combine with a generous --total_steps cap (e.g. 10_000_000) to auto-stop on convergence.")
    ap.add_argument("--latent_smooth_warmup_env_steps", type=int, default=0,
                    help="Curriculum schedule for --latent_action_smooth_coef. While env_steps < N, smoothing "
                         "coef is forced to 0 (lets Glass basin lock undisturbed). At env_steps >= N, coef is "
                         "raised to --latent_action_smooth_coef and the JIT is rebuilt (~3 min compile cost, once).")
    ap.add_argument("--glass_decay_steps", type=int, default=0,
                    help="If > 0, turn Glass loss OFF after env_steps >= N (hybrid Phase-o). Glass active "
                         "during the basin-discovery + representation-learning window [warmup, N], then OFF "
                         "so encoder/dynamics can refine without the partition acting as inductive bias.")
    ap.add_argument("--expl_until", type=int, default=None,
                    help="Override the env_steps duration of initial random exploration (Phase-p, path 1). "
                         "Default: DEFAULTS['EXPL_UNTIL']=25000. Try 500000 to fill replay with diverse "
                         "random rollouts (foot-strike data) before policy takes over.")
    ap.add_argument("--expl_mix_decay_steps", type=int, default=0,
                    help="Benchmark-fair exploration schedule: use a per-env random-policy action mixture "
                         "whose random-action probability decays linearly from 1.0 to 0.0 over N env-steps. "
                         "When >0, this replaces the hard --expl_until random-action phase.")
    # Hierarchical Glass (iteration-4 §7.4 / Path 10).
    ap.add_argument("--glass_num_super_clusters", type=int, default=None,
                    help="If > 0, enable hierarchical Glass: a second-level partition over the K=8 clusters "
                         "into K_super super-clusters. 0 (default) = flat partition (current behaviour).")
    ap.add_argument("--glass_lambda_super_se", type=float, default=None,
                    help="Weight on the super-cluster 2D-SE loss. Try 5e-3 (same as λ_se) when hierarchical "
                         "is on. Has no effect if --glass_num_super_clusters is 0.")
    ap.add_argument("--glass_lambda_super_balance", type=float, default=None,
                    help="Weight on the super-cluster balance hinge loss. Try 1e-2 (same as λ_balance).")
    # Path 5 / Phase-t — reward shaping: penalise knee/torso/thigh contact with floor.
    ap.add_argument("--knee_penalty_coef", type=float, default=0.0,
                    help="Per-step training-only reward penalty when non-foot geoms (torso, nose, pelvis, "
                         "thigh, calf) are within penalty_threshold of the floor. Forces foot-hop technique. "
                         "Eval reward is unmodified. Try 0.1 for HopperHop.")
    ap.add_argument("--knee_penalty_threshold", type=float, default=0.15,
                    help="Z-coordinate threshold (m) below which non-foot geoms incur penalty. Default 0.15.")
    # iter-6 §7.C — Phase-r2 gait penalty bundle
    ap.add_argument("--gait_fall_penalty", type=float, default=0.0,
                    help="Phase-r2: subtract this from training reward each step when torso-foot "
                         "height < --gait_fall_height. Try 0.1.")
    ap.add_argument("--gait_fall_height", type=float, default=0.45,
                    help="Phase-r2: height threshold (m) below which fall penalty fires. Default 0.45.")
    ap.add_argument("--gait_action_smooth", type=float, default=0.0,
                    help="Phase-r2: subtract coef * mean((a_t - a_{t-1})**2) per env from training reward. "
                         "Encourages smooth env-space actions. Try 0.005.")
    # iter-6 §7.B — Phase-r1 soft-reward bundle (v1 = stand_bonus + linear anneal only;
    # speed curriculum + early bonus deferred to v2)
    ap.add_argument("--soft_stand_bonus", type=float, default=0.0,
                    help="Phase-r1: add coef * clip((h - floor)/(0.6 - floor), 0, 1) to training reward. "
                         "Smooth standing bonus that ramps in below the binary 0.6 m cutoff. Try 0.1.")
    ap.add_argument("--soft_stand_floor", type=float, default=0.4,
                    help="Phase-r1: height (m) at which stand_bonus begins to ramp up. Default 0.4.")
    ap.add_argument("--soft_anneal_steps", type=int, default=0,
                    help="Phase-r1: linearly fade --soft_stand_bonus weight from 1.0 -> 0.0 over [0, N] "
                         "env steps so the shaping disappears mid-training. 0 = no fade (full bonus all run).")
    # iter-7 §2.1 — Phase-ar auto-restart on plateau (basin-lottery escape)
    ap.add_argument("--restart_on_plateau", action="store_true",
                    help="Phase-ar: enable plateau-triggered restart of pi+q. At each "
                         "--restart_check_at boundary, if best MPPI is still below "
                         "--restart_threshold, re-init pi+q+target_q+opt (encoder, "
                         "dynamics, reward, replay buffer, env state preserved). "
                         "Up to --restart_max_attempts attempts per seed. Benchmark-fair.")
    ap.add_argument("--restart_check_at", type=int, default=1_000_000,
                    help="Phase-ar: plateau check fires every N env steps (default 1M). "
                         "First check at env_steps>=N; subsequent checks every additional N.")
    ap.add_argument("--restart_threshold", type=float, default=100.0,
                    help="Phase-ar: best MPPI floor below which restart fires (default 100). "
                         "100 is a sensible HopperHop dividing line between 'climbing' and 'basin-locked'.")
    ap.add_argument("--restart_max_attempts", type=int, default=3,
                    help="Phase-ar: max restarts per seed (default 3). Each attempt is a fresh pi+q.")
    ap.add_argument("--mpc_distill_coef", type=float, default=0.0,
                    help="Phase-mpc-lite: coefficient for MPPI-gated actor distillation. "
                         "Uses a small replay anchor batch and planner targets from the current model. 0 = off.")
    ap.add_argument("--mpc_distill_anneal_steps", type=int, default=3_000_000,
                    help="Phase-mpc-lite: linearly anneal distillation coef from its initial value to 0 "
                         "over N env-steps after --expl_until. Default 3M.")
    ap.add_argument("--mpc_distill_disable_gap", type=float, default=100.0,
                    help="Phase-mpc-lite: disable distillation after an eval when pi - mppi >= gap. "
                         "Re-enable automatically on a later eval if the gap drops back below this threshold.")
    ap.add_argument("--mpc_distill_batch_size", type=int, default=16,
                    help="Phase-mpc-lite: replay anchor batch size for planner targets per update cycle. "
                         "Small on purpose to keep MPPI target generation bounded.")
    ap.add_argument("--controller_arbitration", choices=["none", "eval_only"], default="none",
                    help="Iteration 10 i10-a1: log an eval-only arb series that chooses between pi and MPPI "
                         "at each eval. This does not change data collection or replay. Default: none.")
    ap.add_argument("--arbitration_margin", type=float, default=0.0,
                    help="Iteration 10 i10-a1: with --controller_arbitration=eval_only, select MPPI only "
                         "when MPPI reward exceeds pi reward by this margin. Default 0.")
    # Path P / Phase-P — cluster-entropy intrinsic reward (benchmark-fair, no env modification).
    ap.add_argument("--cluster_intrinsic_coef", type=float, default=0.0,
                    help="Path P: add coef * entropy(last W cluster ids) to training reward. Encourages "
                         "policy toward gaits visiting multiple distinct Glass clusters (= proper hopping "
                         "with multiple phases). Pure self-supervised; uses existing Glass partition. "
                         "Try 0.05-0.2 for HopperHop. 0 = off (default).")
    ap.add_argument("--cluster_intrinsic_window", type=int, default=20,
                    help="Window size W for the cluster-entropy intrinsic reward (Path P). Default 20 frames "
                         "(~0.4s at 50Hz). Larger window = longer-horizon diversity prior.")
    ap.add_argument("--cluster_intrinsic_decay_steps", type=int, default=0,
                    help="Phase-Pa: linearly decay --cluster_intrinsic_coef to 0 between --expl_until and "
                         "this env-step. 0 disables decay (static intrinsic, Phase-P behaviour). Recommended "
                         "3,000,000 so by 3M the policy trains on pure extrinsic reward.")
    ap.add_argument("--use_cluster_obs", action="store_true",
                    help="Path 7 / Phase-v: concat the soft Glass cluster distribution S[n_star(z)] "
                         "(K-dim) to z before pi/q lookups. Architectural change, fully benchmark-fair "
                         "(no reward modification). Policy can condition on which gait phase it's in. "
                         "stop_gradient on the cluster computation so Glass keeps its own structural loss.")
    return ap.parse_args()


def main():
    args = parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nBenchmark: {args.algos} × {args.tasks}")
    print(f"Steps: {args.total_steps:,}  Seed: {args.seed}")
    print(f"Output: {EXP_DIR}\n")

    t_total = time.time()
    glass_overrides = {
        "warmup_env_steps": args.glass_warmup_env_steps,
        "every_k_updates": args.glass_every_k_updates,
        "proto_temperature": args.glass_proto_temperature,
        "assignment_temperature": args.glass_assignment_temperature,
        "lambda_se": args.glass_lambda_se,
        "lambda_balance": args.glass_lambda_balance,
        "lambda_temporal": args.glass_lambda_temporal,
        "lambda_temp_stability": args.glass_lambda_temp_stability,
        "num_prototypes": args.glass_num_prototypes,
        "num_clusters": args.glass_num_clusters,
        "num_super_clusters": args.glass_num_super_clusters,
        "lambda_super_se": args.glass_lambda_super_se,
        "lambda_super_balance": args.glass_lambda_super_balance,
        "assign_logits_init_scale": args.glass_assign_logits_init_scale,
        "use_cluster_obs": args.use_cluster_obs,
    }
    if args.glass_stopgrad_graph is not None:
        glass_overrides["stopgrad_graph"] = args.glass_stopgrad_graph == "true"

    # Per-phase benchmark CSV: include TDMPC_GLASS_OUTPUT_TAG in the filename so
    # rows from different phases (phase-f, phase-j, etc.) end up in separate
    # files instead of all interleaving into one rollup.
    _tag = os.environ.get("TDMPC_GLASS_OUTPUT_TAG", "").strip()
    _suffix = f"_{_tag}" if _tag else ""
    failed = False
    for algo in args.algos:
        for task in args.tasks:
            csv_path = EXP_DIR / f"{algo}_{task}{_suffix}.csv"
            try:
                if algo == "ppo":
                    train_ppo(task, args.total_steps, args.seed, csv_path)
                elif algo == "sac":
                    train_sac(task, args.total_steps, args.seed, csv_path)
                elif algo == "tdmpc2":
                    # tdmpc2 path (no Glass). Iter 6 expanded to support the same
                    # config space as tdmpc-glass for direct ablation: NS, knee penalty,
                    # expl_until, latent smoothing, curriculum warmup.
                    train_tdmpc2(
                        task,
                        args.total_steps,
                        args.seed,
                        csv_path,
                        use_glass=False,
                        resume_checkpoint=args.resume_checkpoint,
                        save_full_state=args.save_full_state,
                        mppi_n_samples=args.mppi_n_samples,
                        k_update=args.k_update,
                        latent_action_smooth_coef=args.latent_action_smooth_coef,
                        consistency_coef=args.consistency_coef,
                        early_stop_patience=args.early_stop_patience,
                        latent_smooth_warmup_env_steps=args.latent_smooth_warmup_env_steps,
                        expl_until=args.expl_until,
                        expl_mix_decay_steps=args.expl_mix_decay_steps,
                        knee_penalty_coef=args.knee_penalty_coef,
                        knee_penalty_threshold=args.knee_penalty_threshold,
                        gait_fall_penalty=args.gait_fall_penalty,
                        gait_fall_height=args.gait_fall_height,
                        gait_action_smooth=args.gait_action_smooth,
                        soft_stand_bonus=args.soft_stand_bonus,
                        soft_stand_floor=args.soft_stand_floor,
                        soft_anneal_steps=args.soft_anneal_steps,
                        restart_on_plateau=args.restart_on_plateau,
                        restart_check_at=args.restart_check_at,
                        restart_threshold=args.restart_threshold,
                        restart_max_attempts=args.restart_max_attempts,
                        mpc_distill_coef=args.mpc_distill_coef,
                        mpc_distill_anneal_steps=args.mpc_distill_anneal_steps,
                        mpc_distill_disable_gap=args.mpc_distill_disable_gap,
                        mpc_distill_batch_size=args.mpc_distill_batch_size,
                        controller_arbitration=args.controller_arbitration,
                        arbitration_margin=args.arbitration_margin,
                    )
                elif algo in ("tdmpc-glass", "tdmpc_glass"):
                    q_reset = None
                    if args.q_reset_steps:
                        q_reset = sorted({int(s.strip()) for s in args.q_reset_steps.split(",") if s.strip()})
                    train_tdmpc2(
                        task,
                        args.total_steps,
                        args.seed,
                        csv_path,
                        use_glass=True,
                        resume_checkpoint=args.resume_checkpoint,
                        save_full_state=args.save_full_state,
                        glass_overrides=glass_overrides,
                        act_noise_start=args.act_noise_start,
                        act_noise_end=args.act_noise_end,
                        act_noise_anneal_steps=args.act_noise_anneal_steps,
                        mppi_horizon=args.mppi_horizon,
                        mppi_n_samples=args.mppi_n_samples,
                        k_update=args.k_update,
                        q_reset_steps=q_reset,
                        latent_action_smooth_coef=args.latent_action_smooth_coef,
                        consistency_coef=args.consistency_coef,
                        early_stop_patience=args.early_stop_patience,
                        latent_smooth_warmup_env_steps=args.latent_smooth_warmup_env_steps,
                        glass_decay_steps=args.glass_decay_steps,
                        expl_until=args.expl_until,
                        expl_mix_decay_steps=args.expl_mix_decay_steps,
                        knee_penalty_coef=args.knee_penalty_coef,
                        knee_penalty_threshold=args.knee_penalty_threshold,
                        cluster_intrinsic_coef=args.cluster_intrinsic_coef,
                        cluster_intrinsic_window=args.cluster_intrinsic_window,
                        cluster_intrinsic_decay_steps=args.cluster_intrinsic_decay_steps,
                        gait_fall_penalty=args.gait_fall_penalty,
                        gait_fall_height=args.gait_fall_height,
                        gait_action_smooth=args.gait_action_smooth,
                        soft_stand_bonus=args.soft_stand_bonus,
                        soft_stand_floor=args.soft_stand_floor,
                        soft_anneal_steps=args.soft_anneal_steps,
                        restart_on_plateau=args.restart_on_plateau,
                        restart_check_at=args.restart_check_at,
                        restart_threshold=args.restart_threshold,
                        restart_max_attempts=args.restart_max_attempts,
                        mpc_distill_coef=args.mpc_distill_coef,
                        mpc_distill_anneal_steps=args.mpc_distill_anneal_steps,
                        mpc_distill_disable_gap=args.mpc_distill_disable_gap,
                        mpc_distill_batch_size=args.mpc_distill_batch_size,
                        controller_arbitration=args.controller_arbitration,
                        arbitration_margin=args.arbitration_margin,
                    )
                else:
                    print(f"Unknown algo: {algo}", flush=True)
            except Exception as e:
                failed = True
                print(f"\nERROR in {algo}/{task}: {e}", flush=True)
                import traceback; traceback.print_exc()
            # Encourage GC between runs
            import gc; gc.collect()

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All runs completed in {elapsed/60:.1f} min")
    print(f"CSVs saved to: {EXP_DIR}")

    if not args.no_plot:
        plot_script = Path(__file__).parent / "plot_comparison.py"
        if plot_script.exists():
            import subprocess
            out_dir = EXP_DIR / "plots"
            cmd = [sys.executable, str(plot_script),
                   "--exp_dir", str(EXP_DIR),
                   "--out_dir", str(out_dir)]
            print(f"\nGenerating comparison plot → {out_dir}")
            subprocess.run(cmd, check=True)
        else:
            print(f"\nNo plot script found at {plot_script}. Run manually.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
