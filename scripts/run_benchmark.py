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
    K_UPDATE   = d["K_UPDATE"]     # 64
    BS         = d["BS"]           # 256
    N_ENVS     = d["N_ENVS"]       # 256
    WARMUP     = d["WARMUP_ENV"]   # 25_000
    EXPL_NOISE = d["EXPL_NOISE"]   # 0.3
    EXPL_UNTIL = d["EXPL_UNTIL"]   # 25_000
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
    H          = d["H"]            # 3
    NS         = d["NS"]           # 512
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
    params = {
        "enc": enc_net.init(ek, dummy_obs),
        "dyn": dyn_net.init(dk, dummy_z, dummy_act),
        "rew": rew_net.init(rk, dummy_z, dummy_act),
        "q":   q_net.init(qk, dummy_z, dummy_act),
        "pi":  pi_net.init(pk, dummy_z),
    }
    if use_glass:
        params["glass"] = init_glass_params(
            gk,
            latent_dim=latent_dim,
            num_prototypes=glass_cfg.get("num_prototypes", 32),
            num_clusters=glass_cfg.get("num_clusters", 8),
            assign_logits_init_scale=glass_cfg.get("assign_logits_init_scale", 1.0),
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
        resume_best_mppi = float(resume_payload.get("mppi_reward", -float("inf")))
        print(
            f"  Resumed model checkpoint {resume_checkpoint} "
            f"at env_steps={resume_env_steps:,}",
            flush=True,
        )

    # ── Library update functions
    if use_glass:
        _, multi_step = make_update_fn(
            enc_net, dyn_net, rew_net, q_net, pi_net, tx,
            gamma=gamma, rho=rho, tau=tau, rew_scale=rew_scale,
            glass_enabled=glass_cfg.get("enabled", True),
            glass_every_k_updates=glass_cfg.get("every_k_updates", 4),
            glass_proto_temperature=glass_cfg.get("proto_temperature", 1.0),
            glass_assignment_temperature=glass_cfg.get("assignment_temperature", 1.0),
            glass_lambda_se=glass_cfg.get("lambda_se", 5.0e-3),
            glass_lambda_balance=glass_cfg.get("lambda_balance", 1.0e-2),
            glass_lambda_temporal=glass_cfg.get("lambda_temporal", 1.0e-3),
            glass_stopgrad_graph=glass_cfg.get("stopgrad_graph", True),
            glass_use_cosine_assign=glass_cfg.get("use_cosine_assign", True),
        )
    else:
        _, multi_step = make_update_fn(
            enc_net, dyn_net, rew_net, q_net, pi_net, tx,
            gamma=gamma, rho=rho, tau=tau, rew_scale=rew_scale,
        )

    # ── MPPI planner (for eval)
    plan = make_mppi_fn(
        enc_net, dyn_net, rew_net, q_net, pi_net,
        horizon=H, n_samples=NS, num_elites=elites,
        num_pi_trajs=pi_trajs, n_iter=NI,
        min_std=MIN_STD, max_std=MAX_STD,
        act_low=al, act_high=ah, act_dim=act_dim,
        gamma=gamma, rew_scale=rew_scale,
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
    @jax.jit
    def act_fn_batch(p, obs):
        z = enc_net.apply(p["enc"], obs)
        mu, _ = pi_net.apply(p["pi"], z)
        return jnp.tanh(mu)

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
        mu, _ = pi_net.apply(p["pi"], z)
        return jnp.tanh(mu[0])

    @jax.jit
    def single_env_step(state, act):
        return env.step(state, act[None])

    @jax.jit
    def single_env_reset(key):
        return env.reset(jax.random.split(key, 1))

    def eval_pi(n_eps: int = 5) -> float:
        nonlocal key
        rets = []
        for _ in range(n_eps):
            key, rk2 = jax.random.split(key)
            state = single_env_reset(rk2)
            obs   = jnp.asarray(state.obs[0])
            er = 0.0
            for _ in range(episode_length):
                z   = enc_apply(params, obs)
                act = pi_apply(params, z)
                state = single_env_step(state, act)
                er += float(state.reward[0])
                if bool(state.done[0] > 0.5):
                    break
                obs = jnp.asarray(state.obs[0])
            rets.append(er)
        return float(np.mean(rets))

    def eval_mppi(n_eps: int = 3) -> float:
        nonlocal key
        rets = []
        for _ in range(n_eps):
            key, rk2 = jax.random.split(key)
            state = single_env_reset(rk2)
            obs = jnp.asarray(state.obs[0])
            mu = jnp.zeros((H, act_dim))
            std = jnp.full((H, act_dim), MAX_STD)
            er = 0.0
            t0_mppi = jnp.bool_(True)
            key, pk2 = jax.random.split(key)
            for _ in range(episode_length):
                act, mu, std = plan(params, obs, mu, std, pk2, t0_mppi)
                t0_mppi = jnp.bool_(False)
                key, pk2 = jax.random.split(key)
                state = single_env_step(state, act)
                er += float(state.reward[0])
                if bool(state.done[0] > 0.5):
                    break
                obs = jnp.asarray(state.obs[0])
            rets.append(er)
        return float(np.mean(rets))

    eval_type_csv = None
    ckpt_dir = None
    best_mppi = resume_best_mppi
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
        if not (resume_checkpoint and eval_type_csv.exists()):
            with open(eval_type_csv, "w") as cf:
                cf.write("step,reward,eval_type,seed\n")
    elif env_id == "HopperHop":
        eval_type_csv = EXP_DIR.parent / "tdmpc_dmc" / "hopper-hop-tdmpc2-rerun.csv"
        eval_type_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_type_csv, "w") as cf:
            cf.write("step,reward,eval_type,seed\n")

    if use_glass:
        ckpt_dir = (
            EXP_DIR.parent
            / "tdmpc_glass"
            / _env_dir
            / f"seed_{seed}"
            / "checkpoints"
        )

    # ── Warmup JIT — compile multi_step with dummy data
    print("  Warming up JIT (may take 30-90s)...", flush=True)
    t_jit = time.time()
    _dummy_obs = np.zeros((K_UPDATE, BS, seq_len, obs_dim), np.float32)
    _dummy_act = np.zeros((K_UPDATE, BS, seq_len, act_dim), np.float32)
    _dummy_rew = np.zeros((K_UPDATE, BS, seq_len), np.float32)
    _dummy_don = np.zeros((K_UPDATE, BS, seq_len), np.float32)
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

    with open_csv(csv_path, env_id, seed) as fh:
        while env_steps < total_steps:
            # Collect N_ENVS steps
            if env_steps < EXPL_UNTIL:
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
            buf.add_batch(obs_np, acts_np, rews_np, done_np)
            obs_np    = new_obs
            env_steps += N_ENVS

            # Gradient updates (one vectorised sample + one H2D + lax.scan)
            samp_k = buf.sample_k(K_UPDATE, BS, rng_np)
            if samp_k is not None:
                ob_k, ab_k, rb_k, db_k = [jnp.asarray(x) for x in samp_k]
                if use_glass:
                    glass_active = env_steps >= glass_cfg.get("warmup_env_steps", 100_000)
                    params, tp, opt, key, scale, glass_step, loss_val, aux = multi_step(
                        params, tp, opt, ob_k, ab_k, rb_k, db_k, key, scale,
                        glass_step, glass_active,
                    )
                else:
                    params, tp, opt, key, scale, loss_val, aux = multi_step(
                        params, tp, opt, ob_k, ab_k, rb_k, db_k, key, scale,
                    )

            if env_steps % log_interval < N_ENVS:
                elapsed = time.time() - t0
                print(f"  es={env_steps:>9,}  sps={env_steps/max(elapsed,1):.0f}"
                      f"  loss={float(loss_val):.4f}  scale={float(scale):.2f}", flush=True)

            if env_steps >= next_eval:
                ret = eval_pi(n_eps=5)
                mppi_ret = eval_mppi(n_eps=8 if use_glass else 3)
                elapsed = time.time() - t0
                print(f"  step={env_steps:>9,}  pi_reward={ret:7.1f}"
                      f"  MPPI={mppi_ret:7.1f}"
                      f"  sps={env_steps/max(elapsed,1):.0f}", flush=True)
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
                if ckpt_dir is not None:
                    ckpt_payload = {
                        "algo": "tdmpc-glass",
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
                        "glass_config": dict(glass_cfg),
                    }
                    save_pickle_atomic(ckpt_dir / "latest_eval.pkl", ckpt_payload)
                    if mppi_ret > best_mppi:
                        best_mppi = mppi_ret
                        save_pickle_atomic(ckpt_dir / "best_mppi.pkl", ckpt_payload)
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

    if ckpt_dir is not None:
        final_payload = {
            "algo": "tdmpc-glass",
            "env_id": env_id,
            "seed": seed,
            "env_steps": env_steps,
            "params": jax.device_get(params),
            "target_params": jax.device_get(tp),
            "opt_state": jax.device_get(opt),
            "scale": jax.device_get(scale),
            "glass_step": jax.device_get(glass_step),
            "key": jax.device_get(key),
            "glass_config": dict(glass_cfg),
            "best_mppi": best_mppi,
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
                    help="Resume TD-MPC-Glass model/optimizer state from a pickle checkpoint")
    ap.add_argument("--save_full_state", action="store_true",
                    help="Save replay buffer, vectorized env state, and RNG state for exact TD-MPC-Glass resume")
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
        "num_prototypes": args.glass_num_prototypes,
        "num_clusters": args.glass_num_clusters,
        "assign_logits_init_scale": args.glass_assign_logits_init_scale,
    }
    if args.glass_stopgrad_graph is not None:
        glass_overrides["stopgrad_graph"] = args.glass_stopgrad_graph == "true"

    failed = False
    for algo in args.algos:
        for task in args.tasks:
            csv_path = EXP_DIR / f"{algo}_{task}.csv"
            try:
                if algo == "ppo":
                    train_ppo(task, args.total_steps, args.seed, csv_path)
                elif algo == "sac":
                    train_sac(task, args.total_steps, args.seed, csv_path)
                elif algo == "tdmpc2":
                    train_tdmpc2(task, args.total_steps, args.seed, csv_path)
                elif algo in ("tdmpc-glass", "tdmpc_glass"):
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
