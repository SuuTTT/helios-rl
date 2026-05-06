"""helios-rl training entry point.

Usage (with Hydra):
    python -m helios.main agent=ppo env=mujoco
    python -m helios.main agent=dreamer_v3 env=dm_control
    python -m helios.main agent=tdmpc2 env=mujoco
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(cfg: DictConfig):
    """Create a vectorised Gymnasium environment from config.

    Args:
        cfg: Environment sub-config (env.*).

    Returns:
        Tuple ``(env, obs_space, act_space)``.
    """
    import gymnasium as gym

    env_name = cfg.env_id
    num_envs = int(cfg.num_envs)

    def _make_single():
        env = gym.make(env_name, max_episode_steps=int(cfg.max_episode_steps))
        return env

    if num_envs == 1:
        env = _make_single()
    else:
        env = gym.vector.SyncVectorEnv([_make_single for _ in range(num_envs)])

    obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
    act_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space

    return env, obs_space, act_space


def make_agent(cfg: DictConfig, obs_space, act_space):
    """Instantiate the configured agent.

    Args:
        cfg: Full experiment config.
        obs_space: Observation space.
        act_space: Action space.

    Returns:
        :class:`~helios.algorithms.base.BaseAgent` instance.
    """
    agent_name = cfg.agent.name

    if agent_name == "ppo":
        from helios.algorithms.ppo import PPOAgent

        return PPOAgent(cfg.agent, obs_space, act_space)
    elif agent_name == "dreamer_v3":
        from helios.algorithms.dreamer import DreamerV3Agent

        return DreamerV3Agent(cfg.agent, obs_space, act_space)
    elif agent_name == "tdmpc2":
        from helios.algorithms.tdmpc import TDMPCAgent

        return TDMPCAgent(cfg.agent, obs_space, act_space)
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose ppo | dreamer_v3 | tdmpc2.")


def make_buffer(cfg: DictConfig, obs_space, act_space):
    """Create the appropriate replay buffer for the configured agent.

    PPO uses :class:`~helios.memory.rollout.RolloutBuffer`;
    world-model agents use :class:`~helios.memory.trajectory.TrajectoryBuffer`.

    Args:
        cfg: Full experiment config.
        obs_space: Observation space.
        act_space: Action space.

    Returns:
        Buffer instance.
    """
    import gymnasium as gym

    obs_shape = obs_space.shape
    act_shape = act_space.shape if isinstance(act_space, gym.spaces.Box) else (1,)

    if cfg.agent.name == "ppo":
        from helios.memory.rollout import RolloutBuffer

        return RolloutBuffer(
            num_steps=int(cfg.agent.num_steps),
            num_envs=int(cfg.agent.num_envs),
            obs_shape=obs_shape,
            action_shape=act_shape,
            gamma=float(cfg.agent.gamma),
            gae_lambda=float(cfg.agent.gae_lambda),
        )
    else:
        from helios.memory.trajectory import TrajectoryBuffer

        seq_len = int(getattr(cfg.agent, "batch_length", 64))
        capacity = int(getattr(cfg.agent, "buffer_size", 1_000_000))
        return TrajectoryBuffer(
            capacity=capacity,
            obs_shape=obs_shape,
            action_shape=act_shape,
            seq_len=seq_len,
        )


def log_metrics(metrics: dict[str, float], step: int, wandb_run=None) -> None:
    """Log metrics to stdout and optionally to Weights & Biases.

    Args:
        metrics: Dict of metric name → scalar value.
        step: Global step counter.
        wandb_run: Active W&B run (or None to skip).
    """
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    log.info("step=%d  %s", step, metric_str)
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Warm-up rollout
# ---------------------------------------------------------------------------


def warmup_random(env, buffer, num_steps: int) -> None:
    """Collect random transitions to seed the replay buffer.

    Args:
        env: Gymnasium (vector) environment.
        buffer: Buffer with an ``add_transition`` or ``add`` method.
        num_steps: Number of steps to collect.
    """
    from helios.memory.rollout import RolloutBuffer
    from helios.memory.trajectory import TrajectoryBuffer

    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)

        if isinstance(buffer, TrajectoryBuffer):
            if obs.ndim == 1:
                buffer.add_transition(obs, action, float(reward), bool(done))
            else:
                for i in range(obs.shape[0]):
                    buffer.add_transition(obs[i], action[i], float(reward[i]), bool(done[i]))
        obs = next_obs if not np.all(done) else env.reset()[0]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@hydra.main(config_path="../../configs", config_name="experiment/default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Training entry point driven by a Hydra config.

    Execution flow:
    1. Initialise: config, PRNG, env, buffer, agent.
    2. Warm-up: collect random steps.
    3. Main loop:
       a. act → step env → store.
       b. Every ``train_freq`` steps: sample batch, update agent.
       c. Every ``log_interval`` steps: log metrics.
       d. Every ``eval_interval`` steps: run evaluation episodes.
    """
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ---- Reproducibility ----
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # ---- Weights & Biases ----
    wandb_run = None
    if cfg.wandb.mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity or None,
                config=OmegaConf.to_container(cfg, resolve=True),
                tags=list(cfg.wandb.tags),
                mode=cfg.wandb.mode,
            )
        except ImportError:
            log.warning("wandb not installed; skipping W&B logging.")

    # ---- Environment ----
    env, obs_space, act_space = make_env(cfg.env)
    eval_env, _, _ = make_env(cfg.env)

    # ---- Buffer ----
    buffer = make_buffer(cfg, obs_space, act_space)

    # ---- Agent ----
    agent = make_agent(cfg, obs_space, act_space)
    key, init_key = jax.random.split(key)
    agent_state = agent.initial_state(init_key)

    # ---- Warm-up ----
    warmup_steps = int(getattr(cfg, "warmup_steps", 1000))
    log.info("Collecting %d random warm-up steps …", warmup_steps)
    warmup_random(env, buffer, warmup_steps)

    # ---- Training loop ----
    total_steps = int(cfg.total_steps)
    log_interval = int(cfg.log_interval)
    eval_interval = int(cfg.eval_interval)
    eval_episodes = int(cfg.eval_episodes)
    train_freq = int(getattr(cfg.agent, "num_steps", 1))  # PPO collects full rollout first

    obs, _ = env.reset(seed=seed)
    global_step = 0
    episode_rewards: list[float] = []
    ep_reward = np.zeros(int(cfg.env.num_envs))

    all_metrics: dict[str, float] = {}

    log.info("Starting training for %d steps …", total_steps)
    t_start = time.time()

    while global_step < total_steps:
        # -- Interact --
        key, act_key = jax.random.split(key)
        action, hidden = agent.act(jnp.asarray(obs), agent_state, act_key)
        action_np = np.array(action)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = np.logical_or(terminated, truncated)

        # -- Store --
        _store_transition(buffer, agent_state, obs, action_np, reward, done, hidden, cfg)

        obs = next_obs
        ep_reward += reward
        global_step += int(cfg.env.num_envs)

        # Track episode returns
        for i, d in enumerate(np.atleast_1d(done)):
            if d:
                episode_rewards.append(float(ep_reward[i]) if ep_reward.ndim > 0 else float(ep_reward))
                ep_reward_i = ep_reward[i] if ep_reward.ndim > 0 else ep_reward
                ep_reward = ep_reward.at[i].set(0) if hasattr(ep_reward, "at") else ep_reward.__setitem__(i, 0) or ep_reward

        # -- Train --
        if global_step % train_freq == 0 and _can_train(buffer, cfg):
            batch = _sample_batch(buffer, agent_state, obs, done, hidden, cfg)
            if batch is not None:
                agent_state, metrics = agent.update(batch, agent_state)
                all_metrics.update(metrics)

        # -- Log --
        if global_step % log_interval == 0:
            if episode_rewards:
                all_metrics["train/ep_reward_mean"] = float(np.mean(episode_rewards[-100:]))
                all_metrics["train/ep_reward_max"] = float(np.max(episode_rewards[-100:]))
            all_metrics["train/steps_per_sec"] = global_step / (time.time() - t_start)
            log_metrics(all_metrics, global_step, wandb_run)

        # -- Evaluate --
        if global_step % eval_interval == 0:
            eval_return = evaluate(eval_env, agent, agent_state, eval_episodes, key)
            all_metrics["eval/ep_reward_mean"] = eval_return
            log.info("step=%d  eval_return=%.2f", global_step, eval_return)
            if wandb_run is not None:
                wandb_run.log({"eval/ep_reward_mean": eval_return}, step=global_step)

    log.info("Training complete.  Total steps: %d", global_step)
    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Helpers for the training loop
# ---------------------------------------------------------------------------


def _store_transition(buffer, agent_state, obs, action, reward, done, hidden, cfg) -> None:
    """Route transition storage to the correct buffer type."""
    from helios.memory.rollout import RolloutBuffer
    from helios.memory.trajectory import TrajectoryBuffer

    if isinstance(buffer, RolloutBuffer):
        value = np.array(hidden.get("value", np.zeros(1)))
        log_prob = np.array(hidden.get("log_prob", np.zeros(1)))
        buffer.add(
            jnp.asarray(obs),
            jnp.asarray(action),
            jnp.asarray(reward),
            jnp.asarray(done.astype(np.float32)),
            jnp.asarray(value),
            jnp.asarray(log_prob),
        )
    elif isinstance(buffer, TrajectoryBuffer):
        num_envs = obs.shape[0] if obs.ndim > 1 else 1
        if num_envs == 1:
            buffer.add_transition(obs, action, float(reward), bool(done))
        else:
            for i in range(num_envs):
                buffer.add_transition(obs[i], action[i], float(reward[i]), bool(done[i]))


def _can_train(buffer, cfg) -> bool:
    """Check whether there is enough data to start training."""
    from helios.memory.rollout import RolloutBuffer
    from helios.memory.trajectory import TrajectoryBuffer

    if isinstance(buffer, RolloutBuffer):
        return buffer.full
    elif isinstance(buffer, TrajectoryBuffer):
        batch_size = int(getattr(cfg.agent, "batch_size", 16))
        return buffer.can_sample(batch_size)
    return False


def _sample_batch(buffer, agent_state, obs, done, hidden, cfg):
    """Sample a training batch from the buffer."""
    from helios.memory.rollout import RolloutBuffer
    from helios.memory.trajectory import TrajectoryBuffer

    if isinstance(buffer, RolloutBuffer) and buffer.full:
        last_value = np.array(hidden.get("value", np.zeros(int(cfg.agent.num_envs))))
        last_done = done.astype(np.float32) if done.ndim > 0 else np.array([float(done)])
        return buffer.get(jnp.asarray(last_value), jnp.asarray(last_done))
    elif isinstance(buffer, TrajectoryBuffer):
        batch_size = int(getattr(cfg.agent, "batch_size", 16))
        if buffer.can_sample(batch_size):
            return buffer.sample(batch_size)
    return None


def evaluate(env, agent, agent_state, num_episodes: int, key: jax.Array) -> float:
    """Run ``num_episodes`` evaluation episodes and return mean return.

    Args:
        env: Gymnasium (vector) environment.
        agent: Agent instance.
        agent_state: Current agent state.
        num_episodes: Number of complete episodes to run.
        key: PRNG key.

    Returns:
        Mean episode return over all evaluated episodes.
    """
    total_return = 0.0
    episodes_done = 0

    obs, _ = env.reset()
    ep_return = 0.0

    while episodes_done < num_episodes:
        key, act_key = jax.random.split(key)
        action, _ = agent.act(jnp.asarray(obs), agent_state, act_key, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(np.array(action))
        ep_return += float(np.mean(reward))
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            total_return += ep_return
            episodes_done += 1
            ep_return = 0.0
            obs, _ = env.reset()

    return total_return / max(episodes_done, 1)


if __name__ == "__main__":
    main()
