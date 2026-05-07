import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from flax import struct

@struct.dataclass
class LogEnvState:
    env_state: jax.Array
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray

class LogWrapper:
    """Log the episode returns and lengths."""

    def __init__(self, env):
        self._env = env

    @property
    def default_params(self):
        return self._env.default_params

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done) + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((), dtype=jnp.float32),
            episode_lengths=jnp.zeros((), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros((), dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros((), dtype=jnp.int32),
        )
        return obs, state
