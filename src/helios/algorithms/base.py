"""Abstract base class for all helios-rl agents."""

from __future__ import annotations

import abc
from typing import Any

import jax


class BaseAgent(abc.ABC):
    """Contract that every helios-rl agent must satisfy.

    All state (network parameters, optimizer state, replay buffer, etc.) is
    stored in a plain Python/JAX-compatible ``state`` dict so that agents
    can be serialised, copied, and JIT-compiled without side-effects.

    Sub-classes should implement :meth:`act` and :meth:`update`.
    """

    def __init__(
        self,
        config: Any,
        observation_space: Any,
        action_space: Any,
    ) -> None:
        """Initialise the agent.

        Args:
            config: A Hydra ``DictConfig`` (or plain dict) containing all
                hyperparameters for this agent.
            observation_space: A Gymnasium ``Space`` describing observations.
            action_space: A Gymnasium ``Space`` describing actions.
        """
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def initial_state(self, key: jax.Array) -> dict[str, Any]:
        """Create the initial agent state (parameters + optimizer states).

        Args:
            key: JAX PRNG key used for weight initialisation.

        Returns:
            A serialisable state dict.
        """

    @abc.abstractmethod
    def act(
        self,
        obs: jax.Array,
        state: dict[str, Any],
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Select an action given an observation.

        Args:
            obs: Observation array, shape (batch, *obs_shape) or (batch, *obs_shape).
            state: Current agent state (e.g. hidden state for recurrent agents).
            key: JAX PRNG key for stochastic action sampling.
            deterministic: If True, return the mode/greedy action; no exploration.

        Returns:
            Tuple of ``(action, next_hidden_state)``.
        """

    @abc.abstractmethod
    def update(
        self,
        batch: dict[str, jax.Array],
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Update agent parameters from a batch of experience.

        Args:
            batch: Dictionary of arrays (obs, action, reward, next_obs, done, …).
            state: Current agent state containing network parameters and
                optimizer states.

        Returns:
            Tuple of ``(new_state, metrics_dict)`` where ``metrics_dict``
            maps metric names to scalar float values for logging.
        """
