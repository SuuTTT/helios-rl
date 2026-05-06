"""Sequence-based replay buffer for world-model algorithms.

Stores complete *episodes* as contiguous sequences and samples random
sub-sequences of a fixed length.  Suitable for DreamerV3 and TD-MPC2 which
need temporal context to train the RSSM / latent dynamics.

The implementation uses a circular buffer backed by NumPy arrays for
memory efficiency and copies to JAX on demand.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp


class SequenceBatch(NamedTuple):
    """A batch of fixed-length sequences sampled from the buffer.

    All arrays have shape ``(batch_size, seq_len, ...)``.
    """

    obs: jax.Array        # (B, L, *obs_shape)
    actions: jax.Array    # (B, L, *act_shape)
    rewards: jax.Array    # (B, L)
    dones: jax.Array      # (B, L)


class TrajectoryBuffer:
    """Circular replay buffer storing variable-length episodes.

    Episodes are appended sequentially into a flat ring buffer.  A separate
    *index* array records the start of each valid sub-sequence of length
    ``seq_len``, enabling O(1) random sampling.

    Args:
        capacity: Maximum total number of transitions stored.
        obs_shape: Shape of a single observation.
        action_shape: Shape of a single action.
        seq_len: Length of sequences returned by :meth:`sample`.
        dtype_obs: NumPy dtype for observations (default float32).
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        seq_len: int = 64,
        dtype_obs: np.dtype = np.float32,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.seq_len = seq_len

        # Pre-allocate storage
        self._obs = np.zeros((capacity,) + obs_shape, dtype=dtype_obs)
        self._actions = np.zeros((capacity,) + action_shape, dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        # Circular buffer state
        self._ptr = 0          # next write position
        self._size = 0         # current number of valid steps

        # Tracks which positions are *episode boundaries* (done=True).
        # Sequences that wrap around episode ends are excluded.
        self._episode_ends: set[int] = set()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            obs: Observation array, shape obs_shape.
            action: Action array, shape action_shape.
            reward: Scalar reward.
            done: True if this transition ends an episode.
        """
        idx = self._ptr
        self._obs[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._dones[idx] = float(done)

        if done:
            self._episode_ends.add(idx)
        else:
            self._episode_ends.discard(idx)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Convenience method: store an entire episode at once.

        Args:
            obs: Shape (T, *obs_shape).
            actions: Shape (T, *act_shape).
            rewards: Shape (T,).
            dones: Shape (T,).
        """
        for t in range(len(rewards)):
            self.add_transition(obs[t], actions[t], float(rewards[t]), bool(dones[t]))

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def can_sample(self, batch_size: int) -> bool:
        """True if enough transitions exist for at least ``batch_size`` sequences."""
        return self._size >= self.seq_len * batch_size

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> SequenceBatch:
        """Sample a batch of random sub-sequences.

        Args:
            batch_size: Number of sequences to return.
            rng: Optional NumPy random generator for reproducibility.

        Returns:
            :class:`SequenceBatch` with JAX arrays.
        """
        if rng is None:
            rng = np.random.default_rng()

        valid_size = self._size - self.seq_len + 1
        if valid_size <= 0:
            raise RuntimeError(
                f"Buffer too small to sample sequences of length {self.seq_len}. "
                f"Current size: {self._size}"
            )

        # Collect valid start indices (no episode boundary within the window)
        starts = self._sample_starts(batch_size, valid_size, rng)

        obs_batch = np.zeros((batch_size, self.seq_len) + self.obs_shape, dtype=self._obs.dtype)
        act_batch = np.zeros((batch_size, self.seq_len) + self.action_shape, dtype=np.float32)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        for i, start in enumerate(starts):
            indices = np.arange(start, start + self.seq_len) % self.capacity
            obs_batch[i] = self._obs[indices]
            act_batch[i] = self._actions[indices]
            rew_batch[i] = self._rewards[indices]
            done_batch[i] = self._dones[indices]

        return SequenceBatch(
            obs=jnp.asarray(obs_batch),
            actions=jnp.asarray(act_batch),
            rewards=jnp.asarray(rew_batch),
            dones=jnp.asarray(done_batch),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_starts(
        self,
        batch_size: int,
        valid_size: int,
        rng: np.random.Generator,
        max_retries: int = 10,
    ) -> list[int]:
        """Sample ``batch_size`` start indices that avoid episode boundaries."""
        starts: list[int] = []
        for _ in range(max_retries * batch_size):
            if len(starts) >= batch_size:
                break
            candidate = rng.integers(0, valid_size)
            indices = set(range(candidate, candidate + self.seq_len))
            wrapped = {i % self.capacity for i in indices}
            # Reject if any episode end falls strictly inside the window
            # (allow the very last step to be a "done" as the reward is valid)
            if not (self._episode_ends & {i % self.capacity for i in range(candidate, candidate + self.seq_len - 1)}):
                starts.append(candidate)
        # If we couldn't find enough valid starts, fall back to unconstrained
        while len(starts) < batch_size:
            starts.append(rng.integers(0, valid_size))
        return starts[:batch_size]
