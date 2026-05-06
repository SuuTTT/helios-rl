"""Recurrent State Space Model (RSSM) for DreamerV3.

The RSSM factorises the latent state into:
- **Deterministic** component ``h``: updated by a GRU given the previous
  state and action.  Provides long-term memory.
- **Stochastic** component ``z``: sampled from a categorical (discrete)
  distribution conditioned on ``h`` (prior) or ``h + encoded_obs`` (posterior).

References:
- DreamerV3: Mastering Diverse Domains through World Models (Hafner et al. 2023)
  https://arxiv.org/abs/2301.04104
- RSSM original: Learning Latent Dynamics for Planning from Pixels (Hafner et al. 2019)
  https://arxiv.org/abs/1811.04551
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from helios.core.distributions import OneHotCategorical
from helios.core.networks import MLP, NormedLinear
from helios.dynamics.base import BaseDynamics


# ---------------------------------------------------------------------------
# RSSM Flax module
# ---------------------------------------------------------------------------


class RSSMModule(nn.Module):
    """Core RSSM computation implemented as a Flax Module.

    The module is *stateless*: it takes the previous state and outputs the
    next one, making it easy to vmap/scan over time.

    Args:
        deter_dim: GRU hidden size (deterministic state).
        stoch_dim: Number of stochastic latent variables.
        stoch_classes: Number of categories per latent variable.
        hidden_dim: Width of MLP layers for prior/posterior heads.
        embed_dim: Size of the observation embedding (from CNN or MLP encoder).
    """

    deter_dim: int = 512
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 512
    embed_dim: int = 512

    def setup(self) -> None:
        stoch_size = self.stoch_dim * self.stoch_classes
        # GRU input projection and cell
        self.inp_proj = NormedLinear(self.hidden_dim)
        self.gru_cell = nn.GRUCell(self.deter_dim)
        # Prior head: h -> prior logits
        self.prior_norm = NormedLinear(self.hidden_dim)
        self.prior_head = nn.Dense(stoch_size)
        # Posterior head: [h, embed] -> posterior logits
        self.post_norm = NormedLinear(self.hidden_dim)
        self.post_head = nn.Dense(stoch_size)

    @property
    def stoch_size(self) -> int:
        return self.stoch_dim * self.stoch_classes

    def __call__(
        self,
        prev_h: jax.Array,
        prev_z: jax.Array,
        action: jax.Array,
        embed: jax.Array | None = None,
        key: jax.Array | None = None,
        use_straight_through: bool = True,
    ) -> dict[str, jax.Array]:
        """Forward pass.

        Args:
            prev_h: Previous deterministic state, shape (batch, deter_dim).
            prev_z: Previous stochastic state, shape (batch, stoch_dim, stoch_classes).
            action: Action, shape (batch, action_dim).
            embed: Observation embedding, shape (batch, embed_dim).  If None,
                   only the prior is computed (imagination mode).
            key: PRNG key for sampling z.  Required when ``embed`` is not None
                 and a posterior sample is requested.
            use_straight_through: Pass straight-through gradients through z.

        Returns:
            Dict with keys:
            - ``h``: new deterministic state.
            - ``z``: new stochastic state (posterior if embed given, else prior).
            - ``prior_logits``: prior distribution logits.
            - ``post_logits``: posterior logits (only when embed provided).
        """
        stoch_size = self.stoch_dim * self.stoch_classes

        batch = prev_z.shape[0]

        # 1. Deterministic step via GRU
        z_flat = prev_z.reshape(batch, -1)
        inp = self.inp_proj(jnp.concatenate([z_flat, action], axis=-1))
        # nn.GRUCell returns (new_carry, new_carry) following Flax RNN convention
        h, _ = self.gru_cell(prev_h, inp)

        # 2. Prior
        prior_x = self.prior_norm(h)
        prior_logits = self.prior_head(prior_x).reshape(batch, self.stoch_dim, self.stoch_classes)
        prior_dist = OneHotCategorical(prior_logits, straight_through=use_straight_through)

        if embed is None:
            # Imagination mode: sample from prior
            if key is None:
                z = prior_dist.mode()
            else:
                z = prior_dist.sample(key)
            return {"h": h, "z": z, "prior_logits": prior_logits}

        # 3. Posterior (observe mode)
        post_x = self.post_norm(jnp.concatenate([h, embed], axis=-1))
        post_logits = self.post_head(post_x).reshape(batch, self.stoch_dim, self.stoch_classes)
        post_dist = OneHotCategorical(post_logits, straight_through=use_straight_through)

        if key is None:
            z = post_dist.mode()
        else:
            z = post_dist.sample(key)

        return {
            "h": h,
            "z": z,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
        }


# ---------------------------------------------------------------------------
# RSSM dynamics wrapper implementing BaseDynamics
# ---------------------------------------------------------------------------


class RSSM(BaseDynamics):
    """Recurrent State Space Model wrapping :class:`RSSMModule`.

    This class provides the :class:`~helios.dynamics.base.BaseDynamics`
    interface and manages Flax parameters externally (functional style).

    Args:
        config: DictConfig with RSSM hyperparameters.
        obs_embed_dim: Dimensionality of the observation embedding (encoder output).
        action_dim: Dimensionality of the action vector.
    """

    def __init__(self, config: Any, obs_embed_dim: int, action_dim: int) -> None:
        self.config = config
        self.obs_embed_dim = obs_embed_dim
        self.action_dim = action_dim
        self.module = RSSMModule(
            deter_dim=int(config.deter_dim),
            stoch_dim=int(config.stoch_dim),
            stoch_classes=int(config.stoch_classes),
            hidden_dim=int(config.hidden_dim),
            embed_dim=int(config.embed_dim),
        )

    # ------------------------------------------------------------------
    # BaseDynamics interface
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int) -> dict[str, jax.Array]:
        """Return zeroed (h, z) state."""
        h = jnp.zeros((batch_size, self.module.deter_dim))
        z = jnp.zeros((batch_size, self.module.stoch_dim, self.module.stoch_classes))
        return {"h": h, "z": z}

    def observe(
        self,
        obs: jax.Array,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Posterior update given an observation embedding.

        Note: ``obs`` here is expected to be the *encoded* observation
        (output of the CNN/MLP encoder), not raw pixels.

        Args:
            obs: Encoded observation, shape (batch, embed_dim).
            prev_state: Dict with ``h`` and ``z``.
            action: Action, shape (batch, action_dim).
            key: PRNG key.
            params: Flax parameter dict for ``self.module``.

        Returns:
            ``(new_state, extras)`` where extras contains KL-related tensors.
        """
        out = self.module.apply(
            params,
            prev_state["h"],
            prev_state["z"],
            action,
            embed=obs,
            key=key,
        )
        new_state = {"h": out["h"], "z": out["z"]}
        extras = {
            "prior_logits": out["prior_logits"],
            "post_logits": out["post_logits"],
        }
        return new_state, extras

    def imagine(
        self,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Prior transition without an observation.

        Args:
            prev_state: Dict with ``h`` and ``z``.
            action: Action, shape (batch, action_dim).
            key: PRNG key.
            params: Flax parameter dict for ``self.module``.

        Returns:
            ``(new_state, extras)`` where extras contains prior logits.
        """
        out = self.module.apply(
            params,
            prev_state["h"],
            prev_state["z"],
            action,
            embed=None,
            key=key,
        )
        new_state = {"h": out["h"], "z": out["z"]}
        extras = {"prior_logits": out["prior_logits"]}
        return new_state, extras

    def get_feat(self, state: dict[str, jax.Array]) -> jax.Array:
        """Concatenate h and flattened z as the feature vector."""
        z_flat = state["z"].reshape(state["z"].shape[0], -1)
        return jnp.concatenate([state["h"], z_flat], axis=-1)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the concatenated (h, z) feature."""
        return self.module.deter_dim + self.module.stoch_size


# ---------------------------------------------------------------------------
# KL balancing loss
# ---------------------------------------------------------------------------


def rssm_kl_loss(
    prior_logits: jax.Array,
    post_logits: jax.Array,
    free_nats: float = 1.0,
    alpha: float = 0.8,
) -> jax.Array:
    """KL-balancing loss from DreamerV3.

    Combines a scaled KL(post || sg(prior)) and KL(sg(post) || prior) to
    balance the learning signals between the prior and posterior networks.

    Args:
        prior_logits: Prior logits, shape (..., stoch_dim, stoch_classes).
        post_logits: Posterior logits, shape (..., stoch_dim, stoch_classes).
        free_nats: Minimum KL per latent dimension (free information budget).
        alpha: Weight of KL(post||sg(prior)).  (1-alpha) weights KL(sg(post)||prior).

    Returns:
        Scalar mean KL loss.
    """
    prior_dist = OneHotCategorical(prior_logits)
    post_dist = OneHotCategorical(post_logits)
    prior_sg_dist = OneHotCategorical(jax.lax.stop_gradient(prior_logits))
    post_sg_dist = OneHotCategorical(jax.lax.stop_gradient(post_logits))

    # KL per dimension, then clamp to free nats and sum over dims
    kl_post_prior = jnp.maximum(
        post_dist.kl_divergence(prior_sg_dist), free_nats
    )
    kl_prior_post = jnp.maximum(
        post_sg_dist.kl_divergence(prior_dist), free_nats
    )

    kl = alpha * kl_post_prior + (1.0 - alpha) * kl_prior_post
    return jnp.mean(kl)
