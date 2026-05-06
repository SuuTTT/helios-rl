"""Joint-Embedding Predictive Architecture (JEPA) dynamics model.

JEPA learns a latent space by predicting future latent representations from
past ones, without reconstructing observations in pixel space.  This avoids
the cost of training a pixel decoder while retaining useful predictive structure.

Reference: Assran et al. (2023) - "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (I-JEPA)
https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from helios.core.networks import MLP, NormedLinear
from helios.dynamics.base import BaseDynamics


# ---------------------------------------------------------------------------
# Flax modules
# ---------------------------------------------------------------------------


class ContextEncoder(nn.Module):
    """Encodes a sequence of observations into a context embedding.

    Args:
        hidden_dims: MLP widths.
        embed_dim: Output embedding dimensionality.
        activation: Activation name.
    """

    hidden_dims: tuple[int, ...] = (512, 512)
    embed_dim: int = 512
    activation: str = "silu"

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """
        Args:
            obs: Shape (..., obs_dim).
        Returns:
            Embedding of shape (..., embed_dim).
        """
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.embed_dim,
            activation=self.activation,
        )(obs)


class TargetEncoder(nn.Module):
    """Target (EMA) encoder – same architecture as ContextEncoder.

    Parameters are updated via exponential moving average of the context
    encoder; gradients are **not** propagated through the target.
    """

    hidden_dims: tuple[int, ...] = (512, 512)
    embed_dim: int = 512
    activation: str = "silu"

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.embed_dim,
            activation=self.activation,
        )(obs)


class Predictor(nn.Module):
    """Narrow MLP that predicts target embeddings from context + action.

    A deliberately *smaller* network than the encoders so that the predictor
    cannot trivially collapse to the identity.

    Args:
        hidden_dims: Predictor MLP widths.
        embed_dim: Output (= target) embedding dimension.
    """

    hidden_dims: tuple[int, ...] = (256,)
    embed_dim: int = 512
    activation: str = "silu"

    @nn.compact
    def __call__(self, context: jax.Array, action: jax.Array) -> jax.Array:
        """
        Args:
            context: Context embedding, shape (..., embed_dim).
            action: Action, shape (..., action_dim).
        Returns:
            Predicted target embedding, shape (..., embed_dim).
        """
        x = jnp.concatenate([context, action], axis=-1)
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.embed_dim,
            activation=self.activation,
        )(x)


# ---------------------------------------------------------------------------
# JEPA dynamics
# ---------------------------------------------------------------------------


class JEPA(BaseDynamics):
    """Joint-Embedding Predictive Architecture dynamics model.

    Maintains a *context embedding* as the latent state and provides:
    - **observe**: encode the current observation with the context encoder.
    - **imagine**: use the predictor to forecast the next embedding.

    The target encoder parameters are kept as a separate EMA copy updated
    outside this class (see :func:`update_target_params`).

    Args:
        config: DictConfig with JEPA hyperparameters.
        obs_dim: Dimensionality of the (flat) observation.
        action_dim: Dimensionality of the action vector.
    """

    def __init__(self, config: Any, obs_dim: int, action_dim: int) -> None:
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        embed_dim = int(getattr(config, "embed_dim", 512))
        hidden_dims = tuple(getattr(config, "hidden_dims", (512, 512)))

        self.context_encoder = ContextEncoder(
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
        )
        self.target_encoder = TargetEncoder(
            hidden_dims=hidden_dims,
            embed_dim=embed_dim,
        )
        self.predictor = Predictor(
            hidden_dims=tuple(getattr(config, "predictor_hidden_dims", (256,))),
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim

    # ------------------------------------------------------------------
    # BaseDynamics interface
    # ------------------------------------------------------------------

    def initial_state(self, batch_size: int) -> dict[str, jax.Array]:
        """Return zeroed context embedding."""
        return {"embedding": jnp.zeros((batch_size, self.embed_dim))}

    def observe(
        self,
        obs: jax.Array,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Encode the current observation to update the context embedding.

        Args:
            obs: Raw (or pre-processed) observation, shape (batch, obs_dim).
            prev_state: Dict with ``embedding``.
            action: Unused in JEPA observe step; included for interface compat.
            key: PRNG key (unused; JEPA observe is deterministic).
            params: Flax params dict with keys ``context_encoder`` and
                    ``target_encoder``.

        Returns:
            ``(new_state, extras)`` where extras contains ``target_embedding``.
        """
        context_embed = self.context_encoder.apply(
            params["context_encoder"], obs
        )
        target_embed = self.target_encoder.apply(
            params["target_encoder"], obs
        )
        new_state = {"embedding": context_embed}
        extras = {"target_embedding": target_embed}
        return new_state, extras

    def imagine(
        self,
        prev_state: dict[str, jax.Array],
        action: jax.Array,
        key: jax.Array,
        params: dict,
    ) -> tuple[dict[str, jax.Array], dict[str, Any]]:
        """Predict the next embedding without observing.

        Args:
            prev_state: Dict with ``embedding``.
            action: Action to condition the prediction on.
            key: PRNG key (unused; JEPA imagine is deterministic).
            params: Flax params dict with key ``predictor``.

        Returns:
            ``(new_state, extras)``.
        """
        predicted_embed = self.predictor.apply(
            params["predictor"],
            prev_state["embedding"],
            action,
        )
        new_state = {"embedding": predicted_embed}
        return new_state, {}

    def get_feat(self, state: dict[str, jax.Array]) -> jax.Array:
        return state["embedding"]

    @property
    def feature_dim(self) -> int:
        return self.embed_dim


# ---------------------------------------------------------------------------
# JEPA loss
# ---------------------------------------------------------------------------


def jepa_prediction_loss(
    predicted: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """L2 prediction loss in the normalised embedding space.

    Args:
        predicted: Predictor output, shape (batch, embed_dim).
        target: Target encoder output (stop_gradient applied here),
                shape (batch, embed_dim).

    Returns:
        Scalar mean loss.
    """
    target = jax.lax.stop_gradient(target)
    # Normalise along the embedding dimension to prevent collapse
    pred_norm = predicted / (jnp.linalg.norm(predicted, axis=-1, keepdims=True) + 1e-8)
    tgt_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    return jnp.mean(jnp.sum((pred_norm - tgt_norm) ** 2, axis=-1))


def update_target_params(
    online_params: dict,
    target_params: dict,
    tau: float = 0.99,
) -> dict:
    """EMA update of the target encoder parameters.

    Args:
        online_params: Context (online) encoder parameter tree.
        target_params: Current target encoder parameter tree.
        tau: EMA decay (0.99 = slow, 0.5 = fast).

    Returns:
        Updated target encoder parameters.
    """
    return jax.tree_util.tree_map(
        lambda t, o: tau * t + (1.0 - tau) * o,
        target_params,
        online_params,
    )
