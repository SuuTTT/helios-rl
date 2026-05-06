"""DreamerV3: Imagination-based Actor-Critic with an RSSM world model.

Architecture overview
---------------------
1. **World Model** – encodes observations (CNN), infers the RSSM latent state,
   and decodes reconstructions + reward + continue signal.
2. **Actor** – trained via imagined rollouts inside the world model.
3. **Critic** – trained on imagined trajectories using TD-λ targets.

Reference:
    Hafner et al. (2023) - "Mastering Diverse Domains through World Models"
    https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from helios.algorithms.base import BaseAgent
from helios.core.distributions import OneHotCategorical, TanhNormal
from helios.core.networks import MLP, CNNEncoder, CNNDecoder
from helios.dynamics.rssm import RSSM, RSSMModule, rssm_kl_loss
from helios.memory.trajectory import SequenceBatch


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------


class RewardHead(nn.Module):
    """MLP predicting scalar reward from RSSM feature."""

    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, feat: jax.Array) -> jax.Array:
        x = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(feat)
        return x.squeeze(-1)


class ContinueHead(nn.Module):
    """MLP predicting continue probability (1 - done) from RSSM feature."""

    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, feat: jax.Array) -> jax.Array:
        x = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(feat)
        return nn.sigmoid(x).squeeze(-1)


class Actor(nn.Module):
    """MLP actor that outputs a TanhNormal distribution over actions."""

    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, feat: jax.Array) -> dict[str, jax.Array]:
        x = MLP(hidden_dims=self.hidden_dims, activation=self.activation)(feat)
        mean = nn.Dense(self.action_dim)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.broadcast_to(log_std, mean.shape)
        return {"mean": mean, "log_std": log_std}


class Critic(nn.Module):
    """MLP critic estimating the expected discounted return."""

    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    activation: str = "silu"

    @nn.compact
    def __call__(self, feat: jax.Array) -> jax.Array:
        x = MLP(hidden_dims=self.hidden_dims, output_dim=1, activation=self.activation)(feat)
        return x.squeeze(-1)


# ---------------------------------------------------------------------------
# DreamerV3 Agent
# ---------------------------------------------------------------------------


class DreamerV3Agent(BaseAgent):
    """DreamerV3 imagination-based actor-critic agent.

    Args:
        config: Hydra DictConfig with dreamer_v3 hyperparameters.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.
    """

    def initial_state(self, key: jax.Array) -> dict[str, Any]:
        """Initialise all networks and optimizers.

        Returns a state dict with separate TrainStates for the world model,
        actor, and critic, plus the current RSSM hidden state.
        """
        import gymnasium as gym

        cfg = self.config
        obs_shape = self.observation_space.shape
        is_pixel = len(obs_shape) == 3

        if isinstance(self.action_space, gym.spaces.Box):
            action_dim = int(self.action_space.shape[0])
        else:
            action_dim = int(self.action_space.n)

        embed_dim = int(cfg.rssm.embed_dim)

        # -- Encoder --
        if is_pixel:
            encoder = CNNEncoder(depth=int(cfg.encoder_depth), embed_dim=embed_dim)
        else:
            obs_dim = int(jnp.prod(jnp.array(obs_shape)))
            encoder = MLP(
                hidden_dims=(embed_dim,),
                output_dim=embed_dim,
                activation="silu",
            )

        # -- RSSM module --
        rssm_mod = RSSMModule(
            deter_dim=int(cfg.rssm.deter_dim),
            stoch_dim=int(cfg.rssm.stoch_dim),
            stoch_classes=int(cfg.rssm.stoch_classes),
            hidden_dim=int(cfg.rssm.hidden_dim),
            embed_dim=embed_dim,
        )

        feature_dim = int(cfg.rssm.deter_dim) + int(cfg.rssm.stoch_dim) * int(cfg.rssm.stoch_classes)

        # -- Decoder --
        if is_pixel:
            decoder = CNNDecoder(depth=int(cfg.decoder_depth), output_channels=obs_shape[-1])
        else:
            decoder = MLP(
                hidden_dims=(embed_dim,),
                output_dim=int(obs_shape[0]),
                activation="silu",
            )

        reward_head = RewardHead(hidden_dims=tuple(cfg.actor_hidden_dims))
        continue_head = ContinueHead(hidden_dims=tuple(cfg.critic_hidden_dims))
        actor = Actor(
            action_dim=action_dim,
            hidden_dims=tuple(cfg.actor_hidden_dims),
        )
        critic = Critic(hidden_dims=tuple(cfg.critic_hidden_dims))
        slow_critic = Critic(hidden_dims=tuple(cfg.critic_hidden_dims))

        key, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 8)

        # Dummy inputs for init
        dummy_obs = jnp.zeros((1,) + obs_shape)
        dummy_embed = jnp.zeros((1, embed_dim))
        dummy_h = jnp.zeros((1, int(cfg.rssm.deter_dim)))
        dummy_z = jnp.zeros((1, int(cfg.rssm.stoch_dim), int(cfg.rssm.stoch_classes)))
        dummy_action = jnp.zeros((1, action_dim))
        dummy_feat = jnp.zeros((1, feature_dim))

        enc_params = encoder.init(k1, dummy_obs)
        rssm_params = rssm_mod.init(k2, dummy_h, dummy_z, dummy_action, embed=dummy_embed)
        dec_params = decoder.init(k3, dummy_feat)
        rew_params = reward_head.init(k4, dummy_feat)
        cont_params = continue_head.init(k5, dummy_feat)
        actor_params = actor.init(k6, dummy_feat)
        critic_params = critic.init(k7, dummy_feat)
        slow_critic_params = critic_params  # initialise equal to critic

        # World model combines encoder + rssm + decoder + reward + continue
        wm_params = {
            "encoder": enc_params,
            "rssm": rssm_params,
            "decoder": dec_params,
            "reward": rew_params,
            "continue": cont_params,
        }

        model_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.model_lr), eps=float(cfg.model_eps)),
        )
        actor_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.actor_lr), eps=float(cfg.actor_eps)),
        )
        critic_tx = optax.chain(
            optax.clip_by_global_norm(float(cfg.max_grad_norm)),
            optax.adam(float(cfg.critic_lr), eps=float(cfg.critic_eps)),
        )

        wm_opt_state = model_tx.init(wm_params)
        actor_opt_state = actor_tx.init(actor_params)
        critic_opt_state = critic_tx.init(critic_params)

        # Initial RSSM hidden state (batch_size=1 for acting)
        rssm_h = jnp.zeros((1, int(cfg.rssm.deter_dim)))
        rssm_z = jnp.zeros((1, int(cfg.rssm.stoch_dim), int(cfg.rssm.stoch_classes)))

        return {
            # Network modules (kept for apply_fn reference)
            "encoder": encoder,
            "rssm_mod": rssm_mod,
            "decoder": decoder,
            "reward_head": reward_head,
            "continue_head": continue_head,
            "actor": actor,
            "critic": critic,
            "slow_critic": slow_critic,
            # Parameters
            "wm_params": wm_params,
            "actor_params": actor_params,
            "critic_params": critic_params,
            "slow_critic_params": slow_critic_params,
            # Optimizer states
            "wm_opt_state": wm_opt_state,
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
            # Optimizers
            "model_tx": model_tx,
            "actor_tx": actor_tx,
            "critic_tx": critic_tx,
            # Runtime state
            "rssm_h": rssm_h,
            "rssm_z": rssm_z,
            "step": 0,
        }

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        obs: jax.Array,
        state: dict[str, Any],
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Encode observation, update RSSM state, and sample from actor.

        Args:
            obs: Shape (num_envs, *obs_shape).
            state: Agent state.
            key: PRNG key.
            deterministic: If True, return mode action.

        Returns:
            ``(action, new_rssm_state)``.
        """
        enc = state["encoder"]
        rssm_mod = state["rssm_mod"]
        actor = state["actor"]

        embed = enc.apply(state["wm_params"]["encoder"], obs)

        key, rssm_key, act_key = jax.random.split(key, 3)
        # Dummy action for the first step (we don't have a previous action stored)
        dummy_action = jnp.zeros((obs.shape[0], state["actor"].action_dim))

        rssm_out = rssm_mod.apply(
            state["wm_params"]["rssm"],
            state["rssm_h"],
            state["rssm_z"],
            dummy_action,
            embed=embed,
            key=rssm_key,
        )
        new_h, new_z = rssm_out["h"], rssm_out["z"]
        feat = jnp.concatenate([new_h, new_z.reshape(new_z.shape[0], -1)], axis=-1)

        actor_out = actor.apply(state["actor_params"], feat)
        dist = TanhNormal(actor_out["mean"], actor_out["log_std"])

        if deterministic:
            action = dist.mode()
        else:
            action, _ = dist.sample(act_key)

        new_state = {**state, "rssm_h": new_h, "rssm_z": new_z}
        return action, new_state

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        batch: SequenceBatch,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Train world model, actor, and critic on a sequence batch.

        Args:
            batch: :class:`~helios.memory.trajectory.SequenceBatch`.
            state: Agent state.

        Returns:
            ``(new_state, metrics)``.
        """
        cfg = self.config

        # ---- 1. World model update ----
        wm_params, wm_opt_state, wm_metrics, rssm_states = _wm_update(
            wm_params=state["wm_params"],
            opt_state=state["wm_opt_state"],
            tx=state["model_tx"],
            encoder=state["encoder"],
            rssm_mod=state["rssm_mod"],
            decoder=state["decoder"],
            reward_head=state["reward_head"],
            continue_head=state["continue_head"],
            batch=batch,
            kl_free_nats=float(cfg.kl_free_nats),
            kl_alpha=float(cfg.kl_alpha),
            recon_weight=float(cfg.reconstruction_loss_weight),
            reward_weight=float(cfg.reward_loss_weight),
            continue_weight=float(cfg.continue_loss_weight),
        )

        # ---- 2. Imagination rollout for Actor-Critic ----
        feat_dim = int(cfg.rssm.deter_dim) + int(cfg.rssm.stoch_dim) * int(cfg.rssm.stoch_classes)
        imag_feats, imag_rewards, imag_continues = _imagine_rollout(
            wm_params=wm_params,
            rssm_mod=state["rssm_mod"],
            reward_head=state["reward_head"],
            continue_head=state["continue_head"],
            actor=state["actor"],
            actor_params=state["actor_params"],
            start_states=rssm_states,
            horizon=int(cfg.imagination_horizon),
        )

        # ---- 3. Critic update ----
        critic_params, critic_opt_state, critic_metrics = _critic_update(
            critic_params=state["critic_params"],
            slow_critic_params=state["slow_critic_params"],
            opt_state=state["critic_opt_state"],
            tx=state["critic_tx"],
            critic=state["critic"],
            slow_critic=state["slow_critic"],
            imag_feats=imag_feats,
            imag_rewards=imag_rewards,
            imag_continues=imag_continues,
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
        )

        # ---- 4. Actor update ----
        actor_params, actor_opt_state, actor_metrics = _actor_update(
            actor_params=state["actor_params"],
            opt_state=state["actor_opt_state"],
            tx=state["actor_tx"],
            actor=state["actor"],
            critic=state["critic"],
            critic_params=critic_params,
            imag_feats=imag_feats,
            imag_continues=imag_continues,
            entropy_scale=float(cfg.actor_entropy_scale),
        )

        # ---- 5. Slow critic EMA ----
        slow_critic_params = jax.tree_util.tree_map(
            lambda s, f: (1.0 - float(cfg.slow_target_fraction)) * s
            + float(cfg.slow_target_fraction) * f,
            state["slow_critic_params"],
            critic_params,
        )

        new_state = {
            **state,
            "wm_params": wm_params,
            "actor_params": actor_params,
            "critic_params": critic_params,
            "slow_critic_params": slow_critic_params,
            "wm_opt_state": wm_opt_state,
            "actor_opt_state": actor_opt_state,
            "critic_opt_state": critic_opt_state,
            "step": state["step"] + 1,
        }
        metrics = {**wm_metrics, **critic_metrics, **actor_metrics}
        return new_state, metrics


# ---------------------------------------------------------------------------
# Functional update helpers
# ---------------------------------------------------------------------------


def _wm_update(
    wm_params,
    opt_state,
    tx,
    encoder,
    rssm_mod,
    decoder,
    reward_head,
    continue_head,
    batch: SequenceBatch,
    kl_free_nats: float,
    kl_alpha: float,
    recon_weight: float,
    reward_weight: float,
    continue_weight: float,
):
    """Compute world model gradients and update parameters."""

    def loss_fn(params):
        B, T = batch.obs.shape[:2]
        obs_flat = batch.obs.reshape(B * T, *batch.obs.shape[2:])

        embed = encoder.apply(params["encoder"], obs_flat)
        embed = embed.reshape(B, T, -1)

        # Scan over time to get RSSM states
        init_h = jnp.zeros((B, rssm_mod.deter_dim))
        init_z = jnp.zeros((B, rssm_mod.stoch_dim, rssm_mod.stoch_classes))

        def _step(carry, x):
            h, z = carry
            emb, act = x
            key = jax.random.PRNGKey(0)  # deterministic for loss computation
            out = rssm_mod.apply(params["rssm"], h, z, act, embed=emb, key=key)
            return (out["h"], out["z"]), out

        _, rssm_outs = jax.lax.scan(
            _step,
            (init_h, init_z),
            (jnp.transpose(embed, (1, 0, 2)), jnp.transpose(batch.actions, (1, 0, 2))),
        )
        # rssm_outs: each value has shape (T, B, ...)
        h = jnp.transpose(rssm_outs["h"], (1, 0, 2))       # (B, T, deter_dim)
        z = jnp.transpose(rssm_outs["z"], (1, 0, 2, 3))    # (B, T, stoch_dim, stoch_classes)
        feat = jnp.concatenate([h, z.reshape(B, T, -1)], axis=-1)  # (B, T, feat_dim)

        # KL loss
        kl = rssm_kl_loss(
            rssm_outs["prior_logits"],
            rssm_outs["post_logits"],
            free_nats=kl_free_nats,
            alpha=kl_alpha,
        )

        feat_flat = feat.reshape(B * T, -1)

        # Reconstruction
        recon = decoder.apply(params["decoder"], feat_flat)
        recon_loss = jnp.mean((recon - obs_flat) ** 2)

        # Reward
        pred_rew = reward_head.apply(params["reward"], feat_flat)
        rew_loss = jnp.mean((pred_rew - batch.rewards.reshape(B * T)) ** 2)

        # Continue
        pred_cont = continue_head.apply(params["continue"], feat_flat)
        cont_target = 1.0 - batch.dones.reshape(B * T)
        cont_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(pred_cont, cont_target)
        )

        total_loss = (
            kl
            + recon_weight * recon_loss
            + reward_weight * rew_loss
            + continue_weight * cont_loss
        )
        metrics = {
            "wm/kl": kl,
            "wm/recon": recon_loss,
            "wm/reward": rew_loss,
            "wm/continue": cont_loss,
            "wm/total": total_loss,
        }
        return total_loss, (metrics, (h, z))

    (_, (metrics, (h, z))), grads = jax.value_and_grad(loss_fn, has_aux=True)(wm_params)
    updates, new_opt_state = tx.update(grads, opt_state, wm_params)
    new_params = optax.apply_updates(wm_params, updates)
    return new_params, new_opt_state, metrics, (h, z)


def _imagine_rollout(
    wm_params,
    rssm_mod,
    reward_head,
    continue_head,
    actor,
    actor_params,
    start_states,
    horizon: int,
):
    """Unroll the world model from start states using the actor policy."""
    h0, z0 = start_states
    B, T = h0.shape[:2]
    feat_dim = h0.shape[-1] + z0.shape[-2] * z0.shape[-1]

    # Flatten batch and time dims as independent starting points
    h_flat = h0.reshape(B * T, -1)
    z_flat = z0.reshape(B * T, z0.shape[-2], z0.shape[-1])
    N = B * T

    imag_feats = []
    imag_rewards = []
    imag_continues = []

    h, z = h_flat, z_flat
    for t in range(horizon):
        feat = jnp.concatenate([h, z.reshape(N, -1)], axis=-1)
        actor_out = actor.apply(actor_params, feat)
        dist = TanhNormal(actor_out["mean"], actor_out["log_std"])
        action, _ = dist.sample(jax.random.PRNGKey(t))

        out = rssm_mod.apply(wm_params["rssm"], h, z, action, embed=None, key=jax.random.PRNGKey(t))
        h, z = out["h"], out["z"]
        new_feat = jnp.concatenate([h, z.reshape(N, -1)], axis=-1)

        rew = reward_head.apply(wm_params["reward"], new_feat)
        cont = continue_head.apply(wm_params["continue"], new_feat)

        imag_feats.append(new_feat)
        imag_rewards.append(rew)
        imag_continues.append(cont)

    imag_feats = jnp.stack(imag_feats, axis=0)       # (H, N, feat_dim)
    imag_rewards = jnp.stack(imag_rewards, axis=0)   # (H, N)
    imag_continues = jnp.stack(imag_continues, axis=0)  # (H, N)
    return imag_feats, imag_rewards, imag_continues


def _critic_update(
    critic_params,
    slow_critic_params,
    opt_state,
    tx,
    critic,
    slow_critic,
    imag_feats,
    imag_rewards,
    imag_continues,
    gamma: float,
    gae_lambda: float,
):
    """TD-λ critic update using imagined rollouts."""

    def loss_fn(params):
        values = jax.vmap(lambda f: critic.apply(params, f))(imag_feats)  # (H, N)
        slow_values = jax.vmap(lambda f: slow_critic.apply(slow_critic_params, f))(imag_feats)

        # TD-λ targets
        H = imag_rewards.shape[0]
        targets = jnp.zeros_like(values)
        last_target = slow_values[-1]

        for t in reversed(range(H)):
            last_target = imag_rewards[t] + gamma * imag_continues[t] * (
                (1 - gae_lambda) * slow_values[t] + gae_lambda * last_target
            )
            targets = targets.at[t].set(last_target)

        loss = jnp.mean((values - jax.lax.stop_gradient(targets)) ** 2)
        return loss, {"critic/loss": loss}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic_params)
    updates, new_opt_state = tx.update(grads, opt_state, critic_params)
    new_params = optax.apply_updates(critic_params, updates)
    return new_params, new_opt_state, metrics


def _actor_update(
    actor_params,
    opt_state,
    tx,
    actor,
    critic,
    critic_params,
    imag_feats,
    imag_continues,
    entropy_scale: float,
):
    """Actor update: maximise predicted value + entropy."""

    def loss_fn(params):
        H, N, feat_dim = imag_feats.shape
        # Compute per-step action distributions for entropy
        actor_outs = jax.vmap(lambda f: actor.apply(params, f))(imag_feats)
        dists = TanhNormal(actor_outs["mean"], actor_outs["log_std"])
        entropy = dists.entropy()  # (H, N)

        # Value of imagined states (stop gradient through critic)
        values = jax.vmap(lambda f: critic.apply(critic_params, f))(imag_feats)

        # Discount by continue probabilities
        discounts = jnp.cumprod(
            jnp.concatenate([jnp.ones((1, N)), imag_continues[:-1]], axis=0),
            axis=0,
        )

        actor_loss = -jnp.mean(
            discounts * (jax.lax.stop_gradient(values) + entropy_scale * entropy)
        )
        return actor_loss, {"actor/loss": actor_loss, "actor/entropy": jnp.mean(entropy)}

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_params)
    updates, new_opt_state = tx.update(grads, opt_state, actor_params)
    new_params = optax.apply_updates(actor_params, updates)
    return new_params, new_opt_state, metrics


# Need optax for sigmoid BCE inside the WM loss
import optax  # noqa: E402 – imported at bottom to avoid circular at module top
