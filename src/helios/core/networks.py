import flax.linen as nn
import jax.numpy as jnp
import distrax

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # CleanRL PPO MLP style: unshared Actor and Critic with Orthogonal Initialization
        
        # Critic
        critic = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        
        # Actor
        actor = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor = nn.tanh(actor)
        actor = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor)
        actor = nn.tanh(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)
        pi = distrax.Categorical(logits=logits)
        
        return pi, jnp.squeeze(value, axis=-1)

class ContinuousActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Standard CleanRL Continuous PPO
        critic = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)

        actor_mean = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        action_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor_mean)

        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.Normal(loc=action_mean, scale=jnp.exp(actor_logtstd))
        
        return pi, jnp.squeeze(value, axis=-1)
