import jax
import jax.numpy as jnp
import gymnax

env, env_params = gymnax.make("CartPole-v1")
rng = jax.random.PRNGKey(42)
obs, env_state = env.reset(rng, env_params)
obs, env_state, reward, done, info = env.step(rng, env_state, 0, env_params)
print(info.keys())
