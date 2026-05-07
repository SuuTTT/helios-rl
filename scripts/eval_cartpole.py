import jax
import jax.numpy as jnp
import gymnax
import numpy as np
from run_cartpole import make_train
from helios.core.networks import ActorCritic

print("1. Training the agent to convergence...")
train_fn = make_train()
jitted_train = jax.jit(train_fn)

rng = jax.random.PRNGKey(42)
runner_state, metrics = jitted_train(rng)
train_state = runner_state[0]
print("   -> Training complete. Weights frozen.")

print("\n2. Setting up dedicated evaluation...")
env, env_params = gymnax.make("CartPole-v1")
env_params = env_params.replace(max_steps_in_episode=500)
network = ActorCritic(env.action_space(env_params).n)

def evaluate_episode(rng):
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    
    def step_fn(carry, _):
        obs, state, rng, is_active, cum_reward = carry
        
        pi, _ = network.apply(train_state.params, obs)
        rng, step_rng = jax.random.split(rng)
        
        # PPO policies in Gym environments train via stochastic distributions. 
        # For evaluation we turn entropy to 0 gracefully by taking the peak probability,
        # but pure argmax can sometimes fail out of training-distribution loops in highly chaotic envs if they are under-explored.
        # But generally argmax works. Let's strictly test stochastic as a comparison if argmax is failing.
        action = jnp.argmax(pi.probs, axis=-1)
        
        next_obs, next_state, reward, done, info = env.step(step_rng, state, action, env_params)
        cum_reward += (reward * is_active)
        is_active = is_active * (1.0 - done)
        return (next_obs, next_state, rng, is_active, cum_reward), None

    final_carry, _ = jax.lax.scan(step_fn, (obs, state, rng, jnp.float32(1.0), jnp.float32(0.0)), None, length=500)
    return final_carry[4]

print("3. Running 100 perfectly isolated evaluation episodes...")
eval_rngs = jax.random.split(jax.random.PRNGKey(999), 100)
returns = jax.jit(jax.vmap(evaluate_episode))(eval_rngs)
returns = np.array(returns)

print("\n================ EVALUATION RESULTS ================")
print(f"Total Episodes Run : {len(returns)}")
print(f"Mean Score         : {returns.mean():.2f}")
print(f"Perfect 500s Count : {(returns == 500).sum()}/{len(returns)}")
print("====================================================")
