import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from run_cartpole import make_train

def run_benchmark():
    train_loop = make_train()
    
    # We vmap the entire training loop over a batch of PRNG keys
    vmap_train = jax.jit(jax.vmap(train_loop))
    
    num_seeds = 5
    # Split the main RNG key into 5 unique seeds
    keys = jax.random.split(jax.random.PRNGKey(42), num_seeds)
    
    print(f"Starting {num_seeds}-seed benchmark. This compiles the graph once, then perfectly parallelizes over seeds!")
    start = time.time()
    runner_state, metrics = vmap_train(keys)
    print(f"Benchmark entirely finished in {time.time() - start:.2f} seconds.")
    
    # Extract episodic returns track
    # Shape of returns: [num_seeds, num_updates, num_steps, num_envs]
    returns = metrics["returned_episode_returns"] 
    dones = metrics["returned_episode"]
    
    # To get the average episodic return per update step (across all envs and steps)
    # We only count steps where an episode actually completed
    valid_returns = jnp.where(dones, returns, 0.0)
    # Safely compute the mean, avoiding divide-by-zero if no episodes ended in that update window
    update_means = valid_returns.sum(axis=(2, 3)) / jnp.maximum(dones.sum(axis=(2, 3)), 1)
    
    # Now compute 95% Confidence Interval across the 5 seeds dimension
    mean_update_returns = update_means.mean(axis=0)
    std_update_returns = update_means.std(axis=0)
    ci_95 = 1.96 * (std_update_returns / jnp.sqrt(num_seeds))
    
    # Generate x-axis points based on the number of generated updates
    x = jnp.arange(mean_update_returns.shape[0])
    
    # Plotting using style conventions
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, mean_update_returns, label=f"PPO (Gymnax CartPole-v1)", color="royalblue", linewidth=2)
    plt.fill_between(x, mean_update_returns - ci_95, mean_update_returns + ci_95, color="royalblue", alpha=0.3, label="95% CI")
    
    plt.title(f"PPO Training Performance ({num_seeds} Seeds)", fontsize=16)
    plt.xlabel("Updates (Each update = 128 envs * 128 steps)", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    output_path = "/workspace/helios-rl/cartpole_5seeds_benchmark.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved benchmark plot to: {output_path}")

    print("\nFinal Update Statistics:")
    print(f"Mean episodic return: {mean_update_returns[-1]:.2f}")
    print(f"+/- 95% CI:           {ci_95[-1]:.2f}")

if __name__ == "__main__":
    run_benchmark()
