import os
import subprocess
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def run_cleanrl():
    print("Running CleanRL PPO...")
    seeds = [1, 2, 3, 4, 5]
    cleanrl_dir = "/workspace/cleanrl"
    base_cmd = [
        "python3", "cleanrl/ppo.py",
        "--env-id", "CartPole-v1",
        "--total-timesteps", "1500000",
        "--num-envs", "128",
        "--num-steps", "128",
        "--update-epochs", "4",
        "--num-minibatches", "4",
        "--learning-rate", "2.5e-4",
        "--gamma", "0.99",
        "--gae-lambda", "0.95",
        "--clip-coef", "0.2",
        "--ent-coef", "0.01",
        "--vf-coef", "0.5",
        "--max-grad-norm", "0.5",
    ]
    
    all_vals = []
    run_times = []
    
    for seed in seeds:
        print(f"  CleanRL Seed {seed}...")
        cmd = base_cmd + ["--seed", str(seed)]
        start_time = time.time()
        result = subprocess.run(cmd, cwd=cleanrl_dir, capture_output=True, text=True)
        run_times.append(time.time() - start_time)
        
        steps = []
        returns = []
        for line in result.stdout.split('\n'):
            match = re.search(r'global_step=(\d+), episodic_return=([\d\.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                returns.append(float(match.group(2)))
        
        all_vals.append((steps, returns))
        
    return all_vals, np.mean(run_times)

def generate_comparison():
    c_vals, c_time = run_cleanrl()
    print(f"CleanRL Avg Time: {c_time:.2f}s per seed")
    
    max_steps = 1500000
    common_steps = np.linspace(0, max_steps, num=200)
    
    c_interp_vals = []
    for steps, returns in c_vals:
        if len(steps) > 0:
            d = list(zip(steps, returns))
            d.sort(key=lambda x: x[0])
            steps, returns = zip(*d)
            
            window = max(1, len(returns)//50)
            smoothed_returns = np.convolve(returns, np.ones(window)/window, mode='valid')
            smoothed_steps = steps[:len(smoothed_returns)]
            
            interp_r = np.interp(common_steps, smoothed_steps, smoothed_returns)
            c_interp_vals.append(interp_r)
    
    c_interp_vals = np.array(c_interp_vals)
    c_mean = np.mean(c_interp_vals, axis=0)
    c_std = np.std(c_interp_vals, axis=0)
    c_ci = 1.96 * (c_std / np.sqrt(len(c_interp_vals)))
    
    import sys
    sys.path.append("/workspace/helios-rl/scripts")
    import jax
    import jax.numpy as jnp
    from run_cartpole import make_train
    
    print("Running Helios PPO Benchmark...")
    train_loop = make_train()
    vmap_train = jax.jit(jax.vmap(train_loop))
    keys = jax.random.split(jax.random.PRNGKey(42), 5)
    
    start = time.time()
    runner_state, metrics = vmap_train(keys)
    h_time = time.time() - start
    print(f"Helios 5-seed Time: {h_time:.2f}s TOTAL")
    
    returns = metrics["returned_episode_returns"]
    dones = metrics["returned_episode"]
    
    h_interp_vals = []
    # Shape of dones is (5, NUM_UPDATES, NUM_STEPS, NUM_ENVS)
    num_updates = dones.shape[1]
    num_steps = dones.shape[2]
    num_envs = dones.shape[3]
    
    for seed_idx in range(5):
        seed_steps = []
        seed_returns = []
        for u in range(num_updates):
            global_step = (u + 1) * num_envs * num_steps
            # Get valid returns directly flattened from this block
            step_dones = dones[seed_idx, u]
            step_returns = returns[seed_idx, u]
            v_rets = step_returns[step_dones]
            if len(v_rets) > 0:
                for r in v_rets:
                    seed_steps.append(global_step)
                    seed_returns.append(r)
                    
        if len(seed_steps) > 0:
            window = max(1, len(seed_returns)//50)
            smoothed_returns = np.convolve(seed_returns, np.ones(window)/window, mode='valid')
            smoothed_steps = seed_steps[:len(smoothed_returns)]
            
            interp_r = np.interp(common_steps, smoothed_steps, smoothed_returns)
            h_interp_vals.append(interp_r)
            
    h_interp_vals = np.array(h_interp_vals)
    h_mean = h_interp_vals.mean(axis=0)
    h_std = h_interp_vals.std(axis=0)
    h_ci = 1.96 * (h_std / np.sqrt(5))
    
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(common_steps, c_mean, label=f"CleanRL (PyTorch)", color="orange", linewidth=2)
    plt.fill_between(common_steps, c_mean - c_ci, c_mean + c_ci, color="orange", alpha=0.3)
    
    plt.plot(common_steps, h_mean, label=f"Helios (JAX)", color="royalblue", linewidth=2)
    plt.fill_between(common_steps, h_mean - h_ci, h_mean + h_ci, color="royalblue", alpha=0.3)
    
    plt.title("PPO CartPole-v1: Helios vs CleanRL - True Episodic Rolling Average", fontsize=16)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episodic Return", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    output_path = "/workspace/helios-rl/cartpole_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved benchmark plot to: {output_path}")

if __name__ == "__main__":
    generate_comparison()
