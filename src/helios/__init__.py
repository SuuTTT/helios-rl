"""
helios-rl: High-performance Extensible Latent Inference & Optimization System.

A modular JAX-first reinforcement learning library with:
- Model-free algorithms (PPO)
- World-model-based algorithms (DreamerV3, TD-MPC2)
- Differentiable planners (CEM, MPPI)
- Flexible memory buffers
"""

__version__ = "0.1.0"

from helios.algorithms.base import BaseAgent
from helios.dynamics.base import BaseDynamics

__all__ = ["BaseAgent", "BaseDynamics", "__version__"]
