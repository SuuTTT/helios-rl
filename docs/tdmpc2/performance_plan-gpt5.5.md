I’ll compare our current script against the official TD-MPC2 code and the latest run logs, then turn that into a phased plan focused on closing the algorithmic gap rather than just pushing speed.

Read [](file:///workspace/tdmpc2)

Searched for regex `class TDMPC2|def update|def plan|MPPI|consistency|td_target|rho` (`**/tdmpc2/**`), 80 results

Searched for regex `Hopper|hopper|train_steps|seed_steps|horizon|num_samples|num_pi_trajs|rho|consistency_coef|reward_coef|value_coef|pi_coef|tau` (`**/tdmpc2/**`), 120 results

Read memory [](file:///root/.vscode-server/data/User/workspaceStorage/052d70093d9f9107655bb9d1ce3e1cc8/GitHub.copilot-chat/memory-tool/memories/repo/ppo_jax_findings.md)

I found two useful threads already: the older implementation-gap docs are pointing at MPPI planner differences, and the live v12 run shows speed is solved but MPPI is still lagging policy. I’m going into the official code now to identify the highest-impact mismatches.

Searched for files matching `**/tdmpc2/tdmpc2/**/*.py`, 36 matches

Searched for files matching `**/tdmpc2/**/*.yaml`, 2 matches

Searched for files matching `**/tdmpc2/results/**/*.csv`, 40 matches

Read [](file:///workspace/helios-rl/docs/tdmpc2/IMPL_GAP.md#1-1), lines 1 to 340

Read [](file:///workspace/helios-rl/docs/tdmpc2/ITERATION_LOG.md#442-442), lines 442 to 560

Ran terminal command:  tail -n 80 /tmp/hopper_v12_fused.log | grep -E "step=|MPPI=|es=" | tail -30

Read [](file:///workspace/tdmpc2/tdmpc2/tdmpc2.py)

Read [](file:///workspace/tdmpc2/tdmpc2/config.yaml)

Read [](file:///workspace/tdmpc2/tdmpc2/common/world_model.py)

Read [](file:///workspace/tdmpc2/tdmpc2/common/math.py)

The big official-vs-ours deltas are now pretty clear: our v12 is an efficient trainer for a simplified TD-MPC-like variant, but official TD-MPC2’s planning quality relies on a stronger planner, bigger model/ensemble, separate actor update, and target-Q-only EMA. I’m checking the reference curve and environment setup so the plan has concrete success gates.

Searched for regex `model_size|mt80|mt30|model_size|latent_dim|mlp_dim|hopper-hop|discount_denom|seed_steps` (`**/tdmpc2/**`), 120 results

Searched for regex `train_steps|seed_steps|buffer.add|for .*steps|agent.update|episode_length|eval_freq` (`**/tdmpc2/tdmpc2/trainer/**`), 18 results

Read [](file:///workspace/tdmpc2/results/tdmpc2/hopper-hop.csv#1-1), lines 1 to 80

Read [](file:///workspace/tdmpc2/tdmpc2/common/buffer.py)

Read [](file:///workspace/tdmpc2/tdmpc2/common/layers.py)

Read [](file:///workspace/tdmpc2/tdmpc2/trainer/online_trainer.py)

Read [](file:///workspace/tdmpc2/tdmpc2/common/parser.py)

Read [](file:///workspace/tdmpc2/tdmpc2/common/__init__.py)

Searched for regex `K_UPDATE|SEQ|LATENT|HIDDEN|GAMMA|TAU|H\s+=|NS\s+=|NI\s+=|multi_upd|sample_k|q_net.apply\(tp\["q"\]|new_tp =|QEnsemble` (`**/helios-rl/scripts/train_tdmpc_hopper_v12_fused.py`), 80 results

Searched for regex `reward_coef|value_coef|consistency_coef|num_samples|num_elites|num_pi_trajs|horizon|num_bins|vmin|vmax|num_q|latent_dim|mlp_dim|model_size` (`**/tdmpc2/tdmpc2/config.yaml`), 14 results

Searched for regex `def _plan|num_pi_trajs|num_elites|std =|topk|gumbel|def update_pi|self.scale|soft_update_target_Q|def _td_target|def _update|total_loss` (`**/tdmpc2/tdmpc2/tdmpc2.py`), 28 results

Searched for regex `_target_Qs|_detach_Qs|def soft_update_target_Q|def pi|def Q|randperm|two_hot_inv|self._Qs|init.zero_` (`**/tdmpc2/tdmpc2/common/world_model.py`), 31 results

Searched for regex `MODEL_SIZE|1:|5:|mlp_dim|latent_dim|num_q` (`**/tdmpc2/tdmpc2/common/__init__.py`), 15 results

Searched for regex `model_size|hopper-hop|single-task|DMControl|results/tdmpc2` (`**/tdmpc2/README.md`), 23 results

The next move should be **algorithm parity**, not more speed work. Phase 4 worked: v12 is fast enough now. But compared with official TD-MPC2, our implementation is still a simplified variant, and the MPPI gap is telling us exactly where to focus.

Short answer: **yes, the current learned model/planner stack is not planner-grade yet.** I would not say the whole world model is “bad” because the policy learns and consistency loss is low. But MPPI being worse than pi means the learned reward/dynamics/Q combination is not reliable enough for planning over sampled action sequences. In practice, MPPI is selecting actions that look good inside our learned model but are worse in the real env.

Official TD-MPC2 reaches about **267.9 @500k**, **373.1 @1M**, and **594.2 @4M** on hopper-hop seed 3 in hopper-hop.csv. Our v12 is much faster, but MPPI is still around **0.9 @500k** and **49.9 @1M**, while pi can be much better. That is an algorithm-quality gap.

**Main Differences**
| Area | Official TD-MPC2 | Ours Now |
|---|---|---|
| Collection | MPPI after seed steps | pi + exploration noise |
| Planner | 512 samples, 64 elites, 24 pi trajectories, adaptive std, H=3 | 256 samples, all-softmax, 1 pi trajectory, fixed std, H=5 |
| Model size | default single-task model_size=5: latent 512, MLP 512, Q=5 | latent 128, hidden 128, Q=2 |
| Actor | stochastic pi, separate optimizer, entropy, RunningScale | mostly deterministic mean behavior, no RunningScale |
| Target nets | target Q only | EMA target tree for all params |
| UTD | 1 update per env step, plus seed pretrain | K=64 / N=256 = 0.25 UTD |
| Planning distribution | model trained on MPPI-selected actions | model trained mostly on pi actions, MPPI samples can be OOD |

The planner mismatch is especially important. Official MPPI is in tdmpc2.py: it injects 24 stochastic policy trajectories, samples 512 candidates, keeps top 64 elites, adapts std, and samples the final action from elite scores. Our current planner in train_tdmpc_hopper_v12_fused.py uses 256 samples and a softer optimizer. That alone can make MPPI worse than pi even when the model is passable.

**Multi-Phase Plan**
**Phase 0: Diagnose Before Changing More**
Add diagnostics to v12/v13:
- pi return vs MPPI return with 20 episodes, not 5.
- reward prediction correlation on replay: predicted `two_hot_inv(reward_head)` vs real reward.
- 1-step and H-step latent/reward prediction error.
- Q percentiles and MPPI candidate return spread.
- Compare return of injected pi trajectory vs selected MPPI trajectory.

Success criterion: we know whether MPPI fails because of reward prediction, Q scale, dynamics rollout, or optimizer selection.

**Phase 1: Make MPPI Planner Official-Parity**
Implement official planner first, without touching training:
- `H=3`
- `num_samples=512`
- `num_elites=64`
- `num_pi_trajs=24`
- adaptive std clamped to `[0.05, 2.0]`
- top-k elite selection
- Gumbel/elite final action
- reset mean at episode start

This is the cleanest first experiment. If MPPI immediately rises toward pi or above pi, the world model is usable and our planner was the main issue.

**Phase 2: Fix Target and Actor Semantics**
Match official update structure:
- Use **target Q only**, not EMA copies of encoder/dynamics/reward/pi. Official only soft-updates target Q in world_model.py.
- TD target should use current pi action plus target Q, as official does in tdmpc2.py.
- Separate actor optimizer.
- Add RunningScale for policy Q values, as official does in tdmpc2.py.
- Re-enable entropy correctly, without dividing policy loss by reward scale.

Success criterion: pi remains strong, Q scale is stable, MPPI stops collapsing below pi.

**Phase 3: Match Official Model Capacity**
Create a parity config:
- latent_dim = 512
- mlp_dim = 512
- num_q = 5
- dropout = 0.01
- encoder lr scale = 0.3
- zero-init reward and Q output layers

Official default single-task model size is 5M according to README.md, and the 5M model uses latent 512 / MLP 512 in __init__.py. Our 128×128 model is great for speed, but it may simply be underpowered for official-level MPPI.

Success criterion: match or exceed official seed-3 curve within ~20% by 1M: target MPPI ≥300.

**Phase 4: Match Training Distribution**
This is probably the biggest conceptual gap: official collects with MPPI after seed steps in online_trainer.py. We collect with pi. That means our model is trained on pi-like actions, then MPPI evaluates many off-distribution action sequences.

Run two branches:
- **Parity branch**: batched MPPI collection with small N, e.g. N=8/16/32, K=N for UTD=1.
- **Hybrid branch**: 90-95% pi collection, 5-10% MPPI collection, store MPPI actions for distillation.

Success criterion: MPPI ≥ pi consistently after 500k.

**Phase 5: UTD and Discount Parity**
Once planner/model semantics are fixed:
- Use UTD=1: for N=256, K=256 with fused scan.
- Set discount to official effective value, likely 0.995 for 1000-step episodes through config.yaml.
- Add seed pretraining behavior: official does `seed_steps` updates at the seed boundary.

Success criterion: official-like curve, not just high pi:
- 500k: MPPI 250+
- 1M: MPPI 330-380
- 4M: MPPI 550+

**Phase 6: Recover Speed After Parity**
Only after matching official behavior:
- Bring back Phase 4 fused updates.
- Reduce model size carefully if quality survives.
- Distill MPPI into pi.
- Use sparse MPPI collection rather than every-step MPPI.

My recommended immediate next implementation is **Phase 1: official-parity MPPI planner**. It is isolated, easy to test, and directly answers whether the current model is bad or whether our simplified planner is leaving performance on the table. After that, Phase 2 target/actor semantics is the next highest-impact fix.