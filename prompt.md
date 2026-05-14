# blog0513


- we need punish frequent change of K, especially very short of K be one value for less then 5 frames. which always lead to bad performace.

- "Blog §5.5 reports K=4 seeds average 403.7, K=3 seeds average 310.6 — i.e. K=3 has a structural ceiling around 300 that no downstream tuning can break." what if K=4 also has a ceiling around 550? maybe we need a heirarchical clustering, 

- ! scalar K to semantic: use VLM to transform cluster K to symbol like "stance", "push-off", "flight", "landing" . then we can do symbolic reasoning on the sequence of symbols, e.g. "if next phase is flight, then current phase must be push-off".

- the planning step, canwe also use the abstraction idea? currently mppi would be worse than policy some times why?

- the integration of glass, is there any other possible way to integrate? why we have to design a mu and c to create P? and why the assignment is 16? you mention there is codebook in the tdmpc, do you mean the bin? can we use that?



- when we drop glass: 1. ablation on smoothing term (or we set K=4 on diff seed)


---
- One potential concern about our approach is that final performance may be lower than original td mpc because abstract states might be helpful to learn quickly in the beginning but that could be a blocker later (it may not completely solve the task due to abstractions). Hybrid approach (using both abstraction and raw latent space) may be a good alternative to consider. But let's see