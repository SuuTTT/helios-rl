# finding
all seed move forward, and gait is rythmic(seed 1 jump using foot, but with one fall down, just like human; seed 3 4 use knee than toe,  ). they all complete 10 second jumping 

at begining of video, K all =0 , before hopper was drop to ground(seed 1 the hopper back to floor, seed 2 3 all face to floor) then:
seed4: K=0 when knee on ground, until the torso 90 degree to floor and knee leave ground, then turn to K=5,K=5 when the hopper jump forward(stretching leg then ;leg, torso in one line, 90degree to 0 degree to floor); and k=0 when knee again on ground and nose on ground, then it just keep repeate switching between K=0 and K=5,

seed3: K=2 when jump forward; k=4 when 45 degree to floor; k=5 when torso 0 degree to floor(not touching ground); k -1 when nose touch ground and knee touch ground; k = 4 when torso 90 to ground with knee on ground; k=2 when knee leave ground(just 1 frame); k =  4 when toe on ground;k=2 then loop as begin.

seed1: when in air, switch between K=6 K=7 (6,7,6,7,6) until it lay and begin to sit up; and k=3 when hip leave ground; k=4 when it leg >90degree on knee, and it keep jumping 7 times until 6 second; where he fall down and k=7, until his head touch ground; and k=6 when head leav ground; k=3 when he begin to standup, k= 4 when he begin to jump
# TD-MPC-Glass HopperHop — Video review checklist

When watching a rendered rollout MP4 from `scripts/render_glass_rollout.py`, this
is what to look for. Output for each video should be a short structured note I can
use to update the ../iterations/iteration_2_lessons document.

The MP4 shows 3 episodes back-to-back, each up to 1000 steps. Overlay in the
top-left corner: a coloured square + text `K={cluster_id}  R={cumulative return}`.
The "K=" is `argmax(S[argmax(z·μᵀ)])` — i.e. which Glass cluster the current
latent is currently assigned to. There are at most 4 (in K=4 basin seeds) or 3
(in K=3 basin seeds) distinct colours.

---

## Per-video checklist

For each video (e.g. `phasef/seed_1_best_mppi.mp4`), record:

### A. Behavioural questions (don't need to know RL to answer)

- [ ] **A1.** Does the hopper move *forward* across the frame, or stay in place?
  (Watch the cam0 tracking — if the hopper moves and the camera tracks, the
   floor texture should scroll past.)
- [ ] **A2.** Is the gait *rhythmic*? (Foot-strikes evenly spaced, like a real
  jogger.) Or jerky / irregular / lopsided?
- [ ] **A3.** What's the *failure mode* if it falls? Fall forward / sit down /
  topple sideways / "fold up" with body collapsing?
- [ ] **A4.** Episode length: did each of the 3 episodes complete the full ~1000
  frames? Or did one or more fall early? Roughly how many frames each.
- [ ] **A5.** Estimated hops per second (a hop = foot leaves ground and lands
  again). Just a rough count over the first ~5 seconds.

### B. Cluster-vs-gait alignment (this is the meaty part)

- [ ] **B1.** How many *distinct* cluster numbers (K=0..7) appear over a full
  episode? Expect: K=4 basin → 4 distinct, K=3 basin → 3 distinct.
- [ ] **B2.** Does the cluster number change *synchronously with the gait*?
  i.e. each foot-strike → same cluster transition, then the next phase →
  another consistent cluster. Or do the clusters change at seemingly random
  times (drift, no phase lock)?
- [ ] **B3.** Can you label each cluster with a *gait phase*? Candidates:
   - "Stance" — foot is on ground, body decelerating then pushing off
   - "Push-off" — foot still on ground but body accelerating, ankle extending
   - "Flight" — both feet (here, just one foot for Hopper) off the ground
   - "Landing" — foot touching down, body absorbing impact
   - Or, if it's *not* a real hop: "lean-forward", "lean-back",
   "fold-collapse-recover", etc.
- [ ] **B4.** Does episode 2 use the *same cluster→phase mapping* as episode 1?
  i.e. are clusters consistent across episodes, or do they re-label per episode?

### C. Quality + return calibration

- [ ] **C1.** Final cumulative return shown in the overlay at the end of each
  episode. Should match the CSV's `pi` reward for that checkpoint to within
  ~10 (single-episode variance).
- [ ] **C2.** Subjective rating: 1=falls in <100 frames; 5=runs 1000 frames
  with reasonable hops but slow; 10=textbook continuous hopping forward.

---

## Cross-video comparison (only when 2+ videos available)

When comparing two seeds (e.g. seed 1 vs seed 3, both K=4 basin):

- [ ] **D1.** Are the *gait shapes* visibly different, or do they look the same
  apart from how often the hopper falls?
- [ ] **D2.** Do the *cluster colours used in similar gait phases* match across
  videos? If seed 1's "stance" cluster has the same colour as seed 3's
  "stance" cluster → Glass's K=4 partition has a stable semantic meaning. If
  the colours don't correspond → the partition is unstable across runs.
- [ ] **D3.** For a stuck seed (e.g. seed 3): which cluster does the hopper
  spend most of its time in? If 80%+ of frames are one colour → the hopper is
  "stuck in a phase", e.g. permanently in stance (= sitting / shuffling).

For a K=3 seed (Phase-f seed 4) vs K=4 seed (Phase-f seed 1):

- [ ] **E1.** Does the K=3 seed have a visibly worse gait — e.g. only 3 phases,
  missing the "flight" phase, just shuffling?
- [ ] **E2.** Same comparison as D2 but cross-basin: does any cluster in K=3
  seed look like the "stance" colour from K=4 seed? Would suggest K=3 is just
  K=4 with two phases merged.

---

## What I'll do with the answers

The behavioural notes (A) confirm or refute the blog §5.6.1 "balanced shuffle"
hypothesis for stuck seeds. The cluster-alignment notes (B) tell us whether
Glass is learning a *useful* partition or just any partition with the right
spectral properties. The cross-video notes (D, E) directly answer whether the
K=3 vs K=4 distinction corresponds to a real *gait* distinction, which feeds
directly into the Phase-i / Phase-j basin-stability fix design.

If clusters are *not* gait-aligned (B2 says random) → Glass is doing
representation learning but not behavioural clustering. The structural-entropy
loss is finding *some* partition but not the *right* one. Suggests a different
clustering objective (e.g. timecontrastive) might do better.

If clusters *are* gait-aligned (B2 says phase-locked) → Glass works as intended.
The bottleneck for stuck seeds is then downstream (policy/critic), and we
should focus the next iteration on Q-overestimation fixes (proper Q-reset,
distributional Q, prioritised replay).
