# Stage 5 — Prototype Gate Spec

The project's single biggest research risk: **the JEPA-encoder + latent-dynamics +
sampling-MPC recipe at single-GPU scale is unproven.** V-JEPA 2-AC needed 1M hours
of pretraining video; DINO-WM relies on a strong off-the-shelf frozen DINOv2.
Whether the recipe survives in our regime is a real open question, and the project
itself is partially an answer to it.

This gate is the one-week spike that answers the question *before* committing to
the eight-week build. Outcome is binary. Pass → Stage 5 is the project's headline
and Stages 1–5 build in order. Fail → Stage 5 demotes to optional stretch and
**Stage 2 nano-Dreamer is promoted to the project's marquee**.

The gate is designed to be **un-fudge-able**: pass criteria are quantitative,
held-out splits are defined upfront, and the failure branch is named so the
author cannot talk themselves into passing a failing spike.

---

## Spike scope

Five working days. Total ~300 LOC. PyTorch only.

### Env

Custom 2D continuous-action goal-reaching environment in pure PyTorch (no Gym):
- 84×84 RGB observation with a colored agent dot and a colored goal dot on a
  textured background (random procedural noise per episode — this matters; a
  clean background trivializes the task and hides predictor failure modes).
- 2D continuous action `a ∈ [-1, 1]²` interpreted as a velocity delta.
- Episode length 50 steps.
- Success := agent dot within 5 pixels of goal at episode end.

Why this env: small enough to train end-to-end in <1 hour, has visual noise
JEPA is supposed to handle, has known-correct dynamics (the planner test in
"Hot-zone 3" of ROADMAP.md can use the ground-truth simulator).

### Encoder

Frozen `dinov2_vits14` from HF (`facebook/dinov2-small`). 384-dim patch
embeddings, take the CLS token. Not the Stage-4 I-JEPA encoder — the spike
deliberately avoids depending on Stage 4's success, so spike results
attribute cleanly.

### Predictor

50-line MLP: `(z_t [384], a_t [2]) → z_{t+1} [384]`. 3 hidden layers, width 512,
GeLU, no normalization. Trained with smooth-L1 in the encoder's representation
space against the *next-frame's* DINOv2 embedding.

### Data

Collect 10k rollouts of 50 steps each from a random policy. That's 500k
transitions; trivially fits in RAM. No replay buffer needed for the spike.

### Planner

100-line CEM (Cross-Entropy Method):
- Plan 5 steps ahead.
- 100 candidate trajectories per iteration.
- Top-10 elite refit Gaussian over actions.
- 3 iterations.
- Goal: minimize L2 distance between predicted `z_{t+5}` and a target embedding
  computed from a "goal observation" (the env rendered with the agent placed at
  the goal location).

### Eval split

Held-out: 200 (start, goal) pairs sampled with seeds 1000–1199. The spike never
sees these during training. The training random policy uses seeds 0–999.

---

## Pass criteria

**Both** must hold on the held-out split:

### Criterion A — k-step latent rollout MSE

Plot k-step prediction MSE in the DINOv2 embedding space for `k ∈ {1, 2, 5,
10, 20, 50}`, averaged over 200 held-out trajectories. **Pass** if:
- k=1 MSE < 0.05 (normalized by encoder embedding variance).
- k=10 MSE < 5× the k=1 MSE.
- The curve is monotonically increasing (no weirdness from training collapse).

A visual sanity check accompanies the number: decode `z_t` via a tiny separately-
trained pixel decoder for `k ∈ {1, 5, 10}` and confirm the agent dot is in
roughly the right place. **This visual sanity is part of the gate.** A passing
MSE with visually wrong rollouts fails the gate.

### Criterion B — Goal-reaching success rate

Run the CEM planner on all 200 held-out (start, goal) pairs at fixed budget
(5-step horizon, 100 candidates, 3 iterations). **Pass** if success rate ≥ 50%.

For calibration: a random policy on this env scores ~5% success. A perfect
oracle planner (using ground-truth env as the dynamics model) should score
>95%. The 50% bar is intentionally lenient — the spike is asking *does the
recipe work at all*, not *is it competitive*.

---

## Failure modes to expect and what they tell you

- **k=1 MSE good, k=10 MSE bad, rollouts hallucinate goal locations.**
  Compounding error. Could be fixed in the full Stage 5 (longer horizon
  training, noise augmentation à la DIAMOND, multi-step loss). **Conditional
  pass:** continue to Stage 5 but plan for multi-step training in the budget.
- **k=1 MSE bad.** The predictor isn't learning. Almost certainly an MLP
  capacity or data issue, not a fundamental recipe issue. **Conditional pass:**
  scale predictor to 6 layers / width 1024 and rerun once.
- **k-step MSE good, success rate <30%.** The planner doesn't condition on
  action — predictor is ignoring `a_t`. **Hard fail.** This is the failure mode
  V-JEPA 2-AC papers warn about: action-conditioning a frozen self-supervised
  encoder is non-trivial. If this happens, Stage 5 demotes.
- **k-step MSE good, success rate 30–50%.** Borderline. **Soft fail.** The
  recipe sort-of works but the headline demo would be unconvincing. Demote
  Stage 5; revisit as a stretch chapter only if Stage 4 ships a strong I-JEPA
  encoder that might do better than frozen DINOv2-S.
- **Both criteria pass.** Build the project as planned. Stage 5 is the
  headline.

---

## The failure branch (named explicitly)

If the gate fails:

1. **Stage 5 demotes** to "optional stretch chapter" status, alongside Stage 6.
2. **Stage 2 (nano-Dreamer) becomes the project's marquee.** The README's
   one-tweet demo per stage promotes the imagined cheetah running indefinitely
   as the headline GIF.
3. **Blog post 5's title changes** to: *"What goes wrong when you try to
   replace Dreamer's encoder with JEPA: a postmortem"* — and the post is the
   honest negative result rather than the success story. **This is still a
   valuable artifact** — there is no public negative-result writeup for
   JEPA-WM at single-GPU scale, and the field would learn from it.
4. **The arXiv companion's framing shifts**: from "a pedagogical walkthrough
   ending in the JEPA-WM triad" to "a pedagogical walkthrough of the
   reconstructive lineage, with an honest section on why the JEPA pivot is
   harder than it looks at small scale."
5. **No prestige hit**: the negative result is the contribution. The repo
   still has Stages 1–4 + a Dreamer headline + a postmortem post, which is
   tier-2 at minimum.

---

## What the author does on day 5

Write one document, `STAGE_5_RESULT.md`, with:
- The plotted k-step MSE curve (PNG checked in).
- The 200-pair success rate.
- A 2-paragraph honest assessment.
- A binary decision: **PASS** or **FAIL**.
- If FAIL: the restructured roadmap for Stages 1–4 + postmortem.

This file is committed and pushed before any Stage 1 code is written. It is
the artifact that prevents sunk-cost reasoning over the following two months.
