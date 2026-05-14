# nanoWM — Roadmap

Five stages, each with a hard go/no-go gate to the next. Stage 6 is an explicit
optional stretch. **The Stage-0 spike (week 1) gates whether Stage 5 is the
headline or whether Stage 2 is promoted to marquee** — see `STAGE_5_GATE.md`.

Code budgets are core model + training loop, **PyTorch only**, no RL framework.

---

## Stage 0 (week 1): the Stage-5 prototype spike

**Purpose.** Resolve the project's single biggest research risk *before*
committing to the eight-week build. The Stage-5 recipe (JEPA-encoder + latent
dynamics + sampling MPC) at single-GPU scale is unproven — V-JEPA 2-AC needed
1M hours of pretraining video; DINO-WM uses a strong frozen DINOv2.

**Scope.** Custom 2D continuous-action goal-reaching env, frozen DINOv2-S
encoder, 50-line action-conditioned MLP predictor, 100-line CEM planner. Total
~300 LOC.

**Gate.** See `STAGE_5_GATE.md` for the pass/fail spec. Outcome is binary:
**pass** → Stage 5 is the headline, build Stages 1→5 in order; **fail** →
demote Stage 5 to optional stretch, promote Stage 2 (nano-Dreamer) to the
project headline, restructure the blog series so post 2 is the marquee.

**Timeline.** 5 working days. Day 1: env + DINOv2 wrapper. Day 2: predictor +
data collection. Day 3: training + k-step rollout MSE. Day 4: CEM planner +
success-rate eval. Day 5: write up, decide.

---

## Stage 1 — "Train your dream": the Ha & Schmidhuber tribute

**Conceptual lesson.** V/M/C decomposition; frozen encoder + dynamics + tiny
controller; what's wrong with this picture (VAE wastes capacity on visual
noise, CMA-ES is slow, three-stage pipeline doesn't share representation).
This stage's *flaws* set up the rest of the build.

**Artifact.** 64×64 β-VAE + LSTM-MDN-RNN + linear controller trained with
CMA-ES on **CarRacing-v0**, reaching episode return 750–900.

**Files.** `01_vae.py`, `02_mdnrnn.py`, `03_controller.py`, `04_dream.py`,
`play.py`. Total ~700 LOC.

**Env.** `gymnasium[box2d]`'s CarRacing-v0. Pin the version explicitly — this
env has a notorious history of behavior shifts across gym/gymnasium versions.

**Compute.** Overnight on a single 3090 / 4090. ~$30 of cloud at $0.50/hr.

**Headline visual.** Side-by-side GIF: ground-truth CarRacing rollout vs.
pure-imagination rollout from the same initial frame. `play.py` lets the user
steer inside the model with a temperature slider on the MDN sampling.

**Why this stage exists.** It is the **most likely viral moment** of the
project — the iconic 2018 demo, in 700 lines, in 2026. Despite the existence of
`ctallec/world-models` and friends, nobody has shipped a clean modern
reproduction with a single-tweet "I made my model dream" demo.

**Gate to Stage 2.** Imagination rollouts must visibly track the same track
topology as ground truth for at least 30 steps. Headline GIF is rendered and
shareable. CMA-ES converges within budget.

**Blog post.** *"Dreaming in 700 lines: replaying the Ha & Schmidhuber world model"*

**Timeline.** Full-time: 1 week (3 days code, 2 days writing, 2 days CMA-ES
tuning + video). Part-time @ 1 day/week: 6 weeks.

---

## Stage 2 — "The RSSM is the spine": minimal Dreamer-on-pixels

**Conceptual lesson.** Deterministic/stochastic state split; why imagination
needs a posterior at train time and only a prior at rollout; analytic gradients
through dynamics; actor-critic in imagination; the symlog / two-hot / free-bits
robustness stack and why it matters at scale.

**Artifact.** Reduced DreamerV3 on **DMC `cheetah-run` (vision)** at 500k env
steps. Start with continuous Gaussian latents, upgrade to 32×32 discrete
categoricals (straight-through) midway through the blog post as the upgrade
narrative. Include KL balancing (α≈0.8), free bits (1 nat), and two-hot symlog
critic from day one *even when the env doesn't strictly need them* — they teach
the right lesson.

**Files.** `model.py`, `train.py`, `play.py`. Total ~1200 LOC.

**Env.** `dm_control` cheetah-run (vision). DMC walker-walk as the secondary
sanity check.

**Compute.** 6–10 h on a 4090 per seed. Plan 2 seeds for the headline curve.
~$25/seed cloud.

**Headline visual.** Imagined cheetah running indefinitely in pure latent
rollout (the iconic Dreamer figure). Plus a sample-efficiency curve crossing the
SAC baseline at 200–300k env steps.

**Highest-risk debugging zone.** RSSM. Plan for it: budget 3 of the 2 weeks
explicitly for RSSM debugging. Common failure modes: posterior collapse, KL
explosion early in training, two-hot critic getting stuck at the boundary
buckets. Mitigations: ship symlog and free bits day one; log KL per-dim
histograms; verify the posterior/prior gap doesn't collapse on a single batch
*before* full training.

**Gate to Stage 3.** Open-loop 50-step latent rollout produces a coherent
cheetah motion (visual judgment + per-step decoded reconstruction PSNR > 20).
Sample-efficiency curve crosses SAC baseline by 300k steps.

**Blog post.** *"Dreaming with gradients: building an RSSM from scratch"*

**Timeline.** Full-time: 2 weeks. Part-time: 12 weeks.

---

## Stage 3 — "Tokens or pixels?": one chapter, two transformer WMs

**Conceptual lesson.** World modeling as sequence modeling. VQ-VAE tokenizer +
GPT (IRIS) vs. state-action token interleaving with continuous latents
(STORM). Why STORM dominates IRIS in compute terms (actor-critic in latent
space, not pixel space).

**Artifact.** STORM-shaped transformer-replaces-GRU Dreamer variant — keep the
Stage-2 actor-critic-in-imagination, swap the RSSM GRU for a 6-layer causal
transformer over `(z_t, a_t)` tokens. Evaluate on Atari-100k **Pong** and
**Breakout** only (not the full 26).

**Files.** `model.py` (with a 200-line GPT block deliberately mirroring nanoGPT
naming so LLM-fluent readers recognize it instantly), `train.py`, `play.py`.
Total ~1000 LOC.

**Env.** `envpool` for the Atari speed (16x speedup over Gym).

**Compute.** 5 h/game on a 4090. ~$20/game.

**Headline visual.** A single rollout GIF generated by autoregressive sampling
from the transformer world model. HNS table showing it within 2× of the
published STORM number.

**Gate to Stage 4.** Autoregressive rollouts visibly correct for ≥30 frames on
both games. HNS measured and recorded.

**Blog post.** *"World modeling is sequence modeling: a STORM-shaped transformer"*

**Timeline.** Full-time: 1 week (transformer is fast for an LLM-fluent author).
Part-time: 5 weeks.

---

## Stage 4 — "Predict what's predictable": the JEPA pivot

**Conceptual lesson.** LeCun's argument against pixel reconstruction; I-JEPA
recipe (asymmetric encoder/predictor, EMA target, multi-block mask, smooth-L1
in feature space); why this is *not* contrastive and *not* generative;
collapse-prevention via stop-grad + EMA + asymmetric architecture (the same
trick as BYOL); LeJEPA's theoretical anchor (SIGReg loss).

**Artifact.** Single-file I-JEPA on **STL-10** reaching a respectable linear
probe (target: within 5 points of MAE-equivalent at 5× less compute). Plus a
50-LOC `lejepa_loss.py` as an alternative loss the reader can swap in.

**Files.** `model.py` (~200), `mask_collator.py` (~120), `train.py` (~200),
`probe.py` (~80), `lejepa_loss.py` (~50). Total ~650 LOC.

**Env.** STL-10's 100k-image unlabeled set. CIFAR-10 as a Stage-4 warm-up only;
TinyImageNet as a stretch.

**Compute.** 3–4 h on a 4090. ~$12.

**Highest-risk debugging zone.** The **mask collator** is the genuinely subtle
piece — 4 target blocks at 15–20% each, one context block 85–100% with overlap
removed. Off-by-one errors here silently kill training. Budget two days for
mask-collator debugging with a dedicated unit test that visualizes a batch of
context/target masks before any training.

**Headline visual.** Attention map overlay showing the predictor "looking at"
semantically-relevant unmasked patches when predicting masked targets. Plus a
tiny separately-trained pixel decoder revealing what the predictor "expected"
inside the masked region (the famous I-JEPA "back of the bird" figure).

**Gate to Stage 5.** STL-10 linear probe top-1 > 75%. k-NN probe top-1 > 70%.
Embedding covariance matrix rank > 0.8 × dim (collapse monitor passes).

**Blog post.** *"Predict the latents, not the pixels: I-JEPA in 600 lines"*

**Timeline.** Full-time: 1 week. Part-time: 5 weeks.

---

## Stage 5 — "nanoWM proper": JEPA encoder + latent dynamics + planner

**Conceptual lesson.** How to bolt action-conditioning onto a JEPA-trained or
DINO-frozen encoder; latent dynamics as a small causal transformer or MLP on
`(z, a)`; sampling-MPC (CEM/MPPI) in latent space; the V-JEPA 2-AC / DINO-WM
recipe stated minimally; explicit contrast with Dreamer's actor-critic-in-
imagination (**planning vs. amortized policy** — the cleanest pedagogical
contrast in the entire repo).

**Artifact.** Take the Stage-4 JEPA encoder (or, for speed and robustness, a
frozen DINOv2-Small from HF), freeze it, train a 200-line action-conditioned
predictor on **Crafter** (or **Craftax-Classic** if a JAX dependency is
acceptable for one stage), plan by CEM at test time. **Optional capstone:** a
Genie-style latent action model (~100 LOC) discovering a discrete action
codebook from action-free rollouts and using it for unsupervised pretraining of
the predictor.

**Files.** `encoder.py` (thin wrapper, ~100), `dynamics.py` (~200),
`planner.py` (CEM, ~150), `train.py` (~250), `latent_action_model.py` (~100).
Total ~800 LOC.

**Env.** Crafter as primary (published baseline ladder: human 50.5%, DreamerV3
~14%, EMERALD 58.1%). Craftax-Classic as the speed-up if JAX is acceptable —
reaches 1B env steps in <1 h on a 4090.

**Compute.** 6–8 h on a 4090. ~$25.

**Highest-risk debugging zone.** The **CEM planner**. Common failure: planner
collapses to the same action sequence regardless of goal (the dynamics
predictor doesn't actually condition on action). Mitigation: write a planner
unit test where the dynamics is a hard-coded ground-truth simulator — if CEM
fails there, the planner is wrong; if CEM works there but not with the learned
dynamics, the predictor is the problem.

**Gate to release.** Goal-reaching success rate ≥ 60% at fixed planning budget
on held-out start/goal pairs. **OR** Crafter geometric-mean achievement ≥ 10%
at 1M env steps (matches DreamerV3 within a margin).

**Headline visual.** Goal-conditioned rollout where the agent reaches a goal
location it has never seen, with the planned trajectory drawn over a top-down
view of the env. For Crafter: a GIF of imagined trajectory leading to "make
stone pickaxe" with the achievement bar checking off.

**Blog post.** *"nanoWM: a JEPA-shaped world model in 800 lines"*

**Timeline.** Full-time: 2 weeks (research risk lives here). Part-time:
12 weeks.

---

## Stage 6 (OPTIONAL STRETCH) — "Play your model": diffusion WM

Triggered **only** after Stage 5 ships and the blog series is live.

**Artifact.** Minimal DIAMOND-clone on one Atari game (Pong or Breakout), with
a `play.py` for the user to play *inside* the world model at 10–20 FPS.

**Files.** `unet.py` (~600), `edm.py` (~300), `train.py` (~400),
`play.py` UI (~200). Total ~1500 LOC.

**Compute.** 2–3 days on a 4090. ~$125.

**Headline visual.** Screen recording of the user playing Pong inside the
diffusion world model with no environment running, the player's score going
up. **Second viral candidate of the project.**

**Timeline.** Full-time: 2–3 weeks.

---

## Timeline summary

| Mode | Stages 1–5 | + Stage 6 stretch |
|------|------------|-------------------|
| Full-time | 8–11 weeks | 10–14 weeks |
| Part-time (1 day/week) | 9–12 months | 11–15 months |

The blog series and video can lag the code by 1–2 weeks per stage without losing
momentum. Do not let them lag more — the video is the asymmetric variable and
the code without the video is a fork of an existing repo.

## Top-3 debugging hot zones (allocate explicit budget)

1. **Stage 2 RSSM.** Posterior collapse, KL explosion, two-hot critic stuck.
   Budget 3 days of the Stage-2 fortnight for this *specifically*.
2. **Stage 4 mask collator.** Off-by-one errors silently kill training. Write
   the mask-visualization unit test *first*.
3. **Stage 5 CEM planner.** Action-independence failure mode. Write the
   ground-truth-dynamics planner test *first*.

If any of these three blows the budget by >2×, that stage's go/no-go gate fails
even if numbers look acceptable — quality of the pedagogy degrades when the
author is firefighting.
