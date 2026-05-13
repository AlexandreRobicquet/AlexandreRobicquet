# nanoWM — Blog series outline

Six posts, staged shallow-to-deep. Each post lands within 1–2 weeks of its
paired stage's code. Cross-links are mandatory: every post links back to the
previous, forward to the next, to the relevant arXiv refs, and to the specific
files in the repo.

**The video is the asymmetric variable.** Code without video = tier-3 outcome
(~200★, no cultural moment). Code + post + 60–90 min video walkthrough of even
Stage 1 alone = tier-1 reference link for 2–3 years. Treat the video as a
first-class deliverable, not an afterthought.

---

## Post 1 — Dreaming in 700 lines: replaying the Ha & Schmidhuber world model

**Audience.** Any ML engineer who has heard of world models. The widest entry
point in the series; carries no prerequisites beyond knowing what a VAE is.

**Hook (first 200 words).** Open on a side-by-side GIF: ground-truth CarRacing
on the left, pure-imagination rollout on the right. "In 2018, Ha & Schmidhuber
showed that an agent could learn to drive by *dreaming inside its own model of
the world*. Eight years later, you can reproduce the iconic figure in 700 lines
of PyTorch on a single 4090 overnight. This post walks through every line."

**Key figures.** The 4-panel GIF (ground truth / 1-step / 10-step / 50-step
open-loop). A latent-space PCA plot colored by track curvature. A `play.py`
screenshot with the MDN temperature slider.

**Sections.**
1. *V/M/C decomposition: what does it mean to learn a model of the world?*
2. *VAE → MDN-RNN → CMA-ES, line by line.*
3. *What's wrong with this picture: the three flaws that motivate everything
   that followed.*

**Cross-links.** Forward to Post 2 (the RSSM fixes flaw #1: the three-stage
pipeline doesn't share representation). arXiv `1803.10122`. Repo files
`stage1_ha/*.py`.

**Distribution.** Personal blog (canonical home), HF community blog
cross-post, Twitter thread with the GIF as the lead.

---

## Post 2 — Dreaming with gradients: building an RSSM from scratch

**Audience.** ML engineers comfortable with VAEs and actor-critic, new to
model-based RL.

**Hook.** A learning curve crossing the SAC baseline at 250k steps. "Dreamer
trains an agent entirely inside its own imagination. The actor never sees the
real environment after the first batch of data. Here's how — and the four
robustness tricks that take it from 'works on cheetah' to 'works on Atari and
Minecraft with the same hyperparameters'."

**Key figures.** Imagined cheetah running indefinitely (GIF). Sample-efficiency
curve. A 5-line excerpt of the KL-balancing loss with the α=0.8 highlighted.

**Sections.**
1. *The RSSM in five equations: the deterministic/stochastic split, posterior
   vs prior, free bits.*
2. *Actor-critic in imagination with analytic gradients: why this changes
   everything.*
3. *The DreamerV3 robustness stack: symlog, two-hot, free bits, KL balancing,
   unimix, percentile normalization. What each one buys.*

**Cross-links.** Back to Post 1 (this is V/M/C with shared representations and
learned exploration). Forward to Post 3 (the GRU → transformer swap). arXiv
`1811.04551`, `1912.01603`, `2010.02193`, `2301.04104`. Repo files
`stage2_dreamer/*.py`.

---

## Post 3 — World modeling is sequence modeling: a STORM-shaped transformer

**Audience.** Anyone who has read nanoGPT — explicit bridge from LLMs.

**Hook.** "If you can write a GPT, you can write a world model. The dynamics
RNN in DreamerV3 is just a sequence model. Replace the GRU with a causal
transformer over `(z_t, a_t)` tokens, and you get STORM — same actor-critic in
imagination, faster training, better numbers."

**Key figures.** An autoregressive Atari rollout GIF (Pong + Breakout). A
side-by-side code diff: nanoGPT's `CausalSelfAttention` block vs. our
identical block. An HNS bar chart vs. published STORM.

**Sections.**
1. *State-action token interleaving: how IRIS and STORM differ.*
2. *Identical to nanoGPT: the 200-line GPT block, line by line. Different from
   nanoGPT: the action token positions.*
3. *Sidebar: why STORM dominates IRIS in compute terms (actor-critic in latent
   space, not pixel space).*

**Cross-links.** Back to Post 2 (the RSSM was already a sequence model, we
just made it bigger). Forward to Post 4 (we still reconstruct pixels; what if
we didn't?). arXiv `2310.09615`, `2209.00588`. Repo files `stage3_storm/*.py`.

---

## Post 4 — Predict the latents, not the pixels: I-JEPA in 600 lines

**Audience.** SSL-curious researchers. Pairs with Elon Litman's *Annotated
JEPA* as the runnable companion to that text-only walkthrough.

**Hook.** "Reconstruction is wasteful. Most pixels are noise — leaves blowing,
clouds moving, sensor jitter. I-JEPA predicts not pixels but *latents*, in a
space the encoder has already decided is worth representing. The recipe is
ten lines. Getting it right is fiddly."

**Key figures.** The "back of the bird" decoder reveal. Attention map overlay
showing the predictor "looking at" the right unmasked patches. A
covariance-rank-over-time plot showing the model not collapsing.

**Sections.**
1. *The masking strategy: 4 target blocks at 15–20% each, one context block at
   85–100% with overlap removed. The genuinely subtle bit.*
2. *EMA momentum schedule, asymmetric architecture, smooth-L1 in feature space:
   the collapse-prevention story (contrast with VICReg, BYOL, LeJEPA).*
3. *Probing and visualization: how to know JEPA actually learned something,
   given that the loss curve is uninformative for model selection.*

**Cross-links.** Back to Post 3 (we still had a tokenizer; now we don't).
Forward to Post 5 (now we bolt action-conditioning on top). arXiv `2301.08243`,
`2511.08544` (LeJEPA). Repo files `stage4_ijepa/*.py`.

---

## Post 5 — nanoWM: a JEPA-shaped world model in 800 lines

**Audience.** Researchers tracking V-JEPA 2-AC, DINO-WM, foundation world
models. **The most original post in the series.**

**Hook.** "V-JEPA 2-AC learns to control a robot from 1M hours of internet
video plus 62 hours of robot demos. The recipe is small: frozen JEPA encoder,
a small action-conditioned predictor, plan by MPC in latent space. At
single-GPU scale, does it still work? Here's the experiment, the 800 lines,
and the answer."

(**If Stage 5 gate failed**, title becomes: *"What goes wrong when you try to
replace Dreamer's encoder with JEPA: a postmortem"* — honest negative-result
post, see `STAGE_5_GATE.md`.)

**Key figures.** Imagined Crafter trajectory leading to "make stone pickaxe"
with the achievement bar checking off. Side-by-side: this stage's planned
trajectory vs Stage 2's amortized-policy trajectory on the same task. The
latent-action codebook (capstone) visualized as 8 reconstruction templates.

**Sections.**
1. *Frozen encoder + action-conditioned predictor + sampling-MPC: V-JEPA 2-AC
   and DINO-WM stated minimally.*
2. *Planning vs amortized policy: the cleanest pedagogical contrast in the
   entire repo.*
3. *Latent action discovery (Genie-style capstone): self-supervised action
   codebook, why this is the bridge to internet-scale training.*
4. *Honest comparison to Stage 2's Dreamer: where each wins, where each loses.*

**Cross-links.** Back to Post 4 (the encoder we're freezing). Back to Post 2
(the contrast object). Forward to Post 6 (the diffusion stretch, optional).
arXiv `2411.04983` (DINO-WM), `2506.09985` (V-JEPA 2), `2402.15391` (Genie),
`2603.07083` (Dreamer-CDP). Repo files `stage5_nanowm/*.py`.

---

## Post 6 — (Optional stretch) Play your world: a minimal diffusion WM

**Audience.** Anyone who saw GameNGen or Oasis on Twitter and wondered how.

**Hook.** Screen recording: the author plays Pong inside the diffusion world
model. No env running. Score going up. "DIAMOND showed this works in 2024.
Nobody has done it in ≤1500 lines."

**Key figures.** The screen recording (GIF). A diagram of the EDM denoising
loop. A side-by-side: with vs without noise augmentation on past frames
(GameNGen's key trick) — the second one drifts after 30 frames, the first one
holds for 600+.

**Sections.**
1. *DIAMOND's thesis: discrete tokenizers throw away the wrong information.*
2. *The EDM denoising loop in 100 lines.*
3. *Noise augmentation on past frames: the single trick that prevents
   autoregressive error cascade.*

**Cross-links.** Back to Post 3 (tokens vs pixels was the choice; now we
sidestep both). arXiv `2405.12399` (DIAMOND), `2408.14837` (GameNGen). Repo
files `stage6_diamond/*.py`.

---

## Distribution plan

**Primary**: personal blog (`alexandrerobicquet.com` or equivalent) as the
canonical home. Permanent URLs, full author control, citable.

**Secondary**: cross-post each to **Hugging Face community blog** —
high distribution, low cost, indexed by HF's recommendation system, and HF is
where the world-model audience lives in 2026.

**Citable artifact**: a **single consolidated arXiv companion** at the end of
the series:
> *"nanoWM: a pedagogical walkthrough of world models, 2018–2026"*

This is the artifact that earns syllabus placement. Modeled on *The Annotated
Transformer* for citability, on *Neural Networks: Zero to Hero* for scope.
~40 pages, includes all code listings, every blog post compressed to a
chapter, all six headline figures, all references.

**Backup channel**: Substack for the lazy-discovery audience. Not the primary.

**Video**: a single 60–90 min walkthrough of **Stage 1** as the lead video.
This is non-negotiable — the video is what gates the upside (see
RISKS_AND_BUDGET.md, "Opportunity cost"). Subsequent stages can have shorter
videos (15–30 min) or be skipped, but Stage 1 needs the full Karpathy-style
treatment with the screen recording, the whiteboard derivation, and the live
training watch.

---

## Cadence

A post lands within 1–2 weeks of its paired stage's code. The post can drift
later than the code, but never more than 2 weeks — momentum is the secret
ingredient of a Karpathy-style project, and momentum is lost in 3-week gaps.

If a stage runs over schedule, the post for the *previous* stage gets
delayed accordingly. **Do not publish posts ahead of working code.** The
artifact is the argument; the post without the artifact is marketing.
