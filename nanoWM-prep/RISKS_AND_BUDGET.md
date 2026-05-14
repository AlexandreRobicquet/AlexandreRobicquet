# nanoWM — Risks and Budget

Five characteristic failure modes for educational world-model projects, two
project-specific risks, a per-stage compute budget, and an opportunity-cost
framing.

---

## The five characteristic failure modes

### 1. Scope creep

**Failure pattern.** Trying to support DMC + Atari + Crafter + Minecraft +
ImageNet, ending with no env trained well. The most common cause of
abandonment in this space.

**Mitigation.**
- **Hard ban** on adding envs not in the ROADMAP. CarRacing (S1), DMC (S2),
  Atari Pong+Breakout (S3), STL-10 (S4), Crafter (S5). That is the entire
  list.
- One *optional* stretch env per stage max, called out explicitly in the
  blog post as "if you have a spare GPU-day."
- PRs that add envs are auto-rejected unless they replace an existing one.

### 2. RL plumbing eats LOC

**Failure pattern.** Replay buffers, env vectorization, frame-stacking, reward
normalization, eval rollouts, distributed actors — these quickly bloat to
thousands of lines and obscure the actual model.

**Mitigation.**
- Single 150-line `replay_buffer.py` per stage; copy-paste between stages is
  *fine* and explicitly preferred over a shared utility.
- Use Gymnasium's built-in vectorization (`gym.vector.SyncVectorEnv`,
  `AsyncVectorEnv`). No custom vectorization code.
- **Banned imports**: `stable-baselines3`, `rllib`, `tianshou`, `acme`,
  `cleanrl` (we already learn from CleanRL's philosophy but we don't depend
  on it).

### 3. Wrong env choice

**Failure pattern.** Picking MineRL (Java, days of training, brittle setup) or
the full Atari-26 suite and never finishing.

**Mitigation.**
- The ladder in ROADMAP.md is the contract. Crafter is the headline RL env
  (published baseline ladder from human 50.5% down to DreamerV3 ~14% to early
  baselines ~2%; the curve is legible).
- Craftax-Classic (JAX) is acceptable as a one-stage speedup at Stage 5 if
  the >250× wall-clock speedup pays for itself.
- **Banned**: Minecraft / MineRL, GameNGen-style Doom, full ImageNet (16x A100
  budget), MuJoCo manipulation suites.

### 4. No canonical eval → numbers drift

**Failure pattern.** Each stage's "result" is a different metric, computed
differently. Regressions go invisible. The repo claims X, the user reproduces
0.7X, and the trust evaporates.

**Mitigation.** `EVAL.md` is frozen *before* any model code is written. Four
canonical eval calls, all implemented in a single `eval/eval.py`, all
producing JSON that matches `eval/expected.json`. PRs that fail eval CI are
auto-rejected. A baseline-update protocol forces explicit documentation when
a regression is acknowledged.

### 5. Training instability

**Failure pattern.** Dreamer killer: reward scale on Atari blows up the critic
without symlog/two-hot. JEPA killer: silent posterior collapse — the model
"trains" but learns identity. Diffusion killer: autoregressive error cascade
without noise augmentation.

**Mitigation.**
- **Ship the robustness stack from day 1**, even when the env doesn't strictly
  need it. Stage 2 includes symlog + two-hot + free bits + KL balancing from
  the first commit. The first chapter of post 2 *explains* what each one does;
  it doesn't ablate them.
- **Collapse monitor** (per EVAL.md) logged every 100 steps in Stage 4 and
  Stage 5. Alert thresholds, not just metrics. Unmonitored JEPA = failed JEPA.
- **Noise augmentation** on past frames is in the Stage-6 stretch from line 1.

---

## Two project-specific risks

### Risk A — Naming collision with `simchowitzlabpublic/nano-world-model`

**Reality.** A repo named `nano-world-model` exists from Feb 2026, ~46★, by
the Simchowitz lab. It is a diffusion-forcing video model, multi-file, Hydra-
heavy — but the name is taken in spirit and may sharpen the collision over
the coming year.

**Mitigation.**
- **Differentiate in the README**: lead with "single-file, single-GPU,
  incremental ladder" — characteristics the existing nano-world-model
  deliberately does not have.
- **Different scope framing**: theirs is a model architecture (NanoWM-B/2,
  NanoWM-L/2 are sizes). Ours is a pedagogical artifact spanning the full
  field, with the eponymous Stage 5 being just one component.
- **Fallback name**: `wm-from-scratch`. Decision deferred to post-Stage-1
  launch; if the collision causes user confusion in the first 30 days
  post-launch, rename.

### Risk B — Stage 5 unproven at single-GPU scale

**Reality.** V-JEPA 2-AC pretrained on 1M hours of internet video. DINO-WM
relies on a strong off-the-shelf frozen DINOv2-base. Whether the recipe
survives in our single-4090 regime is a genuine open question.

**Mitigation.**
- **Stage 0 prototype gate** (see `STAGE_5_GATE.md`). One-week spike with
  un-fudge-able pass criteria. If gate fails, Stage 5 demotes to optional
  stretch and Stage 2 nano-Dreamer becomes the marquee.
- The failure branch is named explicitly in the gate doc, so sunk-cost
  reasoning two months in cannot rationalize a failed gate.
- Negative result is still a valuable artifact — there is no public
  negative-result writeup for JEPA-WM at single-GPU scale.

---

## Compute budget

Per-stage estimates assume 3–5 iterations before the final clean training run
that produces a shipped checkpoint. Cloud 4090 at ~$0.50/hr, A100 at $1.50/hr.

| Stage | Compute estimate | Cost |
|-------|-----------------|------|
| 0 (prototype gate) | 5 days × 3–4 h/day | $15 |
| 1 (Ha & Schmidhuber) | 3 nights × 8 h | $30 |
| 2 (Dreamer) | 4 runs × 8 h, 2 seeds | $200 |
| 3 (STORM) | 3 games × 2 seeds × 5 h | $150 |
| 4 (I-JEPA) | 4 runs × 4 h | $50 |
| 5 (nanoWM) | 4 runs × 7 h | $150 |
| **Subtotal Stages 0–5** | | **$595** |
| Misc + final clean runs | | $200 |
| Margin (50%) | | $400 |
| **Budget for Stages 0–5** | | **$1,200** |
| 6 (diffusion, stretch) | 2 runs × 3 days | $500 |
| **Budget with Stage 6** | | **$2,000** |

The actual spend is more likely to come in at $1,200–1,500 for Stages 0–5.
**Budget $2,000 end-to-end** with margin; this is the number to greenlight
the project.

If the Stage-0 gate fails: Stage 5 budget reallocates to extra Stage 2 seeds
and a longer Stage 3, so total spend stays roughly flat.

---

## Timeline budget

| Mode | Stages 0–5 | + Stage 6 |
|------|------------|-----------|
| Full-time | 9–12 weeks | 11–15 weeks |
| Part-time @ 1 day/week | 9–13 months | 11–16 months |

The blog series and video can lag the code by 1–2 weeks per stage without
losing momentum. Beyond 2 weeks lag = lost momentum = tier-3 outcome.

---

## Opportunity cost — the tier-1 vs tier-3 bet

**Tier-1 outcome (the upside, ~30% conditional on shipping the video):**
- nanoWM becomes the default reference link in "how do I learn world models"
  threads for 2–3 years.
- 2k–5k GitHub stars within a year.
- Speaking slot at a major conference (NeurIPS workshop, ICML tutorial).
- The arXiv companion lands on syllabi.
- The author becomes a known voice in the world-model conversation —
  positioned for the next round of foundation-WM research.

**Tier-2 outcome (the realistic middle, ~50%):**
- Solid pedagogical artifact, 500–1500 stars, referenced from a few blog
  posts and one or two papers.
- Negative-result Stage-5 post is a genuine community contribution.
- Author gets recognized within the WM/JEPA community without breaking out
  to wider ML.

**Tier-3 outcome (the floor, ~20% — what happens if the video is skipped):**
- ~200 stars, a few citations, no cultural moment.
- The code is a fork of NM512/dreamerv3-torch in shape, just better-commented.

**What gates the upside.** The video. Specifically the Stage-1 video. The
code alone is necessary but insufficient; the Annotated-Transformer-quality
narrative + a 60–90 min Karpathy-style walkthrough is the combination no
existing project has shipped.

**What's the cost of attempting.** 9–12 weeks full-time (or 9–13 months
part-time) and $1.2–2k of compute. The cost of attempting and ending at
tier-3 is the time; tier-2 alone justifies it; tier-1 is asymmetric upside.

**The recommended bet.** Greenlight Stage 0 immediately. Use the gate to
de-risk the headline. Treat the Stage-1 video as a non-negotiable
deliverable, even if it pushes the Stage-1 timeline from 1 week to 2.

---

## Three deal-breakers (any one of these triggers a project pause)

1. **Stage 0 gate fails AND the author is not personally interested in a
   Dreamer-only project.** The whole pitch was the JEPA-WM synthesis; if
   that's off the table and the author's heart isn't in Dreamer, the
   opportunity cost no longer pencils.
2. **Two of the three top debugging hot zones (RSSM, mask collator, CEM
   planner) each blow their stage budgets by >2×.** Indicates the author is
   firefighting rather than teaching, and the pedagogical quality will degrade.
3. **A direct competitor ships first.** If, in the next 6 months, another
   project ships a single-file pedagogical world-model artifact with a video
   walkthrough that hits the same five-criteria bar, the cultural opening
   closes and the bet's upside shrinks to tier-2 at best. Recovery: pivot
   the project to be explicitly complementary (e.g., the JEPA-only branch
   if they did Dreamer, or vice versa).
