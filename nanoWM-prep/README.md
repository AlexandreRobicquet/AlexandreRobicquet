# nanoWM

> **Status: prep / not yet started.** This directory stages the planning artifacts
> for `nanoWM`. It is not the project itself — when the build kicks off, these files
> migrate to a new standalone repo at `github.com/AlexandreRobicquet/nanoWM`.

## What this is

`nanoWM` is a Karpathy-style, single-file, five-stage build of a **world model**
that climbs from Ha & Schmidhuber's 2018 V/M/C tribute to a JEPA-encoder +
latent-dynamics + planner triad analogous to V-JEPA 2-AC and DINO-WM. Each stage
is standalone-runnable on a single consumer GPU, ships a checkpoint and a
`play.py`, and pairs with one blog post and a short video walkthrough.

## Why this exists

As of mid-2026, **no artifact has the cultural status of nanoGPT for any world
model.** The closest near-misses each miss at least three of the five things a
canonical teaching repo needs: two-file core, an incremental ladder, a line-by-line
narrative, a ≤90-min video walkthrough, ≤1 h to first interesting result. The gap
is not another DreamerV3 port (there are already six). The gap is the synthesis —
walking the reader through the field's 2018→2026 conceptual arc in one place,
ending at the predictive (non-reconstructive) latent triad that V-JEPA 2-AC and
DINO-WM suggest is the next decade's recipe.

## The one-tweet demo per stage

| # | Stage | Headline visual | Code | Compute |
|---|-------|----------------|------|---------|
| 1 | Ha & Schmidhuber tribute | CarRacing rollout vs. pure-imagination dream, side-by-side GIF + `play.py` | ~700 LOC | 1 night / ~$30 |
| 2 | RSSM Dreamer-on-pixels | Imagined cheetah running indefinitely; SAC-beating sample-efficiency curve | ~1200 LOC | 6–10 h / ~$25 |
| 3 | Transformer WM (STORM-shaped) | Autoregressive Atari rollout GIF; HNS within 2× of published STORM | ~1000 LOC | 5 h/game / ~$20 |
| 4 | I-JEPA pivot | "Back-of-the-bird" decoder reveal of what the predictor expected in masked region | ~600 LOC | 3–4 h / ~$12 |
| 5 | **nanoWM proper** (JEPA + latent dynamics + planner) | Imagined Crafter trajectory leading to "make stone pickaxe" achievement | ~800 LOC | 6–8 h / ~$25 |
| 6 | *Stretch*: playable diffusion WM | Screen recording of user playing Pong inside the diffusion model, no env running | ~1500 LOC | 2–3 days / ~$125 |

## Dependency policy (hard rules)

- **Allowed**: `torch`, `gymnasium`, `tqdm`. That's it for the core.
- **Per-stage as needed**: `dm_control` (Stage 2), `envpool` (Stage 3), `crafter`
  or `craftax` (Stage 5), `torchvision` (Stage 4 datasets), `timm` only for the
  frozen DINOv2 baseline in Stage 5.
- **Banned**: Hydra, Lightning, Accelerate, MLflow, Ninjax, `stable-baselines3`,
  `rllib`, `tianshou`, custom build systems. Any of these in a PR is an automatic
  reject — they are the things that turned every existing world-model repo into
  multi-directory mush.

## Honest performance disclaimer

In the spirit of `kc-ml2/SimpleDreamer`: this repo trades performance for
readability. The headline number on the README is **Crafter geometric-mean
achievement after 1M env steps on a single 4090**, paired with the open-loop
rollout GIF. Even matching DreamerV3's original ~14% in a sub-2000-line readable
codebase is the story. SoTA is not the goal; "learning curve visibly going up and
rollouts visibly correct" is.

Specifically, do **not** expect this repo to match:
- DreamerV3's published Crafter numbers (we use a reduced robustness stack).
- IRIS or STORM's published Atari-100k IQM (we ship 3 games, not 26, and we run
  one seed for the headline).
- DIAMOND's Atari-100k IQM (the Stage-6 stretch is one game, not 26, with a
  smaller U-Net).
- V-JEPA 2-AC's robot-manipulation zero-shot (Stage 5 demonstrates the *recipe*
  at single-GPU scale; the V-JEPA 2-AC result required 1M hours of pretraining
  video).

## Repo shape (at kickoff)

```
nanoWM/
├── README.md           # this file, expanded
├── stage1_ha/          # vae.py, mdnrnn.py, controller.py, dream.py, play.py
├── stage2_dreamer/     # model.py, train.py, play.py
├── stage3_storm/       # model.py, train.py, play.py
├── stage4_ijepa/       # model.py, mask_collator.py, train.py, probe.py
├── stage5_nanowm/      # encoder.py, dynamics.py, planner.py, train.py, play.py
├── stage6_diamond/     # OPTIONAL: unet.py, edm.py, train.py, play.py
├── eval/               # eval.py + expected.json (frozen golden numbers)
├── checkpoints/        # released via HF Hub, not committed
└── blog/               # markdown drafts + figures
```

Each `stageN_*/` directory is **self-contained**. A reader can `cd stage1_ha &&
python dream.py` without touching any other directory. Cross-stage shared utils
are explicitly banned to preserve the single-file aesthetic.

## How to use this repo (when it exists)

1. Read the blog post for stage N.
2. `cd stageN_* && python <main script>.py` — trains to first interesting result
   in ≤1 h on a single 4090.
3. `python play.py` — load the released checkpoint, play inside the model.
4. Run `python ../eval/eval.py --stage N` to reproduce the headline numbers
   against `expected.json`.

## Citations

The five-stage build threads the following papers; full BibTeX in `CITATIONS.bib`
once the repo exists:

- Ha & Schmidhuber 2018 (`1803.10122`) — V/M/C decomposition, CarRacing dream.
- Hafner et al., PlaNet (`1811.04551`), DreamerV1–V3 (`1912.01603`,
  `2010.02193`, `2301.04104`) — RSSM, imagination actor-critic, symlog/two-hot.
- Micheli et al., IRIS (`2209.00588`); Robine et al., TWM; Zhang et al., STORM
  (`2310.09615`) — transformer world models.
- Alonso et al., DIAMOND (`2405.12399`); Valevski et al., GameNGen
  (`2408.14837`) — diffusion world models for the Stage-6 stretch.
- Assran et al., I-JEPA (`2301.08243`); V-JEPA (`2404.08471`); V-JEPA 2
  (`2506.09985`); LeJEPA (`2511.08544`) — predict-the-latents.
- Zhou et al., DINO-WM (`2411.04983`) — frozen-encoder + small dynamics +
  MPC, the direct template for Stage 5.
- Bruce et al., Genie (`2402.15391`) — latent action model for the Stage-5
  capstone.

## See also (the projects this repo learns from)

- `karpathy/nanoGPT` — single-file pedagogy, the brand template.
- `kc-ml2/SimpleDreamer` — readable-over-performant Dreamer-V1.
- `NM512/dreamerv3-torch` — the de-facto PyTorch DreamerV3 reference.
- `gaasher/I-JEPA` — the closest single-file I-JEPA implementation.
- `eloialonso/diamond` — the playable-Pong precedent for Stage 6.
- `werner-duvaud/muzero-general` — the best-commented model-based-RL repo in
  existence; the bar for inline pedagogy.
