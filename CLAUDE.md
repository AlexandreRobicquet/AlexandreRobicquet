# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this repo is

A single-file personal intro / research portfolio for Alexandre Robicquet.
The only content is `README.md`, which renders as the landing page on
GitHub (https://github.com/AlexandreRobicquet/Intro).

There is **no code, no build system, no tests, no CI**. Treat this as a
documentation repository.

## Repository layout

```
.
├── README.md     # the entire deliverable
└── CLAUDE.md     # this file (guidance for AI assistants)
```

If you find yourself wanting to add directories, source files, configs,
or tooling, stop and confirm with the user first — it almost certainly
does not belong here. Code projects mentioned in the README
(`pairwise-prm`, `tts-zoo`, `nanoRLM`, etc.) live in their own
repositories.

## Editorial conventions for README.md

The README is short and load-bearing. Match its existing voice when
editing:

- **Framing first.** The README opens with the guiding question
  ("If models improve themselves, how do we know the improvement is
  real?") before listing anything. Keep that ordering.
- **Plain prose, lowercase bullet items.** Sections use sentence-case
  headings (`## Currently Building`, `## Public Repos`, `## Links`).
  Bulleted project names are wrapped in backticks (e.g. `` `pairwise-prm` ``).
- **One-line project descriptions** in the form
  `` `name` - short description ``. Do not pad with adjectives.
- **No emojis, no badges, no shields.** The current README has none —
  don't add them unless explicitly asked.
- **Archived repos stay labeled "archived legacy …"** so the reader
  knows not to look there for current work.
- **Links section** uses bare URLs, not markdown link text. Keep that
  style.

When updating project status (e.g. moving something from "Currently
Building" to "Public Repos"), preserve the existing section structure
rather than introducing new sections.

## Git workflow

- **Branching.** Work happens on `claude/<topic>-<suffix>` branches
  (see `git log` for prior examples like
  `claude/add-claude-documentation-FLk68`,
  `claude/remove-nanowm-prep`).
- **Default branch is `main`.** PRs target `main` and are merged via
  GitHub's merge-commit flow (`Merge pull request #N from …`).
- **Commit messages.** Short, imperative, lowercase-leading where the
  prior style does (`plan: nanoWM project prep`,
  `Refine profile README for research portfolio`). Match what's already
  in `git log` rather than imposing a new convention.
- **One change per PR.** Past PRs are small and single-purpose
  (add the README, remove a planning subdir, etc.). Keep that
  discipline.
- **Never force-push, never amend a pushed commit**, and never push
  directly to `main`.

## Operating principles for assistants

- **Don't invent structure.** No `package.json`, no linters, no
  pre-commit hooks, no GitHub Actions unless the user asks for them.
- **Don't speculate about the research projects.** The README
  intentionally describes them at a high level. If the user asks for
  expansion, ask which project and what level of detail before writing.
- **Don't add a license file, code of conduct, or contributing guide**
  on your own initiative — this is a personal intro repo, not an OSS
  project.
- **Keep edits minimal and reversible.** A README change should usually
  touch a handful of lines, not restructure the whole file.
- **Verify before claiming.** After editing `README.md`, re-read it to
  confirm the rendered Markdown still parses cleanly (headings,
  bullets, backticks balanced).

## When in doubt

Ask. This repo's value is its concision; the failure mode is bloat.
