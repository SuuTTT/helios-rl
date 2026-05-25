# Research Workflow Platform

Date: 2026-05-25

This document generalizes the current dashboard, task queue, and idea queue into
a reusable research operating system. It is designed for any empirical research
project where the loop is:

```text
idea -> literature/benchmark discovery -> local testbed -> probes -> metrics
-> SOTA comparison -> milestone blog -> manuscript -> release
```

The current TD-MPC-Glass stack is the first implementation. The goal is to make
the same system work for future RL, ML systems, robotics, generative modeling,
or benchmark-engineering projects.

## Product Goal

The system should let you write a research idea and target, then have agents do
most of the mechanical work:

- collect and summarize prior work;
- identify SOTA claims, benchmark tasks, metrics, and reproducibility gaps;
- clone or reimplement benchmark code locally;
- create a reproducible environment;
- write baseline and probe launchers;
- queue experiments across GPU/CPU workers;
- monitor metrics, failures, and cost;
- generate next ideas based on evidence;
- escalate promising probes into multi-seed confirmation;
- produce blog posts at milestones;
- produce a LaTeX manuscript once enough evidence exists.

The human remains the principal investigator: approving claims, paper framing,
benchmark fairness, and final publication decisions.

## Core Objects

### Project

A project is the top-level research effort.

Examples:
- `tdmpc-glass-hopperhop`
- `fast-diffusion-sampler`
- `llm-agent-code-repair`
- `robot-learning-contact-rich-control`

Project state should include:
- title;
- target claim;
- benchmark suite;
- SOTA baselines;
- metric definitions;
- environment definition;
- artifact locations;
- active ideas;
- publication status.

Template: `research_system/templates/project.yaml`.

### Idea

An idea is a hypothesis worth testing.

Example:

```text
Glass helps early representation but late Glass loss hurts gait refinement.
Try turning Glass off at 1M, 1.5M, 2M, and compare against always-on.
```

Idea state lives in `scripts/queues/idea_queue.json` and links to central queue
tasks once probes are generated.

### Benchmark

A benchmark is the local executable testbed:
- tasks/datasets;
- SOTA implementations;
- official metrics;
- evaluation scripts;
- baseline configs;
- reproduction notes.

For benchmark work, success is not "our model runs"; success is that the system
can reproduce at least one reference baseline well enough to make future
comparisons defensible.

### Probe

A probe is the smallest executable experiment that can change the decision.

Rules:
- one seed per task;
- fresh output tag;
- explicit pass/kill rule;
- fixed code identity;
- bounded budget;
- tracked in central queue.

### Evidence

Evidence is a structured observation:
- metric table;
- curve;
- log failure;
- video;
- ablation result;
- benchmark reproduction result;
- code diff;
- cost/time summary.

Evidence must link back to project, idea, probe, task id, commit SHA, and
artifact path.

### Milestone

A milestone is a publishable step:
- benchmark built;
- baseline reproduced;
- first SOTA-beating result;
- robust multi-seed result;
- ablation completed;
- paper-ready dataset frozen;
- release candidate.

Milestones trigger blog/manuscript generation.

## System Architecture

```text
Human idea
  |
  v
Idea Queue
  - rough hypothesis
  - target metric
  - priority
  - owner/agent claim
  |
  v
Research Agent
  - reads project state, docs, current results
  - asks GPT Deep Research output for SOTA/benchmarks
  - creates benchmark plan and probe specs
  |
  v
Project Workspace
  - cloned benchmark repos
  - local reimplementation
  - configs
  - launchers
  - paper/blog drafts
  |
  v
Central Run Queue
  - one seed/task
  - GPU/CPU worker assignment
  - cost and ETA
  - stale-running recovery
  |
  v
Workers
  - VastAI, EC2 GPU, local GPUs, CPU boxes
  - Docker/GHCR env
  - W&B metrics
  - S3/HF artifacts
  |
  v
Dashboard
  - task queue
  - idea queue
  - project status
  - SOTA target gap
  - promising probes
  - next idle worker
  |
  v
Evidence Store
  - CSV/JSON metrics
  - summaries
  - plots
  - videos
  - checkpoints
  |
  v
Decision Engine
  - promote
  - mutate
  - retire
  - clean confirmation
  |
  v
Publication Queue
  - GitHub Pages blog
  - LaTeX manuscript
  - release checklist
```

## Workflow

### 1. Capture Idea

The human writes:
- rough idea;
- why it might work;
- target metric;
- constraints;
- desired publication venue or audience.

Use:

```bash
/root/venv/bin/python3 scripts/idea_queue.py add \
  --title "..." \
  --goal "..." \
  --hypothesis "..." \
  --metric "..." \
  --tags "project,topic,benchmark" \
  --priority 5 \
  --owner human
```

### 2. Deep Research Intake

Use GPT Deep Research or another literature tool to produce a report with:
- key papers;
- SOTA table;
- benchmark definitions;
- metric definitions;
- code repos;
- reproduction gotchas;
- compute costs;
- likely gaps.

Store it under:

```text
research/<project>/deep_research/
  report.md
  papers.bib
  sota_table.csv
  repo_links.md
```

Agent task:
- convert the report into `project.yaml`;
- identify the minimum benchmark to reproduce first;
- open setup issues for missing dependencies/data.

### 3. Benchmark Build

The benchmark agent should:
- clone official SOTA repos;
- pin commits;
- create Docker/GHCR environment;
- run smoke tests;
- run one official baseline;
- compare to reported numbers;
- decide whether to use upstream code, wrap it, or reimplement.

Rule:
- If upstream code is slow/hard to modify but produces valid numbers, preserve
  it as a reference and build a local fast implementation beside it.
- Never delete the reference baseline until the local implementation matches
  the metric well enough.

### 4. Probe Generation

The research agent turns ideas into probes:
- flag-only probe if possible;
- code-changing probe if necessary;
- one sentinel seed/task first;
- pass/kill rule;
- expected failure mode.

Probe specs are stored in the idea queue and then enqueued into
`central_queue.json`.

### 5. Run and Monitor

The central queue daemon:
- claims workers;
- syncs code;
- launches tasks;
- marks done/failed;
- auto-promotes seeds according to rules.

The dashboard should show:
- project;
- idea;
- probe;
- task;
- current metric;
- gap to SOTA;
- cost;
- next idle worker;
- promising ideas.

### 6. Evidence and Decision

After each result:
- parse metrics;
- attach evidence to idea;
- update project SOTA gap;
- choose one:
  - retire;
  - mutate idea;
  - add one seed;
  - promote to multi-seed;
  - run clean confirmation.

For TD-MPC-Glass, "partial failed CSV" is useful evidence but not a final
success. Every project needs this kind of evidence policy.

### 7. Milestone Blog

Generate a GitHub Pages blog when one of these happens:
- benchmark reproduction works;
- first meaningful result beats baseline;
- SOTA beaten on one seed/task;
- robust result confirmed;
- paper draft starts.

Blog drafts should live under:

```text
research/<project>/blog/
  YYYY-MM-DD-title.md
```

The blog should include:
- what changed;
- metric table;
- key figure;
- failure modes;
- next step;
- links to code/artifacts.

### 8. Manuscript

Generate LaTeX once there is enough evidence:
- benchmark table;
- method section;
- reproducibility checklist;
- ablations;
- limitations;
- appendix.

Draft path:

```text
research/<project>/paper/
  main.tex
  refs.bib
  figures/
  tables/
```

Use template: `research_system/templates/paper_main.tex`.

## Agent Roles

### Literature Agent

Inputs:
- human idea;
- Deep Research report;
- target benchmark.

Outputs:
- SOTA table;
- benchmark plan;
- risk list;
- BibTeX.

Must not:
- invent SOTA numbers without citation;
- blur train/test metrics;
- compare against incompatible budgets.

### Benchmark Agent

Outputs:
- runnable baseline;
- environment;
- reproduction table;
- benchmark wrapper;
- smoke tests.

Pass condition:
- official or reference baseline runs locally and produces credible numbers.

### Probe Agent

Outputs:
- code diffs;
- launcher/config;
- smoke test;
- queued sentinel probes.

Rules:
- one code-changing idea at a time;
- no silent dirty-code batch;
- fresh output tags;
- pass/kill rule before queueing.

### Analysis Agent

Outputs:
- metric tables;
- plots;
- failure diagnosis;
- promotion/retirement recommendation.

Must distinguish:
- complete run;
- interrupted useful partial;
- invalid header-only run;
- duplicated old CSV history.

### Publication Agent

Outputs:
- blog draft;
- paper draft;
- figure scripts;
- release checklist.

Must not:
- overclaim beyond confirmed evidence;
- hide failed seeds;
- omit benchmark setup details.

## Infrastructure

### Master Node

Use EC2 as control plane:
- dashboard;
- idea queue;
- central queue;
- result mirror;
- backups;
- GitHub/HF/W&B/S3 credentials.

Keep the master CPU-only if possible. Workers do the training.

### Workers

Workers can be:
- VastAI GPUs;
- EC2 GPUs;
- local GPUs;
- CPU boxes for analysis/data processing.

Worker contract:
- Docker/GHCR environment or reproducible venv;
- SSH access;
- enough local disk;
- periodic checkpoint/upload;
- `scripts/` and `src/` synced at launch.

### Storage

Use `storageAWS.md` split:
- GitHub: code/docs/configs.
- GHCR/Docker Hub: environment image.
- W&B: scalar metrics and comparisons.
- S3/B2: queue backups, mirrors, frequent checkpoints.
- Hugging Face: best/final checkpoints and public artifacts.

### Crash Resistance

Minimum:
- queue backup before mutation;
- 15-minute control-plane backup;
- stale-running reset tool;
- one queue master rule;
- worker logs stored locally and mirrored;
- fresh output tags.

Recommended:
- systemd services for dashboard/queue/streamer;
- S3 snapshots;
- Docker image pinning;
- fleet registry file;
- automatic unreachable worker quarantine.

## Dashboard Expansion

The dashboard should evolve from TD-MPC-specific panels into generic panels:

- **Projects**: title, target metric, status, SOTA gap.
- **Idea Queue**: ideas, owners, status, linked probes.
- **Run Queue**: central tasks, worker, ETA, logs.
- **Metrics**: project-specific curves and tables.
- **Evidence**: latest result summaries and artifacts.
- **Milestones**: benchmark built, baseline reproduced, SOTA beaten, draft ready.
- **Publishing**: blog/paper generation status.
- **Fleet**: workers, cost, next idle, reliability.
- **Storage**: backup age, S3 sync age, artifact health.

## Data Model

Use simple files first:

```text
research/
  <project>/
    project.yaml
    deep_research/
    benchmark/
    probes/
    analysis/
    blog/
    paper/
scripts/queues/
  idea_queue.json
  central_queue.json
  publication_queue.json
exp/
  <project>/
    metrics/
    logs/
    remote_mirror/
    artifacts/
```

Graduate to SQLite/Postgres only when file locks become limiting.

## Promotion Rules

Generic default:
- `one positive sentinel`: add one more seed/task.
- `beats SOTA once`: add two more seeds/tasks.
- `beats SOTA robustly`: run clean fixed-SHA confirmation.
- `clean confirmation passes`: milestone blog + paper update.
- `clean confirmation fails`: mark as high-variance, design ablation.

Project-specific rules must define:
- primary metric;
- minimum run budget;
- number of seeds/tasks;
- confidence threshold;
- acceptable variance;
- invalid-run criteria.

## Publication Criteria

Blog is allowed when:
- the result is useful and clearly caveated;
- data and code path are reproducible;
- failure modes are included.

Paper draft starts when:
- benchmark is credible;
- method is stable enough to describe;
- at least one core table exists;
- limitations are known.

SOTA claim requires:
- clean code SHA;
- frozen config;
- complete runs;
- fair baseline;
- artifact backup;
- reproducible command;
- no hidden failed seeds.

## Immediate Roadmap

1. Add `project.yaml` for TD-MPC-Glass.
2. Add dashboard panel for `idea_queue.json`.
3. Add publication queue with blog/paper template generation.
4. Add S3 snapshot cron on EC2.
5. Add fleet registry file to replace hard-coded worker lists.
6. Add research-agent runner that claims one idea and produces a plan/probe.
7. Add benchmark-agent runner that clones and validates external repos.
8. Add paper-agent runner that updates LaTeX from evidence tables.

