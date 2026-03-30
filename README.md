<h1 align="center">ContractSkill</h1>

<p align="center">
  <b>Repairable Contract-Based Skills for Multimodal Web Agents</b>
</p>

<p align="center">
  <i>Zijian Lu, Yiping Zuo, Yupeng Nie, Xin He, Weibei Fan, Chen Dai</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.20340"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.20340-b31b1b.svg"></a>
  <a href="https://doi.org/10.48550/arXiv.2603.20340"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.20340-blue.svg"></a>
</p>

<p align="center">
  Minimal code release accompanying our paper:
  <a href="https://arxiv.org/abs/2603.20340">arXiv:2603.20340</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.20340">Paper</a>
  ·
  <a href="#overview">Overview</a>
  ·
  <a href="#repository-scope">Repository Scope</a>
  ·
  <a href="#getting-started">Getting Started</a>
  ·
  <a href="#citation">Citation</a>
</p>

---

## Overview

Modern multimodal web agents are often strong enough to draft task-level plans, but those plans are brittle when executed in real interfaces. ContractSkill turns free-form web skills into **contracted artifacts** with explicit assumptions, executable steps, and verifiable completion conditions. When execution fails, ContractSkill applies **verifier-guided local repair** instead of rewriting the entire plan from scratch.

At a high level, the method has three stages:

1. Draft a task-level skill from the current web task.
2. Convert the draft into a contracted artifact with preconditions, step-level actions, and success constraints.
3. Use verifier feedback to localize failures and repair the artifact with minimal patches.

This repository contains the core implementation of that pipeline for:

- MiniWoB++
- VisualWebArena

## Highlights

- ContractSkill converts draft skills into executable contracted artifacts with explicit preconditions, step specifications, postconditions, recovery rules, and termination checks.
- The method enables deterministic verification, step-level fault localization, and minimal patch-based repair.
- In the paper, ContractSkill improves self-generated skills on both VisualWebArena and MiniWoB across multiple multimodal models.
- The method treats web-agent skills as explicit procedural artifacts that can be verified, repaired, and reused.

## Why ContractSkill?

ContractSkill is designed around a simple observation: web-agent failures are often not pure reasoning failures. They are frequently caused by:

- missing or stale page assumptions
- partially correct but incomplete task plans
- brittle action targets
- local execution failures that do not require full replanning

By making the artifact structure explicit, ContractSkill makes these failures easier to **localize, diagnose, and patch**.

## Method Components

The released code includes the main pieces of the ContractSkill pipeline:

- **Artifact generation**: produce task-level skills from multimodal observations
- **Contracted representation**: attach preconditions, step-level actions, and success contracts
- **Verification**: detect schema violations, execution failures, and contract failures
- **Repair**: update only the failing part of an artifact through constrained patching
- **Environment integration**: run the same high-level method across supported web-agent benchmarks

## Paper Summary

From the paper abstract:

> ContractSkill converts a draft skill into a contracted executable artifact with explicit preconditions, step specifications, postconditions, recovery rules, and termination checks. This representation enables deterministic verification, step-level fault localization, and minimal patch-based repair, turning skill refinement into localized editing rather than full regeneration.

If you are interested in multimodal GUI agents, reusable skills, or verifier-guided agent repair, the paper is the best place to start:

- Paper page: https://arxiv.org/abs/2603.20340
- PDF: https://arxiv.org/pdf/2603.20340
- DOI: https://doi.org/10.48550/arXiv.2603.20340

## Repository Scope

This is a **minimal code release** focused on method transparency and implementation clarity.

Included in this repository:

- core ContractSkill logic
- benchmark environment wrappers
- main experiment entry points
- setup notes for supported environments
- minimal environment-check and setup scripts

Not included in this repository:

- full experimental outputs
- benchmark task splits used in our paper
- raw traces and task-level result bundles
- internal analysis scripts and development utilities

These materials are omitted in this release because follow-up analysis is still ongoing.

## Repository Layout

```text
.
├── env/                       # ContractSkill logic and benchmark-specific environments
├── docs/                      # Setup notes for MiniWoB++ and VWA
├── scripts/                   # Minimal setup and environment-check utilities
├── run_miniwob_experiment.py  # MiniWoB++ runner
├── run_vwa_experiment.py      # VisualWebArena runner
└── glm_client.py              # Model client wrapper
```

## Getting Started

The fastest entry point is to read the benchmark setup guides:

- `docs/MINIWOB_SETUP.md`
- `docs/VISUALWEBARENA_SETUP.md`
- `docs/VISUALWEBARENA_AMI_AWS.md`

Example environment templates are provided as:

- `.env.api.example`
- `.env.miniwob.example`
- `.env.vwa.example`
- `.env.vwa.ami.example`

## Release Note

This repository is intended as a clean supplementary code release for the paper. It is organized around the **final method implementation**, not around development history, temporary experiment variants, or internal benchmarking artifacts.

## Citation

If this repository or the paper is useful in your work, please cite:

```bibtex
@article{lu2026contractskill,
  title   = {ContractSkill: Repairable Contract-Based Skills for Multimodal Web Agents},
  author  = {Lu, Zijian and Zuo, Yiping and Nie, Yupeng and He, Xin and Fan, Weibei and Dai, Chen},
  journal = {arXiv preprint arXiv:2603.20340},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.20340}
}
```

## Paper Link

- arXiv: https://arxiv.org/abs/2603.20340
