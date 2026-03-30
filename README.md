# ContractSkill

Minimal code release for the ContractSkill pipeline.

This repository contains the core implementation for:

- task-level artifact generation
- verifier-guided repair
- environment integration for MiniWoB, VisualWebArena, and WorkArena

This release is intentionally code-only. It does **not** include:

- full experimental outputs
- task-level result artifacts
- internal analysis scripts
- benchmark split files used in our paper

Those materials are omitted from this release because follow-up analysis is still ongoing.

## Included Contents

- `env/`: core environment wrappers and ContractSkill logic
- `run_miniwob_experiment.py`: MiniWoB experiment entry point
- `run_vwa_experiment.py`: VisualWebArena experiment entry point
- `run_workarena_experiment.py`: WorkArena experiment entry point
- `scripts/`: minimal environment setup and environment checks
- `docs/`: setup notes for supported benchmarks
- `.env.*.example`: example environment configuration files

## What This Release Is For

Use this release to inspect the method implementation and understand how ContractSkill is executed inside supported web environments.

This release is **not** a full reproduction package for the paper's reported tables. In particular, benchmark task definitions and raw result bundles are not included here.

## Setup

See:

- `docs/MINIWOB_SETUP.md`
- `docs/VISUALWEBARENA_SETUP.md`
- `docs/VISUALWEBARENA_AMI_AWS.md`
- `docs/WORKARENA_SETUP.md`

Environment variable templates are provided as:

- `.env.api.example`
- `.env.miniwob.example`
- `.env.vwa.example`
- `.env.vwa.ami.example`
- `.env.workarena.example`

## Notes

- Before publishing, review all example environment files and remove any provider-specific details you do not want to expose.
- If you later decide to release task splits or summary results, they can be added as a separate supplementary package without changing the core code layout.
