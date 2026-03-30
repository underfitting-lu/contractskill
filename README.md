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
  <a href="https://arxiv.org/abs/2603.20340">Paper</a> |
  <a href="https://arxiv.org/pdf/2603.20340">PDF</a> |
  <a href="#overview--简介">Overview</a> |
  <a href="#core-idea--核心思想">Core Idea</a> |
  <a href="#highlights--亮点">Highlights</a> |
  <a href="#getting-started--快速开始">Quick Start</a> |
  <a href="#citation--引用">Citation</a>
</p>

---

## Overview | 简介

**English.**  
ContractSkill is a framework for building web-agent skills that are not only executable, but also **self-correcting**, **self-improving**, and **reusable**. Instead of treating a skill as a fragile free-form plan, ContractSkill turns it into a structured artifact with explicit assumptions, executable steps, and verifiable success conditions. This makes skill execution inspectable, repairable, and easier to transfer across related tasks.

**中文。**  
ContractSkill 旨在构建一种不仅能执行，而且具备 **自我纠错**、**自我演化** 与 **技能复用** 能力的网页智能体技能框架。它不再把 skill 视为脆弱的自由文本计划，而是将其表示为带有显式假设、可执行步骤和可验证成功条件的结构化 artifact，从而让技能执行过程更可检查、更可修复，也更容易迁移到相关任务上。

## Core Idea | 核心思想

**English.**  
ContractSkill emphasizes three properties:

1. **Self-correction.** When execution fails, the system localizes the failing step and applies a minimal patch instead of regenerating everything from scratch.
2. **Self-improvement.** Skills can be iteratively refined through verification and repair, so the agent does not remain stuck with its initial draft.
3. **Skill reuse.** Once expressed as contracted artifacts, skills become more stable building blocks that can be reused across similar web tasks.

**中文。**  
ContractSkill 重点强调三个能力：

1. **自我纠错。** 当执行失败时，系统会定位失败步骤，并只修补必要的局部，而不是整段重写。
2. **自我演化。** skill 可以在验证与修复过程中持续改进，而不是停留在第一次生成的草稿状态。
3. **技能复用。** 一旦 skill 被表达为 contracted artifact，它就更像稳定的能力模块，可以复用于相似网页任务。

## Why ContractSkill? | 为什么是 ContractSkill

**English.**  
Many web-agent failures are not pure reasoning failures. They come from stale page assumptions, brittle targets, missing intermediate steps, or locally broken execution plans. In these cases, full replanning is often unnecessary. What is needed is a representation that supports verification, fault localization, and minimal repair.

**中文。**  
很多网页智能体的失败并不只是“推理错了”，而是来自过时的页面假设、脆弱的目标 grounding、缺失的中间步骤，或者局部执行链条断裂。在这些情况下，整段重规划往往并不必要，更关键的是一种支持验证、定位和最小修复的表示形式。

## Highlights | 亮点

- **Contracted skill artifacts.** Skills are encoded with explicit preconditions, executable step contracts, and success conditions.  
  **结构化 skill artifact。** skill 以显式前置条件、步骤约束和成功条件来表达。

- **Verifier-guided self-correction.** Failures are localized to concrete steps and repaired through small patches.  
  **Verifier 引导的自我纠错。** 失败可以被定位到具体步骤，并通过局部 patch 进行修复。

- **Iterative self-improvement.** Skills are not frozen after first generation; they can be improved through repeated verification and repair.  
  **迭代式自我演化。** skill 不会停留在首次生成状态，而是可以在验证与修复中不断改进。

- **Reusable skill abstraction.** Structured skills are easier to inspect, maintain, and reuse across related tasks.  
  **可复用技能抽象。** 结构化 skill 更易检查、维护，并在相关任务之间复用。

- **Cross-benchmark implementation.** This release contains the core ContractSkill implementation for MiniWoB++ and VisualWebArena.  
  **跨 benchmark 实现。** 本仓库提供 MiniWoB++ 与 VisualWebArena 上的核心 ContractSkill 实现。

## Method at a Glance | 方法概览

**English.**

1. Generate a task-level skill draft for the current web task.
2. Convert the draft into a contracted artifact.
3. Execute the artifact under verifier checks.
4. If execution fails, localize the failing step.
5. Repair only the affected part and continue with an improved artifact.

**中文。**

1. 针对当前网页任务生成 task-level skill draft。
2. 将 draft 转换为 contracted artifact。
3. 在 verifier 检查下执行 artifact。
4. 若执行失败，定位失败步骤。
5. 仅修补受影响部分，并以改进后的 artifact 继续执行。

## Repository Scope | 仓库范围

Included in this repository / 本仓库包含：

- core ContractSkill logic / ContractSkill 核心逻辑
- MiniWoB++ and VisualWebArena runners / MiniWoB++ 与 VisualWebArena 运行入口
- environment wrappers and repair utilities / 环境封装与修复工具
- setup notes and example environment files / 环境说明与示例配置文件

Not included in this repository / 本仓库不包含：

- full experimental outputs / 全量实验结果
- benchmark split files used in the paper / 论文中使用的 benchmark split 文件
- raw traces and task-level result bundles / 原始 trace 与 task 级结果包
- internal analysis or development utilities / 内部分析脚本与开发期工具

## Repository Layout | 目录结构

```text
.
|-- env/                       # ContractSkill logic and benchmark-specific environments
|-- docs/                      # Setup notes for MiniWoB++ and VisualWebArena
|-- scripts/                   # Minimal setup and environment-check utilities
|-- run_miniwob_experiment.py  # MiniWoB++ runner
|-- run_vwa_experiment.py      # VisualWebArena runner
`-- glm_client.py              # Model client wrapper
```

## Getting Started | 快速开始

**English.**  
For environment setup, please start from:

- `docs/MINIWOB_SETUP.md`
- `docs/VISUALWEBARENA_SETUP.md`
- `docs/VISUALWEBARENA_AMI_AWS.md`

Example environment files:

- `.env.api.example`
- `.env.miniwob.example`
- `.env.vwa.example`
- `.env.vwa.ami.example`

**中文。**  
环境配置建议从以下文档开始：

- `docs/MINIWOB_SETUP.md`
- `docs/VISUALWEBARENA_SETUP.md`
- `docs/VISUALWEBARENA_AMI_AWS.md`

示例环境变量文件包括：

- `.env.api.example`
- `.env.miniwob.example`
- `.env.vwa.example`
- `.env.vwa.ami.example`

## Paper | 论文链接

- arXiv page: https://arxiv.org/abs/2603.20340
- arXiv PDF: https://arxiv.org/pdf/2603.20340
- DOI: https://doi.org/10.48550/arXiv.2603.20340

## Citation | 引用

If this repository or the paper is useful in your research, please cite:

如果这个仓库或论文对你的研究有帮助，欢迎引用：

```bibtex
@article{lu2026contractskill,
  title   = {ContractSkill: Repairable Contract-Based Skills for Multimodal Web Agents},
  author  = {Lu, Zijian and Zuo, Yiping and Nie, Yupeng and He, Xin and Fan, Weibei and Dai, Chen},
  journal = {arXiv preprint arXiv:2603.20340},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.20340}
}
```
