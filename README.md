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
  <a href="https://arxiv.org/abs/2603.20340">Paper</a>
  ·
  <a href="https://arxiv.org/pdf/2603.20340">PDF</a>
  ·
  <a href="#overview--简介">Overview</a>
  ·
  <a href="#repository-scope--仓库范围">Scope</a>
  ·
  <a href="#getting-started--快速开始">Quick Start</a>
  ·
  <a href="#citation--引用">Citation</a>
</p>

---

## Overview | 简介

**English.**  
ContractSkill studies how to make multimodal web-agent skills more reliable. Instead of treating a task-level skill as a free-form plan that must be regenerated after failure, ContractSkill converts it into a **contracted artifact** with explicit preconditions, step specifications, and success conditions. This structured representation enables **deterministic verification**, **step-level fault localization**, and **minimal patch-based repair**.

**中文。**  
ContractSkill 关注如何让多模态网页智能体的技能更加可靠。与其把任务级技能看成失败后只能整体重写的自由文本计划，ContractSkill 会将其转化为带有**前置条件、步骤规范和成功约束**的结构化 artifact。这样的表示可以支持**确定性验证**、**步骤级故障定位**以及**最小化 patch 修复**。

## Why ContractSkill? | 为什么需要 ContractSkill

**English.**  
Many web-agent failures are not pure reasoning failures. They often come from stale page assumptions, brittle targets, partially correct but incomplete plans, or local execution errors that do not justify full replanning. ContractSkill makes these failure modes explicit and repairable.

**中文。**  
很多网页智能体失败并不完全是“推理错误”，而是来自过时的页面假设、脆弱的目标 grounding、部分正确但不完整的计划，或者只需要局部修补的执行错误。ContractSkill 的目标，就是把这些失败形式显式化、可定位、可修复。

## Highlights | 亮点

- **Contracted skill artifacts.** Skills are represented with explicit assumptions, executable steps, and verifiable success conditions.  
  **结构化技能 artifact。** 技能不再是松散文本，而是具有显式假设、可执行步骤和可验证完成条件的表示。

- **Verifier-guided repair.** Failures can be localized to concrete steps and repaired with small patches instead of full regeneration.  
  **Verifier 引导的修复。** 失败可以定位到具体步骤，并通过局部 patch 修复，而不是整段重写。

- **Cross-benchmark implementation.** This release contains the core ContractSkill implementation for MiniWoB++ and VisualWebArena.  
  **跨 benchmark 实现。** 本仓库提供 MiniWoB++ 和 VisualWebArena 上的核心 ContractSkill 实现。

## Method at a Glance | 方法概览

**English.**

1. Generate a task-level skill draft from the current web task.
2. Convert the draft into a contracted artifact.
3. Execute the artifact under verifier checks.
4. Localize the failure and repair only the affected part when execution fails.

**中文。**

1. 从当前网页任务生成任务级 skill draft。
2. 将 draft 转换为 contracted artifact。
3. 在 verifier 检查下执行 artifact。
4. 如果失败，只定位并修补受影响的局部步骤。

## Repository Scope | 仓库范围

Included in this repository / 本仓库包含：

- core ContractSkill logic / ContractSkill 核心逻辑
- MiniWoB++ and VisualWebArena runners / MiniWoB++ 与 VisualWebArena 运行入口
- environment wrappers and repair utilities / 环境封装与修复工具
- setup notes and example environment files / 环境说明与示例配置文件

Not included in this repository / 本仓库不包含：

- full experimental outputs / 全量实验结果
- benchmark split files used in the paper / 论文中的 benchmark split 文件
- raw traces and task-level result bundles / 原始 trace 与 task 级结果包
- internal analysis or development utilities / 内部分析脚本与开发期工具

## Repository Layout | 目录结构

```text
.
├── env/                       # ContractSkill logic and benchmark-specific environments
├── docs/                      # Setup notes for MiniWoB++ and VisualWebArena
├── scripts/                   # Minimal setup and environment-check utilities
├── run_miniwob_experiment.py  # MiniWoB++ runner
├── run_vwa_experiment.py      # VisualWebArena runner
└── glm_client.py              # Model client wrapper
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
