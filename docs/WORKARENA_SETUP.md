# WorkArena Setup

This repository keeps WorkArena isolated from the VisualWebArena and MiniWoB stacks.

WorkArena uses:

- `.venv_workarena`
- `.env.workarena`
- BrowserGym's `browsergym-workarena` package
- Playwright Chromium
- gated ServiceNow instances from the official WorkArena release, or explicit SNOW credentials

It does not require Docker or self-hosted websites.

## Scope In This Repo

This repo uses a WorkArena-native action subset instead of the old four-action fallback.

The WorkArena line supports:

- `CLICK`
- `DOUBLE_CLICK`
- `TYPE`
- `SELECT`
- `PRESS`
- `HOVER`
- `FOCUS`
- `CLEAR`
- `DRAG`
- `SCROLL`
- `STOP`

That means:

- the WorkArena runner can express more of the official UI interaction surface
- the VWA and MiniWoB lines stay unchanged
- WorkArena still remains a separate benchmark line with its own env, runner, and outputs

## Official Prerequisites

The current official setup still depends on external instance access:

1. Request access to the gated dataset:
   `https://huggingface.co/datasets/ServiceNow/WorkArena-Instances`
2. Authenticate on the machine:
   `hf auth login`
3. Install `browsergym-workarena`
4. Install Playwright Chromium

The relevant official references are:

- `https://github.com/ServiceNow/WorkArena`
- `https://github.com/ServiceNow/BrowserGym`
- `https://pypi.org/project/browsergym-workarena/`

## 1. Create The Environment

Run:

```bash
bash scripts/setup_workarena_env.sh
```

The setup script:

- prefers `python3.12`
- falls back to `python3.11` only if `python3.12` is unavailable or fails validation
- creates `.venv_workarena`
- installs `browsergym-workarena`, `openai`, `gymnasium`, `playwright`, and `huggingface_hub`
- downloads Chromium for Playwright
- writes `.env.workarena` if it does not already exist
- runs the local WorkArena environment check

## 2. Environment Variables

The tracked example file is:

[.env.workarena.example](/mnt/c/home/lzj/contractskill/.env.workarena.example)

The local runtime file is:

```text
.env.workarena
```

Default contents:

```bash
WORKARENA_DEFAULT_TASK=browsergym/workarena.servicenow.knowledge-base-search
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZAI_API_KEY=
HUGGING_FACE_HUB_TOKEN=
SNOW_INSTANCE_POOL=
SNOW_INSTANCE_URL=
SNOW_INSTANCE_UNAME=
SNOW_INSTANCE_PWD=
```

You can load it with:

```bash
source scripts/load_workarena_env.sh
```

There are two supported access modes:

1. Hugging Face gated access
   - fill `HUGGING_FACE_HUB_TOKEN`, or run `hf auth login`
   - let BrowserGym fetch the official ServiceNow instance pool
2. Direct ServiceNow instance configuration
   - fill `SNOW_INSTANCE_URL`
   - fill `SNOW_INSTANCE_UNAME`
   - fill `SNOW_INSTANCE_PWD`
   - optionally use `SNOW_INSTANCE_POOL` if you already have a local pool file

## 3. Run The Environment Check

Activate the environment or invoke it directly:

```bash
source .venv_workarena/bin/activate
python scripts/check_workarena_env.py
```

The check script reports:

- Python version and virtualenv status
- package import status for `browsergym-workarena`, `browsergym-core`, `gymnasium`, `openai`, `playwright`, and `huggingface_hub`
- Playwright Chromium download and headless launch status
- whether WorkArena L1 environments are registered
- whether `WORKARENA_DEFAULT_TASK` is valid
- whether Hugging Face authentication is available
- whether direct ServiceNow instance configuration is present
- whether a real smoke `gym.make(...).reset()` succeeds

If `.env.workarena` exists, the check script loads it automatically. Shell-exported values still take precedence.

Without Hugging Face auth or direct SNOW credentials, the final smoke reset will stay `BLOCKED`. That is expected.

## 4. Playwright Host Dependencies

If Chromium fails to launch because Linux runtime libraries are missing, install them directly:

```bash
sudo apt-get install -y libnss3 libnspr4 libasound2t64
```

If you are on Ubuntu 22.04 or another non-`t64` release, the audio package may still be named `libasound2`.

You can also use the Playwright helper:

```bash
sudo playwright install-deps
```

## 5. Fixed Splits

This WorkArena line has three tracked splits:

- smoke: [workarena_smoke_5.json](/mnt/c/home/lzj/contractskill/tasks/workarena_smoke_5.json)
- dev: [workarena_dev_20.json](/mnt/c/home/lzj/contractskill/tasks/workarena_dev_20.json)
- main: [workarena_main_50.json](/mnt/c/home/lzj/contractskill/tasks/workarena_main_50.json)

They are all based on WorkArena L1 environment ids and use the repo's WorkArena-native action interface.

## 6. Experiment Runner

WorkArena now has its own experiment entry point:

- [run_workarena_experiment.py](/mnt/c/home/lzj/contractskill/run_workarena_experiment.py)
- outputs under `outputs/workarena_experiments/`

Supported baselines remain the same:

- `no_skill`
- `skill_no_repair`
- `text_only_rewrite`
- `contractskill`

Smoke command:

```bash
cd /mnt/c/home/lzj/contractskill
source scripts/load_workarena_env.sh
source .venv_workarena/bin/activate
python run_workarena_experiment.py --baseline no_skill --split-path tasks/workarena_smoke_5.json
```

Model-backed runs still require `ZAI_API_KEY`.

## 7. Suggested Progression

Use this order:

1. `python scripts/check_workarena_env.py`
2. `workarena_smoke_5.json`
3. `workarena_dev_20.json`
4. `workarena_main_50.json`

The practical gate is simple:

- if `smoke_5` cannot reset, fix auth or SNOW access first
- if `dev_20` shows no repair activity, do not spend time on `main_50` yet

## 8. Separation From Other Benchmarks

This setup is intentionally separate from the other benchmark lines:

- MiniWoB uses `.venv_miniwob` and `.env.miniwob`
- VWA uses `.venv_vwa` and `.env.vwa`
- WorkArena uses `.venv_workarena` and `.env.workarena`

You can keep all three on the same machine without mixing package environments.
