# MiniWoB++ Setup

This repository keeps MiniWoB++ isolated from the VisualWebArena stack.

MiniWoB++ uses:

- `.venv_miniwob`
- `.env.miniwob`
- an external frozen checkout of `miniwob-plusplus`

It does not require Docker.

## Verified Baseline

Verified in WSL on March 11, 2026:

- Python 3.12.3 can install and import `browsergym.miniwob`
- `browsergym-miniwob` 0.14.3 works in a separate virtualenv
- the frozen `miniwob-plusplus` commit `7fd85d71a4b60325c6585396ec4f48377d049838` can reset `browsergym/miniwob.click-test`
- Playwright Chromium can be reused from the local WSL browser cache

## 1. Create The Environment

Run:

```bash
bash scripts/setup_miniwob_env.sh
```

The setup script:

- prefers `python3.12`
- falls back to `python3.11` only if `python3.12` is unavailable or fails validation
- creates `.venv_miniwob`
- installs `openai`, `gymnasium`, `browsergym-miniwob`, and `playwright`
- downloads Chromium for Playwright
- clones `Farama-Foundation/miniwob-plusplus` outside the repo
- pins the external MiniWoB++ checkout to the frozen commit used by BrowserGym docs
- writes `.env.miniwob` if it does not already exist

Default external checkout path:

```text
/mnt/c/home/lzj/benchmark_assets/miniwob-plusplus
```

Override it if needed:

```bash
MINIWOB_REPO_PATH=/mnt/d/benchmark_assets/miniwob-plusplus bash scripts/setup_miniwob_env.sh
```

## 2. Environment Variables

The tracked example file is:

[.env.miniwob.example](/mnt/c/home/lzj/contractskill/.env.miniwob.example)

The local runtime file is:

```text
.env.miniwob
```

Default contents:

```bash
MINIWOB_URL=file:///mnt/c/home/lzj/benchmark_assets/miniwob-plusplus/miniwob/html/miniwob/
MINIWOB_DEFAULT_TASK=browsergym/miniwob.click-test
```

You can load it with:

```bash
source scripts/load_miniwob_env.sh
```

## 3. Run The Environment Check

Activate the environment or invoke it directly:

```bash
source .venv_miniwob/bin/activate
python scripts/check_miniwob_env.py
```

The check script reports:

- Python version and virtualenv status
- package import status for `browsergym-miniwob`, `browsergym-core`, `gymnasium`, `openai`, and `playwright`
- Playwright Chromium download and headless launch status
- whether `MINIWOB_URL` points to a real MiniWoB++ task directory
- whether the external `miniwob-plusplus` checkout exists and matches the frozen commit
- whether the configured smoke task can reset successfully

If `.env.miniwob` exists, the check script loads it automatically. Shell-exported values still take precedence.

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

## 5. Smoke Reset Command

The fastest proof that MiniWoB++ is wired correctly is:

```bash
source scripts/load_miniwob_env.sh
.venv_miniwob/bin/python scripts/check_miniwob_env.py
```

That check runs a real `gym.make(...).reset()` on the configured smoke task.

## 6. Experiment Runner

MiniWoB now has its own experiment entry point:

- [run_miniwob_experiment.py](/mnt/c/home/lzj/contractskill/run_miniwob_experiment.py)
- outputs under `outputs/miniwob_experiments/`
- fixed smoke split: [miniwob_smoke_5.json](/mnt/c/home/lzj/contractskill/tasks/miniwob_smoke_5.json)

It stays separate from the VWA runner and supports the same baseline names:

- `no_skill`
- `skill_no_repair`
- `text_only_rewrite`
- `contractskill`

For paper/default MiniWoB runs, use only these 3 baselines:

- `no_skill`
- `skill_no_repair`
- `contractskill`

Smoke command:

```bash
cd /mnt/c/home/lzj/contractskill
source scripts/load_miniwob_env.sh
source .venv_miniwob/bin/activate
python run_miniwob_experiment.py --baseline no_skill --split-path tasks/miniwob_smoke_5.json
```

`run_miniwob_experiment.py` auto-loads `.env.miniwob` if it exists.

Model-backed runs still require `ZAI_API_KEY`.

## 7. Paper Batch Defaults

The current default MiniWoB paper batch is:

- `4` categories: `M1 / M2 / M3 / M4`
- `5` tasks per category
- `20` tasks total
- `10` instances per task
- `3` repeats per instance
- `3` default baselines: `no_skill / skill_no_repair / contractskill`

That means:

- `600` episodes per baseline
- `1800` episodes for the full default MiniWoB batch

The split builder and handbook batch launcher now default to this configuration:

```bash
python scripts/build_miniwob_splits.py
python scripts/run_miniwob_handbook_batch.py
```

`text_only_rewrite` is still available as an optional baseline, but it is no longer part of the default paper batch.

## 8. Separation From VWA

This setup is intentionally separate from the VisualWebArena stack:

- MiniWoB++ uses `.venv_miniwob`
- VWA uses `.venv_vwa`
- MiniWoB++ uses `.env.miniwob`
- VWA uses `.env.vwa`
- MiniWoB++ does not depend on Docker or VWA site variables

You can keep both environments on the same machine without mixing them.
