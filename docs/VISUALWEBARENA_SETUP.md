# VisualWebArena Setup

This repository keeps VisualWebArena isolated in a dedicated virtual environment named `.venv_vwa`.

## Verified Baseline

Verified in WSL on March 10, 2026:

- Python 3.12.3 can install and import `browsergym.visualwebarena`.
- Python 3.12 is the preferred interpreter for this repo.
- Python 3.11 remains the fallback path if Python 3.12 is unavailable or fails validation.
- `browsergym-visualwebarena` pulls `torch`; use the CPU wheel in WSL to avoid the default CUDA download path.
- `playwright install chromium` and headless Chromium launch both work after installing the required Linux runtime libraries.
- Docker CLI and Docker daemon both work inside Ubuntu 24.04 under systemd after installing native Ubuntu Docker packages.

## 1. Create The Environment

Run:

```bash
bash scripts/setup_vwa_env.sh
```

The setup script:

- prefers `python3.12`
- falls back to `python3.11` only if `python3.12` is unavailable or fails validation
- creates `.venv_vwa`
- installs `openai`, `gymnasium`, CPU-only `torch`, and `browsergym-visualwebarena`
- installs BrowserGym's compatible Playwright version
- downloads Chromium and NLTK `punkt_tab`

If stdlib `venv` is unavailable in Ubuntu, the script bootstraps `virtualenv` automatically.

## 2. Required packages

After setup, the environment should contain:

- `browsergym-visualwebarena`
- `browsergym-core`
- `gymnasium`
- `openai`
- `torch` (CPU-only)
- `playwright`

It also downloads:

- Playwright Chromium
- NLTK `punkt_tab`

## 3. Run The Environment Check

Activate the environment or invoke it directly:

```bash
source .venv_vwa/bin/activate
python scripts/check_vwa_env.py
```

The check script reports:

- Python version, executable path, and whether you are inside a virtualenv
- package import and version status for BrowserGym, Gymnasium, OpenAI, Playwright, Torch, Pillow, and NLTK
- Playwright Chromium download status and headless launch status
- Docker CLI visibility and Docker daemon reachability
- required and optional `VWA_*` environment variables
- HTTP reachability of the configured VWA site URLs when the required values are present

If `.env.vwa` exists in the repo root, the check script loads it automatically for the purpose of the check. Shell-exported variables still take precedence.

It exits with status code `0` only when all required prerequisites are ready and the configured VWA URLs respond over HTTP.

## 4. Fix Playwright Host Dependencies If Needed

If the Browser section reports missing Linux runtime libraries, install them directly:

```bash
sudo apt-get install -y libnss3 libnspr4 libasound2t64
```

If you are on Ubuntu 22.04 or another non-`t64` release, the audio package may still be named `libasound2`.

If you prefer the Playwright helper, you can still run:

```bash
sudo playwright install-deps
```

## 5. Enable Docker Inside WSL

VisualWebArena benchmark websites run through Docker. There are two workable setups:

Option A: Docker Desktop WSL integration on Windows.

1. Open Docker Desktop.
2. Go to `Settings -> Resources -> WSL Integration`.
3. Enable integration for your Ubuntu distro.
4. Click `Apply & Restart`.
5. Restart the distro if needed:

```powershell
wsl --shutdown
```

Then verify inside Ubuntu:

```bash
docker --version
docker ps
```

If `docker ps` still fails, restart Docker Desktop and verify the same distro is enabled under WSL integration.

Option B: native Ubuntu Docker packages inside WSL.

```bash
sudo apt-get install -y docker.io docker-compose-v2
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker.socket docker.service
```

Then restart the shell or the distro so the new `docker` group membership takes effect.

## 6. VisualWebArena Site Deployment

This repository does not bundle the benchmark websites.
Deploy the official VisualWebArena websites separately through Docker / Docker Desktop WSL integration.

Recommended path: use the official AWS AMI and point this repo at the remote host.

The official VisualWebArena Docker docs currently list:

- region: `us-east-2`
- AMI name: `webarena-x`
- AMI ID: `ami-080f6d73cfce497a1`
- recommended instance type: `t3a.xlarge`
- recommended root volume: `1000GB` EBS

If you want the fastest path and do not want to download the full benchmark assets locally, use the official AWS AMI path documented in [VISUALWEBARENA_AMI_AWS.md](/mnt/c/home/lzj/contractskill/docs/VISUALWEBARENA_AMI_AWS.md).

After the instance is up, fill `.env.vwa` in this repo with the EC2 hostname and the official cloud ports:

```bash
VWA_CLASSIFIEDS=http://<ec2-host>:9980
VWA_CLASSIFIEDS_RESET_TOKEN=<token>
VWA_SHOPPING=http://<ec2-host>:7770
VWA_REDDIT=http://<ec2-host>:9999
VWA_WIKIPEDIA=http://<ec2-host>:8888
VWA_HOMEPAGE=http://<ec2-host>:4399
```

Then verify from this repo:

```bash
source .venv_vwa/bin/activate
python scripts/check_vwa_env.py
```

The `Sites` section should show whether the remote EC2 endpoints are actually reachable from your machine.

The official assets are large. As of March 11, 2026, the upstream downloads referenced by the BrowserGym / VisualWebArena docs are approximately:

- shopping image tar: 67.6 GB
- reddit image tar: 53.4 GB
- wikipedia `.zim`: 95.2 GB

Because of that size, do not stage these downloads under `/mnt/c` unless you know you have enough Windows disk space. Prefer the native WSL filesystem under `/`.

## 7. Required Environment Variables

Export the variables directly in your shell:

```bash
export VWA_CLASSIFIEDS="http://<host>:8083"
export VWA_CLASSIFIEDS_RESET_TOKEN="<token>"
export VWA_SHOPPING="http://<host>:8082"
export VWA_REDDIT="http://<host>:8080"
export VWA_WIKIPEDIA="http://<host>:8081"
export VWA_HOMEPAGE="http://<host>:80"
```

`scripts/check_vwa_env.py` treats obvious placeholders such as `replace_me`, `<host>`, and `<token>` as not-ready values.

For the official AWS AMI route, replace those ports with the cloud ports listed above (`9980`, `7770`, `9999`, `8888`, `4399`).

Optional:

```bash
export VWA_FULL_RESET="http://<host>:7565"
export OPENAI_API_KEY="<official-openai-key>"
```

`OPENAI_API_KEY` is not used by the GLM agent itself, but the official VisualWebArena evaluator may still require it for some fuzzy-match checks.

## 8. Optional No-Skill Baseline

Run:

```bash
.venv_vwa/bin/python run_vwa_noskill.py
```

Optional arguments:

```bash
.venv_vwa/bin/python run_vwa_noskill.py \
  --subset-path tasks/vwa_subset_small.json \
  --max-steps 8 \
  --headless true \
  --model glm-4.6v
```

Outputs are isolated under:

- `outputs/vwa/screenshots/`
- `outputs/vwa/traces/`
- `outputs/vwa/results/`

## 9. ContractSkill Experiment Runner

Once `check_vwa_env.py` is green, you can run the multi-baseline experiment entrypoint:

```bash
.venv_vwa/bin/python run_vwa_experiment.py --baseline no_skill --split-path tasks/vwa_smoke_2.json
```

Supported baselines:

- `no_skill`
- `skill_no_repair`
- `text_only_rewrite`
- `contractskill`

Generated benchmark manifests and fixed splits live under `tasks/`:

- `vwa_manifest.json`
- `vwa_smoke_2.json`
- `vwa_dev_20.json`
- `vwa_main_100.json`
- `vwa_stability_20.json`
- `vwa_transfer_30.json`
