#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv_vwa}"

log() {
  printf '[setup_vwa_env] %s\n' "$*"
}

fail() {
  printf '[setup_vwa_env] ERROR: %s\n' "$*" >&2
  exit 1
}

bootstrap_virtualenv() {
  local python_bin="$1"

  log "Bootstrapping pip and virtualenv for $python_bin because stdlib venv is unavailable."
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  "$python_bin" /tmp/get-pip.py --user --break-system-packages
  "$python_bin" -m pip install --user --break-system-packages virtualenv
}

create_virtualenv() {
  local python_bin="$1"
  local expected_version="$2"

  if [ -x "$VENV_PATH/bin/python" ]; then
    local current_version
    current_version="$("$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$current_version" != "$expected_version" ]; then
      log "Removing existing $VENV_PATH because it uses Python $current_version."
      rm -rf "$VENV_PATH"
    fi
  fi

  if [ ! -x "$VENV_PATH/bin/python" ]; then
    if "$python_bin" -m venv "$VENV_PATH" >/dev/null 2>&1; then
      :
    else
      bootstrap_virtualenv "$python_bin"
      "$python_bin" -m virtualenv "$VENV_PATH"
    fi
  fi
}

install_packages() {
  log "Upgrading pip in $VENV_PATH."
  "$VENV_PATH/bin/python" -m pip install --upgrade pip

  log "Installing openai and gymnasium."
  "$VENV_PATH/bin/pip" install openai gymnasium

  log "Installing CPU-only torch to avoid the default CUDA package tree in WSL."
  "$VENV_PATH/bin/pip" install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch

  log "Installing browsergym-visualwebarena and its pinned Playwright dependency."
  "$VENV_PATH/bin/pip" install browsergym-visualwebarena
}

install_runtime_assets() {
  log "Downloading Chromium for Playwright."
  "$VENV_PATH/bin/playwright" install chromium

  log "Downloading NLTK punkt_tab."
  "$VENV_PATH/bin/python" -c "import nltk; nltk.download('punkt_tab')"
}

validate_browsergym() {
  "$VENV_PATH/bin/python" -c 'import browsergym.visualwebarena, platform; print(platform.python_version())'
}

try_python() {
  local python_cmd="$1"
  local python_bin
  local python_version

  python_bin="$(command -v "$python_cmd")"
  python_version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

  log "Trying $python_bin (Python $python_version)."
  create_virtualenv "$python_bin" "$python_version"
  install_packages
  install_runtime_assets
  validate_browsergym

  SELECTED_PYTHON_BIN="$python_bin"
  SELECTED_PYTHON_VERSION="$python_version"
}

SELECTED_PYTHON_BIN=""
SELECTED_PYTHON_VERSION=""

candidates=()
for candidate in python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1; then
    candidates+=("$candidate")
  fi
done

if [ "${#candidates[@]}" -eq 0 ]; then
  fail "Neither python3.12 nor python3.11 was found. Install one of them and rerun this script."
fi

for candidate in "${candidates[@]}"; do
  if try_python "$candidate"; then
    break
  fi

  log "Validation failed for $candidate. Removing $VENV_PATH before trying the next interpreter."
  rm -rf "$VENV_PATH"
done

if [ -z "$SELECTED_PYTHON_VERSION" ]; then
  fail "Unable to prepare a working VisualWebArena environment with Python 3.12 or Python 3.11."
fi

log "Running post-install environment check."
if ! "$VENV_PATH/bin/python" scripts/check_vwa_env.py; then
  cat <<'EOF'
[setup_vwa_env] Post-install check reported missing runtime prerequisites.
This is expected if Docker is not visible in WSL, Playwright host libraries are missing,
or the required VWA_* environment variables are not exported yet.
EOF
fi

cat <<EOF
VisualWebArena environment bootstrap completed.

Selected Python: $SELECTED_PYTHON_VERSION
Selected interpreter: $SELECTED_PYTHON_BIN
Virtualenv: $VENV_PATH

Notes:
- Python 3.12 is preferred for this repo. Python 3.11 is the fallback path only if 3.12 is unavailable or fails validation.
- CPU-only torch is installed on purpose to avoid downloading CUDA wheels inside WSL.
- For benchmark websites, prefer the official cloud AMI route instead of downloading 200GB+ of site assets into WSL.
- If the Browser section of scripts/check_vwa_env.py fails, run:
    sudo playwright install-deps
  or install:
    sudo apt-get install -y libnss3 libnspr4 libasound2
- If the Docker section fails, enable Docker Desktop WSL integration for this Ubuntu distro.

Next:
1. Activate the environment or call it directly:
   source .venv_vwa/bin/activate
   python scripts/check_vwa_env.py
2. Fill `.env.vwa` with either:
   - your cloud EC2 hostname and the official VWA ports (`9980`, `7770`, `9999`, `8888`, `4399`), or
   - your own locally deployed VWA site URLs.
3. Export OPENAI_API_KEY if you plan to use the official evaluator.
EOF
