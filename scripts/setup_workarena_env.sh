#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${WORKARENA_VENV_PATH:-$REPO_ROOT/.venv_workarena}"
LOCAL_ENV_FILE="${WORKARENA_ENV_FILE:-$REPO_ROOT/.env.workarena}"
DEFAULT_TASK="${WORKARENA_DEFAULT_TASK:-browsergym/workarena.servicenow.knowledge-base-search}"

log() {
  printf '[setup_workarena_env] %s\n' "$*"
}

fail() {
  printf '[setup_workarena_env] ERROR: %s\n' "$*" >&2
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

  log "Installing WorkArena benchmark dependencies."
  "$VENV_PATH/bin/pip" install browsergym-workarena openai gymnasium playwright huggingface_hub
}

install_runtime_assets() {
  log "Downloading Chromium for Playwright."
  "$VENV_PATH/bin/python" -m playwright install chromium
}

write_local_env_file() {
  if [ -f "$LOCAL_ENV_FILE" ]; then
    log "$LOCAL_ENV_FILE already exists. Leaving it unchanged."
    return
  fi

  cat >"$LOCAL_ENV_FILE" <<EOF
WORKARENA_DEFAULT_TASK=$DEFAULT_TASK
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZAI_API_KEY=
HUGGING_FACE_HUB_TOKEN=
SNOW_INSTANCE_POOL=
SNOW_INSTANCE_URL=
SNOW_INSTANCE_UNAME=
SNOW_INSTANCE_PWD=
EOF
  log "Wrote $LOCAL_ENV_FILE."
}

validate_browsergym() {
  "$VENV_PATH/bin/python" -c 'import browsergym.workarena, platform; print(platform.python_version())'
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
  write_local_env_file
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
  fail "Unable to prepare a working WorkArena environment with Python 3.12 or Python 3.11."
fi

log "Running post-install environment check."
if ! "$VENV_PATH/bin/python" scripts/check_workarena_env.py; then
  cat <<'EOF'
[setup_workarena_env] Post-install check reported missing WorkArena runtime prerequisites.
This is expected until one of the following is configured:
- Hugging Face gated access + token/login for ServiceNow/WorkArena-Instances
- a custom SNOW_INSTANCE_POOL file
- explicit SNOW_INSTANCE_URL / SNOW_INSTANCE_UNAME / SNOW_INSTANCE_PWD
EOF
fi

cat <<EOF
WorkArena environment bootstrap completed.

Selected Python: $SELECTED_PYTHON_VERSION
Selected interpreter: $SELECTED_PYTHON_BIN
Virtualenv: $VENV_PATH
Default WorkArena task: $DEFAULT_TASK

Notes:
- This setup is isolated from the VisualWebArena and MiniWoB stacks. It uses .venv_workarena and .env.workarena.
- WorkArena does not require Docker, but it does require gated ServiceNow instance access or explicit SNOW credentials.
- The WorkArena runner uses the native WorkArena action subset: CLICK / DOUBLE_CLICK / TYPE / SELECT / PRESS / HOVER / FOCUS / CLEAR / DRAG / SCROLL / STOP.

Next:
1. Activate the environment:
   source .venv_workarena/bin/activate
2. Review or edit:
   .env.workarena
3. Authenticate if needed:
   hf auth login
4. Re-run the check:
   python scripts/check_workarena_env.py
EOF
