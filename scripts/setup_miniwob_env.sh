#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${MINIWOB_VENV_PATH:-$REPO_ROOT/.venv_miniwob}"
MINIWOB_REPO_PATH="${MINIWOB_REPO_PATH:-/mnt/c/home/lzj/benchmark_assets/miniwob-plusplus}"
MINIWOB_REPO_URL="${MINIWOB_REPO_URL:-https://github.com/Farama-Foundation/miniwob-plusplus.git}"
MINIWOB_REPO_COMMIT="${MINIWOB_REPO_COMMIT:-7fd85d71a4b60325c6585396ec4f48377d049838}"
LOCAL_ENV_FILE="${MINIWOB_ENV_FILE:-$REPO_ROOT/.env.miniwob}"
MINIWOB_URL="file://${MINIWOB_REPO_PATH}/miniwob/html/miniwob/"
MINIWOB_DEFAULT_TASK="${MINIWOB_DEFAULT_TASK:-browsergym/miniwob.click-test}"

log() {
  printf '[setup_miniwob_env] %s\n' "$*"
}

fail() {
  printf '[setup_miniwob_env] ERROR: %s\n' "$*" >&2
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

  log "Installing MiniWoB++ benchmark dependencies."
  "$VENV_PATH/bin/pip" install openai gymnasium browsergym-miniwob playwright
}

install_runtime_assets() {
  log "Downloading Chromium for Playwright."
  "$VENV_PATH/bin/playwright" install chromium
}

sync_miniwob_repo() {
  mkdir -p "$(dirname "$MINIWOB_REPO_PATH")"

  if [ -e "$MINIWOB_REPO_PATH" ] && [ ! -d "$MINIWOB_REPO_PATH/.git" ]; then
    fail "$MINIWOB_REPO_PATH exists but is not a git checkout. Remove it or set MINIWOB_REPO_PATH to an empty location."
  fi

  if [ ! -d "$MINIWOB_REPO_PATH/.git" ]; then
    log "Cloning MiniWoB++ into $MINIWOB_REPO_PATH."
    git clone "$MINIWOB_REPO_URL" "$MINIWOB_REPO_PATH"
  fi

  local current_commit=""
  current_commit="$(git -C "$MINIWOB_REPO_PATH" rev-parse HEAD 2>/dev/null || true)"
  if [ "$current_commit" = "$MINIWOB_REPO_COMMIT" ]; then
    log "MiniWoB++ is already pinned to $MINIWOB_REPO_COMMIT."
    return
  fi

  log "Pinning MiniWoB++ to commit $MINIWOB_REPO_COMMIT."
  if ! git -C "$MINIWOB_REPO_PATH" fetch --depth 1 origin "$MINIWOB_REPO_COMMIT"; then
    if git -C "$MINIWOB_REPO_PATH" cat-file -e "${MINIWOB_REPO_COMMIT}^{commit}" 2>/dev/null; then
      log "Fetch failed, but the target commit already exists locally. Reusing the local checkout."
    else
      fail "Unable to fetch MiniWoB++ commit $MINIWOB_REPO_COMMIT."
    fi
  fi
  git -C "$MINIWOB_REPO_PATH" reset --hard "$MINIWOB_REPO_COMMIT"
}

write_local_env_file() {
  if [ -f "$LOCAL_ENV_FILE" ]; then
    log "$LOCAL_ENV_FILE already exists. Leaving it unchanged."
    return
  fi

  cat >"$LOCAL_ENV_FILE" <<EOF
MINIWOB_URL=$MINIWOB_URL
MINIWOB_DEFAULT_TASK=$MINIWOB_DEFAULT_TASK
EOF
  log "Wrote $LOCAL_ENV_FILE."
}

validate_browsergym() {
  MINIWOB_URL="$MINIWOB_URL" "$VENV_PATH/bin/python" -c 'import browsergym.miniwob, platform; print(platform.python_version())'
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
  sync_miniwob_repo
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
  fail "Unable to prepare a working MiniWoB++ environment with Python 3.12 or Python 3.11."
fi

log "Running post-install environment check."
if ! "$VENV_PATH/bin/python" scripts/check_miniwob_env.py; then
  cat <<'EOF'
[setup_miniwob_env] Post-install check reported missing runtime prerequisites.
This is expected if the MiniWoB URL is wrong, Chromium host libraries are missing,
or the frozen MiniWoB++ repository checkout is unavailable.
EOF
fi

cat <<EOF
MiniWoB++ environment bootstrap completed.

Selected Python: $SELECTED_PYTHON_VERSION
Selected interpreter: $SELECTED_PYTHON_BIN
Virtualenv: $VENV_PATH
MiniWoB++ repo: $MINIWOB_REPO_PATH
MiniWoB URL: $MINIWOB_URL

Notes:
- This setup is isolated from the VisualWebArena stack. It uses .venv_miniwob and .env.miniwob.
- MiniWoB++ does not require Docker.
- The frozen MiniWoB++ commit is pinned to $MINIWOB_REPO_COMMIT.

Next:
1. Activate the environment:
   source .venv_miniwob/bin/activate
2. Review or edit:
   .env.miniwob
3. Re-run the check:
   python scripts/check_miniwob_env.py
EOF
