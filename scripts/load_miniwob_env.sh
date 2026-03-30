#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${MINIWOB_ENV_FILE:-$REPO_ROOT/.env.miniwob}"

if [ ! -f "$ENV_FILE" ]; then
  printf '[load_miniwob_env] ERROR: %s not found\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

printf '[load_miniwob_env] Loaded %s\n' "$ENV_FILE"
