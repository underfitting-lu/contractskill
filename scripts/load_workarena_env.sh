#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${WORKARENA_ENV_FILE:-$REPO_ROOT/.env.workarena}"

if [ ! -f "$ENV_FILE" ]; then
  printf '[load_workarena_env] ERROR: %s not found\n' "$ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Keep empty placeholders in .env.workarena from overriding BrowserGym fallback logic.
while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue

  key="${line%%=*}"
  key="${key%"${key##*[![:space:]]}"}"
  [[ -z "$key" ]] && continue

  if [[ -v "$key" && -z "${!key}" ]]; then
    unset "$key"
  fi
done < "$ENV_FILE"

printf '[load_workarena_env] Loaded %s\n' "$ENV_FILE"
