#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${VWA_ENV_FILE:-$REPO_ROOT/.env.vwa}"
API_ENV_FILE="${API_ENV_FILE:-$REPO_ROOT/.env.api}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[load_vwa_env] ERROR: env file not found: $ENV_FILE" >&2
  echo "[load_vwa_env] Copy $REPO_ROOT/.env.vwa.example to $REPO_ROOT/.env.vwa and fill the real values." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -f "$API_ENV_FILE" ]]; then
  set -a
  source "$API_ENV_FILE"
  set +a
  echo "[load_vwa_env] Loaded API variables from $API_ENV_FILE"
fi

set -a
source "$ENV_FILE"
set +a

validate_api_key_var() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi

  python3 - "$var_name" "$value" <<'PY'
import sys

name = sys.argv[1]
value = sys.argv[2].strip()
lowered = value.lower()

if not value.isascii():
    raise SystemExit(
        f"[load_vwa_env] ERROR: {name} contains non-ASCII characters. "
        "A placeholder like '???key' was probably copied literally."
    )

markers = ("your", "newkey", "realkey", "placeholder", "example")
if any(marker in lowered for marker in markers):
    raise SystemExit(
        f"[load_vwa_env] ERROR: {name} looks like a placeholder value, not a real API key."
    )
PY
}

normalize_env_var() {
  local var_name="$1"
  local value="${!var_name:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  value="$(printf '%s' "$value" | tr -d '\r')"
  export "$var_name=$value"
}

if [[ -z "${OPENAI_API_KEY:-}" && -n "${ZAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="$ZAI_API_KEY"
  echo "[load_vwa_env] Bridged OPENAI_API_KEY from ZAI_API_KEY"
fi

if [[ -z "${OPENAI_BASE_URL:-}" && -n "${ZHIPU_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="$ZHIPU_BASE_URL"
  echo "[load_vwa_env] Bridged OPENAI_BASE_URL from ZHIPU_BASE_URL"
fi

for var_name in \
  ZAI_API_KEY \
  ZHIPU_BASE_URL \
  OPENAI_API_KEY \
  OPENAI_BASE_URL \
  VWA_CLASSIFIEDS \
  VWA_CLASSIFIEDS_RESET_TOKEN \
  VWA_SHOPPING \
  VWA_REDDIT \
  VWA_WIKIPEDIA \
  VWA_HOMEPAGE \
  VWA_FULL_RESET
do
  normalize_env_var "$var_name"
done

validate_api_key_var "ZAI_API_KEY"
validate_api_key_var "OPENAI_API_KEY"

echo "[load_vwa_env] Loaded VisualWebArena variables from $ENV_FILE"
