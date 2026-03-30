from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_ENV_PATH = REPO_ROOT / ".env.api"
DEFAULT_QWEN_API_ENV_PATH = REPO_ROOT / ".env.api.qwen"


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def resolve_api_env_file(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)

    env_override = os.getenv("API_ENV_FILE", "").strip()
    if env_override:
        return Path(env_override)

    return DEFAULT_API_ENV_PATH


def load_api_env_file(path: str | Path | None = None, *, overwrite: bool = False) -> Path | None:
    env_path = resolve_api_env_file(path)
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value)
        if key and value and (overwrite or not os.getenv(key)):
            os.environ[key] = value
    return env_path
