from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse, unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATHS = (
    REPO_ROOT / ".env.miniwob",
    REPO_ROOT / ".env.miniwob.example",
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PLACEHOLDER_MARKERS = ("replace_me", "<path>", "<url>", "<task>", "changeme", "todo")
DEFAULT_TASK = "browsergym/miniwob.click-test"
FROZEN_MINIWOB_COMMIT = "7fd85d71a4b60325c6585396ec4f48377d049838"
PACKAGE_SPECS = (
    ("openai", "openai", "openai"),
    ("gymnasium", "gymnasium", "gymnasium"),
    ("playwright", "playwright", "playwright"),
    ("browsergym_core", "browsergym.core", "browsergym-core"),
    ("browsergym_miniwob", "browsergym.miniwob", "browsergym-miniwob"),
)


def make_result(
    ok: bool,
    value: str,
    message: str,
    *,
    required: bool = True,
    label: str | None = None,
) -> dict[str, Any]:
    if label is None:
        label = "OK" if ok else ("WARN" if not required else "FAIL")
    return {
        "ok": ok,
        "required": required,
        "label": label,
        "value": value,
        "message": message,
    }


def clean_message(text: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", text or "")
    return " ".join(cleaned.replace("\r", " ").replace("\n", " ").split())


def strip_wrapping_quotes(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        return trimmed[1:-1]
    return trimmed


def load_local_env_file() -> Path | None:
    env_path = REPO_ROOT / ".env.miniwob"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = strip_wrapping_quotes(value)

    return env_path


def iter_results(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if "ok" in node and "message" in node:
            yield node
        else:
            for value in node.values():
                yield from iter_results(value)


def run_command(command: list[str], timeout: int = 10, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(cwd) if cwd else None,
    )


def looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def check_python() -> dict[str, dict[str, Any]]:
    version = sys.version_info
    version_text = platform.python_version()
    version_tuple = (version.major, version.minor)
    preferred = version_tuple == (3, 12)
    fallback = version_tuple == (3, 11)
    compatible = preferred or fallback

    if preferred:
        version_message = "Python 3.12 is active. This repo prefers 3.12 for MiniWoB++."
    elif fallback:
        version_message = "Python 3.11 is active. This is the fallback interpreter for MiniWoB++."
    else:
        version_message = "Expected Python 3.12 (preferred) or Python 3.11 (fallback)."

    venv_path = os.environ.get("VIRTUAL_ENV", "")
    in_virtualenv = sys.prefix != sys.base_prefix
    if in_virtualenv:
        venv_message = f"Running inside virtualenv: {venv_path or sys.prefix}"
    else:
        venv_message = "Not running inside a virtualenv. Prefer `.venv_miniwob`."

    return {
        "version": make_result(compatible, version_text, version_message),
        "executable": make_result(True, sys.executable, "Python executable path"),
        "virtualenv": make_result(in_virtualenv, venv_path or sys.prefix, venv_message),
    }


def check_package(label: str, module_name: str, dist_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - dependency dependent
        return make_result(False, "", f"{module_name} import failed: {clean_message(str(exc))}")

    version = ""
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        pass

    return make_result(True, version, f"{module_name} import ok")


def check_packages() -> dict[str, dict[str, Any]]:
    return {
        label: check_package(label=label, module_name=module_name, dist_name=dist_name)
        for label, module_name, dist_name in PACKAGE_SPECS
    }


def detect_missing_chromium_libs(browser_cache_root: Path) -> list[str]:
    candidates = sorted(browser_cache_root.glob("chromium-*/chrome-linux/chrome"))
    if not candidates:
        return []

    try:
        result = run_command(["ldd", str(candidates[0])], timeout=10)
    except Exception:
        return []

    missing = []
    for line in result.stdout.splitlines():
        if "=> not found" in line:
            missing.append(line.split("=>", 1)[0].strip())
    return missing


def check_playwright_runtime() -> dict[str, dict[str, Any]]:
    browser_cache_root = Path.home() / ".cache" / "ms-playwright"
    chromium_cached = any(browser_cache_root.glob("chromium-*"))
    browser_install = make_result(
        chromium_cached,
        str(browser_cache_root) if chromium_cached else "",
        (
            f"Chromium download detected under {browser_cache_root}"
            if chromium_cached
            else "Chromium download not found. Run `.venv_miniwob/bin/playwright install chromium`."
        ),
    )

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover
        return {
            "chromium_cache": browser_install,
            "chromium_launch": make_result(
                False,
                "",
                f"playwright runtime import failed: {clean_message(str(exc))}",
            ),
        }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:  # pragma: no cover
        message = clean_message(str(exc))
        missing_libs = detect_missing_chromium_libs(browser_cache_root)
        if missing_libs:
            message = "Chromium is installed, but Linux runtime libraries are missing: " + ", ".join(missing_libs)
        return {
            "chromium_cache": browser_install,
            "chromium_launch": make_result(False, "chromium", message),
        }

    return {
        "chromium_cache": browser_install,
        "chromium_launch": make_result(True, "chromium", "Chromium headless launch ok"),
    }


def parse_file_url(url: str) -> Path | None:
    if not url.startswith("file://"):
        return None
    parsed = urlparse(url)
    return Path(unquote(parsed.path))


def check_miniwob_url() -> dict[str, Any]:
    value = os.getenv("MINIWOB_URL", "")
    if not value:
        return make_result(False, "", "MINIWOB_URL is missing", label="MISSING")
    if looks_like_placeholder(value):
        return make_result(False, value, "MINIWOB_URL is set, but still contains a placeholder value", label="PLACEHOLDER")

    file_path = parse_file_url(value)
    if file_path:
        if not file_path.exists():
            return make_result(False, value, "MINIWOB_URL points to a local path that does not exist")
        sample_file = file_path / "click-test.html"
        if not sample_file.exists():
            return make_result(False, value, "MINIWOB_URL exists, but the MiniWoB task HTML files were not found")
        return make_result(True, value, "MINIWOB_URL points to a local MiniWoB task directory", label="SET")

    return make_result(True, value, "MINIWOB_URL is set", label="SET")


def check_smoke_task_value() -> dict[str, Any]:
    task_id = os.getenv("MINIWOB_DEFAULT_TASK", DEFAULT_TASK)
    if looks_like_placeholder(task_id):
        return make_result(False, task_id, "MINIWOB_DEFAULT_TASK still contains a placeholder value", required=False, label="PLACEHOLDER")
    return make_result(True, task_id, "MiniWoB smoke task id", required=False, label="SET")


def infer_repo_path_from_url(miniwob_url: str) -> Path | None:
    file_path = parse_file_url(miniwob_url)
    if not file_path:
        return None
    try:
        return file_path.parents[2]
    except IndexError:
        return None


def check_repo_checkout(miniwob_url_result: dict[str, Any]) -> dict[str, Any]:
    if not miniwob_url_result["ok"]:
        return make_result(
            True,
            "",
            "Skipped because MINIWOB_URL is not ready.",
            required=False,
            label="SKIP",
        )

    repo_root = infer_repo_path_from_url(os.getenv("MINIWOB_URL", ""))
    if repo_root is None:
        return make_result(
            True,
            "",
            "MINIWOB_URL is not a file:// path. Repository checkout check skipped.",
            required=False,
            label="OPTIONAL",
        )

    if not repo_root.exists():
        return make_result(False, str(repo_root), "Derived MiniWoB++ repository path does not exist")

    if not (repo_root / ".git").exists():
        return make_result(False, str(repo_root), "Derived MiniWoB++ path exists, but is not a git checkout")

    result = run_command(["git", "rev-parse", "HEAD"], timeout=10, cwd=repo_root)
    if result.returncode != 0:
        return make_result(False, str(repo_root), clean_message(result.stderr or result.stdout))

    commit = clean_message(result.stdout)
    if commit == FROZEN_MINIWOB_COMMIT:
        return make_result(True, commit, "MiniWoB++ repository is pinned to the expected frozen commit")

    return make_result(
        True,
        commit,
        f"MiniWoB++ repository exists, but commit differs from the recommended frozen commit {FROZEN_MINIWOB_COMMIT}",
        required=False,
        label="WARN",
    )


def check_smoke_reset(miniwob_url_result: dict[str, Any]) -> dict[str, Any]:
    if not miniwob_url_result["ok"]:
        return make_result(
            True,
            "",
            "Skipped because MINIWOB_URL is not ready.",
            required=False,
            label="SKIP",
        )

    task_id = os.getenv("MINIWOB_DEFAULT_TASK", DEFAULT_TASK)
    if looks_like_placeholder(task_id):
        return make_result(
            True,
            task_id,
            "Skipped because MINIWOB_DEFAULT_TASK still contains a placeholder value.",
            required=False,
            label="SKIP",
        )

    code = (
        "import json\n"
        "import gymnasium as gym\n"
        "import browsergym.miniwob\n"
        f"env = gym.make({task_id!r})\n"
        "obs, info = env.reset(seed=0)\n"
        "payload = {'goal': obs.get('goal', ''), 'url': obs.get('url', ''), 'keys': sorted(obs.keys())}\n"
        "print(json.dumps(payload))\n"
        "env.close()\n"
    )
    result = run_command([sys.executable, "-c", code], timeout=60)
    if result.returncode != 0:
        return make_result(False, task_id, clean_message(result.stderr or result.stdout or "MiniWoB smoke reset failed"))

    payload = json.loads(result.stdout)
    goal = payload.get("goal", "").strip()
    return make_result(
        True,
        task_id,
        f"MiniWoB smoke reset ok. Goal: {goal[:100] if goal else 'empty'}",
    )


def check_env_file_hint(loaded_env_path: Path | None) -> dict[str, Any]:
    for path in ENV_FILE_PATHS:
        if not path.exists():
            continue
        if path.name == ".env.miniwob":
            if loaded_env_path and loaded_env_path == path:
                return make_result(
                    True,
                    str(path),
                    "Loaded local .env.miniwob for this check. Existing shell exports still take precedence.",
                    required=False,
                    label="LOADED",
                )
            return make_result(True, str(path), "Found local .env.miniwob file.", required=False, label="FOUND")
        return make_result(
            True,
            str(path),
            "Template found. Copy it to `.env.miniwob`, fill the real values, then rerun this check.",
            required=False,
            label="TEMPLATE",
        )

    return make_result(True, "", "No .env.miniwob or .env.miniwob.example found.", required=False, label="OPTIONAL")


def build_summary(checks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    package_checks = checks["packages"]
    browser_checks = checks["browser"]
    miniwob_checks = checks["miniwob"]

    package_ready = all(
        package_checks[name]["ok"]
        for name in ("openai", "gymnasium", "playwright", "browsergym_core", "browsergym_miniwob")
    )
    browser_ready = browser_checks["chromium_cache"]["ok"] and browser_checks["chromium_launch"]["ok"]
    miniwob_ready = miniwob_checks["url"]["ok"] and miniwob_checks["smoke_reset"]["ok"]
    overall_ready = all(result["ok"] or not result["required"] for result in iter_results(checks))

    return {
        "package_imports_ready": make_result(package_ready, "yes" if package_ready else "no", "Core MiniWoB Python packages import successfully"),
        "playwright_browser_ready": make_result(browser_ready, "yes" if browser_ready else "no", "Playwright Chromium download and launch checks"),
        "miniwob_url_ready": make_result(miniwob_checks["url"]["ok"], "yes" if miniwob_checks["url"]["ok"] else "no", "MINIWOB_URL is configured correctly"),
        "miniwob_smoke_ready": make_result(miniwob_ready, "yes" if miniwob_ready else "no", "MiniWoB smoke task reset succeeds"),
        "overall_ready": make_result(overall_ready, "yes" if overall_ready else "no", "All required MiniWoB++ prerequisites"),
    }


def render_section(title: str, checks: dict[str, dict[str, Any]]) -> None:
    print(f"{title}:")
    for name, result in checks.items():
        value = f" [{result['value']}]" if result["value"] else ""
        print(f"  [{result['label']}] {name}: {result['message']}{value}")
    print()


def main() -> None:
    loaded_env_path = load_local_env_file()
    miniwob_url = check_miniwob_url()
    checks: dict[str, Any] = {
        "python": check_python(),
        "packages": check_packages(),
        "browser": check_playwright_runtime(),
        "miniwob": {
            "url": miniwob_url,
            "default_task": check_smoke_task_value(),
            "repo_checkout": check_repo_checkout(miniwob_url),
            "smoke_reset": check_smoke_reset(miniwob_url),
        },
        "env_file": {"hint": check_env_file_hint(loaded_env_path)},
    }
    summary = build_summary(checks)
    payload = {"ok": summary["overall_ready"]["ok"], "checks": checks, "summary": summary}

    if "--json" in sys.argv:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("MiniWoB++ Environment Check")
        print(f"Repository: {REPO_ROOT}")
        print()
        render_section("Python", checks["python"])
        render_section("Packages", checks["packages"])
        render_section("Browser", checks["browser"])
        render_section("MiniWoB", checks["miniwob"])
        render_section("Env File", checks["env_file"])
        render_section("Summary", summary)

    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
