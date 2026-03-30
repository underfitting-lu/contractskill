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
from urllib import error as urllib_error
from urllib import request as urllib_request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.vwa_env import REQUIRED_VWA_ENV_VARS


OPTIONAL_VWA_ENV_VARS = ("VWA_FULL_RESET", "OPENAI_API_KEY")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PLACEHOLDER_MARKERS = ("replace_me", "<host>", "<token>", "<your-", "changeme", "todo")
SITE_ENV_VARS = (
    ("classifieds", "VWA_CLASSIFIEDS"),
    ("shopping", "VWA_SHOPPING"),
    ("reddit", "VWA_REDDIT"),
    ("wikipedia", "VWA_WIKIPEDIA"),
    ("homepage", "VWA_HOMEPAGE"),
)
PACKAGE_SPECS = (
    ("openai", "openai", "openai"),
    ("gymnasium", "gymnasium", "gymnasium"),
    ("playwright", "playwright", "playwright"),
    ("torch", "torch", "torch"),
    ("pillow", "PIL", "pillow"),
    ("nltk", "nltk", "nltk"),
    ("browsergym_core", "browsergym.core", "browsergym-core"),
    ("browsergym_visualwebarena", "browsergym.visualwebarena", "browsergym-visualwebarena"),
)
ENV_FILE_PATHS = (
    REPO_ROOT / ".env.vwa",
    REPO_ROOT / ".env.vwa.ami.example",
    REPO_ROOT / ".env.vwa.example",
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
    env_path = REPO_ROOT / ".env.vwa"
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


def check_python() -> dict[str, dict[str, Any]]:
    version = sys.version_info
    version_text = platform.python_version()
    version_tuple = (version.major, version.minor)
    preferred = version_tuple == (3, 12)
    fallback = version_tuple == (3, 11)
    compatible = preferred or fallback

    if preferred:
        version_message = (
            "Python 3.12 is active. This repo validated browsergym.visualwebarena import on 3.12."
        )
    elif fallback:
        version_message = "Python 3.11 is active. This is the fallback interpreter for VisualWebArena."
    else:
        version_message = "Expected Python 3.12 (preferred) or Python 3.11 (fallback)."

    venv_path = os.environ.get("VIRTUAL_ENV", "")
    in_virtualenv = sys.prefix != sys.base_prefix
    if in_virtualenv:
        venv_message = f"Running inside virtualenv: {venv_path or sys.prefix}"
    else:
        venv_message = "Not running inside a virtualenv. Prefer `.venv_vwa` for VisualWebArena."

    return {
        "version": make_result(compatible, version_text, version_message),
        "executable": make_result(True, sys.executable, "Python executable path"),
        "virtualenv": make_result(
            in_virtualenv,
            venv_path or sys.prefix,
            venv_message,
        ),
    }


def check_package(label: str, module_name: str, dist_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - local dependency dependent
        return make_result(False, "", f"{module_name} import failed: {clean_message(str(exc))}")

    version = ""
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:  # pragma: no cover - import already proved presence
        pass

    message = f"{module_name} import ok"
    if label == "browsergym_visualwebarena":
        message = f"{module_name} import ok on Python {platform.python_version()}"
    return make_result(True, version, message)


def check_packages() -> dict[str, dict[str, Any]]:
    return {
        label: check_package(label=label, module_name=module_name, dist_name=dist_name)
        for label, module_name, dist_name in PACKAGE_SPECS
    }


def check_playwright_runtime() -> dict[str, dict[str, Any]]:
    browser_cache_root = Path.home() / ".cache" / "ms-playwright"
    chromium_cached = any(browser_cache_root.glob("chromium-*"))
    cached_value = str(browser_cache_root) if chromium_cached else ""
    browser_install = make_result(
        chromium_cached,
        cached_value,
        (
            f"Chromium download detected under {browser_cache_root}"
            if chromium_cached
            else "Chromium download not found. Run `.venv_vwa/bin/playwright install chromium`."
        ),
    )

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - local dependency dependent
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
    except Exception as exc:  # pragma: no cover - local system dependency dependent
        message = clean_message(str(exc))
        missing_libs = detect_missing_chromium_libs(browser_cache_root)
        if "Host system is missing dependencies to run browsers" in message:
            if missing_libs:
                message = (
                    "Chromium is installed, but Linux runtime libraries are missing: "
                    + ", ".join(missing_libs)
                    + ". Install the missing packages, then retry."
                )
            else:
                message = (
                    "Chromium is installed, but Linux runtime libraries are missing. "
                    "Run `sudo playwright install-deps`, or install `libnss3 libnspr4 libasound2t64` on Ubuntu 24.04."
                )
        return {
            "chromium_cache": browser_install,
            "chromium_launch": make_result(False, "chromium", message),
        }

    return {
        "chromium_cache": browser_install,
        "chromium_launch": make_result(True, "chromium", "Chromium headless launch ok"),
    }


def run_command(command: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def check_docker() -> dict[str, dict[str, Any]]:
    docker_path = shutil.which("docker")
    cli = make_result(
        bool(docker_path),
        docker_path or "",
        (
            f"docker CLI found at {docker_path}"
            if docker_path
            else "docker CLI not found in this WSL distro. Enable Docker Desktop WSL integration or install docker."
        ),
    )
    if not docker_path:
        return {
            "cli": cli,
            "daemon": make_result(
                False,
                "",
                "docker daemon check skipped because the docker CLI is not available.",
            ),
        }

    try:
        result = run_command(["docker", "info", "--format", "{{.ServerVersion}}"])
    except Exception as exc:  # pragma: no cover - local system dependent
        return {
            "cli": cli,
            "daemon": make_result(False, "", f"docker info failed: {clean_message(str(exc))}"),
        }

    server_version = clean_message(result.stdout)
    if result.returncode == 0 and server_version:
        daemon = make_result(
            True,
            server_version,
            f"Docker daemon reachable (server {server_version})",
        )
    else:
        error_text = clean_message(result.stderr or result.stdout)
        if "Cannot connect to the Docker daemon" in error_text:
            error_text = (
                "docker CLI is installed, but the daemon is not reachable. "
                "If you are using native Ubuntu Docker, run "
                "`sudo systemctl enable --now docker.socket docker.service`. "
                "If you just added your user to the `docker` group, restart the shell."
            )
        elif "permission denied" in error_text:
            error_text = (
                "docker daemon exists, but the current user cannot access `/var/run/docker.sock`. "
                "Add the user to the `docker` group and restart the shell."
            )
        daemon = make_result(
            False,
            "",
            error_text or f"`docker info` failed with exit code {result.returncode}",
        )

    return {"cli": cli, "daemon": daemon}


def format_env_value(name: str, value: str) -> str:
    if not value:
        return ""
    if name.endswith("_TOKEN") or name.endswith("_KEY"):
        return f"set(len={len(value)})"
    return value


def looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def check_env_var(name: str, required: bool = True) -> dict[str, Any]:
    value = os.getenv(name, "")
    if value and looks_like_placeholder(value):
        return make_result(
            False if required else True,
            format_env_value(name, value),
            f"{name} is set, but still contains a placeholder value",
            required=required,
            label="PLACEHOLDER",
        )

    if value:
        return make_result(True, format_env_value(name, value), f"{name} is set", label="SET")

    if required:
        return make_result(False, "", f"{name} is missing", label="MISSING")

    return make_result(True, "", f"{name} is not set", required=False, label="OPTIONAL")


def check_env_vars() -> dict[str, dict[str, Any]]:
    checks = {name: check_env_var(name, required=True) for name in REQUIRED_VWA_ENV_VARS}
    for name in OPTIONAL_VWA_ENV_VARS:
        checks[name] = check_env_var(name, required=False)
    return checks


def probe_url(url: str, timeout: int = 10) -> tuple[bool, str]:
    request = urllib_request.Request(
        url,
        headers={"User-Agent": "contractskill-vwa-env-check/1.0"},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            status = getattr(response, "status", 200)
            return True, f"HTTP {status} from {final_url}"
    except urllib_error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {exc.geturl()}"
    except Exception as exc:  # pragma: no cover - network dependent
        return False, clean_message(str(exc))


def check_site_reachability(env_checks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not all(env_checks[name]["ok"] for name in REQUIRED_VWA_ENV_VARS):
        return {
            name: make_result(
                True,
                "",
                "Skipped because required VWA_* values are not ready.",
                required=False,
                label="SKIP",
            )
            for name, _env_name in SITE_ENV_VARS
        } | {
            "full_reset_status": make_result(
                True,
                "",
                "Skipped because required VWA_* values are not ready.",
                required=False,
                label="SKIP",
            )
        }

    checks: dict[str, dict[str, Any]] = {}
    for label, env_name in SITE_ENV_VARS:
        url = os.environ.get(env_name, "")
        ok, detail = probe_url(url)
        checks[label] = make_result(
            ok,
            url,
            f"{env_name} reachable: {detail}" if ok else f"{env_name} unreachable: {detail}",
        )

    full_reset = os.environ.get("VWA_FULL_RESET", "")
    if full_reset and not looks_like_placeholder(full_reset):
        status_url = full_reset.rstrip("/") + "/status"
        ok, detail = probe_url(status_url)
        checks["full_reset_status"] = make_result(
            ok,
            status_url,
            f"VWA_FULL_RESET status endpoint reachable: {detail}"
            if ok
            else f"VWA_FULL_RESET status endpoint unreachable: {detail}",
            required=False,
            label="SET" if ok else "WARN",
        )
    else:
        checks["full_reset_status"] = make_result(
            True,
            "",
            "VWA_FULL_RESET not set. Full-instance reset check skipped.",
            required=False,
            label="OPTIONAL",
        )

    return checks


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


def check_env_file_hint(loaded_env_path: Path | None) -> dict[str, Any]:
    for path in ENV_FILE_PATHS:
        if not path.exists():
            continue
        if path.name == ".env.vwa":
            if loaded_env_path and loaded_env_path == path:
                return make_result(
                    True,
                    str(path),
                    "Loaded local .env.vwa for this check. Existing shell exports still take precedence.",
                    required=False,
                    label="LOADED",
                )
            return make_result(
                True,
                str(path),
                "Found local .env.vwa file.",
                required=False,
                label="FOUND",
            )
        return make_result(
            True,
            str(path),
            f"Template found ({path.name}). Copy it to `.env.vwa`, fill the real URLs/tokens, then rerun this check.",
            required=False,
            label="TEMPLATE",
        )

    return make_result(
        True,
        "",
        "No .env.vwa, .env.vwa.ami.example, or .env.vwa.example found.",
        required=False,
        label="OPTIONAL",
    )


def build_summary(checks: dict[str, Any]) -> dict[str, dict[str, Any]]:
    package_checks = checks["packages"]
    browser_checks = checks["browser"]
    docker_checks = checks["docker"]
    env_checks = checks["env_vars"]
    site_checks = checks["sites"]

    package_imports_ready = all(
        package_checks[name]["ok"]
        for name in (
            "openai",
            "gymnasium",
            "playwright",
            "torch",
            "browsergym_core",
            "browsergym_visualwebarena",
        )
    )
    docker_ready = docker_checks["cli"]["ok"] and docker_checks["daemon"]["ok"]
    env_vars_ready = all(env_checks[name]["ok"] for name in REQUIRED_VWA_ENV_VARS)
    sites_ready = all(
        site_checks[name]["ok"] for name in ("classifieds", "shopping", "reddit", "wikipedia", "homepage")
    )
    overall_ready = all(
        result["ok"] or not result["required"] for result in iter_results(checks)
    )

    return {
        "package_imports_ready": make_result(
            package_imports_ready,
            "yes" if package_imports_ready else "no",
            "Core Python packages import successfully",
        ),
        "playwright_browser_ready": make_result(
            browser_checks["chromium_cache"]["ok"] and browser_checks["chromium_launch"]["ok"],
            "yes"
            if browser_checks["chromium_cache"]["ok"] and browser_checks["chromium_launch"]["ok"]
            else "no",
            "Playwright Chromium download and launch checks",
        ),
        "docker_ready": make_result(
            docker_ready,
            "yes" if docker_ready else "no",
            "docker CLI and daemon visibility from WSL",
        ),
        "required_env_vars_ready": make_result(
            env_vars_ready,
            "yes" if env_vars_ready else "no",
            "Required VWA_* variables",
        ),
        "vwa_sites_reachable": make_result(
            sites_ready,
            "yes" if sites_ready else "no",
            "Configured VWA site URLs respond over HTTP",
        ),
        "overall_ready": make_result(
            overall_ready,
            "yes" if overall_ready else "no",
            "All required VisualWebArena prerequisites",
        ),
    }


def render_section(title: str, checks: dict[str, dict[str, Any]]) -> None:
    print(f"{title}:")
    for name, result in checks.items():
        value = f" [{result['value']}]" if result["value"] else ""
        print(f"  [{result['label']}] {name}: {result['message']}{value}")
    print()


def main() -> None:
    loaded_env_path = load_local_env_file()
    checks: dict[str, Any] = {
        "python": check_python(),
        "packages": check_packages(),
        "browser": check_playwright_runtime(),
        "docker": check_docker(),
        "env_vars": check_env_vars(),
        "sites": {},
        "env_file": {"hint": check_env_file_hint(loaded_env_path)},
    }
    checks["sites"] = check_site_reachability(checks["env_vars"])
    summary = build_summary(checks)
    payload = {"ok": summary["overall_ready"]["ok"], "checks": checks, "summary": summary}

    if "--json" in sys.argv:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("VisualWebArena Environment Check")
        print(f"Repository: {REPO_ROOT}")
        print()
        render_section("Python", checks["python"])
        render_section("Packages", checks["packages"])
        render_section("Browser", checks["browser"])
        render_section("Docker", checks["docker"])
        render_section("Environment Variables", checks["env_vars"])
        render_section("Sites", checks["sites"])
        render_section("Env File", checks["env_file"])
        render_section("Summary", summary)

    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
