from __future__ import annotations

import importlib
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.workarena_runtime import (
    apply_workarena_runtime_patches,
    ensure_workarena_no_proxy_hosts,
)

ENV_FILE_PATHS = (
    REPO_ROOT / ".env.workarena",
    REPO_ROOT / ".env.workarena.example",
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
PROXY_ENABLE_RE = re.compile(r"ProxyEnable\s+REG_DWORD\s+0x([0-9a-fA-F]+)")
PROXY_SERVER_RE = re.compile(r"ProxyServer\s+REG_SZ\s+(.+)")
PLACEHOLDER_MARKERS = ("replace_me", "<path>", "<url>", "<task>", "<token>", "changeme", "todo")
DEFAULT_TASK = "browsergym/workarena.servicenow.knowledge-base-search"
PACKAGE_SPECS = (
    ("openai", "openai", "openai"),
    ("gymnasium", "gymnasium", "gymnasium"),
    ("playwright", "playwright", "playwright"),
    ("huggingface_hub", "huggingface_hub", "huggingface_hub"),
    ("browsergym_core", "browsergym.core", "browsergym-core"),
    ("browsergym_workarena", "browsergym.workarena", "browsergym-workarena"),
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


def running_in_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


def run_optional_command(command: list[str], timeout: int = 10) -> subprocess.CompletedProcess[str] | None:
    try:
        return run_command(command, timeout=timeout)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def detect_wsl_gateway() -> str:
    result = run_optional_command(["ip", "route"], timeout=5)
    if not result or result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("default via "):
            parts = line.split()
            if len(parts) >= 3:
                return parts[2]
    return ""


def parse_windows_proxy_server(proxy_server: str) -> dict[str, str]:
    text = proxy_server.strip()
    if not text:
        return {}

    parsed: dict[str, str] = {}
    if "=" not in text:
        parsed["http"] = text
        parsed["https"] = text
        return parsed

    for chunk in text.split(";"):
        piece = chunk.strip()
        if not piece or "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"http", "https", "socks", "all"} and value:
            parsed[key] = value

    if "https" not in parsed and "http" in parsed:
        parsed["https"] = parsed["http"]
    if "http" not in parsed and "https" in parsed:
        parsed["http"] = parsed["https"]
    return parsed


def normalize_proxy_url(proxy_value: str, gateway: str) -> str:
    value = proxy_value.strip()
    if not value:
        return ""

    if "://" in value:
        scheme, rest = value.split("://", 1)
    else:
        scheme, rest = "http", value

    rest = rest.strip()
    if rest.startswith("127.0.0.1:") or rest.startswith("localhost:"):
        if not gateway:
            return ""
        _, port = rest.rsplit(":", 1)
        rest = f"{gateway}:{port}"

    return f"{scheme}://{rest}"


def bridge_wsl_localhost_proxy() -> None:
    if not running_in_wsl():
        return
    if any(os.environ.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")):
        return

    enable_result = run_optional_command(
        ["reg.exe", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings", "/v", "ProxyEnable"],
        timeout=5,
    )
    server_result = run_optional_command(
        ["reg.exe", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings", "/v", "ProxyServer"],
        timeout=5,
    )
    if not enable_result or not server_result:
        return
    if enable_result.returncode != 0 or server_result.returncode != 0:
        return

    enable_match = PROXY_ENABLE_RE.search(enable_result.stdout)
    if not enable_match or int(enable_match.group(1), 16) == 0:
        return

    server_match = PROXY_SERVER_RE.search(server_result.stdout)
    if not server_match:
        return

    gateway = detect_wsl_gateway()
    proxy_map = parse_windows_proxy_server(server_match.group(1))
    http_proxy = normalize_proxy_url(proxy_map.get("http", ""), gateway)
    https_proxy = normalize_proxy_url(proxy_map.get("https", ""), gateway)
    all_proxy = normalize_proxy_url(proxy_map.get("all", proxy_map.get("socks", "")), gateway)

    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["http_proxy"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        os.environ["https_proxy"] = https_proxy
    if all_proxy:
        os.environ["ALL_PROXY"] = all_proxy
        os.environ["all_proxy"] = all_proxy
    if http_proxy or https_proxy or all_proxy:
        os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
        os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    ensure_workarena_no_proxy_hosts()


def load_local_env_file() -> Path | None:
    env_path = REPO_ROOT / ".env.workarena"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_wrapping_quotes(value)
        if not key or key in os.environ or value == "":
            continue
        os.environ[key] = value

    bridge_wsl_localhost_proxy()
    ensure_workarena_no_proxy_hosts()

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
        version_message = "Python 3.12 is active. This repo prefers 3.12 for WorkArena."
    elif fallback:
        version_message = "Python 3.11 is active. This is the fallback interpreter for WorkArena."
    else:
        version_message = "Expected Python 3.12 (preferred) or Python 3.11 (fallback)."

    venv_path = os.environ.get("VIRTUAL_ENV", "")
    in_virtualenv = sys.prefix != sys.base_prefix
    if in_virtualenv:
        venv_message = f"Running inside virtualenv: {venv_path or sys.prefix}"
    else:
        venv_message = "Not running inside a virtualenv. Prefer `.venv_workarena`."

    return {
        "version": make_result(compatible, version_text, version_message),
        "executable": make_result(True, sys.executable, "Python executable path"),
        "virtualenv": make_result(in_virtualenv, venv_path or sys.prefix, venv_message),
    }


def check_package(module_name: str, dist_name: str) -> dict[str, Any]:
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
        label: check_package(module_name=module_name, dist_name=dist_name)
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
            else "Chromium download not found. Run `.venv_workarena/bin/python -m playwright install chromium`."
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


def check_registered_envs() -> dict[str, Any]:
    try:
        apply_workarena_runtime_patches()
        import browsergym.workarena  # noqa: F401
        from gymnasium.envs.registration import registry
    except Exception as exc:
        return make_result(False, "", f"Could not inspect registered WorkArena envs: {clean_message(str(exc))}")

    env_ids = sorted(
        env_id
        for env_id in registry
        if env_id.startswith("browsergym/workarena.") and "-l2" not in env_id and "-l3" not in env_id
    )
    if not env_ids:
        return make_result(False, "", "No WorkArena L1 environments are registered.")
    return make_result(True, str(len(env_ids)), "Registered WorkArena L1 environments")


def check_default_task() -> dict[str, Any]:
    value = os.getenv("WORKARENA_DEFAULT_TASK", DEFAULT_TASK)
    if looks_like_placeholder(value):
        return make_result(False, value, "WORKARENA_DEFAULT_TASK contains a placeholder value", label="PLACEHOLDER")
    if not value.startswith("browsergym/workarena."):
        return make_result(False, value, "WORKARENA_DEFAULT_TASK must start with browsergym/workarena.")
    return make_result(True, value, "WorkArena smoke task id")


def check_hf_auth() -> dict[str, Any]:
    token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or ""
    if token:
        if looks_like_placeholder(token):
            return make_result(False, "env", "Hugging Face token is present but still contains a placeholder value", label="PLACEHOLDER")
        return make_result(True, "env", "Hugging Face token detected from environment")

    try:
        from huggingface_hub import whoami
    except Exception as exc:
        return make_result(False, "", f"huggingface_hub import failed: {clean_message(str(exc))}")

    try:
        info = whoami()
    except Exception as exc:
        return make_result(False, "", clean_message(str(exc)))

    username = ""
    if isinstance(info, dict):
        username = str(info.get("name") or info.get("fullname") or "")
    return make_result(True, username, "Hugging Face authentication is available")


def check_instance_configuration() -> dict[str, dict[str, Any]]:
    pool = os.getenv("SNOW_INSTANCE_POOL", "")
    url = os.getenv("SNOW_INSTANCE_URL", "")
    uname = os.getenv("SNOW_INSTANCE_UNAME", "")
    pwd = os.getenv("SNOW_INSTANCE_PWD", "")

    if pool:
        if looks_like_placeholder(pool):
            pool_result = make_result(False, pool, "SNOW_INSTANCE_POOL still contains a placeholder value", label="PLACEHOLDER")
        else:
            path = Path(pool).expanduser()
            pool_result = make_result(path.exists(), str(path), "Custom ServiceNow instance pool file")
    else:
        pool_result = make_result(False, "", "SNOW_INSTANCE_POOL is not set", required=False, label="MISSING")

    if url or uname or pwd:
        missing = [name for name, value in (("SNOW_INSTANCE_URL", url), ("SNOW_INSTANCE_UNAME", uname), ("SNOW_INSTANCE_PWD", pwd)) if not value]
        if missing:
            direct_result = make_result(False, ", ".join(missing), "Direct ServiceNow credentials are incomplete", label="PARTIAL")
        elif any(looks_like_placeholder(value) for value in (url, uname, pwd)):
            direct_result = make_result(False, url, "Direct ServiceNow credentials still contain placeholder values", label="PLACEHOLDER")
        else:
            direct_result = make_result(True, url, "Direct ServiceNow instance credentials detected")
    else:
        direct_result = make_result(False, "", "Direct ServiceNow instance credentials are not set", required=False, label="MISSING")

    access_ready = pool_result["ok"] or direct_result["ok"]
    access_result = make_result(
        access_ready,
        "configured" if access_ready else "",
        "Custom pool or direct ServiceNow credentials are configured" if access_ready else "No direct WorkArena instance configuration found",
        required=False,
        label="SET" if access_ready else "MISSING",
    )

    return {
        "custom_pool": pool_result,
        "direct_instance": direct_result,
        "instance_config": access_result,
    }


def check_smoke_reset(default_task: str, auth_ok: bool, instance_ok: bool) -> dict[str, Any]:
    if not (auth_ok or instance_ok):
        return make_result(
            False,
            default_task,
            "WorkArena smoke reset skipped because neither Hugging Face auth nor explicit ServiceNow instance configuration is available.",
            label="BLOCKED",
        )

    try:
        apply_workarena_runtime_patches()
        import browsergym.workarena  # noqa: F401
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet
    except Exception as exc:
        return make_result(False, default_task, f"WorkArena dependencies are unavailable: {clean_message(str(exc))}")

    try:
        env = gym.make(
            default_task,
            headless=True,
            wait_for_user_message=False,
            action_mapping=HighLevelActionSet(subsets=["workarena"], multiaction=False).to_python_code,
        )
    except Exception as exc:
        return make_result(False, default_task, f"Failed to create WorkArena env: {clean_message(str(exc))}")

    try:
        obs, _info = env.reset(seed=0)
    except Exception as exc:
        return make_result(False, default_task, clean_message(str(exc)))
    finally:
        env.close()

    goal = str(obs.get("goal", "") or "").strip()
    message = "WorkArena smoke reset ok."
    if goal:
        message += f" Goal: {goal}"
    return make_result(True, default_task, message)


def print_section(title: str, rows: dict[str, dict[str, Any]]) -> None:
    print(f"{title}:")
    for key, result in rows.items():
        value = result["value"]
        suffix = f" [{value}]" if value else ""
        print(f"  [{result['label']}] {key}: {result['message']}{suffix}")
    print()


def main() -> int:
    env_path = load_local_env_file()

    python_results = check_python()
    package_results = check_packages()
    browser_results = check_playwright_runtime()
    workarena_results = {
        "registered_l1_envs": check_registered_envs(),
        "default_task": check_default_task(),
    }
    auth_result = check_hf_auth()
    instance_results = check_instance_configuration()
    workarena_results["hf_auth"] = auth_result
    workarena_results.update(instance_results)
    smoke_result = check_smoke_reset(
        default_task=workarena_results["default_task"]["value"] or DEFAULT_TASK,
        auth_ok=auth_result["ok"],
        instance_ok=instance_results["instance_config"]["ok"],
    )
    workarena_results["smoke_reset"] = smoke_result

    summary = {
        "package_imports_ready": make_result(
            all(result["ok"] for result in package_results.values()),
            "yes" if all(result["ok"] for result in package_results.values()) else "no",
            "Core WorkArena Python packages import successfully",
        ),
        "playwright_browser_ready": make_result(
            all(result["ok"] for result in browser_results.values()),
            "yes" if all(result["ok"] for result in browser_results.values()) else "no",
            "Playwright Chromium download and launch checks",
        ),
        "instance_access_ready": make_result(
            auth_result["ok"] or instance_results["instance_config"]["ok"],
            "yes" if auth_result["ok"] or instance_results["instance_config"]["ok"] else "no",
            "Hugging Face auth or explicit ServiceNow instance configuration",
        ),
        "workarena_smoke_ready": make_result(
            smoke_result["ok"],
            "yes" if smoke_result["ok"] else "no",
            "WorkArena smoke task reset succeeds",
        ),
    }
    overall_ok = all(result["ok"] for result in summary.values())
    summary["overall_ready"] = make_result(
        overall_ok,
        "yes" if overall_ok else "no",
        "All required WorkArena prerequisites",
    )

    print("WorkArena Environment Check")
    print(f"Repository: {REPO_ROOT}")
    print()
    print_section("Python", python_results)
    print_section("Packages", package_results)
    print_section("Browser", browser_results)
    print_section("WorkArena", workarena_results)
    if env_path:
        print("Env File:")
        print(
            f"  [LOADED] hint: Loaded local .env.workarena for this check. Existing shell exports still take precedence. [{env_path}]"
        )
        print()
    print_section("Summary", summary)

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
