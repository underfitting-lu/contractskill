from __future__ import annotations

import importlib
import os
import time
from urllib import parse


DEFAULT_BROWSER_TIMEOUT_MS = 120_000
DEFAULT_LOGIN_RETRIES = 3
DEFAULT_LOGIN_RETRY_DELAY_MS = 5_000
NO_PROXY_HOSTS = (
    "localhost",
    "127.0.0.1",
    ".service-now.com",
    "service-now.com",
)

_PATCH_APPLIED = False


def _parse_positive_int(raw_value: str | None, default: int) -> int:
    try:
        value = int((raw_value or "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_workarena_browser_timeout_ms() -> int:
    return _parse_positive_int(
        os.environ.get("WORKARENA_BROWSER_TIMEOUT_MS"),
        DEFAULT_BROWSER_TIMEOUT_MS,
    )


def _append_csv_env_var(name: str, values: tuple[str, ...]) -> None:
    current = os.environ.get(name, "")
    items = [item.strip() for item in current.split(",") if item.strip()]
    seen = {item.lower() for item in items}

    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        items.append(value)
        seen.add(lowered)

    if items:
        os.environ[name] = ",".join(items)


def ensure_workarena_no_proxy_hosts() -> None:
    _append_csv_env_var("NO_PROXY", NO_PROXY_HOSTS)
    _append_csv_env_var("no_proxy", NO_PROXY_HOSTS)


def _set_timeout_on_module(module_name: str, timeout_ms: int) -> None:
    module = importlib.import_module(module_name)
    if hasattr(module, "SNOW_BROWSER_TIMEOUT"):
        setattr(module, "SNOW_BROWSER_TIMEOUT", timeout_ms)


def _build_patched_url_login(timeout_ms: int):
    import playwright.sync_api

    def patched_url_login(instance, page: playwright.sync_api.Page) -> None:
        from browsergym.workarena.config import SNOW_BROWSER_TIMEOUT

        retries = _parse_positive_int(
            os.environ.get("WORKARENA_LOGIN_RETRIES"),
            DEFAULT_LOGIN_RETRIES,
        )
        retry_delay_ms = _parse_positive_int(
            os.environ.get("WORKARENA_LOGIN_RETRY_DELAY_MS"),
            DEFAULT_LOGIN_RETRY_DELAY_MS,
        )

        snow_username, snow_password = instance.snow_credentials
        login_url = (
            f"{instance.snow_url}/login.do"
            f"?user_name={parse.quote(snow_username)}"
            f"&user_password={parse.quote(snow_password)}"
            "&sys_action=sysverb_login"
        )

        last_error = "Login did not complete."
        per_attempt_timeout_ms = max(15_000, SNOW_BROWSER_TIMEOUT // max(retries, 1))

        for attempt in range(1, retries + 1):
            page.goto(login_url, wait_until="commit", timeout=SNOW_BROWSER_TIMEOUT)
            deadline = time.monotonic() + (per_attempt_timeout_ms / 1000.0)

            while time.monotonic() < deadline:
                current_url = parse.urlparse(
                    parse.unquote(page.evaluate("() => window.location.href"))
                )
                current_path = current_url.path.lower()

                if "login.do" not in current_path:
                    return

                html = ""
                try:
                    html = page.content()
                except Exception:
                    html = ""

                if "<body" in html.lower():
                    title_text = ""
                    body_text = ""
                    try:
                        title_text = page.title().lower()
                    except Exception:
                        title_text = ""
                    try:
                        body_text = page.locator("body").inner_text(timeout=1000).lower()
                    except Exception:
                        body_text = html.lower()

                    if "user name or password invalid" in body_text:
                        last_error = "Login failed. User name or password invalid."
                        break
                    if "log in | servicenow" in title_text:
                        last_error = "Login page remained active."

                page.wait_for_timeout(1000)

            if attempt < retries:
                page.wait_for_timeout(retry_delay_ms)

        raise RuntimeError(last_error)

    return patched_url_login


def apply_workarena_runtime_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        ensure_workarena_no_proxy_hosts()
        return

    ensure_workarena_no_proxy_hosts()

    import browsergym.workarena  # noqa: F401

    timeout_ms = get_workarena_browser_timeout_ms()
    modules_with_timeout = (
        "browsergym.workarena.config",
        "browsergym.workarena.instance",
        "browsergym.workarena.install",
        "browsergym.workarena.tasks.base",
        "browsergym.workarena.tasks.form",
        "browsergym.workarena.tasks.knowledge",
        "browsergym.workarena.tasks.list",
        "browsergym.workarena.tasks.utils.form",
    )
    for module_name in modules_with_timeout:
        _set_timeout_on_module(module_name, timeout_ms)

    utils_module = importlib.import_module("browsergym.workarena.utils")
    base_module = importlib.import_module("browsergym.workarena.tasks.base")
    install_module = importlib.import_module("browsergym.workarena.install")

    patched_url_login = _build_patched_url_login(timeout_ms)
    utils_module.url_login = patched_url_login
    base_module.url_login = patched_url_login
    install_module.url_login = patched_url_login

    _PATCH_APPLIED = True
