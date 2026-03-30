from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from env.vwa_env import (
    REPO_ROOT,
    _build_element_index,
    _flatten_axtree_text,
    _flatten_generic_text,
    _resolve_target,
    _sanitize_for_json,
    _sorted_entries,
)


def normalize_env_name(env_name: str) -> str:
    text = (env_name or "").strip()
    if not text:
        raise ValueError("env_name must be a non-empty string.")
    if text.startswith("browsergym/"):
        return text
    if text.startswith("miniwob."):
        return f"browsergym/{text}"
    raise ValueError("env_name must be like 'browsergym/miniwob.click-test' or 'miniwob.click-test'.")


def _normalize_target_candidates(target: str) -> list[str]:
    raw = (target or "").strip()
    if not raw:
        return []

    candidates: list[str] = []

    def add(value: str) -> None:
        text = value.strip().strip("\"'")
        if text and text not in candidates:
            candidates.append(text)

    add(raw)

    lowered = raw.lower()
    for prefix in ("name:", "label:", "text:"):
        if lowered.startswith(prefix):
            add(raw[len(prefix) :])

    for prefix in ("input with label ", "field with label "):
        if lowered.startswith(prefix):
            add(raw[len(prefix) :])

    for prefix in ("input field with label ", "input field for ", "field for "):
        if lowered.startswith(prefix):
            add(raw[len(prefix) :])

    for suffix in (" button", " link", " tab", " tab element"):
        if lowered.endswith(suffix):
            add(raw[: -len(suffix)])

    if raw.endswith(")") and "(" in raw:
        add(re.sub(r"\s*\([^)]*\)\s*$", "", raw))

    contains_match = re.fullmatch(r"[a-z]+:contains\((['\"])(.+?)\1\)", raw)
    if contains_match:
        add(contains_match.group(2))

    for pattern in (
        re.compile(r"(?:[a-z]+:)?text\((['\"])(.+?)\1\)", re.IGNORECASE),
        re.compile(r"(?:[a-z]+:)?text\s*=\s*(['\"])(.+?)\1", re.IGNORECASE),
    ):
        match = pattern.fullmatch(raw)
        if match:
            add(match.group(2))

    for quoted in re.findall(r"['\"]([^'\"]+)['\"]", raw):
        add(quoted)

    bid_match = re.search(r"\bbid\s+([A-Za-z0-9]+)\b", raw, flags=re.IGNORECASE)
    if bid_match:
        add(bid_match.group(1))

    return candidates


def _resolve_miniwob_target(entries: list[dict[str, Any]], target_text: str, target_kind: str) -> dict[str, Any]:
    last_failure: dict[str, Any] | None = None
    for candidate in _normalize_target_candidates(target_text):
        resolved = _resolve_target(candidate, entries, target_kind)
        if resolved["success"]:
            return resolved
        last_failure = resolved

        normalized_candidate = candidate.lower()
        exact_matches = [
            entry for entry in entries if entry["display_text"].strip().lower() == normalized_candidate
        ]
        if exact_matches:
            entry = exact_matches[0]
            return {
                "success": True,
                "bid": entry["bid"],
                "matched_text": entry["display_text"],
                "message": f"Resolved first exact {target_kind} target '{candidate}'.",
            }

        partial_matches = [
            entry
            for entry in entries
            if normalized_candidate in entry["display_text"].strip().lower()
            or entry["display_text"].strip().lower() in normalized_candidate
        ]
        if partial_matches:
            entry = partial_matches[0]
            return {
                "success": True,
                "bid": entry["bid"],
                "matched_text": entry["display_text"],
                "message": f"Resolved first fuzzy {target_kind} target '{candidate}'.",
            }

        if target_kind == "input" and entries:
            lowered_candidate = normalized_candidate
            heuristic_entry = None
            if "username" in lowered_candidate or "user name" in lowered_candidate:
                heuristic_entry = entries[0]
            elif "verify" in lowered_candidate or "confirm" in lowered_candidate:
                heuristic_entry = entries[min(1, len(entries) - 1)]
            elif "password" in lowered_candidate:
                heuristic_entry = entries[min(1, len(entries) - 1)] if len(entries) > 1 else entries[0]
            if heuristic_entry is not None:
                return {
                    "success": True,
                    "bid": heuristic_entry["bid"],
                    "matched_text": heuristic_entry["display_text"],
                    "message": f"Resolved heuristic {target_kind} target '{candidate}'.",
                }

    if last_failure is not None:
        return last_failure
    return {
        "success": False,
        "bid": None,
        "matched_text": None,
        "message": f"{target_kind} target is empty.",
    }


def check_miniwob_env_vars() -> dict[str, list[str]]:
    missing = []
    warnings = []
    if not os.getenv("MINIWOB_URL"):
        missing.append("MINIWOB_URL")
    if not os.getenv("ZAI_API_KEY"):
        warnings.append("ZAI_API_KEY is not set. Model-backed MiniWoB runs will fail until it is exported.")
    return {"missing": missing, "warnings": warnings}


def ensure_miniwob_dependencies() -> dict[str, Any]:
    errors = []
    modules: dict[str, Any] = {}

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("gymnasium is not installed in the current MiniWoB environment.")
        modules["gym_error"] = exc
    else:
        modules["gym"] = gym

    try:
        import browsergym.miniwob  # noqa: F401
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("browsergym-miniwob is not installed in the current MiniWoB environment.")
        modules["browsergym_error"] = exc

    try:
        from browsergym.core.action.highlevel import HighLevelActionSet
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append(
            "browsergym-core action set is unavailable. Install browsergym-miniwob first."
        )
        modules["action_set_error"] = exc
    else:
        modules["HighLevelActionSet"] = HighLevelActionSet

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("Pillow is not installed. It is required to save BrowserGym screenshots.")
        modules["image_error"] = exc
    else:
        modules["Image"] = Image

    return {"ok": not errors, "errors": errors, "modules": modules}


class MiniWoBEnv:
    """A thin BrowserGym wrapper for MiniWoB++ tasks."""

    def __init__(self, output_root: str | Path, headless: bool = True):
        self.output_root = Path(output_root).resolve()
        self.headless = headless

        self._deps: dict[str, Any] | None = None
        self._gym = None
        self._action_set = None
        self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self._last_element_index: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        env_check = check_miniwob_env_vars()
        if env_check["missing"]:
            raise RuntimeError(
                "Missing required MiniWoB environment variables: "
                + ", ".join(env_check["missing"])
                + ". See docs/MINIWOB_SETUP.md."
            )

        self._deps = ensure_miniwob_dependencies()
        if not self._deps["ok"]:
            raise RuntimeError("\n".join(self._deps["errors"]))

        self._gym = self._deps["modules"]["gym"]
        HighLevelActionSet = self._deps["modules"]["HighLevelActionSet"]
        self._action_set = HighLevelActionSet(
            subsets=["miniwob_all", "chat"],
            multiaction=False,
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self._last_element_index = {}

    def reset(self, env_name: str, task_key: str, seed: int = 0) -> dict[str, Any]:
        if self._gym is None or self._action_set is None:
            raise RuntimeError("MiniWoBEnv.start() must be called before reset().")

        self.close()
        self.current_env_name = normalize_env_name(env_name)
        self.current_task_key = task_key

        try:
            self.env = self._gym.make(
                self.current_env_name,
                headless=self.headless,
                wait_for_user_message=False,
                action_mapping=self._action_set.to_python_code,
            )
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to create BrowserGym environment '{self.current_env_name}'. "
                "Check Python version, package installation, and MINIWOB_URL."
            ) from exc

        try:
            raw_observation, info = self.env.reset(seed=seed)
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to reset '{self.current_env_name}'. Check MINIWOB_URL and MiniWoB++ checkout."
            ) from exc

        return self._build_observation(
            raw_observation=raw_observation,
            task_key=task_key,
            state_index=0,
            terminated=False,
            truncated=False,
            reward=0.0,
            info=info or {},
        )

    def compile_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_type = action["action_type"]
        click_entries = _sorted_entries(self._last_element_index, kind="click")
        input_entries = _sorted_entries(self._last_element_index, kind="input")

        if action_type == "click":
            resolved = _resolve_miniwob_target(
                click_entries,
                action["target"],
                "click",
            )
            if resolved["success"]:
                return {
                    "success": True,
                    "browsergym_action": f"click({resolved['bid']!r})",
                    "matched_bid": resolved["bid"],
                    "matched_text": resolved["matched_text"],
                    "message": resolved["message"],
                }

            input_resolved = _resolve_miniwob_target(
                input_entries,
                action["target"],
                "input",
            )
            if not input_resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"click({input_resolved['bid']!r})",
                "matched_bid": input_resolved["bid"],
                "matched_text": input_resolved["matched_text"],
                "message": f"{input_resolved['message']} Resolved click via input target.",
            }

        if action_type == "type":
            resolved = _resolve_miniwob_target(
                input_entries,
                action["target"],
                "input",
            )
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"fill({resolved['bid']!r}, {action['value']!r})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "scroll":
            return {
                "success": True,
                "browsergym_action": "scroll(0, 700)",
                "matched_bid": None,
                "matched_text": None,
                "message": "Compiled scroll action.",
            }

        if action_type == "stop":
            reason = action["reason"] or action["value"] or "Done."
            return {
                "success": True,
                "browsergym_action": f"send_msg_to_user({reason!r})",
                "matched_bid": None,
                "matched_text": None,
                "message": "Compiled STOP into send_msg_to_user.",
            }

        return {
            "success": False,
            "browsergym_action": "",
            "matched_bid": None,
            "matched_text": None,
            "message": f"Unsupported action type: {action_type}",
        }

    def step(self, browsergym_action: str, task_key: str, state_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("MiniWoBEnv.reset() must be called before step().")

        try:
            raw_observation, reward, terminated, truncated, info = self.env.step(browsergym_action)
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(f"BrowserGym step failed for action {browsergym_action!r}.") from exc

        observation = self._build_observation(
            raw_observation=raw_observation,
            task_key=task_key,
            state_index=state_index,
            terminated=bool(terminated),
            truncated=bool(truncated),
            reward=float(reward or 0.0),
            info=info or {},
        )

        success = float(reward or 0.0) > 0.0
        fail_reason = ""
        if not success:
            if observation["last_action_error"]:
                fail_reason = observation["last_action_error"]
            elif observation["truncated"]:
                fail_reason = "Environment truncated the episode."
            elif observation["terminated"]:
                fail_reason = "Environment terminated without a positive score."

        step_result = {
            "success": success,
            "reward": float(reward or 0.0),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "fail_reason": fail_reason,
            "last_action_error": observation["last_action_error"],
            "info": _sanitize_for_json(info or {}),
        }
        return observation, step_result

    def _build_observation(
        self,
        raw_observation: dict[str, Any],
        task_key: str,
        state_index: int,
        terminated: bool,
        truncated: bool,
        reward: float,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        screenshot_path = self._persist_screenshot(
            raw_observation.get("screenshot"),
            task_key=task_key,
            state_index=state_index,
        )

        element_index = _build_element_index(
            axtree=raw_observation.get("axtree_object"),
            extra_properties=raw_observation.get("extra_element_properties"),
        )
        _assign_input_aliases(element_index)
        self._last_element_index = element_index

        clickable_elements = [
            {"text": entry["display_text"], "bid": entry["bid"], "role": entry["role"]}
            for entry in _sorted_entries(element_index, kind="click")
        ]
        input_fields = [
            {
                "label": entry["display_text"],
                "bid": entry["bid"],
                "name": entry["display_text"],
                "type": entry["role"] or "input",
                "placeholder": "",
            }
            for entry in _sorted_entries(element_index, kind="input")
        ]

        axtree_text = _flatten_axtree_text(raw_observation.get("axtree_object"))
        dom_text = _flatten_generic_text(raw_observation.get("dom_object"))
        page_text = axtree_text or dom_text

        return {
            "env_id": self.current_env_name,
            "goal": str(raw_observation.get("goal", "") or "").strip(),
            "goal_image_urls": [],
            "url": raw_observation.get("url", ""),
            "page_text": page_text,
            "screenshot_path": screenshot_path,
            "last_action_error": raw_observation.get("last_action_error", ""),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reward": reward,
            "open_pages_urls": raw_observation.get("open_pages_urls", []),
            "open_pages_titles": raw_observation.get("open_pages_titles", []),
            "active_page_index": raw_observation.get("active_page_index", 0),
            "clickable_elements": clickable_elements,
            "input_fields": input_fields,
            "info": _sanitize_for_json(info),
        }

    def _persist_screenshot(self, screenshot: Any, task_key: str, state_index: int) -> str:
        image_module = self._deps["modules"]["Image"]
        screenshot_abs = self.output_root / task_key / f"state_{state_index:02d}.png"
        screenshot_abs.parent.mkdir(parents=True, exist_ok=True)

        if screenshot is None:
            raise RuntimeError("BrowserGym observation does not contain a screenshot.")

        image_module.fromarray(screenshot).save(screenshot_abs)
        return screenshot_abs.relative_to(REPO_ROOT).as_posix()


def _assign_input_aliases(index: dict[str, dict[str, Any]]) -> None:
    counters: dict[str, int] = {}

    for entry in _sorted_entries(index, kind="input"):
        display_text = str(entry.get("display_text", "") or "").strip()
        if display_text and display_text != entry["bid"] and any(char.isalpha() for char in display_text):
            continue

        role = (entry.get("role") or "input").strip().lower() or "input"
        counters[role] = counters.get(role, 0) + 1
        entry["display_text"] = f"{role} {counters[role]}"
