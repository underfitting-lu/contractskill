from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from env.miniwob_env import _assign_input_aliases
from env.workarena_runtime import apply_workarena_runtime_patches
from env.vwa_env import (
    REPO_ROOT,
    _build_element_index,
    _extract_goal_parts,
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
    if text.startswith("workarena."):
        return f"browsergym/{text}"
    raise ValueError(
        "env_name must be like 'browsergym/workarena.servicenow.knowledge-base-search' or "
        "'workarena.servicenow.knowledge-base-search'."
    )


def ensure_workarena_dependencies() -> dict[str, Any]:
    errors = []
    modules: dict[str, Any] = {}

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("gymnasium is not installed in the current WorkArena environment.")
        modules["gym_error"] = exc
    else:
        modules["gym"] = gym

    try:
        import browsergym.workarena  # noqa: F401
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("browsergym-workarena is not installed in the current WorkArena environment.")
        modules["browsergym_error"] = exc

    try:
        from browsergym.core.action.highlevel import HighLevelActionSet
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append(
            "browsergym-core action set is unavailable. Install browsergym-workarena first."
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


class WorkArenaEnv:
    """A BrowserGym wrapper for WorkArena tasks using the native WorkArena action space."""

    def __init__(
        self,
        output_root: str | Path,
        headless: bool = True,
        browser_timeout_ms: int | None = None,
    ):
        self.output_root = Path(output_root).resolve()
        self.headless = headless
        self.browser_timeout_ms = int(browser_timeout_ms or 0) or None

        self._deps: dict[str, Any] | None = None
        self._gym = None
        self._action_set = None
        self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self.planned_seed: int | None = None
        self._last_element_index: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        self._deps = ensure_workarena_dependencies()
        if not self._deps["ok"]:
            raise RuntimeError("\n".join(self._deps["errors"]))

        apply_workarena_runtime_patches()
        self._gym = self._deps["modules"]["gym"]
        HighLevelActionSet = self._deps["modules"]["HighLevelActionSet"]
        self._action_set = HighLevelActionSet(subsets=["workarena"], multiaction=False)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self._last_element_index = {}

    def reset(self, env_name: str, task_key: str, seed: int = 0) -> dict[str, Any]:
        if self._gym is None or self._action_set is None:
            raise RuntimeError("WorkArenaEnv.start() must be called before reset().")

        self.close()
        self.current_env_name = normalize_env_name(env_name)
        self.current_task_key = task_key
        effective_seed = self.planned_seed if self.planned_seed is not None else seed
        self.planned_seed = None

        try:
            self.env = self._gym.make(
                self.current_env_name,
                headless=self.headless,
                wait_for_user_message=False,
                timeout=self.browser_timeout_ms,
                action_mapping=self._action_set.to_python_code,
            )
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to create BrowserGym environment '{self.current_env_name}'. "
                "Check Python version, package installation, and WorkArena setup."
            ) from exc

        try:
            raw_observation, info = self.env.reset(seed=effective_seed)
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to reset '{self.current_env_name}'. Check Hugging Face auth or SNOW instance configuration."
            ) from exc

        raw_observation = self._refresh_raw_observation(raw_observation, delay_sec=0.35)

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

        if action_type == "click":
            resolved = self._resolve_click_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"click({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "double_click":
            resolved = self._resolve_click_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"dblclick({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "type":
            resolved = self._resolve_input_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": (
                    f"fill({json.dumps(resolved['bid'])}, {json.dumps(action['value'])})"
                ),
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "select":
            resolved = self._resolve_input_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": (
                    f"select_option({json.dumps(resolved['bid'])}, {json.dumps(action['value'])})"
                ),
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "press":
            resolved = self._resolve_focus_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": (
                    f"press({json.dumps(resolved['bid'])}, {json.dumps(action['value'])})"
                ),
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "hover":
            resolved = self._resolve_interactive_target(action["target"], target_kind="hover")
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"hover({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "focus":
            resolved = self._resolve_focus_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"focus({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "clear":
            resolved = self._resolve_input_target(action["target"])
            if not resolved["success"]:
                return resolved
            return {
                "success": True,
                "browsergym_action": f"clear({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "drag":
            from_resolved = self._resolve_interactive_target(action["target"], target_kind="drag source")
            if not from_resolved["success"]:
                return from_resolved
            to_resolved = self._resolve_interactive_target(action["value"], target_kind="drag destination")
            if not to_resolved["success"]:
                return to_resolved
            return {
                "success": True,
                "browsergym_action": (
                    f"drag_and_drop({json.dumps(from_resolved['bid'])}, {json.dumps(to_resolved['bid'])})"
                ),
                "matched_bid": from_resolved["bid"],
                "matched_text": f"{from_resolved['matched_text']} -> {to_resolved['matched_text']}",
                "message": (
                    f"{from_resolved['message']} Then resolved drag destination '{action['value']}'."
                ),
            }

        if action_type == "scroll":
            delta_y = -700 if (action.get("value") or "").strip().lower() == "up" else 700
            return {
                "success": True,
                "browsergym_action": f"scroll(0, {delta_y})",
                "matched_bid": None,
                "matched_text": None,
                "message": f"Compiled scroll action ({'up' if delta_y < 0 else 'down'}).",
            }

        if action_type == "stop":
            reason = action["reason"] or action["value"] or "Done."
            return {
                "success": True,
                "browsergym_action": f"send_msg_to_user({json.dumps(reason)})",
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

    def _resolve_click_target(self, target_text: str) -> dict[str, Any]:
        return self._resolve_target_with_preferences(
            target_text=target_text,
            entries=_sorted_entries(self._last_element_index, kind="click"),
            target_kind="click",
            preferred_role_tokens=("button", "link"),
        )

    def _resolve_input_target(self, target_text: str) -> dict[str, Any]:
        return self._resolve_target_with_preferences(
            target_text=target_text,
            entries=_sorted_entries(self._last_element_index, kind="input"),
            target_kind="input",
            preferred_role_tokens=("textbox", "searchbox", "input", "combobox"),
        )

    def _resolve_focus_target(self, target_text: str) -> dict[str, Any]:
        input_result = self._resolve_input_target(target_text)
        if input_result["success"]:
            return input_result
        return self._resolve_interactive_target(target_text, target_kind="focus")

    def _resolve_interactive_target(self, target_text: str, target_kind: str = "interactive") -> dict[str, Any]:
        return self._resolve_target_with_preferences(
            target_text=target_text,
            entries=self._interactive_entries(),
            target_kind=target_kind,
            preferred_role_tokens=("button", "link", "textbox", "combobox"),
        )

    def _resolve_target_with_preferences(
        self,
        *,
        target_text: str,
        entries: list[dict[str, Any]],
        target_kind: str,
        preferred_role_tokens: tuple[str, ...],
    ) -> dict[str, Any]:
        raw_target = (target_text or "").strip()
        if not raw_target:
            return {
                "success": False,
                "bid": None,
                "matched_text": None,
                "message": f"{target_kind} target is empty.",
            }

        normalized_target = raw_target.casefold()
        qualified_target, role_hint_tokens = self._extract_role_hints(normalized_target)
        exact_text_matches = [
            entry
            for entry in entries
            if str(entry.get("display_text", "") or "").strip().casefold() == qualified_target
        ]
        if len(exact_text_matches) == 1:
            entry = exact_text_matches[0]
            return {
                "success": True,
                "bid": entry["bid"],
                "matched_text": entry["display_text"],
                "message": f"Resolved exact {target_kind} target '{raw_target}'.",
            }
        if len(exact_text_matches) > 1:
            entry = self._prefer_entry(
                exact_text_matches,
                role_hint_tokens or preferred_role_tokens,
            )
            return {
                "success": True,
                "bid": entry["bid"],
                "matched_text": entry["display_text"],
                "message": (
                    f"Resolved duplicate exact {target_kind} target '{raw_target}' "
                    f"using role preference (count={len(exact_text_matches)})."
                ),
            }

        return _resolve_target(
            target_text=qualified_target,
            entries=entries,
            target_kind=target_kind,
        )

    @staticmethod
    def _prefer_entry(
        entries: list[dict[str, Any]],
        preferred_role_tokens: tuple[str, ...],
    ) -> dict[str, Any]:
        def _score(entry: dict[str, Any]) -> tuple[int, str, str]:
            role = str(entry.get("role", "") or "").lower()
            for index, token in enumerate(preferred_role_tokens):
                if token in role:
                    return (index, role, str(entry.get("bid", "")))
            return (len(preferred_role_tokens), role, str(entry.get("bid", "")))

        return sorted(entries, key=_score)[0]

    @staticmethod
    def _extract_role_hints(target_text: str) -> tuple[str, tuple[str, ...]]:
        text = (target_text or "").strip()
        if " with " in text:
            text = text.split(" with ", 1)[0].strip()
        if " placeholder " in text:
            text = text.split(" placeholder ", 1)[0].strip()
        role_hint_map = {
            " textbox": ("textbox", "searchbox", "input"),
            " input": ("textbox", "searchbox", "input", "combobox"),
            " button": ("button", "link"),
            " link": ("link", "button"),
            " combobox": ("combobox", "textbox", "input"),
        }

        for suffix, hints in role_hint_map.items():
            if text.endswith(suffix):
                return text[: -len(suffix)].strip(), hints
        return text, ()

    def _interactive_entries(self) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for entry in _sorted_entries(self._last_element_index, kind="click"):
            merged[entry["bid"]] = entry
        for entry in _sorted_entries(self._last_element_index, kind="input"):
            merged[entry["bid"]] = entry
        return sorted(
            merged.values(),
            key=lambda item: (item["display_text"].lower(), item["bid"]),
        )

    def step(
        self,
        action_payload: str | dict[str, Any],
        task_key: str,
        state_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("WorkArenaEnv.reset() must be called before step().")

        translation = (
            action_payload
            if isinstance(action_payload, dict)
            else {
                "success": True,
                "browsergym_action": action_payload,
                "execution_mode": "browsergym",
            }
        )

        try:
            direct_result = self._try_direct_step(translation)
            if direct_result is None:
                raw_observation, reward, terminated, truncated, info = self.env.step(
                    translation["browsergym_action"]
                )
            else:
                raw_observation, reward, terminated, truncated, info = direct_result
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"BrowserGym step failed for action {translation.get('browsergym_action', '')!r}."
            ) from exc

        raw_observation = self._refresh_raw_observation(raw_observation, delay_sec=0.2)

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

    def _try_direct_step(
        self,
        translation: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]] | None:
        browser_env = getattr(self.env, "unwrapped", self.env)
        if browser_env is None:
            return None

        try:
            from browsergym.core.action.base import execute_python_code
        except Exception:
            return None

        browsergym_action = str(translation.get("browsergym_action", "") or "")
        if not browsergym_action:
            return None

        info, send_message_to_user, report_infeasible_instructions = browser_env.pre_step()
        browser_env.last_action = browsergym_action

        try:
            execute_python_code(
                browsergym_action,
                browser_env.page,
                send_message_to_user=send_message_to_user,
                report_infeasible_instructions=report_infeasible_instructions,
            )
            browser_env.last_action_error = ""
        except Exception as exc:
            browser_env.last_action_error = f"{type(exc).__name__}: {exc}"

        return browser_env.post_step(info)

    def _refresh_raw_observation(
        self,
        raw_observation: dict[str, Any],
        *,
        delay_sec: float,
    ) -> dict[str, Any]:
        if self.env is None:
            return raw_observation

        browser_env = getattr(self.env, "unwrapped", self.env)
        get_obs = getattr(browser_env, "_get_obs", None)
        if not callable(get_obs):
            return raw_observation

        try:
            if delay_sec > 0:
                time.sleep(delay_sec)
            refreshed = get_obs()
        except Exception:
            return raw_observation

        if isinstance(refreshed, dict) and refreshed.get("axtree_object"):
            return refreshed
        return raw_observation

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

        goal_text, goal_image_urls = _extract_goal_parts(raw_observation)
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
        interactive_elements = [
            {
                "text": entry["display_text"],
                "bid": entry["bid"],
                "role": entry["role"],
                "action_hints": _entry_action_hints(entry),
            }
            for entry in self._interactive_entries()
        ]

        axtree_text = _flatten_axtree_text(raw_observation.get("axtree_object"))
        dom_text = _flatten_generic_text(raw_observation.get("dom_object"))
        page_text = axtree_text or dom_text

        return {
            "env_id": self.current_env_name,
            "goal": goal_text,
            "goal_image_urls": goal_image_urls,
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
            "interactive_elements": interactive_elements,
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


def _entry_action_hints(entry: dict[str, Any]) -> list[str]:
    role = (entry.get("role") or "").lower()
    hints: list[str] = []
    if entry.get("is_clickable"):
        hints.extend(["CLICK", "DOUBLE_CLICK", "HOVER"])
    if entry.get("is_input"):
        hints.extend(["TYPE", "PRESS", "FOCUS", "CLEAR"])
    if any(token in role for token in ("combobox", "listbox", "select")):
        hints.append("SELECT")
    if entry.get("is_clickable") or entry.get("is_input"):
        hints.append("DRAG")
    return hints
