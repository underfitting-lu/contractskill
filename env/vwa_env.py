from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from functools import wraps
from pathlib import Path
from typing import Any

from env.api_env import load_api_env_file


REQUIRED_VWA_ENV_VARS = (
    "VWA_CLASSIFIEDS",
    "VWA_CLASSIFIEDS_RESET_TOKEN",
    "VWA_SHOPPING",
    "VWA_REDDIT",
    "VWA_WIKIPEDIA",
    "VWA_HOMEPAGE",
)

_BID_PATTERN = re.compile(r"browsergym_id_([A-Za-z0-9]+)")
REPO_ROOT = Path(__file__).resolve().parents[1]
_OBS_STABILIZATION_ATTEMPTS = 4
_OBS_STABILIZATION_DELAY_MS = 750
_LIVE_DOM_ACTION_TIMEOUT_MS = 2_500
_TASK_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {}


def normalize_env_name(env_name: str) -> str:
    text = (env_name or "").strip()
    if not text:
        raise ValueError("env_name must be a non-empty string.")
    if text.startswith("browsergym/"):
        return text
    if text.startswith("visualwebarena."):
        return f"browsergym/{text}"
    if text.isdigit():
        return f"browsergym/visualwebarena.{text}"
    raise ValueError(
        "env_name must be like 'browsergym/visualwebarena.0' or 'visualwebarena.0'."
    )


def check_vwa_env_vars() -> dict:
    load_api_env_file()
    missing = [name for name in REQUIRED_VWA_ENV_VARS if not os.getenv(name)]
    warnings = []
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append(
            "OPENAI_API_KEY is not set. The official VisualWebArena evaluator may need it "
            "for some fuzzy-match validations."
        )
    return {"missing": missing, "warnings": warnings}


def apply_openai_compat_fallbacks() -> None:
    """Bridge Zhipu-compatible env vars into the names VisualWebArena expects."""
    load_api_env_file()
    zai_api_key = os.getenv("ZAI_API_KEY", "").strip()
    zhipu_base_url = os.getenv("ZHIPU_BASE_URL", "").strip()

    if not os.getenv("OPENAI_API_KEY") and zai_api_key:
        os.environ["OPENAI_API_KEY"] = zai_api_key

    if not os.getenv("OPENAI_BASE_URL") and zhipu_base_url:
        os.environ["OPENAI_BASE_URL"] = zhipu_base_url


def ensure_vwa_dependencies() -> dict:
    errors = []
    modules: dict[str, Any] = {}

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append("gymnasium is not installed in the current VWA environment.")
        modules["gym_error"] = exc
    else:
        modules["gym"] = gym

    try:
        import browsergym.visualwebarena  # noqa: F401
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append(
            "browsergym-visualwebarena is not installed in the current VWA environment."
        )
        modules["browsergym_error"] = exc

    try:
        from browsergym.core.action.highlevel import HighLevelActionSet
    except ImportError as exc:  # pragma: no cover - local setup dependent
        errors.append(
            "browsergym-core action set is unavailable. Install browsergym-visualwebarena first."
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


def _dummy_captioning_fn(images: list[Any], prompt: list[str] | None = None, max_new_tokens: int = 32) -> list[str]:
    return [""] * len(images)


def _pil_image_to_data_uri(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text", "") or "").strip()
                if text:
                    parts.append(text)
            else:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _build_remote_captioning_fn() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is unavailable for captioning fallback.") from exc

    load_api_env_file()
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ZAI_API_KEY", "").strip()
    )
    base_url = (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("ZHIPU_BASE_URL", "").strip()
    )
    model = (
        os.getenv("VWA_CAPTION_MODEL", "").strip()
        or os.getenv("VWA_EVAL_VISION_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "glm-4.6v"
    )

    if not api_key:
        raise RuntimeError("No API key available for captioning fallback.")

    client = OpenAI(api_key=api_key, base_url=base_url or None)

    def caption_images(
        images: list[Any],
        prompt: list[str] | None = None,
        max_new_tokens: int = 32,
    ) -> list[str]:
        del max_new_tokens
        captions: list[str] = []
        for index, image in enumerate(images):
            question = "Describe this image briefly."
            if prompt and index < len(prompt) and str(prompt[index] or "").strip():
                question = str(prompt[index]).strip()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": _pil_image_to_data_uri(image)}},
                        ],
                    }
                ],
                temperature=0.0,
            )
            message = response.choices[0].message
            captions.append(_extract_text_content(getattr(message, "content", "")))
        return captions

    return caption_images


def _resolve_vwa_eval_model() -> str:
    for var_name in (
        "VWA_EVAL_TEXT_MODEL",
        "VWA_EVAL_MODEL",
        "OPENAI_MODEL",
        "VWA_EVAL_VISION_MODEL",
        "VWA_CAPTION_MODEL",
    ):
        value = os.getenv(var_name, "").strip()
        if value:
            return value
    return "glm-4.6v"


def _patch_eval_helper_module(helper_module: Any) -> None:
    if getattr(helper_module, "_contractskill_eval_model_patch_applied", False):
        return

    generate = getattr(helper_module, "generate_from_openai_chat_completion", None)
    if generate is None:
        return

    @wraps(getattr(helper_module, "llm_fuzzy_match", None))
    def patched_llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
        message = (
            "Help a teacher to grade the answer of a student given a question. "
            "Keep in mind that the student may use different phrasing or wording to answer the question. "
            "The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
        )
        message += f"question: {question}\n"
        message += f"reference answer: {reference}\n"
        message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        message += f"student answer: {pred}\n"
        message += (
            "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. "
            "Only output one of these options, and nothing else."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message},
        ]
        response = generate(
            model=_resolve_vwa_eval_model(),
            messages=messages,
            temperature=0,
            max_tokens=768,
            top_p=1.0,
            context_length=0,
        ).lower()
        if "partially correct" in response or "incorrect" in response:
            return 0.0
        assert "correct" in response, response
        return 1.0

    @wraps(getattr(helper_module, "llm_ua_match", None))
    def patched_llm_ua_match(pred: str, reference: str, question: str) -> float:
        message = ""
        message += f"task: {question}\n"
        message += f"actual unachievable reason: {reference}\n"
        message += f"reported unachievable reason: {pred}\n"
        message += (
            "The task described above is inherently unachievable due to the reason specified under "
            "'actual unachievable reason'. An individual previously attempted this task and was unable "
            "to complete it. They provided a reason for their failure, which is listed under "
            "'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
            "Determine if the reported reason aligns with the actual reason, even if implicitly. "
            "If the stated reason is in line with the actual reason, respond with 'same'. "
            "Otherwise, respond with 'different'."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message},
        ]
        response = generate(
            model=_resolve_vwa_eval_model(),
            messages=messages,
            temperature=0,
            max_tokens=768,
            top_p=1.0,
            context_length=0,
        ).lower()
        if "different" in response:
            return 0.0
        assert "same" in response, response
        return 1.0

    if hasattr(helper_module, "llm_fuzzy_match"):
        helper_module.llm_fuzzy_match = patched_llm_fuzzy_match
    if hasattr(helper_module, "llm_ua_match"):
        helper_module.llm_ua_match = patched_llm_ua_match
    helper_module._contractskill_eval_model_patch_applied = True
    helper_module._contractskill_patched_llm_fuzzy_match = patched_llm_fuzzy_match
    helper_module._contractskill_patched_llm_ua_match = patched_llm_ua_match


def _patch_eval_evaluator_module(evaluator_module: Any, helper_module: Any) -> None:
    if getattr(evaluator_module, "_contractskill_eval_model_patch_applied", False):
        return
    fuzzy = getattr(helper_module, "_contractskill_patched_llm_fuzzy_match", None)
    ua = getattr(helper_module, "_contractskill_patched_llm_ua_match", None)
    if fuzzy is not None and hasattr(evaluator_module, "llm_fuzzy_match"):
        evaluator_module.llm_fuzzy_match = fuzzy
    if ua is not None and hasattr(evaluator_module, "llm_ua_match"):
        evaluator_module.llm_ua_match = ua
    evaluator_module._contractskill_eval_model_patch_applied = True


def _apply_vwa_eval_patches() -> None:
    try:
        import visualwebarena.evaluation_harness.helper_functions as vwa_helper_functions
        import visualwebarena.evaluation_harness.evaluators as vwa_evaluators
    except Exception:
        vwa_helper_functions = None
        vwa_evaluators = None

    try:
        import webarena.evaluation_harness.helper_functions as wa_helper_functions
        import webarena.evaluation_harness.evaluators as wa_evaluators
    except Exception:
        wa_helper_functions = None
        wa_evaluators = None

    if vwa_helper_functions is not None:
        _patch_eval_helper_module(vwa_helper_functions)
    if vwa_evaluators is not None and vwa_helper_functions is not None:
        _patch_eval_evaluator_module(vwa_evaluators, vwa_helper_functions)
    if wa_helper_functions is not None:
        _patch_eval_helper_module(wa_helper_functions)
    if wa_evaluators is not None and wa_helper_functions is not None:
        _patch_eval_evaluator_module(wa_evaluators, wa_helper_functions)


def _task_configs_need_page_image_query(task_configs: list[dict[str, Any]] | None) -> bool:
    for config in task_configs or []:
        eval_types = ((config or {}).get("eval") or {}).get("eval_types") or []
        if "page_image_query" in eval_types:
            return True
    return False


def _normalize_task_id(task_id: Any) -> str:
    text = str(task_id or "").strip()
    if text.startswith("vwa_"):
        text = text.removeprefix("vwa_")
    return text


def _merge_task_config_overrides(task_configs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged_configs: list[dict[str, Any]] = []
    for config in task_configs or []:
        task_id = _normalize_task_id((config or {}).get("task_id"))
        override = _TASK_CONFIG_OVERRIDES.get(task_id)
        if override:
            merged = dict(config)
            merged.update(override)
            merged_configs.append(merged)
        else:
            merged_configs.append(config)
    return merged_configs


def apply_vwa_captioning_patch() -> None:
    try:
        import browsergym.visualwebarena.task as browsergym_vwa_task
        import visualwebarena.evaluation_harness.image_utils as image_utils
    except Exception:
        return

    if getattr(browsergym_vwa_task, "_contractskill_captioning_patch_applied", False):
        return

    original_init = browsergym_vwa_task.GenericVisualWebArenaTask.__init__
    original_setup = browsergym_vwa_task.GenericVisualWebArenaTask.setup
    original_get_captioning_fn = image_utils.get_captioning_fn

    def resilient_get_captioning_fn(*args: Any, **kwargs: Any) -> Any:
        prefer_remote = os.getenv("VWA_FORCE_REMOTE_CAPTIONING", "1").strip().lower()
        if prefer_remote not in {"0", "false", "no", "off"}:
            try:
                return _build_remote_captioning_fn()
            except Exception:
                pass
        try:
            return original_get_captioning_fn(*args, **kwargs)
        except Exception:
            return _build_remote_captioning_fn()

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.task_configs = _merge_task_config_overrides(getattr(self, "task_configs", None))

    def patched_setup(self: Any, page: Any) -> tuple[str, dict]:
        if _task_configs_need_page_image_query(getattr(self, "task_configs", None)):
            result = original_setup(self, page)
            _apply_vwa_eval_patches()
            return result

        image_utils.get_captioning_fn = lambda *args, **kwargs: _dummy_captioning_fn
        try:
            result = original_setup(self, page)
            _apply_vwa_eval_patches()
            return result
        finally:
            image_utils.get_captioning_fn = resilient_get_captioning_fn

    image_utils.get_captioning_fn = resilient_get_captioning_fn
    browsergym_vwa_task.GenericVisualWebArenaTask.__init__ = patched_init
    browsergym_vwa_task.GenericVisualWebArenaTask.setup = patched_setup
    browsergym_vwa_task._contractskill_captioning_patch_applied = True


def _format_exception(exc: Exception) -> str:
    parts: list[str] = []
    current: BaseException | None = exc

    while current is not None:
        text = f"{type(current).__name__}: {current}"
        if text not in parts:
            parts.append(text)
        current = current.__cause__ or current.__context__

    return " | ".join(parts)


class VisualWebArenaEnv:
    """A thin BrowserGym wrapper for VisualWebArena tasks."""

    def __init__(
        self,
        output_root: str | Path,
        headless: bool = True,
        browser_timeout_ms: int = 45_000,
    ):
        self.output_root = Path(output_root).resolve()
        self.headless = headless
        self.browser_timeout_ms = browser_timeout_ms

        self._deps: dict | None = None
        self._gym = None
        self._action_set = None
        self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self._last_element_index: dict[str, dict] = {}
        self._last_open_pages_titles: list[str] = []
        self._last_active_page_index: int = 0

    def start(self) -> None:
        apply_openai_compat_fallbacks()
        env_check = check_vwa_env_vars()
        if env_check["missing"]:
            raise RuntimeError(
                "Missing required VisualWebArena environment variables: "
                + ", ".join(env_check["missing"])
                + ". See docs/VISUALWEBARENA_SETUP.md."
            )

        self._deps = ensure_vwa_dependencies()
        if not self._deps["ok"]:
            raise RuntimeError("\n".join(self._deps["errors"]))

        apply_vwa_captioning_patch()

        self._gym = self._deps["modules"]["gym"]
        HighLevelActionSet = self._deps["modules"]["HighLevelActionSet"]
        self._action_set = HighLevelActionSet(subsets=["visualwebarena"], multiaction=False)

    def set_task_config_overrides(self, split_items: list[dict[str, Any]] | None) -> None:
        global _TASK_CONFIG_OVERRIDES
        overrides: dict[str, dict[str, Any]] = {}
        for item in split_items or []:
            task_id = _normalize_task_id((item or {}).get("task_id"))
            if not task_id:
                continue
            overrides[task_id] = dict(item)
        _TASK_CONFIG_OVERRIDES = overrides

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.current_env_name = ""
        self.current_task_key = ""
        self._last_element_index = {}
        self._last_open_pages_titles = []
        self._last_active_page_index = 0

    def reset(self, env_name: str, task_key: str, seed: int = 0) -> dict:
        if self._gym is None or self._action_set is None:
            raise RuntimeError("VisualWebArenaEnv.start() must be called before reset().")

        self.close()
        self.current_env_name = normalize_env_name(env_name)
        self.current_task_key = task_key

        try:
            self.env = self._gym.make(
                self.current_env_name,
                headless=self.headless,
                wait_for_user_message=False,
                action_mapping=self._action_set.to_python_code,
                timeout=self.browser_timeout_ms,
            )
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to create BrowserGym environment '{self.current_env_name}'. "
                "Check Python version, package installation, and benchmark deployment. "
                f"Root cause: {_format_exception(exc)}"
            ) from exc

        try:
            raw_observation, info = self.env.reset(seed=seed)
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"Failed to reset '{self.current_env_name}'. "
                "Check VWA_* URLs, site availability, homepage/full-reset services, "
                f"and evaluator dependencies. Root cause: {_format_exception(exc)}"
            ) from exc

        raw_observation = self._stabilize_raw_observation(raw_observation)
        return self._build_observation(
            raw_observation=raw_observation,
            task_key=task_key,
            state_index=0,
            terminated=False,
            truncated=False,
            reward=0.0,
            info=info or {},
        )

    def compile_action(self, action: dict) -> dict:
        action_type = action["action_type"]

        if action_type == "click":
            tab_switch = self._resolve_open_tab_target(action["target"])
            if tab_switch is not None:
                return tab_switch
            if action.get("target_mode") == "bid":
                resolved = _resolve_bid_target(
                    bid=action["target"],
                    entries=_sorted_entries(self._last_element_index, kind="click"),
                    target_kind="click",
                )
            else:
                resolved = _resolve_target(
                    target_text=action["target"],
                    entries=_sorted_entries(self._last_element_index, kind="click"),
                    target_kind="click",
                )
            if not resolved["success"]:
                live_dom_fallback = self._build_live_dom_translation(action)
                if live_dom_fallback is not None:
                    return live_dom_fallback
                return resolved
            return {
                "success": True,
                "browsergym_action": f"click({json.dumps(resolved['bid'])})",
                "matched_bid": resolved["bid"],
                "matched_text": resolved["matched_text"],
                "message": resolved["message"],
            }

        if action_type == "type":
            if action.get("target_mode") == "bid":
                resolved = _resolve_bid_target(
                    bid=action["target"],
                    entries=_sorted_entries(self._last_element_index, kind="input"),
                    target_kind="input",
                )
            else:
                resolved = _resolve_target(
                    target_text=action["target"],
                    entries=_sorted_entries(self._last_element_index, kind="input"),
                    target_kind="input",
                )
            if not resolved["success"]:
                live_dom_fallback = self._build_live_dom_translation(action)
                if live_dom_fallback is not None:
                    return live_dom_fallback
                return resolved
            resolved_role = str(self._last_element_index.get(resolved["bid"], {}).get("role", "") or "").lower()
            resolved_tag = ""
            browser_env = self._get_browser_env()
            page = getattr(browser_env, "page", None) if browser_env is not None else None
            if page is not None:
                resolved_tag = _bid_locator_tag_name(page, resolved["bid"])
            if "combobox" in resolved_role and resolved_tag == "select":
                return {
                    "success": True,
                    "browsergym_action": (
                        f"select_option({json.dumps(resolved['bid'])}, {json.dumps(action['value'])})"
                    ),
                    "matched_bid": resolved["bid"],
                    "matched_text": resolved["matched_text"],
                    "message": (
                        f"{resolved['message']} Compiled as select_option for combobox input "
                        f"(tag={resolved_tag or 'unknown'})."
                    ),
                }
            return {
                "success": True,
                "browsergym_action": (
                    f"fill({json.dumps(resolved['bid'])}, {json.dumps(action['value'])})"
                ),
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

    def step(self, action_payload: str | dict[str, Any], task_key: str, state_index: int) -> tuple[dict, dict]:
        if self.env is None:
            raise RuntimeError("VisualWebArenaEnv.reset() must be called before step().")

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
            if translation.get("execution_mode") == "live_dom":
                raw_observation, reward, terminated, truncated, info = self._execute_live_dom_translation(
                    translation
                )
            else:
                raw_observation, reward, terminated, truncated, info = self.env.step(
                    translation["browsergym_action"]
                )
        except Exception as exc:  # pragma: no cover - local setup dependent
            raise RuntimeError(
                f"BrowserGym step failed for action {translation.get('browsergym_action', '')!r}. "
                f"Root cause: {_format_exception(exc)}"
            ) from exc

        raw_observation = self._stabilize_raw_observation(raw_observation)
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

    def _get_browser_env(self) -> Any | None:
        if self.env is None:
            return None
        return getattr(self.env, "unwrapped", self.env)

    def _stabilize_raw_observation(self, raw_observation: dict) -> dict:
        if not _raw_observation_needs_retry(raw_observation):
            return raw_observation

        browser_env = self._get_browser_env()
        if browser_env is None:
            return raw_observation

        latest = raw_observation
        page = getattr(browser_env, "page", None)
        context = getattr(browser_env, "context", None)
        wait_dom_loaded = getattr(browser_env, "_wait_dom_loaded", None)
        get_obs = getattr(browser_env, "_get_obs", None)

        if page is None or get_obs is None:
            return raw_observation

        for _ in range(_OBS_STABILIZATION_ATTEMPTS):
            try:
                page.wait_for_timeout(_OBS_STABILIZATION_DELAY_MS)
            except Exception:
                time.sleep(_OBS_STABILIZATION_DELAY_MS / 1000.0)

            try:
                if callable(wait_dom_loaded):
                    wait_dom_loaded()
            except Exception:
                pass

            try:
                if context is not None:
                    context.cookies()
            except Exception:
                pass

            try:
                latest = get_obs()
            except Exception:
                continue

            if not _raw_observation_needs_retry(latest):
                return latest

        return latest

    def _build_live_dom_translation(self, action: dict[str, Any]) -> dict[str, Any] | None:
        browser_env = self._get_browser_env()
        page = getattr(browser_env, "page", None)
        if page is None:
            return None

        action_type = str(action.get("action_type", "") or "")
        recipe: dict[str, Any] | None = None
        if action_type == "click":
            recipe = _find_live_click_recipe(
                page=page,
                target=str(action.get("target", "") or ""),
                target_mode=str(action.get("target_mode", "text") or "text"),
                current_url=str(getattr(page, "url", "") or ""),
            )
        elif action_type == "type":
            recipe = _find_live_type_recipe(
                page=page,
                target=str(action.get("target", "") or ""),
                value=str(action.get("value", "") or ""),
                target_mode=str(action.get("target_mode", "text") or "text"),
                current_url=str(getattr(page, "url", "") or ""),
            )

        if recipe is None:
            return None

        return {
            "success": True,
            "browsergym_action": _format_live_dom_action(action),
            "matched_bid": recipe.get("bid"),
            "matched_text": str(action.get("target", "") or ""),
            "message": f"Resolved {action_type} target via live DOM fallback.",
            "execution_mode": "live_dom",
            "dom_recipe": recipe,
            "dom_action_type": action_type,
            "value": action.get("value"),
            "reason": action.get("reason"),
        }

    def _execute_live_dom_translation(self, translation: dict[str, Any]) -> tuple[dict, float, bool, bool, dict]:
        browser_env = self._get_browser_env()
        if browser_env is None:
            raise RuntimeError("Browser environment is unavailable for live DOM execution.")

        page = getattr(browser_env, "page", None)
        if page is None:
            raise RuntimeError("Active BrowserGym page is unavailable for live DOM execution.")

        browser_env.last_action = translation.get("browsergym_action", "")
        info, send_message_to_user, _ = browser_env.pre_step()
        try:
            action_type = translation.get("dom_action_type")
            if action_type == "click":
                _execute_live_click(page, translation["dom_recipe"])
            elif action_type == "type":
                _execute_live_type(page, translation["dom_recipe"], str(translation.get("value", "") or ""))
            elif action_type == "scroll":
                page.evaluate("window.scrollBy(0, 700)")
            elif action_type == "stop":
                send_message_to_user(str(translation.get("reason") or translation.get("value") or "Done."))
            else:
                raise RuntimeError(f"Unsupported live DOM action type: {action_type}")
            browser_env.last_action_error = ""
        except Exception as exc:
            browser_env.last_action_error = f"{type(exc).__name__}: {exc}"

        return browser_env.post_step(info)

    def _build_observation(
        self,
        raw_observation: dict,
        task_key: str,
        state_index: int,
        terminated: bool,
        truncated: bool,
        reward: float,
        info: dict,
    ) -> dict:
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
        self._last_element_index = element_index
        self._last_open_pages_titles = list(raw_observation.get("open_pages_titles", []) or [])
        self._last_active_page_index = _coerce_int(raw_observation.get("active_page_index", 0))

        clickable_elements = [
            {"text": entry["display_text"], "bid": entry["bid"], "role": entry["role"]}
            for entry in _sorted_entries(element_index, kind="click")
        ]
        input_fields = [
            {
                "label": entry["display_text"],
                "bid": entry["bid"],
                "name": "",
                "type": entry["role"],
                "placeholder": "",
            }
            for entry in _sorted_entries(element_index, kind="input")
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
            "active_page_index": _coerce_int(raw_observation.get("active_page_index", 0)),
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

    def _resolve_open_tab_target(self, target_text: str | None) -> dict[str, Any] | None:
        target = (target_text or "").strip()
        if not target:
            return None

        active_index = self._last_active_page_index
        titles = self._last_open_pages_titles
        exact_matches = [
            (index, title)
            for index, title in enumerate(titles)
            if isinstance(title, str) and title.strip() == target
        ]
        if not exact_matches:
            lower_target = target.lower()
            exact_matches = [
                (index, title)
                for index, title in enumerate(titles)
                if isinstance(title, str) and title.strip().lower() == lower_target
            ]
        if not exact_matches:
            return None

        for index, title in exact_matches:
            if index == active_index:
                return {
                    "success": True,
                    "browsergym_action": f"tab_focus({index})",
                    "matched_bid": None,
                    "matched_text": title,
                    "message": f"Matched open tab {title!r}, which is already active.",
                }

            return {
                "success": True,
                "browsergym_action": f"tab_focus({index})",
                "matched_bid": None,
                "matched_text": title,
                "message": f"Resolved tab switch target {title!r} to open tab index {index}.",
            }

        return None


def _extract_goal_parts(raw_observation: dict) -> tuple[str, list[str]]:
    goal_messages = raw_observation.get("goal_object")
    if not isinstance(goal_messages, list):
        goal = str(raw_observation.get("goal", "") or "").strip()
        return goal, []

    text_parts = []
    image_urls = []
    for item in goal_messages:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = str(item.get("text", "") or "").strip()
            if text:
                text_parts.append(text)
        elif item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                url = str(image_url.get("url", "") or "").strip()
                if url:
                    image_urls.append(url)

    return "\n".join(text_parts).strip(), image_urls


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value

    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _coerce_int(value[0], default=default)

    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            pass

    try:
        return int(value)
    except Exception:
        return default


def _flatten_axtree_text(axtree: Any) -> str:
    lines: list[str] = []
    seen = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            role = _normalize_text(_coerce_text(node.get("role")))
            name = _normalize_text(_coerce_text(node.get("name")))
            value = _normalize_text(_coerce_text(node.get("value")))
            desc = _normalize_text(_coerce_text(node.get("description")))

            parts = []
            if role:
                parts.append(role)
            if name:
                parts.append(name)
            if value and value != name:
                parts.append(value)
            if desc and desc not in {name, value}:
                parts.append(desc)

            if parts:
                line = " | ".join(parts)
                if line not in seen:
                    seen.add(line)
                    lines.append(line)

            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(axtree)
    return "\n".join(lines[:400])


def _flatten_generic_text(value: Any) -> str:
    leaves: list[str] = []
    seen = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for child in node.values():
                walk(child)
            return
        if isinstance(node, list):
            for child in node:
                walk(child)
            return

        text = _normalize_text(_coerce_text(node))
        if text and text not in seen:
            seen.add(text)
            leaves.append(text)

    walk(value)
    return "\n".join(leaves[:500])


def _build_element_index(axtree: Any, extra_properties: Any) -> dict[str, dict]:
    extra_properties = extra_properties if isinstance(extra_properties, dict) else {}
    index: dict[str, dict] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            bid = _extract_bid(node)
            role = _normalize_text(_coerce_text(node.get("role")))
            texts = set()

            for key in ("name", "description", "value", "placeholder", "title", "text"):
                text = _normalize_text(_coerce_text(node.get(key)))
                if text:
                    texts.add(text)

            properties = node.get("properties")
            if isinstance(properties, list):
                for item in properties:
                    if not isinstance(item, dict):
                        continue
                    name = _normalize_text(_coerce_text(item.get("name")))
                    value = _normalize_text(_coerce_text(item.get("value")))
                    if not value:
                        continue
                    if name in {
                        "label",
                        "placeholder",
                        "description",
                        "roledescription",
                        "aria-label",
                        "title",
                        "value",
                    }:
                        texts.add(value)

            if bid:
                entry = index.setdefault(
                    bid,
                    {
                        "bid": bid,
                        "role": "",
                        "texts": set(),
                        "visible": _infer_visible(extra_properties.get(bid)),
                    },
                )
                if role and not entry["role"]:
                    entry["role"] = role
                entry["texts"].update(texts)

            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(axtree)

    for entry in index.values():
        entry["display_text"] = _pick_display_text(entry["texts"])
        role = entry["role"].lower()
        entry["is_input"] = any(
            token in role for token in ("textbox", "searchbox", "combobox", "textarea", "spinbutton")
        )
        entry["is_clickable"] = any(
            token in role
            for token in (
                "button",
                "link",
                "menuitem",
                "tab",
                "checkbox",
                "radio",
                "option",
                "switch",
                "combobox",
            )
        )
        if (entry["is_input"] or entry["is_clickable"]) and not entry["display_text"]:
            entry["display_text"] = entry["bid"]

    return index


def _sorted_entries(index: dict[str, dict], kind: str) -> list[dict]:
    if kind == "click":
        entries = [entry for entry in index.values() if entry["visible"] and entry["is_clickable"]]
    else:
        entries = [entry for entry in index.values() if entry["visible"] and entry["is_input"]]

    return sorted(
        (entry for entry in entries if entry["display_text"]),
        key=lambda item: (item["display_text"].lower(), item["bid"]),
    )


def _target_entry_priority(entry: dict[str, Any], target_kind: str) -> tuple[int, int, int, str]:
    role = str(entry.get("role") or "").lower()
    is_input = bool(entry.get("is_input"))

    if target_kind == "click":
        primary = 1
        if "button" in role or "link" in role or "menuitem" in role or "tab" in role:
            primary = 5
        elif "checkbox" in role or "radio" in role or "option" in role or "switch" in role:
            primary = 4
        elif "combobox" in role and not is_input:
            primary = 3
        elif not is_input:
            primary = 2
        secondary = 0 if is_input else 1
        return (primary, secondary, -len(str(entry.get("display_text") or "")), str(entry.get("bid") or ""))

    primary = 5 if is_input else 1
    secondary = 1 if "combobox" in role or "textbox" in role or "textarea" in role else 0
    return (primary, secondary, -len(str(entry.get("display_text") or "")), str(entry.get("bid") or ""))


def _resolve_target(target_text: str, entries: list[dict], target_kind: str) -> dict:
    raw_target = (target_text or "").strip()
    if not raw_target:
        return {
            "success": False,
            "bid": None,
            "matched_text": None,
            "message": f"{target_kind} target is empty.",
        }

    target_variants = _normalized_text_variants(raw_target)
    normalized_target = target_variants[0].lower()
    exact_text_matches = [
        entry
        for entry in entries
        if normalized_target in {variant.lower() for variant in _normalized_text_variants(entry["display_text"])}
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
        entry = sorted(
            exact_text_matches,
            key=lambda item: _target_entry_priority(item, target_kind),
            reverse=True,
        )[0]
        return {
            "success": True,
            "bid": entry["bid"],
            "matched_text": entry["display_text"],
            "message": (
                f"Resolved duplicate exact {target_kind} target '{raw_target}' "
                f"to first matching element (count={len(exact_text_matches)})."
            ),
        }

    exact_bid_matches = [entry for entry in entries if entry["bid"] == raw_target]
    if len(exact_bid_matches) == 1:
        entry = exact_bid_matches[0]
        return {
            "success": True,
            "bid": entry["bid"],
            "matched_text": entry["display_text"],
            "message": f"Resolved exact {target_kind} bid '{raw_target}'.",
        }

    partial_matches = []
    for entry in entries:
        entry_variants = [variant.lower() for variant in _normalized_text_variants(entry["display_text"])]
        matched = False
        for target_variant in target_variants:
            normalized_target_variant = target_variant.lower()
            target_tokens = re.findall(r"[a-z0-9]+", normalized_target_variant)
            for entry_variant in entry_variants:
                if normalized_target_variant in entry_variant:
                    matched = True
                    break
                if entry_variant not in normalized_target_variant:
                    continue
                if target_kind == "click":
                    entry_tokens = re.findall(r"[a-z0-9]+", entry_variant)
                    # Allow truncated visible prefixes like long product titles, but reject tiny generic
                    # controls such as "cookie" from hijacking a product-title click.
                    if not normalized_target_variant.startswith(entry_variant):
                        continue
                    if len(entry_variant) < min(12, len(normalized_target_variant)):
                        continue
                    if len(entry_tokens) < 2 and len(target_tokens) >= 2:
                        continue
                matched = True
                break
            if matched:
                break
        if matched:
            partial_matches.append(entry)

    if len(partial_matches) == 1:
        entry = partial_matches[0]
        return {
            "success": True,
            "bid": entry["bid"],
            "matched_text": entry["display_text"],
            "message": f"Resolved fuzzy {target_kind} target '{raw_target}'.",
        }
    if len(partial_matches) > 1:
        return {
            "success": False,
            "bid": None,
            "matched_text": None,
            "message": (
                f"Ambiguous fuzzy {target_kind} target '{raw_target}' "
                f"(count={len(partial_matches)}). Matching candidates: "
                + ", ".join(entry["display_text"] for entry in partial_matches[:10])
            ),
        }

    available = ", ".join(entry["display_text"] for entry in entries[:25]) or "none"
    return {
        "success": False,
        "bid": None,
        "matched_text": None,
        "message": (
            f"Could not resolve {target_kind} target '{raw_target}'. "
            f"Available {target_kind} targets: {available}"
        ),
    }


def _resolve_bid_target(bid: str, entries: list[dict], target_kind: str) -> dict:
    raw_bid = (bid or "").strip()
    if not raw_bid:
        return {
            "success": False,
            "bid": None,
            "matched_text": None,
            "message": f"{target_kind} bid target is empty.",
        }

    exact_bid_matches = [entry for entry in entries if entry["bid"] == raw_bid]
    if len(exact_bid_matches) == 1:
        entry = exact_bid_matches[0]
        return {
            "success": True,
            "bid": entry["bid"],
            "matched_text": entry["display_text"],
            "message": f"Resolved {target_kind} bid target '{raw_bid}'.",
        }

    available_bids = ", ".join(entry["bid"] for entry in entries[:40]) or "none"
    return {
        "success": False,
        "bid": None,
        "matched_text": None,
        "message": (
            f"Could not resolve {target_kind} bid target '{raw_bid}'. "
            f"Available bids: {available_bids}"
        ),
    }


def _extract_bid(node: dict) -> str | None:
    for key in ("browsergym_id", "bid"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in node.values():
        bid = _extract_bid_from_value(value)
        if bid:
            return bid
    return None


def _extract_bid_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        match = _BID_PATTERN.search(value)
        return match.group(1) if match else None
    if isinstance(value, dict):
        for child in value.values():
            bid = _extract_bid_from_value(child)
            if bid:
                return bid
        return None
    if isinstance(value, list):
        for child in value:
            bid = _extract_bid_from_value(child)
            if bid:
                return bid
        return None
    return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "value" in value:
            return _coerce_text(value["value"])
        parts = []
        for key in ("name", "text", "description", "label"):
            if key in value:
                text = _coerce_text(value[key])
                if text:
                    parts.append(text)
        return " ".join(parts)
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return ""


def _normalize_text(text: str) -> str:
    raw = " ".join((text or "").split())
    if not raw:
        return ""

    cleaned = raw.replace("\uFFFD", " ")
    cleaned = re.sub(r"(?:^|\s)[^\x00-\x7F?]+[?]?(?=[A-Za-z0-9])", " ", cleaned)
    cleaned = re.sub(r"[\uE000-\uF8FF]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    ascii_match = re.search(r"[A-Za-z0-9]", cleaned)
    if ascii_match and ascii_match.start() > 0:
        prefix = cleaned[: ascii_match.start()]
        if not any(ch.isascii() and ch.isalnum() for ch in prefix):
            cleaned = cleaned[ascii_match.start() :].strip()

    return " ".join(cleaned.split())


def _normalized_text_variants(text: str) -> list[str]:
    variants: list[str] = []

    def add(value: str) -> None:
        normalized = _normalize_text(value)
        if normalized and normalized not in variants:
            variants.append(normalized)

    raw = (text or "").strip()
    add(raw)
    ellipsis_trimmed = re.sub(r"\s*(?:\.{3}|.)\s*$", "", raw).strip()
    add(ellipsis_trimmed)
    add(ellipsis_trimmed.replace("...", " ").replace(".", " "))
    add(re.sub(r"\s*\([^)]*\)\s*$", "", ellipsis_trimmed))
    add(re.sub(r"\(\s*\d[\d,]*\s+items?\s*\)", "", raw, flags=re.IGNORECASE))
    add(re.sub(r"\s+\d[\d,]*\s+items?\s*$", "", raw, flags=re.IGNORECASE))
    for separator in (" | ", "|", " - ", " - ", " - ", ": "):
        if separator in ellipsis_trimmed:
            prefix, suffix = ellipsis_trimmed.split(separator, 1)
            if len(prefix.strip()) >= 3:
                add(prefix.strip())
            if len(suffix.strip()) >= 3:
                add(suffix.strip())
    add(re.sub(r"^[^A-Za-z0-9]+", "", raw))
    add(re.sub(r"(?:^|\s)[^\x00-\x7F?]+[?]?(?=[A-Za-z0-9])", " ", raw))
    add(re.sub(r"[\uE000-\uF8FF]", " ", raw))
    return variants


def _pick_display_text(texts: set[str]) -> str:
    cleaned = [text for text in sorted({_normalize_text(text) for text in texts}) if text]
    if not cleaned:
        return ""
    return sorted(cleaned, key=lambda item: (len(item), item.lower()))[0]


def _infer_visible(extra_props: Any) -> bool:
    if not isinstance(extra_props, dict):
        return True
    for key in ("visibility", "visible", "is_visible"):
        value = extra_props.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "hidden", "none", "0"}:
                return False
            if lowered in {"true", "visible", "1"}:
                return True
    return True


def _raw_observation_needs_retry(raw_observation: dict) -> bool:
    axtree = raw_observation.get("axtree_object")
    extra_properties = raw_observation.get("extra_element_properties")
    element_index = _build_element_index(axtree, extra_properties)
    clickable_count = len(_sorted_entries(element_index, kind="click"))
    input_count = len(_sorted_entries(element_index, kind="input"))

    page_text = _flatten_axtree_text(axtree) or _flatten_generic_text(raw_observation.get("dom_object"))
    normalized_page_text = page_text.lower()
    no_interactives = clickable_count == 0 and input_count == 0
    interactive_roles_present = any(
        token in normalized_page_text
        for token in ("link |", "button |", "textbox |", "searchbox |", "combobox |", "textarea |")
    )
    too_short = len([line for line in normalized_page_text.splitlines() if line.strip()]) <= 6

    return no_interactives and (
        "busy | 1" in normalized_page_text or interactive_roles_present or too_short
    )


def _format_live_dom_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type", "") or "").upper()
    target = str(action.get("target", "") or "")
    value = action.get("value")
    reason = action.get("reason")

    if action_type == "TYPE":
        return f"LIVE_TYPE[{target}={value}]"
    if action_type == "CLICK":
        return f"LIVE_CLICK[{target}]"
    if action_type == "SCROLL":
        return "LIVE_SCROLL"
    if action_type == "STOP":
        return f"LIVE_STOP[{reason or value or 'Done.'}]"
    return f"LIVE_{action_type}[{target}]"


def _is_classifieds_url(current_url: str) -> bool:
    normalized = (current_url or "").strip().lower()
    return ":9980" in normalized or "classifieds" in normalized


def _build_bid_css_selector(bid: str) -> str:
    safe_bid = (bid or "").strip()
    return f'[bid="{safe_bid}"], [browsergym_id="{safe_bid}"], [data-testid="{safe_bid}"]'


def _find_live_click_recipe(
    page: Any,
    target: str,
    target_mode: str,
    current_url: str,
) -> dict[str, Any] | None:
    raw_target = (target or "").strip()
    if not raw_target:
        return None

    recipes: list[dict[str, Any]] = []
    if target_mode == "bid":
        recipes.append(
            {
                "kind": "css",
                "selector": _build_bid_css_selector(raw_target),
                "bid": raw_target,
            }
        )
    else:
        normalized_target = _normalize_text(raw_target)
        recipes.extend(
            [
                {"kind": "role", "role": "link", "name": normalized_target, "exact": True},
                {"kind": "role", "role": "button", "name": normalized_target, "exact": True},
                {"kind": "text", "text": normalized_target, "exact": True},
                {"kind": "label", "label": normalized_target, "exact": True},
                {"kind": "role", "role": "textbox", "name": normalized_target, "exact": True},
                {"kind": "role", "role": "searchbox", "name": normalized_target, "exact": True},
                {"kind": "text", "text": normalized_target, "exact": False},
                {"kind": "label", "label": normalized_target, "exact": False},
                {"kind": "role", "role": "link", "name": normalized_target, "exact": False},
                {"kind": "role", "role": "button", "name": normalized_target, "exact": False},
                {"kind": "role", "role": "textbox", "name": normalized_target, "exact": False},
                {"kind": "role", "role": "searchbox", "name": normalized_target, "exact": False},
            ]
        )
        if _is_classifieds_url(current_url) and normalized_target.lower() in {"search", "??search"}:
            recipes.append({"kind": "css", "selector": "button[type='submit'], button.btn-primary"})

    return _first_live_recipe(page, recipes)


def _find_live_type_recipe(
    page: Any,
    target: str,
    value: str,
    target_mode: str,
    current_url: str,
) -> dict[str, Any] | None:
    del value

    raw_target = (target or "").strip()
    if not raw_target:
        return None

    recipes: list[dict[str, Any]] = []
    if target_mode == "bid":
        recipes.append(
            {
                "kind": "css",
                "selector": _build_bid_css_selector(raw_target),
                "bid": raw_target,
            }
        )
    else:
        normalized_target = _normalize_text(raw_target)
        recipes.extend(
            [
                {"kind": "role", "role": "combobox", "name": normalized_target, "exact": True},
                {"kind": "label", "label": normalized_target, "exact": True},
                {"kind": "placeholder", "placeholder": normalized_target, "exact": True},
                {"kind": "role", "role": "textbox", "name": normalized_target, "exact": True},
                {"kind": "role", "role": "searchbox", "name": normalized_target, "exact": True},
                {"kind": "label", "label": normalized_target, "exact": False},
                {"kind": "placeholder", "placeholder": normalized_target, "exact": False},
                {"kind": "role", "role": "textbox", "name": normalized_target, "exact": False},
                {"kind": "role", "role": "searchbox", "name": normalized_target, "exact": False},
            ]
        )
        if _is_classifieds_url(current_url):
            lower_target = normalized_target.lower()
            if lower_target == "select a category":
                recipes.append(
                    {
                        "kind": "css",
                        "selector": "select[name='sCategory'], select",
                        "force_select": True,
                    }
                )
            if lower_target == "e.g., a blue used car":
                recipes.append(
                    {
                        "kind": "css",
                        "selector": "input[name='sPattern'], input[type='search'], input[type='text']",
                    }
                )

    return _first_live_recipe(page, recipes)


def _first_live_recipe(page: Any, recipes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for recipe in recipes:
        try:
            locator = _locator_from_recipe(page, recipe)
            if locator.count() > 0:
                return recipe
        except Exception:
            continue
    return None


def _bid_locator_tag_name(page: Any, bid: str) -> str:
    raw_bid = (bid or "").strip()
    if not raw_bid:
        return ""
    try:
        locator = page.locator(_build_bid_css_selector(raw_bid))
        if locator.count() < 1:
            return ""
        return str(locator.first.evaluate("(el) => (el.tagName || '').toLowerCase()") or "")
    except Exception:
        return ""


def _locator_from_recipe(page: Any, recipe: dict[str, Any]) -> Any:
    kind = recipe.get("kind")
    if kind == "css":
        return page.locator(recipe["selector"])
    if kind == "role":
        return page.get_by_role(
            recipe["role"],
            name=recipe.get("name"),
            exact=bool(recipe.get("exact", False)),
        )
    if kind == "text":
        return page.get_by_text(recipe["text"], exact=bool(recipe.get("exact", False)))
    if kind == "label":
        return page.get_by_label(recipe["label"], exact=bool(recipe.get("exact", False)))
    if kind == "placeholder":
        return page.get_by_placeholder(recipe["placeholder"], exact=bool(recipe.get("exact", False)))
    raise ValueError(f"Unsupported live DOM locator kind: {kind}")


def _execute_live_click(page: Any, recipe: dict[str, Any]) -> None:
    locator = _locator_from_recipe(page, recipe)
    if locator.count() < 1:
        raise RuntimeError(f"Live DOM click target disappeared: {recipe}")

    target = locator.first
    try:
        target.scroll_into_view_if_needed(timeout=_LIVE_DOM_ACTION_TIMEOUT_MS)
    except Exception:
        pass
    try:
        target.click(timeout=_LIVE_DOM_ACTION_TIMEOUT_MS, no_wait_after=True)
        return
    except Exception:
        pass

    try:
        target.evaluate(
            """
(el) => {
  el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});
}
            """
        )
        page.wait_for_timeout(150)
    except Exception:
        pass

    try:
        target.click(timeout=_LIVE_DOM_ACTION_TIMEOUT_MS, no_wait_after=True, force=True)
        return
    except Exception:
        pass

    target.evaluate(
        """
(el) => {
  el.scrollIntoView({block: 'center', inline: 'center', behavior: 'instant'});
  el.click();
}
        """
    )


def _execute_live_type(page: Any, recipe: dict[str, Any], value: str) -> None:
    locator = _locator_from_recipe(page, recipe)
    if locator.count() < 1:
        raise RuntimeError(f"Live DOM input target disappeared: {recipe}")

    target = locator.first
    try:
        target.scroll_into_view_if_needed(timeout=_LIVE_DOM_ACTION_TIMEOUT_MS)
    except Exception:
        pass

    tag_name = ""
    try:
        tag_name = str(target.evaluate("(el) => el.tagName.toLowerCase()") or "")
    except Exception:
        pass

    if recipe.get("force_select") or tag_name == "select":
        try:
            target.select_option(label=value, timeout=_LIVE_DOM_ACTION_TIMEOUT_MS)
            return
        except Exception:
            pass

        option_value = None
        try:
            option_value = target.evaluate(
                """(el, desired) => {
                    const goal = (desired || "").trim().toLowerCase();
                    const match = Array.from(el.options || []).find((option) => {
                        const label = (option.label || option.text || "").trim().toLowerCase();
                        return label === goal || label.includes(goal) || goal.includes(label);
                    });
                    return match ? match.value : null;
                }""",
                value,
            )
        except Exception:
            option_value = None

        if option_value:
            target.select_option(value=str(option_value), timeout=_LIVE_DOM_ACTION_TIMEOUT_MS)
            return

        raise RuntimeError(f"Could not select option {value!r} for recipe {recipe}")

    target.fill(value, timeout=_LIVE_DOM_ACTION_TIMEOUT_MS)


def _sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    return str(value)
