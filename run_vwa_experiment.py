from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import re
import signal
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote_plus, urlparse

from env.action_parser import parse_action
from env.skill_utils import (
    allowed_patch_types_for_failure,
    build_translator_error_target_repair,
    build_contract_repair_prompt,
    build_skill_generation_prompt,
    build_text_rewrite_prompt,
    has_execution_equivalent_update,
    introduced_invalid_repair_targets,
    localize_failure,
    observation_contract_status,
    parse_skill_response,
    preserve_navigation_prefix_for_repair,
    repair_targets_against_observation,
    should_block_generic_repair_target,
    skill_to_action_string,
    stabilize_contractskill_miniwob_m3_skill,
    stabilize_contractskill_miniwob_skill,
    summarize_skill_diff,
)
from env.api_env import DEFAULT_QWEN_API_ENV_PATH, load_api_env_file
from env.vwa_env import VisualWebArenaEnv
from glm_client import GLMClient, ModelRequestError


DEFAULT_MODEL = "glm-4.6v"
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_REPAIRS = 5
DEFAULT_MAX_MODEL_CALLS = 6
DEFAULT_BROWSER_TIMEOUT_MS = 45_000
DEFAULT_RESET_TIMEOUT_SEC = 0
DEFAULT_STEP_TIMEOUT_SEC = 0
DEFAULT_RESET_RETRIES = 1
DEFAULT_TASK_TIMEOUT_SEC = 0
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = REPO_ROOT / "tasks" / "vwa_smoke_2.json"

OUTPUT_ROOT = REPO_ROOT / "outputs" / "vwa_experiments"
SCREENSHOT_ROOT = OUTPUT_ROOT / "screenshots"
TRACE_ROOT = OUTPUT_ROOT / "traces"
RESULT_ROOT = OUTPUT_ROOT / "results"
SKILL_ROOT = OUTPUT_ROOT / "skills"
INITIAL_SKILL_ROOT = OUTPUT_ROOT / "shared_initial_skills"
INITIAL_SKILL_CACHE_VERSION = "v3"
CANONICAL_SITES = ("classifieds", "reddit", "shopping")
CONSERVATIVE_REPAIR_MODELS: set[str] = set()

ACTION_SYSTEM_PROMPT = """You are a web agent acting inside BrowserGym VisualWebArena.
Given the task goal, the current page state, and the current page screenshot, output exactly one next action.

Allowed formats:
CLICK[text]
CLICK[bid=123]
TYPE[field=value]
TYPE[bid=123=value]
SCROLL[down]
STOP[answer]

Rules:
1. Output one action only.
2. CLICK[text] should use exact visible English text from the provided clickable targets whenever possible.
3. CLICK[bid=123] is preferred when a clickable target has a clear bid.
4. TYPE[field=value] should use the exact input label from the provided input field list whenever possible.
5. TYPE[bid=123=value] is allowed when the input has a clear bid.
6. For TYPE, do not invent prefixes such as field=, label=, or input=. Use the label itself unless you use bid=...
7. If the page provides an input label like Search query, write TYPE[Search query=your text].
8. Do not click generic navigation like Home unless it is clearly required by the task goal.
9. Use STOP[answer] only when the task is already complete or explicitly asks for a final textual answer.
10. Do not output Python code, explanations, or markdown. Output only the action string.
11. Never output Playwright or BrowserGym code such as fill(...), click(...), or LIVE_TYPE[...].
12. Never output bare numeric field names like TYPE[170=value]. If you use a number, it must be TYPE[bid=170=value].

Examples:
- Valid: TYPE[Search query=blue kayak]
- Valid: CLICK[bid=117]
- Valid: TYPE[bid=50=blue kayak]
- Invalid: fill("57", "spiderman")
- Invalid: LIVE_TYPE[Comment=hello]
- Invalid: TYPE[170=blue kayak]
- Invalid: TYPE[field=Search query]=blue kayak
- Invalid: TYPE[field=e.g., a blue used car]=blue kayak
"""

CURRENT_PAGE_ANSWER_SYSTEM_PROMPT = """You answer current-page visual web questions.
Use the provided goal, current page text, clickable list, input list, and screenshot.

Output exactly one line:
- the exact answer text only, if the answer is visible on the CURRENT page
- NOT_VISIBLE, if the answer is not visible on the CURRENT page

Rules:
1. Do not explain.
2. Do not click or plan.
3. Prefer extracting from visible page evidence, not guessing.
4. Preserve units when the goal asks for units.
"""

CONTRACT_REPAIR_BASELINES = {
    "contractskill",
    "contractskill_no_patch_constraints",
    "contractskill_no_failure_localization",
    "contractskill_unconstrained_repair",
}
PATCH_CONSTRAINED_BASELINES = {
    "contractskill",
    "contractskill_no_failure_localization",
}
FAILURE_AWARE_PATCH_POLICY_BASELINES = {
    "contractskill",
}
LOCALIZED_REPAIR_BASELINES = {
    "contractskill",
    "contractskill_no_patch_constraints",
    "contractskill_unconstrained_repair",
}
NAV_PREFIX_REPAIR_BASELINES = {
    "contractskill",
    "contractskill_no_patch_constraints",
}
M3_STABILIZED_BASELINES = {
    "contractskill",
    "contractskill_no_patch_constraints",
}

CLASSIFIEDS_RESULT_PICKER_SYSTEM_PROMPT = """You pick the single best visible classifieds listing for the task.

Output exactly one line:
- the exact visible listing title from the candidate list
- or NOT_VISIBLE if none of the visible candidates can satisfy the task

Rules:
1. Choose only from the provided visible listing titles.
2. Do not invent titles.
3. Prefer the cheapest candidate when the task asks for the cheapest item.
4. Use the screenshot to judge visual cues like cover color, grass, water, or whether an image matches the goal.
5. Respect simple constraints such as color, product type, and price range if visible in the candidate text.
"""

REDDIT_POST_PICKER_SYSTEM_PROMPT = """You pick the single best visible reddit click target for the task.

Output exactly one line:
- the exact visible click target from the candidate list
- or NOT_VISIBLE if none of the visible candidates clearly advances the task

Rules:
1. Choose only from the provided visible candidate targets.
2. Do not invent targets.
3. If the goal names a subreddit or sort order, assume that prefix navigation is already handled and focus on the best visible post-level target.
4. Prefer an exact full post title or a visible comment-count link over image thumbnails, usernames, or generic UI chrome.
5. If the goal asks for a photo/image post, use the screenshot and visible text to choose the best matching post.
"""

SHOPPING_CARD_PICKER_SYSTEM_PROMPT = """You pick the single best visible shopping product card for the task.

Output exactly one line:
- the exact visible product title from the candidate list
- or NOT_VISIBLE if none of the visible candidates clearly satisfies the task

Rules:
1. Choose only from the provided visible product titles.
2. Do not invent titles.
3. Use the screenshot to judge visual cues like color, shape, packaging, and row/column position.
4. Respect textual constraints such as stars, weight, red/yellow, round, or other visible attributes when possible.
5. Prefer the most directly matching visible product card over generic category links.
"""

REDDIT_GENERIC_CLICK_TARGETS = {
    "bans",
    "comments",
    "forums",
    "home",
    "moderation log",
    "notifications (0)",
    "submit",
    "submissions",
    "subscribe no subscribers",
    "subscribe via rss",
    "upvote",
    "downvote",
    "wiki",
}


def is_conservative_repair_model(model: str) -> bool:
    return str(model or "").strip().lower() in CONSERVATIVE_REPAIR_MODELS


def effective_repair_limits(model: str, max_repairs: int, max_model_calls: int) -> tuple[int, int]:
    if not is_conservative_repair_model(model):
        return max_repairs, max_model_calls
    return min(max_repairs, 1), min(max_model_calls, 3)


def primary_site_for_task(task_item: dict[str, Any]) -> str:
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    return sites[0] if len(sites) == 1 else ""


def preserve_successful_prefix_for_repair(
    current_skill: dict[str, Any],
    repaired_skill: dict[str, Any],
    failure_info: dict[str, Any],
) -> dict[str, Any]:
    failed_step_index = int(failure_info.get("failed_step_index") or 0)
    keep_count = max(0, failed_step_index - 1)
    if keep_count <= 0:
        return repaired_skill

    previous_steps = [dict(step) for step in current_skill.get("steps", []) or []]
    updated_steps = [dict(step) for step in repaired_skill.get("steps", []) or []]
    if keep_count > len(previous_steps):
        return repaired_skill
    if keep_count > len(updated_steps):
        merged = dict(repaired_skill)
        merged["steps"] = previous_steps[:keep_count] + updated_steps
        repair_history = list(merged.get("repair_history") or [])
        repair_history.append(f"preserved executed prefix through step {keep_count}")
        merged["repair_history"] = repair_history
        return merged
    if previous_steps[:keep_count] == updated_steps[:keep_count]:
        return repaired_skill

    merged = dict(repaired_skill)
    merged["steps"] = previous_steps[:keep_count] + updated_steps[keep_count:]
    repair_history = list(merged.get("repair_history") or [])
    repair_history.append(f"preserved executed prefix through step {keep_count}")
    merged["repair_history"] = repair_history
    return merged


def should_treat_navigation_timeout_as_success(
    task_item: dict[str, Any],
    *,
    previous_observation: dict[str, Any],
    next_observation: dict[str, Any],
    step_result: dict[str, Any],
    final_action_error: str,
    skill_step: dict[str, Any] | None = None,
) -> bool:
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if sites != ["shopping"]:
        return False
    if not _needs_shopping_product_page(str(task_item.get("intent", "") or "")):
        return False
    if skill_step is not None and str(skill_step.get("action") or "").upper() != "CLICK":
        return False

    error_text = " ".join(
        str(value or "")
        for value in (
            final_action_error,
            step_result.get("fail_reason"),
            step_result.get("last_action_error"),
        )
    ).lower()
    if "locator.click" not in error_text or "timeout" not in error_text:
        return False

    previous_url = str(previous_observation.get("url", "") or "")
    next_url = str(next_observation.get("url", "") or "")
    if not next_url or next_url == previous_url:
        return False
    if ".html" not in next_url.lower():
        return False

    target_text = str((skill_step or {}).get("target") or "").strip().lower()
    page_text = str(next_observation.get("page_text", "") or "").lower()
    if target_text and target_text not in page_text:
        return False
    return True


def should_treat_shopping_checkout_terminal_as_success(
    task_item: dict[str, Any],
    *,
    observation: dict[str, Any],
) -> bool:
    if primary_site_for_task(task_item) != "shopping":
        return False
    if str(task_item.get("task_family") or "") != "shopping_checkout":
        return False

    url = str(observation.get("url", "") or "").lower()
    page_text = str(observation.get("page_text", "") or "").lower()
    clickable_texts = {
        str(item.get("text") or "").strip().lower()
        for item in observation.get("clickable_elements") or []
        if str(item.get("text") or "").strip()
    }
    if "checkout/onepage/success" in url:
        return True
    if "thank you for your purchase" in page_text:
        return True
    return "print receipt" in clickable_texts


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def zero_usage() -> dict[str, float]:
    return {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "estimated_cost_usd": 0.0,
    }


def _to_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def estimate_cost_usd(prompt_tokens: float, completion_tokens: float) -> float:
    prompt_rate = _to_float(os.getenv("GLM_PROMPT_COST_PER_1K_USD"))
    completion_rate = _to_float(os.getenv("GLM_COMPLETION_COST_PER_1K_USD"))
    return (prompt_tokens / 1000.0) * prompt_rate + (completion_tokens / 1000.0) * completion_rate


def normalize_usage(usage: Any) -> dict[str, float]:
    if usage is None:
        prompt_tokens = 0.0
        completion_tokens = 0.0
        total_tokens = 0.0
    elif isinstance(usage, dict):
        prompt_tokens = _to_float(usage.get("prompt_tokens"))
        completion_tokens = _to_float(usage.get("completion_tokens"))
        total_tokens = _to_float(usage.get("total_tokens"))
    else:
        prompt_tokens = _to_float(getattr(usage, "prompt_tokens", 0))
        completion_tokens = _to_float(getattr(usage, "completion_tokens", 0))
        total_tokens = _to_float(getattr(usage, "total_tokens", 0))

    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost_usd(prompt_tokens, completion_tokens),
    }


def add_usage(target: dict[str, float], usage: dict[str, float]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"):
        target[key] = _to_float(target.get(key)) + _to_float(usage.get(key))


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return slug.strip("._-") or "default"


def skills_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right,
        sort_keys=True,
        ensure_ascii=False,
    )


def append_repair_history_note(skill: dict[str, Any], note: str) -> dict[str, Any]:
    text = str(note or "").strip()
    if not text:
        return skill

    updated = dict(skill)
    history = [str(item or "").strip() for item in updated.get("repair_history", [])]
    history = [item for item in history if item]
    if text not in history:
        history.append(text)
    updated["repair_history"] = history
    return updated


def detect_implicit_navigation_targets(
    skill: dict[str, Any],
    observation: dict[str, Any],
) -> list[str]:
    visible_clicks = {
        str(item.get("text") or "").strip().lower()
        for item in observation.get("clickable_elements") or []
        if str(item.get("text") or "").strip()
    }
    visible_tabs = {
        str(title or "").strip().lower()
        for title in observation.get("open_pages_titles") or []
        if str(title or "").strip()
    }
    forbidden = {"back", "forward", "refresh"}
    violations: list[str] = []
    for step in skill.get("steps", []) or []:
        if str(step.get("action") or "").upper() != "CLICK":
            continue
        target = str(step.get("target") or "").strip()
        lowered = target.lower()
        if lowered in forbidden and lowered not in visible_clicks and lowered not in visible_tabs:
            violations.append(target)
    return violations


def parse_bool(text: str) -> bool:
    value = text.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {text}")


class EnvCallTimeoutError(RuntimeError):
    pass


def _run_with_timeout(timeout_sec: int, label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    timeout = max(0, int(timeout_sec or 0))
    if timeout <= 0 or not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        return fn(*args, **kwargs)

    def _handle_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise EnvCallTimeoutError(f"{label} timed out after {timeout} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def reset_env_with_watchdog(
    env: VisualWebArenaEnv,
    env_name: str,
    *,
    task_key: str,
    seed: int,
    reset_timeout_sec: int,
    reset_retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    attempts = max(1, int(reset_retries or 0) + 1)

    for attempt_index in range(1, attempts + 1):
        try:
            return _run_with_timeout(
                reset_timeout_sec,
                f"env.reset({env_name})",
                env.reset,
                env_name,
                task_key=task_key,
                seed=seed,
            )
        except Exception as exc:
            last_error = exc
            env.close()
            if attempt_index >= attempts:
                break
            print(
                f"[watchdog] reset failed for {task_key} "
                f"(attempt {attempt_index}/{attempts}): {exc}"
            )

    raise RuntimeError(str(last_error) if last_error is not None else f"env.reset({env_name}) failed")


def step_env_with_watchdog(
    env: VisualWebArenaEnv,
    translation: dict[str, Any],
    *,
    task_key: str,
    state_index: int,
    step_timeout_sec: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _run_with_timeout(
            step_timeout_sec,
            f"env.step({task_key}#{state_index})",
            env.step,
            translation,
            task_key=task_key,
            state_index=state_index,
        )
    except Exception:
        env.close()
        raise


def load_split(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Split file must contain a non-empty list.")

    required = ("task_id", "env_name", "notes")
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Each split item must be a JSON object.")
        for key in required:
            if not str(item.get(key, "") or "").strip():
                raise ValueError(f"Each split item must contain non-empty {key!r}.")
    return payload


def encode_image_as_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def trim_text(text: str, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def summarize_clickables(observation: dict[str, Any]) -> str:
    items = observation.get("clickable_elements") or []
    if not items:
        return "none"
    lines = []
    for item in items[:40]:
        lines.append(f"{item.get('text', '')} (bid={item.get('bid', '')}, role={item.get('role', '')})")
    return "\n".join(lines)


def summarize_inputs(observation: dict[str, Any]) -> str:
    fields = observation.get("input_fields") or []
    if not fields:
        return "none"

    lines = []
    for field in fields[:30]:
        lines.append(
            f"label={field.get('label', '')}; type={field.get('type', '')}; bid={field.get('bid', '')}"
        )
    return "\n".join(lines)


def is_current_page_answer_task(task_item: dict[str, Any], observation: dict[str, Any]) -> bool:
    notes = str(task_item.get("notes", "") or "").lower()
    goal = str(observation.get("goal", "") or "").strip().lower()
    if "current-page answer task" in notes:
        return True
    if goal.startswith(("what is", "what are", "how many", "how much", "which price range")):
        return True
    if any(token in goal for token in ("on this page", "in this page", "current page")):
        return any(
            marker in goal
            for marker in (" what is", " what are", " how many", " how much", " which ")
        )
    return False


def is_visual_current_page_comparison_task(
    task_item: dict[str, Any],
    observation: dict[str, Any],
) -> bool:
    goal = str(observation.get("goal", "") or task_item.get("intent", "") or "").strip().lower()
    if not goal:
        return False
    if not any(token in goal for token in ("image", "picture", "photo")):
        return False
    if not any(token in goal for token in ("on this page", "in this page", "current page", "all post", "all posts")):
        return False
    return any(
        token in goal
        for token in ("largest", "smallest", "biggest", "least", "most", "takes up", "proportion", "closest", "furthest")
    )


def needs_visual_current_page_exploration(goal_text: str) -> bool:
    goal_lc = str(goal_text or "").lower()
    return any(
        marker in goal_lc
        for marker in ("takes up", "largest proportion", "smallest proportion", "largest", "smallest", "most of the image", "least of the image")
    )


def normalize_current_page_answer(goal_text: str, answer_text: str) -> str:
    text = str(answer_text or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    for prefix in ("answer:", "final answer:", "the answer is", "it is"):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" :.-")
            lowered = text.lower()

    text = text.strip().strip("\"'").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"^[\-\u2022]\s*", "", text)

    if not text:
        return ""

    goal_lc = str(goal_text or "").lower()
    answer_lc = text.lower()

    if "in inches" in goal_lc and re.fullmatch(r"\d+(?:\.\d+)?", text):
        text = f"{text} inches"
        answer_lc = text.lower()
    elif " in inches" in goal_lc and answer_lc.endswith(" inch"):
        text = text + "es"
        answer_lc = text.lower()

    if "price range" in goal_lc:
        text = re.sub(r"\s*[---]\s*", " to ", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def is_not_visible_answer(answer_text: str) -> bool:
    lowered = str(answer_text or "").strip().lower()
    return lowered in {
        "",
        "not_visible",
        "not visible",
        "not visible on current page",
        "not available",
        "information not available",
        "unknown",
        "cannot determine",
    }


def is_classifieds_search_task(task_item: dict[str, Any], observation: dict[str, Any] | None = None) -> bool:
    sites = list(task_item.get("sites", []) or [])
    if sites != ["classifieds"]:
        return False
    if task_item.get("task_family") != "information_seeking":
        return False
    if observation is not None and is_current_page_answer_task(task_item, observation):
        return False
    notes = str(task_item.get("notes", "") or "").lower()
    if "classifieds search task" in notes:
        return True
    goal = str((observation or {}).get("goal") or task_item.get("intent", "") or "").lower()
    return any(token in goal for token in ("cheapest", "find me", "show me"))


def is_visual_classifieds_goal(goal_text: str) -> bool:
    goal_lc = str(goal_text or "").lower()
    visual_markers = (
        "image",
        "picture",
        "cover",
        "photo",
        "on water",
        "on grass",
        "in a basket",
        "shirt",
    )
    return any(marker in goal_lc for marker in visual_markers)


def needs_visual_classifieds_exploration(goal_text: str) -> bool:
    goal_lc = str(goal_text or "").lower()
    return any(
        marker in goal_lc
        for marker in ("on grass", "on water", "in a basket", "next to", "beside", "background", "taken during", "on the cover", "set on")
    )


def extract_classifieds_query(goal_text: str, observation: dict[str, Any] | None = None) -> str:
    url = str((observation or {}).get("url", "") or "")
    if url:
        parsed = urlparse(url)
        query_values = parse_qs(parsed.query).get("sPattern")
        if query_values:
            return unquote_plus(query_values[0]).strip()

    text = str(goal_text or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    patterns = (
        r"find me the cheapest (.+?)(?: on this site)?(?:\.|$)",
        r"show me the cheapest (.+?)(?: on this site)?(?:\.|$)",
        r"find the cheapest (.+?)(?: on this site)?(?:\.|$)",
        r"show the cheapest (.+?)(?: on this site)?(?:\.|$)",
    )
    candidate = ""
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            candidate = match.group(1)
            break
    if not candidate:
        candidate = lowered

    candidate = re.sub(r"\bit should be between\b.*$", "", candidate)
    candidate = re.sub(r"\bbetween\b.*$", "", candidate)
    candidate = re.sub(r"\bpriced between\b.*$", "", candidate)
    candidate = re.sub(r"\bwith\b", " ", candidate)
    candidate = re.sub(r"[^a-z0-9$ ]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def find_search_field_label(observation: dict[str, Any]) -> str | None:
    fields = observation.get("input_fields") or []
    if not fields:
        return None
    for field in fields:
        label = str(field.get("label", "") or "").strip()
        if label:
            return label
    return None


def find_search_click_target(observation: dict[str, Any]) -> str | None:
    clickables = observation.get("clickable_elements") or []
    for item in clickables:
        text = str(item.get("text", "") or "").strip()
        if "search" in text.lower():
            bid = str(item.get("bid", "") or "").strip()
            return f"bid={bid}" if bid else text
    return None


def collect_classifieds_listing_titles(observation: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in observation.get("clickable_elements") or []:
        role = str(item.get("role", "") or "").strip().lower()
        if role != "link":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        if text.isdigit():
            continue
        lowered = text.lower()
        if lowered in {
            "classifieds",
            "my account",
            "logout",
            "publish ad",
            "subscribe now!",
            "all categories",
            "antiques",
            "appliances",
            "arts + crafts",
            "auto parts",
            "beauty + health",
            "bikes",
            "boats",
            "books",
            "apply",
            "contact",
            "listings with pictures",
            "search",
            "??search",
        }:
            continue
        if len(text) < 4:
            continue
        if text not in seen:
            seen.add(text)
            candidates.append(text)
    return candidates


CLASSIFIEDS_SEARCH_STOPWORDS = {
    "the",
    "a",
    "an",
    "me",
    "show",
    "find",
    "cheapest",
    "most",
    "recently",
    "posted",
    "with",
    "between",
    "and",
    "this",
    "site",
    "it",
    "should",
    "be",
}

CLASSIFIEDS_ACCESSORY_TERMS = {
    "mats",
    "jackers",
    "seat",
    "floor",
    "trunk",
    "kit",
    "kits",
    "parts",
    "accessory",
    "accessories",
}

CLASSIFIEDS_COLOR_TERMS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "silver",
    "gray",
    "grey",
    "orange",
    "purple",
    "brown",
    "gold",
    "pink",
}

CLASSIFIEDS_PAGINATION_PREV_TOKENS = {"<", "<<", "prev", "previous"}
CLASSIFIEDS_PAGINATION_NEXT_TOKENS = {">", ">>", "next"}


def _tokenize_classifieds_query(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [token for token in tokens if token not in CLASSIFIEDS_SEARCH_STOPWORDS]


def extract_classifieds_focus_tokens(goal_text: str, query_text: str) -> list[str]:
    focus_tokens: list[str] = []
    seen: set[str] = set()
    for token in _tokenize_classifieds_query(query_text) or _tokenize_classifieds_query(goal_text):
        if token in CLASSIFIEDS_COLOR_TERMS or token in CLASSIFIEDS_ACCESSORY_TERMS:
            continue
        if token in {"used", "new", "sale", "posted", "listing", "listings"}:
            continue
        if token in seen:
            continue
        seen.add(token)
        focus_tokens.append(token)
    return focus_tokens


def extract_classifieds_focus_phrase(goal_text: str, query_text: str) -> str:
    return " ".join(extract_classifieds_focus_tokens(goal_text, query_text)).strip()


def _count_classifieds_token_matches(tokens: list[str], text: str) -> int:
    lowered = f" {(text or '').lower()} "
    return sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", lowered))


def order_classifieds_pagination_targets(targets: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_target in targets:
        target = str(raw_target or "").strip()
        if not target or target in seen:
            continue
        seen.add(target)
        deduped.append(target)

    def sort_key(target: str) -> tuple[int, int, str]:
        lowered = target.lower()
        if lowered in CLASSIFIEDS_PAGINATION_PREV_TOKENS:
            return (3, 0, lowered)
        if target.isdigit():
            return (0, int(target), lowered)
        if lowered in CLASSIFIEDS_PAGINATION_NEXT_TOKENS:
            return (1, 0, lowered)
        if len(target) <= 2 and not any(ch.isalnum() for ch in target):
            return (1, 1, lowered)
        return (2, 0, lowered)

    return [target for target in sorted(deduped, key=sort_key) if target.lower() not in CLASSIFIEDS_PAGINATION_PREV_TOKENS]


def build_classifieds_query_variants(goal_text: str, observation: dict[str, Any] | None = None) -> list[str]:
    exact = extract_classifieds_query(goal_text, observation)
    if not exact:
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def add_variant(value: str) -> None:
        normalized = re.sub(r"\s+", " ", (value or "").strip().lower())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        variants.append(normalized)

    add_variant(exact)

    exact_tokens = _tokenize_classifieds_query(exact)
    stripped = [
        token
        for token in exact_tokens
        if token not in CLASSIFIEDS_COLOR_TERMS and token not in CLASSIFIEDS_ACCESSORY_TERMS
    ]
    add_variant(" ".join(stripped))

    noun_only = [
        token
        for token in stripped
        if token not in {"toyota", "honda", "ford", "bmw", "nissan", "chevy", "chevrolet"}
    ]
    if noun_only:
        add_variant(" ".join(noun_only))

    if "toyota" in exact_tokens:
        add_variant("toyota")
    if "bike" in exact_tokens or "bicycle" in exact_tokens:
        add_variant("bike")
        add_variant("bicycle")
    if "kayak" in exact_tokens:
        add_variant("kayak")

    return variants


def _parse_classifieds_price(text: str) -> float | None:
    match = re.search(r"\$?\s*(\d[\d,]*)(?:\.(\d{2}))?", text or "")
    if not match:
        return None
    integer_part = match.group(1).replace(",", "")
    fractional_part = match.group(2) or "00"
    try:
        return float(f"{integer_part}.{fractional_part}")
    except ValueError:
        return None


def extract_classifieds_price_bounds(goal_text: str) -> tuple[float | None, float | None]:
    lowered = (goal_text or "").lower()
    between_match = re.search(
        r"(?:between|from)\s*\$?([\d,]+(?:\.\d{1,2})?)\s*(?:to|-)\s*\$?([\d,]+(?:\.\d{1,2})?)",
        lowered,
    )
    if between_match:
        low = _parse_classifieds_price(between_match.group(1))
        high = _parse_classifieds_price(between_match.group(2))
        return low, high

    minimum = None
    maximum = None

    at_least_match = re.search(r"(?:at least|more than|over|above)\s*\$?([\d,]+(?:\.\d{1,2})?)", lowered)
    if at_least_match:
        minimum = _parse_classifieds_price(at_least_match.group(1))

    at_most_match = re.search(r"(?:at most|less than|under|below|no more than)\s*\$?([\d,]+(?:\.\d{1,2})?)", lowered)
    if at_most_match:
        maximum = _parse_classifieds_price(at_most_match.group(1))

    return minimum, maximum


def extract_classifieds_goal_colors(goal_text: str) -> set[str]:
    lowered = (goal_text or "").lower()
    return {
        color
        for color in CLASSIFIEDS_COLOR_TERMS
        if re.search(rf"\b{re.escape(color)}\b", lowered)
    }


def is_classifieds_cheapest_goal(goal_text: str) -> bool:
    lowered = (goal_text or "").lower()
    return "cheapest" in lowered or "lowest price" in lowered


def get_live_vwa_page(env: VisualWebArenaEnv) -> Any | None:
    if getattr(env, "env", None) is None:
        return None
    try:
        return env.env.unwrapped.page
    except Exception:
        return None


def collect_live_classifieds_search_candidates(env: VisualWebArenaEnv) -> list[dict[str, Any]]:
    page = get_live_vwa_page(env)
    if page is None:
        return []

    try:
        items = page.evaluate(
            """
() => {
  const cardSelectors = [
    'li.listing-card',
    '.listing-card',
    '.products .item',
    '.search-list .item',
    'article.listing-card',
    '.latest-listings .item',
  ];
  const cards = cardSelectors.flatMap((selector) => [...document.querySelectorAll(selector)]);
  const fallbackCards = [...document.querySelectorAll('a[href*="page=item&id="]')]
    .map((link) => link.closest('li, article, div') || link);
  const nodes = cards.length ? cards : fallbackCards;
  const deduped = [...new Set(nodes)];
    const seenHrefs = new Set();
    const rows = [];

  for (const node of deduped) {
    const link = node.matches?.('a[href*="page=item&id="]')
      ? node
      : node.querySelector?.('a[href*="page=item&id="]');
    if (!link) {
      continue;
    }
    const href = String(link.href || '').trim();
    if (!href || seenHrefs.has(href)) {
      continue;
    }
    seenHrefs.add(href);

    const rawText = String(node.innerText || '').trim();
    const text = rawText.replace(/\\s+/g, ' ').trim();
    const rawTitle = String(link.innerText || link.textContent || '').trim() || rawText;
    const title = rawTitle.split(/\\n+/)[0].replace(/\\s+/g, ' ').trim() || text;
    if (!title) {
      continue;
    }

    const priceNode = node.querySelector?.(
      '.price, .currency-value, .item-price, .price__amount, strong.price, span.price'
    );
    const priceText =
      String(priceNode?.textContent || '').replace(/\\s+/g, ' ').trim() ||
      (text.match(/\\$\\s?[\\d,]+(?:\\.\\d{2})?/) || [''])[0];
    const rect = node.getBoundingClientRect();
    rows.push({
      title,
      href,
      text,
      raw_title: rawTitle.replace(/\\s+/g, ' ').trim(),
      price_text: String(priceText || '').trim(),
      top: rect.top,
      left: rect.left,
      visible: rect.bottom > 0 && rect.top < window.innerHeight * 1.5,
    });
  }

  rows.sort((a, b) => (a.top - b.top) || (a.left - b.left));
  return rows;
}
            """
        )
    except Exception:
        return []

    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def collect_live_classifieds_pagination_targets(env: VisualWebArenaEnv) -> list[str]:
    page = get_live_vwa_page(env)
    if page is None:
        return []

    try:
        items = page.evaluate(
            """
() => {
  const selectors = [
    '.pagination a',
    '.paginate a',
    '.searchPagination a',
    'a[href*="page=search"][href*="iPage="]',
    'a[href*="page=search"][href*="sPage="]',
  ];
  const nodes = selectors.flatMap((selector) => [...document.querySelectorAll(selector)]);
  const seen = new Set();
  const rows = [];
  for (const node of nodes) {
    const text = String(node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    rows.push(text);
  }
  return rows;
}
            """
        )
    except Exception:
        return []

    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def collect_live_classifieds_sort_targets(env: VisualWebArenaEnv) -> list[str]:
    page = get_live_vwa_page(env)
    if page is None:
        return []

    try:
        items = page.evaluate(
            """
() => {
  const selectors = [
    'a[href*="page=search"][href*="sOrder="]',
    '.sort a',
    '.sorting a',
    '.searchFilters a',
  ];
  const nodes = selectors.flatMap((selector) => [...document.querySelectorAll(selector)]);
  const seen = new Set();
  const rows = [];
  for (const node of nodes) {
    const text = String(node.textContent || '').replace(/\\s+/g, ' ').trim();
    if (!text || seen.has(text)) {
      continue;
    }
    seen.add(text);
    rows.push(text);
  }
  return rows;
}
            """
        )
    except Exception:
        return []

    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def extract_live_classifieds_item_details(env: VisualWebArenaEnv) -> dict[str, Any]:
    page = get_live_vwa_page(env)
    if page is None:
        return {}

    try:
        payload = page.evaluate(
            """
() => {
  const bodyText = String(document.body?.innerText || '').replace(/\\s+/g, ' ').trim();
  const titleNode = document.querySelector('h1, .item-title, .title h1, .name h1, .name, [itemprop="name"]');
  const priceNode = document.querySelector('.price, .currency-value, .item-price, #price, strong.price, [itemprop="price"]');
  const descNode = document.querySelector('#description, .description, .item-description, [itemprop="description"]');
  const breadcrumb = [...document.querySelectorAll('.breadcrumb a, .breadcrumb li, .meta a, .category a')]
    .map((node) => String(node.textContent || '').replace(/\\s+/g, ' ').trim())
    .filter(Boolean);
  return {
    url: location.href,
    title: String(titleNode?.innerText || '').replace(/\\s+/g, ' ').trim(),
    price_text: String(priceNode?.innerText || '').replace(/\\s+/g, ' ').trim(),
    description: String(descNode?.innerText || '').replace(/\\s+/g, ' ').trim(),
    breadcrumb,
    text: bodyText.slice(0, 5000),
  };
}
            """
        )
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def align_classifieds_listing_title_to_clickables(title: str, observation: dict[str, Any]) -> str:
    normalized_title = str(title or "").strip()
    if not normalized_title:
        return normalized_title

    clickable_titles = collect_classifieds_listing_titles(observation)
    if not clickable_titles:
        return normalized_title

    title_lc = normalized_title.lower()
    exact = next((item for item in clickable_titles if item == normalized_title), None)
    if exact:
        return exact

    exact_ci = next((item for item in clickable_titles if item.lower() == title_lc), None)
    if exact_ci:
        return exact_ci

    ranked = sorted(
        clickable_titles,
        key=lambda item: (
            item.lower().startswith(title_lc) or title_lc.startswith(item.lower()),
            title_lc in item.lower() or item.lower() in title_lc,
            len(set(_tokenize_classifieds_query(item)).intersection(_tokenize_classifieds_query(normalized_title))),
            -abs(len(item) - len(normalized_title)),
        ),
        reverse=True,
    )
    best = ranked[0]
    best_tokens = set(_tokenize_classifieds_query(best))
    title_tokens = set(_tokenize_classifieds_query(normalized_title))
    if best_tokens and title_tokens and best_tokens.intersection(title_tokens):
        return best
    return normalized_title


def score_classifieds_listing(goal_text: str, query_text: str, title: str) -> int:
    title_lc = title.lower()
    title_tokens = set(_tokenize_classifieds_query(title_lc))
    query_tokens = _tokenize_classifieds_query(query_text)
    goal_tokens = _tokenize_classifieds_query(goal_text)
    focus_tokens = extract_classifieds_focus_tokens(goal_text, query_text)
    focus_phrase = extract_classifieds_focus_phrase(goal_text, query_text)
    normalized_title = " ".join(_tokenize_classifieds_query(title_lc)).strip()
    score = 0

    for token in query_tokens:
        if token in title_tokens or token in title_lc:
            score += 5
    for token in goal_tokens:
        if token in title_tokens or token in title_lc:
            score += 2

    focus_title_matches = _count_classifieds_token_matches(focus_tokens, title_lc)
    score += 8 * focus_title_matches
    if focus_tokens and focus_title_matches == 0:
        score -= 18
    if focus_phrase and normalized_title == focus_phrase:
        score += 18
    elif focus_phrase and normalized_title.startswith(focus_phrase + " "):
        score += 6

    if any(term in title_lc for term in CLASSIFIEDS_ACCESSORY_TERMS):
        score -= 8

    if "toyota" in goal_text.lower():
        if "yaris" in title_lc:
            score += 4
        if "corolla" in title_lc:
            score += 3
        if "tacoma" in title_lc or "4runner" in title_lc:
            score -= 2
        if "prius" in title_lc:
            score -= 1

    if "bike" in goal_text.lower():
        if "bike" in title_lc or "bicycle" in title_lc:
            score += 4

    if "kayak" in goal_text.lower():
        if "kayak" in title_lc:
            score += 6
        if "aquarium" in title_lc or "sofa" in title_lc:
            score -= 6

    return score


def score_classifieds_candidate(
    goal_text: str,
    query_text: str,
    *,
    title: str,
    detail_text: str = "",
    price_text: str = "",
) -> int:
    title_lc = title.lower()
    combined_text = f"{title} {detail_text} {price_text}".strip().lower()
    score = score_classifieds_listing(goal_text, query_text, title)
    focus_tokens = extract_classifieds_focus_tokens(goal_text, query_text)

    for token in _tokenize_classifieds_query(query_text):
        if token in combined_text and token not in title_lc:
            score += 2
    for token in _tokenize_classifieds_query(goal_text):
        if token in combined_text and token not in title_lc:
            score += 1

    focus_combined_matches = _count_classifieds_token_matches(focus_tokens, combined_text)
    focus_title_matches = _count_classifieds_token_matches(focus_tokens, title_lc)
    if focus_combined_matches > focus_title_matches:
        score += 3 * (focus_combined_matches - focus_title_matches)
    if focus_tokens and focus_combined_matches == 0:
        score -= 24

    required_colors = extract_classifieds_goal_colors(goal_text)
    if required_colors:
        matched_colors = {color for color in required_colors if re.search(rf"\b{re.escape(color)}\b", combined_text)}
        score += 5 * len(matched_colors)
        conflicting_colors = {
            color
            for color in CLASSIFIEDS_COLOR_TERMS
            if color not in required_colors and re.search(rf"\b{re.escape(color)}\b", combined_text)
        }
        if conflicting_colors and not matched_colors:
            score -= 4

    if "handlebars" in goal_text.lower():
        if "handlebar" in combined_text:
            score += 8
        else:
            score -= 4

    low_price, high_price = extract_classifieds_price_bounds(goal_text)
    candidate_price = _parse_classifieds_price(price_text) or _parse_classifieds_price(detail_text)
    if candidate_price is not None:
        if low_price is not None and candidate_price < low_price:
            score -= 25
        if high_price is not None and candidate_price > high_price:
            score -= 25
        if low_price is not None and high_price is not None and low_price <= candidate_price <= high_price:
            score += 12
    elif low_price is not None or high_price is not None:
        score -= 4

    return score


def rank_live_classifieds_candidates(
    goal_text: str,
    query_text: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    goal_lc = (goal_text or "").lower()
    deduped: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for candidate in candidates:
        title = str(candidate.get("title", "") or "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        price = _parse_classifieds_price(str(candidate.get("price_text", "") or ""))
        score = score_classifieds_candidate(
            goal_text,
            query_text,
            title=title,
            detail_text=str(candidate.get("text", "") or ""),
            price_text=str(candidate.get("price_text", "") or ""),
        )
        enriched = dict(candidate)
        enriched["title"] = title
        enriched["price"] = price
        enriched["score"] = score
        deduped.append(enriched)

    def sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        score = -int(candidate.get("score", 0) or 0)
        price = candidate.get("price")
        top = float(candidate.get("top", 0.0) or 0.0)
        title = str(candidate.get("title", "") or "").lower()
        if "cheapest" in goal_lc:
            return (
                score,
                0 if price is not None else 1,
                price if price is not None else float("inf"),
                top,
                title,
            )
        if "most recently posted" in goal_lc:
            return (score, top, title)
        return (score, top, title)

    deduped.sort(key=sort_key)
    return [candidate for candidate in deduped if int(candidate.get("score", 0) or 0) > 0]


def analyze_classifieds_item_page(goal_text: str, query_text: str, item_details: dict[str, Any]) -> dict[str, Any]:
    title = str(item_details.get("title", "") or "").strip()
    detail_text = " ".join(
        part
        for part in (
            title,
            str(item_details.get("description", "") or ""),
            str(item_details.get("text", "") or ""),
            " ".join(str(item) for item in item_details.get("breadcrumb") or []),
        )
        if part
    )
    price_text = str(item_details.get("price_text", "") or "")
    score = score_classifieds_candidate(
        goal_text,
        query_text,
        title=title,
        detail_text=detail_text,
        price_text=price_text,
    )

    reasons: list[str] = []
    combined_text = detail_text.lower()
    focus_tokens = extract_classifieds_focus_tokens(goal_text, query_text)
    if focus_tokens:
        missing_focus = [token for token in focus_tokens if not re.search(rf"\b{re.escape(token)}\b", combined_text)]
        if len(missing_focus) == len(focus_tokens):
            reasons.append("missing_focus_tokens")
    required_colors = extract_classifieds_goal_colors(goal_text)
    for color in sorted(required_colors):
        if not re.search(rf"\b{re.escape(color)}\b", combined_text):
            reasons.append(f"missing_color:{color}")

    if "handlebars" in goal_text.lower() and "handlebar" not in combined_text:
        reasons.append("missing_handlebars")

    low_price, high_price = extract_classifieds_price_bounds(goal_text)
    price = _parse_classifieds_price(price_text) or _parse_classifieds_price(detail_text)
    if price is None and (low_price is not None or high_price is not None):
        reasons.append("price_not_visible")
    elif price is not None:
        if low_price is not None and price < low_price:
            reasons.append("price_too_low")
        if high_price is not None and price > high_price:
            reasons.append("price_too_high")

    if any(term in combined_text for term in CLASSIFIEDS_ACCESSORY_TERMS):
        reasons.append("accessory_like")

    if score <= 0:
        reasons.append("weak_goal_match")

    return {
        "title": title,
        "score": score,
        "price": price,
        "reasons": reasons,
    }


def choose_classifieds_listing_title(goal_text: str, query_text: str, titles: list[str]) -> str | None:
    if not titles:
        return None
    ranked = sorted(
        ((score_classifieds_listing(goal_text, query_text, title), title) for title in titles),
        key=lambda item: (item[0], -len(item[1])),
        reverse=True,
    )
    best_score, best_title = ranked[0]
    if best_score <= 0:
        return None
    return best_title


def rank_classifieds_listing_titles(goal_text: str, query_text: str, titles: list[str]) -> list[str]:
    ranked = sorted(
        ((score_classifieds_listing(goal_text, query_text, title), title) for title in titles),
        key=lambda item: (item[0], -len(item[1])),
        reverse=True,
    )
    return [title for score, title in ranked if score > 0]


def build_classifieds_search_skill(
    task_item: dict[str, Any],
    *,
    query: str,
    search_field_label: str,
    search_click_target: str,
    category_value: str | None = None,
    listing_title: str | None = None,
    post_search_click_targets: list[str] | None = None,
    stop_value: str | None = None,
    history_tags: list[str] | None = None,
    scroll_count_after_search: int = 0,
    include_stop: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    if category_value:
        steps.append({"action": "TYPE", "target": "Select a category", "value": category_value})
    steps.extend(
        [
            {"action": "TYPE", "target": search_field_label, "value": query},
            {"action": "CLICK", "target": search_click_target, "value": None},
        ]
    )
    for _ in range(max(0, int(scroll_count_after_search))):
        steps.append({"action": "SCROLL", "target": None, "value": "down"})
    for click_target in post_search_click_targets or []:
        if click_target:
            steps.append({"action": "CLICK", "target": click_target, "value": None})
    if listing_title:
        steps.append({"action": "CLICK", "target": listing_title, "value": None})
    if include_stop:
        final_stop_value = stop_value or listing_title or "Task completed"
        steps.append({"action": "STOP", "target": None, "value": final_stop_value})
    repair_history = ["heuristic_classifieds_search_fallback"]
    if history_tags:
        repair_history.extend(history_tags)
    return {
        "skill_id": f"{task_item['task_id']}_classifieds_search_fallback",
        "task": task_item.get("intent", task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": steps,
        "repair_history": repair_history,
        "patches": [],
    }


def extract_existing_classifieds_listing_click(
    current_skill: dict[str, Any],
    *,
    search_click_target: str,
) -> str | None:
    steps = current_skill.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).upper() != "CLICK":
            continue
        target = str(step.get("target", "") or "").strip()
        if not target:
            continue
        target_lc = target.lower()
        if target == search_click_target:
            continue
        if target_lc in {"select a category", "classifieds", "my account", "logout", "publish ad"}:
            continue
        if target_lc.startswith("bid="):
            continue
        return target
    return None


def infer_classifieds_category(goal_text: str) -> str | None:
    goal = (goal_text or "").lower()
    if any(token in goal for token in ("kayak", "boat", "canoe")):
        return "Boats"
    if any(token in goal for token in ("bike", "bicycle", "handlebars")):
        return "Bikes"
    if any(token in goal for token in ("toyota", "honda", "ford", "chevy", "chevrolet", "car", "truck", "suv")):
        return "Cars + trucks"
    return None


def extract_current_skill_search_query(current_skill: dict[str, Any], search_field_label: str) -> str | None:
    for step in current_skill.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if str(step.get("action", "")).upper() != "TYPE":
            continue
        target = str(step.get("target", "") or "").strip()
        if target != search_field_label:
            continue
        value = str(step.get("value", "") or "").strip()
        if value:
            return value
    return None


def extract_classifieds_history_tags(current_skill: dict[str, Any], prefix: str) -> list[str]:
    tags: list[str] = []
    for entry in current_skill.get("repair_history") or []:
        if not isinstance(entry, str):
            continue
        if entry.startswith(prefix):
            tags.append(entry[len(prefix) :])
    return tags


def extract_classifieds_scroll_count(current_skill: dict[str, Any], query: str) -> int:
    count = 0
    for entry in current_skill.get("repair_history") or []:
        if not isinstance(entry, str):
            continue
        if entry == f"classf_scroll:{query}":
            count += 1
            continue
        if entry.startswith(f"classf_scroll:{query}#"):
            count += 1
    return count


def extract_classifieds_pool_union(current_skill: dict[str, Any]) -> list[str]:
    union: list[str] = []
    seen: set[str] = set()
    for entry in current_skill.get("repair_history") or []:
        if not isinstance(entry, str) or not entry.startswith("classf_pool:"):
            continue
        payload = entry[len("classf_pool:") :]
        for item in payload.split(" || "):
            title = item.strip()
            if not title or title in seen:
                continue
            seen.add(title)
            union.append(title)
    return union


def extract_classifieds_pagination_targets(current_skill: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for entry in current_skill.get("repair_history") or []:
        if not isinstance(entry, str) or not entry.startswith("classf_pages:"):
            continue
        payload = entry[len("classf_pages:") :]
        for item in payload.split(" || "):
            target = item.strip()
            if not target or target in seen:
                continue
            seen.add(target)
            targets.append(target)
    return targets


def extract_classifieds_nav_targets(current_skill: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for target in extract_classifieds_history_tags(current_skill, "classf_navpick:"):
        normalized = str(target or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        targets.append(normalized)
    return targets


def extract_reddit_subreddit(goal_text: str) -> str | None:
    match = re.search(r"(?:/|\b)f/([A-Za-z0-9_]+)", goal_text or "", flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


def is_reddit_navigation_task(task_item: dict[str, Any], observation: dict[str, Any] | None = None) -> bool:
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if "reddit" not in sites:
        return False
    family = str(task_item.get("task_family", "") or "").lower()
    goal_text = str(task_item.get("intent", "") or "")
    if family == "navigation":
        return True
    return "comments section" in goal_text.lower() or "/f/" in goal_text.lower()


def is_reddit_image_like_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(token in lowered for token in ("/submission_images/", ".jpg", ".jpeg", ".png", ".gif", ".webp"))


def collect_reddit_post_title_candidates(observation: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in observation.get("clickable_elements") or []:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        if lowered in REDDIT_GENERIC_CLICK_TARGETS:
            continue
        if lowered.startswith("sort by:") or lowered.startswith("from:"):
            continue
        if re.fullmatch(r"\d+\s+comments?", lowered):
            continue
        if len(text) < 12:
            continue
        if " " not in text and text.lower() == text and all(ch.isalnum() or ch in "_-" for ch in text):
            continue
        seen.add(lowered)
        candidates.append(text)
    return candidates


def collect_reddit_comment_link_candidates(observation: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for item in observation.get("clickable_elements") or []:
        text = str(item.get("text", "") or "").strip()
        lowered = text.lower()
        if not text or lowered in seen:
            continue
        if not re.fullmatch(r"\d+\s+comments?", lowered):
            continue
        seen.add(lowered)
        candidates.append(text)
    return candidates


def select_best_reddit_click_target_with_usage(
    glm: GLMClient,
    *,
    task_item: dict[str, Any],
    observation: dict[str, Any],
    candidates: list[str],
) -> tuple[str, dict[str, float], str]:
    if not candidates:
        return "NOT_VISIBLE", zero_usage(), "NOT_VISIBLE"
    screenshot_path = REPO_ROOT / observation["screenshot_path"]
    prompt = (
        f"Goal:\n{observation.get('goal', task_item.get('intent', ''))}\n\n"
        f"Current URL:\n{observation.get('url', '')}\n\n"
        "Visible reddit click targets on the current page:\n"
        + "\n".join(f"- {title}" for title in candidates[:20])
    )
    raw_text, usage = ask_model_with_images_and_usage(
        glm,
        system_prompt=REDDIT_POST_PICKER_SYSTEM_PROMPT,
        text_prompt=prompt,
        screenshot_path=screenshot_path,
        goal_image_urls=observation.get("goal_image_urls") or [],
    )
    normalized = normalize_current_page_answer(task_item.get("intent", ""), raw_text)
    if normalized in candidates:
        return normalized, usage, raw_text
    for title in candidates:
        if normalized.lower() == title.lower():
            return title, usage, raw_text
    return "NOT_VISIBLE", usage, raw_text


def build_reddit_navigation_skill(
    task_item: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    history_tags: list[str] | None = None,
) -> dict[str, Any]:
    repair_history = ["heuristic_reddit_navigation_fallback"]
    if history_tags:
        repair_history.extend(history_tags)
    return {
        "skill_id": f"{task_item['task_id']}_reddit_navigation_fallback",
        "task": task_item.get("intent", task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": steps,
        "repair_history": repair_history,
        "patches": [],
    }


def find_reddit_repair_observation(
    latest_observation: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    subreddit: str | None,
) -> dict[str, Any]:
    latest_url = str(latest_observation.get("url", "") or "")
    if latest_url and not is_reddit_image_like_url(latest_url):
        return latest_observation

    for step in reversed(execution_trace or []):
        candidate = step.get("post_action_observation") or step.get("observation") or {}
        if not isinstance(candidate, dict):
            continue
        url = str(candidate.get("url", "") or "")
        if not url or is_reddit_image_like_url(url):
            continue
        if subreddit and f"/f/{subreddit.lower()}" not in url.lower() and "/forums/all" not in url.lower():
            continue
        if candidate.get("clickable_elements"):
            return candidate
    return latest_observation


def maybe_build_reddit_navigation_repair(
    glm: GLMClient,
    *,
    task_item: dict[str, Any],
    seed_observation: dict[str, Any],
    latest_observation: dict[str, Any],
    failure_info: dict[str, Any],
    current_skill: dict[str, Any],
    execution_trace: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not is_reddit_navigation_task(task_item, seed_observation):
        return None

    fail_reason = str(failure_info.get("fail_reason", "") or "").lower()
    if not any(
        token in fail_reason
        for token in (
            "translator_error",
            "repair_no_meaningful_change",
            "skill_exhausted_without_success",
            "environment terminated without a positive score",
            "invalid_skill_json",
        )
    ):
        return None

    goal_text = str(task_item.get("intent", "") or "")
    subreddit = extract_reddit_subreddit(goal_text)
    if not subreddit:
        return None

    analysis_observation = find_reddit_repair_observation(latest_observation, execution_trace, subreddit)
    analysis_url = str(analysis_observation.get("url", "") or "")

    steps: list[dict[str, Any]] = [{"action": "CLICK", "target": subreddit, "value": None}]
    history_tags = list(current_skill.get("repair_history") or [])
    history_tags.append(f"reddit_subreddit:{subreddit}")

    needs_top_all_time = ("all time" in goal_text.lower()) and ("top" in goal_text.lower() or "top ranked" in goal_text.lower())
    if needs_top_all_time:
        steps.extend(
            [
                {"action": "CLICK", "target": "Sort by: Hot", "value": None},
                {"action": "CLICK", "target": "Top", "value": None},
                {"action": "CLICK", "target": "From: Past 24 hours", "value": None},
                {"action": "CLICK", "target": "All time", "value": None},
            ]
        )
        history_tags.append("reddit_sort:top_all_time")

    should_pick_post = needs_top_all_time or (f"/f/{subreddit.lower()}" in analysis_url.lower())
    if not should_pick_post:
        return {
            "skill": build_reddit_navigation_skill(task_item, steps=steps, history_tags=history_tags),
            "usage": zero_usage(),
            "raw_output": subreddit,
            "source": "reddit_navigation_prefix",
        }

    prefer_comment_links = any(
        token in goal_text.lower()
        for token in ("comments section", "comment section", "image post", "photo")
    )
    candidates = (
        collect_reddit_comment_link_candidates(analysis_observation)
        if prefer_comment_links
        else collect_reddit_post_title_candidates(analysis_observation)
    )
    if not candidates and prefer_comment_links:
        candidates = collect_reddit_post_title_candidates(analysis_observation)
    chosen_target, usage, raw_output = select_best_reddit_click_target_with_usage(
        glm,
        task_item=task_item,
        observation=analysis_observation,
        candidates=candidates,
    )
    if chosen_target == "NOT_VISIBLE":
        return None

    steps.append({"action": "CLICK", "target": chosen_target, "value": None})
    steps.append({"action": "STOP", "target": None, "value": "Done"})
    history_tags.append(f"reddit_pick:{chosen_target}")
    return {
        "skill": build_reddit_navigation_skill(task_item, steps=steps, history_tags=history_tags),
        "usage": usage,
        "raw_output": raw_output,
        "source": "reddit_navigation_picker",
    }


def select_best_classifieds_listing_with_usage(
    glm: GLMClient,
    *,
    task_item: dict[str, Any],
    observation: dict[str, Any],
    query_text: str,
    titles: list[str],
) -> tuple[str, dict[str, float], str]:
    if not titles:
        return "NOT_VISIBLE", zero_usage(), "NOT_VISIBLE"
    screenshot_path = REPO_ROOT / observation["screenshot_path"]
    prompt = (
        f"Goal:\n{observation.get('goal', task_item.get('intent', ''))}\n\n"
        f"Search query already used:\n{query_text}\n\n"
        "Visible listing titles on the current classifieds results page:\n"
        + "\n".join(f"- {title}" for title in titles[:20])
    )
    raw_text, usage = ask_model_with_images_and_usage(
        glm,
        system_prompt=CLASSIFIEDS_RESULT_PICKER_SYSTEM_PROMPT,
        text_prompt=prompt,
        screenshot_path=screenshot_path,
        goal_image_urls=observation.get("goal_image_urls") or [],
    )
    normalized = normalize_current_page_answer(task_item.get("intent", ""), raw_text)
    if normalized in titles:
        return normalized, usage, raw_text
    for title in titles:
        if normalized.lower() == title.lower():
            return title, usage, raw_text
    return "NOT_VISIBLE", usage, raw_text


def maybe_build_classifieds_search_repair(
    glm: GLMClient,
    *,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    seed_observation: dict[str, Any],
    latest_observation: dict[str, Any],
    failure_info: dict[str, Any],
    current_skill: dict[str, Any],
) -> dict[str, Any] | None:
    if not is_classifieds_search_task(task_item, seed_observation):
        return None

    fail_reason = str(failure_info.get("fail_reason", "") or "").lower()
    if not any(
        token in fail_reason
        for token in (
            "translator_error",
            "repair_no_meaningful_change",
            "skill_exhausted_without_success",
            "environment terminated without a positive score".lower(),
            "invalid_skill_json",
        )
    ):
        return None

    query_variants = build_classifieds_query_variants(task_item.get("intent", ""), latest_observation or seed_observation)
    query_text = extract_classifieds_query(task_item.get("intent", ""), latest_observation or seed_observation)
    category_value = infer_classifieds_category(task_item.get("intent", ""))
    search_field_label = find_search_field_label(seed_observation) or find_search_field_label(latest_observation)
    search_click_target = find_search_click_target(seed_observation) or find_search_click_target(latest_observation)
    if not query_text or not search_field_label or not search_click_target:
        return None
    current_query = extract_current_skill_search_query(current_skill, search_field_label) or query_text
    existing_listing_click = extract_existing_classifieds_listing_click(
        current_skill,
        search_click_target=search_click_target,
    )
    tried_queries = set(extract_classifieds_history_tags(current_skill, "classf_query:"))
    tried_listings = set(extract_classifieds_history_tags(current_skill, "classf_pick:"))
    scroll_count = extract_classifieds_scroll_count(current_skill, current_query)
    inherited_nav_targets = extract_classifieds_nav_targets(current_skill)
    candidate_pool = extract_classifieds_history_tags(current_skill, "classf_pool:")
    previous_pool: list[str] = []
    if candidate_pool:
        previous_pool = [item for item in candidate_pool[-1].split(" || ") if item]
    pool_union = extract_classifieds_pool_union(current_skill)

    def build_search_plan(
        chosen_query: str,
        *,
        listing_title: str | None = None,
        post_search_click_targets: list[str] | None = None,
        source: str,
        usage: dict[str, float] | None = None,
        raw_output: str | None = None,
        pool: list[str] | None = None,
        page_targets: list[str] | None = None,
        target_scroll_count: int = 0,
        include_stop: bool = False,
    ) -> dict[str, Any]:
        combined_nav_targets: list[str] = []
        seen_nav_targets: set[str] = set()
        for click_target in inherited_nav_targets + list(post_search_click_targets or []):
            normalized_target = str(click_target or "").strip()
            if not normalized_target or normalized_target in seen_nav_targets:
                continue
            seen_nav_targets.add(normalized_target)
            combined_nav_targets.append(normalized_target)
        history_tags = list(current_skill.get("repair_history") or [])
        history_tags.append(f"classf_query:{chosen_query}")
        if pool:
            history_tags.append("classf_pool:" + " || ".join(pool))
        if page_targets:
            history_tags.append("classf_pages:" + " || ".join(page_targets))
        for click_target in combined_nav_targets:
            history_tags.append(f"classf_navpick:{click_target}")
        if listing_title:
            history_tags.append(f"classf_pick:{listing_title}")
        current_known_scroll_count = extract_classifieds_scroll_count(current_skill, chosen_query)
        for next_scroll_count in range(current_known_scroll_count + 1, max(current_known_scroll_count, target_scroll_count) + 1):
            history_tags.append(f"classf_scroll:{chosen_query}#{next_scroll_count}")
        deduped_history: list[str] = []
        seen_history: set[str] = set()
        for entry in history_tags:
            if not isinstance(entry, str):
                continue
            if entry in seen_history:
                continue
            seen_history.add(entry)
            deduped_history.append(entry)
        return {
            "skill": build_classifieds_search_skill(
                task_item,
                query=chosen_query,
                search_field_label=search_field_label,
                search_click_target=search_click_target,
                category_value=category_value,
                listing_title=listing_title,
                post_search_click_targets=combined_nav_targets,
                stop_value=listing_title,
                history_tags=deduped_history,
                scroll_count_after_search=target_scroll_count,
                include_stop=include_stop,
            ),
            "usage": usage or zero_usage(),
            "raw_output": raw_output or listing_title or chosen_query,
            "source": source,
        }

    latest_url = str(latest_observation.get("url", "") or "")
    if "page=search" in latest_url:
        live_candidates = collect_live_classifieds_search_candidates(env)
        pagination_targets = order_classifieds_pagination_targets(collect_live_classifieds_pagination_targets(env))
        sort_targets = collect_live_classifieds_sort_targets(env)
        tried_nav_targets = set(extract_classifieds_history_tags(current_skill, "classf_navpick:"))
        ranked_candidates = rank_live_classifieds_candidates(
            task_item.get("intent", ""),
            current_query,
            live_candidates,
        )
        ranked_titles = list(
            dict.fromkeys(
                str(candidate.get("title", "") or "").strip()
                for candidate in ranked_candidates
                if candidate.get("title")
            )
        )
        titles = ranked_titles or collect_classifieds_listing_titles(latest_observation)
        if not ranked_titles:
            ranked_titles = rank_classifieds_listing_titles(task_item.get("intent", ""), current_query, titles)
        prefers_visual_picker = is_visual_classifieds_goal(task_item.get("intent", ""))
        cheapest_sort_target = next(
            (target for target in sort_targets if target.strip().lower() == "lower price first"),
            None,
        )
        already_sorted_low_to_high = ("sorder=i_price" in latest_url.lower()) and ("iordertype=asc" in latest_url.lower())
        if (
            is_classifieds_cheapest_goal(task_item.get("intent", ""))
            and cheapest_sort_target
            and not already_sorted_low_to_high
            and cheapest_sort_target not in tried_nav_targets
        ):
            return build_search_plan(
                current_query,
                post_search_click_targets=[cheapest_sort_target],
                source="classifieds_search_sort_cheapest",
                pool=titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
                include_stop=False,
            )
        if prefers_visual_picker:
            chosen_title, usage, raw_output = select_best_classifieds_listing_with_usage(
                glm,
                task_item=task_item,
                observation=latest_observation,
                query_text=current_query,
                titles=titles,
            )
            if chosen_title != "NOT_VISIBLE":
                return build_search_plan(
                    current_query,
                    listing_title=chosen_title,
                    source="classifieds_result_picker_visual_first",
                    usage=usage,
                    raw_output=raw_output,
                    pool=titles,
                    page_targets=pagination_targets,
                    target_scroll_count=scroll_count,
                )
        if existing_listing_click and existing_listing_click in ranked_titles and existing_listing_click not in tried_listings:
            return build_search_plan(
                current_query,
                listing_title=existing_listing_click,
                source="classifieds_result_prefix_preserved",
                pool=ranked_titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
            )
        chosen_title = next((title for title in ranked_titles if title not in tried_listings), None)
        if chosen_title:
            return build_search_plan(
                current_query,
                listing_title=chosen_title,
                source="classifieds_result_structured_ranker",
                pool=ranked_titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
            )

        chosen_title, usage, raw_output = select_best_classifieds_listing_with_usage(
            glm,
            task_item=task_item,
            observation=latest_observation,
            query_text=current_query,
            titles=titles,
        )
        if chosen_title != "NOT_VISIBLE":
            return build_search_plan(
                current_query,
                listing_title=chosen_title,
                source="classifieds_result_picker",
                usage=usage,
                raw_output=raw_output,
                pool=titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
            )

        next_page_target = next((target for target in pagination_targets if target not in tried_nav_targets), None)
        if next_page_target:
            return build_search_plan(
                current_query,
                post_search_click_targets=[next_page_target],
                source="classifieds_search_next_page",
                pool=titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
                include_stop=False,
            )

        if scroll_count < 2:
            return build_search_plan(
                current_query,
                source="classifieds_result_scroll",
                pool=titles,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count + 1,
                include_stop=False,
            )

        next_query = next((candidate for candidate in query_variants if candidate not in tried_queries and candidate != current_query), None)
        if next_query:
            return build_search_plan(
                next_query,
                source="classifieds_search_variant_from_results",
            )

    if "page=item" in latest_url and (previous_pool or pool_union):
        item_details = extract_live_classifieds_item_details(env)
        item_review = analyze_classifieds_item_page(task_item.get("intent", ""), current_query, item_details)
        candidate_source = previous_pool or pool_union
        ranked_source = rank_classifieds_listing_titles(task_item.get("intent", ""), current_query, candidate_source) or candidate_source
        pagination_targets = order_classifieds_pagination_targets(extract_classifieds_pagination_targets(current_skill))
        tried_nav_targets = set(extract_classifieds_history_tags(current_skill, "classf_navpick:"))
        required_attribute_missing = any(
            reason.startswith("missing_") or reason in {"accessory_like", "weak_goal_match"}
            for reason in item_review.get("reasons") or []
        )
        next_page_target = next(
            (target for target in pagination_targets if target not in tried_nav_targets),
            None,
        )
        if required_attribute_missing and next_page_target:
            return build_search_plan(
                current_query,
                post_search_click_targets=[next_page_target],
                source="classifieds_item_critic_next_page",
                raw_output="; ".join(item_review.get("reasons") or []) or next_page_target,
                pool=ranked_source,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
                include_stop=False,
            )
        next_listing = next((title for title in ranked_source if title not in tried_listings), None)
        if next_listing:
            return build_search_plan(
                current_query,
                listing_title=next_listing,
                source="classifieds_item_critic_next_candidate",
                raw_output="; ".join(item_review.get("reasons") or []) or str(item_review.get("title") or next_listing),
                pool=ranked_source,
                page_targets=pagination_targets,
                target_scroll_count=scroll_count,
            )

        if scroll_count < 2:
            return build_search_plan(
                current_query,
                source="classifieds_item_critic_scroll_retry",
                raw_output="; ".join(item_review.get("reasons") or []) or current_query,
                pool=ranked_source,
                target_scroll_count=scroll_count + 1,
                include_stop=False,
            )

    next_query = next((candidate for candidate in query_variants if candidate not in tried_queries), None)
    if next_query:
        return build_search_plan(
            next_query,
            source="classifieds_search_variant_prefix",
        )

    return build_search_plan(
        current_query,
        source="classifieds_search_prefix",
    )


def build_classifieds_visual_navigation_skill(
    task_item: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    history_tags: list[str] | None = None,
) -> dict[str, Any]:
    repair_history = ["heuristic_classifieds_visual_navigation"]
    if history_tags:
        repair_history.extend(history_tags)
    return {
        "skill_id": f"{task_item['task_id']}_classifieds_visual_navigation",
        "task": task_item.get("intent", task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": steps,
        "repair_history": repair_history,
        "patches": [],
    }


def maybe_build_classifieds_visual_navigation_repair(
    glm: GLMClient,
    *,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    latest_observation: dict[str, Any],
    failure_info: dict[str, Any],
    current_skill: dict[str, Any],
) -> dict[str, Any] | None:
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if sites != ["classifieds"]:
        return None

    fail_reason = str(failure_info.get("fail_reason", "") or "").lower()
    if not any(
        token in fail_reason
        for token in (
            "translator_error",
            "repair_no_meaningful_change",
            "skill_exhausted_without_success",
            "invalid_skill_json",
            "environment terminated without a positive score",
        )
    ):
        return None

    goal_text = str(task_item.get("intent", "") or "")
    lowered_goal = goal_text.lower()
    visual_goal = any(
        token in lowered_goal
        for token in ("image is", "cover has", "cover with", "taken during", "on the cover")
    )
    if not visual_goal:
        return None

    live_candidates = collect_live_classifieds_search_candidates(env)
    titles = [
        str(item.get("title", "") or "").strip()
        for item in live_candidates
        if str(item.get("title", "") or "").strip()
    ]
    if not titles:
        titles = collect_classifieds_listing_titles(latest_observation)
    titles = list(dict.fromkeys(titles))
    if not titles:
        return None

    chosen_title, usage, raw_output = select_best_classifieds_listing_with_usage(
        glm,
        task_item=task_item,
        observation=latest_observation,
        query_text=goal_text,
        titles=titles,
    )
    if chosen_title == "NOT_VISIBLE":
        return None

    existing_steps = [dict(step) for step in current_skill.get("steps", [])]
    prefix_steps: list[dict[str, Any]] = []
    for step in existing_steps:
        action = str(step.get("action") or "").upper()
        target = str(step.get("target") or "").strip()
        if action not in {"SCROLL", "TYPE", "CLICK"}:
            continue
        if action == "CLICK":
            lowered = target.lower()
            if target == chosen_title:
                continue
            if lowered in {"classifieds", "search", "apply"} or lowered in {
                "all categories",
                "antiques",
                "appliances",
                "arts + crafts",
                "auto parts",
                "beauty + health",
                "bikes",
                "boats",
                "books",
                "clothing",
                "computers",
                "electronics",
                "furniture",
                "video gaming",
            }:
                prefix_steps.append(dict(step))
                continue
            continue
        prefix_steps.append(dict(step))

    steps = prefix_steps + [{"action": "CLICK", "target": chosen_title, "value": None}]
    history_tags = list(current_skill.get("repair_history") or [])
    history_tags.append(f"classf_visual_pick:{chosen_title}")
    return {
        "skill": build_classifieds_visual_navigation_skill(
            task_item,
            steps=steps,
            history_tags=history_tags,
        ),
        "usage": usage,
        "raw_output": raw_output,
        "source": "classifieds_visual_picker",
    }


def build_current_page_answer_prompt(task_item: dict[str, Any], observation: dict[str, Any]) -> str:
    goal_text = observation.get("goal", "").strip()
    page_text = trim_text(observation.get("page_text", ""), limit=9000)
    return (
        f"Benchmark task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Task notes: {task_item.get('notes', '')}\n"
        f"Goal:\n{goal_text}\n\n"
        f"Current URL: {observation.get('url', '')}\n"
        f"Open tab titles: {', '.join(observation.get('open_pages_titles') or []) or 'none'}\n"
        f"Clickable targets:\n{summarize_clickables(observation)}\n\n"
        f"Input fields:\n{summarize_inputs(observation)}\n\n"
        f"Current page text:\n{page_text}\n"
    )


_ORDINAL_MAP = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "last": -1,
}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_SHOPPING_CHECKOUT_QUANTITY_WORDS = {
    "a dozen": 12,
    "dozen": 12,
    "double": 2,
    "octuplets": 8,
    "quadruplets": 4,
    "quintuplets": 5,
    "septuplets": 7,
    "sextuplets": 6,
    "triplets": 3,
    "twin": 2,
    "twins": 2,
    **_NUMBER_WORDS,
}
_COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
_GENERIC_SHOPPING_TARGETS = {
    "advanced search",
    "beauty & personal care",
    "cell phones & accessories",
    "clothing, shoes & jewelry",
    "electronics",
    "grocery & gourmet food",
    "health & household",
    "home",
    "home & kitchen",
    "my account",
    "my cart",
    "my wish list",
    "office products",
    "patio, lawn & garden",
    "search",
    "set ascending direction",
    "sign out",
    "sports & outdoors",
    "store logo",
    "tools & home improvement",
    "video games",
    "view as list",
}
_SHOPPING_GOAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "around",
    "for",
    "from",
    "in",
    "item",
    "me",
    "my",
    "of",
    "page",
    "please",
    "product",
    "show",
    "the",
    "this",
    "to",
    "with",
}
_SHOPPING_SPEC_TOKENS = {
    "ct",
    "count",
    "fl",
    "gram",
    "grams",
    "g",
    "kg",
    "lb",
    "lbs",
    "net",
    "ounce",
    "ounces",
    "oz",
    "pack",
    "weight",
}
_SHOPPING_ACCESSORY_TOKENS = {
    "adapter",
    "battery",
    "batteries",
    "bundle",
    "cable",
    "cables",
    "case",
    "cases",
    "charger",
    "chargers",
    "charging",
    "cover",
    "covers",
    "dock",
    "docks",
    "kit",
    "pack",
    "packs",
    "replacement",
    "shell",
    "skin",
    "skins",
    "strap",
    "straps",
}
_SHOPPING_STRONG_ACCESSORY_TOKENS = {
    "adapter",
    "battery",
    "batteries",
    "cable",
    "cables",
    "charger",
    "chargers",
    "charging",
    "dock",
    "docks",
    "kit",
    "replacement",
}


def _extract_axis_target(goal_text: str, axis: str) -> int | tuple[int, int] | None:
    lowered = (goal_text or "").lower()
    if axis == "row":
        between_match = re.search(
            r"\b(the\s+)?(?P<left>first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+and\s+"
            r"(?P<right>first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?|last)\s+rows?\b",
            lowered,
        )
        if between_match:
            left = _ORDINAL_MAP.get(between_match.group("left"), 0)
            right = _ORDINAL_MAP.get(between_match.group("right"), 0)
            if left and right:
                return (left, right)

    match = re.search(
        rf"\b(the\s+)?(?P<label>first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th)?)\s+{axis}s?\b",
        lowered,
    )
    if not match:
        return None
    return _ORDINAL_MAP.get(match.group("label"))


def _parse_currency_value(text: str) -> float | None:
    match = re.search(r"(\d[\d,]*\.\d{2})", text or "")
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _format_price_range(values: list[float], style: str) -> str | None:
    if not values:
        return None
    low = min(values)
    high = max(values)
    def _format_value(value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}"
        return f"{value:,.2f}"
    if style == "classifieds":
        return f"{_format_value(low)} $ to {_format_value(high)} $"
    return f"${low:,.2f} to ${high:,.2f}"


def _group_by_coordinate(
    items: list[dict[str, Any]],
    *,
    coord_key: str,
    tolerance: float,
) -> list[list[dict[str, Any]]]:
    sorted_items = sorted(items, key=lambda item: (float(item.get(coord_key, 0.0)), float(item.get("secondary", 0.0))))
    groups: list[list[dict[str, Any]]] = []
    anchors: list[float] = []
    for item in sorted_items:
        coord = float(item.get(coord_key, 0.0))
        placed = False
        for index, anchor in enumerate(anchors):
            if abs(coord - anchor) <= tolerance:
                groups[index].append(item)
                placed = True
                break
        if not placed:
            anchors.append(coord)
            groups.append([item])
    return groups


def _select_group(groups: list[list[dict[str, Any]]], target: int | tuple[int, int] | None) -> list[dict[str, Any]]:
    if not groups or target is None:
        return []
    if isinstance(target, tuple):
        selected: list[dict[str, Any]] = []
        for part in target:
            selected.extend(_select_group(groups, part))
        return selected
    if target == -1:
        return groups[-1]
    index = target - 1
    if 0 <= index < len(groups):
        return groups[index]
    return []


def collect_live_shopping_product_cards(
    env: VisualWebArenaEnv,
    *,
    visible_only: bool = True,
) -> list[dict[str, Any]]:
    if env.env is None:
        return []

    page = env.env.unwrapped.page
    cards = page.evaluate(
        """
() => {
  const readBid = (node) => {
    if (!node) return null;
    return node.getAttribute('bid') || node.getAttribute('browsergym_id') || null;
  };
  const nodes = [...document.querySelectorAll('.product-item')];
    return nodes.map((el) => {
      const rect = el.getBoundingClientRect();
      const isVisible =
        rect.width > 0 &&
        rect.height > 0 &&
        rect.bottom > 0 &&
        rect.right > 0 &&
        rect.top < window.innerHeight &&
        rect.left < window.innerWidth;
      const imageLink =
        el.querySelector('.product-image-photo')?.closest('a') ||
        el.querySelector('.product-item-photo') ||
        el.querySelector('a.product.photo') ||
        null;
      const titleLink =
        el.querySelector('.product-item-link') ||
        el.querySelector('.product.name a') ||
        el.querySelector('a.product-item-link') ||
        el.querySelector('a');
      const titleText =
        (titleLink?.getAttribute('title') || '').trim() ||
        (titleLink?.getAttribute('aria-label') || '').trim() ||
        (titleLink?.textContent || '').trim();
      const addToCart =
        [...el.querySelectorAll('button, a, span')].find((node) =>
          /add to cart/i.test((node.innerText || '').trim())
        ) || null;
    const addToWishlist =
      [...el.querySelectorAll('button, a, span')].find((node) =>
        /wish list|wishlist/i.test((node.innerText || '').trim())
      ) || null;
    return {
      visible: isVisible,
      text: (el.innerText || '').trim(),
      title: titleText,
      title_bid: readBid(titleLink),
      image_text:
        (imageLink?.getAttribute('title') || '').trim() ||
        (imageLink?.getAttribute('aria-label') || '').trim() ||
        (imageLink?.innerText || '').trim(),
      image_bid: readBid(imageLink),
      add_to_cart_text: (addToCart?.innerText || '').trim(),
      add_to_cart_bid: readBid(addToCart),
      add_to_wishlist_text: (addToWishlist?.innerText || '').trim(),
      add_to_wishlist_bid: readBid(addToWishlist),
      top: rect.top,
      left: rect.left,
      secondary: rect.left,
    };
  }).filter((item) => item.text);
}
        """
    )
    if not isinstance(cards, list):
        return []
    if visible_only:
        cards = [card for card in cards if bool(card.get("visible"))]
    return cards


def _is_generic_shopping_target(target: str) -> bool:
    normalized = _normalize_goal_fragment(target)
    if not normalized:
        return True
    if normalized in _GENERIC_SHOPPING_TARGETS:
        return True
    return normalized.endswith(" item")


def _looks_product_like_target(target: str) -> bool:
    normalized = _normalize_goal_fragment(target)
    if not normalized or _is_generic_shopping_target(normalized):
        return False
    tokens = [token for token in normalized.split() if token]
    if len(tokens) >= 4:
        return True
    return any(char.isdigit() for char in normalized) or "," in str(target or "")


def _is_shopping_grid_position_task(goal_text: str) -> bool:
    lowered = (goal_text or "").lower()
    return any(token in lowered for token in (" row", " rows", " column", " columns", "below "))


def _needs_shopping_product_page(goal_text: str) -> bool:
    lowered = (goal_text or "").lower()
    return "product page" in lowered or "on the product page" in lowered


def _is_shopping_action_goal(task_item: dict[str, Any], goal_text: str) -> bool:
    if str(task_item.get("task_family") or "") == "shopping_checkout":
        return True

    lowered = (goal_text or "").lower()
    return any(
        token in lowered
        for token in (
            "to my cart",
            "shopping cart",
            "wish list",
            "wishlist",
            "product page",
            "on the product page",
        )
    )


def _is_shopping_checkout_task(task_item: dict[str, Any]) -> bool:
    return (
        primary_site_for_task(task_item) == "shopping"
        and str(task_item.get("task_family") or "") == "shopping_checkout"
    )


def _requires_explicit_shopping_checkout(task_item: dict[str, Any]) -> bool:
    if not _is_shopping_checkout_task(task_item):
        return False

    goal_text = str(task_item.get("intent", "") or "")
    lowered = " ".join(goal_text.lower().split())
    if not lowered:
        return False

    explicit_checkout_phrases = (
        "place order",
        "proceed to checkout",
        "proceed checkout",
        "checkout",
        "ship here",
        "complete the purchase",
        "complete purchase",
        "finish the purchase",
        "finish purchase",
        "buy it",
        "buy them",
        "order enough",
        "so each can get their own",
    )
    if any(phrase in lowered for phrase in explicit_checkout_phrases):
        return True

    return _extract_shopping_checkout_quantity(task_item) > 1


def _extract_shopping_checkout_quantity(task_item: dict[str, Any]) -> int:
    goal_text = str(task_item.get("intent", "") or "")
    lowered = " ".join(goal_text.lower().split())
    if "each can get their own" not in lowered:
        return 1

    for phrase, quantity in sorted(
        _SHOPPING_CHECKOUT_QUANTITY_WORDS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            return quantity

    match = re.search(
        r"\b(\d+)\s+(?:sons|daughters|kids|children|guests|people|friends|students)\b",
        lowered,
    )
    if match:
        return max(1, int(match.group(1)))
    return 1


def _extract_shopping_checkout_query(
    task_item: dict[str, Any],
    observation: dict[str, Any] | None = None,
) -> str:
    url = str((observation or {}).get("url", "") or "")
    if url:
        parsed = urlparse(url)
        query_values = parse_qs(parsed.query).get("q")
        if query_values:
            return unquote_plus(query_values[0]).strip()

    goal_text = str(task_item.get("intent", "") or "")
    lowered = " ".join(goal_text.lower().split())
    candidate = ""

    if "each can get their own" in lowered:
        match = re.search(
            r"order enough(?: of)? (.+?) so each can get their own",
            lowered,
        )
        if match:
            candidate = match.group(1).strip()

    if not candidate:
        candidate = _extract_shopping_goal_anchor(goal_text)

    candidate = re.sub(r"\([^)]*\)", " ", candidate)
    candidate = re.sub(r"\bin the (?:first|second|third|fourth|fifth|last) row\b", " ", candidate)
    candidate = re.sub(r"\bin the (?:first|second|third|fourth|fifth|last) column\b", " ", candidate)
    candidate = re.sub(r"\bbelow the .*$", " ", candidate)
    candidate = re.sub(r"\bto my cart\b", " ", candidate)
    candidate = re.sub(r"\bfor each\b.*$", " ", candidate)
    candidate = re.sub(r"\bso each can get their own\b.*$", " ", candidate)
    candidate = re.sub(r"^(?:of\s+)?(?:the\s+)?", "", candidate.strip())
    candidate = re.sub(r"[^a-z0-9 ]+", " ", candidate.lower())
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def _build_shopping_checkout_query_variants(query_text: str) -> list[str]:
    base = re.sub(r"\s+", " ", str(query_text or "").strip().lower())
    if not base:
        return []

    base_tokens = [
        token
        for token in base.split()
        if token
        and token not in _SHOPPING_GOAL_STOPWORDS
        and token not in _COLOR_WORDS
    ]
    prioritized_variants: list[str] = []
    if len(base_tokens) >= 3:
        tail_variant = " ".join(base_tokens[-2:]).strip()
        head_variant = base_tokens[-1].strip()
        for candidate in (tail_variant, head_variant):
            if candidate:
                prioritized_variants.append(candidate)

    variants = prioritized_variants + [base]
    queue = [base]
    seen_variant_inputs = {base, *prioritized_variants}

    while queue:
        candidate = queue.pop(0)
        tokens = candidate.split()
        if not tokens:
            continue

        last = tokens[-1]
        singular = ""
        if len(last) >= 5 and last.endswith("ies"):
            singular = last[:-3] + "y"
        elif len(last) >= 4 and last.endswith("s") and not last.endswith("ss"):
            singular = last[:-1]
        if singular and singular != last:
            singular_variant = " ".join(tokens[:-1] + [singular]).strip()
            if singular_variant and singular_variant not in seen_variant_inputs:
                variants.append(singular_variant)
                queue.append(singular_variant)
                seen_variant_inputs.add(singular_variant)

        if any(token in _COLOR_WORDS for token in tokens):
            without_colors = " ".join(token for token in tokens if token not in _COLOR_WORDS).strip()
            if without_colors and without_colors not in seen_variant_inputs:
                variants.append(without_colors)
                queue.append(without_colors)
                seen_variant_inputs.add(without_colors)

        informative_tokens = [
            token
            for token in tokens
            if token
            and token not in _SHOPPING_GOAL_STOPWORDS
            and token not in _COLOR_WORDS
        ]
        if informative_tokens:
            tail_variant = " ".join(informative_tokens[-2:]).strip()
            if tail_variant and tail_variant not in seen_variant_inputs:
                variants.append(tail_variant)
                queue.append(tail_variant)
                seen_variant_inputs.add(tail_variant)
            head_variant = informative_tokens[-1].strip()
            if head_variant and head_variant not in seen_variant_inputs:
                variants.append(head_variant)
                queue.append(head_variant)
                seen_variant_inputs.add(head_variant)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped


def _find_shopping_search_input_target(observation: dict[str, Any]) -> str | None:
    for field in observation.get("input_fields") or []:
        label = str(field.get("label", "") or field.get("name", "") or "").strip()
        if "search" not in label.lower():
            continue
        return label
    return None


def _find_shopping_click_target(
    observation: dict[str, Any],
    text: str,
    *,
    preferred_roles: tuple[str, ...] = (),
) -> str | None:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None

    matches: list[dict[str, Any]] = []
    for item in observation.get("clickable_elements") or []:
        item_text = str(item.get("text", "") or "").strip()
        if item_text.lower() != normalized:
            continue
        matches.append(item)

    if not matches:
        return None

    if preferred_roles:
        for role in preferred_roles:
            for item in matches:
                if str(item.get("role", "") or "").strip().lower() == role:
                    return str(item.get("text", "") or "").strip()

    item = matches[0]
    return str(item.get("text", "") or "").strip()


def _find_shopping_search_click_target(observation: dict[str, Any]) -> str | None:
    return _find_shopping_click_target(
        observation,
        "Search",
        preferred_roles=("button", "link"),
    )


def _find_shopping_qty_target(observation: dict[str, Any]) -> str:
    for field in observation.get("input_fields") or []:
        label = str(field.get("label", "") or field.get("name", "") or "").strip()
        field_type = str(field.get("type", "") or "").strip().lower()
        if label.lower() in {"qty", "quantity", "1"} or field_type == "spinbutton":
            return label or "1"
    return "1"


def _preview_shopping_checkout_product_card(
    env: VisualWebArenaEnv,
    observation: dict[str, Any],
    query_text: str,
) -> dict[str, Any] | None:
    if env.env is None:
        return None

    page = env.env.unwrapped.page
    search_bid = ""
    search_button_bid = ""
    for field in observation.get("input_fields") or []:
        label = str(field.get("label", "") or field.get("name", "") or "").strip().lower()
        if "search" not in label:
            continue
        search_bid = str(field.get("bid", "") or field.get("browsergym_id", "") or "").strip()
        if search_bid:
            break
    for item in observation.get("clickable_elements") or []:
        if str(item.get("text", "") or "").strip().lower() != "search":
            continue
        if str(item.get("role", "") or "").strip().lower() != "button":
            continue
        search_button_bid = str(item.get("bid", "") or item.get("browsergym_id", "") or "").strip()
        if search_button_bid:
            break

    if not search_bid:
        return None

    try:
        search_input = page.locator(f'[bid="{search_bid}"]').first
        search_input.fill(query_text, timeout=2_000)
        page.wait_for_timeout(300)

        option_locator = page.get_by_text(query_text, exact=True).first
        try:
            if option_locator.is_visible():
                option_locator.click(timeout=3_000)
            elif search_button_bid:
                page.locator(f'[bid="{search_button_bid}"]').first.click(timeout=3_000)
            else:
                return None
        except Exception:
            if not search_button_bid:
                return None
            page.locator(f'[bid="{search_button_bid}"]').first.click(timeout=3_000)

        try:
            page.wait_for_load_state("networkidle", timeout=5_000)
        except Exception:
            pass

        cards = collect_live_shopping_product_cards(env, visible_only=True)
        goal_text = str(observation.get("goal", "") or "")
        color = _extract_goal_color(goal_text) or ""
        non_accessory_cards = [
            card
            for card in cards
            if not _shopping_card_has_conflicting_accessory_spec(card, query_text)
        ]
        if non_accessory_cards:
            cards = non_accessory_cards
        return _pick_best_shopping_card(cards, anchor=query_text, color=color) if cards else None
    except Exception:
        return None


def _preview_shopping_checkout_product_title(
    env: VisualWebArenaEnv,
    observation: dict[str, Any],
    query_text: str,
) -> str:
    selected = _preview_shopping_checkout_product_card(env, observation, query_text)
    return str((selected or {}).get("title") or "").strip()


def build_shopping_checkout_skill(
    task_item: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    history_tags: list[str] | None = None,
) -> dict[str, Any]:
    repair_history = ["heuristic_shopping_checkout_fallback"]
    if history_tags:
        repair_history.extend(history_tags)
    deduped_history = list(dict.fromkeys(str(entry) for entry in repair_history if str(entry).strip()))
    return {
        "skill_id": f"{task_item['task_id']}_shopping_checkout_fallback",
        "task": task_item.get("intent", task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": steps,
        "repair_history": deduped_history,
        "patches": [],
    }


def build_shopping_checkout_candidate(
    *,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    current_skill: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any] | None:
    if not _requires_explicit_shopping_checkout(task_item):
        return None

    quantity = _extract_shopping_checkout_quantity(task_item)
    query_text = _extract_shopping_checkout_query(task_item, observation=observation)
    query_variants = _build_shopping_checkout_query_variants(query_text)
    if not query_variants:
        return None
    selected_query = query_variants[0]
    goal_text = str(observation.get("goal", "") or task_item.get("intent", "") or "")
    color = _extract_goal_color(goal_text) or ""

    history_tags = list(current_skill.get("repair_history") or [])
    history_tags.extend(
        [
            f"shopping_checkout:query={selected_query}",
            f"shopping_checkout:qty={quantity}",
        ]
    )

    ship_here_target = _find_shopping_click_target(
        observation,
        "Ship Here",
        preferred_roles=("button", "link"),
    )
    next_target = _find_shopping_click_target(
        observation,
        "Next",
        preferred_roles=("button", "link"),
    )
    place_order_target = _find_shopping_click_target(
        observation,
        "Place Order",
        preferred_roles=("button", "link"),
    )
    if ship_here_target:
        steps = [{"action": "CLICK", "target": ship_here_target, "value": None}]
        if next_target:
            steps.append({"action": "CLICK", "target": next_target, "value": None})
        steps.append({"action": "CLICK", "target": "Place Order", "value": None})
        return {
            "skill": build_shopping_checkout_skill(
                task_item,
                steps=steps,
                history_tags=history_tags + ["shopping_checkout:stage=shipping"],
            ),
            "usage": zero_usage(),
            "raw_output": json.dumps(steps, ensure_ascii=False),
            "source": "shopping_checkout_shipping",
        }

    if next_target and not place_order_target:
        steps = [
            {"action": "CLICK", "target": next_target, "value": None},
            {"action": "CLICK", "target": "Place Order", "value": None},
        ]
        return {
            "skill": build_shopping_checkout_skill(
                task_item,
                steps=steps,
                history_tags=history_tags + ["shopping_checkout:stage=payment_step"],
            ),
            "usage": zero_usage(),
            "raw_output": json.dumps(steps, ensure_ascii=False),
            "source": "shopping_checkout_payment_step",
        }

    if place_order_target:
        steps = [{"action": "CLICK", "target": place_order_target, "value": None}]
        return {
            "skill": build_shopping_checkout_skill(
                task_item,
                steps=steps,
                history_tags=history_tags + ["shopping_checkout:stage=place_order"],
            ),
            "usage": zero_usage(),
            "raw_output": json.dumps(steps, ensure_ascii=False),
            "source": "shopping_checkout_place_order",
        }

    proceed_target = _find_shopping_click_target(
        observation,
        "Proceed to Checkout",
        preferred_roles=("button", "link"),
    )
    if proceed_target:
        steps: list[dict[str, Any]] = []
        if quantity > 1:
            steps.append(
                {
                    "action": "TYPE",
                    "target": _find_shopping_qty_target(observation),
                    "value": str(quantity),
                }
            )
        steps.append({"action": "CLICK", "target": proceed_target, "value": None})
        steps.append({"action": "CLICK", "target": "Ship Here", "value": None})
        steps.append({"action": "CLICK", "target": "Next", "value": None})
        steps.append({"action": "CLICK", "target": "Place Order", "value": None})
        return {
            "skill": build_shopping_checkout_skill(
                task_item,
                steps=steps,
                history_tags=history_tags + ["shopping_checkout:stage=mini_cart"],
            ),
            "usage": zero_usage(),
            "raw_output": json.dumps(steps, ensure_ascii=False),
            "source": "shopping_checkout_minicart",
        }

    raw_cards = collect_live_shopping_product_cards(env, visible_only=True)
    visible_cards = filter_shopping_cards_to_observation(raw_cards, observation)
    non_accessory_cards = [
        card
        for card in visible_cards
        if not _shopping_card_has_conflicting_accessory_spec(card, selected_query)
    ]
    if non_accessory_cards:
        visible_cards = non_accessory_cards
    selected = None
    selected_anchor = selected_query
    if visible_cards:
        for query_variant in query_variants:
            selected = _pick_best_shopping_card(visible_cards, anchor=query_variant, color=color)
            if selected is not None:
                selected_anchor = query_variant
                break

    steps: list[dict[str, Any]] = []
    current_scroll_count = 0
    if selected is not None:
        current_scroll_count = _scroll_steps_to_card(
            steps,
            raw_cards=raw_cards or visible_cards,
            card=selected,
            current_scroll_count=current_scroll_count,
        )
        selected_title = str(selected.get("title") or "").strip()
        if selected_title:
            history_tags.append(f"shopping_checkout:pick={selected_title}")
        title_text = str(selected.get("title") or "").strip()
        image_text = str(selected.get("image_text") or "").strip()
        if title_text:
            steps.append({"action": "CLICK", "target": title_text, "value": None})
        elif image_text and not _is_generic_shopping_target(image_text):
            steps.append({"action": "CLICK", "target": image_text, "value": None})
        if steps and quantity > 1:
            steps.append({"action": "TYPE", "target": "1", "value": str(quantity)})
        if steps:
            steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})

    search_input_target = _find_shopping_search_input_target(observation)
    search_click_target = _find_shopping_search_click_target(observation)
    if not steps and search_input_target and search_click_target:
        best_preview: tuple[tuple[int, int, int, int, int, int], str, str] | None = None
        for query_variant in query_variants:
            preview_card = _preview_shopping_checkout_product_card(env, observation, query_variant)
            if not preview_card:
                continue
            preview_title = str(preview_card.get("title") or "").strip()
            if not preview_title:
                continue
            anchor_score, spec_score, color_score = _shopping_card_match_score(
                preview_card,
                anchor=query_variant,
                color=color,
            )
            head_hits = _shopping_card_primary_goal_hit_count(preview_card, query_variant)
            preview_tuple = (
                1 if head_hits > 0 else 0,
                head_hits,
                color_score,
                anchor_score,
                spec_score,
                len(query_variant),
            )
            candidate_preview = (preview_tuple, query_variant, preview_title)
            if best_preview is None or candidate_preview[0] > best_preview[0]:
                best_preview = candidate_preview
        preview_query = selected_query
        preview_title = ""
        if best_preview is not None:
            _, preview_query, preview_title = best_preview
        selected_query = preview_query
        steps = [{"action": "TYPE", "target": search_input_target, "value": selected_query}]
        if preview_title:
            steps.extend(
                [
                    {"action": "CLICK", "target": search_click_target, "value": None},
                    {"action": "CLICK", "target": preview_title, "value": None},
                ]
            )
            if quantity > 1:
                steps.append({"action": "TYPE", "target": "1", "value": str(quantity)})
            steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})
            history_tags.extend(
                [
                    "shopping_checkout:stage=search_preview",
                    f"shopping_checkout:pick={preview_title}",
                ]
            )
        else:
            steps.extend(
                [
                    {"action": "CLICK", "target": search_click_target, "value": None},
                    {"action": "CLICK", "target": "Add to Cart", "value": None},
                ]
            )
            history_tags.append("shopping_checkout:stage=search_bootstrap")
    elif steps:
        if selected_anchor and selected_anchor != query_text:
            history_tags.append(f"shopping_checkout:query_variant={selected_anchor}")
        history_tags.append("shopping_checkout:stage=result_pick")

    if not steps:
        return None

    steps.append({"action": "CLICK", "target": "My Cart", "value": None})
    steps.append({"action": "CLICK", "target": "Proceed to Checkout", "value": None})
    steps.append({"action": "CLICK", "target": "Ship Here", "value": None})
    steps.append({"action": "CLICK", "target": "Next", "value": None})
    steps.append({"action": "CLICK", "target": "Place Order", "value": None})

    return {
        "skill": build_shopping_checkout_skill(
            task_item,
            steps=steps,
            history_tags=history_tags,
        ),
        "usage": zero_usage(),
        "raw_output": json.dumps(steps, ensure_ascii=False),
        "source": "shopping_checkout_structured",
    }


def _first_shopping_product_target(steps: list[dict[str, Any]]) -> str:
    for step in steps:
        if str(step.get("action") or "").upper() != "CLICK":
            continue
        target = str(step.get("target") or "").strip()
        if _looks_product_like_target(target):
            return target
    return ""


def _normalize_goal_fragment(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _shopping_token_regex(token: str) -> str:
    normalized = _normalize_goal_fragment(token)
    if not normalized:
        return r"$^"

    variants = {normalized}
    if len(normalized) >= 4 and normalized.endswith("y"):
        variants.add(f"{normalized[:-1]}ies")
    if len(normalized) >= 3 and not normalized.endswith("s"):
        variants.add(f"{normalized}s")
    elif len(normalized) >= 4 and normalized.endswith("s"):
        variants.add(normalized[:-1])

    pattern = "|".join(sorted(re.escape(variant) for variant in variants if variant))
    return rf"\b(?:{pattern})\b"


def _shopping_card_match_breakdown(
    card: dict[str, Any],
    *,
    anchor: str = "",
    color: str = "",
) -> dict[str, Any]:
    text = _normalize_goal_fragment(str(card.get("text", "") or ""))
    title = _normalize_goal_fragment(str(card.get("title", "") or ""))
    combined = " ".join(part for part in (title, text) if part).strip()

    anchor_score = 0
    spec_score = 0
    core_hits = 0
    spec_hits = 0
    if anchor:
        goal_tokens = [token for token in anchor.split() if token and token not in _SHOPPING_GOAL_STOPWORDS]
        spec_tokens = [
            token
            for token in goal_tokens
            if any(char.isdigit() for char in token) or token in _SHOPPING_SPEC_TOKENS
        ]
        core_tokens = [token for token in goal_tokens if token not in spec_tokens]

        if anchor in combined:
            anchor_score += 4
            spec_score += 2
        if anchor in title:
            anchor_score += 3
            spec_score += 1

        for left, right in zip(core_tokens, core_tokens[1:]):
            phrase = f"{left} {right}".strip()
            if len(phrase) < 3:
                continue
            if phrase in combined:
                anchor_score += 4
                if phrase in title:
                    anchor_score += 2

        for token in core_tokens:
            if len(token) < 3:
                continue
            token_pattern = _shopping_token_regex(token)
            if re.search(token_pattern, combined):
                core_hits += 1
                anchor_score += 3
                if re.search(token_pattern, title):
                    anchor_score += 2

        if core_tokens:
            if core_hits == 0:
                anchor_score -= 6
            elif len(core_tokens) >= 2 and core_hits == 1:
                anchor_score -= 2

        for token in spec_tokens:
            token_pattern = _shopping_token_regex(token)
            if re.search(token_pattern, combined):
                spec_hits += 1
                spec_score += 2
                if re.search(token_pattern, title):
                    spec_score += 1
            else:
                spec_score -= 1

        if not core_tokens and spec_tokens:
            anchor_score += spec_score
        elif core_hits > 0 and spec_hits > 0:
            spec_score += min(core_hits, spec_hits)

    color_score = 0
    if color:
        if re.search(_shopping_token_regex(color), combined):
            color_score += 2
        else:
            conflicting_colors = {
                candidate
                for candidate in _COLOR_WORDS
                if candidate != color and re.search(_shopping_token_regex(candidate), combined)
            }
            if conflicting_colors:
                color_score -= 2
    return {
        "anchor_score": anchor_score,
        "spec_score": spec_score,
        "color_score": color_score,
        "core_hits": core_hits,
        "spec_hits": spec_hits,
        "title": title,
        "text": text,
        "combined": combined,
    }


def _shopping_card_match_score(card: dict[str, Any], *, anchor: str = "", color: str = "") -> tuple[int, int, int]:
    breakdown = _shopping_card_match_breakdown(card, anchor=anchor, color=color)
    return breakdown["anchor_score"], breakdown["spec_score"], breakdown["color_score"]


def _shopping_card_title_specificity(card: dict[str, Any]) -> int:
    title = _normalize_goal_fragment(str(card.get("title", "") or ""))
    if not title:
        return 0
    informative_tokens = [
        token
        for token in title.split()
        if len(token) >= 3
        and token not in _SHOPPING_GOAL_STOPWORDS
        and token not in _SHOPPING_SPEC_TOKENS
        and not token.isdigit()
    ]
    informative_count = len(dict.fromkeys(informative_tokens))
    if informative_count <= 0:
        return 0
    return max(0, 12 - abs(informative_count - 6))


def _rank_shopping_cards(
    cards: list[dict[str, Any]],
    *,
    anchor: str = "",
    color: str = "",
) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    for card in cards:
        anchor_score, spec_score, color_score = _shopping_card_match_score(
            card,
            anchor=anchor,
            color=color,
        )
        title_specificity = _shopping_card_title_specificity(card)
        title_length = len(str(card.get("title", "") or ""))
        ranked.append(
            (
                (
                    anchor_score,
                    color_score,
                    spec_score,
                    title_specificity,
                    title_length,
                ),
                card,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [card for _, card in ranked]


def _pick_best_shopping_card(
    cards: list[dict[str, Any]],
    *,
    anchor: str = "",
    color: str = "",
) -> dict[str, Any] | None:
    if not cards:
        return None
    ranked = _rank_shopping_cards(cards, anchor=anchor, color=color)
    if not ranked:
        return None
    return ranked[0]


def _select_card_by_group_index(
    groups: list[list[dict[str, Any]]],
    group_index: int,
) -> list[dict[str, Any]]:
    if not groups:
        return []
    if group_index == -1:
        return groups[-1]
    if 0 <= group_index < len(groups):
        return groups[group_index]
    return []


def infer_shopping_card_row_index(
    cards: list[dict[str, Any]],
    selected_card: dict[str, Any],
) -> int | None:
    if not cards:
        return None
    rows = _group_by_coordinate(cards, coord_key="top", tolerance=80.0)
    selected_title = str(selected_card.get("title") or "").strip().lower()
    selected_bid = str(selected_card.get("title_bid") or "").strip().lower()
    for row_index, row in enumerate(rows):
        for card in row:
            card_title = str(card.get("title") or "").strip().lower()
            card_bid = str(card.get("title_bid") or "").strip().lower()
            if selected_title and card_title == selected_title:
                return row_index
            if selected_bid and card_bid == selected_bid:
                return row_index
    return None


def infer_shopping_card_scroll_count(
    selected_card: dict[str, Any],
    *,
    row_index: int | None = None,
) -> int:
    top = _to_float(selected_card.get("top"))
    scroll_count = max(0, int(row_index or 0))
    if top > 380.0:
        scroll_count = max(scroll_count, int(math.ceil((top - 380.0) / 700.0)))
    return max(0, scroll_count)


def shopping_scroll_assisted_step_indices(skill: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    seen_scroll = False
    for index, step in enumerate(skill.get("steps", []) or [], start=1):
        action = str(step.get("action") or "").upper()
        if action == "SCROLL":
            seen_scroll = True
            continue
        if seen_scroll and action == "CLICK":
            indices.add(index)
    return indices


def filter_shopping_cards_to_observation(
    cards: list[dict[str, Any]],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    clickable_texts = {
        str(item.get("text") or "").strip().lower()
        for item in observation.get("clickable_elements") or []
        if str(item.get("text") or "").strip()
    }
    clickable_bids = {
        str(item.get("bid") or "").strip().lower()
        for item in observation.get("clickable_elements") or []
        if str(item.get("bid") or "").strip()
    }

    filtered: list[dict[str, Any]] = []
    for card in cards:
        normalized = dict(card)
        title = str(card.get("title") or "").strip()
        title_bid = str(card.get("title_bid") or "").strip()
        image_text = str(card.get("image_text") or "").strip()
        image_bid = str(card.get("image_bid") or "").strip()

        if title and title.lower() not in clickable_texts:
            normalized["title"] = ""
        if title_bid and title_bid.lower() not in clickable_bids:
            normalized["title_bid"] = ""
        if image_text and image_text.lower() not in clickable_texts:
            normalized["image_text"] = ""
        if image_bid and image_bid.lower() not in clickable_bids:
            normalized["image_bid"] = ""

        if not any(
            str(normalized.get(key) or "").strip()
            for key in ("title", "title_bid", "image_text", "image_bid")
        ):
            continue
        filtered.append(normalized)

    return filtered


def _extract_shopping_goal_anchor(goal_text: str) -> str:
    lowered = _normalize_goal_fragment(goal_text)
    if not lowered:
        return ""

    patterns = (
        r"product page of (?:the )?(.*)",
        r"add (?:the )?(.*) to my cart",
        r"add (?:the )?(.*) to my wish list",
        r"show me (?:the )?(.*)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            lowered = match.group(1).strip()
            break

    lowered = re.sub(
        r"\b(please|can you|with|around|about|item|product|page|show|me|the|a|an)\b",
        " ",
        lowered,
    )
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _extract_shopping_size_specs(text: str) -> set[str]:
    lowered = _normalize_goal_fragment(text)
    if not lowered:
        return set()

    specs: set[str] = set()
    for number, unit in re.findall(
        r"\b(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|lbs|pound|pounds)\b",
        lowered,
    ):
        normalized_unit = "oz" if unit.startswith("o") else "lb"
        normalized_number = number.rstrip("0").rstrip(".") if "." in number else number
        specs.add(f"{normalized_number} {normalized_unit}")
    return specs


def _extract_shopping_accessory_specs(text: str) -> set[str]:
    lowered = _normalize_goal_fragment(text)
    if not lowered:
        return set()
    return {
        token
        for token in _SHOPPING_ACCESSORY_TOKENS
        if re.search(_shopping_token_regex(token), lowered)
    }


def _extract_primary_shopping_goal_tokens(text: str) -> list[str]:
    normalized = _normalize_goal_fragment(text)
    if not normalized:
        return []

    filtered_tokens = [
        token
        for token in normalized.split()
        if token
        and token not in _SHOPPING_GOAL_STOPWORDS
        and token not in _SHOPPING_SPEC_TOKENS
        and token not in _COLOR_WORDS
    ]
    if not filtered_tokens:
        return []

    primary = filtered_tokens[-1:]
    expanded: list[str] = []
    seen: set[str] = set()
    for token in primary:
        candidates = [token]
        if len(token) >= 5 and token.endswith("ies"):
            candidates.append(token[:-3] + "y")
        elif len(token) >= 4 and token.endswith("s") and not token.endswith("ss"):
            candidates.append(token[:-1])
        elif len(token) >= 4:
            candidates.append(token + "s")
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def _shopping_card_primary_goal_hit_count(card: dict[str, Any], goal_text: str) -> int:
    tokens = _extract_primary_shopping_goal_tokens(goal_text)
    if not tokens:
        return 0
    combined = _normalize_goal_fragment(
        " ".join(
            part
            for part in (
                str(card.get("title") or "").strip(),
                str(card.get("text") or "").strip(),
            )
            if part
        )
    )
    matched: set[str] = set()
    for token in tokens:
        if re.search(_shopping_token_regex(token), combined):
            matched.add(token)
    return len(matched)


def _shopping_token_position(tokens: list[str], candidates: set[str]) -> int | None:
    if not tokens or not candidates:
        return None
    for index, token in enumerate(tokens):
        if token in candidates:
            return index
    return None


def _shopping_card_has_conflicting_size_spec(card: dict[str, Any], goal_text: str) -> bool:
    goal_specs = _extract_shopping_size_specs(goal_text)
    if not goal_specs:
        return False

    card_text = " ".join(
        part
        for part in (
            str(card.get("title") or "").strip(),
            str(card.get("text") or "").strip(),
        )
        if part
    )
    card_specs = _extract_shopping_size_specs(card_text)
    if not card_specs:
        return False
    return goal_specs.isdisjoint(card_specs)


def _shopping_card_has_conflicting_accessory_spec(card: dict[str, Any], goal_text: str) -> bool:
    goal_specs = _extract_shopping_accessory_specs(goal_text)
    if goal_specs:
        return False

    card_title = _normalize_goal_fragment(str(card.get("title") or "").strip())
    card_text = " ".join(
        part
        for part in (
            str(card.get("title") or "").strip(),
            str(card.get("text") or "").strip(),
        )
        if part
    )
    normalized_card_text = _normalize_goal_fragment(card_text)
    card_specs = _extract_shopping_accessory_specs(normalized_card_text)
    if not card_specs:
        return False

    primary_goal_tokens = _extract_primary_shopping_goal_tokens(goal_text)
    if not primary_goal_tokens:
        return False

    title_tokens = [token for token in card_title.split() if token]
    primary_goal_set = set(primary_goal_tokens)
    strong_accessory_specs = set(card_specs).intersection(_SHOPPING_STRONG_ACCESSORY_TOKENS)
    if not strong_accessory_specs:
        return False

    first_goal_position = _shopping_token_position(title_tokens, primary_goal_set)
    first_strong_accessory_position = _shopping_token_position(
        title_tokens,
        _SHOPPING_STRONG_ACCESSORY_TOKENS,
    )
    if first_strong_accessory_position is None:
        return False

    if "not include" in normalized_card_text:
        return True
    if first_goal_position is None:
        return True
    if first_strong_accessory_position < first_goal_position:
        return True

    for separator in (" for ", " compatible with ", " fits ", " fit "):
        prefix = f" {card_title} ".split(separator, 1)[0].strip()
        if not prefix:
            continue
        prefix_tokens = [token for token in prefix.split() if token]
        prefix_goal_position = _shopping_token_position(prefix_tokens, primary_goal_set)
        prefix_strong_accessory_position = _shopping_token_position(
            prefix_tokens,
            _SHOPPING_STRONG_ACCESSORY_TOKENS,
        )
        if prefix_strong_accessory_position is not None and prefix_goal_position is None:
            return True

    return False


def _extract_shopping_between_targets(goal_text: str) -> tuple[str, str] | None:
    raw_lowered = " ".join((goal_text or "").lower().split())
    if "between " not in raw_lowered:
        return None

    patterns = (
        r"between the (?P<left>.+?) and the (?P<right>.+?)(?:,\s*add the cheaper|,\s*just leave|,\s*leave the other| and just| and leave| and add|\.|$)",
        r"between (?P<left>.+?) and (?P<right>.+?)(?:,\s*add the cheaper|,\s*just leave|,\s*leave the other| and just| and leave| and add|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_lowered)
        if not match:
            continue
        left = _normalize_goal_fragment(match.group("left") or "")
        right = _normalize_goal_fragment(match.group("right") or "")
        right = re.sub(
            r"\b(add the cheaper.*|just leave.*|leave the other.*|to my cart.*|to my wishlist.*)\b",
            "",
            right,
        ).strip()
        if left and right and left != right:
            return left, right
    return None


def _extract_shopping_goal_colors(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {
        color
        for color in _COLOR_WORDS
        if re.search(rf"\b{re.escape(color)}\b", lowered)
    }


def _select_shopping_compare_anchor_card(
    cards: list[dict[str, Any]],
    *,
    anchor: str,
    exclude_bid: str = "",
) -> dict[str, Any] | None:
    ranked = _rank_shopping_cards(cards, anchor=anchor)
    if not ranked:
        return None

    required_colors = _extract_shopping_goal_colors(anchor)
    required_tokens = {
        token
        for token in _normalize_goal_fragment(anchor).split()
        if len(token) >= 4 and token not in _SHOPPING_GOAL_STOPWORDS
    }
    for card in ranked:
        if exclude_bid and str(card.get("title_bid") or "").strip() == exclude_bid:
            continue
        breakdown = _shopping_card_match_breakdown(card, anchor=anchor)
        combined = breakdown["combined"]
        if breakdown["anchor_score"] < 5:
            continue
        if required_tokens and not any(token in combined for token in required_tokens):
            continue
        if required_colors and not any(
            re.search(_shopping_token_regex(color), combined) for color in required_colors
        ):
            continue
        return card
    return None


def _scroll_steps_to_card(
    steps: list[dict[str, Any]],
    *,
    raw_cards: list[dict[str, Any]],
    card: dict[str, Any],
    current_scroll_count: int,
) -> int:
    row_index = infer_shopping_card_row_index(raw_cards, card)
    target_scroll_count = infer_shopping_card_scroll_count(card, row_index=row_index)
    for _ in range(max(0, target_scroll_count - current_scroll_count)):
        steps.append({"action": "SCROLL", "target": None, "value": "down"})
    return target_scroll_count


def build_shopping_compare_candidate(
    task_item: dict[str, Any],
    raw_cards: list[dict[str, Any]],
) -> dict[str, Any] | None:
    goal_text = str(task_item.get("intent", "") or "")
    lowered_goal = goal_text.lower()
    if "cheaper" not in lowered_goal:
        return None
    if "wishlist" not in lowered_goal or "cart" not in lowered_goal:
        return None

    between_targets = _extract_shopping_between_targets(goal_text)
    if between_targets is None:
        return None
    left_anchor, right_anchor = between_targets

    candidate_cards = [card for card in raw_cards if str(card.get("title") or "").strip()]
    if len(candidate_cards) < 2:
        return None

    left_card = _select_shopping_compare_anchor_card(candidate_cards, anchor=left_anchor)
    if left_card is None:
        return None
    right_card = _select_shopping_compare_anchor_card(
        candidate_cards,
        anchor=right_anchor,
        exclude_bid=str(left_card.get("title_bid") or "").strip(),
    )
    if right_card is None:
        return None

    left_price = _parse_currency_value(str(left_card.get("text") or ""))
    right_price = _parse_currency_value(str(right_card.get("text") or ""))
    if left_price is None or right_price is None:
        return None

    cheaper_card, other_card = (
        (left_card, right_card) if left_price <= right_price else (right_card, left_card)
    )
    cheaper_cart_bid = str(cheaper_card.get("add_to_cart_bid") or "").strip()
    other_wishlist_bid = str(other_card.get("add_to_wishlist_bid") or "").strip()
    if not cheaper_cart_bid or not other_wishlist_bid:
        return None

    steps: list[dict[str, Any]] = []
    current_scroll_count = 0
    for action_card, action_target in (
        (cheaper_card, f"bid={cheaper_cart_bid}"),
        (other_card, f"bid={other_wishlist_bid}"),
    ):
        current_scroll_count = _scroll_steps_to_card(
            steps,
            raw_cards=raw_cards,
            card=action_card,
            current_scroll_count=current_scroll_count,
        )
        steps.append({"action": "CLICK", "target": action_target, "value": None})

    cheaper_title = str(cheaper_card.get("title") or "").strip()
    other_title = str(other_card.get("title") or "").strip()
    return {
        "skill": build_shopping_grid_skill(
            task_item,
            steps=steps,
            history_tags=[
                "shopping_compare:price_direct_actions",
                f"shopping_compare:left={left_anchor}",
                f"shopping_compare:right={right_anchor}",
                f"shopping_compare:cheaper={cheaper_title}",
                f"shopping_compare:other={other_title}",
            ],
        ),
        "usage": zero_usage(),
        "raw_output": json.dumps(steps, ensure_ascii=False),
        "source": "shopping_compare_structured",
    }


def _extract_history_tag_value(history: list[Any], prefix: str) -> str:
    for entry in reversed(history or []):
        text = str(entry or "").strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""


def select_best_shopping_card_with_usage(
    glm: GLMClient,
    *,
    task_item: dict[str, Any],
    observation: dict[str, Any],
    cards: list[dict[str, Any]],
) -> tuple[str, dict[str, float], str]:
    candidates = [str(card.get("title", "") or "").strip() for card in cards if str(card.get("title", "") or "").strip()]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return "NOT_VISIBLE", zero_usage(), "NOT_VISIBLE"

    screenshot_path = REPO_ROOT / observation["screenshot_path"]
    prompt = (
        f"Goal:\n{observation.get('goal', task_item.get('intent', ''))}\n\n"
        f"Current URL:\n{observation.get('url', '')}\n\n"
        "Visible shopping product titles on the current page:\n"
        + "\n".join(f"- {title}" for title in candidates[:20])
    )
    raw_text, usage = ask_model_with_images_and_usage(
        glm,
        system_prompt=SHOPPING_CARD_PICKER_SYSTEM_PROMPT,
        text_prompt=prompt,
        screenshot_path=screenshot_path,
        goal_image_urls=observation.get("goal_image_urls") or [],
    )
    normalized = normalize_current_page_answer(task_item.get("intent", ""), raw_text)
    if normalized in candidates:
        return normalized, usage, raw_text
    for title in candidates:
        if normalized.lower() == title.lower():
            return title, usage, raw_text
    return "NOT_VISIBLE", usage, raw_text


def select_structured_shopping_card(task_item: dict[str, Any], observation: dict[str, Any], cards: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cards:
        return None

    goal_text = str(observation.get("goal", "") or task_item.get("intent", "") or "")
    non_conflicting_cards = [
        card
        for card in cards
        if not _shopping_card_has_conflicting_size_spec(card, goal_text)
        and not _shopping_card_has_conflicting_accessory_spec(card, goal_text)
    ]
    if non_conflicting_cards:
        cards = non_conflicting_cards
    row_target = _extract_axis_target(goal_text, "row")
    column_target = _extract_axis_target(goal_text, "column")
    relative_target = _extract_relative_row_target(goal_text)
    color = _extract_goal_color(goal_text)
    anchor = _extract_shopping_goal_anchor(goal_text)

    rows = _group_by_coordinate(cards, coord_key="top", tolerance=80.0)
    normalized_rows = [sorted(row, key=lambda item: float(item.get("left", 0.0))) for row in rows]
    columns = _group_by_coordinate(cards, coord_key="left", tolerance=140.0)
    normalized_columns = [sorted(column, key=lambda item: float(item.get("top", 0.0))) for column in columns]

    if relative_target:
        offset, anchor = relative_target
        best_anchor: tuple[int, int, tuple[int, int]] | None = None
        for row_index, row in enumerate(normalized_rows):
            for col_index, card in enumerate(row):
                score = _shopping_card_match_score(card, anchor=anchor)
                if score[0] <= 0:
                    continue
                candidate = (row_index, col_index, score)
                if best_anchor is None or candidate[2] > best_anchor[2]:
                    best_anchor = candidate
        if best_anchor is not None:
            target_row_index = best_anchor[0] + offset
            if 0 <= target_row_index < len(normalized_rows):
                target_row = normalized_rows[target_row_index]
                target_col_index = min(best_anchor[1], max(0, len(target_row) - 1))
                return target_row[target_col_index]

    if row_target is not None and column_target is not None and isinstance(row_target, int) and isinstance(column_target, int):
        row_index = len(normalized_rows) - 1 if row_target == -1 else row_target - 1
        col_index = None
        if row_index >= 0 and row_index < len(normalized_rows):
            row = normalized_rows[row_index]
            if column_target == -1:
                col_index = len(row) - 1
            else:
                col_index = column_target - 1
            if col_index is not None and 0 <= col_index < len(row):
                return row[col_index]

    if row_target is not None:
        row_cards = _select_group(normalized_rows, row_target)
        if row_cards:
            best_row_card = _pick_best_shopping_card(row_cards, anchor=anchor, color=color or "")
            if best_row_card is not None:
                return best_row_card
            return row_cards[0]

    if column_target is not None and isinstance(column_target, int):
        column_index = len(normalized_columns) - 1 if column_target == -1 else column_target - 1
        column_cards = _select_card_by_group_index(normalized_columns, column_index)
        if column_cards:
            best_column_card = _pick_best_shopping_card(column_cards, anchor=anchor, color=color or "")
            if best_column_card is not None:
                return best_column_card
            return column_cards[0]

    if color:
        color_matches = [card for card in cards if _shopping_card_match_score(card, color=color)[2] > 0]
        if len(color_matches) == 1:
            return color_matches[0]
        if color_matches:
            best_color_card = _pick_best_shopping_card(color_matches, anchor=anchor, color=color)
            if best_color_card is not None:
                return best_color_card

    if anchor:
        ranked_cards = _rank_shopping_cards(cards, anchor=anchor, color=color or "")
        if ranked_cards:
            best_card = ranked_cards[0]
            if _shopping_card_match_score(best_card, anchor=anchor, color=color or "")[0] > 0:
                return best_card

    return None


def build_shopping_grid_skill(
    task_item: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    history_tags: list[str] | None = None,
) -> dict[str, Any]:
    repair_history = ["heuristic_shopping_grid_fallback"]
    if history_tags:
        repair_history.extend(history_tags)
    return {
        "skill_id": f"{task_item['task_id']}_shopping_grid_fallback",
        "task": task_item.get("intent", task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": steps,
        "repair_history": repair_history,
        "patches": [],
    }


def build_shopping_grid_candidate(
    *,
    glm: GLMClient | None,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    current_skill: dict[str, Any],
    observation: dict[str, Any],
    base_observation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    goal_text = str(task_item.get("intent", "") or "")
    lowered_goal = goal_text.lower()
    is_position_task = _is_shopping_grid_position_task(goal_text)
    needs_product_page = _needs_shopping_product_page(goal_text)
    if not any(
        token in lowered_goal
        for token in ("row", "column", "below", "to my cart", "product page", "wish list")
    ):
        return None

    execution_observation = base_observation or observation
    raw_cards = collect_live_shopping_product_cards(
        env,
        visible_only=not (is_position_task or needs_product_page),
    )
    compare_candidate = build_shopping_compare_candidate(task_item, raw_cards)
    if compare_candidate is not None:
        return compare_candidate
    if not _is_shopping_action_goal(task_item, goal_text):
        return None
    cards = (
        raw_cards
        if (is_position_task or needs_product_page)
        else filter_shopping_cards_to_observation(raw_cards, observation)
    )
    history_tags = list(current_skill.get("repair_history") or [])
    selected = None
    preferred_bid = _extract_history_tag_value(history_tags, "shopping_grid:title_bid=")
    preferred_title = _extract_history_tag_value(history_tags, "shopping_grid:picker=")
    if preferred_bid:
        for card in cards:
            if str(card.get("title_bid") or "").strip() == preferred_bid:
                selected = card
                break
    if selected is None and preferred_title:
        for card in cards:
            if str(card.get("title", "") or "").strip().lower() == preferred_title.lower():
                selected = card
                break
    if selected is None:
        selected = select_structured_shopping_card(task_item, observation, cards)
    chosen_title = ""
    picker_usage = zero_usage()
    picker_raw_output = ""
    if selected is None and glm is not None and not is_position_task:
        chosen_title, picker_usage, picker_raw_output = select_best_shopping_card_with_usage(
            glm,
            task_item=task_item,
            observation=observation,
            cards=cards,
        )
    if selected is None and chosen_title and chosen_title != "NOT_VISIBLE":
        for card in cards:
            if str(card.get("title", "") or "").strip().lower() == chosen_title.lower():
                selected = card
                break
    if not selected:
        return None

    is_checkout = "to my cart" in goal_text.lower() or "add the" in goal_text.lower()
    is_wishlist = "wish list" in goal_text.lower() or "wishlist" in goal_text.lower()
    click_steps: list[dict[str, Any]] = []
    if is_position_task:
        history_tags.append("shopping_grid:mode=position")
    if needs_product_page:
        history_tags.append("shopping_grid:mode=product_page")
    if chosen_title and chosen_title != "NOT_VISIBLE":
        history_tags.append(f"shopping_grid:picker={chosen_title}")

    title_bid = str(selected.get("title_bid") or "").strip()
    image_bid = str(selected.get("image_bid") or "").strip()
    title_text = str(selected.get("title") or "").strip()
    image_text = str(selected.get("image_text") or "").strip()
    selected_row_index = (
        infer_shopping_card_row_index(raw_cards, selected)
        if (is_position_task or needs_product_page)
        else None
    )
    scroll_count = (
        infer_shopping_card_scroll_count(selected, row_index=selected_row_index)
        if needs_product_page
        else 0
    )
    visible_clickable_texts = {
        str(item.get("text") or "").strip().lower()
        for item in execution_observation.get("clickable_elements") or []
        if str(item.get("text") or "").strip()
    }
    title_visible_now = title_text.lower() in visible_clickable_texts if title_text else False
    image_visible_now = image_text.lower() in visible_clickable_texts if image_text else False

    if needs_product_page and title_text and not title_visible_now and scroll_count > 0:
        for _ in range(scroll_count):
            click_steps.append({"action": "SCROLL", "target": None, "value": "down"})
        click_steps.append({"action": "CLICK", "target": title_text, "value": None})
        history_tags.append(f"shopping_grid:title={title_text}")
        history_tags.append(f"shopping_grid:scrolls={scroll_count}")
    elif needs_product_page and image_text and not image_visible_now and scroll_count > 0 and not _is_generic_shopping_target(image_text):
        for _ in range(scroll_count):
            click_steps.append({"action": "SCROLL", "target": None, "value": "down"})
        click_steps.append({"action": "CLICK", "target": image_text, "value": None})
        history_tags.append(f"shopping_grid:image_text={image_text}")
        history_tags.append(f"shopping_grid:scrolls={scroll_count}")
    elif needs_product_page and title_text:
        click_steps.append({"action": "CLICK", "target": title_text, "value": None})
        history_tags.append(f"shopping_grid:title={title_text}")
        if is_wishlist:
            click_steps.append({"action": "CLICK", "target": "Add to Wish List", "value": None})
        elif is_checkout:
            click_steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})
    elif needs_product_page and image_text and not _is_generic_shopping_target(image_text):
        click_steps.append({"action": "CLICK", "target": image_text, "value": None})
        history_tags.append(f"shopping_grid:image_text={image_text}")
        if is_wishlist:
            click_steps.append({"action": "CLICK", "target": "Add to Wish List", "value": None})
        elif is_checkout:
            click_steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})
    elif title_bid:
        click_steps.append({"action": "CLICK", "target": f"bid={str(selected.get('title_bid')).strip()}", "value": None})
        history_tags.append(f"shopping_grid:title_bid={selected.get('title_bid')}")
        if is_wishlist:
            click_steps.append({"action": "CLICK", "target": "Add to Wish List", "value": None})
        elif is_checkout:
            click_steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})
    elif image_bid:
        click_steps.append({"action": "CLICK", "target": f"bid={image_bid}", "value": None})
        history_tags.append(f"shopping_grid:image_bid={image_bid}")
        if is_wishlist:
            click_steps.append({"action": "CLICK", "target": "Add to Wish List", "value": None})
        elif is_checkout:
            click_steps.append({"action": "CLICK", "target": "Add to Cart", "value": None})
    elif is_checkout and str(selected.get("add_to_cart_bid") or "").strip():
        click_steps.append({"action": "CLICK", "target": f"bid={str(selected.get('add_to_cart_bid')).strip()}", "value": None})
        history_tags.append(f"shopping_grid:add_bid={selected.get('add_to_cart_bid')}")
    elif is_wishlist and str(selected.get("add_to_wishlist_bid") or "").strip():
        click_steps.append({"action": "CLICK", "target": f"bid={str(selected.get('add_to_wishlist_bid')).strip()}", "value": None})
        history_tags.append(f"shopping_grid:wishlist_bid={selected.get('add_to_wishlist_bid')}")
    elif is_checkout and str(selected.get("add_to_cart_text") or "").strip():
        add_text = str(selected.get("add_to_cart_text") or "").strip()
        click_steps.append({"action": "CLICK", "target": add_text, "value": None})
        history_tags.append(f"shopping_grid:add_text={add_text}")
    elif is_wishlist and str(selected.get("add_to_wishlist_text") or "").strip():
        click_steps.append({"action": "CLICK", "target": str(selected.get("add_to_wishlist_text") or "").strip(), "value": None})
        history_tags.append(f"shopping_grid:wishlist_text={str(selected.get('add_to_wishlist_text') or '').strip()}")
    elif title_text:
        click_steps.append({"action": "CLICK", "target": title_text, "value": None})
        history_tags.append(f"shopping_grid:title={title_text}")
    elif image_text and not _is_generic_shopping_target(image_text):
        click_steps.append({"action": "CLICK", "target": image_text, "value": None})
        history_tags.append(f"shopping_grid:image_text={image_text}")

    if not click_steps:
        return None

    return {
        "skill": build_shopping_grid_skill(
            task_item,
            steps=click_steps,
            history_tags=history_tags,
        ),
        "usage": picker_usage,
        "raw_output": picker_raw_output or json.dumps(click_steps, ensure_ascii=False),
        "source": "shopping_grid_structured",
    }


def build_shopping_heuristic_candidate(
    *,
    glm: GLMClient | None,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    current_skill: dict[str, Any],
    observation: dict[str, Any],
    base_observation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    checkout_candidate = build_shopping_checkout_candidate(
        env=env,
        task_item=task_item,
        current_skill=current_skill,
        observation=observation,
    )
    if checkout_candidate is not None:
        return checkout_candidate

    return build_shopping_grid_candidate(
        glm=glm,
        env=env,
        task_item=task_item,
        current_skill=current_skill,
        observation=observation,
        base_observation=base_observation,
    )


def maybe_build_shopping_grid_repair(
    *,
    glm: GLMClient | None,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    seed_observation: dict[str, Any],
    latest_observation: dict[str, Any],
    failure_info: dict[str, Any],
    current_skill: dict[str, Any],
) -> dict[str, Any] | None:
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if sites != ["shopping"]:
        return None

    fail_reason = str(failure_info.get("fail_reason", "") or "").lower()
    if not any(
        token in fail_reason
        for token in (
            "contract_satisfied_but_reward_zero",
            "translator_error",
            "repair_no_meaningful_change",
            "skill_exhausted_without_success",
            "invalid_skill_json",
            "success_contract_failed",
        )
    ):
        return None

    return build_shopping_heuristic_candidate(
        glm=glm,
        env=env,
        task_item=task_item,
        current_skill=current_skill,
        observation=latest_observation or seed_observation,
        base_observation=seed_observation or latest_observation,
    )


def extract_structured_current_page_answer(
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    observation: dict[str, Any],
) -> str | None:
    if env.env is None:
        return None

    page = env.env.unwrapped.page
    goal_text = observation.get("goal", "")
    template_id = int(task_item.get("intent_template_id") or 0)
    site = ((task_item.get("sites") or [""])[0] or "").lower()

    if template_id == 17 and site == "classifieds":
        items = page.evaluate(
            """
() => {
  const selectors = ['li.listing-card', '.listing-card', '.products .item', '.search-list .item'];
  const nodes = selectors.flatMap((selector) => [...document.querySelectorAll(selector)]);
  const deduped = [...new Set(nodes)];
  return deduped.map((el) => {
    const rect = el.getBoundingClientRect();
    const text = (el.innerText || '').trim();
    return {
      text,
      top: rect.top,
      left: rect.left,
      secondary: rect.left,
    };
  }).filter((item) => item.text);
}
            """
        )
        row_target = _extract_axis_target(goal_text, "row")
        rows = _group_by_coordinate(items, coord_key="top", tolerance=60.0)
        selected_items = _select_group(rows, row_target)
        values = [_parse_currency_value(item.get("text", "")) for item in selected_items]
        return _format_price_range([value for value in values if value is not None], "classifieds")

    if template_id == 4 and site == "shopping":
        items = page.evaluate(
            """
() => {
  return [...document.querySelectorAll('.product-item')].map((el) => {
    const rect = el.getBoundingClientRect();
    const prices = [...el.querySelectorAll('.price')]
      .map((node) => (node.innerText || '').trim())
      .filter(Boolean);
    return {
      text: (el.innerText || '').trim(),
      prices,
      top: rect.top,
      left: rect.left,
      secondary: rect.left,
    };
  }).filter((item) => item.text && item.prices.length > 0);
}
            """
        )
        if not items:
            return None

        row_target = _extract_axis_target(goal_text, "row")
        column_target = _extract_axis_target(goal_text, "column")
        selected_items: list[dict[str, Any]]
        if row_target is not None:
            rows = _group_by_coordinate(items, coord_key="top", tolerance=80.0)
            selected_items = _select_group(rows, row_target)
        elif column_target is not None:
            columns = _group_by_coordinate(items, coord_key="left", tolerance=80.0)
            selected_items = _select_group(columns, column_target)
        else:
            return None

        values: list[float] = []
        for item in selected_items:
            for price_text in item.get("prices", []):
                value = _parse_currency_value(price_text)
                if value is not None:
                    values.append(value)
                    break
        return _format_price_range(values, "shopping")

    return None


def _extract_relative_row_target(goal_text: str) -> tuple[int, str] | None:
    lowered = (goal_text or "").lower()
    match = re.search(
        r"\b(?P<count>one|two|three|four|five|\d+)\s+rows?\s+below\s+the\s+(?P<anchor>.+?)(?:\s+to\b|[?.]|$)",
        lowered,
    )
    if not match:
        return None
    raw_count = match.group("count")
    if raw_count.isdigit():
        count = int(raw_count)
    else:
        count = _NUMBER_WORDS.get(raw_count, 0)
    anchor = re.sub(r"[^a-z0-9 ]+", " ", match.group("anchor") or "")
    anchor = re.sub(r"\s+", " ", anchor).strip()
    if count <= 0 or not anchor:
        return None
    return count, anchor


def _extract_goal_color(goal_text: str) -> str | None:
    lowered = (goal_text or "").lower()
    for color in _COLOR_WORDS:
        if re.search(rf"\b{re.escape(color)}\b", lowered):
            return color
    return None


def extract_current_page_answer_with_fallback(
    glm: GLMClient,
    env: VisualWebArenaEnv,
    *,
    task_item: dict[str, Any],
    observation: dict[str, Any],
) -> tuple[str | None, dict[str, float], str, str]:
    structured_answer = extract_structured_current_page_answer(env, task_item, observation)
    if structured_answer:
        return structured_answer, zero_usage(), structured_answer, "structured"

    answer, usage, raw_text = extract_current_page_answer_with_usage(
        glm,
        task_item=task_item,
        observation=observation,
    )
    return answer, usage, raw_text, "model"


def extract_current_page_answer_with_usage(
    glm: GLMClient,
    *,
    task_item: dict[str, Any],
    observation: dict[str, Any],
) -> tuple[str | None, dict[str, float], str]:
    screenshot_path = REPO_ROOT / observation["screenshot_path"]
    raw_text, usage = ask_model_with_images_and_usage(
        glm,
        system_prompt=CURRENT_PAGE_ANSWER_SYSTEM_PROMPT,
        text_prompt=build_current_page_answer_prompt(task_item, observation),
        screenshot_path=screenshot_path,
        goal_image_urls=observation.get("goal_image_urls") or [],
    )
    normalized = normalize_current_page_answer(observation.get("goal", ""), raw_text)
    if is_not_visible_answer(normalized):
        return None, usage, raw_text
    return normalized, usage, raw_text


def build_direct_answer_skill(task_item: dict[str, Any], answer_text: str) -> dict[str, Any]:
    return {
        "skill_id": task_item["task_id"],
        "task": str(task_item.get("notes") or task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": [
            {
                "action": "STOP",
                "target": None,
                "value": answer_text,
            }
        ],
        "repair_history": [],
        "patches": [],
    }


def build_visual_classifieds_exploration_skill(task_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": f"{task_item['task_id']}_visual_explore",
        "task": str(task_item.get("intent") or task_item.get("notes") or task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": [
            {
                "action": "SCROLL",
                "target": None,
                "value": "down",
            }
        ],
        "repair_history": ["seed_visual_classifieds_exploration"],
        "patches": [],
    }


def build_current_page_visual_exploration_skill(task_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": f"{task_item['task_id']}_page_visual_explore",
        "task": str(task_item.get("intent") or task_item.get("notes") or task_item["task_id"]),
        "preconditions": {},
        "success_contract": {},
        "steps": [
            {
                "action": "SCROLL",
                "target": None,
                "value": "down",
            }
        ],
        "repair_history": ["seed_current_page_visual_exploration"],
        "patches": [],
    }


def build_action_user_prompt(task_item: dict[str, Any], observation: dict[str, Any]) -> str:
    goal_text = observation.get("goal", "").strip()
    last_action_error = observation.get("last_action_error", "").strip() or "none"
    page_text = trim_text(observation.get("page_text", ""))
    open_tabs = observation.get("open_pages_titles") or []
    open_tab_text = ", ".join(open_tabs) if open_tabs else "none"
    direct_answer_hint = ""
    if is_current_page_answer_task(task_item, observation) and not is_visual_current_page_comparison_task(
        task_item, observation
    ):
        direct_answer_hint = (
            "High-priority strategy for this task:\n"
            "- This looks like a current-page answer task.\n"
            "- First try to read the answer from the CURRENT page text and screenshot.\n"
            "- If the answer is already visible, output STOP[exact answer] immediately.\n"
            "- Do not click or scroll unless the answer is clearly not visible on the current page.\n\n"
        )
    elif is_current_page_answer_task(task_item, observation):
        direct_answer_hint = (
            "High-priority strategy for this task:\n"
            "- This looks like a current-page visual comparison task.\n"
            "- Compare the visible posts or images on the CURRENT page before answering.\n"
            "- Do not output STOP immediately unless the winner is visually obvious already.\n"
            "- Prefer scrolling on the same page over jumping into detail pages.\n\n"
        )

    return (
        f"Benchmark task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Task family: {task_item.get('task_family', 'unknown')}\n"
        f"Task notes: {task_item['notes']}\n\n"
        f"Goal:\n{goal_text}\n\n"
        f"Current URL: {observation.get('url', '')}\n"
        f"Open tab titles: {open_tab_text}\n"
        f"Last action error: {last_action_error}\n"
        f"{direct_answer_hint}"
        "Important action constraints:\n"
        "- Use only the exact clickable text or exact bid from the clickable target list.\n"
        "- For TYPE, copy one exact input label or exact bid from the input field list.\n"
        "- Prefer bid-based actions when text looks long, duplicated, or noisy.\n"
        "- Valid bid examples: CLICK[bid=117], TYPE[bid=50=blue kayak]\n"
        "- Do not write TYPE[field=...]=... or TYPE[label=...]=...\n"
        "- If an input field says Search query, the correct format is TYPE[Search query=...]\n"
        "- Avoid generic navigation like Home unless the task explicitly requires it.\n\n"
        f"Clickable targets: {summarize_clickables(observation)}\n"
        f"Input fields:\n{summarize_inputs(observation)}\n\n"
        f"Page text:\n{page_text}\n\n"
        "Return exactly one action."
    )


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(message_content).strip()


def ask_model_with_images(
    glm: GLMClient,
    *,
    system_prompt: str,
    text_prompt: str,
    screenshot_path: Path,
    goal_image_urls: list[str] | None = None,
) -> str:
    text, _ = ask_model_with_images_and_usage(
        glm,
        system_prompt=system_prompt,
        text_prompt=text_prompt,
        screenshot_path=screenshot_path,
        goal_image_urls=goal_image_urls,
    )
    return text


def ask_model_with_images_and_usage(
    glm: GLMClient,
    *,
    system_prompt: str,
    text_prompt: str,
    screenshot_path: Path,
    goal_image_urls: list[str] | None = None,
) -> tuple[str, dict[str, float]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for image_url in goal_image_urls or []:
        user_content.append({"type": "image_url", "image_url": {"url": image_url}})
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": encode_image_as_data_url(screenshot_path)},
        }
    )

    response = glm.chat_completion(
        model=glm.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
    return (
        extract_message_text(response.choices[0].message.content),
        normalize_usage(getattr(response, "usage", None)),
    )


def trace_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "env_id": observation.get("env_id", ""),
        "goal": trim_text(observation.get("goal", ""), 800),
        "url": observation.get("url", ""),
        "page_text_excerpt": trim_text(observation.get("page_text", ""), 1800),
        "screenshot_path": observation.get("screenshot_path", ""),
        "last_action_error": observation.get("last_action_error", ""),
        "terminated": observation.get("terminated", False),
        "truncated": observation.get("truncated", False),
        "reward": observation.get("reward", 0.0),
        "clickable_elements": observation.get("clickable_elements", []),
        "input_fields": observation.get("input_fields", []),
        "open_pages_titles": observation.get("open_pages_titles", []),
    }


def build_task_key(run_id: str, baseline: str, task_id: str, attempt_label: str) -> str:
    return f"{baseline}/{run_id}/{task_id}/{attempt_label}"


def ask_model_for_skill(
    glm: GLMClient,
    *,
    text_prompt: str,
    observation: dict[str, Any],
) -> str:
    text, _ = ask_model_for_skill_with_usage(
        glm,
        text_prompt=text_prompt,
        observation=observation,
    )
    return text


def ask_model_for_skill_with_usage(
    glm: GLMClient,
    *,
    text_prompt: str,
    observation: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    screenshot_path = REPO_ROOT / observation["screenshot_path"]
    return ask_model_with_images_and_usage(
        glm,
        system_prompt="Return JSON only.",
        text_prompt=text_prompt,
        screenshot_path=screenshot_path,
        goal_image_urls=observation.get("goal_image_urls") or [],
    )


def build_initial_skill_cache_path(
    model: str,
    split_path: Path,
    task_id: str,
    *,
    cache_key: str | None = None,
) -> Path:
    cache_stem = cache_key or split_path.stem
    return (
        INITIAL_SKILL_ROOT
        / INITIAL_SKILL_CACHE_VERSION
        / sanitize_slug(model)
        / sanitize_slug(cache_stem)
        / f"{sanitize_slug(task_id)}.json"
    )


def load_initial_skill_override_map(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("initial skill map must be a JSON object of task_id -> skill path")
    overrides: dict[str, Path] = {}
    for task_id, raw_path in payload.items():
        if not isinstance(task_id, str) or not isinstance(raw_path, str):
            raise ValueError("initial skill map entries must be string task_id -> string path")
        skill_path = Path(raw_path)
        if not skill_path.is_absolute():
            skill_path = (REPO_ROOT / skill_path).resolve()
        overrides[task_id] = skill_path
    return overrides


def load_or_create_shared_initial_skill(
    glm: GLMClient,
    env: VisualWebArenaEnv,
    *,
    task_item: dict[str, Any],
    split_path: Path,
    cache_key: str | None,
    injected_skill_map: dict[str, Path] | None,
    seed_observation: dict[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    prompt = build_skill_generation_prompt(task_item, seed_observation, max_steps=max_steps)
    prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    injected_skill_path = (injected_skill_map or {}).get(task_item["task_id"])
    if injected_skill_path is not None:
        if not injected_skill_path.exists():
            raise FileNotFoundError(
                f"Injected initial skill for {task_item['task_id']} not found: {injected_skill_path}"
            )
        raw_output = injected_skill_path.read_text(encoding="utf-8")
        try:
            injected_skill = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            return {
                "cache_hit": True,
                "cache_path": injected_skill_path.relative_to(REPO_ROOT).as_posix(),
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "raw_output": raw_output,
                "usage": zero_usage(),
                "skill": None,
                "parse_error": str(exc),
                "generation_mode": "injected_artifact",
                "counts_as_model_call": False,
                "counts_as_live_model_call": False,
            }
        return {
            "cache_hit": True,
            "cache_path": injected_skill_path.relative_to(REPO_ROOT).as_posix(),
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "raw_output": raw_output,
            "usage": zero_usage(),
            "skill": injected_skill,
            "parse_error": None,
            "generation_mode": "injected_artifact",
            "counts_as_model_call": False,
            "counts_as_live_model_call": False,
        }
    cache_path = build_initial_skill_cache_path(
        glm.model,
        split_path,
        task_item["task_id"],
        cache_key=cache_key,
    )
    cache_relpath = cache_path.relative_to(REPO_ROOT).as_posix()

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("prompt_hash") == prompt_hash:
            return {
                "cache_hit": True,
                "cache_path": cache_relpath,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "raw_output": payload.get("raw_output", ""),
                "usage": normalize_usage(payload.get("usage")),
                "skill": payload.get("skill"),
                "parse_error": payload.get("parse_error"),
            }

    if (
        list(task_item.get("sites", []) or []) == ["classifieds"]
        and str(task_item.get("task_family", "") or "") == "navigation"
        and is_visual_classifieds_goal(seed_observation.get("goal", ""))
        and needs_visual_classifieds_exploration(seed_observation.get("goal", ""))
    ):
        exploratory_skill = build_visual_classifieds_exploration_skill(task_item)
        raw_output = json.dumps(exploratory_skill, ensure_ascii=False, indent=2)
        payload = {
            "task_id": task_item["task_id"],
            "env_name": task_item["env_name"],
            "model": glm.model,
            "split_name": split_path.stem,
            "prompt_hash": prompt_hash,
            "raw_output": raw_output,
            "usage": zero_usage(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "skill": exploratory_skill,
            "parse_error": None,
            "generation_mode": "seed_visual_classifieds_exploration",
        }
        dump_json(cache_path, payload)
        return {
            "cache_hit": False,
            "cache_path": cache_relpath,
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "raw_output": raw_output,
            "usage": zero_usage(),
            "skill": exploratory_skill,
            "parse_error": None,
            "generation_mode": "seed_visual_classifieds_exploration",
        }

    if is_visual_current_page_comparison_task(task_item, seed_observation) and needs_visual_current_page_exploration(
        seed_observation.get("goal", "")
    ):
        exploratory_skill = build_current_page_visual_exploration_skill(task_item)
        raw_output = json.dumps(exploratory_skill, ensure_ascii=False, indent=2)
        payload = {
            "task_id": task_item["task_id"],
            "env_name": task_item["env_name"],
            "model": glm.model,
            "split_name": split_path.stem,
            "prompt_hash": prompt_hash,
            "raw_output": raw_output,
            "usage": zero_usage(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "skill": exploratory_skill,
            "parse_error": None,
            "generation_mode": "seed_current_page_visual_exploration",
        }
        dump_json(cache_path, payload)
        return {
            "cache_hit": False,
            "cache_path": cache_relpath,
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "raw_output": raw_output,
            "usage": zero_usage(),
            "skill": exploratory_skill,
            "parse_error": None,
            "generation_mode": "seed_current_page_visual_exploration",
        }

    if is_current_page_answer_task(task_item, seed_observation) and not is_visual_current_page_comparison_task(
        task_item, seed_observation
    ):
        extracted_answer, usage, raw_text, generation_mode = extract_current_page_answer_with_fallback(
            glm,
            env,
            task_item=task_item,
            observation=seed_observation,
        )
        if extracted_answer:
            direct_skill = build_direct_answer_skill(task_item, extracted_answer)
            payload = {
                "task_id": task_item["task_id"],
                "env_name": task_item["env_name"],
                "model": glm.model,
                "split_name": split_path.stem,
                "prompt_hash": prompt_hash,
                "raw_output": raw_text,
                "usage": usage,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "skill": direct_skill,
                "parse_error": None,
                "generation_mode": generation_mode,
            }
            dump_json(cache_path, payload)
            return {
                "cache_hit": False,
                "cache_path": cache_relpath,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "raw_output": raw_text,
                "usage": usage,
                "skill": direct_skill,
                "parse_error": None,
                "generation_mode": generation_mode,
            }

    raw_output, usage = ask_model_for_skill_with_usage(
        glm,
        text_prompt=prompt,
        observation=seed_observation,
    )
    payload: dict[str, Any] = {
        "task_id": task_item["task_id"],
        "env_name": task_item["env_name"],
        "model": glm.model,
        "split_name": split_path.stem,
        "prompt_hash": prompt_hash,
        "raw_output": raw_output,
        "usage": usage,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        payload["skill"] = parse_skill_response(raw_output)
        payload["parse_error"] = None
    except ValueError as exc:
        payload["skill"] = None
        payload["parse_error"] = str(exc)

    dump_json(cache_path, payload)
    return {
        "cache_hit": False,
        "cache_path": cache_relpath,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "raw_output": raw_output,
        "usage": usage,
        "skill": payload.get("skill"),
        "parse_error": payload.get("parse_error"),
    }


def save_skill_artifact(
    *,
    baseline: str,
    run_id: str,
    task_id: str,
    label: str,
    skill: dict[str, Any],
) -> str:
    path = SKILL_ROOT / baseline / run_id / task_id / f"{label}.json"
    dump_json(path, skill)
    return path.relative_to(REPO_ROOT).as_posix()


def get_task_seed(task_item: dict[str, Any]) -> int:
    value = task_item.get("seed", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_task_metadata(task_item: dict[str, Any]) -> dict[str, Any]:
    sites = list(task_item.get("sites", []) or [])
    site_label = "+".join(sites) if sites else "unknown"
    metadata = {
        "task_family": task_item.get("task_family", ""),
        "task_category": task_item.get("task_category", task_item.get("task_family", "")),
        "task_template": task_item.get("task_template", task_item.get("task_id", "")),
        "sites": sites,
        "site_label": site_label,
        "primary_site": sites[0] if len(sites) == 1 else ("multi_site" if sites else "unknown"),
        "site_scope": task_item.get("site_scope", ""),
        "statefulness": task_item.get("statefulness", ""),
        "goal_modality": task_item.get("goal_modality", ""),
    }
    for key in ("instance_id", "repeat_id", "seed", "benchmark_name"):
        if key in task_item:
            metadata[key] = task_item[key]
    return metadata


def summarize_result_bucket(bucket: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(bucket)
    success_count = sum(1 for result in bucket if result["success"])
    infra_failed_count = sum(1 for result in bucket if result.get("failure_source") == "infra_failed")
    total_prompt_tokens = sum(_to_float(result.get("prompt_tokens")) for result in bucket)
    total_completion_tokens = sum(_to_float(result.get("completion_tokens")) for result in bucket)
    total_tokens = sum(_to_float(result.get("total_tokens")) for result in bucket)
    total_cost = sum(_to_float(result.get("estimated_cost_usd")) for result in bucket)
    return {
        "total_tasks": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "infra_failed_count": infra_failed_count,
        "infra_failed_rate": infra_failed_count / total if total else 0.0,
        "average_steps": sum(result["steps_taken"] for result in bucket) / total if total else 0.0,
        "average_model_calls": sum(result["model_calls"] for result in bucket) / total if total else 0.0,
        "average_live_model_calls": (
            sum(_to_float(result.get("live_model_calls")) for result in bucket) / total if total else 0.0
        ),
        "average_repair_count": sum(result["repair_count"] for result in bucket) / total if total else 0.0,
        "average_patch_count": (
            sum(_to_float(result.get("patch_count")) for result in bucket) / total if total else 0.0
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_estimated_cost_usd": total_cost,
        "average_prompt_tokens": total_prompt_tokens / total if total else 0.0,
        "average_completion_tokens": total_completion_tokens / total if total else 0.0,
        "average_total_tokens": total_tokens / total if total else 0.0,
        "average_estimated_cost_usd": total_cost / total if total else 0.0,
    }


def build_site_summary_rows(site_summary: dict[str, dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    ordered_sites = [site for site in CANONICAL_SITES if site in site_summary]
    extra_sites = sorted(site for site in site_summary if site not in CANONICAL_SITES)
    for site in ordered_sites + extra_sites:
        rows.append({"site": site, **site_summary[site]})
    return rows


def run_noskill_task(
    glm: GLMClient,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    run_id: str,
    max_steps: int,
    reset_timeout_sec: int,
    step_timeout_sec: int,
    reset_retries: int,
    *,
    baseline_label: str = "no_skill",
    max_model_calls: int | None = None,
    disable_answer_extractor: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = task_item["task_id"]
    task_seed = get_task_seed(task_item)
    task_metadata = get_task_metadata(task_item)
    task_key = build_task_key(run_id, "no_skill", task_id, "attempt_00")
    trace = {
        "run_id": run_id,
        "baseline": baseline_label,
        "task_id": task_id,
        "env_name": task_item["env_name"],
        "notes": task_item["notes"],
        "model": glm.model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
        "model_calls": 0,
        "live_model_calls": 0,
        **task_metadata,
    }
    usage_totals = zero_usage()

    observation = reset_env_with_watchdog(
        env,
        task_item["env_name"],
        task_key=task_key,
        seed=task_seed,
        reset_timeout_sec=reset_timeout_sec,
        reset_retries=reset_retries,
    )
    trace["initial_observation"] = trace_observation(observation)

    success = False
    fail_reason = ""
    final_action_error = observation.get("last_action_error", "")
    steps_taken = 0

    effective_max_steps = max_steps
    if max_model_calls is not None and max_model_calls > 0:
        effective_max_steps = min(max_steps, max_model_calls)

    for step_index in range(1, effective_max_steps + 1):
        steps_taken = step_index
        step_record = {
            "step_index": step_index,
            "observation": trace_observation(observation),
        }

        if max_model_calls is not None and max_model_calls > 0 and trace["model_calls"] >= max_model_calls:
            fail_reason = "max_model_calls_exceeded"
            trace["steps"].append(step_record)
            break

        if (
            step_index == 1
            and not disable_answer_extractor
            and is_current_page_answer_task(task_item, observation)
        ):
            answer_text, call_usage, raw_answer, extractor_mode = extract_current_page_answer_with_fallback(
                glm,
                env,
                task_item=task_item,
                observation=observation,
            )
            step_record["answer_extractor_raw"] = raw_answer
            step_record["answer_extractor_normalized"] = answer_text
            step_record["answer_extractor_mode"] = extractor_mode
            if answer_text:
                raw_action = f"STOP[{answer_text}]"
            else:
                screenshot_path = REPO_ROOT / observation["screenshot_path"]
                raw_action, call_usage = ask_model_with_images_and_usage(
                    glm,
                    system_prompt=ACTION_SYSTEM_PROMPT,
                    text_prompt=build_action_user_prompt(task_item, observation),
                    screenshot_path=screenshot_path,
                    goal_image_urls=observation.get("goal_image_urls") or [],
                )
        else:
            screenshot_path = REPO_ROOT / observation["screenshot_path"]
            raw_action, call_usage = ask_model_with_images_and_usage(
                glm,
                system_prompt=ACTION_SYSTEM_PROMPT,
                text_prompt=build_action_user_prompt(task_item, observation),
                screenshot_path=screenshot_path,
                goal_image_urls=observation.get("goal_image_urls") or [],
            )
        trace["model_calls"] += 1
        trace["live_model_calls"] += 1
        add_usage(usage_totals, call_usage)
        step_record["raw_action"] = raw_action
        step_record["token_usage"] = call_usage

        try:
            parsed_action = parse_action(raw_action)
        except ValueError as exc:
            fail_reason = f"invalid_action_format: {exc}"
            step_record["parse_error"] = str(exc)
            trace["steps"].append(step_record)
            break

        step_record["parsed_action"] = parsed_action
        translation = env.compile_action(parsed_action)
        step_record["translation"] = translation

        if not translation["success"]:
            fail_reason = f"translator_error: {translation['message']}"
            trace["steps"].append(step_record)
            break

        try:
            next_observation, step_result = step_env_with_watchdog(
                env,
                translation,
                task_key=task_key,
                state_index=step_index,
                step_timeout_sec=step_timeout_sec,
            )
        except Exception as exc:
            fail_reason = f"browsergym_step_error: {exc}"
            step_record["env_step_error"] = str(exc)
            trace["steps"].append(step_record)
            break

        step_record["env_step"] = step_result
        step_record["post_action_observation"] = trace_observation(next_observation)
        trace["steps"].append(step_record)

        observation = next_observation
        final_action_error = observation.get("last_action_error", "")

        if should_treat_navigation_timeout_as_success(
            task_item,
            previous_observation=step_record["observation"],
            next_observation=next_observation,
            step_result=step_result,
            final_action_error=final_action_error,
            skill_step=None,
        ):
            step_record["timeout_navigation_promoted_to_success"] = True
            success = True
            fail_reason = ""
            break

        if step_result["success"]:
            success = True
            fail_reason = ""
            break

        if observation.get("terminated") or observation.get("truncated"):
            fail_reason = step_result["fail_reason"] or "environment_terminated_without_success"
            break

    if not success and not fail_reason:
        fail_reason = "max_steps_exceeded"
        if final_action_error:
            fail_reason = f"{fail_reason}: {final_action_error}"

    failure_info = localize_failure(trace["steps"], fail_reason)
    trace["final_observation"] = trace_observation(observation)
    trace["success"] = success
    trace["fail_reason"] = fail_reason
    trace["steps_taken"] = steps_taken
    trace["final_action_error"] = final_action_error
    trace["failure_info"] = failure_info
    trace["token_usage"] = usage_totals
    trace["finished_at"] = datetime.now().isoformat(timespec="seconds")

    result = {
        "task_id": task_id,
        "env_name": task_item["env_name"],
        "baseline": baseline_label,
        **task_metadata,
        "success": success,
        "initial_skill_success": None,
        "post_repair_success": None,
        "steps_taken": steps_taken,
        "model_calls": trace["model_calls"],
        "live_model_calls": trace["live_model_calls"],
        "repair_count": 0,
        "patch_count": 0,
        "final_fail_reason": fail_reason,
        "final_action_error": final_action_error,
        "initial_skill_path": None,
        "final_skill_path": None,
        "repair_skill_paths": [],
        "patch_types": [],
        **usage_totals,
        **failure_info,
    }
    return result, trace


def execute_skill_attempt(
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    skill: dict[str, Any],
    *,
    task_key: str,
    max_steps: int,
    seed: int,
    reset_timeout_sec: int,
    step_timeout_sec: int,
    reset_retries: int,
) -> dict[str, Any]:
    observation = reset_env_with_watchdog(
        env,
        task_item["env_name"],
        task_key=task_key,
        seed=seed,
        reset_timeout_sec=reset_timeout_sec,
        reset_retries=reset_retries,
    )
    attempt_trace: dict[str, Any] = {
        "attempt_label": task_key.rsplit("/", 1)[-1],
        "skill_id": skill["skill_id"],
        "initial_observation": trace_observation(observation),
        "precondition_status": observation_contract_status(observation, skill["preconditions"]),
        "steps": [],
    }

    if not attempt_trace["precondition_status"]["success"]:
        fail_reason = "precondition_failed: " + "; ".join(attempt_trace["precondition_status"]["unmet"])
        failure_info = localize_failure([], fail_reason)
        return {
            "success": False,
            "fail_reason": fail_reason,
            "final_action_error": observation.get("last_action_error", ""),
            "steps_taken": 0,
            "final_observation": observation,
            "execution_trace": attempt_trace,
            "failure_info": failure_info,
            "success_contract_status": {"success": False, "unmet": ["preconditions_not_met"]},
        }

    success = False
    fail_reason = ""
    steps_taken = 0
    final_action_error = observation.get("last_action_error", "")
    has_success_contract = any(skill.get("success_contract", {}).values())
    success_contract_status = (
        observation_contract_status(observation, skill["success_contract"])
        if has_success_contract
        else {"success": False, "unmet": []}
    )

    for step_index, skill_step in enumerate(skill["steps"], start=1):
        if step_index > max_steps:
            fail_reason = "max_steps_exceeded"
            break

        steps_taken = step_index
        step_record = {
            "step_index": step_index,
            "observation": trace_observation(observation),
            "skill_step": skill_step,
            "raw_action": skill_to_action_string(skill_step),
        }

        try:
            parsed_action = parse_action(step_record["raw_action"])
        except ValueError as exc:
            fail_reason = f"invalid_action_format: {exc}"
            step_record["parse_error"] = str(exc)
            attempt_trace["steps"].append(step_record)
            break

        step_record["parsed_action"] = parsed_action
        translation = env.compile_action(parsed_action)
        step_record["translation"] = translation

        if not translation["success"]:
            fail_reason = f"translator_error: {translation['message']}"
            attempt_trace["steps"].append(step_record)
            break

        try:
            next_observation, step_result = step_env_with_watchdog(
                env,
                translation,
                task_key=task_key,
                state_index=step_index,
                step_timeout_sec=step_timeout_sec,
            )
        except Exception as exc:
            fail_reason = f"browsergym_step_error: {exc}"
            step_record["env_step_error"] = str(exc)
            attempt_trace["steps"].append(step_record)
            break

        step_record["env_step"] = step_result
        step_record["post_action_observation"] = trace_observation(next_observation)
        attempt_trace["steps"].append(step_record)

        observation = next_observation
        final_action_error = observation.get("last_action_error", "")
        success_contract_status = (
            observation_contract_status(observation, skill["success_contract"])
            if has_success_contract
            else {"success": False, "unmet": []}
        )
        step_record["success_contract_status"] = success_contract_status

        if should_treat_shopping_checkout_terminal_as_success(
            task_item,
            observation=observation,
        ):
            step_record["shopping_checkout_terminal_promoted_to_success"] = True
            success = True
            fail_reason = ""
            break

        if step_result["success"]:
            success = True
            fail_reason = ""
            break

        if observation.get("terminated") or observation.get("truncated"):
            fail_reason = step_result["fail_reason"] or "environment_terminated_without_success"
            break

    if not success and not fail_reason:
        if has_success_contract and not success_contract_status["success"]:
            fail_reason = "success_contract_failed: " + "; ".join(success_contract_status["unmet"])
        elif has_success_contract:
            fail_reason = "contract_satisfied_but_reward_zero"
        else:
            fail_reason = "skill_exhausted_without_success"

    failure_info = localize_failure(attempt_trace["steps"], fail_reason)
    attempt_trace["final_observation"] = trace_observation(observation)
    attempt_trace["success"] = success
    attempt_trace["fail_reason"] = fail_reason
    attempt_trace["steps_taken"] = steps_taken
    attempt_trace["final_action_error"] = final_action_error
    attempt_trace["success_contract_status"] = success_contract_status
    attempt_trace["failure_info"] = failure_info

    return {
        "success": success,
        "fail_reason": fail_reason,
        "final_action_error": final_action_error,
        "steps_taken": steps_taken,
        "final_observation": observation,
        "execution_trace": attempt_trace,
        "failure_info": failure_info,
        "success_contract_status": success_contract_status,
    }


def run_skill_baseline_task(
    glm: GLMClient,
    env: VisualWebArenaEnv,
    task_item: dict[str, Any],
    run_id: str,
    split_path: Path,
    *,
    initial_skill_cache_key: str | None = None,
    initial_skill_map: dict[str, Path] | None = None,
    baseline: str,
    max_steps: int,
    max_repairs: int,
    max_model_calls: int,
    reset_timeout_sec: int,
    step_timeout_sec: int,
    reset_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = task_item["task_id"]
    task_seed = get_task_seed(task_item)
    task_metadata = get_task_metadata(task_item)
    trace: dict[str, Any] = {
        "run_id": run_id,
        "baseline": baseline,
        "task_id": task_id,
        "env_name": task_item["env_name"],
        "notes": task_item.get("notes", ""),
        "model": glm.model,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model_calls": 0,
        "live_model_calls": 0,
        "attempts": [],
        "repairs": [],
        **task_metadata,
    }
    usage_totals = zero_usage()
    live_usage_totals = zero_usage()
    effective_max_repairs, effective_max_model_calls = effective_repair_limits(
        glm.model,
        max_repairs,
        max_model_calls,
    )
    conservative_repair_model = is_conservative_repair_model(glm.model)

    seed_observation = reset_env_with_watchdog(
        env,
        task_item["env_name"],
        task_key=build_task_key(run_id, baseline, task_id, "seed"),
        seed=task_seed,
        reset_timeout_sec=reset_timeout_sec,
        reset_retries=reset_retries,
    )
    trace["seed_observation"] = trace_observation(seed_observation)

    initial_skill_record = load_or_create_shared_initial_skill(
        glm,
        env,
        task_item=task_item,
        split_path=split_path,
        cache_key=initial_skill_cache_key,
        injected_skill_map=initial_skill_map,
        seed_observation=seed_observation,
        max_steps=max_steps,
    )
    initial_prompt = initial_skill_record["prompt"]
    raw_skill = initial_skill_record["raw_output"]
    counts_as_model_call = initial_skill_record.get("counts_as_model_call", True)
    counts_as_live_model_call = initial_skill_record.get(
        "counts_as_live_model_call",
        not initial_skill_record["cache_hit"],
    )
    if counts_as_model_call:
        trace["model_calls"] += 1
    if counts_as_live_model_call:
        trace["live_model_calls"] += 1
        add_usage(live_usage_totals, initial_skill_record["usage"])
    add_usage(usage_totals, initial_skill_record["usage"])
    trace["initial_skill_cache"] = {
        "path": initial_skill_record["cache_path"],
        "cache_hit": initial_skill_record["cache_hit"],
        "prompt_hash": initial_skill_record["prompt_hash"],
        "usage": initial_skill_record["usage"],
        "generation_mode": initial_skill_record.get("generation_mode", "model_generation"),
    }
    trace["initial_skill_prompt"] = initial_prompt
    trace["initial_skill_raw_output"] = raw_skill

    parse_error = initial_skill_record.get("parse_error")
    current_skill = initial_skill_record.get("skill")
    shopping_bootstrap = None
    if baseline in CONTRACT_REPAIR_BASELINES:
        shopping_bootstrap = build_shopping_heuristic_candidate(
            glm=glm,
            env=env,
            task_item=task_item,
            current_skill=current_skill or {
                "skill_id": task_item["task_id"],
                "task": task_item.get("intent", task_item["task_id"]),
                "preconditions": {},
                "success_contract": {},
                "steps": [],
                "repair_history": [],
                "patches": [],
            },
            observation=seed_observation,
            base_observation=seed_observation,
        )
        if shopping_bootstrap is not None:
            current_skill = shopping_bootstrap["skill"]

    if parse_error:
        if shopping_bootstrap is not None and current_skill is not None:
            parse_error = None
        else:
            fail_reason = f"invalid_skill_json: {parse_error}"
            failure_info = localize_failure([], fail_reason)
            trace["success"] = False
            trace["fail_reason"] = fail_reason
            trace["failure_info"] = failure_info
            trace["token_usage"] = usage_totals
            trace["live_token_usage"] = live_usage_totals
            trace["finished_at"] = datetime.now().isoformat(timespec="seconds")
            result = {
                "task_id": task_id,
                "env_name": task_item["env_name"],
                "baseline": baseline,
                **task_metadata,
                "success": False,
                "initial_skill_success": False,
                "post_repair_success": False,
                "steps_taken": 0,
                "model_calls": trace["model_calls"],
                "live_model_calls": trace["live_model_calls"],
                "repair_count": 0,
                "patch_count": 0,
                "final_fail_reason": fail_reason,
                "final_action_error": "",
                "initial_skill_path": None,
                "final_skill_path": None,
                "repair_skill_paths": [],
                "patch_types": [],
                **usage_totals,
                "live_prompt_tokens": live_usage_totals["prompt_tokens"],
                "live_completion_tokens": live_usage_totals["completion_tokens"],
                "live_total_tokens": live_usage_totals["total_tokens"],
                "live_estimated_cost_usd": live_usage_totals["estimated_cost_usd"],
                **failure_info,
            }
            return result, trace

    if baseline in M3_STABILIZED_BASELINES:
        current_skill = stabilize_contractskill_miniwob_skill(
            task_item,
            current_skill,
            observation=seed_observation,
        )

    initial_skill_path = save_skill_artifact(
        baseline=baseline,
        run_id=run_id,
        task_id=task_id,
        label="skill_initial",
        skill=current_skill,
    )
    trace["initial_skill"] = current_skill
    trace["initial_skill_path"] = initial_skill_path

    attempt = execute_skill_attempt(
        env,
        task_item,
        current_skill,
        task_key=build_task_key(run_id, baseline, task_id, "attempt_00"),
        max_steps=max_steps,
        seed=task_seed,
        reset_timeout_sec=reset_timeout_sec,
        step_timeout_sec=step_timeout_sec,
        reset_retries=reset_retries,
    )
    trace["attempts"].append(attempt["execution_trace"])

    initial_skill_success = attempt["success"]
    success = attempt["success"]
    final_fail_reason = attempt["fail_reason"]
    final_action_error = attempt["final_action_error"]
    steps_taken = attempt["steps_taken"]
    failure_info = attempt["failure_info"]
    repair_paths: list[str] = []
    repair_count = 0
    final_skill_path = initial_skill_path

    if not success and baseline in {
        "text_only_rewrite",
        "contractskill",
        "contractskill_no_patch_constraints",
        "contractskill_no_failure_localization",
        "contractskill_unconstrained_repair",
    }:
        repair_limit = 1 if baseline == "text_only_rewrite" else effective_max_repairs
        latest_observation = attempt["final_observation"]
        site_name = primary_site_for_task(task_item)

        for repair_round in range(1, repair_limit + 1):
            if trace["model_calls"] >= effective_max_model_calls:
                final_fail_reason = "max_model_calls_exceeded"
                failure_info = localize_failure([], final_fail_reason)
                break

            heuristic_repair = None
            if baseline in CONTRACT_REPAIR_BASELINES:
                heuristic_skill = build_translator_error_target_repair(
                    current_skill,
                    latest_observation,
                    failure_info,
                    site=site_name,
                )
                if heuristic_skill is not None:
                    heuristic_repair = {
                        "skill": heuristic_skill,
                        "usage": zero_usage(),
                        "raw_output": json.dumps(heuristic_skill, ensure_ascii=False),
                        "source": "translator_target_replacement",
                    }
                if heuristic_repair is None:
                    heuristic_repair = maybe_build_shopping_grid_repair(
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        seed_observation=seed_observation,
                        latest_observation=latest_observation,
                        failure_info=failure_info,
                        current_skill=current_skill,
                    )
                if heuristic_repair is None:
                    heuristic_repair = maybe_build_classifieds_search_repair(
                        glm,
                        env=env,
                        task_item=task_item,
                        seed_observation=seed_observation,
                        latest_observation=latest_observation,
                        failure_info=failure_info,
                        current_skill=current_skill,
                    )
                if heuristic_repair is None:
                    heuristic_repair = maybe_build_classifieds_visual_navigation_repair(
                        glm,
                        env=env,
                        task_item=task_item,
                        latest_observation=latest_observation,
                        failure_info=failure_info,
                        current_skill=current_skill,
                    )
                if heuristic_repair is None:
                    heuristic_repair = maybe_build_reddit_navigation_repair(
                        glm,
                        task_item=task_item,
                        seed_observation=seed_observation,
                        latest_observation=latest_observation,
                        failure_info=failure_info,
                        current_skill=current_skill,
                        execution_trace=attempt["execution_trace"]["steps"],
                    )

            if heuristic_repair is not None:
                raw_repair = heuristic_repair["raw_output"]
                repair_usage = heuristic_repair["usage"]
                if _to_float(repair_usage.get("total_tokens")) > 0:
                    trace["model_calls"] += 1
                    trace["live_model_calls"] += 1
                repair_prompt = None
            else:
                if baseline == "text_only_rewrite":
                    repair_prompt = build_text_rewrite_prompt(
                        task_item=task_item,
                        latest_observation=latest_observation,
                        current_skill=current_skill,
                        failure_info=failure_info,
                        execution_trace=attempt["execution_trace"]["steps"],
                        max_steps=max_steps,
                    )
                else:
                    repair_prompt = build_contract_repair_prompt(
                        task_item=task_item,
                        latest_observation=latest_observation,
                        current_skill=current_skill,
                        failure_info=failure_info,
                        execution_trace=attempt["execution_trace"]["steps"],
                        max_steps=max_steps,
                        repair_round=repair_round,
                        include_failure_localization=baseline in LOCALIZED_REPAIR_BASELINES,
                        constrained_patch_repair=baseline != "contractskill_unconstrained_repair",
                        include_structured_repair_context=baseline != "contractskill_unconstrained_repair",
                    )
                    if conservative_repair_model:
                        failed_step_index = int(failure_info.get("failed_step_index") or 0)
                        repair_prompt += (
                            "\nConservative repair mode:\n"
                            "1. Make exactly one minimal executable repair.\n"
                            "2. Keep the existing working steps unchanged.\n"
                            "3. Do not rewrite the whole skill.\n"
                        )
                        if failed_step_index > 1:
                            repair_prompt += (
                                f"4. Steps 1-{failed_step_index - 1} already executed. "
                                "Keep them unchanged and only repair the suffix.\n"
                            )

                raw_repair, repair_usage = ask_model_for_skill_with_usage(
                    glm,
                    text_prompt=repair_prompt,
                    observation=latest_observation,
                )
                trace["model_calls"] += 1
                trace["live_model_calls"] += 1
            add_usage(usage_totals, repair_usage)
            add_usage(live_usage_totals, repair_usage)

            repair_record: dict[str, Any] = {
                "repair_round": repair_round,
                "raw_output": raw_repair,
                "failure_info": failure_info,
                "token_usage": repair_usage,
            }

            if heuristic_repair is not None:
                repaired_skill = heuristic_repair["skill"]
                repair_record["repair_source"] = heuristic_repair["source"]
            else:
                try:
                    repaired_skill = parse_skill_response(raw_repair)
                except ValueError as exc:
                    repair_record["parse_error"] = str(exc)
                    trace["repairs"].append(repair_record)
                    if repair_round < repair_limit:
                        current_skill = append_repair_history_note(
                            current_skill,
                            f"repair retry note: previous repair output was invalid JSON ({exc}). "
                            "Next repair must return a valid skill with a non-empty executable steps list.",
                        )
                        continue
                    final_fail_reason = f"invalid_skill_json: {exc}"
                    failure_info = localize_failure([], final_fail_reason)
                    break

            if baseline in NAV_PREFIX_REPAIR_BASELINES:
                repaired_skill = preserve_navigation_prefix_for_repair(
                    current_skill,
                    repaired_skill,
                    trace["seed_observation"],
                    latest_observation,
                )
            repaired_skill = preserve_successful_prefix_for_repair(
                current_skill,
                repaired_skill,
                failure_info,
            )
            precondition_status = (
                attempt["execution_trace"].get("precondition_status", {})
                if isinstance(attempt.get("execution_trace"), dict)
                else {}
            )
            if precondition_status.get("success"):
                repaired_skill["preconditions"] = current_skill.get("preconditions", {})
                repair_record["preserved_preconditions"] = True
            if baseline in M3_STABILIZED_BASELINES:
                repaired_skill = stabilize_contractskill_miniwob_skill(
                    task_item,
                    repaired_skill,
                    observation=latest_observation,
                )

            shopping_sites = [str(site).lower() for site in task_item.get("sites", []) or []]
            previous_steps = current_skill.get("steps", []) or []
            repaired_steps = repaired_skill.get("steps", []) or []
            if shopping_sites == ["shopping"] and previous_steps and repaired_steps:
                previous_first = previous_steps[0]
                repaired_first = repaired_steps[0]
                previous_target = str(previous_first.get("target") or "").strip()
                repaired_target = str(repaired_first.get("target") or "").strip()
                if (
                    previous_first.get("action") == "CLICK"
                    and repaired_first.get("action") == "CLICK"
                    and _looks_product_like_target(previous_target)
                    and _is_generic_shopping_target(repaired_target)
                ):
                    repair_record["shopping_navigation_regression"] = {
                        "previous_target": previous_target,
                        "repaired_target": repaired_target,
                    }
                    trace["repairs"].append(repair_record)
                    if repair_round < repair_limit:
                        current_skill = append_repair_history_note(
                            current_skill,
                            "repair retry note: previous repair replaced a specific shopping product target "
                            f"'{previous_target}' with generic navigation '{repaired_target}'. "
                            "Next repair must keep or improve the product-level target.",
                        )
                        continue
                    final_fail_reason = "shopping_repair_regressed_to_generic_navigation"
                    failure_info = localize_failure([], final_fail_reason)
                    break
                previous_product_target = _first_shopping_product_target(previous_steps)
                repaired_product_target = _first_shopping_product_target(repaired_steps)
                if previous_product_target and not repaired_product_target:
                    first_click_target = next(
                        (
                            str(step.get("target") or "").strip()
                            for step in repaired_steps
                            if str(step.get("action") or "").upper() == "CLICK"
                        ),
                        "",
                    )
                    if not first_click_target or _is_generic_shopping_target(first_click_target):
                        repair_record["shopping_product_regression"] = {
                            "previous_product_target": previous_product_target,
                            "repaired_first_click_target": first_click_target,
                        }
                        trace["repairs"].append(repair_record)
                        final_fail_reason = "shopping_repair_regressed_from_product_to_generic"
                        failure_info = localize_failure([], final_fail_reason)
                        break

            repair_target_observation = (
                seed_observation
                if shopping_sites == ["shopping"]
                else latest_observation
            )
            if baseline in CONTRACT_REPAIR_BASELINES:
                repaired_skill = repair_targets_against_observation(
                    current_skill,
                    repaired_skill,
                    repair_target_observation,
                    site=site_name,
                )
                repaired_steps = repaired_skill.get("steps", []) or []
            invalid_target_violations = introduced_invalid_repair_targets(
                current_skill,
                repaired_skill,
                repair_target_observation,
                failure_info=failure_info,
            )
            if (
                repair_record.get("repair_source") in {"shopping_grid_structured", "shopping_compare_structured"}
                and invalid_target_violations
            ):
                scroll_assisted_indices = shopping_scroll_assisted_step_indices(repaired_skill)
                invalid_target_violations = [
                    violation
                    for violation in invalid_target_violations
                    if int(violation.get("step_index") or 0) not in scroll_assisted_indices
                ]
            if invalid_target_violations:
                repair_record["invalid_target_violations"] = invalid_target_violations
                trace["repairs"].append(repair_record)
                if repair_round < repair_limit:
                    invalid_target_summary = ", ".join(
                        f"step {int(violation.get('step_index') or 0)} -> {violation.get('action')}[{violation.get('target')}]"
                        for violation in invalid_target_violations[:4]
                    )
                    current_skill = append_repair_history_note(
                        current_skill,
                        "repair retry note: previous repair introduced immediate-prefix targets not grounded in "
                        f"the latest observation ({invalid_target_summary}). Keep the reachable repaired prefix "
                        "strictly executable now, and defer future-page targets to later suffix steps.",
                    )
                    continue
                final_fail_reason = "repair_target_not_in_observation"
                failure_info = localize_failure([], final_fail_reason)
                break

            blocked_generic_targets = [
                {
                    "step_index": index,
                    "target": str(step.get("target") or "").strip(),
                }
                for index, step in enumerate(repaired_steps, start=1)
                if str(step.get("action") or "").upper() == "CLICK"
                and should_block_generic_repair_target(site_name, str(step.get("target") or ""))
                and (
                    index > len(previous_steps)
                    or str((previous_steps[index - 1] or {}).get("action") or "").upper() != "CLICK"
                    or str((previous_steps[index - 1] or {}).get("target") or "").strip().lower()
                    != str(step.get("target") or "").strip().lower()
                )
            ]
            if blocked_generic_targets:
                repair_record["blocked_generic_targets"] = blocked_generic_targets
                trace["repairs"].append(repair_record)
                final_fail_reason = "repair_regressed_to_generic_navigation"
                failure_info = localize_failure([], final_fail_reason)
                break

            diff_summary = summarize_skill_diff(current_skill, repaired_skill)
            repair_record["diff_summary"] = diff_summary
            repair_record["skill"] = repaired_skill

            implicit_nav_violations = detect_implicit_navigation_targets(
                repaired_skill,
                latest_observation,
            )
            if implicit_nav_violations:
                repair_record["implicit_navigation_violations"] = implicit_nav_violations
                trace["repairs"].append(repair_record)
                if repair_round < repair_limit:
                    current_skill = append_repair_history_note(
                        current_skill,
                        "repair retry note: previous repair used implicit browser navigation targets "
                        f"{', '.join(implicit_nav_violations)}. Next repair must use only visible page targets.",
                    )
                    continue
                final_fail_reason = "repair_invalid_navigation_target"
                failure_info = localize_failure([], final_fail_reason)
                break

            if skills_equivalent(current_skill, repaired_skill):
                repair_record["equivalent_to_previous"] = True
                trace["repairs"].append(repair_record)
                if repair_round < repair_limit:
                    current_skill = append_repair_history_note(
                        current_skill,
                        "repair retry note: previous repair made no structural change. "
                        "Next repair must change at least one executable step or precondition.",
                    )
                    continue
                final_fail_reason = "repair_no_meaningful_change"
                failure_info = localize_failure([], final_fail_reason)
                break

            if baseline in PATCH_CONSTRAINED_BASELINES:
                if has_execution_equivalent_update(current_skill, repaired_skill):
                    repair_record["execution_equivalent_to_previous"] = True
                    trace["repairs"].append(repair_record)
                    if repair_round < repair_limit:
                        current_skill = append_repair_history_note(
                            current_skill,
                            "repair retry note: previous repair changed only non-executable fields. "
                            "Next repair must change an executable step or precondition.",
                        )
                        continue
                    final_fail_reason = "repair_no_meaningful_change"
                    failure_info = localize_failure([], final_fail_reason)
                    break

                if baseline in FAILURE_AWARE_PATCH_POLICY_BASELINES:
                    allowed_patch_types = allowed_patch_types_for_failure(failure_info)
                    repaired_patch_types = {
                        patch["type"] for patch in repaired_skill.get("patches", []) if patch.get("type")
                    }
                    if repaired_patch_types and repaired_patch_types.isdisjoint(allowed_patch_types):
                        final_fail_reason = "repair_patch_policy_violation"
                        failure_info = localize_failure([], final_fail_reason)
                        repair_record["allowed_patch_types"] = sorted(allowed_patch_types)
                        repair_record["observed_patch_types"] = sorted(repaired_patch_types)
                        trace["repairs"].append(repair_record)
                        break

            repair_skill_path = save_skill_artifact(
                baseline=baseline,
                run_id=run_id,
                task_id=task_id,
                label=f"skill_repair_{repair_round:02d}",
                skill=repaired_skill,
            )
            repair_paths.append(repair_skill_path)
            repair_record["skill_path"] = repair_skill_path
            trace["repairs"].append(repair_record)
            repair_count = repair_round

            current_skill = repaired_skill
            final_skill_path = repair_skill_path
            attempt = execute_skill_attempt(
                env,
                task_item,
                current_skill,
                task_key=build_task_key(run_id, baseline, task_id, f"attempt_{repair_round:02d}"),
                max_steps=max_steps,
                seed=task_seed,
                reset_timeout_sec=reset_timeout_sec,
                step_timeout_sec=step_timeout_sec,
                reset_retries=reset_retries,
            )
            trace["attempts"].append(attempt["execution_trace"])

            success = attempt["success"]
            final_fail_reason = attempt["fail_reason"]
            final_action_error = attempt["final_action_error"]
            steps_taken = attempt["steps_taken"]
            failure_info = attempt["failure_info"]
            latest_observation = attempt["final_observation"]

            if success:
                break

    trace["success"] = success
    trace["fail_reason"] = final_fail_reason
    trace["final_action_error"] = final_action_error
    trace["failure_info"] = failure_info
    trace["repair_count"] = repair_count
    trace["token_usage"] = usage_totals
    trace["live_token_usage"] = live_usage_totals
    trace["finished_at"] = datetime.now().isoformat(timespec="seconds")

    patch_types = [
        patch["type"]
        for repair_record in trace["repairs"]
        for patch in repair_record.get("skill", {}).get("patches", [])
    ]
    patch_count = len(patch_types)

    result = {
        "task_id": task_id,
        "env_name": task_item["env_name"],
        "baseline": baseline,
        **task_metadata,
        "success": success,
        "initial_skill_success": initial_skill_success,
        "post_repair_success": bool(success and repair_count > 0),
        "steps_taken": steps_taken,
        "model_calls": trace["model_calls"],
        "live_model_calls": trace["live_model_calls"],
        "repair_count": repair_count,
        "patch_count": patch_count,
        "final_fail_reason": final_fail_reason,
        "final_action_error": final_action_error,
        "initial_skill_path": initial_skill_path,
        "final_skill_path": final_skill_path,
        "repair_skill_paths": repair_paths,
        "patch_types": patch_types,
        **usage_totals,
        "live_prompt_tokens": live_usage_totals["prompt_tokens"],
        "live_completion_tokens": live_usage_totals["completion_tokens"],
        "live_total_tokens": live_usage_totals["total_tokens"],
        "live_estimated_cost_usd": live_usage_totals["estimated_cost_usd"],
        **failure_info,
    }
    return result, trace


def aggregate_summary(
    *,
    run_id: str,
    split_path: Path,
    baseline: str,
    model: str,
    max_steps: int,
    max_repairs: int,
    max_model_calls: int,
    headless: bool,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        split_path_value = str(split_path.relative_to(REPO_ROOT))
    except ValueError:
        split_path_value = str(split_path)

    total_tasks = len(results)
    success_count = sum(1 for result in results if result["success"])
    infra_failed_count = sum(1 for result in results if result.get("failure_source") == "infra_failed")
    average_steps = (sum(result["steps_taken"] for result in results) / total_tasks) if total_tasks else 0.0
    average_model_calls = (
        sum(result["model_calls"] for result in results) / total_tasks if total_tasks else 0.0
    )
    average_live_model_calls = (
        sum(_to_float(result.get("live_model_calls")) for result in results) / total_tasks
        if total_tasks
        else 0.0
    )
    average_repair_count = (
        sum(result["repair_count"] for result in results) / total_tasks if total_tasks else 0.0
    )
    average_patch_count = (
        sum(_to_float(result.get("patch_count")) for result in results) / total_tasks
        if total_tasks
        else 0.0
    )
    total_prompt_tokens = sum(_to_float(result.get("prompt_tokens")) for result in results)
    total_completion_tokens = sum(_to_float(result.get("completion_tokens")) for result in results)
    total_tokens = sum(_to_float(result.get("total_tokens")) for result in results)
    total_cost = sum(_to_float(result.get("estimated_cost_usd")) for result in results)
    total_live_prompt_tokens = sum(_to_float(result.get("live_prompt_tokens")) for result in results)
    total_live_completion_tokens = sum(
        _to_float(result.get("live_completion_tokens")) for result in results
    )
    total_live_tokens = sum(_to_float(result.get("live_total_tokens")) for result in results)
    total_live_cost = sum(_to_float(result.get("live_estimated_cost_usd")) for result in results)

    repair_trigger_count = sum(
        1
        for result in results
        if result["initial_skill_success"] is False and result["baseline"] != "no_skill"
    )
    repair_success_count = sum(1 for result in results if result["post_repair_success"])
    conditional_repair_success = (
        repair_success_count / repair_trigger_count if repair_trigger_count else 0.0
    )

    failure_source_distribution = Counter(
        result["failure_source"] for result in results if not result["success"]
    )
    patch_type_distribution = Counter(
        patch_type
        for result in results
        for patch_type in result.get("patch_types", [])
    )
    per_family_counts: dict[str, dict[str, float]] = {}
    family_buckets: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        family = str(result.get("task_family", "") or "unknown")
        family_buckets.setdefault(family, []).append(result)
    for family, family_results in family_buckets.items():
        per_family_counts[family] = summarize_result_bucket(family_results)

    site_buckets: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        site_label = str(result.get("site_label", "") or "unknown")
        site_buckets.setdefault(site_label, []).append(result)
    per_site_counts = {
        site_label: summarize_result_bucket(site_results)
        for site_label, site_results in site_buckets.items()
    }
    site_summary_rows = build_site_summary_rows(per_site_counts)

    return {
        "run_id": run_id,
        "baseline": baseline,
        "model": model,
        "split_name": split_path.stem,
        "split_path": split_path_value,
        "total_tasks": total_tasks,
        "success_count": success_count,
        "success_rate": success_count / total_tasks if total_tasks else 0.0,
        "infra_failed_count": infra_failed_count,
        "infra_failed_rate": infra_failed_count / total_tasks if total_tasks else 0.0,
        "average_steps": average_steps,
        "average_model_calls": average_model_calls,
        "average_live_model_calls": average_live_model_calls,
        "average_repair_count": average_repair_count,
        "average_patch_count": average_patch_count,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_estimated_cost_usd": total_cost,
        "average_prompt_tokens": total_prompt_tokens / total_tasks if total_tasks else 0.0,
        "average_completion_tokens": total_completion_tokens / total_tasks if total_tasks else 0.0,
        "average_total_tokens": total_tokens / total_tasks if total_tasks else 0.0,
        "average_estimated_cost_usd": total_cost / total_tasks if total_tasks else 0.0,
        "total_live_prompt_tokens": total_live_prompt_tokens,
        "total_live_completion_tokens": total_live_completion_tokens,
        "total_live_tokens": total_live_tokens,
        "total_live_estimated_cost_usd": total_live_cost,
        "average_live_prompt_tokens": total_live_prompt_tokens / total_tasks if total_tasks else 0.0,
        "average_live_completion_tokens": (
            total_live_completion_tokens / total_tasks if total_tasks else 0.0
        ),
        "average_live_total_tokens": total_live_tokens / total_tasks if total_tasks else 0.0,
        "average_live_estimated_cost_usd": total_live_cost / total_tasks if total_tasks else 0.0,
        "repair_trigger_count": repair_trigger_count,
        "repair_success_count": repair_success_count,
        "conditional_repair_success": conditional_repair_success,
        "failure_source_distribution": dict(failure_source_distribution),
        "patch_type_distribution": dict(patch_type_distribution),
        "per_family_summary": per_family_counts,
        "per_site_summary": per_site_counts,
        "site_summary_rows": site_summary_rows,
        "max_steps": max_steps,
        "max_repairs": max_repairs,
        "max_model_calls": max_model_calls,
        "headless": headless,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ContractSkill baselines on BrowserGym VisualWebArena.")
    parser.add_argument(
        "--baseline",
        choices=(
            "no_skill",
            "skill_no_repair",
            "text_only_rewrite",
            "contractskill",
            "contractskill_no_patch_constraints",
            "contractskill_no_failure_localization",
            "contractskill_unconstrained_repair",
        ),
        required=True,
    )
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS)
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument(
        "--browser-timeout-ms",
        type=int,
        default=DEFAULT_BROWSER_TIMEOUT_MS,
    )
    parser.add_argument(
        "--reset-timeout-sec",
        type=int,
        default=DEFAULT_RESET_TIMEOUT_SEC,
        help="Hard timeout for env.reset in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--step-timeout-sec",
        type=int,
        default=DEFAULT_STEP_TIMEOUT_SEC,
        help="Hard timeout for env.step in seconds. Use 0 to disable.",
    )
    parser.add_argument("--reset-retries", type=int, default=DEFAULT_RESET_RETRIES)
    parser.add_argument(
        "--task-timeout-sec",
        type=int,
        default=DEFAULT_TASK_TIMEOUT_SEC,
        help="Hard timeout for a full task attempt in seconds. Use 0 to disable.",
    )
    parser.add_argument("--headless", type=parse_bool, default=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--initial-skill-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping task_id to an existing skill artifact path for hard replay.",
    )
    return parser.parse_args()


def configure_api_env_for_model(model: str) -> Path | None:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("qwen") and DEFAULT_QWEN_API_ENV_PATH.exists():
        os.environ["API_ENV_FILE"] = str(DEFAULT_QWEN_API_ENV_PATH)
        loaded = load_api_env_file(DEFAULT_QWEN_API_ENV_PATH, overwrite=True)
        os.environ.setdefault("VWA_EVAL_TEXT_MODEL", str(model))
        os.environ.setdefault("VWA_CAPTION_MODEL", str(model))
        return loaded

    os.environ.pop("API_ENV_FILE", None)
    loaded = load_api_env_file(overwrite=False)
    if str(model or "").strip():
        os.environ.setdefault("VWA_EVAL_TEXT_MODEL", str(model))
        os.environ.setdefault("VWA_CAPTION_MODEL", str(model))
    return loaded


def main() -> None:
    args = parse_args()
    args.split_path = args.split_path.resolve()
    split = load_split(args.split_path)
    initial_skill_map = load_initial_skill_override_map(args.initial_skill_map)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    configured_api_env = configure_api_env_for_model(args.model)
    if configured_api_env is not None:
        print(f"Using API env file: {configured_api_env}")

    for root in (SCREENSHOT_ROOT, TRACE_ROOT, RESULT_ROOT, SKILL_ROOT, INITIAL_SKILL_ROOT):
        ensure_dir(root)

    glm = GLMClient(model=args.model)
    env = VisualWebArenaEnv(
        output_root=SCREENSHOT_ROOT / args.baseline / run_id,
        headless=args.headless,
        browser_timeout_ms=args.browser_timeout_ms,
    )
    env.start()
    env.set_task_config_overrides(split)

    results: list[dict[str, Any]] = []

    try:
        for task_item in split:
            print(f"\n[{args.baseline}] {task_item['task_id']} {task_item['env_name']}")
            try:
                if args.baseline == "no_skill":
                    result, trace = _run_with_timeout(
                        args.task_timeout_sec,
                        f"task {task_item['task_id']}",
                        run_noskill_task,
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        run_id=run_id,
                        max_steps=args.max_steps,
                        reset_timeout_sec=args.reset_timeout_sec,
                        step_timeout_sec=args.step_timeout_sec,
                        reset_retries=args.reset_retries,
                    )
                else:
                    result, trace = _run_with_timeout(
                        args.task_timeout_sec,
                        f"task {task_item['task_id']}",
                        run_skill_baseline_task,
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        run_id=run_id,
                        split_path=args.split_path,
                        baseline=args.baseline,
                        initial_skill_map=initial_skill_map,
                        max_steps=args.max_steps,
                        max_repairs=args.max_repairs,
                        max_model_calls=args.max_model_calls,
                        reset_timeout_sec=args.reset_timeout_sec,
                        step_timeout_sec=args.step_timeout_sec,
                        reset_retries=args.reset_retries,
                    )
            except Exception as exc:
                fail_reason = str(exc)
                failure_info = localize_failure([], fail_reason)
                result = {
                    "task_id": task_item["task_id"],
                    "env_name": task_item["env_name"],
                    "baseline": args.baseline,
                    **get_task_metadata(task_item),
                    "success": False,
                    "initial_skill_success": None if args.baseline == "no_skill" else False,
                    "post_repair_success": False if args.baseline != "no_skill" else None,
                    "steps_taken": 0,
                    "model_calls": 0,
                    "live_model_calls": 0,
                    "repair_count": 0,
                    "patch_count": 0,
                    "final_fail_reason": fail_reason,
                    "final_action_error": "",
                    "initial_skill_path": None,
                    "final_skill_path": None,
                    "repair_skill_paths": [],
                    "patch_types": [],
                    **zero_usage(),
                    "live_prompt_tokens": 0.0,
                    "live_completion_tokens": 0.0,
                    "live_total_tokens": 0.0,
                    "live_estimated_cost_usd": 0.0,
                    **failure_info,
                }
                trace = {
                    "run_id": run_id,
                    "baseline": args.baseline,
                    "task_id": task_item["task_id"],
                    "env_name": task_item["env_name"],
                    "notes": task_item.get("notes", ""),
                    "model": glm.model,
                    "success": False,
                    "fail_reason": fail_reason,
                    "failure_info": failure_info,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }

            results.append(result)
            dump_json(TRACE_ROOT / args.baseline / run_id / f"{task_item['task_id']}.json", trace)

            status_text = "SUCCESS" if result["success"] else "FAIL"
            print(f"{status_text}: {result['final_fail_reason'] or 'score > 0'}")
    finally:
        env.close()

    summary = aggregate_summary(
        run_id=run_id,
        split_path=args.split_path,
        baseline=args.baseline,
        model=args.model,
        max_steps=args.max_steps,
        max_repairs=args.max_repairs,
        max_model_calls=args.max_model_calls,
        headless=args.headless,
        results=results,
    )

    summary_base = f"{args.baseline}_{args.split_path.stem}_{run_id}"
    dump_json(RESULT_ROOT / args.baseline / f"{summary_base}.json", summary)
    dump_json(RESULT_ROOT / args.baseline / f"{args.baseline}_{args.split_path.stem}_latest.json", summary)
    dump_jsonl(RESULT_ROOT / args.baseline / f"{summary_base}.jsonl", results)
    dump_csv(RESULT_ROOT / args.baseline / f"{summary_base}_site_table.csv", summary["site_summary_rows"])
    dump_csv(
        RESULT_ROOT / args.baseline / f"{args.baseline}_{args.split_path.stem}_latest_site_table.csv",
        summary["site_summary_rows"],
    )

    print(f"\nBaseline: {summary['baseline']}")
    print(f"Split: {summary['split_name']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Success rate: {summary['success_rate']:.2f}")
    print(f"Infra failed: {summary['infra_failed_count']}")
    print(f"Average steps: {summary['average_steps']:.2f}")
    print(f"Average model calls: {summary['average_model_calls']:.2f}")
    print(f"Average total tokens: {summary['average_total_tokens']:.2f}")
    print(f"Estimated total cost (USD): {summary['total_estimated_cost_usd']:.4f}")
    if summary["site_summary_rows"]:
        print("Per-site summary:")
        for row in summary["site_summary_rows"]:
            print(
                "  "
                f"{row['site']}: "
                f"{int(row['success_count'])}/{int(row['total_tasks'])} "
                f"({float(row['success_rate']):.2f}), "
                f"infra={int(row.get('infra_failed_count', 0))}, "
                f"steps={float(row['average_steps']):.2f}, "
                f"repairs={float(row['average_repair_count']):.2f}, "
                f"avg_tokens={float(row['average_total_tokens']):.2f}, "
                f"avg_cost=${float(row['average_estimated_cost_usd']):.4f}"
            )


if __name__ == "__main__":
    main()

