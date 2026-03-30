from __future__ import annotations

import ast
import difflib
import json
import re
from typing import Any


ALLOWED_ACTIONS = ("CLICK", "TYPE", "SCROLL", "STOP")
ALLOWED_CONTRACT_KEYS = ("url_contains", "text_contains", "clickable_contains", "input_contains")
CONTRACT_KEY_ALIASES = {
    "clickable_texts": "clickable_contains",
    "input_fields": "input_contains",
    "input_labels": "input_contains",
}
ALLOWED_PATCH_TYPES = (
    "selector_replacement",
    "argument_correction",
    "precondition_insertion",
    "postcondition_insertion",
    "recovery_insertion",
)
_QUOTED_TEXT_PATTERN = re.compile(r"""['"]([^'"]+)['"]""")
_CONTAINS_PATTERN = re.compile(r"""[a-z]+:contains\((['"])(.+?)\1\)""", re.IGNORECASE)
_TEXT_SELECTOR_PATTERNS = (
    re.compile(r"""(?:[a-z]+:)?text\((['"])(.+?)\1\)""", re.IGNORECASE),
    re.compile(r"""(?:[a-z]+:)?text\s*=\s*(['"])(.+?)\1""", re.IGNORECASE),
)
_DOUBLE_QUOTED_TEXT_PATTERN = re.compile(r'"([^"]+)"')
_COMMON_GOAL_STOPWORDS = {
    "what",
    "which",
    "how",
    "much",
    "many",
    "the",
    "this",
    "that",
    "from",
    "with",
    "into",
    "onto",
    "your",
    "page",
    "pages",
    "item",
    "items",
    "listing",
    "listings",
    "first",
    "second",
    "third",
    "smaller",
    "larger",
    "piece",
    "product",
    "find",
}
_REDDIT_TITLE_UI_TEXTS = {
    "bans",
    "comments",
    "forums",
    "home",
    "moderation log",
    "notifications (0)",
    "submit",
    "submissions",
    "subscribe via rss",
    "upvote",
    "downvote",
    "wiki",
}
_GENERIC_NAVIGATION_CLICK_TARGETS = {
    "all categories",
    "apply",
    "back",
    "boats",
    "classifieds",
    "comments",
    "forums",
    "furniture",
    "home",
    "more",
    "my account",
    "notifications (0)",
    "pics",
    "publish ad",
    "reddit",
    "search",
    "select a category",
    "space",
    "submissions",
    "submit",
    "subscribe now!",
    "subscribe no subscribers",
    "subscribe via rss",
    "wiki",
}
_REPAIR_BLOCKED_GENERIC_TARGETS = _GENERIC_NAVIGATION_CLICK_TARGETS | {
    "home",
    "search",
    "submissions",
    "forums",
    "my account",
    "store logo",
}


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


def _candidate_json_prefixes(text: str) -> list[str]:
    candidates = [text]
    for index, char in enumerate(text):
        if char in "{[":
            candidates.append(text[index:])
    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_payload(text: str) -> Any:
    cleaned = strip_code_fences(text or "")
    if not cleaned:
        raise ValueError("Model output is empty.")

    decoder = json.JSONDecoder()
    candidates = _candidate_json_prefixes(cleaned)
    partial_payload: Any | None = None
    for candidate in candidates:
        try:
            payload, end_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        remainder = candidate[end_index:].strip()
        if not remainder:
            return payload
        if partial_payload is None and isinstance(payload, (dict, list)):
            partial_payload = payload

    if partial_payload is not None:
        return partial_payload

    for candidate in candidates:
        try:
            payload = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, (dict, list)):
            return payload

    raise ValueError("Could not extract a valid JSON payload from the model output.")


def normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings.")

    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, (dict, list)):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True).strip()
        else:
            text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def normalize_contract(contract: Any, field_name: str) -> dict[str, list[str]]:
    if contract is None:
        return {}
    if isinstance(contract, list):
        contract = {"text_contains": contract}
    if not isinstance(contract, dict):
        raise ValueError(f"{field_name} must be an object.")

    normalized: dict[str, list[str]] = {}
    for key, value in contract.items():
        normalized_key = CONTRACT_KEY_ALIASES.get(key, key)
        if normalized_key not in ALLOWED_CONTRACT_KEYS:
            continue
        normalized.setdefault(normalized_key, [])
        normalized[normalized_key].extend(
            normalize_string_list(value, f"{field_name}.{key}")
        )

    for key, values in normalized.items():
        deduped: list[str] = []
        seen = set()
        for item in values:
            lowered = item.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(item)
        normalized[key] = deduped
    return normalized


def _infer_patch_type(item: dict[str, Any]) -> str:
    candidate = str(item.get("type") or item.get("patch_type") or "").strip()
    if candidate in ALLOWED_PATCH_TYPES:
        return candidate

    location = str(item.get("target") or item.get("location") or "").strip().lower()
    reason = " ".join(
        str(item.get(key) or "").strip().lower() for key in ("summary", "description", "reason")
    )

    if item.get("old_target") or item.get("original_target") or item.get("new_target"):
        return "selector_replacement"
    if item.get("old_value") is not None or item.get("new_value") is not None:
        return "argument_correction"
    if "precondition" in location:
        return "precondition_insertion"
    if "postcondition" in location or "success_contract" in location:
        return "postcondition_insertion"
    if "recover" in reason:
        return "recovery_insertion"
    return ""


def _infer_patch_summary(item: dict[str, Any], patch_type: str) -> str:
    for key in ("summary", "description", "reason"):
        value = str(item.get(key) or "").strip()
        if value:
            return value

    old_target = str(item.get("old_target") or item.get("original_target") or "").strip()
    new_target = str(item.get("new_target") or "").strip()
    if old_target or new_target:
        old_text = old_target or "previous target"
        new_text = new_target or "new target"
        return f"{patch_type}: {old_text} -> {new_text}"

    if item.get("old_value") is not None or item.get("new_value") is not None:
        old_value = json.dumps(item.get("old_value"), ensure_ascii=False)
        new_value = json.dumps(item.get("new_value"), ensure_ascii=False)
        return f"{patch_type}: {old_value} -> {new_value}"

    location = str(item.get("target") or item.get("location") or "").strip()
    if location:
        return f"{patch_type}: update {location}"

    step_index = item.get("step_index")
    if isinstance(step_index, int):
        return f"{patch_type}: step {step_index}"

    affected_steps = item.get("affected_steps")
    if isinstance(affected_steps, list) and affected_steps:
        return f"{patch_type}: steps {', '.join(str(step) for step in affected_steps)}"

    return patch_type.replace("_", " ")


def normalize_patches(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("patches must be a list.")

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"patches[{index}] must be an object.")

        patch_type = _infer_patch_type(item)
        if not patch_type:
            continue
        summary = _infer_patch_summary(item, patch_type)
        normalized.append({"type": patch_type, "summary": summary})
    return normalized


def _coerce_steps_payload(payload: dict[str, Any]) -> Any:
    steps = payload.get("steps")

    if isinstance(steps, str):
        raw_steps = steps.strip()
        if raw_steps:
            try:
                parsed_steps = extract_json_payload(raw_steps)
            except Exception:
                parsed_steps = None
            if isinstance(parsed_steps, dict):
                steps = [parsed_steps]
            elif isinstance(parsed_steps, list):
                steps = parsed_steps

    if isinstance(steps, dict):
        return [steps]
    if isinstance(steps, list):
        return steps

    for alternate_key in ("actions", "plan", "step_sequence", "step"):
        alternate_steps = payload.get(alternate_key)
        if isinstance(alternate_steps, dict):
            return [alternate_steps]
        if isinstance(alternate_steps, list):
            return alternate_steps

    if str(payload.get("action", "") or "").strip():
        return [
            {
                "action": payload.get("action"),
                "target": payload.get(
                    "target",
                    payload.get("field", payload.get("label", payload.get("input", payload.get("name")))),
                ),
                "value": payload.get("value", payload.get("text", payload.get("reason"))),
            }
        ]

    return steps


def validate_skill_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Skill payload must be a JSON object.")

    skill_id = str(
        payload.get("skill_id", "")
        or payload.get("task_id", "")
        or payload.get("task", "")
        or "generated_skill"
    ).strip()
    task = str(payload.get("task", "") or payload.get("task_id", "") or skill_id).strip()
    if not task:
        raise ValueError("task must be non-empty.")

    steps = _coerce_steps_payload(payload)
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list.")

    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{index}] must be an object.")

        raw_action = str(step.get("action", "") or "").strip().upper()
        action = {
            "SELECT": "CLICK",
            "FOCUS": "CLICK",
            "HOVER": "CLICK",
            "WAIT": "",
            "PAUSE": "",
        }.get(raw_action, raw_action)
        if not action:
            continue

        target = step.get(
            "target",
            step.get("field", step.get("label", step.get("input", step.get("name")))),
        )
        value = step.get("value", step.get("text", step.get("reason")))

        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"steps[{index}].action must be one of: {', '.join(ALLOWED_ACTIONS)}.")

        if action == "CLICK":
            target = normalize_step_target_text(target)
            if not target:
                raise ValueError(f"steps[{index}] CLICK requires non-empty target.")
            value = None
        elif action == "TYPE":
            target = normalize_step_target_text(target)
            value = str(value or "").strip()
            if not target:
                raise ValueError(f"steps[{index}] TYPE requires non-empty target.")
            if not value:
                raise ValueError(f"steps[{index}] TYPE requires non-empty value.")
        elif action == "SCROLL":
            target = None
            value = "down"
        elif action == "STOP":
            target = None
            value = str(value or step.get("target") or step.get("answer") or "").strip() or None
        else:
            target = None
            value = None

        normalized_steps.append(
            {
                "action": action,
                "target": target,
                "value": value,
            }
        )

    if not normalized_steps:
        raise ValueError("steps must contain at least one executable action.")

    repair_history = normalize_string_list(payload.get("repair_history", []), "repair_history")

    return {
        "skill_id": skill_id,
        "task": task,
        "preconditions": normalize_contract(payload.get("preconditions"), "preconditions"),
        "success_contract": normalize_contract(payload.get("success_contract"), "success_contract"),
        "steps": normalized_steps,
        "repair_history": repair_history,
        "patches": normalize_patches(payload.get("patches", [])),
    }


def parse_skill_response(raw_output: str) -> dict[str, Any]:
    payload = extract_json_payload(raw_output)
    return validate_skill_payload(payload)


def normalize_step_target_text(target: Any) -> str:
    text = str(target or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    for prefix in ("field=", "label=", "input=", "target="):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            lowered = text.lower()
            break

    bid_match = re.search(r"\bbid\s+([A-Za-z0-9]+)\b", text, flags=re.IGNORECASE)
    if bid_match:
        return bid_match.group(1).strip()

    for pattern in _TEXT_SELECTOR_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return match.group(2).strip()

    contains_match = _CONTAINS_PATTERN.fullmatch(text)
    if contains_match:
        return contains_match.group(2).strip()

    quoted = _QUOTED_TEXT_PATTERN.findall(text)
    if quoted:
        return quoted[0].strip()

    lowered = text.lower()
    for prefix in (
        "input with label ",
        "field with label ",
        "input field with label ",
        "input field for ",
        "field for ",
    ):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip().strip("\"'")

    for suffix in (" button", " link", " tab", " checkbox", " radio", " option"):
        if lowered.endswith(suffix):
            return text[: -len(suffix)].strip()

    return text


def contract_to_lines(contract: dict[str, list[str]]) -> list[str]:
    lines: list[str] = []
    for key in ALLOWED_CONTRACT_KEYS:
        values = contract.get(key, [])
        if values:
            lines.append(f"- {key}: {', '.join(values)}")
    if not lines:
        lines.append("- empty")
    return lines


def skill_to_action_string(step: dict[str, Any]) -> str:
    action = step["action"]
    if action == "CLICK":
        return f"CLICK[{step['target']}]"
    if action == "TYPE":
        return f"TYPE[{step['target']}={step['value']}]"
    if action == "SCROLL":
        return "SCROLL[down]"
    stop_value = str(step.get("value") or "").strip()
    return f"STOP[{stop_value or 'Skill completed'}]"


def build_observation_summary(observation: dict[str, Any], text_limit: int = 5000) -> str:
    page_text = observation.get("page_text", "") or ""
    if len(page_text) > text_limit:
        page_text = page_text[:text_limit] + " ..."

    clickables = observation.get("clickable_elements") or []
    inputs = observation.get("input_fields") or []

    clickable_summary = ", ".join(item.get("text", "") for item in clickables[:40]) or "none"
    input_summary = (
        "\n".join(
            f"- label={item.get('label', '')}; bid={item.get('bid', '')}; type={item.get('type', '')}"
            for item in inputs[:30]
        )
        or "none"
    )
    goal_image_count = len(observation.get("goal_image_urls") or [])
    open_tab_titles = observation.get("open_pages_titles") or []
    active_page_index = _coerce_int(observation.get("active_page_index", 0))
    active_tab_title = ""
    if 0 <= active_page_index < len(open_tab_titles):
        active_tab_title = str(open_tab_titles[active_page_index] or "").strip()

    return (
        f"Goal:\n{observation.get('goal', '').strip()}\n\n"
        f"Current URL: {observation.get('url', '')}\n"
        f"Active tab index: {active_page_index}\n"
        f"Active tab title: {active_tab_title or 'unknown'}\n"
        f"Open tab titles: {', '.join(observation.get('open_pages_titles') or []) or 'none'}\n"
        f"Last action error: {observation.get('last_action_error', '') or 'none'}\n"
        f"Goal image count: {goal_image_count}\n"
        f"Clickable targets: {clickable_summary}\n"
        f"Input fields:\n{input_summary}\n\n"
        f"Page text:\n{page_text}"
    )


def is_current_page_answer_task(
    task_item: dict[str, Any],
    observation: dict[str, Any],
) -> bool:
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


def is_visual_classifieds_goal(goal_text: str) -> bool:
    goal_lc = str(goal_text or "").lower()
    return any(
        marker in goal_lc
        for marker in ("image", "picture", "cover", "photo", "on water", "on grass", "in a basket", "shirt")
    )


def _goal_keywords(goal_text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", goal_text.lower())
    keywords: list[str] = []
    for token in tokens:
        if len(token) < 3 or token in _COMMON_GOAL_STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:8]


def extract_focused_page_evidence(
    *,
    page_text: str,
    goal_text: str,
    max_lines: int = 28,
) -> str:
    if not page_text.strip():
        return "none"

    keywords = _goal_keywords(goal_text)
    if not keywords:
        return "none"

    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    selected_indexes: list[int] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(keyword in lowered for keyword in keywords):
            for candidate in range(max(0, index - 1), min(len(lines), index + 3)):
                if candidate not in selected_indexes:
                    selected_indexes.append(candidate)
        if len(selected_indexes) >= max_lines:
            break

    if not selected_indexes:
        return "none"

    selected_lines = [lines[index] for index in selected_indexes[:max_lines]]
    return "\n".join(f"- {line}" for line in selected_lines)


def build_skill_generation_prompt(task_item: dict[str, Any], observation: dict[str, Any], max_steps: int) -> str:
    clickables = [item.get("text", "") for item in observation.get("clickable_elements") or [] if item.get("text")]
    inputs = [
        item.get("label", "") or item.get("name", "")
        for item in observation.get("input_fields") or []
        if item.get("label") or item.get("name")
    ]
    open_tab_titles = [title for title in observation.get("open_pages_titles") or [] if title]
    active_page_index = _coerce_int(observation.get("active_page_index", 0))
    clickable_lines = "\n".join(f"- {text}" for text in clickables[:40]) or "- none"
    input_lines = "\n".join(f"- {text}" for text in inputs[:20]) or "- none"
    tab_lines = "\n".join(
        f"- [{index}] {title}" + (" (active)" if index == active_page_index else "")
        for index, title in enumerate(open_tab_titles[:10])
    ) or "- none"
    current_page_hint = ""
    if is_current_page_answer_task(task_item, observation) and not is_visual_current_page_comparison_task(
        task_item, observation
    ):
        current_page_hint = (
            "Current-page answer bias:\n"
            "1. This task appears answerable from the CURRENT page without navigation.\n"
            "2. Read the visible page text and screenshot first; do not default to clicking product titles.\n"
            "3. The default and preferred output is a ONE-STEP skill: a single STOP step with the exact answer in step.value.\n"
            "4. Only use CLICK or SCROLL if the answer is genuinely not visible on the current page.\n"
            "5. Do not navigate to detail pages just because an item title is clickable.\n"
            "6. If the page already contains enough information to compare items or read a spec, answer directly.\n\n"
        )
    elif is_current_page_answer_task(task_item, observation):
        current_page_hint = (
            "Current-page visual comparison bias:\n"
            "1. This task depends on comparing multiple visible images or posts on the CURRENT page.\n"
            "2. Do not default to a one-step STOP answer unless the winner is visually obvious now.\n"
            "3. Prefer staying on the same page and using SCROLL to inspect more visible candidates before answering.\n"
            "4. Avoid jumping to unrelated detail pages just because a title is clickable.\n\n"
        )
    focused_evidence = ""
    if is_current_page_answer_task(task_item, observation):
        focused_evidence = (
            "Focused page evidence for answer extraction:\n"
            f"{extract_focused_page_evidence(page_text=observation.get('page_text', ''), goal_text=observation.get('goal', ''))}\n\n"
        )
    classifieds_visual_hint = ""
    if (
        task_item.get("sites") == ["classifieds"]
        and str(task_item.get("task_family", "") or "") == "navigation"
        and is_visual_classifieds_goal(observation.get("goal", ""))
    ):
        classifieds_visual_hint = (
            "Visual classifieds search bias:\n"
            "1. This task is decided by listing images, not only titles.\n"
            "2. Do not assume the first visible listing is correct.\n"
            "3. If the decisive visual cue is not clearly visible in the current viewport, prefer SCROLL[down] as step 1 to inspect more listings on the same results page.\n"
            "4. After scrolling, click the single listing whose image best matches the goal.\n\n"
        )
    site_specific_hint = ""
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if "reddit" in sites:
        site_specific_hint = (
            "Reddit/Postmill navigation bias:\n"
            "1. If the goal names a subreddit like /f/boston, first click that exact subreddit link when it is visible.\n"
            "2. Use exact visible sort controls such as Sort by: Hot, Top, From: Past 24 hours, and All time.\n"
            "3. If the goal asks for a post or the comments section, prefer clicking the exact full post title or a visible comment-count link.\n"
            "4. Do not click image thumbnails, shortened title fragments, or invented navigation like Back unless that exact text is currently visible.\n\n"
        )
    return (
        "You are generating a task-level contracted skill for BrowserGym VisualWebArena.\n"
        "Return JSON only. Do not wrap the JSON in markdown.\n\n"
        "Required schema:\n"
        "{\n"
        '  "skill_id": "string",\n'
        '  "task": "string",\n'
        '  "preconditions": {"url_contains": [], "text_contains": [], "clickable_contains": [], "input_contains": []},\n'
        '  "success_contract": {"url_contains": [], "text_contains": [], "clickable_contains": [], "input_contains": []},\n'
        '  "steps": [{"action": "CLICK|TYPE|SCROLL|STOP", "target": "string or null", "value": "string or null"}],\n'
        '  "repair_history": [],\n'
        '  "patches": []\n'
        "}\n\n"
        f"Task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Task family: {task_item.get('task_family', 'unknown')}\n"
        f"Notes: {task_item.get('notes', '')}\n"
        f"Max allowed steps: {max_steps}\n\n"
        "Rules:\n"
        "1. Use only CLICK, TYPE, SCROLL, STOP.\n"
        "2. Use exact visible English target text when possible.\n"
        "3. Keep the skill concise and executable.\n"
        "4. If the task likely needs a final answer, include a STOP step at the end, and put the exact answer text in step.value.\n"
        "5. If you are unsure about a contract field, leave it empty instead of guessing.\n"
        "6. Step targets must be plain visible text or plain input labels only.\n"
        "7. Do not emit CSS/XPath/Playwright selectors such as text=..., a:text(...), :contains(...), #id, or [attr=value].\n"
        "8. Do not invent unsupported schema keys such as input_fields_count or clickable_count.\n\n"
        f"{current_page_hint}"
        f"{focused_evidence}"
        f"{classifieds_visual_hint}"
        f"{site_specific_hint}"
        "Current page executable targets:\n"
        f"Open tab titles (you may switch tabs by emitting CLICK[exact tab title]):\n{tab_lines}\n"
        f"Clickable texts:\n{clickable_lines}\n"
        f"Input labels:\n{input_lines}\n\n"
        "Execution discipline:\n"
        "1. The first 1-3 steps must be executable on the CURRENT page only.\n"
        "2. Do not reference future-page elements that are not visible yet.\n"
        "3. Every CLICK target in the first 1-3 steps must exactly match either one current clickable text above or one open tab title above.\n"
        "4. Every TYPE target in the first 1-3 steps must exactly match one current input label above.\n"
        "5. Prefer a short executable prefix over a long speculative plan.\n"
        "6. If another relevant page is already open in a browser tab, prefer switching to that tab with CLICK[exact tab title] instead of hallucinating a navigation path.\n"
        "7. For multi-page tasks, only plan later-page targets after using realistic navigation from the current page or a real open-tab switch.\n\n"
        "8. For current-page answer tasks, if the answer is visible now, return exactly one step: STOP with the exact answer in value.\n\n"
        "Examples:\n"
        '- Good TYPE step: {"action": "TYPE", "target": "Search query", "value": "blue kayak"}\n'
        '- Good tab switch step: {"action": "CLICK", "target": "One Stop Market", "value": null}\n'
        '- Good answer step: {"action": "STOP", "target": null, "value": "16 inches"}\n'
        '- Good current-page answer skill: {"steps": [{"action": "STOP", "target": null, "value": "16 inches"}], "...": "..."}\n'
        '- Bad current-page answer skill: {"steps": [{"action": "CLICK", "target": "Product title", "value": null}], "...": "..."}\n'
        '- Bad TYPE step: {"action": "TYPE", "target": "", "value": "blue kayak"}\n'
        '- Bad CLICK step: {"action": "CLICK", "target": "a:text(\\"Home\\")", "value": null}\n\n'
        f"Observation summary:\n{build_observation_summary(observation)}\n"
    )


def build_text_rewrite_prompt(
    task_item: dict[str, Any],
    latest_observation: dict[str, Any],
    current_skill: dict[str, Any],
    failure_info: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    max_steps: int,
) -> str:
    _ = current_skill, failure_info, execution_trace
    return (
        "The previous skill failed on BrowserGym VisualWebArena.\n"
        "Rewrite the entire skill from scratch using only the latest observation as grounding.\n"
        "Do not rely on the previous skill, failure metadata, or execution trace.\n"
        "Return JSON only and follow the same schema as before.\n\n"
        f"Task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Max allowed steps: {max_steps}\n"
        "Rewrite rules:\n"
        "1. Start from the task goal and the latest observation only.\n"
        "2. Do not assume any previous step was correct.\n"
        "3. If the answer is already visible on the current page, return a single STOP step.\n"
        "4. Otherwise produce a short executable plan grounded in currently visible targets.\n"
        "5. Keep targets as plain visible text or plain input labels only.\n\n"
        f"Latest observation:\n{build_observation_summary(latest_observation)}\n"
    )


def build_contract_repair_prompt(
    task_item: dict[str, Any],
    latest_observation: dict[str, Any],
    current_skill: dict[str, Any],
    failure_info: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    max_steps: int,
    repair_round: int,
    *,
    include_failure_localization: bool = True,
    constrained_patch_repair: bool = True,
    include_structured_repair_context: bool = True,
) -> str:
    trace_excerpt = json.dumps(execution_trace[-6:], ensure_ascii=False, indent=2)
    error_code = str(failure_info.get("error_code", "") or "").strip() or "unknown_failure"
    allowed_patch_types = allowed_patch_types_for_failure(failure_info)
    allowed_patch_text = ", ".join(sorted(allowed_patch_types)) if allowed_patch_types else "selector_replacement, argument_correction"
    failed_step_index = int(failure_info.get("failed_step_index") or 0)
    clickables = [item.get("text", "") for item in latest_observation.get("clickable_elements") or [] if item.get("text")]
    inputs = [
        item.get("label", "") or item.get("name", "")
        for item in latest_observation.get("input_fields") or []
        if item.get("label") or item.get("name")
    ]
    open_tab_titles = [title for title in latest_observation.get("open_pages_titles") or [] if title]
    active_page_index = _coerce_int(latest_observation.get("active_page_index", 0))
    clickable_lines = "\n".join(f"- {text}" for text in clickables[:40]) or "- none"
    input_lines = "\n".join(f"- {text}" for text in inputs[:20]) or "- none"
    tab_lines = "\n".join(
        f"- [{index}] {title}" + (" (active)" if index == active_page_index else "")
        for index, title in enumerate(open_tab_titles[:10])
    ) or "- none"
    current_steps = current_skill.get("steps", []) or []
    stop_only_answer_attempt = (
        is_current_page_answer_task(task_item, latest_observation)
        and not is_visual_current_page_comparison_task(task_item, latest_observation)
        and len(current_steps) == 1
        and current_steps[0].get("action") == "STOP"
        and str(current_steps[0].get("value") or "").strip()
    )
    current_page_hint = ""
    if include_structured_repair_context and is_current_page_answer_task(
        task_item, latest_observation
    ) and not is_visual_current_page_comparison_task(task_item, latest_observation):
        current_page_hint = (
            "Current-page answer repair bias:\n"
            "1. This task appears answerable from the CURRENT page.\n"
            "2. If the answer is already visible in the latest observation, repair toward a single STOP step whose value is the exact answer.\n"
            "3. Do not add clicks, tab switches, or scrolls unless the trace clearly shows the answer was not visible yet.\n"
            "4. If the previous skill navigated away before answering, prefer removing that unnecessary navigation.\n"
            "5. For current-page answer tasks, a repair that still clicks a listing without evidence that the answer was hidden is likely wrong.\n\n"
        )
    elif include_structured_repair_context and is_current_page_answer_task(task_item, latest_observation):
        current_page_hint = (
            "Current-page visual comparison repair bias:\n"
            "1. This task depends on comparing multiple visible posts or images on the same page.\n"
            "2. Do not force the repair back to a one-step STOP unless the latest observation makes the winner visually obvious.\n"
            "3. Prefer repairs that keep the agent on the same page and add SCROLL or a better-supported comparison step.\n\n"
        )
    focused_evidence = ""
    if include_structured_repair_context and is_current_page_answer_task(task_item, latest_observation):
        focused_evidence = (
            "Focused page evidence for answer extraction:\n"
            f"{extract_focused_page_evidence(page_text=latest_observation.get('page_text', ''), goal_text=latest_observation.get('goal', ''))}\n\n"
        )
    answer_revision_hint = ""
    if include_structured_repair_context and stop_only_answer_attempt:
        answer_revision_hint = (
            "Answer-revision rule:\n"
            "1. The previous skill already attempted a direct STOP answer from the current page.\n"
            "2. If reward stayed zero, first assume the answer text itself was wrong or imprecise.\n"
            "3. Prefer repairing by changing STOP.value while keeping the skill as a single STOP step.\n"
            "4. Do not add navigation or clicks unless the latest observation clearly proves the answer is not visible on the current page.\n\n"
        )
    site_specific_repair_hint = ""
    sites = [str(site).lower() for site in task_item.get("sites", []) or []]
    if "reddit" in sites:
        site_specific_repair_hint = (
            "Reddit/Postmill repair bias:\n"
            "1. If the goal names a subreddit like /f/boston, preserve or add CLICK on that exact subreddit link as the first navigation step.\n"
            "2. Prefer exact visible sort controls such as Sort by: Hot, Top, From: Past 24 hours, and All time over invented abstractions.\n"
            "3. If the goal asks for a post or comments section, repair toward an exact full post title or a visible comment-count link.\n"
            "4. Do not repair by clicking image thumbnails, shortened title snippets, or Back unless those exact texts are visible on the current page.\n\n"
        )
    localization_rule = (
        "7. If failed_step_index is 0 or 1, you must change step 1 or step 2.\n"
        if include_failure_localization
        else ""
    )
    trace_excerpt_text = (
        f"Execution trace excerpt:\n{trace_excerpt}\n\n"
        if include_failure_localization
        else "Execution trace excerpt omitted. Infer a repair only from the current skill and the latest observation.\n\n"
    )
    if constrained_patch_repair:
        intro_text = (
            "The previous contracted skill failed on BrowserGym VisualWebArena.\n"
            "Repair the skill with minimal changes. Keep the working parts. Return a full repaired JSON skill.\n"
            "Use the optional patches field to record one or more patch operators.\n\n"
            "Allowed patch types:\n"
            "- selector_replacement\n"
            "- argument_correction\n"
            "- precondition_insertion\n"
            "- postcondition_insertion\n"
            "- recovery_insertion\n\n"
            "Return JSON only and follow the same schema as before.\n\n"
        )
        repair_rules = (
            "Repair rules:\n"
            "1. Preserve task-critical literals that appear in the goal, especially quoted strings.\n"
            "2. Do not change the target entity to a different semantic target just because it is visible.\n"
            "3. Step targets must remain plain visible text or plain input labels only.\n"
            "4. Do not emit CSS/XPath/Playwright selectors such as text=..., a:text(...), :contains(...), #id, or [attr=value].\n"
            "5. Do not add unsupported schema keys such as input_fields_count or clickable_count.\n"
            "6. If the failure is execution-related, do not only edit success_contract or patches; change at least one executable step or a blocking precondition.\n"
            "8. At least one executable step must change after repair; patch-only edits are invalid.\n"
            "9. Any CLICK/TYPE target you introduce for the currently reachable repair prefix must exist in the current page target lists below.\n"
            "10. You may switch to an already-open tab by emitting CLICK[exact tab title] if that tab title appears below.\n"
            "11. If a relevant tab is already open, prefer switching to it over inventing a new navigation path.\n"
            "12. If the trace shows that an earlier navigation or tab-switch step reached the current page successfully, preserve that working prefix and append new later-page steps after it.\n"
            "13. Do not replace a working tab-switch/navigation step with a later-page element that only becomes visible after that step.\n"
            "14. Later suffix steps may reference targets that appear only after the repaired prefix executes, but the changed immediate prefix must stay grounded in the latest observation.\n\n"
        )
        repair_rules = repair_rules.replace(
            "8. At least one executable step must change after repair; patch-only edits are invalid.\n",
            localization_rule + "8. At least one executable step must change after repair; patch-only edits are invalid.\n",
        )
    else:
        intro_text = (
            "The previous contracted skill failed on BrowserGym VisualWebArena.\n"
            "Repair the skill, but you may freely rewrite any part of it. You do not need to keep the repair minimal.\n"
            "Return a full repaired JSON skill. The optional patches field may be omitted or used only as loose notes.\n\n"
            "Return JSON only and follow the same schema as before.\n\n"
        )
        repair_rules = (
            "Repair rules:\n"
            "1. Preserve task-critical literals that appear in the goal, especially quoted strings.\n"
            "2. Do not change the target entity to a different semantic target just because it is visible.\n"
            "3. Step targets must remain plain visible text or plain input labels only.\n"
            "4. Do not emit CSS/XPath/Playwright selectors such as text=..., a:text(...), :contains(...), #id, or [attr=value].\n"
            "5. Do not add unsupported schema keys such as input_fields_count or clickable_count.\n"
            "6. You may rewrite the whole step sequence if needed.\n"
            "7. Use the latest observation summary as your only grounding context for what is currently visible and actionable.\n"
            "8. Later suffix steps may reference targets that appear only after the repaired prefix executes, but the changed immediate prefix must stay grounded in the latest observation.\n\n"
        )
    if include_failure_localization:
        failure_header = (
            f"Failure error code: {error_code}\n"
            f"Failed step index: {failed_step_index}\n"
            f"Preferred patch types for this failure: {allowed_patch_text}\n"
            f"Failure info:\n{json.dumps(failure_info, ensure_ascii=False, indent=2)}\n\n"
        )
    else:
        failure_header = (
            "Failure summary:\n"
            "The previous skill failed to complete the task. No explicit failed-step index, failure source, or patch preference is provided.\n\n"
        )
    return (
        intro_text
        + f"Repair round: {repair_round}\n"
        + f"Task id: {task_item['task_id']}\n"
        + f"Environment: {task_item['env_name']}\n"
        + f"Max allowed steps: {max_steps}\n"
        + failure_header
        + repair_rules
        + f"{current_page_hint}"
        + f"{focused_evidence}"
        + f"{answer_revision_hint}"
        + f"{site_specific_repair_hint}"
        + (
            "Current page executable targets:\n"
            + f"Open tab titles (valid immediate CLICK targets):\n{tab_lines}\n"
            + f"Clickable texts:\n{clickable_lines}\n"
            + f"Input labels:\n{input_lines}\n\n"
            if include_structured_repair_context
            else ""
        )
        + "Step formatting reminders:\n"
        + '- Every TYPE step must include both a non-empty "target" and a non-empty "value".\n'
        + '- Every final-answer STOP step should store the exact answer text in "value".\n'
        + '- Use plain visible text labels only, for example "Search query" or "One Stop Market", not selector syntax.\n'
        + '- Good multi-page repair: keep CLICK["One Stop Market"] as step 1 and add CLICK["Electronics"] as a later step.\n'
        + '- Bad multi-page repair: replace CLICK["One Stop Market"] with CLICK["Electronics"] in step 1 when Electronics only exists after switching tabs.\n\n'
        + f"Current skill:\n{json.dumps(current_skill, ensure_ascii=False, indent=2)}\n\n"
        + trace_excerpt_text
        + f"Latest observation:\n{build_observation_summary(latest_observation)}\n"
    )


def _insert_step_before_stop(
    steps: list[dict[str, Any]],
    *,
    action: str,
    target: str | None,
    value: str | None = None,
) -> bool:
    candidate = {
        "action": action,
        "target": target,
        "value": value,
    }
    for step in steps:
        if (
            step.get("action") == candidate["action"]
            and step.get("target") == candidate["target"]
            and step.get("value") == candidate["value"]
        ):
            return False

    insert_at = len(steps)
    for index, step in enumerate(steps):
        if step.get("action") == "STOP":
            insert_at = index
            break
    steps.insert(insert_at, candidate)
    return True


def _quoted_goal_targets(goal_text: str) -> list[str]:
    if not goal_text:
        return []
    return [text.strip() for text in _DOUBLE_QUOTED_TEXT_PATTERN.findall(goal_text) if text.strip()]


def stabilize_contractskill_miniwob_m3_skill(
    task_item: dict[str, Any],
    skill: dict[str, Any],
    *,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env_name = str(task_item.get("env_name", "") or "")
    task_family = str(task_item.get("task_family", "") or "")
    if "miniwob" not in env_name or task_family != "M3_selection_and_composition":
        return skill

    goal_text = str((observation or {}).get("goal") or skill.get("task") or task_item.get("notes") or "").strip()
    steps = [dict(step) for step in skill.get("steps", [])]
    repair_history = list(skill.get("repair_history", []))
    changed = False

    if env_name.endswith("click-checkboxes") or env_name.endswith("click-checkboxes-soft"):
        if _insert_step_before_stop(steps, action="CLICK", target="Submit", value=None):
            repair_history.append("m3_completion: appended CLICK[Submit] for checkbox task")
            changed = True
        if skill.get("success_contract"):
            skill = {**skill, "success_contract": {}}
            repair_history.append("m3_completion: cleared weak success_contract for checkbox task")
            changed = True

    elif env_name.endswith("click-menu") or env_name.endswith("click-menu-2"):
        quoted_targets = [text for text in _quoted_goal_targets(goal_text) if text.lower() != "menu"]
        target_item = quoted_targets[-1] if quoted_targets else ""
        if _insert_step_before_stop(steps, action="CLICK", target="Menu", value=None):
            repair_history.append("m3_completion: ensured CLICK[Menu] prefix for menu task")
            changed = True
        if target_item and _insert_step_before_stop(steps, action="CLICK", target=target_item, value=None):
            repair_history.append(f"m3_completion: appended CLICK[{target_item}] for menu task")
            changed = True
        if skill.get("success_contract"):
            skill = {**skill, "success_contract": {}}
            repair_history.append("m3_completion: cleared weak success_contract for menu task")
            changed = True

    if not changed:
        return skill

    deduped_history: list[str] = []
    seen = set()
    for item in repair_history:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped_history.append(text)

    return {
        **skill,
        "steps": steps,
        "repair_history": deduped_history,
    }


def _clickable_bid_for_text(
    observation: dict[str, Any] | None,
    target_text: str,
    *,
    preferred_roles: set[str] | None = None,
) -> str:
    if not observation:
        return ""
    normalized_target = normalize_step_target_text(target_text).lower()
    if not normalized_target:
        return ""
    matches: list[dict[str, Any]] = []
    for item in observation.get("clickable_elements") or []:
        text = normalize_step_target_text(item.get("text", ""))
        if text.lower() != normalized_target:
            continue
        matches.append(item)
    if not matches:
        return ""
    if preferred_roles:
        for item in matches:
            role = str(item.get("role") or "").strip().lower()
            if role in preferred_roles:
                bid = str(item.get("bid") or item.get("browsergym_id") or "").strip()
                if bid:
                    return bid
    for item in matches:
        bid = str(item.get("bid") or item.get("browsergym_id") or "").strip()
        if bid:
            return bid
    return ""


def _extract_goal_path_segments(goal_text: str) -> list[str]:
    normalized_goal = str(goal_text or "").strip()
    if not normalized_goal:
        return []
    if normalized_goal.lower().startswith("select "):
        normalized_goal = normalized_goal[7:].strip()
    parts = [
        normalize_step_target_text(part)
        for part in re.split(r"\s*>\s*", normalized_goal)
    ]
    return [part for part in parts if part]


def _first_click_target_excluding(
    steps: list[dict[str, Any]],
    *,
    excluded_targets: set[str],
) -> str:
    excluded = {normalize_step_target_text(value).lower() for value in excluded_targets}
    for step in steps:
        if str(step.get("action") or "").upper() != "CLICK":
            continue
        target = normalize_step_target_text(step.get("target"))
        if not target or target.lower() in excluded:
            continue
        return target
    return ""


def stabilize_contractskill_miniwob_skill(
    task_item: dict[str, Any],
    skill: dict[str, Any],
    *,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skill = stabilize_contractskill_miniwob_m3_skill(
        task_item,
        skill,
        observation=observation,
    )

    env_name = str(task_item.get("env_name", "") or "")
    if "miniwob" not in env_name:
        return skill

    goal_text = str((observation or {}).get("goal") or skill.get("task") or task_item.get("notes") or "").strip()
    steps = [dict(step) for step in skill.get("steps", [])]
    repair_history = list(skill.get("repair_history", []))
    changed = False

    if env_name.endswith("click-menu"):
        goal_path = _extract_goal_path_segments(goal_text)
        if len(goal_path) >= 2:
            new_steps = [
                {"action": "CLICK", "target": goal_path[0], "value": None},
                {"action": "CLICK", "target": goal_path[1], "value": None},
            ]
            if steps[:2] != new_steps or len(steps) != 2:
                steps = new_steps
                repair_history.append(
                    f"menu_path_stabilization: enforced CLICK[{goal_path[0]}] -> CLICK[{goal_path[1]}]"
                )
                changed = True
            if skill.get("success_contract"):
                skill = {**skill, "success_contract": {}}
                repair_history.append("menu_path_stabilization: cleared weak success_contract")
                changed = True

    elif env_name.endswith("click-collapsible"):
        expand_target = _first_click_target_excluding(steps, excluded_targets={"Submit"})
        if not expand_target and observation:
            expand_target = next(
                (
                    normalize_step_target_text(item.get("text", ""))
                    for item in observation.get("clickable_elements") or []
                    if str(item.get("role") or "").strip().lower() == "tab"
                    and normalize_step_target_text(item.get("text", "")).lower() != "submit"
                ),
                "",
            )
        submit_button_bid = _clickable_bid_for_text(
            observation,
            "Submit",
            preferred_roles={"button"},
        )
        if expand_target and submit_button_bid:
            new_steps = [
                {"action": "CLICK", "target": expand_target, "value": None},
                {"action": "CLICK", "target": f"bid={submit_button_bid}", "value": None},
            ]
            if steps[:2] != new_steps or len(steps) != 2:
                steps = new_steps
                repair_history.append(
                    f"collapsible_bid_stabilization: enforced CLICK[{expand_target}] -> CLICK[bid={submit_button_bid}]"
                )
                changed = True

    elif env_name.endswith("enter-password"):
        password_values = _quoted_goal_targets(goal_text)
        password_value = password_values[0] if password_values else ""
        input_bids = [
            str(item.get("bid") or item.get("browsergym_id") or "").strip()
            for item in (observation or {}).get("input_fields") or []
            if str(item.get("bid") or item.get("browsergym_id") or "").strip()
        ]
        deduped_input_bids = list(dict.fromkeys(input_bids))
        submit_bid = _clickable_bid_for_text(
            observation,
            "Submit",
            preferred_roles={"button"},
        )
        if password_value and len(deduped_input_bids) >= 2 and submit_bid:
            new_steps = [
                {"action": "TYPE", "target": f"bid={deduped_input_bids[0]}", "value": password_value},
                {"action": "TYPE", "target": f"bid={deduped_input_bids[1]}", "value": password_value},
                {"action": "CLICK", "target": f"bid={submit_bid}", "value": None},
            ]
            if steps[:3] != new_steps:
                steps = new_steps
                repair_history.append(
                    "enter_password_bid_stabilization: switched textbox labels to stable bid targets"
                )
                changed = True

    if not changed:
        return skill

    deduped_history: list[str] = []
    seen = set()
    for item in repair_history:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped_history.append(text)

    return {
        **skill,
        "steps": steps,
        "repair_history": deduped_history,
    }


def observation_contract_status(
    observation: dict[str, Any],
    contract: dict[str, list[str]],
) -> dict[str, Any]:
    if not contract:
        return {"success": True, "unmet": []}

    page_text = observation.get("page_text", "") or ""
    url = observation.get("url", "") or ""
    clickable_texts = [item.get("text", "") or "" for item in observation.get("clickable_elements") or []]
    input_labels = [
        item.get("label", "") or item.get("name", "") or ""
        for item in observation.get("input_fields") or []
    ]

    unmet: list[str] = []
    page_text_lc = page_text.lower()
    url_lc = url.lower()
    clickable_texts_lc = [text.lower() for text in clickable_texts if text]
    input_labels_lc = [text.lower() for text in input_labels if text]

    for expected in contract.get("url_contains", []):
        expected_lc = expected.lower()
        if expected_lc not in url_lc:
            unmet.append(f"url_missing:{expected}")

    for expected in contract.get("text_contains", []):
        expected_lc = expected.lower()
        if expected_lc not in page_text_lc:
            unmet.append(f"text_missing:{expected}")

    for expected in contract.get("clickable_contains", []):
        expected_lc = expected.lower()
        if not any(expected_lc in text for text in clickable_texts_lc):
            unmet.append(f"clickable_missing:{expected}")

    for expected in contract.get("input_contains", []):
        expected_lc = expected.lower()
        if not any(expected_lc in text for text in input_labels_lc) and expected_lc not in page_text_lc:
            unmet.append(f"input_missing:{expected}")

    return {"success": not unmet, "unmet": unmet}


def observation_executable_targets(observation: dict[str, Any]) -> dict[str, list[str]]:
    clickable_texts: list[str] = []
    clickable_bids: list[str] = []
    for item in observation.get("clickable_elements") or []:
        text = normalize_step_target_text(item.get("text", ""))
        bid = str(item.get("bid") or item.get("browsergym_id") or "").strip()
        if text:
            clickable_texts.append(text)
        if bid:
            clickable_bids.append(bid)

    input_labels: list[str] = []
    input_bids: list[str] = []
    for item in observation.get("input_fields") or []:
        label = normalize_step_target_text(item.get("label", "") or item.get("name", ""))
        bid = str(item.get("bid") or item.get("browsergym_id") or "").strip()
        if label:
            input_labels.append(label)
        if bid:
            input_bids.append(bid)

    tab_titles = [
        normalize_step_target_text(title)
        for title in observation.get("open_pages_titles") or []
        if normalize_step_target_text(title)
    ]
    return {
        "clickable_texts": list(dict.fromkeys(clickable_texts)),
        "clickable_bids": list(dict.fromkeys(clickable_bids)),
        "input_labels": list(dict.fromkeys(input_labels)),
        "input_bids": list(dict.fromkeys(input_bids)),
        "tab_titles": list(dict.fromkeys(tab_titles)),
    }


def _normalized_target_set(values: list[str]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        text = normalize_step_target_text(value)
        if text:
            normalized.add(text.lower())
    return normalized


def is_observation_target_executable(
    observation: dict[str, Any],
    *,
    action: str,
    target: str | None,
) -> bool:
    normalized_target = normalize_step_target_text(target)
    if not normalized_target:
        return False

    targets = observation_executable_targets(observation)
    if action == "CLICK":
        allowed = (
            _normalized_target_set(targets["clickable_texts"])
            | _normalized_target_set(targets["tab_titles"])
            | {value.lower() for value in targets["clickable_bids"]}
        )
        return normalized_target.lower() in allowed
    if action == "TYPE":
        allowed = _normalized_target_set(targets["input_labels"]) | {
            value.lower() for value in targets["input_bids"]
        }
        return normalized_target.lower() in allowed
    return True


def introduced_invalid_repair_targets(
    previous: dict[str, Any],
    updated: dict[str, Any],
    observation: dict[str, Any],
    *,
    failure_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    previous_steps = previous.get("steps", []) or []
    updated_steps = updated.get("steps", []) or []
    failed_step_index = int((failure_info or {}).get("failed_step_index") or 0)
    validation_limit = failed_step_index if failed_step_index > 0 else min(3, len(updated_steps))

    for index, updated_step in enumerate(updated_steps, start=1):
        if index > validation_limit:
            break
        action = str(updated_step.get("action") or "").upper()
        if action not in {"CLICK", "TYPE"}:
            continue

        target = normalize_step_target_text(updated_step.get("target"))
        if not target:
            continue

        previous_step = previous_steps[index - 1] if index - 1 < len(previous_steps) else None
        previous_action = str(previous_step.get("action") or "").upper() if previous_step else ""
        previous_target = normalize_step_target_text(previous_step.get("target")) if previous_step else ""
        if previous_action == action and previous_target.lower() == target.lower():
            continue

        if not is_observation_target_executable(observation, action=action, target=target):
            violations.append(
                {
                    "step_index": index,
                    "action": action,
                    "target": target,
                    "reason": "not_in_current_observation",
                }
            )

    return violations


def extract_translator_error_target(failure_info: dict[str, Any]) -> dict[str, str] | None:
    fail_reason = str(failure_info.get("fail_reason", "") or "")
    match = re.search(
        r"translator_error:\s+Could not resolve\s+(click|input|type)\s+target\s+'([^']+)'",
        fail_reason,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    raw_action = match.group(1).strip().lower()
    action = "TYPE" if raw_action in {"input", "type"} else "CLICK"
    target = normalize_step_target_text(match.group(2))
    if not target:
        return None
    return {"action": action, "target": target}


def should_block_generic_repair_target(site: str, target: str) -> bool:
    normalized_target = normalize_step_target_text(target).lower()
    if not normalized_target:
        return False
    if site not in {"reddit", "shopping"}:
        return False
    return normalized_target in _REPAIR_BLOCKED_GENERIC_TARGETS


def find_nearest_legal_target(
    observation: dict[str, Any],
    *,
    action: str,
    target: str,
    site: str = "",
) -> str | None:
    normalized_target = normalize_step_target_text(target)
    if not normalized_target:
        return None

    executable = observation_executable_targets(observation)
    if action == "CLICK":
        candidates = executable["clickable_texts"] + executable["tab_titles"] + executable["clickable_bids"]
    elif action == "TYPE":
        candidates = executable["input_labels"] + executable["input_bids"]
    else:
        return None

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized_candidate = normalize_step_target_text(candidate)
        if not normalized_candidate:
            continue
        if action == "CLICK" and should_block_generic_repair_target(site, normalized_candidate):
            continue
        lowered = normalized_candidate.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped_candidates.append(normalized_candidate)

    if not deduped_candidates:
        return None

    target_tokens = set(re.findall(r"[a-z0-9]+", normalized_target.lower()))
    scored_candidates: list[tuple[float, int, str]] = []
    for candidate in deduped_candidates:
        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        overlap = len(target_tokens & candidate_tokens)
        ratio = difflib.SequenceMatcher(None, normalized_target.lower(), candidate.lower()).ratio()
        scored_candidates.append((ratio, overlap, candidate))

    scored_candidates.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
    best_ratio, best_overlap, best_candidate = scored_candidates[0]
    if best_overlap <= 0 and best_ratio < 0.55:
        return None
    if best_ratio < 0.4:
        return None
    return best_candidate


def build_translator_error_target_repair(
    current_skill: dict[str, Any],
    observation: dict[str, Any],
    failure_info: dict[str, Any],
    *,
    site: str = "",
) -> dict[str, Any] | None:
    translator_target = extract_translator_error_target(failure_info)
    if translator_target is None:
        return None

    repaired = {
        "skill_id": current_skill.get("skill_id", "generated_skill"),
        "task": current_skill.get("task", ""),
        "preconditions": dict(current_skill.get("preconditions", {}) or {}),
        "success_contract": dict(current_skill.get("success_contract", {}) or {}),
        "steps": [dict(step) for step in current_skill.get("steps", []) or []],
        "repair_history": list(current_skill.get("repair_history", []) or []),
        "patches": list(current_skill.get("patches", []) or []),
    }

    replacement = find_nearest_legal_target(
        observation,
        action=translator_target["action"],
        target=translator_target["target"],
        site=site,
    )
    if replacement is None:
        return None

    for step in repaired["steps"]:
        action = str(step.get("action") or "").upper()
        target = normalize_step_target_text(step.get("target"))
        if action != translator_target["action"]:
            continue
        if target.lower() != translator_target["target"].lower():
            continue
        step["target"] = replacement
        repaired["repair_history"].append(
            f"translator target replacement: {translator_target['target']} -> {replacement}"
        )
        return repaired

    return None


def repair_targets_against_observation(
    previous: dict[str, Any],
    updated: dict[str, Any],
    observation: dict[str, Any],
    *,
    site: str = "",
) -> dict[str, Any]:
    previous_steps = previous.get("steps", []) or []
    updated_steps = updated.get("steps", []) or []
    if not updated_steps:
        return updated

    repaired = {
        **updated,
        "steps": [dict(step) for step in updated_steps],
        "repair_history": list(updated.get("repair_history", []) or []),
    }
    changed = False

    for index, step in enumerate(repaired["steps"], start=1):
        action = str(step.get("action") or "").upper()
        if action not in {"CLICK", "TYPE"}:
            continue

        target = normalize_step_target_text(step.get("target"))
        if not target:
            continue

        previous_step = previous_steps[index - 1] if index - 1 < len(previous_steps) else None
        previous_action = str(previous_step.get("action") or "").upper() if previous_step else ""
        previous_target = normalize_step_target_text(previous_step.get("target")) if previous_step else ""

        if previous_action == action and previous_target.lower() == target.lower():
            continue

        target_is_executable = is_observation_target_executable(
            observation,
            action=action,
            target=target,
        )
        blocked_generic = action == "CLICK" and should_block_generic_repair_target(site, target)
        if target_is_executable and not blocked_generic:
            continue

        replacement = None
        if previous_action == action and previous_target:
            previous_is_executable = is_observation_target_executable(
                observation,
                action=action,
                target=previous_target,
            )
            if previous_is_executable and not (
                action == "CLICK" and should_block_generic_repair_target(site, previous_target)
            ):
                replacement = previous_target
            else:
                replacement = find_nearest_legal_target(
                    observation,
                    action=action,
                    target=previous_target,
                    site=site,
                )

        if replacement is None:
            replacement = find_nearest_legal_target(
                observation,
                action=action,
                target=target,
                site=site,
            )

        if not replacement or replacement.lower() == target.lower():
            continue

        step["target"] = replacement
        repaired["repair_history"].append(
            f"observation target replacement: {target} -> {replacement}"
        )
        changed = True

    if not changed:
        return updated
    return repaired


def localize_failure(execution_trace: list[dict[str, Any]], fail_reason: str) -> dict[str, Any]:
    if not fail_reason:
        failed_step_index = execution_trace[-1]["step_index"] if execution_trace else 0
        return {
            "failed_step_index": failed_step_index,
            "failure_source": "",
            "error_code": "",
            "fail_reason": "",
        }

    failed_step_index = execution_trace[-1]["step_index"] if execution_trace else 0
    error_code = (fail_reason or "unknown_failure").split(":", 1)[0].strip() or "unknown_failure"

    failure_source = "contract_failed"
    if error_code in {"invalid_skill_json", "invalid_skill_schema"}:
        failure_source = "schema_invalid"
    elif error_code.startswith("infra_"):
        failure_source = "infra_failed"
    elif error_code in {
        "precondition_failed",
        "success_contract_failed",
        "contract_satisfied_but_reward_zero",
        "max_steps_exceeded",
    }:
        failure_source = "contract_failed"
    elif error_code in {
        "invalid_action_format",
        "translator_error",
        "browsergym_step_error",
        "environment_terminated_without_success",
    }:
        failure_source = "execution_failed"

    return {
        "failed_step_index": failed_step_index,
        "failure_source": failure_source,
        "error_code": error_code,
        "fail_reason": fail_reason,
    }


def summarize_skill_diff(previous: dict[str, Any], updated: dict[str, Any]) -> list[str]:
    lines: list[str] = []

    previous_steps = previous.get("steps", [])
    updated_steps = updated.get("steps", [])
    if len(previous_steps) != len(updated_steps):
        lines.append(f"step_count: {len(previous_steps)} -> {len(updated_steps)}")

    for index, (before, after) in enumerate(zip(previous_steps, updated_steps), start=1):
        if before != after:
            lines.append(
                f"step_{index}: {skill_to_action_string(before)} -> {skill_to_action_string(after)}"
            )

    if previous.get("preconditions") != updated.get("preconditions"):
        lines.append("preconditions updated")
    if previous.get("success_contract") != updated.get("success_contract"):
        lines.append("success_contract updated")

    previous_history = previous.get("repair_history", [])
    updated_history = updated.get("repair_history", [])
    if len(updated_history) > len(previous_history):
        lines.append("repair_history appended")

    previous_patches = previous.get("patches", [])
    updated_patches = updated.get("patches", [])
    if previous_patches != updated_patches:
        lines.append("patches updated")

    if not lines:
        lines.append("no_structural_change_detected")
    return lines


def preserve_navigation_prefix_for_repair(
    previous: dict[str, Any],
    updated: dict[str, Any],
    initial_observation: dict[str, Any],
    latest_observation: dict[str, Any],
) -> dict[str, Any]:
    previous_steps = previous.get("steps", [])
    updated_steps = updated.get("steps", [])
    if not previous_steps or not updated_steps:
        return updated

    previous_first = previous_steps[0]
    updated_first = updated_steps[0]
    previous_target = str(previous_first.get("target") or "").strip()
    updated_target = str(updated_first.get("target") or "").strip()

    if previous_first.get("action") != "CLICK" or updated_first.get("action") != "CLICK":
        return updated
    if not previous_target or not updated_target:
        return updated
    if previous_target.lower() == updated_target.lower():
        return updated

    initial_tabs = {
        str(title).strip().lower()
        for title in (initial_observation.get("open_pages_titles") or [])
        if str(title).strip()
    }
    initial_clicks = {
        str(item.get("text") or "").strip().lower()
        for item in (initial_observation.get("clickable_elements") or [])
        if str(item.get("text") or "").strip()
    }
    previous_target_lc = previous_target.lower()
    updated_target_lc = updated_target.lower()

    def is_navigation_like_click_target(target_lc: str) -> bool:
        if target_lc in initial_tabs:
            return True
        if target_lc in _GENERIC_NAVIGATION_CLICK_TARGETS:
            return True
        if target_lc.startswith(("sort by:", "from:")):
            return True
        if re.fullmatch(r"(?:[0-9]+|>|<|next|prev|previous)", target_lc):
            return True
        return False

    is_navigation_prefix = is_navigation_like_click_target(previous_target_lc)
    is_later_page_target = (
        updated_target_lc not in initial_clicks and updated_target_lc not in initial_tabs
    )

    if not is_navigation_prefix or not is_later_page_target:
        return updated

    merged = dict(updated)
    merged["steps"] = [dict(previous_first)] + [dict(step) for step in updated_steps]
    merged_history = list(merged.get("repair_history") or [])
    merged_history.append(
        f"preserved navigation prefix: keep CLICK[{previous_target}] before CLICK[{updated_target}]"
    )
    merged["repair_history"] = merged_history
    return merged


def allowed_patch_types_for_failure(failure_info: dict[str, Any]) -> set[str]:
    error_code = str(failure_info.get("error_code", "") or "").strip()
    if error_code == "precondition_failed":
        return {"selector_replacement", "argument_correction", "precondition_insertion"}
    if error_code in {"translator_error", "browsergym_step_error"}:
        return {"selector_replacement", "argument_correction", "recovery_insertion"}
    if error_code in {
        "contract_satisfied_but_reward_zero",
        "success_contract_failed",
        "environment_terminated_without_success",
        "max_steps_exceeded",
    }:
        return {"selector_replacement", "argument_correction", "recovery_insertion"}
    return set(ALLOWED_PATCH_TYPES)


def has_execution_equivalent_update(
    previous: dict[str, Any],
    updated: dict[str, Any],
) -> bool:
    return previous.get("steps", []) == updated.get("steps", [])



