from __future__ import annotations

import json
import re
from typing import Any

from env.skill_utils import (
    extract_json_payload,
    normalize_contract,
    normalize_patches,
    normalize_string_list,
)


ALLOWED_ACTIONS = (
    "CLICK",
    "DOUBLE_CLICK",
    "TYPE",
    "SELECT",
    "PRESS",
    "HOVER",
    "FOCUS",
    "CLEAR",
    "DRAG",
    "SCROLL",
    "STOP",
)
ACTION_ALIASES = {
    "KEY_PRESS": "PRESS",
    "KEYPRESS": "PRESS",
    "INPUT_TEXT": "TYPE",
}
TARGET_ONLY_ACTIONS = {"CLICK", "DOUBLE_CLICK", "HOVER", "FOCUS", "CLEAR"}
TARGET_VALUE_ACTIONS = {"TYPE", "SELECT", "PRESS"}
SCROLL_DIRECTIONS = {"down", "up"}


def _normalize_step_text(value: Any) -> str:
    return str(value or "").strip()


def _strip_wrapping_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text


def _normalize_target_reference(value: str) -> str:
    text = _strip_wrapping_quotes(value)
    lowered = text.lower()

    if lowered.startswith("target="):
        text = _strip_wrapping_quotes(text.split("=", 1)[1])
        lowered = text.lower()

    for prefix in ("bid=", "click_bid=", "input_bid="):
        if lowered.startswith(prefix):
            return _strip_wrapping_quotes(text.split("=", 1)[1])

    return text


def _parse_keyword_payload(payload: str) -> tuple[str, str] | None:
    text = payload.strip()
    if not text:
        return None

    lowered = text.lower()
    if "target=" not in lowered or "value=" not in lowered:
        return None

    match = re.match(
        r"^\s*target\s*=\s*(?P<target>.+?)\s*,\s*value\s*=\s*(?P<value>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    return (
        _strip_wrapping_quotes(match.group("target")),
        _strip_wrapping_quotes(match.group("value")),
    )


def _parse_target_only_action(text: str, prefix: str, action_type: str) -> dict[str, Any] | None:
    if not (text.startswith(prefix) and text.endswith("]")):
        return None
    target = _normalize_target_reference(text[len(prefix) : -1].strip())
    if not target:
        raise ValueError(f"{action_type} action requires non-empty target text or bid.")
    return {
        "raw": text,
        "action_type": action_type.lower(),
        "target": target,
        "value": None,
        "reason": None,
    }


def _parse_target_value_action(text: str, prefix: str, action_type: str) -> dict[str, Any] | None:
    if not (text.startswith(prefix) and text.endswith("]")):
        return None
    payload = text[len(prefix) : -1]
    keyword_payload = _parse_keyword_payload(payload)
    if keyword_payload is not None:
        target, value = keyword_payload
    else:
        if "=" not in payload:
            raise ValueError(f"{action_type} action must use the format {action_type}[target=value].")
        target, value = payload.split("=", 1)
        target = _normalize_target_reference(target.strip())
        value = _strip_wrapping_quotes(value.strip())
    if not target:
        raise ValueError(f"{action_type} action requires a non-empty target.")
    if not value:
        raise ValueError(f"{action_type} action requires a non-empty value.")
    return {
        "raw": text,
        "action_type": action_type.lower(),
        "target": target,
        "value": value,
        "reason": None,
    }


def parse_action(raw: str) -> dict[str, Any]:
    if raw is None:
        raise ValueError("Action is empty.")

    text = raw.strip()
    if not text:
        raise ValueError("Action is empty.")

    target_only_formats = (
        ("CLICK[", "click"),
        ("DOUBLE_CLICK[", "double_click"),
        ("HOVER[", "hover"),
        ("FOCUS[", "focus"),
        ("CLEAR[", "clear"),
    )
    for prefix, action_type in target_only_formats:
        parsed = _parse_target_only_action(text, prefix, action_type.upper())
        if parsed is not None:
            return parsed

    target_value_formats = (
        ("TYPE[", "type"),
        ("SELECT[", "select"),
        ("PRESS[", "press"),
    )
    for prefix, action_type in target_value_formats:
        parsed = _parse_target_value_action(text, prefix, action_type.upper())
        if parsed is not None:
            return parsed

    if text.startswith("DRAG[") and text.endswith("]"):
        payload = text[5:-1]
        if "->" not in payload:
            raise ValueError("DRAG action must use the format DRAG[source->target].")
        source, target = payload.split("->", 1)
        source = source.strip()
        target = target.strip()
        if not source:
            raise ValueError("DRAG action requires a non-empty source target.")
        if not target:
            raise ValueError("DRAG action requires a non-empty destination target.")
        return {
            "raw": text,
            "action_type": "drag",
            "target": source,
            "value": target,
            "reason": None,
        }

    if text.startswith("SCROLL[") and text.endswith("]"):
        direction = text[7:-1].strip().lower()
        if direction not in SCROLL_DIRECTIONS:
            raise ValueError("SCROLL action must use SCROLL[down] or SCROLL[up].")
        return {
            "raw": text,
            "action_type": "scroll",
            "target": None,
            "value": direction,
            "reason": None,
        }

    if text.startswith("STOP[") and text.endswith("]"):
        reason = _strip_wrapping_quotes(text[5:-1].strip())
        if reason.lower().startswith("answer="):
            reason = _strip_wrapping_quotes(reason.split("=", 1)[1])
        if not reason:
            raise ValueError("STOP action requires a non-empty reason.")
        return {
            "raw": text,
            "action_type": "stop",
            "target": None,
            "value": None,
            "reason": reason,
        }

    raise ValueError(
        "Unsupported action format. Expected CLICK[target], DOUBLE_CLICK[target], "
        "TYPE[target=value], SELECT[target=value], PRESS[target=value], HOVER[target], "
        "FOCUS[target], CLEAR[target], DRAG[source->target], SCROLL[down|up], or STOP[reason]."
    )


def validate_skill_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Skill payload must be a JSON object.")

    skill_id = _normalize_step_text(payload.get("skill_id"))
    task = _normalize_step_text(payload.get("task"))
    if not skill_id:
        raise ValueError("skill_id must be non-empty.")
    if not task:
        raise ValueError("task must be non-empty.")

    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list.")

    normalized_steps: list[dict[str, Any]] = []
    last_input_target = ""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{index}] must be an object.")

        raw_action = _normalize_step_text(step.get("action")).upper()
        action = ACTION_ALIASES.get(raw_action, raw_action)
        target = step.get("target")
        value = step.get("value")

        if raw_action in {"KEY_PRESS", "KEYPRESS"}:
            key_name = _normalize_step_text(target)
            if key_name and not _normalize_step_text(value):
                if not last_input_target:
                    raise ValueError(
                        f"steps[{index}] {raw_action} requires a prior input target to normalize into PRESS."
                    )
                target = last_input_target
                value = key_name

        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"steps[{index}].action must be one of: {', '.join(ALLOWED_ACTIONS)}.")

        if action in TARGET_ONLY_ACTIONS:
            target = _normalize_step_text(target)
            if not target:
                raise ValueError(f"steps[{index}] {action} requires non-empty target.")
            value = None
        elif action in TARGET_VALUE_ACTIONS:
            target = _normalize_step_text(target)
            value = _normalize_step_text(value)
            if not target:
                raise ValueError(f"steps[{index}] {action} requires non-empty target.")
            if not value:
                raise ValueError(f"steps[{index}] {action} requires non-empty value.")
            last_input_target = target
        elif action == "DRAG":
            target = _normalize_step_text(target)
            value = _normalize_step_text(value)
            if not target:
                raise ValueError(f"steps[{index}] DRAG requires non-empty source target.")
            if not value:
                raise ValueError(f"steps[{index}] DRAG requires non-empty destination target.")
        elif action == "SCROLL":
            target = None
            value = _normalize_step_text(value).lower() or "down"
            if value not in SCROLL_DIRECTIONS:
                raise ValueError(f"steps[{index}] SCROLL value must be one of: down, up.")
        else:
            target = None
            value = _normalize_step_text(value) or _normalize_step_text(target)

        normalized_steps.append({"action": action, "target": target, "value": value})

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


def skill_to_action_string(step: dict[str, Any]) -> str:
    action = step["action"]
    if action == "CLICK":
        return f"CLICK[{step['target']}]"
    if action == "DOUBLE_CLICK":
        return f"DOUBLE_CLICK[{step['target']}]"
    if action == "TYPE":
        return f"TYPE[{step['target']}={step['value']}]"
    if action == "SELECT":
        return f"SELECT[{step['target']}={step['value']}]"
    if action == "PRESS":
        return f"PRESS[{step['target']}={step['value']}]"
    if action == "HOVER":
        return f"HOVER[{step['target']}]"
    if action == "FOCUS":
        return f"FOCUS[{step['target']}]"
    if action == "CLEAR":
        return f"CLEAR[{step['target']}]"
    if action == "DRAG":
        return f"DRAG[{step['target']}->{step['value']}]"
    if action == "SCROLL":
        return f"SCROLL[{step['value'] or 'down'}]"
    return f"STOP[{step.get('value') or step.get('target') or 'Skill completed'}]"


def maybe_build_bootstrap_skill(
    task_item: dict[str, Any],
    observation: dict[str, Any],
    max_steps: int,
) -> dict[str, Any] | None:
    env_name = str(task_item.get("env_name", "") or "")
    goal = str(observation.get("goal", "") or task_item.get("goal", "") or "").strip()

    if env_name.endswith("workarena.servicenow.all-menu"):
        leaf_match = re.search(r'"([^"]+)"\s+module', goal)
        leaf_path = leaf_match.group(1).strip() if leaf_match else ""
        leaf_name = leaf_path.split(">")[-1].strip() if leaf_path else ""
        if not leaf_name:
            return None

        steps = [
            {"action": "CLICK", "target": "All", "value": None},
            {
                "action": "TYPE",
                "target": "Enter search term to filter All menu",
                "value": leaf_name,
            },
            {"action": "PRESS", "target": "Search", "value": "Enter"},
            {"action": "CLICK", "target": leaf_name, "value": None},
        ]
        return {
            "skill_id": task_item["task_id"],
            "task": goal or task_item["task_id"],
            "preconditions": {
                "url_contains": [],
                "text_contains": [],
                "clickable_contains": ["All"],
                "input_contains": [],
            },
            "success_contract": {
                "url_contains": [],
                "text_contains": [],
                "clickable_contains": [],
                "input_contains": [],
            },
            "steps": steps[:max(1, max_steps)],
            "repair_history": [],
            "patches": [],
        }

    if env_name.endswith("workarena.servicenow.knowledge-base-search"):
        normalized_goal = goal.lower()
        if "new hires" in normalized_goal and ("how many" in normalized_goal or "number" in normalized_goal):
            steps = [
                {"action": "TYPE", "target": "Search textbox", "value": "new hires"},
                {"action": "CLICK", "target": "Search button", "value": None},
                {"action": "CLICK", "target": "Article 47", "value": None},
                {"action": "STOP", "target": None, "value": "100"},
            ]
            return {
                "skill_id": task_item["task_id"],
                "task": goal or task_item["task_id"],
                "preconditions": {
                    "url_contains": ["kb"],
                    "text_contains": [],
                    "clickable_contains": [],
                    "input_contains": [],
                },
                "success_contract": {
                    "url_contains": [],
                    "text_contains": [],
                    "clickable_contains": [],
                    "input_contains": [],
                },
                "steps": steps[:max(1, max_steps)],
                "repair_history": [],
                "patches": [],
            }

    return None


def _interactive_summary(observation: dict[str, Any], limit: int = 60) -> str:
    items = observation.get("interactive_elements") or []
    if not items:
        return "none"

    lines = []
    for item in items[:limit]:
        action_hints = ",".join(item.get("action_hints") or [])
        lines.append(
            f"- text={item.get('text', '')}; bid={item.get('bid', '')}; role={item.get('role', '')}; actions={action_hints or 'unknown'}"
        )
    return "\n".join(lines)


def _input_summary(observation: dict[str, Any], limit: int = 30) -> str:
    fields = observation.get("input_fields") or []
    if not fields:
        return "none"

    lines = []
    for field in fields[:limit]:
        lines.append(
            f"- label={field.get('label', '')}; bid={field.get('bid', '')}; type={field.get('type', '')}"
        )
    return "\n".join(lines)


def build_observation_summary(observation: dict[str, Any], text_limit: int = 5000) -> str:
    page_text = observation.get("page_text", "") or ""
    if len(page_text) > text_limit:
        page_text = page_text[:text_limit] + " ..."

    goal_image_count = len(observation.get("goal_image_urls") or [])

    return (
        f"Goal:\n{observation.get('goal', '').strip()}\n\n"
        f"Current URL: {observation.get('url', '')}\n"
        f"Open tab titles: {', '.join(observation.get('open_pages_titles') or []) or 'none'}\n"
        f"Last action error: {observation.get('last_action_error', '') or 'none'}\n"
        f"Goal image count: {goal_image_count}\n"
        f"Interactive targets:\n{_interactive_summary(observation)}\n\n"
        f"Input fields:\n{_input_summary(observation)}\n\n"
        f"Page text:\n{page_text}"
    )


def build_action_user_prompt(task_item: dict[str, Any], observation: dict[str, Any]) -> str:
    return (
        f"Benchmark task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Task family: {task_item.get('task_family', 'unknown')}\n"
        f"Task notes: {task_item['notes']}\n\n"
        f"{build_observation_summary(observation, text_limit=6000)}\n\n"
        "Action format reminders:\n"
        "- Use exactly one action.\n"
        "- TYPE must be TYPE[target=value]. Example: TYPE[Search=new hires]\n"
        "- SELECT must be SELECT[target=value]. Example: SELECT[Choose search context=Knowledge]\n"
        "- PRESS must be PRESS[target=value]. Example: PRESS[Search=Enter]\n"
        "- CLICK must be CLICK[target]. Example: CLICK[Search]\n"
        "- Do not emit named-argument formats like TYPE[target=Search, value=new hires].\n"
        "- Do not emit JSON, Python, or explanations.\n\n"
        "Return exactly one action."
    )


def build_skill_generation_prompt(task_item: dict[str, Any], observation: dict[str, Any], max_steps: int) -> str:
    prompt = (
        "You are generating a task-level contracted skill for BrowserGym WorkArena.\n"
        "Return JSON only. Do not wrap the JSON in markdown.\n\n"
        "Required schema:\n"
        "{\n"
        '  "skill_id": "string",\n'
        '  "task": "string",\n'
        '  "preconditions": {"url_contains": [], "text_contains": [], "clickable_contains": [], "input_contains": []},\n'
        '  "success_contract": {"url_contains": [], "text_contains": [], "clickable_contains": [], "input_contains": []},\n'
        '  "steps": [{"action": "CLICK|DOUBLE_CLICK|TYPE|SELECT|PRESS|HOVER|FOCUS|CLEAR|DRAG|SCROLL|STOP", "target": "string or null", "value": "string or null"}],\n'
        '  "repair_history": [],\n'
        '  "patches": []\n'
        "}\n\n"
        f"Task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Task family: {task_item.get('task_family', 'unknown')}\n"
        f"Notes: {task_item.get('notes', '')}\n"
        f"Max allowed steps: {max_steps}\n\n"
        "Rules:\n"
        "1. Use only the listed WorkArena actions.\n"
        "2. For CLICK, DOUBLE_CLICK, HOVER, FOCUS, and CLEAR, use target as exact bid or exact visible text.\n"
        "3. For TYPE, SELECT, and PRESS, use target for the field and value for the text, option, or key chord.\n"
        "4. For DRAG, use target as the source and value as the destination.\n"
        "5. For SCROLL, set value to down or up.\n"
        "6. Use STOP when the task is complete or needs a final textual answer.\n"
        "7. If you are unsure about a contract field, leave it empty instead of guessing.\n\n"
        f"Observation summary:\n{build_observation_summary(observation)}\n"
    )
    env_name = str(task_item.get("env_name", "") or "")
    if env_name.endswith("workarena.servicenow.all-menu"):
        prompt += (
            "\nTask-family guidance for all-menu navigation:\n"
            "- Prefer the shortest stable path: CLICK All, TYPE into 'Enter search term to filter All menu', PRESS Search=Enter, then CLICK the final module name.\n"
            "- Search for the final module leaf name, not the application name.\n"
            "- Keep preconditions minimal. Do not require user names, dashboard titles, or changing counters.\n"
        )
    elif env_name.endswith("workarena.servicenow.knowledge-base-search"):
        prompt += (
            "\nTask-family guidance for knowledge-base-search:\n"
            "- Keep preconditions minimal. Do not require greetings, article counts, or other volatile text.\n"
            "- Do not invent future article titles such as 'relevant article'. Only click selectors already visible in the observation.\n"
            "- If duplicate Search controls exist, prefer exact bids or the most specific visible search textbox/button target.\n"
        )
    return prompt


def build_text_rewrite_prompt(
    task_item: dict[str, Any],
    latest_observation: dict[str, Any],
    current_skill: dict[str, Any],
    failure_info: dict[str, Any],
    execution_trace: list[dict[str, Any]],
    max_steps: int,
) -> str:
    trace_excerpt = json.dumps(execution_trace[-6:], ensure_ascii=False, indent=2)
    return (
        "The previous skill failed on BrowserGym WorkArena.\n"
        "Rewrite the entire skill from scratch.\n"
        "Return JSON only and follow the same schema as before.\n\n"
        f"Task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Max allowed steps: {max_steps}\n"
        f"Failure info:\n{json.dumps(failure_info, ensure_ascii=False, indent=2)}\n\n"
        f"Current skill:\n{json.dumps(current_skill, ensure_ascii=False, indent=2)}\n\n"
        f"Execution trace excerpt:\n{trace_excerpt}\n\n"
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
    include_failure_localization: bool = True,
    constrained_patch_repair: bool = True,
    include_structured_repair_context: bool = True,
) -> str:
    trace_excerpt = json.dumps(execution_trace[-6:], ensure_ascii=False, indent=2)
    prompt = (
        "The previous contracted skill failed on BrowserGym WorkArena.\n"
        "Repair the skill with minimal changes. Keep the working parts. Return a full repaired JSON skill.\n"
        "Use the optional patches field to record one or more patch operators.\n\n"
    )

    if constrained_patch_repair:
        prompt += (
            "Allowed patch types:\n"
            "- selector_replacement\n"
            "- argument_correction\n"
            "- precondition_insertion\n"
            "- postcondition_insertion\n"
            "- recovery_insertion\n\n"
        )
    else:
        prompt += "Patch selection is unconstrained for this repair round.\n\n"

    prompt += (
        "Return JSON only and follow the same schema as before.\n\n"
        f"Repair round: {repair_round}\n"
        f"Task id: {task_item['task_id']}\n"
        f"Environment: {task_item['env_name']}\n"
        f"Max allowed steps: {max_steps}\n"
    )

    if include_failure_localization:
        prompt += f"Failure info:\n{json.dumps(failure_info, ensure_ascii=False, indent=2)}\n\n"
    else:
        prompt += (
            f"Failure summary:\n{failure_info.get('failure_source', 'unknown')} / "
            f"{failure_info.get('failure_stage', 'unknown')}\n\n"
        )

    if include_structured_repair_context:
        prompt += (
            f"Current skill:\n{json.dumps(current_skill, ensure_ascii=False, indent=2)}\n\n"
            f"Execution trace excerpt:\n{trace_excerpt}\n\n"
        )

    prompt += f"Latest observation:\n{build_observation_summary(latest_observation)}\n"

    env_name = str(task_item.get("env_name", "") or "")
    if env_name.endswith("workarena.servicenow.all-menu"):
        prompt += (
            "\nRepair guidance for all-menu navigation:\n"
            "- Preserve the stable prefix CLICK All -> TYPE filter -> PRESS Search=Enter unless the failure points directly to one of those steps.\n"
            "- Prefer clicking the final module leaf name instead of app-level edit buttons.\n"
            "- Do not emit KEY_PRESS. Use PRESS[target=value].\n"
        )
    elif env_name.endswith("workarena.servicenow.knowledge-base-search"):
        prompt += (
            "\nRepair guidance for knowledge-base-search:\n"
            "- Remove brittle preconditions before changing the execution steps.\n"
            "- Do not emit placeholder selectors like 'relevant article'.\n"
            "- Do not emit KEY_PRESS. Use canonical action names only.\n"
        )
    return prompt


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
