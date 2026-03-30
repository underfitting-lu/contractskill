from __future__ import annotations

import re


def _extract_action_core(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    text = text.strip("`").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    # Recover common non-canonical action spellings that some models emit.
    fill_match = re.fullmatch(r"""fill\(\s*['"]?([^,'")]+)['"]?\s*,\s*['"](.+?)['"]\s*\)""", text)
    if fill_match:
        target = fill_match.group(1).strip()
        value = fill_match.group(2).strip()
        if target.isdigit():
            return f"TYPE[bid={target}={value}]"
        return f"TYPE[{target}={value}]"

    click_match = re.fullmatch(r"""click\(\s*['"]?([^'")]+)['"]?\s*\)""", text)
    if click_match:
        target = click_match.group(1).strip()
        if target.isdigit():
            return f"CLICK[bid={target}]"
        return f"CLICK[{target}]"

    for prefix, replacement in (
        ("LIVE_TYPE[", "TYPE["),
        ("LIVE_CLICK[", "CLICK["),
        ("LIVE_SCROLL[", "SCROLL["),
        ("LIVE_STOP[", "STOP["),
    ):
        if text.startswith(prefix):
            return replacement + text[len(prefix) :]

    return text


def _extract_bid_target(target: str) -> tuple[str, str | None]:
    cleaned = target.strip()
    lowered = cleaned.lower()
    for prefix in ("bid=", "input_bid=", "click_bid="):
        if lowered.startswith(prefix):
            suffix = cleaned[len(prefix) :].strip()
            if "=" in suffix:
                suffix = suffix.split("=", 1)[0].strip()
            return suffix, "bid"
    return cleaned, None


def _normalize_type_target(target: str) -> str:
    cleaned = target.strip()
    bid_target, bid_mode = _extract_bid_target(cleaned)
    if bid_mode == "bid":
        return bid_target
    for prefix in ("field=", "label=", "input="):
        if cleaned.lower().startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


def parse_action(raw: str) -> dict:
    """Parse a model action string into a normalized action dictionary."""
    if raw is None:
        raise ValueError("Action is empty.")

    text = _extract_action_core(raw)
    if not text:
        raise ValueError("Action is empty.")

    if text == "SCROLL[down]":
        return {
            "raw": text,
            "action_type": "scroll",
            "target": None,
            "value": "down",
            "reason": None,
        }

    if text.startswith("CLICK[") and text.endswith("]"):
        target = text[6:-1].strip()
        if not target:
            raise ValueError("CLICK action requires non-empty text inside brackets.")
        normalized_target, target_mode = _extract_bid_target(target)
        return {
            "raw": text,
            "action_type": "click",
            "target": normalized_target,
            "value": None,
            "reason": None,
            "target_mode": target_mode or "text",
        }

    if text.startswith("TYPE["):
        target = ""
        value = ""

        if text.endswith("]"):
            payload = text[5:-1]
            lowered_payload = payload.lower()
            if lowered_payload.startswith(("bid=", "input_bid=")):
                prefix = "input_bid=" if lowered_payload.startswith("input_bid=") else "bid="
                second_equals = payload.find("=", len(prefix))
                if second_equals == -1:
                    raise ValueError("TYPE action must use the format TYPE[field=value].")
                target = payload[:second_equals]
                value = payload[second_equals + 1 :]
            elif "=" not in payload:
                raise ValueError("TYPE action must use the format TYPE[field=value].")
            else:
                target, value = payload.split("=", 1)
        elif "]=" in text:
            close_index = text.find("]=")
            if close_index <= 5:
                raise ValueError("TYPE action must use the format TYPE[field=value].")
            target = text[5:close_index]
            value = text[close_index + 2 :]
        else:
            raise ValueError("TYPE action must use the format TYPE[field=value].")

        raw_target = target.strip()
        target = _normalize_type_target(target)
        _, target_mode = _extract_bid_target(raw_target)
        if target_mode is None and raw_target.isdigit():
            target_mode = "bid"
        value = value.strip()
        if not target:
            raise ValueError("TYPE action requires a non-empty field name.")
        if not value:
            raise ValueError("TYPE action requires a non-empty value.")
        return {
            "raw": text,
            "action_type": "type",
            "target": target,
            "value": value,
            "reason": None,
            "target_mode": target_mode or "text",
        }

    if text.startswith("STOP[") and text.endswith("]"):
        reason = text[5:-1].strip()
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
        "Unsupported action format. Expected CLICK[text], CLICK[bid=123], TYPE[field=value], "
        "SCROLL[down], or STOP[reason]."
    )
