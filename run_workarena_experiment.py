from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import run_vwa_experiment as base
from env.workarena_env import WorkArenaEnv
from env.workarena_runtime import (
    apply_workarena_runtime_patches,
    get_workarena_browser_timeout_ms,
)
from env.workarena_skill_utils import (
    build_action_user_prompt,
    build_contract_repair_prompt,
    build_skill_generation_prompt,
    build_text_rewrite_prompt,
    maybe_build_bootstrap_skill,
    parse_action,
    parse_skill_response,
    skill_to_action_string,
    summarize_skill_diff,
)
from glm_client import GLMClient
from scripts.check_workarena_env import load_local_env_file


DEFAULT_MODEL = "glm-4.6v"
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_REPAIRS = 5
DEFAULT_MAX_MODEL_CALLS = 6
DEFAULT_BROWSER_TIMEOUT_MS = get_workarena_browser_timeout_ms()
DEFAULT_RESET_TIMEOUT_SEC = 0
DEFAULT_STEP_TIMEOUT_SEC = 0
DEFAULT_RESET_RETRIES = 1
DEFAULT_TASK_TIMEOUT_SEC = 0
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = REPO_ROOT / "tasks" / "workarena_smoke_5.json"

OUTPUT_ROOT = REPO_ROOT / "outputs" / "workarena_experiments"
SCREENSHOT_ROOT = OUTPUT_ROOT / "screenshots"
TRACE_ROOT = OUTPUT_ROOT / "traces"
RESULT_ROOT = OUTPUT_ROOT / "results"
SKILL_ROOT = OUTPUT_ROOT / "skills"
INITIAL_SKILL_ROOT = OUTPUT_ROOT / "shared_initial_skills"
INITIAL_SKILL_CACHE_VERSION = "wa_v1"

ACTION_SYSTEM_PROMPT = """You are a web agent acting inside BrowserGym WorkArena.
Given the task goal, the current page state, and the current page screenshot, output exactly one next action.

Allowed formats:
CLICK[target]
DOUBLE_CLICK[target]
TYPE[target=value]
SELECT[target=value]
PRESS[target=value]
HOVER[target]
FOCUS[target]
CLEAR[target]
DRAG[source->target]
SCROLL[down]
SCROLL[up]
STOP[answer]

Rules:
1. Output one action only.
2. target may be an exact visible text label or an exact bid from the provided interactive targets.
3. Prefer bids when there are duplicate texts or the widget has little visible text.
4. TYPE and SELECT use the target field plus a value after '='.
5. PRESS uses the target field plus a key or key chord after '=' such as Enter or ControlOrMeta+a.
6. DRAG uses source->target inside the brackets.
7. Use STOP[answer] when you believe the task is complete or the task requires a final textual answer.
8. Do not output Python code, explanations, or markdown.

Canonical examples:
- CLICK[Search]
- TYPE[Search=new hires]
- SELECT[Choose search context=Knowledge]
- PRESS[Search=Enter]
- STOP[250]

Invalid examples:
- TYPE[target=Search, value=new hires]
- CLICK[target=Search]
- {"action": "TYPE", "target": "Search", "value": "new hires"}
"""

_ORIGINAL_LOAD_OR_CREATE_SHARED_INITIAL_SKILL = base.load_or_create_shared_initial_skill

def patch_base_module() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SCREENSHOT_ROOT = SCREENSHOT_ROOT
    base.TRACE_ROOT = TRACE_ROOT
    base.RESULT_ROOT = RESULT_ROOT
    base.SKILL_ROOT = SKILL_ROOT
    base.INITIAL_SKILL_ROOT = INITIAL_SKILL_ROOT
    base.INITIAL_SKILL_CACHE_VERSION = INITIAL_SKILL_CACHE_VERSION
    base.ACTION_SYSTEM_PROMPT = ACTION_SYSTEM_PROMPT
    base.parse_action = parse_action
    base.parse_skill_response = parse_skill_response
    base.skill_to_action_string = skill_to_action_string
    base.summarize_skill_diff = summarize_skill_diff
    base.build_action_user_prompt = build_action_user_prompt
    base.build_skill_generation_prompt = build_skill_generation_prompt
    base.build_text_rewrite_prompt = build_text_rewrite_prompt
    base.build_contract_repair_prompt = build_contract_repair_prompt
    base.load_or_create_shared_initial_skill = load_or_create_shared_initial_skill


def load_or_create_shared_initial_skill(
    glm: GLMClient,
    env: WorkArenaEnv,
    *,
    task_item: dict,
    split_path: Path,
    cache_key: str | None,
    seed_observation: dict,
    max_steps: int,
) -> dict:
    bootstrap_skill = maybe_build_bootstrap_skill(task_item, seed_observation, max_steps)
    if bootstrap_skill is None:
        return _ORIGINAL_LOAD_OR_CREATE_SHARED_INITIAL_SKILL(
            glm,
            env,
            task_item=task_item,
            split_path=split_path,
            cache_key=cache_key,
            seed_observation=seed_observation,
            max_steps=max_steps,
        )

    prompt = build_skill_generation_prompt(task_item, seed_observation, max_steps=max_steps)
    prompt_hash = hashlib.sha1((prompt + "\nbootstrap:workarena").encode("utf-8")).hexdigest()
    cache_path = base.build_initial_skill_cache_path(
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
                "usage": base.normalize_usage(payload.get("usage")),
                "skill": payload.get("skill"),
                "parse_error": payload.get("parse_error"),
                "generation_mode": payload.get("generation_mode", "bootstrap"),
            }

    raw_output = json.dumps(bootstrap_skill, ensure_ascii=False, indent=2)
    payload = {
        "task_id": task_item["task_id"],
        "env_name": task_item["env_name"],
        "model": glm.model,
        "split_name": split_path.stem,
        "prompt_hash": prompt_hash,
        "raw_output": raw_output,
        "usage": base.zero_usage(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "skill": bootstrap_skill,
        "parse_error": None,
        "generation_mode": "bootstrap_all_menu_v1",
    }
    base.dump_json(cache_path, payload)
    return {
        "cache_hit": False,
        "cache_path": cache_relpath,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "raw_output": raw_output,
        "usage": base.zero_usage(),
        "skill": bootstrap_skill,
        "parse_error": None,
        "generation_mode": "bootstrap_all_menu_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ContractSkill baselines on BrowserGym WorkArena.")
    parser.add_argument(
        "--baseline",
        choices=("no_skill", "skill_no_repair", "text_only_rewrite", "contractskill"),
        required=True,
    )
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-repairs", type=int, default=DEFAULT_MAX_REPAIRS)
    parser.add_argument("--max-model-calls", type=int, default=DEFAULT_MAX_MODEL_CALLS)
    parser.add_argument("--browser-timeout-ms", type=int, default=DEFAULT_BROWSER_TIMEOUT_MS)
    parser.add_argument("--reset-timeout-sec", type=int, default=DEFAULT_RESET_TIMEOUT_SEC)
    parser.add_argument("--step-timeout-sec", type=int, default=DEFAULT_STEP_TIMEOUT_SEC)
    parser.add_argument("--reset-retries", type=int, default=DEFAULT_RESET_RETRIES)
    parser.add_argument("--task-timeout-sec", type=int, default=DEFAULT_TASK_TIMEOUT_SEC)
    parser.add_argument("--headless", type=base.parse_bool, default=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    patch_base_module()
    load_local_env_file()
    apply_workarena_runtime_patches()
    args = parse_args()
    split_path = args.split_path
    if not split_path.is_absolute():
        split_path = (REPO_ROOT / split_path).resolve()
    split = base.load_split(split_path)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for root in (SCREENSHOT_ROOT, TRACE_ROOT, RESULT_ROOT, SKILL_ROOT, INITIAL_SKILL_ROOT):
        base.ensure_dir(root)

    glm = GLMClient(model=args.model)
    env = WorkArenaEnv(
        output_root=SCREENSHOT_ROOT / args.baseline / run_id,
        headless=args.headless,
        browser_timeout_ms=args.browser_timeout_ms,
    )
    env.start()

    results: list[dict] = []

    try:
        for task_item in split:
            env.planned_seed = int(task_item.get("seed", 0) or 0)
            print(f"\n[{args.baseline}] {task_item['task_id']} {task_item['env_name']}")
            try:
                if args.baseline == "no_skill":
                    result, trace = base._run_with_timeout(
                        args.task_timeout_sec,
                        f"task {task_item['task_id']}",
                        base.run_noskill_task,
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
                    result, trace = base._run_with_timeout(
                        args.task_timeout_sec,
                        f"task {task_item['task_id']}",
                        base.run_skill_baseline_task,
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        run_id=run_id,
                        split_path=split_path,
                        baseline=args.baseline,
                        max_steps=args.max_steps,
                        max_repairs=args.max_repairs,
                        max_model_calls=args.max_model_calls,
                        reset_timeout_sec=args.reset_timeout_sec,
                        step_timeout_sec=args.step_timeout_sec,
                        reset_retries=args.reset_retries,
                    )
            except Exception as exc:
                fail_reason = str(exc)
                failure_info = base.localize_failure([], fail_reason)
                result = {
                    "task_id": task_item["task_id"],
                    "env_name": task_item["env_name"],
                    "baseline": args.baseline,
                    **base.get_task_metadata(task_item),
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
                    **base.zero_usage(),
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
                    **base.get_task_metadata(task_item),
                    "notes": task_item.get("notes", ""),
                    "model": glm.model,
                    "success": False,
                    "fail_reason": fail_reason,
                    "failure_info": failure_info,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                }

            results.append(result)
            base.dump_json(TRACE_ROOT / args.baseline / run_id / f"{task_item['task_id']}.json", trace)

            status_text = "SUCCESS" if result["success"] else "FAIL"
            print(f"{status_text}: {result['final_fail_reason'] or 'score > 0'}")
    finally:
        env.close()

    summary = base.aggregate_summary(
        run_id=run_id,
        split_path=split_path,
        baseline=args.baseline,
        model=args.model,
        max_steps=args.max_steps,
        max_repairs=args.max_repairs,
        max_model_calls=args.max_model_calls,
        headless=args.headless,
        results=results,
    )

    summary_base = f"{args.baseline}_{args.split_path.stem}_{run_id}"
    base.dump_json(RESULT_ROOT / args.baseline / f"{summary_base}.json", summary)
    base.dump_json(RESULT_ROOT / args.baseline / f"{args.baseline}_{args.split_path.stem}_latest.json", summary)
    base.dump_jsonl(RESULT_ROOT / args.baseline / f"{summary_base}.jsonl", results)

    print(f"\nBaseline: {summary['baseline']}")
    print(f"Split: {summary['split_name']}")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Success count: {summary['success_count']}")
    print(f"Success rate: {summary['success_rate']:.2f}")
    print(f"Average steps: {summary['average_steps']:.2f}")
    print(f"Average model calls: {summary['average_model_calls']:.2f}")


if __name__ == "__main__":
    main()
