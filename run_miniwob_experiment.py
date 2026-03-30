from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import run_vwa_experiment as base
from env.miniwob_env import MiniWoBEnv
from glm_client import GLMClient
from scripts.check_miniwob_env import load_local_env_file


DEFAULT_MODEL = "glm-4.6v"
DEFAULT_MAX_STEPS = 12
DEFAULT_MAX_REPAIRS = 4
DEFAULT_MAX_MODEL_CALLS = 4
DEFAULT_RESET_TIMEOUT_SEC = 0
DEFAULT_STEP_TIMEOUT_SEC = 0
DEFAULT_RESET_RETRIES = 1
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_PATH = REPO_ROOT / "tasks" / "miniwob_smoke_5.json"

OUTPUT_ROOT = REPO_ROOT / "outputs" / "miniwob_experiments"
SCREENSHOT_ROOT = OUTPUT_ROOT / "screenshots"
TRACE_ROOT = OUTPUT_ROOT / "traces"
RESULT_ROOT = OUTPUT_ROOT / "results"
SKILL_ROOT = OUTPUT_ROOT / "skills"

ACTION_SYSTEM_PROMPT = """You are a web agent acting inside BrowserGym MiniWoB++.
Given the task goal, the current page state, and the current page screenshot, output exactly one next action.

Allowed formats:
CLICK[text]
TYPE[field=value]
SCROLL[down]
STOP[answer]

Rules:
1. Output one action only.
2. CLICK[text] should use exact visible English text from the provided clickable targets whenever possible.
3. TYPE[field=value] should use the exact input label from the provided input field list whenever possible.
4. Use STOP[answer] when you believe the task is complete or the task requires a final textual answer.
5. Do not output bids, Python code, explanations, or markdown.
"""


def adapt_prompt(prompt: str) -> str:
    return prompt.replace("BrowserGym VisualWebArena", "BrowserGym MiniWoB++").replace(
        "VisualWebArena", "MiniWoB++"
    )


def patch_base_module() -> None:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.SCREENSHOT_ROOT = SCREENSHOT_ROOT
    base.TRACE_ROOT = TRACE_ROOT
    base.RESULT_ROOT = RESULT_ROOT
    base.SKILL_ROOT = SKILL_ROOT
    base.ACTION_SYSTEM_PROMPT = ACTION_SYSTEM_PROMPT

    original_generation = base.build_skill_generation_prompt
    original_rewrite = base.build_text_rewrite_prompt
    original_repair = base.build_contract_repair_prompt

    def build_skill_generation_prompt(*args, **kwargs):
        return adapt_prompt(original_generation(*args, **kwargs))

    def build_text_rewrite_prompt(*args, **kwargs):
        return adapt_prompt(original_rewrite(*args, **kwargs))

    def build_contract_repair_prompt(*args, **kwargs):
        return adapt_prompt(original_repair(*args, **kwargs))

    base.build_skill_generation_prompt = build_skill_generation_prompt
    base.build_text_rewrite_prompt = build_text_rewrite_prompt
    base.build_contract_repair_prompt = build_contract_repair_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ContractSkill baselines on BrowserGym MiniWoB++.")
    parser.add_argument(
        "--baseline",
        choices=(
            "no_skill",
            "no_skill_budget_matched",
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
    parser.add_argument("--reset-timeout-sec", type=int, default=DEFAULT_RESET_TIMEOUT_SEC)
    parser.add_argument("--step-timeout-sec", type=int, default=DEFAULT_STEP_TIMEOUT_SEC)
    parser.add_argument("--reset-retries", type=int, default=DEFAULT_RESET_RETRIES)
    parser.add_argument("--headless", type=base.parse_bool, default=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--initial-skill-cache-key",
        type=str,
        default=None,
        help="Optional cache namespace for shared initial skills. Use the same value across split shards to keep the same initial-skill pool.",
    )
    parser.add_argument(
        "--initial-skill-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping task_id to an existing skill artifact path for hard replay.",
    )
    return parser.parse_args()


def main() -> None:
    patch_base_module()
    load_local_env_file()
    args = parse_args()
    configured_api_env = base.configure_api_env_for_model(args.model)
    split_path = args.split_path
    if not split_path.is_absolute():
        split_path = (REPO_ROOT / split_path).resolve()
    split = base.load_split(split_path)
    initial_skill_map = base.load_initial_skill_override_map(args.initial_skill_map)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for root in (SCREENSHOT_ROOT, TRACE_ROOT, RESULT_ROOT, SKILL_ROOT):
        base.ensure_dir(root)

    if configured_api_env is not None:
        print(f"Using API env file: {configured_api_env}")

    glm = GLMClient(model=args.model)
    env = MiniWoBEnv(
        output_root=SCREENSHOT_ROOT / args.baseline / run_id,
        headless=args.headless,
    )
    env.start()

    results: list[dict] = []

    try:
        for task_item in split:
            print(f"\n[{args.baseline}] {task_item['task_id']} {task_item['env_name']}")
            try:
                if args.baseline in {"no_skill", "no_skill_budget_matched"}:
                    result, trace = base.run_noskill_task(
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        run_id=run_id,
                        max_steps=args.max_steps,
                        reset_timeout_sec=args.reset_timeout_sec,
                        step_timeout_sec=args.step_timeout_sec,
                        reset_retries=args.reset_retries,
                        baseline_label=args.baseline,
                        max_model_calls=(args.max_model_calls if args.baseline == "no_skill_budget_matched" else None),
                        disable_answer_extractor=(args.baseline == "no_skill_budget_matched"),
                    )
                else:
                    result, trace = base.run_skill_baseline_task(
                        glm=glm,
                        env=env,
                        task_item=task_item,
                        run_id=run_id,
                        split_path=split_path,
                        initial_skill_cache_key=args.initial_skill_cache_key,
                        initial_skill_map=initial_skill_map,
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
                task_metadata = base.get_task_metadata(task_item)
                result = {
                    "task_id": task_item["task_id"],
                    "env_name": task_item["env_name"],
                    "baseline": args.baseline,
                    **task_metadata,
                    "success": False,
                    "initial_skill_success": None if args.baseline in {"no_skill", "no_skill_budget_matched"} else False,
                    "post_repair_success": False if args.baseline not in {"no_skill", "no_skill_budget_matched"} else None,
                    "steps_taken": 0,
                    "model_calls": 0,
                    "repair_count": 0,
                    "final_fail_reason": fail_reason,
                    "final_action_error": "",
                    "initial_skill_path": None,
                    "final_skill_path": None,
                    "repair_skill_paths": [],
                    "patch_types": [],
                    **failure_info,
                }
                trace = {
                    "run_id": run_id,
                    "baseline": args.baseline,
                    "task_id": task_item["task_id"],
                    "env_name": task_item["env_name"],
                    "notes": task_item.get("notes", ""),
                    "model": glm.model,
                    **task_metadata,
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
