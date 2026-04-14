"""
Multi-Model Validation Experiment

Tests dual-process configurations across multiple LLM families to
demonstrate that findings generalize beyond a single model provider.

Usage:
    python -m src.experiments.multi_model --n_samples 50
    python -m src.experiments.multi_model --families openai,deepseek --n_samples 100
"""

import argparse
import json
import os
import sys
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from api_config import (
    get_openai_client, get_deepseek_client,
    get_together_client, get_dashscope_client,
    validate_api_keys,
)
from tasks.task_loader import TaskLoader
from evaluation.evaluator import DualProcessEvaluator

# ============================================================
# Model family definitions
# ============================================================

# When USE_OPENROUTER=1, all families use get_openai_client() (pointed at OpenRouter)
# with provider-prefixed model names. Otherwise, each family uses its own provider.
_USE_OPENROUTER = os.environ.get("USE_OPENROUTER", "0") == "1"

if _USE_OPENROUTER:
    MODEL_FAMILIES = {
        "openai": {
            "system1": {"model": "openai/gpt-4o-mini", "provider_func": "get_openai_client"},
            "system2": {"model": "openai/gpt-4o", "provider_func": "get_openai_client"},
        },
        "deepseek": {
            "system1": {"model": "deepseek/deepseek-chat", "provider_func": "get_openai_client"},
            "system2": {
                "model": "deepseek/deepseek-r1",
                "provider_func": "get_openai_client",
                "no_temperature": True,
            },
        },
        "qwen": {
            "system1": {"model": "qwen/qwen-2.5-7b-instruct", "provider_func": "get_openai_client"},
            "system2": {"model": "qwen/qwen-2.5-72b-instruct", "provider_func": "get_openai_client"},
        },
        "llama": {
            "system1": {"model": "meta-llama/llama-3.1-8b-instruct", "provider_func": "get_openai_client"},
            "system2": {"model": "meta-llama/llama-3.3-70b-instruct", "provider_func": "get_openai_client"},
        },
    }
else:
    MODEL_FAMILIES = {
        "openai": {
            "system1": {"model": "gpt-4o-mini", "provider_func": "get_openai_client"},
            "system2": {"model": "gpt-4o", "provider_func": "get_openai_client"},
        },
        "deepseek": {
            "system1": {"model": "deepseek-chat", "provider_func": "get_deepseek_client"},
            "system2": {
                "model": "deepseek-reasoner",
                "provider_func": "get_deepseek_client",
                "no_temperature": True,
            },
        },
        "qwen": {
            "system1": {"model": "qwen-turbo", "provider_func": "get_dashscope_client"},
            "system2": {"model": "qwen-max", "provider_func": "get_dashscope_client"},
        },
        "llama": {
            "system1": {
                "model": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                "provider_func": "get_together_client",
            },
            "system2": {
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "provider_func": "get_together_client",
            },
        },
    }

# Map provider names to their keys in validate_api_keys() output
_PROVIDER_KEY_MAP: Dict[str, str] = {
    "get_openai_client": "openai",
    "get_deepseek_client": "deepseek",
    "get_together_client": "together",
    "get_dashscope_client": "dashscope",
}

# ============================================================
# System prompts (same as ablation_factorial.py for consistency)
# ============================================================

ZERO_SHOT_PROMPT = """You are a fast, intuitive thinker. Answer immediately with your first instinct.
Give a brief, direct answer. Trust your gut feeling.

Respond in JSON format: {"answer": "your answer", "confidence": 0.0-1.0}"""

COT_PROMPT = """You are a careful, analytical thinker. Think step by step.
Break down the problem systematically before reaching a conclusion.

Respond in JSON format:
{
    "reasoning_steps": ["step 1", "step 2", ...],
    "answer": "your final answer",
    "confidence": 0.0-1.0
}"""

# ============================================================
# Provider / client helpers
# ============================================================

_PROVIDER_FUNC_MAP = {
    "get_openai_client": get_openai_client,
    "get_deepseek_client": get_deepseek_client,
    "get_together_client": get_together_client,
    "get_dashscope_client": get_dashscope_client,
}


def get_client(provider_func_name: str):
    """
    Look up and call a provider factory by its string name.

    Args:
        provider_func_name: One of the keys in _PROVIDER_FUNC_MAP.

    Returns:
        An OpenAI-compatible client.

    Raises:
        KeyError: If the provider name is not recognised.
    """
    func = _PROVIDER_FUNC_MAP[provider_func_name]
    return func()


# ============================================================
# Core trial runner
# ============================================================

def run_trial(
    client,
    model: str,
    temperature: Optional[float],
    prompt_mode: str,
    question: str,
    max_tokens: int,
    no_temperature: bool = False,
) -> Dict[str, Any]:
    """
    Run a single API call and return structured result.

    Args:
        client: OpenAI-compatible client.
        model: Model name.
        temperature: Sampling temperature (ignored when no_temperature=True).
        prompt_mode: "zero_shot" or "cot".
        question: The user question text.
        max_tokens: Maximum tokens in the completion.
        no_temperature: If True, omit temperature from the API call.

    Returns:
        Dict with keys: answer, confidence, tokens_used, parse_error, raw_content.
    """
    system_prompt = ZERO_SHOT_PROMPT if prompt_mode == "zero_shot" else COT_PROMPT

    def _call_api():
        call_kwargs: Dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        if not no_temperature:
            call_kwargs["temperature"] = temperature

        # First try with json_object response format
        try:
            call_kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**call_kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            # If the model doesn't support response_format, retry without it
            if "response_format" in err_msg or "json" in err_msg or "unsupported" in err_msg:
                call_kwargs.pop("response_format", None)
                return client.chat.completions.create(**call_kwargs)
            raise

    # Try once; on failure, wait 2 s and retry once more
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            response = _call_api()
            break
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2)
    else:
        # Both attempts failed
        return {
            "answer": f"ERROR: {last_error}",
            "confidence": 0.5,
            "tokens_used": 0,
            "parse_error": True,
            "raw_content": "",
        }

    raw_content: str = response.choices[0].message.content or ""
    tokens_used: int = response.usage.total_tokens if response.usage else 0

    # For deepseek-reasoner, the model wraps its thinking in <think>...</think>
    # Extract the answer that follows those tags.
    answer_content = raw_content
    if model == "deepseek-reasoner" and "</think>" in raw_content:
        parts = raw_content.split("</think>", 1)
        answer_content = parts[1].strip()

    # Attempt JSON parse
    answer: Any = None
    confidence: float = 0.5
    parse_error: bool = False

    # Try to find a JSON object even if the model returned surrounding text
    json_str = _extract_json(answer_content)
    if json_str:
        try:
            parsed = json.loads(json_str)
            answer = parsed.get("answer", "")
            confidence = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            answer = answer_content
            parse_error = True
    else:
        answer = answer_content
        parse_error = True

    return {
        "answer": answer,
        "confidence": confidence,
        "tokens_used": tokens_used,
        "parse_error": parse_error,
        "raw_content": raw_content,
    }


def _extract_json(text: str) -> Optional[str]:
    """
    Try to extract the first JSON object from a string.

    Returns the JSON substring, or None if none found.
    """
    text = text.strip()
    # If it already starts with { try direct parse
    if text.startswith("{"):
        return text
    # Look for the first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


# ============================================================
# Family runner
# ============================================================

def run_family(
    family_name: str,
    family_config: Dict[str, Any],
    tasks: Dict[str, List],
    n_samples: int,
    evaluator: DualProcessEvaluator,
) -> Dict[str, Any]:
    """
    Run System 1 and System 2 configurations for one model family.

    Args:
        family_name: E.g. "openai".
        family_config: Dict with "system1" and "system2" sub-dicts from MODEL_FAMILIES.
        tasks: Dict with keys "intuitive", "analytical", "conflict", each a list of Task objs.
        n_samples: Number of samples per category.
        evaluator: Shared DualProcessEvaluator instance.

    Returns:
        Dict summarising results for both systems.
    """
    family_results: Dict[str, Any] = {}

    for system_key in ("system1", "system2"):
        cfg = family_config[system_key]
        model: str = cfg["model"]
        provider_func: str = cfg["provider_func"]
        no_temperature: bool = cfg.get("no_temperature", False)

        temperature = 0.9 if system_key == "system1" else 0.2
        prompt_mode = "zero_shot" if system_key == "system1" else "cot"
        max_tokens = 150 if system_key == "system1" else 1000

        print(f"\n  [{family_name}] {system_key}: {model}")

        # Obtain client (may raise if API key is missing / invalid)
        client = get_client(provider_func)

        results_by_category: Dict[str, Dict[str, Any]] = {}

        for cat_name, cat_tasks in tasks.items():
            correct_count = 0
            total_count = 0
            confidence_sum = 0.0
            tokens_sum = 0
            parse_errors = 0
            trial_records: List[Dict] = []

            for task in tqdm(
                cat_tasks,
                desc=f"    {system_key}/{cat_name}",
                unit="task",
                leave=False,
            ):
                trial = run_trial(
                    client=client,
                    model=model,
                    temperature=temperature,
                    prompt_mode=prompt_mode,
                    question=task.question,
                    max_tokens=max_tokens,
                    no_temperature=no_temperature,
                )

                # Evaluate
                is_correct = False
                if not trial["parse_error"]:
                    eval_result = evaluator.evaluate_response(
                        question=task.question,
                        response={"answer": trial["answer"], "confidence": trial["confidence"]},
                        correct_answer=task.correct_answer,
                        task_type=task.task_type,
                        options=task.options if task.options else None,
                    )
                    is_correct = eval_result["is_correct"]

                trial_records.append({
                    "task_id": task.id,
                    "answer": trial["answer"],
                    "correct_answer": task.correct_answer,
                    "is_correct": is_correct,
                    "confidence": trial["confidence"],
                    "tokens_used": trial["tokens_used"],
                    "parse_error": trial["parse_error"],
                })

                correct_count += int(is_correct)
                total_count += 1
                confidence_sum += trial["confidence"]
                tokens_sum += trial["tokens_used"]
                parse_errors += int(trial["parse_error"])

                # Polite rate-limit pause
                time.sleep(0.3)

            accuracy = correct_count / total_count if total_count > 0 else float("nan")
            results_by_category[cat_name] = {
                "n": total_count,
                "correct": correct_count,
                "accuracy": accuracy,
                "avg_confidence": confidence_sum / total_count if total_count > 0 else float("nan"),
                "avg_tokens": tokens_sum / total_count if total_count > 0 else float("nan"),
                "parse_errors": parse_errors,
                "trials": trial_records,
            }
            print(
                f"    {cat_name}: accuracy={accuracy:.3f}  "
                f"(correct={correct_count}/{total_count}, parse_errors={parse_errors})"
            )

        family_results[system_key] = {
            "model": model,
            "temperature": None if no_temperature else temperature,
            "prompt_mode": prompt_mode,
            "results_by_category": results_by_category,
        }

    # Compute S2 - S1 gap per category
    s2_s1_gap: Dict[str, float] = {}
    for cat_name in tasks:
        s1_acc = family_results["system1"]["results_by_category"][cat_name]["accuracy"]
        s2_acc = family_results["system2"]["results_by_category"][cat_name]["accuracy"]
        if not (np.isnan(s1_acc) or np.isnan(s2_acc)):
            s2_s1_gap[cat_name] = round(s2_acc - s1_acc, 4)
        else:
            s2_s1_gap[cat_name] = float("nan")

    family_results["s2_s1_gap"] = s2_s1_gap
    return family_results


# ============================================================
# Cross-model consistency analysis
# ============================================================

def compute_cross_model_consistency(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each family, compute the S2-S1 accuracy gap per task category.
    Report whether the expected pattern (larger gap on analytical tasks,
    smaller gap on intuitive tasks) holds consistently across families.

    Args:
        all_results: The "families" sub-dict of the experiment output.

    Returns:
        Dict with pattern_holds flag and gap range stats.
    """
    analytical_gaps: List[float] = []
    intuitive_gaps: List[float] = []
    conflict_gaps: List[float] = []

    for family_name, family_data in all_results.items():
        gap = family_data.get("s2_s1_gap", {})
        a = gap.get("analytical")
        i = gap.get("intuitive")
        c = gap.get("conflict")

        if a is not None and not np.isnan(a):
            analytical_gaps.append(a)
        if i is not None and not np.isnan(i):
            intuitive_gaps.append(i)
        if c is not None and not np.isnan(c):
            conflict_gaps.append(c)

    def _safe_range(vals: List[float]) -> Optional[List[float]]:
        if not vals:
            return None
        return [round(min(vals), 4), round(max(vals), 4)]

    # The pattern holds if analytical gap > intuitive gap in the majority of families
    pattern_count = 0
    valid_families = 0
    for family_name, family_data in all_results.items():
        gap = family_data.get("s2_s1_gap", {})
        a = gap.get("analytical")
        i = gap.get("intuitive")
        if a is not None and i is not None and not np.isnan(a) and not np.isnan(i):
            valid_families += 1
            if a > i:
                pattern_count += 1

    pattern_holds = (pattern_count / valid_families >= 0.5) if valid_families > 0 else False

    return {
        "pattern_holds": pattern_holds,
        "families_showing_pattern": pattern_count,
        "total_valid_families": valid_families,
        "analytical_gap_range": _safe_range(analytical_gaps),
        "intuitive_gap_range": _safe_range(intuitive_gaps),
        "conflict_gap_range": _safe_range(conflict_gaps),
        "mean_analytical_gap": round(float(np.mean(analytical_gaps)), 4) if analytical_gaps else float("nan"),
        "mean_intuitive_gap": round(float(np.mean(intuitive_gaps)), 4) if intuitive_gaps else float("nan"),
    }


# ============================================================
# Summary table printer
# ============================================================

def _print_summary_table(family_results: Dict[str, Any], consistency: Dict[str, Any]) -> None:
    """Print a Unicode box-drawing table of S2-S1 gaps per family."""
    categories = ["intuitive", "analytical", "conflict"]

    col_width = 22
    fam_width = 14

    top    = "╔" + "═" * fam_width + "╦" + ("═" * col_width + "╦") * (len(categories) - 1) + "═" * col_width + "╗"
    sep    = "╠" + "═" * fam_width + "╬" + ("═" * col_width + "╬") * (len(categories) - 1) + "═" * col_width + "╣"
    bottom = "╚" + "═" * fam_width + "╩" + ("═" * col_width + "╩") * (len(categories) - 1) + "═" * col_width + "╝"

    def header_cell(text: str, width: int) -> str:
        return text.center(width)

    def data_cell(val: Optional[float], width: int) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A".center(width)
        sign = "+" if val >= 0 else ""
        return f"{sign}{val * 100:.1f}%".center(width)

    print("\n" + top)
    # Header row
    header_row = (
        "║" + header_cell("Family", fam_width)
        + "║" + "║".join(header_cell(f"{c.capitalize()} (S2-S1)", col_width) for c in categories)
        + "║"
    )
    print(header_row)
    print(sep)

    family_display_names = {
        "openai": "OpenAI",
        "deepseek": "DeepSeek",
        "qwen": "Qwen",
        "llama": "Llama",
    }

    for fam_key, fam_data in family_results.items():
        gap = fam_data.get("s2_s1_gap", {})
        display_name = family_display_names.get(fam_key, fam_key.capitalize())
        cells = "║".join(
            data_cell(gap.get(cat), col_width) for cat in categories
        )
        print("║" + display_name.center(fam_width) + "║" + cells + "║")

    print(bottom)

    # Consistency summary
    print(f"\nCross-model consistency:")
    print(f"  Pattern holds (analytical gap > intuitive gap): {consistency['pattern_holds']}")
    print(f"  Families showing pattern: {consistency['families_showing_pattern']}/{consistency['total_valid_families']}")
    if consistency["analytical_gap_range"]:
        lo, hi = consistency["analytical_gap_range"]
        print(f"  Analytical gap range:  [{lo*100:+.1f}%, {hi*100:+.1f}%]")
    if consistency["intuitive_gap_range"]:
        lo, hi = consistency["intuitive_gap_range"]
        print(f"  Intuitive gap range:   [{lo*100:+.1f}%, {hi*100:+.1f}%]")
    print()


# ============================================================
# Main orchestrator
# ============================================================

def run_experiment(
    families: List[str],
    n_samples: int = 50,
    output_dir: str = "results/multi_model/",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate the multi-model validation experiment.

    Args:
        families: List of family names to run (subset of MODEL_FAMILIES keys).
        n_samples: Number of tasks per category.
        output_dir: Directory to save JSON results.
        dry_run: If True, use only 1 sample per category.

    Returns:
        Full results dict.
    """
    if dry_run:
        n_samples = 1
        print("[DRY RUN] Running 1 sample per category to verify the pipeline.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Validate requested families
    unknown = [f for f in families if f not in MODEL_FAMILIES]
    if unknown:
        print(f"WARNING: Unknown families ignored: {unknown}")
        families = [f for f in families if f in MODEL_FAMILIES]

    if not families:
        print("ERROR: No valid families specified. Exiting.")
        return {}

    # Check API key availability and skip families whose keys are absent
    api_status = validate_api_keys()
    active_families: List[str] = []
    for fam in families:
        cfg = MODEL_FAMILIES[fam]
        # Gather all provider functions used by this family
        providers_needed = {
            _PROVIDER_KEY_MAP[cfg["system1"]["provider_func"]],
            _PROVIDER_KEY_MAP[cfg["system2"]["provider_func"]],
        }
        missing = [p for p in providers_needed if not api_status.get(p, False)]
        if missing:
            print(f"WARNING: Skipping family '{fam}' — missing API key(s) for: {missing}")
        else:
            active_families.append(fam)

    if not active_families:
        print("ERROR: No families have valid API keys configured. Exiting.")
        return {}

    print(f"\nRunning experiment for families: {active_families}")

    # Setup output directory
    out_path = project_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Load tasks once (shared across all families)
    loader = TaskLoader()
    loaded = loader.load()
    if not loaded:
        print("WARNING: Dataset could not be loaded. Results may be empty.")

    s1_tasks = loader.get_system1_tasks(n=n_samples, shuffle=True)
    s2_tasks = loader.get_system2_tasks(n=n_samples, shuffle=True)
    conflict_tasks = loader.get_conflict_tasks(n=n_samples, shuffle=True)

    tasks = {
        "intuitive": s1_tasks,
        "analytical": s2_tasks,
        "conflict": conflict_tasks,
    }

    print(
        f"Tasks per system: {sum(len(v) for v in tasks.values())} "
        f"({len(s1_tasks)} intuitive, {len(s2_tasks)} analytical, {len(conflict_tasks)} conflict)"
    )

    evaluator = DualProcessEvaluator()
    all_family_results: Dict[str, Any] = {}

    for fam_name in active_families:
        print(f"\n{'='*60}")
        print(f"Family: {fam_name.upper()}")
        print(f"{'='*60}")
        try:
            fam_result = run_family(
                family_name=fam_name,
                family_config=MODEL_FAMILIES[fam_name],
                tasks=tasks,
                n_samples=n_samples,
                evaluator=evaluator,
            )
            all_family_results[fam_name] = fam_result
        except Exception as exc:
            print(f"ERROR: Family '{fam_name}' failed with: {exc}")
            all_family_results[fam_name] = {"error": str(exc)}

    # Cross-model consistency
    # Only pass families that did not error out
    valid_results = {k: v for k, v in all_family_results.items() if "error" not in v}
    consistency = compute_cross_model_consistency(valid_results)

    # Print summary table
    _print_summary_table(valid_results, consistency)

    # Assemble final output
    full_output: Dict[str, Any] = {
        "metadata": {
            "timestamp": timestamp,
            "n_samples": n_samples,
            "dry_run": dry_run,
            "families_requested": families,
            "families_run": active_families,
        },
        "families": all_family_results,
        "cross_model_consistency": consistency,
    }

    # Save JSON (strip non-serialisable numpy values)
    output_file = out_path / f"multi_model_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {output_file}")

    return full_output


# ============================================================
# CLI entry point
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Model Validation Experiment for Dual-Process Theory in LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of tasks per category (default: 50).",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="openai,deepseek,qwen,llama",
        help="Comma-separated list of model families to test (default: openai,deepseek,qwen,llama).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/multi_model/",
        help="Output directory for result JSON files (default: results/multi_model/).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 1 sample per category to verify the pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    requested_families = [f.strip() for f in args.families.split(",") if f.strip()]

    run_experiment(
        families=requested_families,
        n_samples=args.n_samples,
        output_dir=args.output,
        dry_run=args.dry_run,
    )
