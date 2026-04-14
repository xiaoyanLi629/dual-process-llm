"""
Factorial Ablation Experiment for Dual-Process Theory in LLMs
(Main experiment for the BIBM 2026 paper)

2x2x2 factorial design isolating:
- Model size (gpt-4o-mini vs gpt-4o)
- Temperature (0.2 vs 0.9)
- Prompting strategy (zero-shot vs chain-of-thought)

Key comparisons:
- C5 vs C8: same-model comparison (primary finding)
  Holds model constant (gpt-4o); isolates temperature + prompting
- C1 vs C8: canonical S1 vs S2 comparison (secondary)
  Varies all three factors simultaneously

Control baselines:
- Random baseline: expected accuracy by chance per task category
- Variance baseline: C8 run twice to establish the noise floor

Usage:
    python -m src.experiments.ablation_factorial --n_samples 50
    python -m src.experiments.ablation_factorial --n_samples 100 --output results/ablation/
    python -m src.experiments.ablation_factorial --n_samples 100 --no_variance_baseline
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
src_root = Path(__file__).parent.parent  # .../src/
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(src_root))  # allows bare 'tasks.task_loader' imports used by src/__init__ files

from api_config import get_openai_client
from tasks.task_loader import TaskLoader
from evaluation.evaluator import DualProcessEvaluator

# ============================================================
# 2x2x2 Factorial Conditions
# ============================================================
CONDITIONS = [
    {"id": "C1", "model": "gpt-4o-mini", "temperature": 0.9, "prompt": "zero_shot", "label": "System 1 (canonical)"},
    {"id": "C2", "model": "gpt-4o-mini", "temperature": 0.9, "prompt": "cot",       "label": "mini + high-T + CoT"},
    {"id": "C3", "model": "gpt-4o-mini", "temperature": 0.2, "prompt": "zero_shot", "label": "mini + low-T + zero-shot"},
    {"id": "C4", "model": "gpt-4o-mini", "temperature": 0.2, "prompt": "cot",       "label": "mini + low-T + CoT"},
    {"id": "C5", "model": "gpt-4o",      "temperature": 0.9, "prompt": "zero_shot", "label": "4o + high-T + zero-shot"},
    {"id": "C6", "model": "gpt-4o",      "temperature": 0.9, "prompt": "cot",       "label": "4o + high-T + CoT"},
    {"id": "C7", "model": "gpt-4o",      "temperature": 0.2, "prompt": "zero_shot", "label": "4o + low-T + zero-shot"},
    {"id": "C8", "model": "gpt-4o",      "temperature": 0.2, "prompt": "cot",       "label": "System 2 (canonical)"},
]

# Map condition IDs to their factor levels for easy lookup
# Factors: model (mini=0, 4o=1), temperature (high=0.9, low=0.2), prompt (zero_shot, cot)
_CONDITION_INDEX: Dict[str, Dict] = {c["id"]: c for c in CONDITIONS}

# ============================================================
# System Prompts
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
# Core run_condition function
# ============================================================
def run_condition(
    condition: Dict[str, Any],
    tasks: List[Any],
    client,
    evaluator: DualProcessEvaluator,
) -> List[Dict[str, Any]]:
    """
    Run a single factorial condition against the given tasks.

    Args:
        condition: Condition dict with id, model, temperature, prompt, label.
        tasks: List of Task objects from TaskLoader.
        client: OpenAI client instance.
        evaluator: DualProcessEvaluator instance.

    Returns:
        List of per-trial result dicts.
    """
    model = condition["model"]
    temperature = condition["temperature"]
    prompt_type = condition["prompt"]
    condition_id = condition["id"]

    system_prompt = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else COT_PROMPT
    max_tokens = 150 if prompt_type == "zero_shot" else 1000

    trial_results: List[Dict[str, Any]] = []

    for task in tasks:
        result = _run_single_trial(
            condition_id=condition_id,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            task=task,
            client=client,
            evaluator=evaluator,
        )
        trial_results.append(result)

    return trial_results


def _run_single_trial(
    condition_id: str,
    model: str,
    temperature: float,
    system_prompt: str,
    max_tokens: int,
    task: Any,
    client,
    evaluator: DualProcessEvaluator,
) -> Dict[str, Any]:
    """
    Execute one API call for a single task under a given condition.

    Returns a flat dict with all recorded fields.
    """
    answer = None
    confidence = 0.5
    tokens_used = 0
    is_correct = False
    parse_error = False

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": task.question},
            ],
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

        try:
            parsed = json.loads(raw_content)
            answer = parsed.get("answer", "")
            confidence = float(parsed.get("confidence", 0.5))
        except (json.JSONDecodeError, ValueError):
            # Treat unparseable output as an incorrect response
            answer = raw_content
            parse_error = True

    except Exception as api_err:
        answer = f"ERROR: {api_err}"
        parse_error = True

    # Evaluate correctness
    if answer is not None and not parse_error:
        eval_result = evaluator.evaluate_response(
            question=task.question,
            response={"answer": answer, "confidence": confidence},
            correct_answer=task.correct_answer,
            task_type=task.task_type,
            options=task.options if task.options else None,
        )
        is_correct = eval_result["is_correct"]
    else:
        is_correct = False

    return {
        "condition_id": condition_id,
        "task_id": task.id,
        "task_category": task.metadata.get("category", task.task_type),
        "answer": answer,
        "correct_answer": task.correct_answer,
        "is_correct": is_correct,
        "confidence": confidence,
        "tokens_used": tokens_used,
        "parse_error": parse_error,
    }


# ============================================================
# Control baselines
# ============================================================
def compute_random_baseline(tasks_by_category: dict) -> dict:
    """
    Compute expected random accuracy for each task category.

    For MCQ tasks: 1/n_options
    For open-ended: 0%
    """
    baselines = {}
    for category, tasks in tasks_by_category.items():
        if not tasks:
            baselines[category] = 0.0
            continue

        total_expected = 0.0
        for task in tasks:
            if hasattr(task, 'options') and task.options:
                total_expected += 1.0 / len(task.options)
            else:
                total_expected += 0.0  # open-ended: 0% random chance

        baselines[category] = total_expected / len(tasks)

    return baselines


def compute_variance_baseline(
    condition: Dict[str, Any],
    tasks: List[Any],
    client,
    evaluator: "DualProcessEvaluator",
    n_subset: int = 50,
) -> Dict[str, Any]:
    """
    Run the same condition (C8) twice on a subset of tasks and measure
    agreement rate and accuracy difference to establish the noise floor.

    Args:
        condition: Condition dict (typically C8).
        tasks: Full task list; first n_subset items are used.
        client: OpenAI client instance.
        evaluator: DualProcessEvaluator instance.
        n_subset: Number of tasks per run (default 50).

    Returns:
        Dict with agreement_rate, accuracy_run1, accuracy_run2,
        accuracy_difference, and per-trial details.
    """
    subset = tasks[:n_subset]

    print(f"\n[Variance Baseline] Running {condition['id']} twice on {len(subset)} tasks …")

    run1 = run_condition(condition, subset, client, evaluator)
    run2 = run_condition(condition, subset, client, evaluator)

    n = len(subset)
    acc1 = sum(t["is_correct"] for t in run1) / n if n else float("nan")
    acc2 = sum(t["is_correct"] for t in run2) / n if n else float("nan")

    # Agreement: both runs give the same correctness outcome for each item
    agreements = [r1["is_correct"] == r2["is_correct"] for r1, r2 in zip(run1, run2)]
    agreement_rate = sum(agreements) / len(agreements) if agreements else float("nan")

    print(f"  Run 1 accuracy: {acc1:.3f}  Run 2 accuracy: {acc2:.3f}")
    print(f"  Agreement rate: {agreement_rate:.3f}  |Δacc|: {abs(acc1 - acc2):.3f}")

    return {
        "condition_id": condition["id"],
        "n_subset": n_subset,
        "accuracy_run1": acc1,
        "accuracy_run2": acc2,
        "accuracy_difference": abs(acc1 - acc2),
        "agreement_rate": agreement_rate,
        "run1_trials": run1,
        "run2_trials": run2,
    }


# ============================================================
# Analysis: main effects and interactions
# ============================================================
def _accuracy_for_conditions(results: Dict[str, Any], condition_ids: List[str]) -> float:
    """Return mean accuracy across the specified condition IDs."""
    scores: List[float] = []
    for cid in condition_ids:
        if cid in results and results[cid]["trials"]:
            cond_correct = sum(t["is_correct"] for t in results[cid]["trials"])
            cond_total = len(results[cid]["trials"])
            scores.append(cond_correct / cond_total)
    return float(np.mean(scores)) if scores else float("nan")


def compute_main_effects(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute main effects by averaging over other factors.

    Main effect of Model       = mean(C5,C6,C7,C8) - mean(C1,C2,C3,C4)
    Main effect of Temperature = mean(low-T conditions) - mean(high-T conditions)
    Main effect of Prompt      = mean(CoT conditions) - mean(zero-shot conditions)
    """
    # Model: mini = C1-C4, 4o = C5-C8
    mini_acc = _accuracy_for_conditions(results, ["C1", "C2", "C3", "C4"])
    gpt4o_acc = _accuracy_for_conditions(results, ["C5", "C6", "C7", "C8"])
    model_effect = gpt4o_acc - mini_acc

    # Temperature: high (0.9) = C1,C2,C5,C6; low (0.2) = C3,C4,C7,C8
    high_t_acc = _accuracy_for_conditions(results, ["C1", "C2", "C5", "C6"])
    low_t_acc  = _accuracy_for_conditions(results, ["C3", "C4", "C7", "C8"])
    temp_effect = low_t_acc - high_t_acc  # positive = lower temp helps

    # Prompt: zero_shot = C1,C3,C5,C7; cot = C2,C4,C6,C8
    zero_shot_acc = _accuracy_for_conditions(results, ["C1", "C3", "C5", "C7"])
    cot_acc       = _accuracy_for_conditions(results, ["C2", "C4", "C6", "C8"])
    prompt_effect = cot_acc - zero_shot_acc  # positive = CoT helps

    return {
        "model": {
            "gpt-4o-mini_accuracy": mini_acc,
            "gpt-4o_accuracy": gpt4o_acc,
            "effect_gpt4o_minus_mini": model_effect,
        },
        "temperature": {
            "high_temp_accuracy": high_t_acc,
            "low_temp_accuracy": low_t_acc,
            "effect_low_minus_high": temp_effect,
        },
        "prompt": {
            "zero_shot_accuracy": zero_shot_acc,
            "cot_accuracy": cot_acc,
            "effect_cot_minus_zero_shot": prompt_effect,
        },
    }


def compute_interactions(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute 2-way interactions.

    Model x Prompt interaction:
        Does the effect of CoT differ between mini and 4o?
        = (CoT_4o - ZS_4o) - (CoT_mini - ZS_mini)

    Model x Temperature interaction:
        Does the effect of temperature differ between models?
        = (LowT_4o - HighT_4o) - (LowT_mini - HighT_mini)

    Temperature x Prompt interaction:
        Does the effect of CoT differ across temperatures?
        = (CoT_lowT - ZS_lowT) - (CoT_highT - ZS_highT)
    """
    # Accuracy for each individual condition
    def acc(cid: str) -> float:
        if cid not in results or not results[cid]["trials"]:
            return float("nan")
        trials = results[cid]["trials"]
        return sum(t["is_correct"] for t in trials) / len(trials)

    # --- Model x Prompt ---
    # mini: ZS=C1, CoT=C2 (high-T); ZS=C3, CoT=C4 (low-T)  -> average over temperature
    cot_effect_mini = (
        (acc("C2") - acc("C1")) + (acc("C4") - acc("C3"))
    ) / 2
    cot_effect_4o = (
        (acc("C6") - acc("C5")) + (acc("C8") - acc("C7"))
    ) / 2
    model_x_prompt = cot_effect_4o - cot_effect_mini

    # --- Model x Temperature ---
    # mini: high-T=C1/C2, low-T=C3/C4; 4o: high-T=C5/C6, low-T=C7/C8
    lowt_effect_mini = (
        (acc("C3") - acc("C1")) + (acc("C4") - acc("C2"))
    ) / 2
    lowt_effect_4o = (
        (acc("C7") - acc("C5")) + (acc("C8") - acc("C6"))
    ) / 2
    model_x_temp = lowt_effect_4o - lowt_effect_mini

    # --- Temperature x Prompt ---
    # high-T: ZS=C1/C5, CoT=C2/C6; low-T: ZS=C3/C7, CoT=C4/C8
    cot_effect_highT = (
        (acc("C2") - acc("C1")) + (acc("C6") - acc("C5"))
    ) / 2
    cot_effect_lowT = (
        (acc("C4") - acc("C3")) + (acc("C8") - acc("C7"))
    ) / 2
    temp_x_prompt = cot_effect_lowT - cot_effect_highT

    return {
        "model_x_prompt": {
            "cot_effect_in_mini": cot_effect_mini,
            "cot_effect_in_4o": cot_effect_4o,
            "interaction": model_x_prompt,
            "interpretation": "Positive = CoT helps more for gpt-4o than gpt-4o-mini",
        },
        "model_x_temperature": {
            "lowT_effect_in_mini": lowt_effect_mini,
            "lowT_effect_in_4o": lowt_effect_4o,
            "interaction": model_x_temp,
            "interpretation": "Positive = low temperature helps more for gpt-4o than gpt-4o-mini",
        },
        "temperature_x_prompt": {
            "cot_effect_at_highT": cot_effect_highT,
            "cot_effect_at_lowT": cot_effect_lowT,
            "interaction": temp_x_prompt,
            "interpretation": "Positive = CoT helps more at low temperature than high temperature",
        },
    }


# ============================================================
# Results aggregation helpers
# ============================================================
def _compute_condition_summary(condition: Dict, trials: List[Dict]) -> Dict:
    """Compute per-condition accuracy broken down by task category."""
    if not trials:
        return {"accuracy_overall": float("nan"), "by_category": {}}

    # Overall
    overall_correct = sum(t["is_correct"] for t in trials)
    overall_acc = overall_correct / len(trials)

    # By category
    categories: Dict[str, List[bool]] = {}
    for t in trials:
        cat = t["task_category"]
        categories.setdefault(cat, []).append(t["is_correct"])

    by_category = {
        cat: {
            "n": len(vals),
            "correct": sum(vals),
            "accuracy": sum(vals) / len(vals),
        }
        for cat, vals in categories.items()
    }

    return {
        "condition_id": condition["id"],
        "label": condition["label"],
        "model": condition["model"],
        "temperature": condition["temperature"],
        "prompt": condition["prompt"],
        "n_trials": len(trials),
        "n_correct": overall_correct,
        "accuracy_overall": overall_acc,
        "avg_confidence": float(np.mean([t["confidence"] for t in trials])),
        "avg_tokens": float(np.mean([t["tokens_used"] for t in trials])),
        "parse_errors": sum(t["parse_error"] for t in trials),
        "by_category": by_category,
    }


def _print_summary_table(results: Dict, main_effects: Dict, interactions: Dict) -> None:
    """Print a readable summary to stdout."""
    print("\n" + "=" * 80)
    print("2x2x2 FACTORIAL ABLATION — RESULTS SUMMARY")
    print("=" * 80)

    header = f"{'ID':<4} {'Label':<30} {'Model':<14} {'Temp':>5} {'Prompt':<12} {'Acc':>7} {'N':>5}"
    print(header)
    print("-" * 80)
    for cid, cdata in results.items():
        summary = cdata["summary"]
        acc_str = f"{summary['accuracy_overall']:.3f}" if not np.isnan(summary['accuracy_overall']) else "  N/A"
        print(
            f"{cid:<4} {summary['label']:<30} {summary['model']:<14} "
            f"{summary['temperature']:>5.1f} {summary['prompt']:<12} {acc_str:>7} {summary['n_trials']:>5}"
        )

    # ---- Key comparisons ----
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    def _acc(cid: str) -> float:
        if cid in results and results[cid]["trials"]:
            trials = results[cid]["trials"]
            return sum(t["is_correct"] for t in trials) / len(trials)
        return float("nan")

    c5_acc = _acc("C5")
    c8_acc = _acc("C8")
    c1_acc = _acc("C1")

    if not np.isnan(c5_acc) and not np.isnan(c8_acc):
        print(f"  [PRIMARY]   C5 vs C8 (same-model, gpt-4o): "
              f"C5={c5_acc:.3f}  C8={c8_acc:.3f}  delta={c8_acc - c5_acc:+.3f}")
        print(f"              Interpretation: effect of low-T + CoT within gpt-4o")
    if not np.isnan(c1_acc) and not np.isnan(c8_acc):
        print(f"  [SECONDARY] C1 vs C8 (canonical S1 vs S2): "
              f"C1={c1_acc:.3f}  C8={c8_acc:.3f}  delta={c8_acc - c1_acc:+.3f}")
        print(f"              Interpretation: full model+temperature+prompt contrast")

    print("\n" + "=" * 80)
    print("MAIN EFFECTS")
    print("=" * 80)
    me = main_effects
    print(f"  Model:       gpt-4o-mini={me['model']['gpt-4o-mini_accuracy']:.3f}  "
          f"gpt-4o={me['model']['gpt-4o_accuracy']:.3f}  "
          f"effect={me['model']['effect_gpt4o_minus_mini']:+.3f}")
    print(f"  Temperature: high={me['temperature']['high_temp_accuracy']:.3f}  "
          f"low={me['temperature']['low_temp_accuracy']:.3f}  "
          f"effect(low-high)={me['temperature']['effect_low_minus_high']:+.3f}")
    print(f"  Prompt:      zero-shot={me['prompt']['zero_shot_accuracy']:.3f}  "
          f"CoT={me['prompt']['cot_accuracy']:.3f}  "
          f"effect(CoT-ZS)={me['prompt']['effect_cot_minus_zero_shot']:+.3f}")

    print("\n" + "=" * 80)
    print("2-WAY INTERACTIONS")
    print("=" * 80)
    ix = interactions
    print(f"  Model x Prompt:       interaction={ix['model_x_prompt']['interaction']:+.3f}  "
          f"({ix['model_x_prompt']['interpretation']})")
    print(f"  Model x Temperature:  interaction={ix['model_x_temperature']['interaction']:+.3f}  "
          f"({ix['model_x_temperature']['interpretation']})")
    print(f"  Temp x Prompt:        interaction={ix['temperature_x_prompt']['interaction']:+.3f}  "
          f"({ix['temperature_x_prompt']['interpretation']})")
    print("=" * 80 + "\n")


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(
    n_samples: int = 50,
    output_dir: str = "results/ablation/",
    dry_run: bool = False,
    condition_ids: List[str] = None,
    with_variance_baseline: bool = True,
) -> Dict[str, Any]:
    """
    Run the full 2x2x2 factorial ablation experiment (main experiment).

    Args:
        n_samples: Number of tasks per category per condition.
        output_dir: Directory to save result JSON files.
        dry_run: If True, run only 1 task per condition for verification.
        condition_ids: List of condition IDs to run (default: all).
        with_variance_baseline: If True, run C8 a second time on a 50-item
            subset to establish the noise floor (default: True).

    Returns:
        Full results dict.
    """
    if dry_run:
        n_samples = 1
        print("[DRY RUN] Running 1 sample per condition to verify pipeline.")

    # Filter conditions
    active_conditions = CONDITIONS
    if condition_ids:
        active_conditions = [c for c in CONDITIONS if c["id"] in condition_ids]
        print(f"Running conditions: {[c['id'] for c in active_conditions]}")

    # Setup
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    client = get_openai_client()
    evaluator = DualProcessEvaluator()

    # Load tasks
    loader = TaskLoader()
    loader.load()

    s1_tasks = loader.get_system1_tasks(n=n_samples, shuffle=True)
    s2_tasks = loader.get_system2_tasks(n=n_samples, shuffle=True)
    conflict_tasks = loader.get_conflict_tasks(n=n_samples, shuffle=True)
    all_tasks = s1_tasks + s2_tasks + conflict_tasks

    tasks_by_category = {
        "system1": s1_tasks,
        "system2": s2_tasks,
        "conflict": conflict_tasks,
    }

    print(f"\nTasks per condition: {len(all_tasks)} "
          f"({len(s1_tasks)} intuitive, {len(s2_tasks)} analytical, {len(conflict_tasks)} conflict)")

    # Compute random baseline (no API calls)
    random_baseline = compute_random_baseline(tasks_by_category)
    print(f"\nRandom baselines: " + "  ".join(f"{k}={v:.3f}" for k, v in random_baseline.items()))

    # Run each condition
    results: Dict[str, Any] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for condition in active_conditions:
        cid = condition["id"]
        print(f"\n--- Condition {cid}: {condition['label']} ---")

        trials: List[Dict] = []
        for task in tqdm(all_tasks, desc=cid, unit="task"):
            trial = _run_single_trial(
                condition_id=cid,
                model=condition["model"],
                temperature=condition["temperature"],
                system_prompt=ZERO_SHOT_PROMPT if condition["prompt"] == "zero_shot" else COT_PROMPT,
                max_tokens=150 if condition["prompt"] == "zero_shot" else 1000,
                task=task,
                client=client,
                evaluator=evaluator,
            )
            trials.append(trial)
            # Respect rate limits
            time.sleep(0.3)

        summary = _compute_condition_summary(condition, trials)
        results[cid] = {
            "condition": condition,
            "summary": summary,
            "trials": trials,
        }
        print(f"  Accuracy: {summary['accuracy_overall']:.3f}  "
              f"(n={summary['n_trials']}, parse_errors={summary['parse_errors']})")

    # Compute main effects and interactions
    main_effects = compute_main_effects(results)
    interactions = compute_interactions(results)

    # Print summary table (includes key comparisons)
    _print_summary_table(results, main_effects, interactions)

    # Variance baseline: run C8 again on a subset to measure noise floor
    variance_baseline: Optional[Dict[str, Any]] = None
    if with_variance_baseline and not dry_run:
        # Only run if C8 was included in this run
        c8_condition = _CONDITION_INDEX.get("C8")
        if c8_condition and ("C8" in results or condition_ids is None):
            variance_baseline = compute_variance_baseline(
                condition=c8_condition,
                tasks=all_tasks,
                client=client,
                evaluator=evaluator,
                n_subset=min(50, len(all_tasks)),
            )
        else:
            print("\n[Variance Baseline] Skipped — C8 was not in the active conditions.")
    elif dry_run:
        print("\n[Variance Baseline] Skipped in dry-run mode.")

    # Assemble full output
    full_output = {
        "experiment": "2x2x2_factorial_ablation",
        "timestamp": timestamp,
        "config": {
            "n_samples_per_category": n_samples,
            "dry_run": dry_run,
            "conditions_run": [c["id"] for c in active_conditions],
            "with_variance_baseline": with_variance_baseline,
        },
        "conditions": results,
        "main_effects": main_effects,
        "interactions": interactions,
        "random_baseline": random_baseline,
        "variance_baseline": variance_baseline,
        "key_comparisons": {
            "primary": {
                "label": "C5 vs C8 (same-model: gpt-4o, isolates temperature + prompting)",
                "C5": _accuracy_for_conditions(results, ["C5"]),
                "C8": _accuracy_for_conditions(results, ["C8"]),
                "delta_C8_minus_C5": (
                    _accuracy_for_conditions(results, ["C8"])
                    - _accuracy_for_conditions(results, ["C5"])
                ),
            },
            "secondary": {
                "label": "C1 vs C8 (canonical S1 vs S2, all three factors vary)",
                "C1": _accuracy_for_conditions(results, ["C1"]),
                "C8": _accuracy_for_conditions(results, ["C8"]),
                "delta_C8_minus_C1": (
                    _accuracy_for_conditions(results, ["C8"])
                    - _accuracy_for_conditions(results, ["C1"])
                ),
            },
        },
    }

    # Save JSON
    output_file = output_path / f"ablation_factorial_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {output_file}")

    return full_output


# ============================================================
# CLI Entry Point
# ============================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2x2x2 Factorial Ablation Experiment for Dual-Process Theory in LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples per task category per condition (default: 50).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation/",
        help="Output directory for result JSON files (default: results/ablation/).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 1 sample per condition to verify the pipeline (does not save full results).",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated condition IDs to run, e.g. --conditions C1,C8 (default: all).",
    )
    parser.add_argument(
        "--no_variance_baseline",
        action="store_true",
        help="Disable the same-config variance baseline (C8 run twice). Enabled by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    condition_ids: List[str] = None
    if args.conditions:
        condition_ids = [c.strip() for c in args.conditions.split(",")]
        # Validate
        valid_ids = {c["id"] for c in CONDITIONS}
        invalid = [cid for cid in condition_ids if cid not in valid_ids]
        if invalid:
            print(f"ERROR: Unknown condition IDs: {invalid}. Valid IDs: {sorted(valid_ids)}")
            sys.exit(1)

    run_experiment(
        n_samples=args.n_samples,
        output_dir=args.output,
        dry_run=args.dry_run,
        condition_ids=condition_ids,
        with_variance_baseline=not args.no_variance_baseline,
    )
