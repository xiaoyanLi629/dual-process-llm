"""
Comprehensive Statistical Analysis for Dual-Process Experiments

Computes complete statistical reports with no selective omission.
All p-values are reported, including non-significant ones.

Usage:
    python -m src.experiments.statistical_analysis --input results/ablation/
    python -m src.experiments.statistical_analysis --input results/paper_results.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))


# ============================================================
# Token-Based Cognitive Metrics
# ============================================================

def compute_token_efficiency(accuracy: float, tokens_used: float) -> float:
    """
    Token Efficiency Ratio: accuracy per unit of computational effort.

    TER = accuracy / (tokens_used / 100)

    Higher is better — more accuracy per token.
    This replaces response time as the effort metric.

    Args:
        accuracy: Proportion correct (0–1).
        tokens_used: Mean tokens consumed per response.

    Returns:
        Token Efficiency Ratio (float). Returns 0.0 if tokens_used is 0.
    """
    if tokens_used == 0:
        return 0.0
    return accuracy / (tokens_used / 100)


def compute_reasoning_density(reasoning_steps: int, tokens_used: int) -> float:
    """
    Reasoning Density: explicit reasoning steps per token.

    RD = reasoning_steps / tokens_used

    Args:
        reasoning_steps: Number of explicit reasoning steps in the response.
        tokens_used: Total tokens used in the response.

    Returns:
        Reasoning Density (float). Returns 0.0 if tokens_used is 0.
    """
    if tokens_used == 0:
        return 0.0
    return reasoning_steps / tokens_used


def compute_marginal_accuracy_gain(
    s1_accuracy: float,
    s2_accuracy: float,
    s1_tokens: float,
    s2_tokens: float,
) -> float:
    """
    Marginal Accuracy Gain: additional accuracy per additional token.

    MAG = (s2_accuracy - s1_accuracy) / (s2_tokens - s1_tokens)

    Can be computed per task category to show where extra effort pays off.

    Args:
        s1_accuracy: System 1 accuracy (0–1).
        s2_accuracy: System 2 accuracy (0–1).
        s1_tokens: Mean tokens used by System 1.
        s2_tokens: Mean tokens used by System 2.

    Returns:
        Marginal Accuracy Gain (float). Returns 0.0 if token counts are equal.
    """
    token_diff = s2_tokens - s1_tokens
    if token_diff == 0:
        return 0.0
    return (s2_accuracy - s1_accuracy) / token_diff


# ============================================================
# Effect Size
# ============================================================

def cohens_d(group1: List[float], group2: List[float]) -> Tuple[float, str]:
    """
    Compute Cohen's d effect size with interpretation.

    Thresholds (absolute value):
        |d| < 0.2:             negligible
        0.2 <= |d| < 0.5:     small
        0.5 <= |d| < 0.8:     medium
        |d| >= 0.8:            large

    Args:
        group1: Observations for group 1.
        group2: Observations for group 2.

    Returns:
        Tuple of (d, interpretation_string).
    """
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0, "N/A (insufficient data)"

    pooled_var = (
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2)
    )
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        d = 0.0
    else:
        d = (np.mean(g1) - np.mean(g2)) / pooled_std

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "negligible"
    elif abs_d < 0.5:
        interp = "small"
    elif abs_d < 0.8:
        interp = "medium"
    else:
        interp = "large"

    return float(d), interp


# ============================================================
# Pairwise Comparison
# ============================================================

def pairwise_comparison(
    group1: List[float],
    group2: List[float],
    label: str = "",
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> Dict[str, Any]:
    """
    Complete statistical comparison between two groups.

    Performs a two-tailed independent-samples t-test and reports ALL
    statistics, including non-significant results. Applies Bonferroni
    correction when n_comparisons > 1.

    Args:
        group1: Observations for group 1.
        group2: Observations for group 2.
        label: Human-readable label for this comparison.
        alpha: Significance threshold (default 0.05).
        n_comparisons: Total number of comparisons for Bonferroni correction.

    Returns:
        Dict with full statistics:
            label, n1, n2, mean1, mean2, std1, std2,
            difference, t_statistic, df, p_value, p_value_corrected,
            cohens_d, cohens_d_interpretation,
            ci_95_lower, ci_95_upper,
            significant_uncorrected, significant_corrected
    """
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)

    n1, n2 = len(g1), len(g2)
    mean1, mean2 = float(np.mean(g1)), float(np.mean(g2))
    std1 = float(np.std(g1, ddof=1)) if n1 > 1 else float("nan")
    std2 = float(np.std(g2, ddof=1)) if n2 > 1 else float("nan")
    difference = mean2 - mean1

    # t-test
    if n1 > 1 and n2 > 1:
        t_stat, p_value = stats.ttest_ind(g1, g2)
        df = float(n1 + n2 - 2)
        t_stat = float(t_stat)
        p_value = float(p_value)
    else:
        t_stat, p_value, df = float("nan"), float("nan"), float("nan")

    # Bonferroni correction
    p_value_corrected = min(p_value * n_comparisons, 1.0) if not np.isnan(p_value) else float("nan")

    # Cohen's d
    d, d_interp = cohens_d(group1, group2)

    # 95% CI for the difference (Welch-style)
    if n1 > 1 and n2 > 1:
        se_diff = np.sqrt(np.var(g1, ddof=1) / n1 + np.var(g2, ddof=1) / n2)
        # Degrees of freedom via Welch-Satterthwaite
        var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        welch_df = (var1 / n1 + var2 / n2) ** 2 / denom if denom > 0 else df
        t_crit = stats.t.ppf(0.975, df=welch_df)
        ci_lower = float(difference - t_crit * se_diff)
        ci_upper = float(difference + t_crit * se_diff)
    else:
        ci_lower, ci_upper = float("nan"), float("nan")

    return {
        "label": label,
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "std1": std1,
        "std2": std2,
        "difference": difference,
        "t_statistic": t_stat,
        "df": df,
        "p_value": p_value,
        "p_value_corrected": p_value_corrected,
        "cohens_d": d,
        "cohens_d_interpretation": d_interp,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_uncorrected": (p_value < alpha) if not np.isnan(p_value) else False,
        "significant_corrected": (p_value_corrected < alpha) if not np.isnan(p_value_corrected) else False,
    }


# ============================================================
# Full Statistical Report
# ============================================================

def full_statistical_report(
    results: dict,
    n_comparisons: int = None,
    alpha: float = 0.05,
) -> dict:
    """
    Generate COMPLETE statistical report from experiment results.

    For EVERY comparison, reports:
    - t-statistic, df, p-value (two-tailed)
    - Bonferroni-corrected p-value (if n_comparisons provided)
    - Cohen's d effect size with interpretation
    - 95% CI for the difference
    - Whether significant at uncorrected and corrected thresholds

    Comparisons included (NONE omitted):
    - Overall: S1 vs S2 accuracy
    - Per category: S1 vs S2 on intuitive, analytical, conflict (ALL three)
    - Confidence: S1 vs S2
    - Token usage: S1 vs S2
    - Token efficiency: S1 vs S2

    Args:
        results: Experiment results dict (as produced by run_experiment).
        n_comparisons: Total comparisons for Bonferroni correction.
                       If None, auto-detected from number of comparisons run.
        alpha: Significance threshold.

    Returns:
        Dict with comparisons list, token_metrics, and summary.
    """
    comparisons: List[Dict[str, Any]] = []

    # --- Extract trial-level data by system and category ---
    def _extract(key: str, category: Optional[str] = None) -> Tuple[List[float], List[float]]:
        """Return (s1_values, s2_values) for a given field, optionally filtered by category."""
        s1_vals: List[float] = []
        s2_vals: List[float] = []

        # Handle both flat results dicts and nested condition-based results
        if "conditions" in results:
            # Factorial ablation format
            for cid, cdata in results["conditions"].items():
                trials = cdata.get("trials", [])
                cond = cdata.get("condition", {})
                model = cond.get("model", "")
                prompt = cond.get("prompt", "")
                # Heuristic: C1 is canonical S1, C8 is canonical S2
                is_s1 = cid == "C1"
                is_s2 = cid == "C8"
                if not (is_s1 or is_s2):
                    continue
                for t in trials:
                    if category and t.get("task_category") != category:
                        continue
                    val = t.get(key)
                    if val is not None:
                        if is_s1:
                            s1_vals.append(float(val))
                        else:
                            s2_vals.append(float(val))
        elif "system1" in results and "system2" in results:
            # Direct S1/S2 format
            for t in results["system1"].get("trials", []):
                if category and t.get("task_category") != category:
                    continue
                val = t.get(key)
                if val is not None:
                    s1_vals.append(float(val))
            for t in results["system2"].get("trials", []):
                if category and t.get("task_category") != category:
                    continue
                val = t.get(key)
                if val is not None:
                    s2_vals.append(float(val))

        return s1_vals, s2_vals

    # Determine categories present
    categories: List[str] = []
    if "conditions" in results:
        sample_trials = []
        for cdata in results["conditions"].values():
            sample_trials = cdata.get("trials", [])
            if sample_trials:
                break
        categories = list({t.get("task_category", "") for t in sample_trials if t.get("task_category")})
    elif "system1" in results:
        categories = list({t.get("task_category", "") for t in results["system1"].get("trials", []) if t.get("task_category")})

    # Ensure expected categories are present even if not detected
    for expected in ["intuitive", "analytical", "conflict"]:
        if expected not in categories:
            categories.append(expected)

    # Total comparisons for Bonferroni:
    # 1 overall + len(categories) per-category + 1 confidence + 1 tokens + 1 TER = 5 + len(categories)
    detected_n = 4 + len(categories)
    if n_comparisons is None:
        n_comparisons = detected_n

    # --- Overall accuracy ---
    s1_acc, s2_acc = _extract("is_correct")
    if s1_acc or s2_acc:
        comparisons.append(pairwise_comparison(
            s1_acc, s2_acc,
            label="Overall: S1 vs S2 accuracy",
            alpha=alpha,
            n_comparisons=n_comparisons,
        ))

    # --- Per-category accuracy (ALL categories, no omission) ---
    for cat in sorted(categories):
        s1_cat, s2_cat = _extract("is_correct", category=cat)
        comparisons.append(pairwise_comparison(
            s1_cat, s2_cat,
            label=f"Category '{cat}': S1 vs S2 accuracy",
            alpha=alpha,
            n_comparisons=n_comparisons,
        ))

    # --- Confidence ---
    s1_conf, s2_conf = _extract("confidence")
    if s1_conf or s2_conf:
        comparisons.append(pairwise_comparison(
            s1_conf, s2_conf,
            label="Confidence: S1 vs S2",
            alpha=alpha,
            n_comparisons=n_comparisons,
        ))

    # --- Token usage ---
    s1_tok, s2_tok = _extract("tokens_used")
    if s1_tok or s2_tok:
        comparisons.append(pairwise_comparison(
            s1_tok, s2_tok,
            label="Token usage: S1 vs S2",
            alpha=alpha,
            n_comparisons=n_comparisons,
        ))

    # --- Token Efficiency Ratio ---
    s1_acc_vals, s2_acc_vals = _extract("is_correct")
    s1_tok_vals, s2_tok_vals = _extract("tokens_used")
    if s1_acc_vals and s1_tok_vals and s2_acc_vals and s2_tok_vals:
        s1_ter = [
            compute_token_efficiency(float(a), float(t))
            for a, t in zip(s1_acc_vals, s1_tok_vals)
        ]
        s2_ter = [
            compute_token_efficiency(float(a), float(t))
            for a, t in zip(s2_acc_vals, s2_tok_vals)
        ]
        comparisons.append(pairwise_comparison(
            s1_ter, s2_ter,
            label="Token Efficiency Ratio: S1 vs S2",
            alpha=alpha,
            n_comparisons=n_comparisons,
        ))

    # --- Token-based summary metrics (no comparison, just descriptive) ---
    s1_acc_mean = float(np.mean(s1_acc_vals)) if s1_acc_vals else float("nan")
    s2_acc_mean = float(np.mean(s2_acc_vals)) if s2_acc_vals else float("nan")
    s1_tok_mean = float(np.mean(s1_tok_vals)) if s1_tok_vals else float("nan")
    s2_tok_mean = float(np.mean(s2_tok_vals)) if s2_tok_vals else float("nan")

    token_metrics = {
        "s1_token_efficiency_ratio": compute_token_efficiency(s1_acc_mean, s1_tok_mean),
        "s2_token_efficiency_ratio": compute_token_efficiency(s2_acc_mean, s2_tok_mean),
        "marginal_accuracy_gain": compute_marginal_accuracy_gain(
            s1_acc_mean, s2_acc_mean, s1_tok_mean, s2_tok_mean
        ),
        "s1_avg_tokens": s1_tok_mean,
        "s2_avg_tokens": s2_tok_mean,
        "s1_avg_accuracy": s1_acc_mean,
        "s2_avg_accuracy": s2_acc_mean,
    }

    # --- Summary ---
    n_sig_uncorrected = sum(c.get("significant_uncorrected", False) for c in comparisons)
    n_sig_corrected = sum(c.get("significant_corrected", False) for c in comparisons)

    return {
        "metadata": {
            "n_comparisons_total": n_comparisons,
            "alpha": alpha,
            "correction_method": "bonferroni",
            "timestamp": datetime.now().isoformat(),
        },
        "comparisons": comparisons,
        "token_metrics": token_metrics,
        "summary": {
            "total_comparisons": len(comparisons),
            "n_significant_uncorrected": n_sig_uncorrected,
            "n_significant_corrected": n_sig_corrected,
            "note": (
                "All comparisons are reported regardless of significance. "
                "p_value_corrected uses Bonferroni correction."
            ),
        },
    }


# ============================================================
# Factorial ANOVA Report
# ============================================================

def factorial_anova_report(results: dict) -> dict:
    """
    For factorial ablation results, compute:
    - Main effect of each factor (with effect size)
    - All 2-way interactions
    - Summary table

    Note: With only 8 conditions and aggregated data, full ANOVA is
    not appropriate. Main effects are reported as mean differences
    averaged over the other factors. Effect sizes are Cohen's d on
    per-trial accuracy arrays.

    Args:
        results: Output from the 2x2x2 ablation experiment.

    Returns:
        Dict with main_effects, interactions, and summary_table.
    """
    if "conditions" not in results:
        return {"error": "No 'conditions' key found. Expected factorial ablation results."}

    conditions = results["conditions"]

    def _trials_for(cids: List[str]) -> List[float]:
        vals: List[float] = []
        for cid in cids:
            if cid in conditions:
                for t in conditions[cid].get("trials", []):
                    vals.append(float(t.get("is_correct", 0)))
        return vals

    # Factor membership
    mini_ids = ["C1", "C2", "C3", "C4"]
    gpt4o_ids = ["C5", "C6", "C7", "C8"]
    high_t_ids = ["C1", "C2", "C5", "C6"]
    low_t_ids = ["C3", "C4", "C7", "C8"]
    zs_ids = ["C1", "C3", "C5", "C7"]
    cot_ids = ["C2", "C4", "C6", "C8"]

    mini_acc = _trials_for(mini_ids)
    gpt4o_acc = _trials_for(gpt4o_ids)
    high_t_acc = _trials_for(high_t_ids)
    low_t_acc = _trials_for(low_t_ids)
    zs_acc = _trials_for(zs_ids)
    cot_acc = _trials_for(cot_ids)

    def _effect(a: List[float], b: List[float], label: str) -> Dict[str, Any]:
        d, interp = cohens_d(a, b)
        diff = float(np.mean(b) - np.mean(a)) if a and b else float("nan")
        t_stat, p_val = (stats.ttest_ind(a, b) if (len(a) > 1 and len(b) > 1)
                         else (float("nan"), float("nan")))
        return {
            "label": label,
            "mean_level_0": float(np.mean(a)) if a else float("nan"),
            "mean_level_1": float(np.mean(b)) if b else float("nan"),
            "mean_difference": diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": d,
            "cohens_d_interpretation": interp,
        }

    main_effects = {
        "model": _effect(mini_acc, gpt4o_acc, "Model: gpt-4o-mini vs gpt-4o"),
        "temperature": _effect(high_t_acc, low_t_acc, "Temperature: high (0.9) vs low (0.2)"),
        "prompt": _effect(zs_acc, cot_acc, "Prompt: zero-shot vs CoT"),
    }

    # 2-way interactions as difference-of-differences
    def _interaction(
        a_lo: List[float], a_hi: List[float],
        b_lo: List[float], b_hi: List[float],
        label: str,
    ) -> Dict[str, Any]:
        effect_a = float(np.mean(a_hi) - np.mean(a_lo)) if a_lo and a_hi else float("nan")
        effect_b = float(np.mean(b_hi) - np.mean(b_lo)) if b_lo and b_hi else float("nan")
        interaction = effect_b - effect_a if not (np.isnan(effect_a) or np.isnan(effect_b)) else float("nan")
        return {
            "label": label,
            "effect_in_level_0": effect_a,
            "effect_in_level_1": effect_b,
            "interaction": interaction,
        }

    # Model x Prompt: does CoT benefit differ between mini and gpt-4o?
    mini_zs = _trials_for(["C1", "C3"])
    mini_cot = _trials_for(["C2", "C4"])
    gpt4o_zs = _trials_for(["C5", "C7"])
    gpt4o_cot = _trials_for(["C6", "C8"])

    # Model x Temperature: does low-T benefit differ between models?
    mini_high = _trials_for(["C1", "C2"])
    mini_low = _trials_for(["C3", "C4"])
    gpt4o_high = _trials_for(["C5", "C6"])
    gpt4o_low = _trials_for(["C7", "C8"])

    # Temperature x Prompt: does CoT benefit differ across temperatures?
    high_t_zs = _trials_for(["C1", "C5"])
    high_t_cot = _trials_for(["C2", "C6"])
    low_t_zs = _trials_for(["C3", "C7"])
    low_t_cot = _trials_for(["C4", "C8"])

    interactions = {
        "model_x_prompt": _interaction(
            mini_zs, mini_cot, gpt4o_zs, gpt4o_cot,
            "Model x Prompt: CoT benefit in mini vs gpt-4o",
        ),
        "model_x_temperature": _interaction(
            mini_high, mini_low, gpt4o_high, gpt4o_low,
            "Model x Temperature: low-T benefit in mini vs gpt-4o",
        ),
        "temperature_x_prompt": _interaction(
            high_t_zs, high_t_cot, low_t_zs, low_t_cot,
            "Temperature x Prompt: CoT benefit at high-T vs low-T",
        ),
    }

    # Summary table rows
    summary_rows = []
    for factor, data in main_effects.items():
        summary_rows.append({
            "type": "main_effect",
            "factor": factor,
            "mean_difference": data["mean_difference"],
            "cohens_d": data["cohens_d"],
            "p_value": data["p_value"],
        })
    for name, data in interactions.items():
        summary_rows.append({
            "type": "interaction",
            "factor": name,
            "interaction_value": data["interaction"],
        })

    return {
        "main_effects": main_effects,
        "interactions": interactions,
        "summary_table": summary_rows,
    }


# ============================================================
# Formatting
# ============================================================

def format_report_table(report: dict) -> str:
    """
    Format the statistical report as a clean text table.

    Includes ALL comparisons. Significant results (uncorrected) are
    flagged with '*', Bonferroni-corrected significance with '**'.

    Args:
        report: Output from full_statistical_report().

    Returns:
        Multi-line string suitable for printing to stdout or saving.
    """
    lines: List[str] = []
    meta = report.get("metadata", {})
    alpha = meta.get("alpha", 0.05)
    correction = meta.get("correction_method", "bonferroni")
    n_comp = meta.get("n_comparisons_total", "?")

    lines.append("=" * 100)
    lines.append("COMPLETE STATISTICAL REPORT — DUAL-PROCESS EXPERIMENT")
    lines.append(f"  alpha={alpha}  correction={correction}  n_comparisons={n_comp}")
    lines.append("=" * 100)
    lines.append(
        f"  {'Comparison':<50} {'N1':>5} {'N2':>5} {'Mean1':>7} {'Mean2':>7} "
        f"{'Diff':>7} {'t':>7} {'df':>6} {'p':>8} {'p_corr':>8} {'d':>6} {'Sig':>5}"
    )
    lines.append("-" * 100)

    for c in report.get("comparisons", []):
        sig = ""
        if c.get("significant_corrected"):
            sig = "**"
        elif c.get("significant_uncorrected"):
            sig = "*"

        def _fmt(v, fmt=".3f"):
            return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "   N/A"

        lines.append(
            f"  {c.get('label', ''):<50} "
            f"{c.get('n1', 0):>5} {c.get('n2', 0):>5} "
            f"{_fmt(c.get('mean1', float('nan'))):>7} "
            f"{_fmt(c.get('mean2', float('nan'))):>7} "
            f"{_fmt(c.get('difference', float('nan'))):>7} "
            f"{_fmt(c.get('t_statistic', float('nan'))):>7} "
            f"{_fmt(c.get('df', float('nan')), '.1f'):>6} "
            f"{_fmt(c.get('p_value', float('nan')), '.4f'):>8} "
            f"{_fmt(c.get('p_value_corrected', float('nan')), '.4f'):>8} "
            f"{_fmt(c.get('cohens_d', float('nan'))):>6} "
            f"{sig:>5}"
        )

    lines.append("-" * 100)
    lines.append("  * p < alpha (uncorrected)   ** p < alpha (Bonferroni-corrected)")
    lines.append("")

    summ = report.get("summary", {})
    lines.append(
        f"  Total comparisons: {summ.get('total_comparisons', '?')}   "
        f"Significant uncorrected: {summ.get('n_significant_uncorrected', '?')}   "
        f"Significant corrected: {summ.get('n_significant_corrected', '?')}"
    )

    tok = report.get("token_metrics", {})
    if tok:
        lines.append("")
        lines.append("  TOKEN-BASED METRICS")
        lines.append(f"    S1 Token Efficiency Ratio : {tok.get('s1_token_efficiency_ratio', float('nan')):.4f}")
        lines.append(f"    S2 Token Efficiency Ratio : {tok.get('s2_token_efficiency_ratio', float('nan')):.4f}")
        lines.append(f"    Marginal Accuracy Gain    : {tok.get('marginal_accuracy_gain', float('nan')):.6f} acc/token")
        lines.append(f"    S1 avg tokens             : {tok.get('s1_avg_tokens', float('nan')):.1f}")
        lines.append(f"    S2 avg tokens             : {tok.get('s2_avg_tokens', float('nan')):.1f}")

    lines.append("=" * 100)
    return "\n".join(lines)


# ============================================================
# I/O Helpers
# ============================================================

def _load_results(input_path: Path) -> dict:
    """Load experiment results from a JSON file or the newest JSON in a directory."""
    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {input_path}")
        input_path = json_files[-1]
        print(f"Loading newest results file: {input_path}")
    elif not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# CLI Entry Point
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive statistical analysis for dual-process experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results JSON file or directory containing JSON files.",
    )
    parser.add_argument(
        "--output",
        default="results/analysis/",
        help="Output directory for analysis JSON and text report (default: results/analysis/).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--correction",
        choices=["bonferroni", "none"],
        default="bonferroni",
        help="Multiple comparison correction method (default: bonferroni).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    input_path = Path(args.input)
    output_path = project_root / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_path}")
    results = _load_results(input_path)

    # Full statistical report
    report = full_statistical_report(results, alpha=args.alpha)

    # Factorial ANOVA report (if applicable)
    anova = {}
    if "conditions" in results:
        anova = factorial_anova_report(results)

    # Print table
    table = format_report_table(report)
    print(table)

    # Save JSON report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"statistical_report_{timestamp}.json"
    combined = {"statistical_report": report, "factorial_anova": anova}
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nReport saved to: {report_file}")

    # Save text table
    table_file = output_path / f"statistical_report_{timestamp}.txt"
    with open(table_file, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Text table saved to: {table_file}")
