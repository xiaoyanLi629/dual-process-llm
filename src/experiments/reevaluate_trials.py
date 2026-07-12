"""
Re-evaluate all stored trials with the fixed evaluator and recompute
every downstream statistic (factorial, multi-model, human comparison,
variance baseline, key comparisons, 3-way ANOVA with F / partial eta^2).

No API re-calls: we re-derive `is_correct` from stored answer / correct_answer
pairs. This eliminates the substring-match bug in the old evaluator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))

from evaluation.evaluator import DualProcessEvaluator  # noqa: E402

_EVAL = DualProcessEvaluator()


def _recheck(ans: Any, correct: Any) -> bool:
    """Re-evaluate a single trial with the fixed evaluator."""
    if ans is None or correct is None:
        return False
    return _EVAL._check_correctness(str(ans), correct)


# ============================================================
# Factorial ablation
# ============================================================
def reevaluate_factorial(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    for cid, cond in data["conditions"].items():
        for t in cond["trials"]:
            t["is_correct"] = _recheck(t.get("answer"), t.get("correct_answer"))
        cond["summary"] = _condition_summary(cond, cond["trials"])

    # Re-run summaries and analyses
    data["main_effects"] = _main_effects(data["conditions"])
    data["interactions"] = _interactions(data["conditions"])
    data["key_comparisons"] = _key_comparisons(data["conditions"])
    data["anova"] = _three_way_anova(data["conditions"])
    data["category_main_effects"] = _main_effects_by_category(data["conditions"])
    data["pairwise_tests"] = _pairwise_tests(data["conditions"])

    # Variance baseline
    if data.get("variance_baseline"):
        vb = data["variance_baseline"]
        for run_key in ("run1_trials", "run2_trials"):
            for t in vb.get(run_key, []):
                t["is_correct"] = _recheck(t.get("answer"), t.get("correct_answer"))
        run1 = vb.get("run1_trials", [])
        run2 = vb.get("run2_trials", [])
        n = len(run1)
        acc1 = sum(t["is_correct"] for t in run1) / n if n else float("nan")
        acc2 = sum(t["is_correct"] for t in run2) / n if n else float("nan")
        agreement = (sum(r1["is_correct"] == r2["is_correct"] for r1, r2 in zip(run1, run2))
                     / n if n else float("nan"))
        vb["accuracy_run1"] = acc1
        vb["accuracy_run2"] = acc2
        vb["accuracy_difference"] = abs(acc1 - acc2)
        vb["agreement_rate"] = agreement

    return data


def _condition_summary(cond: Dict, trials: List[Dict]) -> Dict:
    overall_correct = sum(t["is_correct"] for t in trials)
    overall_acc = overall_correct / len(trials) if trials else float("nan")

    categories: Dict[str, List[int]] = {}
    for t in trials:
        categories.setdefault(t["task_category"], []).append(int(t["is_correct"]))

    by_category = {
        cat: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
        for cat, v in categories.items()
    }

    return {
        "condition_id": cond.get("condition", {}).get("id") or cond.get("summary", {}).get("condition_id"),
        "label": cond.get("condition", {}).get("label") or cond.get("summary", {}).get("label"),
        "model": cond.get("condition", {}).get("model") or cond.get("summary", {}).get("model"),
        "temperature": cond.get("condition", {}).get("temperature") or cond.get("summary", {}).get("temperature"),
        "prompt": cond.get("condition", {}).get("prompt") or cond.get("summary", {}).get("prompt"),
        "n_trials": len(trials),
        "n_correct": overall_correct,
        "accuracy_overall": overall_acc,
        "avg_confidence": float(np.mean([t.get("confidence", 0.0) for t in trials])) if trials else float("nan"),
        "avg_tokens": float(np.mean([t.get("tokens_used", 0) for t in trials])) if trials else float("nan"),
        "parse_errors": sum(bool(t.get("parse_error", False)) for t in trials),
        "by_category": by_category,
    }


def _accs_by_cid(conditions: Dict, cids: List[str]) -> List[float]:
    """Per-condition mean accuracies for the given CIDs (used for averages-of-means)."""
    out: List[float] = []
    for cid in cids:
        trials = conditions[cid]["trials"]
        if trials:
            out.append(sum(t["is_correct"] for t in trials) / len(trials))
    return out


def _trial_correctness(conditions: Dict, cids: List[str]) -> List[int]:
    """Flat list of 0/1 correctness across given CIDs."""
    vals: List[int] = []
    for cid in cids:
        for t in conditions[cid]["trials"]:
            vals.append(int(t["is_correct"]))
    return vals


def _main_effects(conditions: Dict) -> Dict:
    mini = _accs_by_cid(conditions, ["C1", "C2", "C3", "C4"])
    full = _accs_by_cid(conditions, ["C5", "C6", "C7", "C8"])
    high_t = _accs_by_cid(conditions, ["C1", "C2", "C5", "C6"])
    low_t = _accs_by_cid(conditions, ["C3", "C4", "C7", "C8"])
    zs = _accs_by_cid(conditions, ["C1", "C3", "C5", "C7"])
    cot = _accs_by_cid(conditions, ["C2", "C4", "C6", "C8"])
    return {
        "model": {
            "gpt-4o-mini_accuracy": float(np.mean(mini)),
            "gpt-4o_accuracy": float(np.mean(full)),
            "effect_gpt4o_minus_mini": float(np.mean(full) - np.mean(mini)),
        },
        "temperature": {
            "high_temp_accuracy": float(np.mean(high_t)),
            "low_temp_accuracy": float(np.mean(low_t)),
            "effect_low_minus_high": float(np.mean(low_t) - np.mean(high_t)),
        },
        "prompt": {
            "zero_shot_accuracy": float(np.mean(zs)),
            "cot_accuracy": float(np.mean(cot)),
            "effect_cot_minus_zero_shot": float(np.mean(cot) - np.mean(zs)),
        },
    }


def _interactions(conditions: Dict) -> Dict:
    def acc(cid: str) -> float:
        t = conditions[cid]["trials"]
        return sum(r["is_correct"] for r in t) / len(t) if t else float("nan")

    cot_in_mini = ((acc("C2") - acc("C1")) + (acc("C4") - acc("C3"))) / 2
    cot_in_4o = ((acc("C6") - acc("C5")) + (acc("C8") - acc("C7"))) / 2
    lowt_in_mini = ((acc("C3") - acc("C1")) + (acc("C4") - acc("C2"))) / 2
    lowt_in_4o = ((acc("C7") - acc("C5")) + (acc("C8") - acc("C6"))) / 2
    cot_at_highT = ((acc("C2") - acc("C1")) + (acc("C6") - acc("C5"))) / 2
    cot_at_lowT = ((acc("C4") - acc("C3")) + (acc("C8") - acc("C7"))) / 2
    return {
        "model_x_prompt": {
            "cot_effect_in_mini": cot_in_mini,
            "cot_effect_in_4o": cot_in_4o,
            "interaction": cot_in_4o - cot_in_mini,
            "interpretation": "Positive = CoT helps more for gpt-4o than gpt-4o-mini",
        },
        "model_x_temperature": {
            "lowT_effect_in_mini": lowt_in_mini,
            "lowT_effect_in_4o": lowt_in_4o,
            "interaction": lowt_in_4o - lowt_in_mini,
            "interpretation": "Positive = low temperature helps more for gpt-4o than gpt-4o-mini",
        },
        "temperature_x_prompt": {
            "cot_effect_at_highT": cot_at_highT,
            "cot_effect_at_lowT": cot_at_lowT,
            "interaction": cot_at_lowT - cot_at_highT,
            "interpretation": "Positive = CoT helps more at low temperature than high temperature",
        },
    }


def _key_comparisons(conditions: Dict) -> Dict:
    def acc(cid: str) -> float:
        t = conditions[cid]["trials"]
        return sum(r["is_correct"] for r in t) / len(t) if t else float("nan")
    return {
        "primary": {
            "label": "C5 vs C8 (same-model: gpt-4o, isolates temperature + prompting)",
            "C5": acc("C5"),
            "C8": acc("C8"),
            "delta_C8_minus_C5": acc("C8") - acc("C5"),
        },
        "secondary": {
            "label": "C1 vs C8 (canonical S1 vs S2, all three factors vary)",
            "C1": acc("C1"),
            "C8": acc("C8"),
            "delta_C8_minus_C1": acc("C8") - acc("C1"),
        },
    }


# ============================================================
# 3-way factorial ANOVA with F and partial eta^2
# ============================================================
def _three_way_anova(conditions: Dict) -> Dict:
    """Type III-style 3-way ANOVA on 0/1 correctness using statsmodels."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    rows = []
    for cid, cond in conditions.items():
        cfg = cond["condition"]
        model_lvl = "full" if cfg["model"] == "gpt-4o" else "mini"
        temp_lvl = "low" if cfg["temperature"] <= 0.5 else "high"
        prompt_lvl = "cot" if cfg["prompt"] == "cot" else "zs"
        for t in cond["trials"]:
            rows.append({
                "correct": int(t["is_correct"]),
                "model": model_lvl,
                "temp": temp_lvl,
                "prompt": prompt_lvl,
                "category": t["task_category"],
            })
    df = pd.DataFrame(rows)

    # Overall 3-way ANOVA (pooled across categories)
    model = ols("correct ~ C(model) * C(temp) * C(prompt)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    ss_resid = anova_table.loc["Residual", "sum_sq"]

    def _row(name: str) -> Dict[str, float]:
        r = anova_table.loc[name]
        ss = float(r["sum_sq"])
        df_num = float(r["df"])
        f = float(r["F"]) if not np.isnan(r["F"]) else float("nan")
        p = float(r["PR(>F)"]) if not np.isnan(r["PR(>F)"]) else float("nan")
        eta2_p = ss / (ss + ss_resid) if (ss + ss_resid) > 0 else float("nan")
        return {"ss": ss, "df": df_num, "F": f, "p": p, "partial_eta2": eta2_p}

    overall = {
        "model": _row("C(model)"),
        "temp": _row("C(temp)"),
        "prompt": _row("C(prompt)"),
        "model_x_temp": _row("C(model):C(temp)"),
        "model_x_prompt": _row("C(model):C(prompt)"),
        "temp_x_prompt": _row("C(temp):C(prompt)"),
        "model_x_temp_x_prompt": _row("C(model):C(temp):C(prompt)"),
        "residual_df": float(anova_table.loc["Residual", "df"]),
    }

    # Prompt x Category (central dual-process test): 2x3 ANOVA
    model_pc = ols("correct ~ C(prompt) * C(category)", data=df).fit()
    anova_pc = sm.stats.anova_lm(model_pc, typ=2)
    ss_resid_pc = anova_pc.loc["Residual", "sum_sq"]

    def _row_pc(name: str) -> Dict[str, float]:
        r = anova_pc.loc[name]
        ss = float(r["sum_sq"])
        return {
            "ss": ss,
            "df": float(r["df"]),
            "F": float(r["F"]) if not np.isnan(r["F"]) else float("nan"),
            "p": float(r["PR(>F)"]) if not np.isnan(r["PR(>F)"]) else float("nan"),
            "partial_eta2": ss / (ss + ss_resid_pc) if (ss + ss_resid_pc) > 0 else float("nan"),
        }

    prompt_x_cat = {
        "prompt": _row_pc("C(prompt)"),
        "category": _row_pc("C(category)"),
        "prompt_x_category": _row_pc("C(prompt):C(category)"),
        "residual_df": float(anova_pc.loc["Residual", "df"]),
    }

    return {"overall": overall, "prompt_x_category": prompt_x_cat}


def _main_effects_by_category(conditions: Dict) -> Dict:
    """Per-category main effect of each factor (paper Table II)."""
    cats = ["system1", "system2", "conflict"]
    by_cat = {c: {} for c in cats}
    sel = {
        "model": (["C1", "C2", "C3", "C4"], ["C5", "C6", "C7", "C8"]),
        "temperature": (["C1", "C2", "C5", "C6"], ["C3", "C4", "C7", "C8"]),
        "prompt": (["C1", "C3", "C5", "C7"], ["C2", "C4", "C6", "C8"]),
    }
    for cat in cats:
        for factor, (lo_cids, hi_cids) in sel.items():
            lo_accs = _cat_accs(conditions, lo_cids, cat)
            hi_accs = _cat_accs(conditions, hi_cids, cat)
            by_cat[cat][factor] = {
                "lo_mean": float(np.mean(lo_accs)) if lo_accs else float("nan"),
                "hi_mean": float(np.mean(hi_accs)) if hi_accs else float("nan"),
                "effect": float(np.mean(hi_accs) - np.mean(lo_accs)) if lo_accs and hi_accs else float("nan"),
            }
    # Overall (pooled across categories, averages-of-means across conditions)
    by_cat["overall"] = {
        f: {
            "effect": _main_effects(conditions)[f][list(_main_effects(conditions)[f].keys())[-1]]
        }
        for f in ["model", "temperature", "prompt"]
    }
    return by_cat


def _cat_accs(conditions: Dict, cids: List[str], cat: str) -> List[float]:
    out = []
    for cid in cids:
        trials = [t for t in conditions[cid]["trials"] if t["task_category"] == cat]
        if trials:
            out.append(sum(t["is_correct"] for t in trials) / len(trials))
    return out


def _pairwise_tests(conditions: Dict) -> Dict:
    """t-tests and Cohen's d for C1/C5/C8 by category, for the paper narrative."""
    out = {}
    for cat in ["system1", "system2", "conflict"]:
        c1 = [int(t["is_correct"]) for t in conditions["C1"]["trials"] if t["task_category"] == cat]
        c5 = [int(t["is_correct"]) for t in conditions["C5"]["trials"] if t["task_category"] == cat]
        c8 = [int(t["is_correct"]) for t in conditions["C8"]["trials"] if t["task_category"] == cat]

        def _pair(a, b, label):
            if len(a) < 2 or len(b) < 2:
                return None
            t_stat, p = stats.ttest_ind(a, b, equal_var=False)
            n1, n2 = len(a), len(b)
            v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
            pooled_sd = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))) or 1e-9
            d = float((np.mean(b) - np.mean(a)) / pooled_sd)
            return {
                "label": label,
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "delta": float(np.mean(b) - np.mean(a)),
                "df": n1 + n2 - 2,
                "t": float(t_stat), "p": float(p),
                "cohens_d": d,
            }

        out[cat] = {
            "C1_vs_C8": _pair(c1, c8, "C1 vs C8 (canonical)"),
            "C5_vs_C8": _pair(c5, c8, "C5 vs C8 (same-model)"),
        }
    # Overall
    for key in ["C1_vs_C8", "C5_vs_C8"]:
        a_cid = key.split("_")[0]
        c_a = [int(t["is_correct"]) for t in conditions[a_cid]["trials"]]
        c_b = [int(t["is_correct"]) for t in conditions["C8"]["trials"]]
        out.setdefault("overall", {})
        t_stat, p = stats.ttest_ind(c_a, c_b, equal_var=False)
        n1, n2 = len(c_a), len(c_b)
        v1, v2 = np.var(c_a, ddof=1), np.var(c_b, ddof=1)
        pooled_sd = float(np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))) or 1e-9
        d = float((np.mean(c_b) - np.mean(c_a)) / pooled_sd)
        out["overall"][key] = {
            "label": key,
            "mean_a": float(np.mean(c_a)), "mean_b": float(np.mean(c_b)),
            "delta": float(np.mean(c_b) - np.mean(c_a)),
            "df": n1 + n2 - 2,
            "t": float(t_stat), "p": float(p),
            "cohens_d": d,
        }
    return out


# ============================================================
# Multi-model
# ============================================================
def reevaluate_multi_model(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)

    for fam, fdata in data["families"].items():
        for sys_name in ["system1", "system2"]:
            rbc = fdata[sys_name]["results_by_category"]
            for cat, cdata in rbc.items():
                trials = cdata.get("trials", [])
                for t in trials:
                    t["is_correct"] = _recheck(t.get("answer"), t.get("correct_answer"))
                n = len(trials)
                correct = sum(t["is_correct"] for t in trials)
                cdata["n"] = n
                cdata["correct"] = correct
                cdata["accuracy"] = correct / n if n else float("nan")

        gap = {}
        for cat in ["intuitive", "analytical", "conflict"]:
            s1 = fdata["system1"]["results_by_category"][cat]["accuracy"]
            s2 = fdata["system2"]["results_by_category"][cat]["accuracy"]
            gap[cat] = round(s2 - s1, 4)
        fdata["s2_s1_gap"] = gap

    # Cross-model consistency
    analytical_gaps = [fdata["s2_s1_gap"]["analytical"] for fdata in data["families"].values()]
    intuitive_gaps = [fdata["s2_s1_gap"]["intuitive"] for fdata in data["families"].values()]
    conflict_gaps = [fdata["s2_s1_gap"]["conflict"] for fdata in data["families"].values()]
    overall_gaps = [np.mean([fdata["s2_s1_gap"][c] for c in ["intuitive", "analytical", "conflict"]])
                    for fdata in data["families"].values()]
    pattern_holds = all(g > 0 for g in analytical_gaps)
    data["cross_model_consistency"] = {
        "pattern_holds": bool(pattern_holds),
        "families_showing_pattern": sum(1 for g in analytical_gaps if g > 0),
        "total_valid_families": len(analytical_gaps),
        "analytical_gap_range": [float(min(analytical_gaps)), float(max(analytical_gaps))],
        "intuitive_gap_range": [float(min(intuitive_gaps)), float(max(intuitive_gaps))],
        "conflict_gap_range": [float(min(conflict_gaps)), float(max(conflict_gaps))],
        "mean_analytical_gap": float(np.mean(analytical_gaps)),
        "mean_intuitive_gap": float(np.mean(intuitive_gaps)),
        "mean_conflict_gap": float(np.mean(conflict_gaps)),
        "mean_overall_gap": float(np.mean(overall_gaps)),
    }
    return data


# ============================================================
# Human comparison (recomputes with new C1 / C8 numbers + multi-model means)
# ============================================================
def build_human_comparison(factorial_data: dict, multi_model_data: dict) -> dict:
    # Human ground truth (from prior data file; conflict = 0.39, not the 0.31 in the paper)
    human_gaps = {
        "intuitive": {"mean_gap": 0.03, "n_studies": 2,
                      "study_ids": ["simple_pattern_recognition", "semantic_association"]},
        "analytical": {"mean_gap": 0.25, "n_studies": 2,
                       "study_ids": ["syllogistic_reasoning", "arithmetic_word_problems"]},
        "conflict": {"mean_gap": 0.39, "n_studies": 4,
                     "study_ids": ["crt_classic", "base_rate_neglect",
                                   "framing_effects", "conjunction_fallacy"]},
    }

    cat_map = {"intuitive": "system1", "analytical": "system2", "conflict": "conflict"}

    # LLM gaps (C1 vs C8, canonical)
    llm_gaps_canonical = {}
    for human_cat, internal_cat in cat_map.items():
        c1_acc = factorial_data["conditions"]["C1"]["summary"]["by_category"][internal_cat]["accuracy"]
        c8_acc = factorial_data["conditions"]["C8"]["summary"]["by_category"][internal_cat]["accuracy"]
        llm_gaps_canonical[human_cat] = {
            "s1_accuracy": c1_acc,
            "s2_accuracy": c8_acc,
            "s2_s1_gap": c8_acc - c1_acc,
        }

    # LLM gaps (multi-model mean)
    llm_gaps_multi = {}
    for cat in ["intuitive", "analytical", "conflict"]:
        gaps = [fdata["s2_s1_gap"][cat] for fdata in multi_model_data["families"].values()]
        llm_gaps_multi[cat] = {
            "mean_gap": float(np.mean(gaps)),
            "per_family": {fam: fdata["s2_s1_gap"][cat]
                           for fam, fdata in multi_model_data["families"].items()},
        }

    # Similarity (canonical C1 vs C8)
    cats = ["intuitive", "analytical", "conflict"]
    human_gap_list = [human_gaps[c]["mean_gap"] for c in cats]
    llm_gap_list = [llm_gaps_canonical[c]["s2_s1_gap"] for c in cats]
    rho, _ = stats.spearmanr(human_gap_list, llm_gap_list)
    human_ordering = [c for _, c in sorted(zip(human_gap_list, cats), reverse=True)]
    llm_ordering = [c for _, c in sorted(zip(llm_gap_list, cats), reverse=True)]

    per_cat = {}
    for c in cats:
        h = human_gaps[c]["mean_gap"]
        l_ = llm_gaps_canonical[c]["s2_s1_gap"]
        diff = abs(h - l_)
        if diff < 0.08 and np.sign(h) == np.sign(l_):
            match = "consistent"
        elif np.sign(h) == np.sign(l_):
            match = "partial"
        else:
            match = "divergent"
        per_cat[c] = {"human_gap": h, "llm_gap": l_, "pattern_match": match}

    return {
        "human_gaps": human_gaps,
        "llm_gaps_canonical": llm_gaps_canonical,
        "llm_gaps_multi_model": llm_gaps_multi,
        "similarity": {
            "spearman_rho": float(rho),
            "ordering_match": human_ordering == llm_ordering,
            "human_ordering": human_ordering,
            "llm_ordering": llm_ordering,
            "per_category": per_cat,
        },
    }


# ============================================================
# Entry point
# ============================================================
def main() -> None:
    root = project_root / "results" / "bibm_2026"
    factorial_in = root / "main_experiment_v2" / "ablation_factorial_20260415_034946.json"
    multi_in = root / "multi_model" / "multi_model_20260415_072827.json"
    human_out = root / "human_comparison" / "human_comparison.json"

    print(f"[1/3] Re-evaluating factorial trials: {factorial_in.name}")
    fac = reevaluate_factorial(factorial_in)
    factorial_out = root / "main_experiment_v2" / "ablation_factorial_corrected.json"
    with factorial_out.open("w") as f:
        json.dump(fac, f, indent=2, default=str)
    print(f"   -> {factorial_out.name}")
    _print_factorial_summary(fac)

    print(f"\n[2/3] Re-evaluating multi-model trials: {multi_in.name}")
    mm = reevaluate_multi_model(multi_in)
    mm_out = root / "multi_model" / "multi_model_corrected.json"
    with mm_out.open("w") as f:
        json.dump(mm, f, indent=2, default=str)
    print(f"   -> {mm_out.name}")
    _print_multi_model_summary(mm)

    print(f"\n[3/3] Rebuilding human comparison")
    hc = build_human_comparison(fac, mm)
    hc_out = root / "human_comparison" / "human_comparison_corrected.json"
    with hc_out.open("w") as f:
        json.dump(hc, f, indent=2, default=str)
    print(f"   -> {hc_out.name}")
    _print_human_comparison_summary(hc)


def _print_factorial_summary(fac: dict) -> None:
    print("\n--- CONDITION ACCURACIES (by category) ---")
    print(f"{'CID':<4} {'overall':>8} {'sys1':>7} {'sys2':>7} {'conflict':>9}  tokens")
    for cid in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
        s = fac["conditions"][cid]["summary"]
        bc = s["by_category"]
        print(f"{cid:<4} {s['accuracy_overall']:>8.3f} "
              f"{bc['system1']['accuracy']:>7.3f} "
              f"{bc['system2']['accuracy']:>7.3f} "
              f"{bc['conflict']['accuracy']:>9.3f}  "
              f"{s['avg_tokens']:.1f}")

    me = fac["main_effects"]
    print("\n--- MAIN EFFECTS (averages over other factors) ---")
    print(f"  Model (full - mini):    {me['model']['effect_gpt4o_minus_mini']:+.4f}")
    print(f"  Temp (low - high):      {me['temperature']['effect_low_minus_high']:+.4f}")
    print(f"  Prompt (CoT - zero):    {me['prompt']['effect_cot_minus_zero_shot']:+.4f}")

    ix = fac["interactions"]
    print("\n--- 2-WAY INTERACTIONS ---")
    print(f"  Model x Prompt: {ix['model_x_prompt']['interaction']:+.4f}")
    print(f"  Model x Temp:   {ix['model_x_temperature']['interaction']:+.4f}")
    print(f"  Temp x Prompt:  {ix['temperature_x_prompt']['interaction']:+.4f}")

    print("\n--- KEY COMPARISONS ---")
    kc = fac["key_comparisons"]
    print(f"  Primary C5 vs C8: C5={kc['primary']['C5']:.3f} C8={kc['primary']['C8']:.3f} "
          f"delta={kc['primary']['delta_C8_minus_C5']:+.4f}")
    print(f"  Second. C1 vs C8: C1={kc['secondary']['C1']:.3f} C8={kc['secondary']['C8']:.3f} "
          f"delta={kc['secondary']['delta_C8_minus_C1']:+.4f}")

    print("\n--- 3-WAY ANOVA (F, p, partial eta^2) ---")
    a = fac["anova"]["overall"]
    for key in ["model", "temp", "prompt",
                "model_x_temp", "model_x_prompt", "temp_x_prompt",
                "model_x_temp_x_prompt"]:
        r = a[key]
        print(f"  {key:<22} F({r['df']:.0f}, {a['residual_df']:.0f})={r['F']:.2f}  "
              f"p={r['p']:.4f}  eta2_p={r['partial_eta2']:.4f}")

    print("\n--- PROMPT x CATEGORY (central dual-process test) ---")
    pc = fac["anova"]["prompt_x_category"]
    r = pc["prompt_x_category"]
    print(f"  prompt_x_category  F({r['df']:.0f}, {pc['residual_df']:.0f})={r['F']:.2f}  "
          f"p={r['p']:.4g}  eta2_p={r['partial_eta2']:.4f}")

    print("\n--- PER-CATEGORY MAIN EFFECTS (paper Table II) ---")
    cme = fac["category_main_effects"]
    for factor in ["prompt", "model", "temperature"]:
        print(f"  {factor}:")
        for cat in ["system2", "system1", "conflict"]:  # analytical, intuitive, conflict
            lbl = {"system1": "intuitive", "system2": "analytical", "conflict": "conflict"}[cat]
            e = cme[cat][factor]
            print(f"    {lbl:<12} effect={e['effect']:+.4f}")

    vb = fac.get("variance_baseline")
    if vb:
        print(f"\n--- VARIANCE BASELINE (C8 run twice, n={vb['n_subset']}) ---")
        print(f"  acc1={vb['accuracy_run1']:.3f}  acc2={vb['accuracy_run2']:.3f}  "
              f"|delta|={vb['accuracy_difference']:.3f}  agreement={vb['agreement_rate']:.3f}")


def _print_multi_model_summary(mm: dict) -> None:
    print("\n--- MULTI-MODEL S2-S1 GAPS ---")
    print(f"{'family':<10} {'intuitive':>10} {'analytical':>11} {'conflict':>10} {'mean':>8}")
    for fam, fdata in mm["families"].items():
        g = fdata["s2_s1_gap"]
        mean_gap = np.mean([g["intuitive"], g["analytical"], g["conflict"]])
        print(f"{fam:<10} {g['intuitive']:>+10.3f} {g['analytical']:>+11.3f} "
              f"{g['conflict']:>+10.3f} {mean_gap:>+8.3f}")
    c = mm["cross_model_consistency"]
    print(f"\nMean across families: intuitive={c['mean_intuitive_gap']:+.3f}  "
          f"analytical={c['mean_analytical_gap']:+.3f}  "
          f"conflict={c['mean_conflict_gap']:+.3f}")


def _print_human_comparison_summary(hc: dict) -> None:
    print("\n--- HUMAN vs LLM (canonical C1 vs C8) ---")
    print(f"{'category':<12} {'human':>8} {'llm':>8} {'match':>12}")
    for cat in ["intuitive", "analytical", "conflict"]:
        h = hc["human_gaps"][cat]["mean_gap"]
        l_ = hc["llm_gaps_canonical"][cat]["s2_s1_gap"]
        m = hc["similarity"]["per_category"][cat]["pattern_match"]
        print(f"{cat:<12} {h:>+8.3f} {l_:>+8.3f} {m:>12}")
    print(f"Spearman rho={hc['similarity']['spearman_rho']:.3f}  "
          f"ordering_match={hc['similarity']['ordering_match']}")


if __name__ == "__main__":
    main()
