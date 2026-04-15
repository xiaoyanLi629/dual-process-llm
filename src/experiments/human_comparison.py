"""
Systematic Human-LLM Comparison Framework

Compares the PATTERN of dual-process performance differences between
humans (from published cognitive psychology literature) and LLMs, not
the absolute performance levels.

The key question: Does the S2-S1 gap follow the same pattern across
task categories (intuitive / analytical / conflict) in humans and LLMs?

Methodology notes
-----------------
- Human benchmarks are drawn from 8 peer-reviewed studies spanning
  1973-2011.  Each study provides a within-study S2-S1 contrast; the
  absolute accuracy values are NOT comparable to LLM scores because the
  task content differs entirely.
- The comparison is purely structural: do the three-category gap
  magnitudes rank in the same order, and how correlated are they?
- This script is deliberately transparent about mismatches.  Reviewers
  criticised the paper for claiming human parity; this analysis reports
  divergences explicitly.

Usage
-----
    # Against the main paper results JSON
    python -m src.experiments.human_comparison \
        --llm_results results/paper_parallel_20260125_214106/paper_results.json

    # Against an ablation directory
    python -m src.experiments.human_comparison \
        --llm_results results/ablation/ \
        --output results/human_comparison/ \
        --format both

    # Against a multi-model JSON
    python -m src.experiments.human_comparison \
        --llm_results results/multi_model/multi_model_20260125_214106.json \
        --format both
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup so the module works both as a script and as a package import
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_SRC_ROOT = _HERE.parent.parent          # .../src/
_PROJECT_ROOT = _SRC_ROOT.parent         # project root

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Default path to the human benchmarks dataset shipped with the repo
_DEFAULT_BENCHMARKS_PATH = _PROJECT_ROOT / "data" / "human_benchmarks.json"

# Default output directory
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "results" / "human_comparison"


# ===========================================================================
# 1. Load human benchmarks
# ===========================================================================

def load_human_benchmarks(path: Optional[str] = None) -> dict:
    """
    Load human benchmark data from a JSON file.

    Parameters
    ----------
    path : str or None
        Path to the benchmarks JSON.  Defaults to
        ``data/human_benchmarks.json`` in the project root.

    Returns
    -------
    dict
        The full benchmarks dict with keys ``metadata`` and ``benchmarks``.
    """
    resolved = Path(path) if path else _DEFAULT_BENCHMARKS_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"Human benchmarks file not found: {resolved}\n"
            "Expected: data/human_benchmarks.json in the project root."
        )
    with open(resolved, encoding="utf-8") as fh:
        data = json.load(fh)
    return data


# ===========================================================================
# 2. Aggregate human gaps by task category
# ===========================================================================

def aggregate_human_gaps_by_category(benchmarks: dict) -> dict:
    """
    Compute the average human S2-S1 gap per task category.

    Parameters
    ----------
    benchmarks : dict
        The dict returned by :func:`load_human_benchmarks` (or its
        ``benchmarks`` sub-dict).

    Returns
    -------
    dict
        Example::

            {
                "intuitive": {
                    "mean_gap": 0.03,
                    "n_studies": 2,
                    "studies": [
                        {"id": "simple_pattern_recognition", "gap": 0.03, ...},
                        ...
                    ]
                },
                "analytical": { ... },
                "conflict":   { ... }
            }
    """
    # Accept either the full benchmarks dict or just the inner "benchmarks" key
    if "benchmarks" in benchmarks:
        items: dict = benchmarks["benchmarks"]
    else:
        items = benchmarks

    categories: Dict[str, List[dict]] = {
        "intuitive": [],
        "analytical": [],
        "conflict": [],
    }

    for study_id, study in items.items():
        cat = study.get("task_category_mapping")
        if cat not in categories:
            continue
        gap = study.get("s2_s1_gap")
        if gap is None or (isinstance(gap, float) and math.isnan(gap)):
            continue
        categories[cat].append({
            "id": study_id,
            "gap": gap,
            "source": study.get("source", ""),
            "n_subjects": study.get("n_subjects"),
            "notes": study.get("notes", ""),
        })

    result: dict = {}
    for cat, studies in categories.items():
        if studies:
            mean_gap = sum(s["gap"] for s in studies) / len(studies)
        else:
            mean_gap = float("nan")
        result[cat] = {
            "mean_gap": round(mean_gap, 4),
            "n_studies": len(studies),
            "studies": studies,
        }

    return result


# ===========================================================================
# 3. Load LLM results
# ===========================================================================

def load_llm_results(path: str) -> dict:
    """
    Load LLM experiment results from a JSON file or directory of JSON files.

    Supports three result formats produced by this project:

    1. **Main experiment** (``paper_results.json``): contains a
       ``category_breakdown`` key with ``system1_tasks``, ``system2_tasks``,
       and ``conflict_tasks`` sub-dicts.

    2. **Ablation factorial** (``ablation_factorial_*.json``): contains a
       ``conditions`` key.  C1 is the canonical System 1 condition; C8 is
       the canonical System 2 condition.

    3. **Multi-model** (``multi_model_*.json``): contains a ``families`` key
       with per-family ``system1`` / ``system2`` sub-results.

    When *path* is a directory, every ``*.json`` file in that directory is
    loaded and the results are merged (averaged) across files.

    Parameters
    ----------
    path : str
        Path to a JSON file or a directory of JSON files.

    Returns
    -------
    dict
        Normalised result dict with the following structure::

            {
                "source": "<path>",
                "format": "main" | "ablation" | "multi_model" | "merged",
                "by_category": {
                    "intuitive":  {"s1_accuracy": 0.497, "s2_accuracy": 0.560, "s2_s1_gap": 0.063},
                    "analytical": {"s1_accuracy": 0.720, "s2_accuracy": 0.933, "s2_s1_gap": 0.213},
                    "conflict":   {"s1_accuracy": 0.950, "s2_accuracy": 0.967, "s2_s1_gap": 0.017},
                },
                "raw": <original loaded data>
            }
    """
    p = Path(path)
    if p.is_dir():
        json_files = sorted(p.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {p}")
        all_results = [_load_single_llm_json(str(f)) for f in json_files]
        return _merge_llm_results(all_results, source=str(p))
    else:
        return _load_single_llm_json(str(p))


def _load_single_llm_json(path: str) -> dict:
    """Load and normalise a single LLM result JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # ---- Detect format ----
    if "category_breakdown" in raw:
        return _parse_main_format(raw, source=path)
    elif "conditions" in raw or "experiment" in raw:
        return _parse_ablation_format(raw, source=path)
    elif "families" in raw:
        return _parse_multi_model_format(raw, source=path)
    else:
        # Try best-effort parse
        return _parse_generic_format(raw, source=path)


def _parse_main_format(raw: dict, source: str) -> dict:
    """Parse the main experiment result format."""
    cb = raw["category_breakdown"]

    def _cat(cb_key: str) -> dict:
        entry = cb.get(cb_key, {})
        s1 = entry.get("system1_accuracy", {})
        s2 = entry.get("system2_accuracy", {})
        s1_mean = s1.get("mean", float("nan"))
        s2_mean = s2.get("mean", float("nan"))
        gap = (s2_mean - s1_mean) if not (math.isnan(s1_mean) or math.isnan(s2_mean)) else float("nan")
        return {
            "s1_accuracy": round(s1_mean, 4),
            "s2_accuracy": round(s2_mean, 4),
            "s2_s1_gap":   round(gap, 4),
        }

    return {
        "source": source,
        "format": "main",
        "by_category": {
            "intuitive":  _cat("system1_tasks"),
            "analytical": _cat("system2_tasks"),
            "conflict":   _cat("conflict_tasks"),
        },
        "raw": raw,
    }


def _parse_ablation_format(raw: dict, source: str) -> dict:
    """
    Parse ablation factorial results.

    Uses C1 (System 1 canonical) and C8 (System 2 canonical) conditions.
    Falls back to averaging all zero-shot / CoT conditions if C1/C8 are absent.
    """
    conditions = raw.get("conditions", {})

    # Canonical System 1 = C1, System 2 = C8
    c1 = conditions.get("C1", {})
    c8 = conditions.get("C8", {})

    def _acc_from_condition(cond: dict, cat_key: str) -> float:
        """Extract per-category accuracy from a condition dict."""
        summary = cond.get("summary", {})
        by_cat = summary.get("by_category", {})
        cat_data = by_cat.get(cat_key, {})
        return cat_data.get("accuracy", float("nan"))

    # Category key mapping: ablation uses raw task_category values from TaskLoader
    # which can be "system1_tasks", "system2_tasks", "conflict_tasks" or
    # "intuitive", "analytical", "conflict"
    def _find_cat_key(cond: dict, candidates: List[str]) -> Optional[str]:
        summary = cond.get("summary", {})
        by_cat = summary.get("by_category", {})
        for k in candidates:
            if k in by_cat:
                return k
        return None

    intuitive_key  = _find_cat_key(c1, ["intuitive", "system1_tasks", "system1"]) or "intuitive"
    analytical_key = _find_cat_key(c1, ["analytical", "system2_tasks", "system2"]) or "analytical"
    conflict_key   = _find_cat_key(c1, ["conflict", "conflict_tasks"])  or "conflict"

    def _build_cat(cat_key: str) -> dict:
        s1 = _acc_from_condition(c1, cat_key)
        s2 = _acc_from_condition(c8, cat_key)
        gap = (s2 - s1) if not (math.isnan(s1) or math.isnan(s2)) else float("nan")
        return {
            "s1_accuracy": round(s1, 4),
            "s2_accuracy": round(s2, 4),
            "s2_s1_gap":   round(gap, 4),
        }

    return {
        "source": source,
        "format": "ablation",
        "by_category": {
            "intuitive":  _build_cat(intuitive_key),
            "analytical": _build_cat(analytical_key),
            "conflict":   _build_cat(conflict_key),
        },
        "raw": raw,
    }


def _parse_multi_model_format(raw: dict, source: str) -> dict:
    """
    Parse multi-model experiment results.

    Averages S2-S1 gaps across all model families that completed successfully.
    """
    families = raw.get("families", {})
    cats = ["intuitive", "analytical", "conflict"]

    cat_s1: Dict[str, List[float]] = {c: [] for c in cats}
    cat_s2: Dict[str, List[float]] = {c: [] for c in cats}

    for fam_name, fam_data in families.items():
        if "error" in fam_data:
            continue
        s1_data = fam_data.get("system1", {}).get("results_by_category", {})
        s2_data = fam_data.get("system2", {}).get("results_by_category", {})
        for cat in cats:
            s1_acc = s1_data.get(cat, {}).get("accuracy", float("nan"))
            s2_acc = s2_data.get(cat, {}).get("accuracy", float("nan"))
            if not (math.isnan(s1_acc) or math.isnan(s2_acc)):
                cat_s1[cat].append(s1_acc)
                cat_s2[cat].append(s2_acc)

    def _mean(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    by_category: dict = {}
    for cat in cats:
        s1 = _mean(cat_s1[cat])
        s2 = _mean(cat_s2[cat])
        gap = (s2 - s1) if not (math.isnan(s1) or math.isnan(s2)) else float("nan")
        by_category[cat] = {
            "s1_accuracy": round(s1, 4),
            "s2_accuracy": round(s2, 4),
            "s2_s1_gap":   round(gap, 4),
        }

    return {
        "source": source,
        "format": "multi_model",
        "by_category": by_category,
        "raw": raw,
    }


def _parse_generic_format(raw: dict, source: str) -> dict:
    """Fallback: try to find accuracy fields in an unrecognised JSON."""
    # Attempt to surface something useful without crashing
    by_category: dict = {}
    for cat in ["intuitive", "analytical", "conflict"]:
        entry = raw.get(cat, {})
        s1 = entry.get("s1_accuracy", entry.get("system1_accuracy", float("nan")))
        s2 = entry.get("s2_accuracy", entry.get("system2_accuracy", float("nan")))
        gap_raw = entry.get("s2_s1_gap", entry.get("gap", None))
        if gap_raw is None:
            gap = (s2 - s1) if not (math.isnan(float(s1)) or math.isnan(float(s2))) else float("nan")
        else:
            gap = float(gap_raw)
        by_category[cat] = {
            "s1_accuracy": round(float(s1), 4) if not math.isnan(float(s1)) else float("nan"),
            "s2_accuracy": round(float(s2), 4) if not math.isnan(float(s2)) else float("nan"),
            "s2_s1_gap":   round(gap, 4) if not math.isnan(gap) else float("nan"),
        }
    return {
        "source": source,
        "format": "generic",
        "by_category": by_category,
        "raw": raw,
    }


def _merge_llm_results(results: List[dict], source: str) -> dict:
    """Average by_category values across multiple result dicts."""
    cats = ["intuitive", "analytical", "conflict"]
    fields = ["s1_accuracy", "s2_accuracy", "s2_s1_gap"]

    merged: dict = {}
    for cat in cats:
        field_vals: Dict[str, List[float]] = {f: [] for f in fields}
        for r in results:
            for field in fields:
                val = r["by_category"].get(cat, {}).get(field, float("nan"))
                if isinstance(val, float) and math.isnan(val):
                    continue
                field_vals[field].append(val)
        merged[cat] = {
            f: round(sum(v) / len(v), 4) if v else float("nan")
            for f, v in field_vals.items()
        }

    return {
        "source": source,
        "format": "merged",
        "n_files": len(results),
        "by_category": merged,
        "raw": [r["raw"] for r in results],
    }


# ===========================================================================
# 4. Pattern similarity
# ===========================================================================

def _spearman_rank_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Compute Spearman rank correlation for two equal-length lists.

    Returns None if either list contains NaN or if n < 2.
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    if any(math.isnan(v) for v in x + y):
        return None

    def _rank(vals: List[float]) -> List[float]:
        sorted_vals = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        for rank_idx, (orig_idx, _) in enumerate(sorted_vals):
            ranks[orig_idx] = rank_idx + 1.0
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    rho = 1 - (6 * d2) / (n * (n * n - 1))
    return round(rho, 4)


def compute_pattern_similarity(human_gaps: dict, llm_gaps: dict) -> dict:
    """
    Compare the structural pattern of S2-S1 gaps across task categories.

    Parameters
    ----------
    human_gaps : dict
        Output of :func:`aggregate_human_gaps_by_category`.
    llm_gaps : dict
        ``by_category`` sub-dict from :func:`load_llm_results`, or any dict
        with keys ``intuitive``, ``analytical``, ``conflict`` each containing
        a ``s2_s1_gap`` value.

    Returns
    -------
    dict
        Keys:

        * ``spearman_rho`` – Spearman rank correlation of the three gap values.
        * ``ordering_match`` – bool: does conflict > analytical > intuitive
          hold in both humans and LLMs?
        * ``human_ordering`` – actual rank order for humans.
        * ``llm_ordering``   – actual rank order for LLMs.
        * ``per_category``   – per-category gap comparison with match flag.
        * ``divergences``    – list of categories where the pattern clearly
          differs (one positive, one near-zero/negative, or large absolute
          difference).
        * ``summary_verdict`` – plain-English summary.
    """
    cats = ["intuitive", "analytical", "conflict"]

    human_vals: List[float] = []
    llm_vals:   List[float] = []

    per_cat: dict = {}
    for cat in cats:
        h_gap = human_gaps.get(cat, {}).get("mean_gap", float("nan"))
        # llm_gaps may come directly from by_category (with s2_s1_gap key)
        # or from a dict keyed by gap directly
        llm_entry = llm_gaps.get(cat, {})
        if isinstance(llm_entry, dict):
            l_gap = llm_entry.get("s2_s1_gap", llm_entry.get("mean_gap", float("nan")))
        else:
            l_gap = float(llm_entry) if llm_entry is not None else float("nan")

        human_vals.append(h_gap)
        llm_vals.append(l_gap)

        # Pattern match: both positive, both small, or both large
        if math.isnan(h_gap) or math.isnan(l_gap):
            match_flag = "unknown"
        else:
            abs_diff = abs(h_gap - l_gap)
            same_sign = (h_gap >= 0) == (l_gap >= 0)
            # Both "large" means > 0.15; both "small" means <= 0.10
            both_large = h_gap > 0.15 and l_gap > 0.15
            both_small = h_gap <= 0.10 and l_gap <= 0.10
            if both_large or both_small:
                match_flag = "consistent"
            elif same_sign and abs_diff < 0.20:
                match_flag = "partial"
            else:
                match_flag = "divergent"

        per_cat[cat] = {
            "human_gap": h_gap,
            "llm_gap":   l_gap,
            "pattern_match": match_flag,
        }

    # Spearman correlation
    rho = _spearman_rank_correlation(human_vals, llm_vals)

    # Ordering match: expected human order is conflict > analytical > intuitive
    def _rank_order(vals: List[float]) -> List[Tuple[str, float]]:
        """Return cats sorted descending by gap value."""
        paired = list(zip(cats, vals))
        paired.sort(key=lambda t: t[1] if not math.isnan(t[1]) else -999, reverse=True)
        return paired

    human_order = _rank_order(human_vals)
    llm_order   = _rank_order(llm_vals)

    human_ordering = [c for c, _ in human_order]
    llm_ordering   = [c for c, _ in llm_order]
    ordering_match = human_ordering == llm_ordering

    # Divergences: categories where the match flag is "divergent"
    divergences = [cat for cat, info in per_cat.items() if info["pattern_match"] == "divergent"]

    # Verdict
    if rho is not None and rho >= 0.9 and ordering_match and not divergences:
        verdict = (
            "Strong pattern similarity: LLM gap ordering matches humans across all categories "
            f"(Spearman rho = {rho})."
        )
    elif rho is not None and rho >= 0.6 and not divergences:
        verdict = (
            f"Moderate pattern similarity (Spearman rho = {rho}). "
            "Gap ordering partially matches; no strongly divergent categories."
        )
    elif divergences:
        div_str = ", ".join(divergences)
        verdict = (
            f"Pattern divergence detected in: {div_str}. "
            f"Spearman rho = {rho}. "
            "LLMs and humans show qualitatively different dual-process effects in these categories. "
            "This should be reported transparently."
        )
    else:
        verdict = (
            f"Weak or indeterminate pattern similarity (Spearman rho = {rho}). "
            "Gap ordering does not reliably match humans."
        )

    return {
        "spearman_rho": rho,
        "ordering_match": ordering_match,
        "human_ordering": human_ordering,
        "llm_ordering": llm_ordering,
        "per_category": per_cat,
        "divergences": divergences,
        "summary_verdict": verdict,
    }


# ===========================================================================
# 5. Comparison table
# ===========================================================================

def generate_comparison_table(human_gaps: dict, llm_gaps: dict) -> str:
    """
    Return a formatted plain-text comparison table.

    Parameters
    ----------
    human_gaps : dict
        Output of :func:`aggregate_human_gaps_by_category`.
    llm_gaps : dict
        ``by_category`` sub-dict from :func:`load_llm_results`.

    Returns
    -------
    str
        Multi-line string with the comparison table.
    """
    cats = [
        ("intuitive",  "Intuitive"),
        ("analytical", "Analytical"),
        ("conflict",   "Conflict"),
    ]

    similarity = compute_pattern_similarity(human_gaps, llm_gaps)
    per_cat = similarity["per_category"]

    # Column widths
    w_cat   = 12
    w_human = 17
    w_llm   = 15
    w_match = 22
    w_note  = 40

    sep = (
        f"{'':->{ w_cat + 2 }}"
        f"+{'':->{ w_human + 2 }}"
        f"+{'':->{ w_llm + 2 }}"
        f"+{'':->{ w_match + 2 }}"
        f"+{'':->{ w_note + 2 }}"
    )

    def _pct(val: float) -> str:
        if math.isnan(val):
            return "N/A"
        sign = "+" if val >= 0 else ""
        return f"{sign}{val * 100:.1f}%"

    def _match_symbol(flag: str) -> str:
        symbols = {
            "consistent": "yes (same magnitude)",
            "partial":    "partial",
            "divergent":  "NO (divergent)",
            "unknown":    "?",
        }
        return symbols.get(flag, flag)

    def _note_for(cat: str, info: dict) -> str:
        h = info["human_gap"]
        l = info["llm_gap"]
        if math.isnan(h) or math.isnan(l):
            return "insufficient data"
        diff = l - h
        if cat == "conflict":
            if l > 0.15 and h > 0.30:
                return "LLM gap much smaller than humans"
            elif abs(diff) < 0.05:
                return "gaps close"
        elif cat == "analytical":
            if l > 0.15 and h > 0.15:
                return "both show large S2 advantage"
        elif cat == "intuitive":
            if l <= 0.10 and h <= 0.10:
                return "both show small S2 advantage"
        return f"LLM {'+' if diff >= 0 else ''}{diff*100:.1f}pp vs human"

    lines: List[str] = []
    lines.append("")
    lines.append("=" * (w_cat + w_human + w_llm + w_match + w_note + 10))
    lines.append("HUMAN vs LLM DUAL-PROCESS PATTERN COMPARISON (S2-S1 accuracy gap)")
    lines.append("=" * (w_cat + w_human + w_llm + w_match + w_note + 10))
    lines.append(
        f"  {'Category':<{w_cat}}  {'Human S2-S1 Gap':>{w_human}}"
        f"  {'LLM S2-S1 Gap':>{w_llm}}"
        f"  {'Pattern Match':>{w_match}}"
        f"  {'Note':<{w_note}}"
    )
    lines.append(sep)

    for cat_key, cat_label in cats:
        info = per_cat.get(cat_key, {})
        h_gap = info.get("human_gap", float("nan"))
        l_gap = info.get("llm_gap",   float("nan"))
        flag  = info.get("pattern_match", "unknown")

        n_studies = human_gaps.get(cat_key, {}).get("n_studies", "?")
        h_str = f"{_pct(h_gap)} ({n_studies} studies)"
        l_str = _pct(l_gap)
        m_str = _match_symbol(flag)
        note  = _note_for(cat_key, info)

        lines.append(
            f"  {cat_label:<{w_cat}}  {h_str:>{w_human}}"
            f"  {l_str:>{w_llm}}"
            f"  {m_str:>{w_match}}"
            f"  {note:<{w_note}}"
        )

    lines.append(sep)
    lines.append("")
    lines.append(f"  Spearman rank correlation (human vs LLM gaps): {similarity['spearman_rho']}")
    lines.append(f"  Expected gap ordering (conflict>analytical>intuitive):")
    lines.append(f"    Human ordering : {' > '.join(similarity['human_ordering'])}")
    lines.append(f"    LLM ordering   : {' > '.join(similarity['llm_ordering'])}")
    lines.append(f"  Ordering matches: {similarity['ordering_match']}")
    if similarity["divergences"]:
        lines.append(f"  DIVERGENT categories: {', '.join(similarity['divergences'])}")
    lines.append("")
    lines.append(f"  Verdict: {similarity['summary_verdict']}")
    lines.append("")
    lines.append(
        "  NOTE: Absolute accuracy values are NOT compared — human and LLM task content differs.\n"
        "  Only the cross-category pattern of S2-S1 gap magnitudes is compared."
    )
    lines.append("")
    return "\n".join(lines)


# ===========================================================================
# 6. Full comparison run
# ===========================================================================

def run_comparison(
    llm_results_path: str,
    output_dir: str = str(_DEFAULT_OUTPUT_DIR),
    output_format: str = "both",
    benchmarks_path: Optional[str] = None,
) -> dict:
    """
    Run the full human-LLM comparison and optionally save results.

    Parameters
    ----------
    llm_results_path : str
        Path to the LLM experiment result JSON (or directory of JSONs).
    output_dir : str
        Directory to write output files.
    output_format : str
        One of ``"table"``, ``"json"``, ``"both"``.
    benchmarks_path : str or None
        Override path to human benchmarks JSON.

    Returns
    -------
    dict
        Full results dict with keys:
        ``human_gaps``, ``llm_gaps``, ``similarity``, ``table``.
    """
    # 1. Load data
    benchmarks = load_human_benchmarks(benchmarks_path)
    human_gaps = aggregate_human_gaps_by_category(benchmarks)
    llm_result = load_llm_results(llm_results_path)
    llm_by_cat = llm_result["by_category"]

    # 2. Compute similarity
    similarity = compute_pattern_similarity(human_gaps, llm_by_cat)

    # 3. Generate table
    table = generate_comparison_table(human_gaps, llm_by_cat)

    # 4. Assemble output
    full_output = {
        "human_gaps": human_gaps,
        "llm_result_source": llm_result["source"],
        "llm_result_format": llm_result["format"],
        "llm_gaps": {
            cat: {
                "s1_accuracy": llm_by_cat[cat].get("s1_accuracy"),
                "s2_accuracy": llm_by_cat[cat].get("s2_accuracy"),
                "s2_s1_gap":   llm_by_cat[cat].get("s2_s1_gap"),
            }
            for cat in ["intuitive", "analytical", "conflict"]
        },
        "similarity": similarity,
        "table": table,
    }

    # 5. Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if output_format in ("table", "both"):
        print(table)
        table_file = out_path / "human_comparison_table.txt"
        table_file.write_text(table, encoding="utf-8")
        print(f"Table saved to: {table_file}")

    if output_format in ("json", "both"):
        # Strip raw LLM data (can be very large) from saved JSON
        saveable = {k: v for k, v in full_output.items() if k != "table"}
        # Remove raw sub-dicts from human_gaps for cleaner output
        saveable_human_gaps: dict = {}
        for cat, info in human_gaps.items():
            saveable_human_gaps[cat] = {
                "mean_gap":  info["mean_gap"],
                "n_studies": info["n_studies"],
                "study_ids": [s["id"] for s in info.get("studies", [])],
            }
        saveable["human_gaps"] = saveable_human_gaps

        json_file = out_path / "human_comparison.json"
        with open(json_file, "w", encoding="utf-8") as fh:
            json.dump(saveable, fh, indent=2, ensure_ascii=False, default=str)
        print(f"JSON saved to: {json_file}")

    return full_output


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Systematic Human-LLM Dual-Process Comparison\n"
            "Compares S2-S1 accuracy gap patterns across task categories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm_results",
        type=str,
        required=True,
        help=(
            "Path to LLM experiment result JSON file or directory of JSONs. "
            "Supports main experiment, ablation, and multi-model formats."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["table", "json", "both"],
        help="Output format: table, json, or both (default: both).",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help=(
            "Path to human benchmarks JSON. "
            f"Default: {_DEFAULT_BENCHMARKS_PATH}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_comparison(
        llm_results_path=args.llm_results,
        output_dir=args.output,
        output_format=args.format,
        benchmarks_path=args.benchmarks,
    )
