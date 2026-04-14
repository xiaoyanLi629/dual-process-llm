"""
Prepare BIBM 2026 Dataset

Fixes task categorization by removing benchmarks that appear in multiple categories:
- ARC: removed (appears in both intuitive and analytical)
- MMLU: removed (appears in both intuitive and analytical)

Kept categories:
- Intuitive (S1): HellaSwag, PIQA, SIQA, CommonsenseQA, WinoGrande
- Analytical (S2): GSM8K, LogiQA
- Conflict: TruthfulQA + novel conflict tasks

Usage:
    python data/prepare_bibm_dataset.py
"""

import json
from pathlib import Path
from collections import Counter


def count_by_source(items):
    """Return a Counter of source -> count for a list of task dicts."""
    return Counter(item.get("source", "unknown") for item in items)


def format_source_counts(counter):
    """Format source counts as a readable string."""
    parts = [f"{src}: {cnt}" for src, cnt in sorted(counter.items(), key=lambda x: -x[1])]
    return ", ".join(parts)


def main():
    project_root = Path(__file__).parent.parent

    input_path = project_root / "data" / "processed" / "dual_process_dataset.json"
    novel_path = project_root / "data" / "novel_conflict_tasks.json"
    output_path = project_root / "data" / "processed" / "bibm_dataset.json"

    # --- Load existing dataset ---
    print("Loading dual_process_dataset.json ...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    orig_s1 = raw_data.get("system1_tasks", [])
    orig_s2 = raw_data.get("system2_tasks", [])
    orig_conflict = raw_data.get("conflict_tasks", [])

    orig_s1_counts = count_by_source(orig_s1)
    orig_s2_counts = count_by_source(orig_s2)

    print("\n=== BIBM Dataset Preparation ===")
    print("Original:")
    print(f"  system1_tasks: {len(orig_s1)} ({format_source_counts(orig_s1_counts)})")
    print(f"  system2_tasks: {len(orig_s2)} ({format_source_counts(orig_s2_counts)})")
    print(f"  conflict_tasks: {len(orig_conflict)}")

    # --- Define keep lists ---
    s1_keep_sources = {"HellaSwag", "PIQA", "SIQA", "CommonsenseQA", "WinoGrande"}
    s2_keep_sources = {"GSM8K", "LogiQA"}
    remove_sources = {"ARC", "MMLU"}

    # --- Filter system1 ---
    new_s1 = [item for item in orig_s1 if item.get("source", "") in s1_keep_sources]
    removed_arc_s1 = orig_s1_counts.get("ARC", 0)
    removed_mmlu_s1 = orig_s1_counts.get("MMLU", 0)

    # --- Filter system2 ---
    new_s2 = [item for item in orig_s2 if item.get("source", "") in s2_keep_sources]
    removed_arc_s2 = orig_s2_counts.get("ARC", 0)
    removed_mmlu_s2 = orig_s2_counts.get("MMLU", 0)

    # --- Conflict tasks: keep all existing + add novel ---
    # Keep all original conflict tasks (TruthfulQA)
    new_conflict = list(orig_conflict)

    # Load novel conflict tasks
    novel_count = 0
    if novel_path.exists():
        with open(novel_path, "r", encoding="utf-8") as f:
            novel_data = json.load(f)
        novel_tasks = novel_data.get("novel_conflict_tasks", [])

        # Check for duplicates by ID to avoid double-loading
        existing_ids = {item.get("id", "") for item in new_conflict}
        added_novel = []
        for task in novel_tasks:
            if task.get("id", "") not in existing_ids:
                added_novel.append(task)
                existing_ids.add(task.get("id", ""))

        new_conflict.extend(added_novel)
        novel_count = len(added_novel)
    else:
        print(f"Warning: novel conflict tasks file not found at {novel_path}")

    # --- Print removal/addition summary ---
    print("\nRemoved:")
    print(f"  ARC from system1: {removed_arc_s1} items")
    print(f"  MMLU from system1: {removed_mmlu_s1} items")
    print(f"  ARC from system2: {removed_arc_s2} items")
    print(f"  MMLU from system2: {removed_mmlu_s2} items")

    print("\nAdded:")
    print(f"  Novel conflict tasks: {novel_count} items")

    # --- Print final counts ---
    new_s1_counts = count_by_source(new_s1)
    new_s2_counts = count_by_source(new_s2)
    new_conflict_counts = count_by_source(new_conflict)
    total = len(new_s1) + len(new_s2) + len(new_conflict)

    print("\nFinal:")
    print(f"  system1_tasks: {len(new_s1)} ({format_source_counts(new_s1_counts)})")
    print(f"  system2_tasks: {len(new_s2)} ({format_source_counts(new_s2_counts)})")
    truthfulqa_count = new_conflict_counts.get("TruthfulQA", 0)
    print(f"  conflict_tasks: {len(new_conflict)} (TruthfulQA: {truthfulqa_count}, novel: {novel_count})")
    print(f"  Total: {total}")

    # --- Build output dataset ---
    output_dataset = {
        "system1_tasks": new_s1,
        "system2_tasks": new_s2,
        "conflict_tasks": new_conflict,
        "metadata": {
            "created": "2026-04-14",
            "description": "BIBM 2026 dataset with principled task categorization",
            "source_dataset": "dual_process_dataset.json",
            "excluded_sources": sorted(remove_sources),
            "exclusion_reason": (
                "These benchmarks span both intuitive and analytical demands, "
                "creating category overlap"
            ),
            "categorization_criteria": {
                "system1_intuitive": (
                    "Tasks solvable by pattern matching, associative retrieval, "
                    "or commonsense heuristics"
                ),
                "system2_analytical": (
                    "Tasks requiring explicit multi-step reasoning or rule application"
                ),
                "conflict": (
                    "Tasks where the intuitive response is systematically wrong"
                ),
            },
            "counts": {
                "system1_tasks": len(new_s1),
                "system2_tasks": len(new_s2),
                "conflict_tasks": len(new_conflict),
                "total": total,
            },
            "sources": {
                "system1": dict(new_s1_counts),
                "system2": dict(new_s2_counts),
                "conflict": dict(new_conflict_counts),
            },
        },
    }

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_dataset, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
