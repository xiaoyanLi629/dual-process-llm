"""
BIBM 2026 Master Experiment Runner

Orchestrates all experiments needed for the paper:
1. Main experiment — 2x2x2 factorial ablation (8 conditions, core of the paper)
   - Includes random baseline and same-config variance baseline (C8 x2)
   - Novel conflict tasks are part of the main dataset (bibm_dataset.json)
2. Multi-model validation — 4 model families (OpenAI, DeepSeek, Qwen, LLaMA)
3. Post-hoc analysis — statistical analysis + human comparison (no API calls)

Key comparisons:
    - C5 vs C8: primary same-model comparison (gpt-4o, isolates T + prompting)
    - C1 vs C8: canonical S1 vs S2 comparison (all three factors vary)

Usage:
    # Run everything
    python -m src.experiments.run_all_bibm --all --n_samples 100

    # Run specific experiments
    python -m src.experiments.run_all_bibm --experiment main --n_samples 50
    python -m src.experiments.run_all_bibm --experiment multi_model --n_samples 50

    # Run only post-hoc analyses (no API calls)
    python -m src.experiments.run_all_bibm --analyze_only --input results/bibm_2026/

    # Dry run (1 sample per condition, verify setup)
    python -m src.experiments.run_all_bibm --all --dry_run

Estimated costs (at n_samples=100):
    - Main experiment (8 conditions + variance baseline): ~5,100 API calls, ~$22-35
    - Multi-model validation (4 families): ~2,400 API calls per family, ~$15-40 total
    Total: ~$37-75
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / "src"))


# ===========================================================================
# Sub-experiment wrappers
# ===========================================================================

def run_main_experiment(output_dir: Path, n_samples: int = 100, dry_run: bool = False) -> Optional[dict]:
    """
    Run the main 2x2x2 factorial ablation experiment (all 8 conditions).

    Includes:
    - Random baseline computed from the task set
    - Same-config variance baseline (C8 run twice on 50-item subset)
    - Novel conflict tasks are already embedded in bibm_dataset.json
    """
    from experiments.ablation_factorial import run_experiment as run_ablation_exp

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 (MAIN): 2x2x2 Factorial Ablation")
    print("  Primary comparison:   C5 vs C8 (same-model, gpt-4o)")
    print("  Secondary comparison: C1 vs C8 (canonical S1 vs S2)")
    print("=" * 60)

    main_out = output_dir / "main_experiment"
    main_out.mkdir(parents=True, exist_ok=True)

    return run_ablation_exp(
        n_samples=n_samples,
        output_dir=str(main_out.relative_to(project_root)),
        dry_run=dry_run,
        condition_ids=None,          # run all 8 conditions
        with_variance_baseline=True, # always include noise-floor baseline
    )


def run_multi_model(
    output_dir: Path,
    n_samples: int = 100,
    families: str = "openai,deepseek,qwen,llama",
    dry_run: bool = False,
) -> Optional[dict]:
    """Run multi-model validation across the specified model families."""
    from experiments.multi_model import run_experiment as run_mm_exp

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-Model Validation")
    print("=" * 60)

    mm_out = output_dir / "multi_model"
    mm_out.mkdir(parents=True, exist_ok=True)

    family_list = [f.strip() for f in families.split(",") if f.strip()]

    return run_mm_exp(
        families=family_list,
        n_samples=n_samples,
        output_dir=str(mm_out.relative_to(project_root)),
        dry_run=dry_run,
    )


# ===========================================================================
# Post-hoc analysis wrapper
# ===========================================================================

def run_analysis(output_dir: Path, input_dir: Optional[Path] = None) -> None:
    """
    Run all post-hoc analyses (no API calls required).

    Searches input_dir for result JSON files produced by prior runs:
    - If ablation results exist, runs statistical_analysis on them.
    - If any S1/S2 results (ablation, multi-model, or main experiment) exist,
      runs human_comparison on them.

    All outputs are saved under output_dir/analysis/.
    """
    from experiments.statistical_analysis import (
        full_statistical_report,
        factorial_anova_report,
        format_report_table,
        _load_results,
    )
    from experiments.human_comparison import run_comparison

    if input_dir is None:
        input_dir = output_dir

    analysis_out = output_dir / "analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("POST-HOC ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Collect candidate result files
    # ------------------------------------------------------------------
    # Priority order for statistical analysis:
    #   main_experiment (factorial ablation) > multi_model > paper_results
    # Note: novel_conflict tasks are embedded in main_experiment results.
    main_files  = sorted((input_dir / "main_experiment").glob("*.json")) if (input_dir / "main_experiment").is_dir() else []
    mm_files    = sorted((input_dir / "multi_model").glob("*.json"))     if (input_dir / "multi_model").is_dir()     else []
    # Also check for a flat paper_results.json in the input directory itself
    paper_files = sorted(input_dir.glob("paper_results*.json"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 2. Statistical analysis
    # ------------------------------------------------------------------
    print("\n--- Statistical Analysis ---")

    stat_candidates = main_files or paper_files
    if stat_candidates:
        stat_file = stat_candidates[-1]  # newest / highest priority
        print(f"  Loading results from: {stat_file}")

        try:
            results = _load_results(stat_file)

            report = full_statistical_report(results)
            anova  = factorial_anova_report(results) if "conditions" in results else {}

            table_str = format_report_table(report)
            print(table_str)

            # Save JSON report
            report_json = analysis_out / f"statistical_report_{timestamp}.json"
            combined = {"statistical_report": report, "factorial_anova": anova}
            with open(report_json, "w", encoding="utf-8") as fh:
                json.dump(combined, fh, indent=2, ensure_ascii=False, default=str)
            print(f"  Statistical report saved to: {report_json}")

            # Save text table
            report_txt = analysis_out / f"statistical_report_{timestamp}.txt"
            with open(report_txt, "w", encoding="utf-8") as fh:
                fh.write(table_str)
            print(f"  Text table saved to: {report_txt}")

        except Exception as exc:
            print(f"  WARNING: Statistical analysis failed: {exc}")
    else:
        print("  No result files found for statistical analysis. Skipping.")

    # ------------------------------------------------------------------
    # 3. Human comparison
    # ------------------------------------------------------------------
    print("\n--- Human-LLM Comparison ---")

    # Prefer main_experiment (contains all 8 conditions + baselines);
    # fall back to multi-model or paper results.
    hc_candidates: List[Path] = []
    for candidate_dir in [
        input_dir / "main_experiment",
        input_dir / "multi_model",
    ]:
        if candidate_dir.is_dir():
            files = sorted(candidate_dir.glob("*.json"))
            if files:
                hc_candidates.append(files[-1])  # newest file per sub-dir

    if not hc_candidates and paper_files:
        hc_candidates = [paper_files[-1]]

    if hc_candidates:
        hc_file = hc_candidates[0]  # highest priority source
        print(f"  Running human comparison against: {hc_file}")
        try:
            run_comparison(
                llm_results_path=str(hc_file),
                output_dir=str(analysis_out),
                output_format="both",
                benchmarks_path=None,  # use default data/human_benchmarks.json
            )
        except FileNotFoundError as exc:
            print(f"  WARNING: Human comparison skipped: {exc}")
        except Exception as exc:
            print(f"  WARNING: Human comparison failed: {exc}")
    else:
        print("  No suitable result files found for human comparison. Skipping.")

    print(f"\nAll analysis outputs saved to: {analysis_out}")


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BIBM 2026 Master Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments followed by post-hoc analyses.",
    )
    parser.add_argument(
        "--experiment", type=str,
        choices=["main", "multi_model"],
        help="Run a single named experiment: 'main' (factorial ablation) or 'multi_model' (validation).",
    )
    parser.add_argument(
        "--analyze_only", action="store_true",
        help="Run only post-hoc analyses (no API calls). Requires --input or --output with existing results.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of samples per task category per condition (default: 100).",
    )
    parser.add_argument(
        "--families", type=str, default="openai,deepseek,qwen,llama",
        help="Comma-separated model families for the multi-model experiment (default: all four).",
    )
    parser.add_argument(
        "--output", type=str, default="results/bibm_2026",
        help="Root output directory (default: results/bibm_2026).",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input directory containing existing results for --analyze_only mode. "
             "Defaults to --output if not specified.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run exactly 1 sample per condition to verify the pipeline without incurring real costs.",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root so the script is location-agnostic
    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist run configuration alongside results
    config = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    with open(output_dir / "run_config.json", "w") as fh:
        json.dump(config, fh, indent=2)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if args.analyze_only:
        input_dir = (project_root / args.input).resolve() if args.input else output_dir
        run_analysis(output_dir, input_dir)

    elif args.all:
        run_main_experiment(output_dir, args.n_samples, args.dry_run)
        run_multi_model(output_dir, args.n_samples, args.families, args.dry_run)
        run_analysis(output_dir)

    elif args.experiment:
        if args.experiment == "main":
            run_main_experiment(output_dir, args.n_samples, args.dry_run)
        elif args.experiment == "multi_model":
            run_multi_model(output_dir, args.n_samples, args.families, args.dry_run)
    else:
        parser.print_help()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
