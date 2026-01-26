"""
Parallel Experiment Runner for Project 1: Dual Process Theory

This script uses async API calls for ~8-10x faster experiment execution.

Usage:
    python run_parallel_experiment.py --mode quick_test
    python run_parallel_experiment.py --mode full
    python run_parallel_experiment.py --mode paper --paper_mode full
"""

import argparse
import asyncio
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from evaluation.parallel_experiment_runner import ParallelExperimentRunner
from evaluation.metrics import MetricsCalculator
from tasks.task_loader import TaskLoader


# Human benchmarks (same as original)
HUMAN_BENCHMARKS = {
    "crt": {
        "intuitive_accuracy": 0.33,
        "reflective_accuracy": 0.67,
        "source": "Frederick (2005) - Cognitive Reflection Test",
        "n_subjects": 3428
    },
    "logical_reasoning": {
        "intuitive_accuracy": 0.55,
        "reflective_accuracy": 0.78,
        "source": "Evans et al. (1983)",
        "n_subjects": 256
    },
    "response_time": {
        "system1_mean_ms": 1500,
        "system2_mean_ms": 8000,
        "ratio": 5.3,
        "source": "Thompson et al. (2011) meta-analysis"
    }
}


def run_quick_test(n_samples: int = 10):
    """Quick test with parallel processing"""
    print("\n" + "="*60)
    print("QUICK TEST (Parallel) - Dual Process Theory")
    print("="*60)
    
    runner = ParallelExperimentRunner(
        system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
        system2_config={"model_name": "gpt-4o", "temperature": 0.2},
        s1_concurrent=20,
        s2_concurrent=10
    )
    
    results = runner.run_experiment(
        experiment_name="quick_test_parallel",
        task_categories=["conflict_tasks"],
        n_samples_per_category=n_samples,
        save_results=True
    )
    
    return results


def run_full_experiment(n_samples: int = 100):
    """Full experiment with parallel processing"""
    print("\n" + "="*60)
    print("FULL EXPERIMENT (Parallel) - Dual Process Theory")
    print("="*60)
    
    runner = ParallelExperimentRunner(
        system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
        system2_config={"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True},
        s1_concurrent=20,
        s2_concurrent=10
    )
    
    results = runner.run_experiment(
        experiment_name="full_experiment_parallel",
        task_categories=["system1_tasks", "system2_tasks", "conflict_tasks"],
        n_samples_per_category=n_samples,
        save_results=True
    )
    
    return results


class ParallelPaperExperimentRunner:
    """Paper-level experiments with parallel processing"""
    
    def __init__(self, output_dir: str = None):
        self.metrics = MetricsCalculator()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "results" / f"paper_parallel_{self.timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_runs_results = []
    
    def run_single_experiment(self,
                             experiment_name: str,
                             n_samples_per_category: Optional[int],
                             system1_config: Dict,
                             system2_config: Dict,
                             task_categories: List[str],
                             s1_concurrent: int = 20,
                             s2_concurrent: int = 10) -> Dict:
        """Run a single experiment with parallel processing"""
        
        experiment_output_dir = self.output_dir / experiment_name
        
        runner = ParallelExperimentRunner(
            output_dir=experiment_output_dir,
            system1_config=system1_config,
            system2_config=system2_config,
            s1_concurrent=s1_concurrent,
            s2_concurrent=s2_concurrent
        )
        
        results = runner.run_experiment(
            experiment_name=experiment_name,
            task_categories=task_categories,
            n_samples_per_category=n_samples_per_category,
            save_results=True
        )
        
        return results
    
    def run_multiple_rounds(self,
                           n_rounds: int = 3,
                           n_samples_per_category: int = 200,
                           task_categories: List[str] = None) -> Dict:
        """Run multiple rounds with parallel processing"""
        
        if task_categories is None:
            task_categories = ["system1_tasks", "system2_tasks", "conflict_tasks"]
        
        print("\n" + "="*70)
        print(f"PAPER-LEVEL EXPERIMENT (Parallel): {n_rounds} Rounds")
        print(f"Samples per category: {n_samples_per_category}")
        print("="*70)
        
        system1_config = {"model_name": "gpt-4o-mini", "temperature": 0.9}
        system2_config = {"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True}
        
        round_results = []
        
        for round_num in range(1, n_rounds + 1):
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}/{n_rounds}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            results = self.run_single_experiment(
                experiment_name=f"round_{round_num}",
                n_samples_per_category=n_samples_per_category,
                system1_config=system1_config,
                system2_config=system2_config,
                task_categories=task_categories
            )
            
            duration = time.time() - start_time
            results["round"] = round_num
            results["duration_seconds"] = duration
            
            round_results.append(results)
            
            print(f"\nRound {round_num} Summary:")
            print(f"  System 1 Accuracy: {results['system1_results']['accuracy']:.2%}")
            print(f"  System 2 Accuracy: {results['system2_results']['accuracy']:.2%}")
            print(f"  Duration: {duration/60:.1f} minutes")
        
        # Aggregate results
        aggregated = self._aggregate_rounds(round_results)
        
        return aggregated
    
    def _aggregate_rounds(self, round_results: List[Dict]) -> Dict:
        """Aggregate results across rounds"""
        
        s1_accuracies = [r["system1_results"]["accuracy"] for r in round_results]
        s2_accuracies = [r["system2_results"]["accuracy"] for r in round_results]
        s1_confidences = [r["system1_results"]["avg_confidence"] for r in round_results]
        s2_confidences = [r["system2_results"]["avg_confidence"] for r in round_results]
        s1_times = [r["system1_results"]["avg_response_time_ms"] for r in round_results]
        s2_times = [r["system2_results"]["avg_response_time_ms"] for r in round_results]
        s1_tokens = [r["system1_results"]["avg_tokens_used"] for r in round_results]
        s2_tokens = [r["system2_results"]["avg_tokens_used"] for r in round_results]
        
        def calc_stats(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                "sem": float(stats.sem(values)) if len(values) > 1 else 0,
                "ci_95": tuple(stats.t.interval(0.95, len(values)-1, 
                                                loc=np.mean(values), 
                                                scale=stats.sem(values))) if len(values) > 1 else (np.mean(values), np.mean(values)),
                "values": values
            }
        
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if n1+n2 > 2 else 1
            if pooled_std == 0:
                return 0.0
            return (np.mean(group2) - np.mean(group1)) / pooled_std
        
        # Statistical tests
        if len(s1_accuracies) > 1:
            t_stat, p_value = stats.ttest_rel(s1_accuracies, s2_accuracies)
        else:
            t_stat, p_value = 0, 1.0
        
        effect_size = cohens_d(s1_accuracies, s2_accuracies)
        
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            "n_rounds": len(round_results),
            "total_samples": sum(r["config"]["total_tasks"] for r in round_results),
            "system1_accuracy": calc_stats(s1_accuracies),
            "system2_accuracy": calc_stats(s2_accuracies),
            "system1_confidence": calc_stats(s1_confidences),
            "system2_confidence": calc_stats(s2_confidences),
            "system1_response_time_ms": calc_stats(s1_times),
            "system2_response_time_ms": calc_stats(s2_times),
            "system1_tokens": calc_stats(s1_tokens),
            "system2_tokens": calc_stats(s2_tokens),
            "statistical_analysis": {
                "paired_t_test": {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant_at_05": p_value < 0.05,
                    "significant_at_01": p_value < 0.01,
                    "significant_at_001": p_value < 0.001
                },
                "effect_size": {
                    "cohens_d": float(effect_size),
                    "interpretation": effect_interpretation
                },
                "accuracy_difference": {
                    "mean": float(np.mean(s2_accuracies) - np.mean(s1_accuracies)),
                    "std": float(np.std([s2-s1 for s1, s2 in zip(s1_accuracies, s2_accuracies)], ddof=1)) if len(s1_accuracies) > 1 else 0
                }
            },
            "round_details": round_results
        }
    
    def compare_with_human_benchmarks(self, experiment_results: Dict) -> Dict:
        """Compare with human benchmarks"""
        
        print("\n" + "="*70)
        print("HUMAN BENCHMARK COMPARISON")
        print("="*70)
        
        s1_acc = experiment_results["system1_accuracy"]["mean"]
        s2_acc = experiment_results["system2_accuracy"]["mean"]
        
        comparisons = {
            "crt": {
                "human_intuitive": HUMAN_BENCHMARKS["crt"]["intuitive_accuracy"],
                "human_reflective": HUMAN_BENCHMARKS["crt"]["reflective_accuracy"],
                "llm_system1": s1_acc,
                "llm_system2": s2_acc,
                "source": HUMAN_BENCHMARKS["crt"]["source"]
            },
            "response_time": {
                "human_ratio": HUMAN_BENCHMARKS["response_time"]["ratio"],
                "llm_s1_mean_ms": experiment_results["system1_response_time_ms"]["mean"],
                "llm_s2_mean_ms": experiment_results["system2_response_time_ms"]["mean"],
                "llm_ratio": experiment_results["system2_response_time_ms"]["mean"] / 
                            max(experiment_results["system1_response_time_ms"]["mean"], 1),
                "source": HUMAN_BENCHMARKS["response_time"]["source"]
            }
        }
        
        print(f"\n--- CRT Comparison ---")
        print(f"  Human Intuitive: {HUMAN_BENCHMARKS['crt']['intuitive_accuracy']:.1%}")
        print(f"  Human Reflective: {HUMAN_BENCHMARKS['crt']['reflective_accuracy']:.1%}")
        print(f"  LLM System 1: {s1_acc:.1%}")
        print(f"  LLM System 2: {s2_acc:.1%}")
        
        return comparisons
    
    def run_category_breakdown(self, n_samples: int = 100, n_rounds: int = 3) -> Dict:
        """Run category breakdown analysis"""
        
        print("\n" + "="*70)
        print("CATEGORY BREAKDOWN ANALYSIS (Parallel)")
        print("="*70)
        
        categories = {
            "system1_tasks": "Intuitive tasks",
            "system2_tasks": "Analytical tasks",
            "conflict_tasks": "Conflict tasks"
        }
        
        category_results = {}
        
        for category, description in categories.items():
            print(f"\n--- {category}: {description} ---")
            
            all_s1_acc = []
            all_s2_acc = []
            
            for round_num in range(1, n_rounds + 1):
                print(f"  Round {round_num}/{n_rounds}...")
                
                results = self.run_single_experiment(
                    experiment_name=f"category_{category}_round_{round_num}",
                    n_samples_per_category=n_samples,
                    system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
                    system2_config={"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True},
                    task_categories=[category]
                )
                
                all_s1_acc.append(results["system1_results"]["accuracy"])
                all_s2_acc.append(results["system2_results"]["accuracy"])
            
            # Calculate statistics
            if len(all_s1_acc) > 1:
                t_stat, p_value = stats.ttest_rel(all_s1_acc, all_s2_acc)
            else:
                t_stat, p_value = 0, 1.0
            
            effect_size = self.metrics.calculate_effect_size(all_s1_acc, all_s2_acc)
            
            category_results[category] = {
                "description": description,
                "system1_accuracy": {
                    "mean": float(np.mean(all_s1_acc)),
                    "std": float(np.std(all_s1_acc, ddof=1)) if len(all_s1_acc) > 1 else 0,
                    "values": all_s1_acc
                },
                "system2_accuracy": {
                    "mean": float(np.mean(all_s2_acc)),
                    "std": float(np.std(all_s2_acc, ddof=1)) if len(all_s2_acc) > 1 else 0,
                    "values": all_s2_acc
                },
                "difference": {
                    "mean": float(np.mean(all_s2_acc) - np.mean(all_s1_acc)),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": effect_size
                }
            }
            
            print(f"  S1: {np.mean(all_s1_acc):.1%} ± {np.std(all_s1_acc, ddof=1):.1%}" if len(all_s1_acc) > 1 else f"  S1: {np.mean(all_s1_acc):.1%}")
            print(f"  S2: {np.mean(all_s2_acc):.1%} ± {np.std(all_s2_acc, ddof=1):.1%}" if len(all_s2_acc) > 1 else f"  S2: {np.mean(all_s2_acc):.1%}")
        
        return category_results
    
    def generate_paper_figures(self, results: Dict, human_comparison: Dict, category_results: Dict):
        """Generate paper figures"""
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300
        })
        
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Figure 1: Main comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(2)
        s1_mean = results["system1_accuracy"]["mean"]
        s2_mean = results["system2_accuracy"]["mean"]
        s1_ci = results["system1_accuracy"]["ci_95"]
        s2_ci = results["system2_accuracy"]["ci_95"]
        
        bars = ax.bar(x, [s1_mean, s2_mean],
                     yerr=[[s1_mean - s1_ci[0], s2_mean - s2_ci[0]],
                           [s1_ci[1] - s1_mean, s2_ci[1] - s2_mean]],
                     capsize=5, color=['#E64B35', '#4DBBD5'], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Dual Process Theory: System Comparison\n(Mean ± 95% CI)')
        ax.set_ylim(0, 1)
        
        p_val = results["statistical_analysis"]["paired_t_test"]["p_value"]
        effect_d = results["statistical_analysis"]["effect_size"]["cohens_d"]
        
        sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        ax.annotate(f'{sig_text}\np={p_val:.3f}\nd={effect_d:.2f}',
                   xy=(0.5, max(s1_mean, s2_mean) + 0.08),
                   ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig1_main_comparison.png", dpi=300)
        plt.close()
        
        # Figure 2: Category breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cats = list(category_results.keys())
        x = np.arange(len(cats))
        width = 0.35
        
        s1_means = [category_results[c]["system1_accuracy"]["mean"] for c in cats]
        s2_means = [category_results[c]["system2_accuracy"]["mean"] for c in cats]
        s1_stds = [category_results[c]["system1_accuracy"]["std"] for c in cats]
        s2_stds = [category_results[c]["system2_accuracy"]["std"] for c in cats]
        
        ax.bar(x - width/2, s1_means, width, yerr=s1_stds,
              label='System 1', color='#E64B35', alpha=0.8, capsize=3)
        ax.bar(x + width/2, s2_means, width, yerr=s2_stds,
              label='System 2', color='#4DBBD5', alpha=0.8, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Intuitive\nTasks', 'Analytical\nTasks', 'Conflict\nTasks'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance by Task Category')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig2_category_breakdown.png", dpi=300)
        plt.close()
        
        print(f"\nFigures saved to {figures_dir}")
    
    def save_results(self, results: Dict, human_comparison: Dict, category_results: Dict):
        """Save all results"""
        
        full_results = {
            "timestamp": self.timestamp,
            "main_experiment": results,
            "human_comparison": human_comparison,
            "category_breakdown": category_results
        }
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(i) for i in obj)
            return obj
        
        full_results = convert_numpy(full_results)
        
        output_file = self.output_dir / "paper_results.json"
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")


def run_paper_experiments_parallel(mode: str = "full"):
    """Run paper-level experiments with parallel processing"""
    
    runner = ParallelPaperExperimentRunner()
    
    if mode == "quick_validation":
        n_rounds = 2
        n_samples = 50
        category_samples = 30
    elif mode == "medium":
        n_rounds = 3
        n_samples = 100
        category_samples = 50
    else:  # full
        n_rounds = 3
        n_samples = 200
        category_samples = 100
    
    print("\n" + "="*70)
    print(f"PAPER-LEVEL EXPERIMENTS (Parallel) - Mode: {mode}")
    print(f"Rounds: {n_rounds}, Samples/category: {n_samples}")
    print("Expected speedup: ~8-10x compared to sequential")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Main experiment
    print("\n[1/4] Running main experiment...")
    main_results = runner.run_multiple_rounds(
        n_rounds=n_rounds,
        n_samples_per_category=n_samples
    )
    
    # 2. Human comparison
    print("\n[2/4] Comparing with human benchmarks...")
    human_comparison = runner.compare_with_human_benchmarks(main_results)
    
    # 3. Category breakdown
    print("\n[3/4] Running category breakdown...")
    category_results = runner.run_category_breakdown(
        n_samples=category_samples,
        n_rounds=n_rounds
    )
    
    # 4. Generate outputs
    print("\n[4/4] Generating outputs...")
    runner.generate_paper_figures(main_results, human_comparison, category_results)
    runner.save_results(main_results, human_comparison, category_results)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nTotal Duration: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results Directory: {runner.output_dir}")
    
    print("\n--- Key Findings ---")
    print(f"System 1 Accuracy: {main_results['system1_accuracy']['mean']:.1%} ± {main_results['system1_accuracy']['std']:.1%}")
    print(f"System 2 Accuracy: {main_results['system2_accuracy']['mean']:.1%} ± {main_results['system2_accuracy']['std']:.1%}")
    print(f"Difference: {main_results['statistical_analysis']['accuracy_difference']['mean']:.1%}")
    print(f"p-value: {main_results['statistical_analysis']['paired_t_test']['p_value']:.4f}")
    print(f"Cohen's d: {main_results['statistical_analysis']['effect_size']['cohens_d']:.2f}")
    
    return main_results, human_comparison, category_results


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Dual Process Theory Experiments (~8-10x faster)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test:
    python run_parallel_experiment.py --mode quick_test --n_samples 10

  Full experiment:
    python run_parallel_experiment.py --mode full --n_samples 100

  Paper-level (recommended):
    python run_parallel_experiment.py --mode paper --paper_mode quick_validation
    python run_parallel_experiment.py --mode paper --paper_mode full
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick_test",
        choices=["quick_test", "full", "paper"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Samples per category (for quick_test and full modes)"
    )
    parser.add_argument(
        "--paper_mode",
        type=str,
        default="full",
        choices=["quick_validation", "medium", "full"],
        help="Paper experiment mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick_test":
        run_quick_test(args.n_samples)
    elif args.mode == "full":
        run_full_experiment(args.n_samples)
    elif args.mode == "paper":
        run_paper_experiments_parallel(args.paper_mode)


if __name__ == "__main__":
    main()
