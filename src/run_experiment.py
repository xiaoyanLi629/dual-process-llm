"""
Unified Experiment Runner for Project 1: Dual Process Theory

This script provides both basic experiments and paper-level experiments:

Basic Experiments:
    python run_experiment.py --experiment quick_test --n_samples 10
    python run_experiment.py --experiment full --n_samples 100
    python run_experiment.py --experiment ablation --n_samples 50
    python run_experiment.py --experiment category --n_samples 50
    python run_experiment.py --experiment stepped --n_samples 30

Paper-Level Experiments (ICLR/NeurIPS Standard):
    python run_experiment.py --experiment paper --mode quick_validation
    python run_experiment.py --experiment paper --mode medium
    python run_experiment.py --experiment paper --mode full
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from tqdm import tqdm

# Add project root to path for api_config access
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from evaluation.experiment_runner import ExperimentRunner
from evaluation.metrics import MetricsCalculator
from tasks.task_loader import TaskLoader


# ============================================================================
# Human Benchmark Data from Cognitive Psychology Literature
# ============================================================================

HUMAN_BENCHMARKS = {
    "crt": {
        # Cognitive Reflection Test - Frederick (2005)
        "intuitive_accuracy": 0.33,  # Proportion giving intuitive (wrong) answer
        "reflective_accuracy": 0.67,  # Proportion giving correct answer after reflection
        "source": "Frederick (2005) - Cognitive Reflection Test",
        "n_subjects": 3428
    },
    "base_rate": {
        # Base rate neglect - Kahneman & Tversky (1973)
        "intuitive_accuracy": 0.15,  # Using base rates correctly
        "reflective_accuracy": 0.45,  # After explicit instruction
        "source": "Kahneman & Tversky (1973)",
        "n_subjects": 152
    },
    "logical_reasoning": {
        # Syllogistic reasoning - Evans et al. (1983)
        "intuitive_accuracy": 0.55,  # Belief-based responding
        "reflective_accuracy": 0.78,  # Logic-based responding
        "source": "Evans et al. (1983)",
        "n_subjects": 256
    },
    "framing_effect": {
        # Framing effects - Tversky & Kahneman (1981)
        "susceptibility": 0.72,  # Proportion affected by framing
        "resistance_after_reflection": 0.45,  # After deliberation
        "source": "Tversky & Kahneman (1981)",
        "n_subjects": 307
    },
    "response_time": {
        # Dual process response times - meta-analysis
        "system1_mean_ms": 1500,
        "system1_std_ms": 500,
        "system2_mean_ms": 8000,
        "system2_std_ms": 3000,
        "ratio": 5.3,
        "source": "Thompson et al. (2011) meta-analysis"
    }
}


# ============================================================================
# Basic Experiment Functions
# ============================================================================

def run_quick_test(n_samples: int = 10):
    """Run quick test"""
    print("\n" + "="*60)
    print("QUICK TEST - Dual Process Theory")
    print("="*60)
    
    runner = ExperimentRunner(
        system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
        system2_config={"model_name": "gpt-4o", "temperature": 0.2}
    )
    
    results = runner.run_experiment(
        experiment_name="quick_test",
        task_categories=["conflict_tasks"],  # Test conflict tasks only
        n_samples_per_category=n_samples,
        save_results=True
    )
    
    return results


def run_full_experiment(n_samples: int = 100):
    """Run full experiment"""
    print("\n" + "="*60)
    print("FULL EXPERIMENT - Dual Process Theory")
    print("="*60)
    
    runner = ExperimentRunner(
        system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
        system2_config={"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True}
    )
    
    results = runner.run_experiment(
        experiment_name="full_experiment",
        task_categories=["system1_tasks", "system2_tasks", "conflict_tasks"],
        n_samples_per_category=n_samples,
        save_results=True
    )
    
    return results


def run_ablation_study(n_samples: int = 50):
    """Run ablation study"""
    print("\n" + "="*60)
    print("ABLATION STUDY - Dual Process Theory")
    print("="*60)
    
    runner = ExperimentRunner()
    
    # Define ablation configurations
    ablation_configs = [
        # Baseline: Same model, different temperature
        {
            "name": "same_model_diff_temp",
            "system1": {"model_name": "gpt-4o", "temperature": 0.9},
            "system2": {"model_name": "gpt-4o", "temperature": 0.2}
        },
        # Different model, same temperature
        {
            "name": "diff_model_same_temp",
            "system1": {"model_name": "gpt-4o-mini", "temperature": 0.5},
            "system2": {"model_name": "gpt-4o", "temperature": 0.5}
        },
        # Full difference
        {
            "name": "full_difference",
            "system1": {"model_name": "gpt-4o-mini", "temperature": 0.9},
            "system2": {"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True}
        },
        # CoT only difference
        {
            "name": "cot_only",
            "system1": {"model_name": "gpt-4o", "temperature": 0.3},
            "system2": {"model_name": "gpt-4o", "temperature": 0.3, "use_cot": True}
        }
    ]
    
    results = runner.run_ablation_study(
        base_experiment_name="ablation",
        ablation_configs=ablation_configs,
        n_samples=n_samples
    )
    
    return results


def run_category_analysis(n_samples: int = 100):
    """Run detailed analysis by category"""
    print("\n" + "="*60)
    print("CATEGORY ANALYSIS - Dual Process Theory")
    print("="*60)
    
    runner = ExperimentRunner(
        system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
        system2_config={"model_name": "gpt-4o", "temperature": 0.2}
    )
    
    # Load task statistics
    loader = TaskLoader()
    loader.load()
    stats = loader.get_statistics()
    
    print("\nDataset Statistics:")
    for category, info in stats["categories"].items():
        print(f"\n{category}:")
        print(f"  Total: {info['count']}")
        print(f"  Sources: {info['sources']}")
    
    # Run experiment by data source
    sources_to_test = ["TruthfulQA", "GSM8K", "LogiQA", "CommonsenseQA", "PIQA"]
    
    all_results = {}
    for source in sources_to_test:
        print(f"\n--- Testing {source} ---")
        
        # Get tasks from this source
        tasks = []
        for category in ["system1_tasks", "system2_tasks", "conflict_tasks"]:
            category_tasks = loader.get_tasks(category, source_filter=source, n=n_samples)
            tasks.extend(category_tasks)
        
        if not tasks:
            print(f"No tasks found for {source}")
            continue
        
        results = runner.run_experiment(
            experiment_name=f"category_{source}",
            task_categories=["system1_tasks", "system2_tasks", "conflict_tasks"],
            n_samples_per_category=min(n_samples, len(tasks)),
            save_results=True
        )
        
        all_results[source] = results
        runner.reset()
    
    return all_results


def run_stepped_experiment(n_samples: int = 50):
    """Run stepped experiment to isolate variable effects
    
    This experiment systematically varies one factor at a time to determine
    which factors contribute most to the System 1 vs System 2 difference.
    
    Factors tested:
    1. Model size (gpt-4o-mini vs gpt-4o)
    2. Temperature (0.9 vs 0.2)
    3. Prompt style (zero-shot vs CoT)
    4. Max tokens (100 vs 1000)
    """
    print("\n" + "="*60)
    print("STEPPED EXPERIMENT - Isolating Variable Effects")
    print("="*60)
    
    # Define stepped configurations
    # Each step changes only ONE variable from baseline
    stepped_configs = [
        {
            "name": "baseline",
            "description": "Baseline: Same model, same params",
            "system1": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 500},
            "system2": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 500}
        },
        {
            "name": "model_only",
            "description": "Change: Model size only (mini vs full)",
            "system1": {"model_name": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 500},
            "system2": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 500}
        },
        {
            "name": "temperature_only",
            "description": "Change: Temperature only (0.9 vs 0.2)",
            "system1": {"model_name": "gpt-4o", "temperature": 0.9, "max_tokens": 500},
            "system2": {"model_name": "gpt-4o", "temperature": 0.2, "max_tokens": 500}
        },
        {
            "name": "cot_only",
            "description": "Change: CoT prompting only",
            "system1": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 500},
            "system2": {"model_name": "gpt-4o", "temperature": 0.5, "use_cot": True, "max_tokens": 500}
        },
        {
            "name": "tokens_only",
            "description": "Change: Max tokens only (100 vs 1000)",
            "system1": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 100},
            "system2": {"model_name": "gpt-4o", "temperature": 0.5, "max_tokens": 1000}
        },
        {
            "name": "model_and_temp",
            "description": "Change: Model + Temperature",
            "system1": {"model_name": "gpt-4o-mini", "temperature": 0.9, "max_tokens": 500},
            "system2": {"model_name": "gpt-4o", "temperature": 0.2, "max_tokens": 500}
        },
        {
            "name": "full_difference",
            "description": "Full S1/S2: All factors different",
            "system1": {"model_name": "gpt-4o-mini", "temperature": 0.9, "max_tokens": 100},
            "system2": {"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True, "max_tokens": 1000}
        }
    ]
    
    all_results = []
    
    for config in stepped_configs:
        print(f"\n--- {config['name']}: {config['description']} ---")
        
        runner = ExperimentRunner(
            system1_config=config["system1"],
            system2_config=config["system2"]
        )
        
        results = runner.run_experiment(
            experiment_name=f"stepped_{config['name']}",
            task_categories=["conflict_tasks"],  # Focus on conflict tasks
            n_samples_per_category=n_samples,
            save_results=False  # Save all at end
        )
        
        step_result = {
            "config_name": config["name"],
            "description": config["description"],
            "system1_config": config["system1"],
            "system2_config": config["system2"],
            "system1_accuracy": results["system1_results"]["accuracy"],
            "system2_accuracy": results["system2_results"]["accuracy"],
            "accuracy_difference": results["comparison"]["accuracy_diff"],
            "response_time_ratio": results["comparison"]["response_time_ratio"],
            "tokens_ratio": results["comparison"]["tokens_ratio"]
        }
        
        all_results.append(step_result)
        
        print(f"  S1 Accuracy: {step_result['system1_accuracy']:.2%}")
        print(f"  S2 Accuracy: {step_result['system2_accuracy']:.2%}")
        print(f"  Difference: {step_result['accuracy_difference']:+.2%}")
    
    # Analyze which factors contribute most
    print("\n" + "="*60)
    print("FACTOR CONTRIBUTION ANALYSIS")
    print("="*60)
    
    baseline_diff = all_results[0]["accuracy_difference"]
    
    factor_contributions = {}
    for result in all_results[1:]:
        contribution = result["accuracy_difference"] - baseline_diff
        factor_contributions[result["config_name"]] = {
            "contribution": contribution,
            "description": result["description"]
        }
        print(f"\n{result['config_name']}:")
        print(f"  {result['description']}")
        print(f"  Contribution to S1/S2 difference: {contribution:+.2%}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"stepped_experiment_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "experiment": "stepped_variable_isolation",
            "timestamp": timestamp,
            "n_samples": n_samples,
            "results": all_results,
            "factor_contributions": factor_contributions
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return all_results


# ============================================================================
# Paper-Level Experiment Class (ICLR/NeurIPS Standard)
# ============================================================================

class PaperExperimentRunner:
    """Runner for paper-level experiments with full statistical rigor"""
    
    def __init__(self, output_dir: str = None):
        self.metrics = MetricsCalculator()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent.parent / "results" / f"paper_{self.timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store all results for cross-run analysis
        self.all_runs_results = []
    
    def run_single_experiment(self, 
                             experiment_name: str,
                             n_samples_per_category: int,
                             system1_config: Dict,
                             system2_config: Dict,
                             task_categories: List[str],
                             save_results: bool = False) -> Dict:
        """Run a single experiment iteration
        
        Args:
            experiment_name: Name of the experiment
            n_samples_per_category: Number of samples per category
            system1_config: System 1 configuration
            system2_config: System 2 configuration
            task_categories: List of task categories
            save_results: Whether to save individual round results
        """
        
        # Create subdirectory for this experiment within the main output directory
        experiment_output_dir = self.output_dir / experiment_name
        
        runner = ExperimentRunner(
            output_dir=experiment_output_dir,
            system1_config=system1_config,
            system2_config=system2_config
        )
        
        results = runner.run_experiment(
            experiment_name=experiment_name,
            task_categories=task_categories,
            n_samples_per_category=n_samples_per_category,
            save_results=save_results
        )
        
        return results
    
    def run_multiple_rounds(self,
                           n_rounds: int = 3,
                           n_samples_per_category: int = 200,
                           task_categories: List[str] = None,
                           use_all_data: bool = False) -> Dict:
        """Run multiple independent rounds for statistical reliability
        
        Args:
            n_rounds: Number of independent rounds
            n_samples_per_category: Samples per category per round (ignored if use_all_data=True)
            task_categories: List of task categories to use
            use_all_data: If True, use all available data in each round
        """
        
        if task_categories is None:
            task_categories = ["system1_tasks", "system2_tasks", "conflict_tasks"]
        
        # Get dataset sizes if using all data
        if use_all_data:
            loader = TaskLoader()
            loader.load()
            stats = loader.get_statistics()
            
            # Get the minimum category size to ensure balanced sampling
            category_sizes = {cat: stats["categories"][cat]["count"] for cat in task_categories}
            
            print("\n" + "="*70)
            print(f"PAPER-LEVEL EXPERIMENT: {n_rounds} Independent Rounds (FULL DATA)")
            print("Using ALL available data:")
            for cat, size in category_sizes.items():
                print(f"  {cat}: {size:,} samples")
            total_samples = sum(category_sizes.values()) * n_rounds
            print(f"Total samples across all rounds: {total_samples:,}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print(f"PAPER-LEVEL EXPERIMENT: {n_rounds} Independent Rounds")
            print(f"Samples per category per round: {n_samples_per_category}")
            print(f"Total samples: {n_rounds * n_samples_per_category * len(task_categories)}")
            print("="*70)
        
        system1_config = {"model_name": "gpt-4o-mini", "temperature": 0.9}
        system2_config = {"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True}
        
        round_results = []
        
        for round_num in range(1, n_rounds + 1):
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}/{n_rounds}")
            print(f"{'='*50}")
            
            start_time = time.time()
            
            # Determine samples per category for this round
            if use_all_data:
                # Use all data, but shuffle differently each round
                samples_this_round = None  # Signal to use all
            else:
                samples_this_round = n_samples_per_category
            
            results = self.run_single_experiment(
                experiment_name=f"round_{round_num}",
                n_samples_per_category=samples_this_round,
                system1_config=system1_config,
                system2_config=system2_config,
                task_categories=task_categories
            )
            
            duration = time.time() - start_time
            results["round"] = round_num
            results["duration_seconds"] = duration
            
            round_results.append(results)
            
            # Print round summary
            print(f"\nRound {round_num} Summary:")
            print(f"  System 1 Accuracy: {results['system1_results']['accuracy']:.2%}")
            print(f"  System 2 Accuracy: {results['system2_results']['accuracy']:.2%}")
            print(f"  Duration: {duration/60:.1f} minutes")
        
        # Aggregate results across rounds
        aggregated = self._aggregate_rounds(round_results)
        
        return aggregated
    
    def _aggregate_rounds(self, round_results: List[Dict]) -> Dict:
        """Aggregate results across multiple rounds with statistics"""
        
        # Extract metrics from each round
        s1_accuracies = [r["system1_results"]["accuracy"] for r in round_results]
        s2_accuracies = [r["system2_results"]["accuracy"] for r in round_results]
        s1_confidences = [r["system1_results"]["avg_confidence"] for r in round_results]
        s2_confidences = [r["system2_results"]["avg_confidence"] for r in round_results]
        s1_times = [r["system1_results"]["avg_response_time_ms"] for r in round_results]
        s2_times = [r["system2_results"]["avg_response_time_ms"] for r in round_results]
        s1_tokens = [r["system1_results"]["avg_tokens_used"] for r in round_results]
        s2_tokens = [r["system2_results"]["avg_tokens_used"] for r in round_results]
        
        # Calculate statistics
        def calc_stats(values):
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "sem": float(stats.sem(values)),
                "ci_95": tuple(stats.t.interval(0.95, len(values)-1, 
                                                loc=np.mean(values), 
                                                scale=stats.sem(values))),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values
            }
        
        # Effect size calculation (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            if pooled_std == 0:
                return 0.0
            return (np.mean(group2) - np.mean(group1)) / pooled_std
        
        # Statistical tests
        t_stat, p_value = stats.ttest_rel(s1_accuracies, s2_accuracies)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(s1_accuracies, s2_accuracies)
        except:
            wilcoxon_stat, wilcoxon_p = None, None
        
        effect_size = cohens_d(s1_accuracies, s2_accuracies)
        
        # Interpret effect size
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_interpretation = "negligible"
        elif abs_d < 0.5:
            effect_interpretation = "small"
        elif abs_d < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        aggregated = {
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
                "wilcoxon_test": {
                    "statistic": float(wilcoxon_stat) if wilcoxon_stat else None,
                    "p_value": float(wilcoxon_p) if wilcoxon_p else None
                },
                "effect_size": {
                    "cohens_d": float(effect_size),
                    "interpretation": effect_interpretation
                },
                "accuracy_difference": {
                    "mean": float(np.mean(s2_accuracies) - np.mean(s1_accuracies)),
                    "std": float(np.std([s2-s1 for s1, s2 in zip(s1_accuracies, s2_accuracies)], ddof=1))
                }
            },
            "round_details": round_results
        }
        
        return aggregated
    
    def compare_with_human_benchmarks(self, experiment_results: Dict) -> Dict:
        """Compare LLM results with human cognitive benchmarks"""
        
        print("\n" + "="*70)
        print("HUMAN BENCHMARK COMPARISON")
        print("="*70)
        
        comparisons = {}
        
        # System 1 vs Human intuitive responses
        s1_acc = experiment_results["system1_accuracy"]["mean"]
        s2_acc = experiment_results["system2_accuracy"]["mean"]
        
        # CRT comparison
        human_crt = HUMAN_BENCHMARKS["crt"]
        comparisons["crt"] = {
            "human_intuitive": human_crt["intuitive_accuracy"],
            "human_reflective": human_crt["reflective_accuracy"],
            "llm_system1": s1_acc,
            "llm_system2": s2_acc,
            "s1_vs_human_intuitive": s1_acc - human_crt["intuitive_accuracy"],
            "s2_vs_human_reflective": s2_acc - human_crt["reflective_accuracy"],
            "source": human_crt["source"]
        }
        
        # Logical reasoning comparison
        human_logic = HUMAN_BENCHMARKS["logical_reasoning"]
        comparisons["logical_reasoning"] = {
            "human_intuitive": human_logic["intuitive_accuracy"],
            "human_reflective": human_logic["reflective_accuracy"],
            "llm_system1": s1_acc,
            "llm_system2": s2_acc,
            "s1_vs_human_intuitive": s1_acc - human_logic["intuitive_accuracy"],
            "s2_vs_human_reflective": s2_acc - human_logic["reflective_accuracy"],
            "source": human_logic["source"]
        }
        
        # Response time comparison
        human_rt = HUMAN_BENCHMARKS["response_time"]
        s1_rt = experiment_results["system1_response_time_ms"]["mean"]
        s2_rt = experiment_results["system2_response_time_ms"]["mean"]
        llm_ratio = s2_rt / s1_rt if s1_rt > 0 else 0
        
        comparisons["response_time"] = {
            "human_s1_mean_ms": human_rt["system1_mean_ms"],
            "human_s2_mean_ms": human_rt["system2_mean_ms"],
            "human_ratio": human_rt["ratio"],
            "llm_s1_mean_ms": s1_rt,
            "llm_s2_mean_ms": s2_rt,
            "llm_ratio": llm_ratio,
            "ratio_similarity": 1 - abs(llm_ratio - human_rt["ratio"]) / human_rt["ratio"],
            "source": human_rt["source"]
        }
        
        # Print comparison summary
        print("\n--- CRT Comparison ---")
        print(f"  Human Intuitive: {human_crt['intuitive_accuracy']:.1%}")
        print(f"  Human Reflective: {human_crt['reflective_accuracy']:.1%}")
        print(f"  LLM System 1: {s1_acc:.1%}")
        print(f"  LLM System 2: {s2_acc:.1%}")
        
        print("\n--- Response Time Ratio ---")
        print(f"  Human S2/S1 Ratio: {human_rt['ratio']:.1f}x")
        print(f"  LLM S2/S1 Ratio: {llm_ratio:.1f}x")
        
        return comparisons
    
    def run_category_breakdown(self, 
                              n_samples: int = 100,
                              n_rounds: int = 3,
                              use_all_data: bool = False) -> Dict:
        """Run detailed analysis by task category
        
        Args:
            n_samples: Samples per category per round (ignored if use_all_data=True)
            n_rounds: Number of rounds
            use_all_data: If True, use all available data for each category
        """
        
        print("\n" + "="*70)
        print("CATEGORY BREAKDOWN ANALYSIS")
        if use_all_data:
            print("(Using ALL available data)")
        print("="*70)
        
        categories = {
            "system1_tasks": "Intuitive tasks (commonsense, pattern recognition)",
            "system2_tasks": "Analytical tasks (math, logic)",
            "conflict_tasks": "Conflict tasks (CRT-style, cognitive traps)"
        }
        
        category_results = {}
        
        for category, description in categories.items():
            print(f"\n--- {category}: {description} ---")
            
            all_s1_acc = []
            all_s2_acc = []
            
            for round_num in range(1, n_rounds + 1):
                print(f"  Round {round_num}/{n_rounds}...")
                
                # Use None for n_samples to signal "use all data"
                samples_this_round = None if use_all_data else n_samples
                
                results = self.run_single_experiment(
                    experiment_name=f"category_{category}_round_{round_num}",
                    n_samples_per_category=samples_this_round,
                    system1_config={"model_name": "gpt-4o-mini", "temperature": 0.9},
                    system2_config={"model_name": "gpt-4o", "temperature": 0.2, "use_cot": True},
                    task_categories=[category]
                )
                
                all_s1_acc.append(results["system1_results"]["accuracy"])
                all_s2_acc.append(results["system2_results"]["accuracy"])
            
            # Calculate statistics for this category
            t_stat, p_value = stats.ttest_rel(all_s1_acc, all_s2_acc)
            effect_size = self.metrics.calculate_effect_size(all_s1_acc, all_s2_acc)
            
            category_results[category] = {
                "description": description,
                "system1_accuracy": {
                    "mean": float(np.mean(all_s1_acc)),
                    "std": float(np.std(all_s1_acc, ddof=1)),
                    "values": all_s1_acc
                },
                "system2_accuracy": {
                    "mean": float(np.mean(all_s2_acc)),
                    "std": float(np.std(all_s2_acc, ddof=1)),
                    "values": all_s2_acc
                },
                "difference": {
                    "mean": float(np.mean(all_s2_acc) - np.mean(all_s1_acc)),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": effect_size
                }
            }
            
            print(f"  S1: {np.mean(all_s1_acc):.1%} ± {np.std(all_s1_acc, ddof=1):.1%}")
            print(f"  S2: {np.mean(all_s2_acc):.1%} ± {np.std(all_s2_acc, ddof=1):.1%}")
            print(f"  p-value: {p_value:.4f}, Cohen's d: {effect_size['cohens_d']:.2f}")
        
        return category_results
    
    def generate_paper_figures(self, results: Dict, human_comparison: Dict, category_results: Dict):
        """Generate publication-quality figures"""
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Set publication style
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'serif',
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 13,
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Figure 1: Main accuracy comparison with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(2)
        s1_mean = results["system1_accuracy"]["mean"]
        s2_mean = results["system2_accuracy"]["mean"]
        s1_ci = results["system1_accuracy"]["ci_95"]
        s2_ci = results["system2_accuracy"]["ci_95"]
        
        bars = ax.bar(x, [s1_mean, s2_mean], 
                     yerr=[[s1_mean - s1_ci[0], s2_mean - s2_ci[0]], 
                           [s1_ci[1] - s1_mean, s2_ci[1] - s2_mean]],
                     capsize=5, color=['#3498db', '#e74c3c'], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Dual Process Theory: System Comparison\n(Mean ± 95% CI)')
        ax.set_ylim(0, 1)
        
        # Add significance annotation
        p_val = results["statistical_analysis"]["paired_t_test"]["p_value"]
        effect_d = results["statistical_analysis"]["effect_size"]["cohens_d"]
        
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        elif p_val < 0.05:
            sig_text = "*"
        else:
            sig_text = "n.s."
        
        ax.annotate(f'{sig_text}\np={p_val:.3f}\nd={effect_d:.2f}', 
                   xy=(0.5, max(s1_mean, s2_mean) + 0.08),
                   ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig1_main_comparison.png", dpi=300)
        plt.close()
        
        # Figure 2: Category breakdown
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(category_results.keys())
        x = np.arange(len(categories))
        width = 0.35
        
        s1_means = [category_results[c]["system1_accuracy"]["mean"] for c in categories]
        s2_means = [category_results[c]["system2_accuracy"]["mean"] for c in categories]
        s1_stds = [category_results[c]["system1_accuracy"]["std"] for c in categories]
        s2_stds = [category_results[c]["system2_accuracy"]["std"] for c in categories]
        
        bars1 = ax.bar(x - width/2, s1_means, width, yerr=s1_stds, 
                      label='System 1', color='#3498db', alpha=0.8, capsize=3)
        bars2 = ax.bar(x + width/2, s2_means, width, yerr=s2_stds,
                      label='System 2', color='#e74c3c', alpha=0.8, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Intuitive\nTasks', 'Analytical\nTasks', 'Conflict\nTasks'])
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance by Task Category')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add significance markers
        for i, cat in enumerate(categories):
            p_val = category_results[cat]["difference"]["p_value"]
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""
            if sig:
                ax.annotate(sig, xy=(i, max(s1_means[i], s2_means[i]) + s1_stds[i] + 0.03),
                           ha='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig2_category_breakdown.png", dpi=300)
        plt.close()
        
        # Figure 3: Human comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        ax1 = axes[0]
        x = np.arange(2)
        width = 0.25
        
        human_intuitive = human_comparison["crt"]["human_intuitive"]
        human_reflective = human_comparison["crt"]["human_reflective"]
        llm_s1 = human_comparison["crt"]["llm_system1"]
        llm_s2 = human_comparison["crt"]["llm_system2"]
        
        ax1.bar(x - width, [human_intuitive, human_reflective], width, 
               label='Human', color='#2ecc71', alpha=0.8)
        ax1.bar(x, [llm_s1, llm_s2], width,
               label='LLM', color='#9b59b6', alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Intuitive/S1', 'Reflective/S2'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Human vs LLM: Accuracy')
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Response time ratio comparison
        ax2 = axes[1]
        human_ratio = human_comparison["response_time"]["human_ratio"]
        llm_ratio = human_comparison["response_time"]["llm_ratio"]
        
        ax2.bar([0, 1], [human_ratio, llm_ratio], color=['#2ecc71', '#9b59b6'], alpha=0.8)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Human', 'LLM'])
        ax2.set_ylabel('S2/S1 Response Time Ratio')
        ax2.set_title('Response Time Ratio Comparison')
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig3_human_comparison.png", dpi=300)
        plt.close()
        
        # Figure 4: Effect size visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        effect_sizes = []
        labels = []
        
        for cat in category_results:
            effect_sizes.append(category_results[cat]["difference"]["effect_size"]["cohens_d"])
            labels.append(cat.replace("_", "\n"))
        
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in effect_sizes]
        bars = ax.barh(labels, effect_sizes, color=colors, alpha=0.8)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_title("Effect Size by Task Category\n(Positive = S2 > S1)")
        
        # Add interpretation regions
        ax.text(0.1, -0.5, 'small', fontsize=9, alpha=0.7)
        ax.text(0.35, -0.5, 'medium', fontsize=9, alpha=0.7)
        ax.text(0.65, -0.5, 'large', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(figures_dir / "fig4_effect_sizes.png", dpi=300)
        plt.close()
        
        print(f"\nFigures saved to {figures_dir}")
        
        return figures_dir
    
    def save_results(self, results: Dict, human_comparison: Dict, 
                    category_results: Dict, filename: str = "paper_results.json"):
        """Save all results to JSON"""
        
        full_results = {
            "timestamp": self.timestamp,
            "main_experiment": results,
            "human_comparison": human_comparison,
            "category_breakdown": category_results,
            "human_benchmarks_used": HUMAN_BENCHMARKS
        }
        
        # Convert numpy types for JSON serialization
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
        
        output_file = self.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        
        return output_file
    
    def generate_latex_tables(self, results: Dict, category_results: Dict):
        """Generate LaTeX tables for paper"""
        
        latex_dir = self.output_dir / "latex"
        latex_dir.mkdir(exist_ok=True)
        
        # Table 1: Main results
        table1 = r"""
\begin{table}[h]
\centering
\caption{Main Experimental Results: System 1 vs System 2 Comparison}
\label{tab:main_results}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{System 1} & \textbf{System 2} \\
\midrule
Accuracy & %.1f\%% $\pm$ %.1f\%% & %.1f\%% $\pm$ %.1f\%% \\
Confidence & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f \\
Response Time (ms) & %.0f $\pm$ %.0f & %.0f $\pm$ %.0f \\
Tokens Used & %.0f $\pm$ %.0f & %.0f $\pm$ %.0f \\
\midrule
\multicolumn{3}{l}{\textit{Statistical Analysis}} \\
\midrule
Accuracy Difference & \multicolumn{2}{c}{%.1f\%% (p=%.4f)} \\
Cohen's d & \multicolumn{2}{c}{%.2f (%s)} \\
\bottomrule
\end{tabular}
\end{table}
""" % (
            results["system1_accuracy"]["mean"] * 100,
            results["system1_accuracy"]["std"] * 100,
            results["system2_accuracy"]["mean"] * 100,
            results["system2_accuracy"]["std"] * 100,
            results["system1_confidence"]["mean"],
            results["system1_confidence"]["std"],
            results["system2_confidence"]["mean"],
            results["system2_confidence"]["std"],
            results["system1_response_time_ms"]["mean"],
            results["system1_response_time_ms"]["std"],
            results["system2_response_time_ms"]["mean"],
            results["system2_response_time_ms"]["std"],
            results["system1_tokens"]["mean"],
            results["system1_tokens"]["std"],
            results["system2_tokens"]["mean"],
            results["system2_tokens"]["std"],
            results["statistical_analysis"]["accuracy_difference"]["mean"] * 100,
            results["statistical_analysis"]["paired_t_test"]["p_value"],
            results["statistical_analysis"]["effect_size"]["cohens_d"],
            results["statistical_analysis"]["effect_size"]["interpretation"]
        )
        
        with open(latex_dir / "table1_main_results.tex", "w") as f:
            f.write(table1)
        
        # Table 2: Category breakdown
        table2 = r"""
\begin{table}[h]
\centering
\caption{Performance by Task Category}
\label{tab:category_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Category} & \textbf{S1 Acc.} & \textbf{S2 Acc.} & \textbf{Diff.} & \textbf{p-value} & \textbf{Cohen's d} \\
\midrule
"""
        for cat, data in category_results.items():
            cat_name = cat.replace("_", " ").title()
            table2 += r"%s & %.1f\%% & %.1f\%% & %.1f\%% & %.4f & %.2f \\" % (
                cat_name,
                data["system1_accuracy"]["mean"] * 100,
                data["system2_accuracy"]["mean"] * 100,
                data["difference"]["mean"] * 100,
                data["difference"]["p_value"],
                data["difference"]["effect_size"]["cohens_d"]
            )
            table2 += "\n"
        
        table2 += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(latex_dir / "table2_category_results.tex", "w") as f:
            f.write(table2)
        
        print(f"\nLaTeX tables saved to {latex_dir}")
        
        return latex_dir


# ============================================================================
# Paper Experiment Entry Point
# ============================================================================

def run_paper_experiments(mode: str = "full"):
    """Run paper-level experiments
    
    Modes:
        quick_validation: Fast test (2 rounds, 50 samples/category)
        medium: Medium run (3 rounds, 100 samples/category)
        full: Standard paper run (3 rounds, 200 samples/category)
        full_data: Use ALL available data (3 rounds, all samples)
    """
    
    runner = PaperExperimentRunner()
    
    if mode == "quick_validation":
        # Quick validation run
        n_rounds = 2
        n_samples = 50
        category_samples = 30
        use_all_data = False
    elif mode == "medium":
        # Medium run
        n_rounds = 3
        n_samples = 100
        category_samples = 50
        use_all_data = False
    elif mode == "full_data":
        # Use ALL available data
        n_rounds = 3
        n_samples = None  # Will use all data
        category_samples = None  # Will use all data
        use_all_data = True
    else:  # full
        # Full paper-level run
        n_rounds = 3
        n_samples = 200
        category_samples = 100
        use_all_data = False
    
    print("\n" + "="*70)
    print(f"PAPER-LEVEL EXPERIMENTS - Mode: {mode}")
    if use_all_data:
        print("Using ALL available data!")
    else:
        print(f"Rounds: {n_rounds}, Samples/category: {n_samples}")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Main experiment with multiple rounds
    print("\n[1/4] Running main experiment with multiple rounds...")
    main_results = runner.run_multiple_rounds(
        n_rounds=n_rounds,
        n_samples_per_category=n_samples,
        use_all_data=use_all_data
    )
    
    # 2. Human benchmark comparison
    print("\n[2/4] Comparing with human benchmarks...")
    human_comparison = runner.compare_with_human_benchmarks(main_results)
    
    # 3. Category breakdown
    print("\n[3/4] Running category breakdown analysis...")
    category_results = runner.run_category_breakdown(
        n_samples=category_samples,
        n_rounds=n_rounds,
        use_all_data=use_all_data
    )
    
    # 4. Generate outputs
    print("\n[4/4] Generating paper outputs...")
    runner.generate_paper_figures(main_results, human_comparison, category_results)
    runner.generate_latex_tables(main_results, category_results)
    runner.save_results(main_results, human_comparison, category_results)
    
    total_time = time.time() - start_time
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nTotal Duration: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"Results Directory: {runner.output_dir}")
    
    print("\n--- Key Findings ---")
    print(f"System 1 Accuracy: {main_results['system1_accuracy']['mean']:.1%} ± {main_results['system1_accuracy']['std']:.1%}")
    print(f"System 2 Accuracy: {main_results['system2_accuracy']['mean']:.1%} ± {main_results['system2_accuracy']['std']:.1%}")
    print(f"Difference: {main_results['statistical_analysis']['accuracy_difference']['mean']:.1%}")
    print(f"p-value: {main_results['statistical_analysis']['paired_t_test']['p_value']:.4f}")
    print(f"Cohen's d: {main_results['statistical_analysis']['effect_size']['cohens_d']:.2f} ({main_results['statistical_analysis']['effect_size']['interpretation']})")
    
    sig = main_results['statistical_analysis']['paired_t_test']['significant_at_05']
    print(f"\nStatistically Significant (p<0.05): {'Yes ✓' if sig else 'No ✗'}")
    
    return main_results, human_comparison, category_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual Process Theory Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic experiments:
    python run_experiment.py --experiment quick_test --n_samples 10
    python run_experiment.py --experiment full --n_samples 100
    python run_experiment.py --experiment ablation --n_samples 50
    python run_experiment.py --experiment category --n_samples 50
    python run_experiment.py --experiment stepped --n_samples 30

  Paper-level experiments (ICLR/NeurIPS standard):
    python run_experiment.py --experiment paper --mode quick_validation
    python run_experiment.py --experiment paper --mode medium
    python run_experiment.py --experiment paper --mode full
        """
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="quick_test",
        choices=["quick_test", "full", "ablation", "category", "stepped", "paper"],
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of samples per category (for basic experiments)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["quick_validation", "medium", "full", "full_data"],
        help="Paper experiment mode: quick_validation, medium, full, or full_data (use ALL data)"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "quick_test":
        run_quick_test(args.n_samples)
    elif args.experiment == "full":
        run_full_experiment(args.n_samples)
    elif args.experiment == "ablation":
        run_ablation_study(args.n_samples)
    elif args.experiment == "category":
        run_category_analysis(args.n_samples)
    elif args.experiment == "stepped":
        run_stepped_experiment(args.n_samples)
    elif args.experiment == "paper":
        run_paper_experiments(args.mode)


if __name__ == "__main__":
    main()
