"""
Experiment Runner

Manage and execute dual process theory experiments
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm

from systems.system1 import System1
from systems.system2 import System2
from tasks.task_loader import TaskLoader, Task
from evaluation.evaluator import DualProcessEvaluator
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import DualProcessVisualizer


class ExperimentRunner:
    """
    Experiment Runner
    
    Manage the complete workflow of dual process theory experiments
    """
    
    def __init__(self, 
                 output_dir: str = None,
                 system1_config: Dict = None,
                 system2_config: Dict = None,
                 create_timestamp_dir: bool = True):
        """
        Initialize experiment runner
        
        Args:
            output_dir: Output directory
            system1_config: System 1 Configuration
            system2_config: System 2 Configuration
            create_timestamp_dir: Whether to create a timestamped subdirectory
        """
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            # Default: create timestamped directory under results/
            base_dir = Path(__file__).parent.parent.parent / "results"
            self.output_dir = base_dir / self.run_timestamp
        else:
            # Use provided directory directly (no timestamp subdirectory)
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize system
        self.system1_config = system1_config or {}
        self.system2_config = system2_config or {}
        
        self.system1 = System1(**self.system1_config)
        self.system2 = System2(**self.system2_config)
        
        # Initialize evaluator and metrics calculator
        self.evaluator = DualProcessEvaluator()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize visualizer
        figures_dir = self.output_dir / "figures"
        self.visualizer = DualProcessVisualizer(output_dir=figures_dir)
        
        # Initialize Task Loader
        self.task_loader = TaskLoader()
        
        # experiment record
        self.experiment_log: List[Dict] = []
    
    def run_experiment(self,
                      experiment_name: str,
                      task_categories: List[str] = None,
                      n_samples_per_category: int = 100,
                      shuffle: bool = True,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Run full experiment
        
        Args:
            experiment_name: Experiment name
            task_categories: List of task categories
            n_samples_per_category: Samples per category (None = use all data)
            shuffle: Whether to shuffle task order
            save_results: Whether to save results
            
        Returns:
            Dict: Experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load tasks
        self.task_loader.load()
        
        if task_categories is None:
            task_categories = ["system1_tasks", "system2_tasks", "conflict_tasks"]
        
        # Collect all tasks
        all_tasks = []
        for category in task_categories:
            # If n_samples_per_category is None, get all tasks
            if n_samples_per_category is None:
                tasks = self.task_loader.get_tasks(
                    category, 
                    n=None,  # Get all
                    shuffle=shuffle
                )
            else:
                tasks = self.task_loader.get_tasks(
                    category, 
                    n=n_samples_per_category, 
                    shuffle=shuffle
                )
            for task in tasks:
                task.metadata["category"] = category
            all_tasks.extend(tasks)
        
        print(f"Loaded dataset:")
        stats = self.task_loader.get_statistics()
        for cat in task_categories:
            print(f"  - {cat}: {stats['categories'][cat]['count']}")
        print(f"Total tasks: {len(all_tasks)}")
        
        # Run System 1
        print("\n--- Running System 1 ---")
        system1_results = self._run_system(
            self.system1, all_tasks, "System1"
        )
        
        # Run System 2
        print("\n--- Running System 2 ---")
        system2_results = self._run_system(
            self.system2, all_tasks, "System2"
        )
        
        # Calculate comparison metrics
        comparison = self.metrics_calculator.compare_systems(
            system1_results, system2_results
        )
        
        # Analyze by task category
        category_analysis = self._analyze_by_category(
            all_tasks, system1_results, system2_results
        )
        
        # Aggregate results
        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "config": {
                "task_categories": task_categories,
                "n_samples_per_category": n_samples_per_category,
                "total_tasks": len(all_tasks),
                "system1_config": self.system1_config,
                "system2_config": self.system2_config
            },
            "system1_results": {
                "accuracy": system1_results["accuracy"],
                "avg_confidence": system1_results["avg_confidence"],
                "avg_response_time_ms": system1_results["avg_response_time_ms"],
                "avg_tokens_used": system1_results["avg_tokens_used"],
                "avg_reasoning_steps": system1_results["avg_reasoning_steps"]
            },
            "system2_results": {
                "accuracy": system2_results["accuracy"],
                "avg_confidence": system2_results["avg_confidence"],
                "avg_response_time_ms": system2_results["avg_response_time_ms"],
                "avg_tokens_used": system2_results["avg_tokens_used"],
                "avg_reasoning_steps": system2_results["avg_reasoning_steps"]
            },
            "comparison": comparison,
            "category_analysis": category_analysis
        }
        
        # Save results
        if save_results:
            self._save_results(experiment_name, experiment_results, 
                             system1_results, system2_results)
        
        # Generate visualizations
        print("\n--- Generating Visualizations ---")
        experiment_results['system1_results']['results'] = system1_results.get('results', [])
        experiment_results['system2_results']['results'] = system2_results.get('results', [])
        self.visualizer.generate_all_figures(experiment_results, save=True)
        self.visualizer.close_all()
        
        # print summary
        self._print_summary(experiment_results)
        
        return experiment_results
    
    def _run_system(self, 
                   system: Any, 
                   tasks: List[Task],
                   system_name: str) -> Dict[str, Any]:
        """
        Run a single cognitive system.

        Args:
            system: Cognitive system instance.
            tasks: List of tasks.
            system_name: System name.

        Returns:
            Dict: System results.
        """
        responses = []
        
        for task in tqdm(tasks, desc=f"Processing {system_name}"):
            try:
                response = system.process(task.question, task.task_type)
                responses.append(response)
            except Exception as e:
                print(f"Error processing task {task.id}: {e}")
                responses.append({"answer": "Error", "confidence": 0.0})
        
        # Evaluation result
        task_dicts = [task.to_dict() for task in tasks]
        results = self.evaluator.evaluate_batch(task_dicts, responses, system_name)
        
        # Add cognitive effort metrics
        effort_metrics = system.get_cognitive_effort_metrics()
        results.update(effort_metrics)
        
        return results
    
    def _analyze_by_category(self,
                            tasks: List[Task],
                            system1_results: Dict,
                            system2_results: Dict) -> Dict[str, Any]:
        """
        Analyze results by task category.

        Args:
            tasks: List of tasks.
            system1_results: System 1 results.
            system2_results: System 2 results.

        Returns:
            Dict: Category analysis results.
        """
        categories = {}
        
        s1_results = system1_results.get("results", [])
        s2_results = system2_results.get("results", [])
        
        for i, task in enumerate(tasks):
            category = task.metadata.get("category", "unknown")
            
            if category not in categories:
                categories[category] = {
                    "system1": {"correct": 0, "total": 0},
                    "system2": {"correct": 0, "total": 0}
                }
            
            categories[category]["system1"]["total"] += 1
            categories[category]["system2"]["total"] += 1
            
            if i < len(s1_results) and s1_results[i].get("is_correct"):
                categories[category]["system1"]["correct"] += 1
            
            if i < len(s2_results) and s2_results[i].get("is_correct"):
                categories[category]["system2"]["correct"] += 1
        
        # Calculate accuracy
        for category in categories:
            for system in ["system1", "system2"]:
                total = categories[category][system]["total"]
                correct = categories[category][system]["correct"]
                categories[category][system]["accuracy"] = correct / total if total > 0 else 0
        
        return categories
    
    def _save_results(self,
                     experiment_name: str,
                     experiment_results: Dict,
                     system1_results: Dict,
                     system2_results: Dict):
        """
        Save experiment results
        
        Args:
            experiment_name: Experiment name
            experiment_results: Experiment results summary
            system1_results: System 1 detailed results
            system2_results: System 2 detailed results
        """
        # Save summary results
        summary_path = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed results
        detailed_results = {
            "system1_detailed": system1_results,
            "system2_detailed": system2_results
        }
        detailed_path = self.output_dir / f"{experiment_name}_detailed.json"
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save experiment config
        config_path = self.output_dir / "experiment_config.json"
        config_data = {
            "run_timestamp": self.run_timestamp,
            "experiment_name": experiment_name,
            "system1_config": self.system1_config,
            "system2_config": self.system2_config
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to folder: {self.output_dir}")
        print(f"  - {summary_path.name}")
        print(f"  - {detailed_path.name}")
        print(f"  - {config_path.name}")
    
    def _print_summary(self, results: Dict):
        """
        print experiment summary
        
        Args:
            results: experiment results
        """
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        
        print(f"\nExperiment: {results['experiment_name']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Total Tasks: {results['config']['total_tasks']}")
        
        print(f"\n--- System 1 (Intuitive) ---")
        s1 = results['system1_results']
        print(f"  Accuracy: {s1['accuracy']:.2%}")
        print(f"  Avg Confidence: {s1['avg_confidence']:.2f}")
        print(f"  Avg Response Time: {s1['avg_response_time_ms']:.0f} ms")
        print(f"  Avg Tokens Used: {s1['avg_tokens_used']:.0f}")
        
        print(f"\n--- System 2 (Analytical) ---")
        s2 = results['system2_results']
        print(f"  Accuracy: {s2['accuracy']:.2%}")
        print(f"  Avg Confidence: {s2['avg_confidence']:.2f}")
        print(f"  Avg Response Time: {s2['avg_response_time_ms']:.0f} ms")
        print(f"  Avg Tokens Used: {s2['avg_tokens_used']:.0f}")
        print(f"  Avg Reasoning Steps: {s2['avg_reasoning_steps']:.1f}")
        
        print(f"\n--- Comparison ---")
        comp = results['comparison']
        print(f"  Accuracy Difference (S2-S1): {comp['accuracy_diff']:+.2%}")
        print(f"  Response Time Ratio (S2/S1): {comp['response_time_ratio']:.2f}x")
        print(f"  Tokens Ratio (S2/S1): {comp['tokens_ratio']:.2f}x")
        
        if "statistical_test" in comp:
            test = comp["statistical_test"]
            print(f"\n  Statistical Test: {test['test']}")
            print(f"  p-value: {test['p_value']:.4f}")
            print(f"  Significant: {test['significant']}")
        
        print(f"\n--- Category Analysis ---")
        for category, data in results['category_analysis'].items():
            print(f"\n  {category}:")
            print(f"    System 1 Accuracy: {data['system1']['accuracy']:.2%}")
            print(f"    System 2 Accuracy: {data['system2']['accuracy']:.2%}")
    
    def run_ablation_study(self,
                          base_experiment_name: str,
                          ablation_configs: List[Dict],
                          n_samples: int = 50) -> List[Dict]:
        """
        Run ablation study.

        Args:
            base_experiment_name: Base experiment name.
            ablation_configs: List of ablation configurations.
            n_samples: Number of samples per configuration.

        Returns:
            List[Dict]: List of ablation experiment results.
        """
        results = []
        
        for i, config in enumerate(ablation_configs):
            print(f"\n--- Ablation {i+1}/{len(ablation_configs)} ---")
            print(f"Config: {config}")
            
            # Update system configuration
            if "system1" in config:
                self.system1 = System1(**config["system1"])
            if "system2" in config:
                self.system2 = System2(**config["system2"])
            
            # Run experiment
            exp_name = f"{base_experiment_name}_ablation_{i+1}"
            exp_results = self.run_experiment(
                experiment_name=exp_name,
                n_samples_per_category=n_samples,
                save_results=True
            )
            
            exp_results["ablation_config"] = config
            results.append(exp_results)
        
        return results
    
    def reset(self):
        """Reset experiment state."""
        self.system1.reset()
        self.system2.reset()
        self.evaluator.reset()
        self.experiment_log = []
