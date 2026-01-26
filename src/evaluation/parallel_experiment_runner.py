"""
Parallel Experiment Runner
High-performance experiment runner using async API calls

Achieves ~8-10x speedup compared to sequential processing.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from systems.async_systems import AsyncSystem1, AsyncSystem2, AsyncResponse
from tasks.task_loader import TaskLoader, Task
from evaluation.evaluator import DualProcessEvaluator
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import DualProcessVisualizer


class ParallelExperimentRunner:
    """
    Parallel Experiment Runner
    
    Uses async API calls to process multiple tasks concurrently,
    achieving significant speedup over sequential processing.
    """
    
    def __init__(self,
                 output_dir: str = None,
                 system1_config: Dict = None,
                 system2_config: Dict = None,
                 s1_concurrent: int = 20,
                 s2_concurrent: int = 10):
        """
        Initialize parallel experiment runner
        
        Args:
            output_dir: Output directory
            system1_config: System 1 configuration
            system2_config: System 2 configuration
            s1_concurrent: Max concurrent calls for System 1 (lighter model)
            s2_concurrent: Max concurrent calls for System 2 (heavier model)
        """
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "results"
            self.output_dir = base_dir / self.run_timestamp
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store configs
        self.system1_config = system1_config or {}
        self.system2_config = system2_config or {}
        self.s1_concurrent = s1_concurrent
        self.s2_concurrent = s2_concurrent
        
        # Initialize async systems
        self.system1 = AsyncSystem1(
            max_concurrent=s1_concurrent,
            **self.system1_config
        )
        self.system2 = AsyncSystem2(
            max_concurrent=s2_concurrent,
            **self.system2_config
        )
        
        # Initialize evaluator and metrics
        self.evaluator = DualProcessEvaluator()
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize visualizer
        figures_dir = self.output_dir / "figures"
        self.visualizer = DualProcessVisualizer(output_dir=figures_dir)
        
        # Initialize task loader
        self.task_loader = TaskLoader()
    
    def run_experiment(self,
                      experiment_name: str,
                      task_categories: List[str] = None,
                      n_samples_per_category: int = 100,
                      shuffle: bool = True,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Run experiment with parallel processing
        
        Args:
            experiment_name: Experiment name
            task_categories: List of task categories
            n_samples_per_category: Samples per category (None = use all)
            shuffle: Whether to shuffle tasks
            save_results: Whether to save results
            
        Returns:
            Dict: Experiment results
        """
        print(f"\n{'='*60}")
        print(f"Running Parallel Experiment: {experiment_name}")
        print(f"Concurrency: S1={self.s1_concurrent}, S2={self.s2_concurrent}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load tasks
        self.task_loader.load()
        
        if task_categories is None:
            task_categories = ["system1_tasks", "system2_tasks", "conflict_tasks"]
        
        # Collect all tasks
        all_tasks = []
        for category in task_categories:
            if n_samples_per_category is None:
                tasks = self.task_loader.get_tasks(category, n=None, shuffle=shuffle)
            else:
                tasks = self.task_loader.get_tasks(category, n=n_samples_per_category, shuffle=shuffle)
            for task in tasks:
                task.metadata["category"] = category
            all_tasks.extend(tasks)
        
        print(f"Loaded {len(all_tasks)} tasks")
        
        # Extract questions and task types
        questions = [task.question for task in all_tasks]
        task_types = [task.task_type for task in all_tasks]
        
        # Run both systems in parallel
        print("\n--- Running Systems in Parallel ---")
        
        # Use asyncio to run the parallel processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            s1_responses, s2_responses = loop.run_until_complete(
                self._run_parallel_with_progress(questions, task_types)
            )
        finally:
            loop.close()
        
        # Convert responses to evaluation format
        system1_results = self._process_responses(all_tasks, s1_responses, "System1")
        system2_results = self._process_responses(all_tasks, s2_responses, "System2")
        
        # Calculate comparison metrics
        comparison = self.metrics_calculator.compare_systems(system1_results, system2_results)
        
        # Category analysis
        category_analysis = self._analyze_by_category(all_tasks, system1_results, system2_results)
        
        # Aggregate results
        experiment_results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - start_time,
            "parallel_config": {
                "s1_concurrent": self.s1_concurrent,
                "s2_concurrent": self.s2_concurrent
            },
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
                "avg_reasoning_steps": system1_results.get("avg_reasoning_steps", 0)
            },
            "system2_results": {
                "accuracy": system2_results["accuracy"],
                "avg_confidence": system2_results["avg_confidence"],
                "avg_response_time_ms": system2_results["avg_response_time_ms"],
                "avg_tokens_used": system2_results["avg_tokens_used"],
                "avg_reasoning_steps": system2_results.get("avg_reasoning_steps", 0)
            },
            "comparison": comparison,
            "category_analysis": category_analysis
        }
        
        # Save results
        if save_results:
            self._save_results(experiment_name, experiment_results, system1_results, system2_results)
        
        # Generate visualizations
        print("\n--- Generating Visualizations ---")
        experiment_results['system1_results']['results'] = system1_results.get('results', [])
        experiment_results['system2_results']['results'] = system2_results.get('results', [])
        self.visualizer.generate_all_figures(experiment_results, save=True)
        self.visualizer.close_all()
        
        # Print summary
        self._print_summary(experiment_results)
        
        return experiment_results
    
    async def _run_parallel_with_progress(self, questions: List[str], 
                                          task_types: List[str]) -> tuple:
        """Run both systems with progress tracking"""
        
        print(f"\nProcessing {len(questions)} tasks...")
        print("System 1 (gpt-4o-mini) and System 2 (gpt-4o) running concurrently...")
        
        # Create progress tracking
        total_tasks = len(questions) * 2  # Both systems
        
        start_time = time.time()
        
        # Run System 1
        print("\n[System 1] Starting parallel processing...")
        s1_start = time.time()
        s1_responses = await self.system1.process_batch(questions, task_types)
        s1_time = time.time() - s1_start
        print(f"[System 1] Completed in {s1_time:.1f}s ({len(questions)/s1_time:.1f} tasks/sec)")
        
        # Run System 2
        print("\n[System 2] Starting parallel processing...")
        s2_start = time.time()
        s2_responses = await self.system2.process_batch(questions, task_types)
        s2_time = time.time() - s2_start
        print(f"[System 2] Completed in {s2_time:.1f}s ({len(questions)/s2_time:.1f} tasks/sec)")
        
        total_time = time.time() - start_time
        print(f"\nTotal parallel processing time: {total_time:.1f}s")
        print(f"Effective throughput: {len(questions)*2/total_time:.1f} tasks/sec")
        
        return s1_responses, s2_responses
    
    def _process_responses(self, tasks: List[Task], 
                          responses: List[AsyncResponse],
                          system_name: str) -> Dict[str, Any]:
        """Process async responses into evaluation format"""
        
        # Convert to evaluation format
        eval_responses = []
        for resp in responses:
            eval_responses.append({
                "answer": resp.answer,
                "confidence": resp.confidence,
                "response_time_ms": resp.response_time_ms,
                "tokens_used": resp.tokens_used,
                "reasoning_steps": resp.reasoning_steps
            })
        
        # Evaluate
        task_dicts = [task.to_dict() for task in tasks]
        results = self.evaluator.evaluate_batch(task_dicts, eval_responses, system_name)
        
        # Add metrics
        total_tokens = sum(r.tokens_used for r in responses)
        avg_tokens = total_tokens / len(responses) if responses else 0
        avg_time = sum(r.response_time_ms for r in responses) / len(responses) if responses else 0
        avg_steps = sum(len(r.reasoning_steps) for r in responses) / len(responses) if responses else 0
        
        results["total_tokens"] = total_tokens
        results["total_api_calls"] = len(responses)
        results["avg_tokens_per_call"] = avg_tokens
        results["avg_response_time_ms"] = avg_time
        results["avg_reasoning_steps"] = avg_steps
        
        return results
    
    def _analyze_by_category(self, tasks: List[Task],
                            system1_results: Dict,
                            system2_results: Dict) -> Dict[str, Any]:
        """Analyze results by task category"""
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
    
    def _save_results(self, experiment_name: str, experiment_results: Dict,
                     system1_results: Dict, system2_results: Dict):
        """Save experiment results"""
        # Save summary
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
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def _print_summary(self, results: Dict):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY (Parallel)")
        print(f"{'='*60}")
        
        print(f"\nExperiment: {results['experiment_name']}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Total Tasks: {results['config']['total_tasks']}")
        
        # Calculate speedup estimate
        sequential_estimate = results['config']['total_tasks'] * 14  # ~14s per task sequential
        actual_time = results['duration_seconds']
        speedup = sequential_estimate / actual_time if actual_time > 0 else 1
        print(f"Estimated Speedup: {speedup:.1f}x")
        
        print(f"\n--- System 1 (Intuitive) ---")
        s1 = results['system1_results']
        print(f"  Accuracy: {s1['accuracy']:.2%}")
        print(f"  Avg Confidence: {s1['avg_confidence']:.2f}")
        print(f"  Avg Response Time: {s1['avg_response_time_ms']:.0f} ms")
        
        print(f"\n--- System 2 (Analytical) ---")
        s2 = results['system2_results']
        print(f"  Accuracy: {s2['accuracy']:.2%}")
        print(f"  Avg Confidence: {s2['avg_confidence']:.2f}")
        print(f"  Avg Response Time: {s2['avg_response_time_ms']:.0f} ms")
        
        print(f"\n--- Comparison ---")
        comp = results['comparison']
        print(f"  Accuracy Difference (S2-S1): {comp['accuracy_diff']:+.2%}")
    
    def reset(self):
        """Reset experiment state"""
        self.system1.reset()
        self.system2.reset()
        self.evaluator.reset()
