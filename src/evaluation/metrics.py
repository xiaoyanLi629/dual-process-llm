"""
Metrics Calculator
Evaluation Metrics Module

Calculate various evaluation metrics for dual process theory experiments
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
from dataclasses import dataclass


@dataclass
class ExperimentMetrics:
    """Experiment metrics data structure"""
    accuracy: float
    avg_confidence: float
    avg_response_time_ms: float
    avg_tokens_used: float
    total_api_calls: int
    reasoning_steps_avg: float
    calibration_error: float
    overconfidence_rate: float
    underconfidence_rate: float


class MetricsCalculator:
    """
    Metrics Calculator
    
    Calculate various evaluation metrics for dual process theory experiments
    """
    
    def __init__(self):
        """Initialize the Metrics Calculator."""
        self.results_cache: Dict[str, List] = {}
    
    def calculate_accuracy(self, predictions: List[bool]) -> float:
        """
        Calculate accuracy.

        Args:
            predictions: List of booleans indicating whether each prediction is correct.

        Returns:
            float: Accuracy value.
        """
        if not predictions:
            return 0.0
        return sum(predictions) / len(predictions)
    
    def calculate_confidence_metrics(self, 
                                    confidences: List[float],
                                    correct: List[bool]) -> Dict[str, float]:
        """
        Calculate confidence-related metrics.

        Args:
            confidences: List of confidence values.
            correct: List of booleans indicating whether each prediction is correct.

        Returns:
            Dict: Confidence metrics.
        """
        if not confidences or not correct:
            return {
                "avg_confidence": 0.0,
                "calibration_error": 0.0,
                "overconfidence_rate": 0.0,
                "underconfidence_rate": 0.0
            }
        
        confidences = np.array(confidences)
        correct = np.array(correct)
        
        avg_confidence = np.mean(confidences)
        accuracy = np.mean(correct)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, correct)
        
        # Overconfidence rate: high confidence but incorrect
        high_conf_mask = confidences > 0.7
        if high_conf_mask.sum() > 0:
            overconfidence_rate = 1 - np.mean(correct[high_conf_mask])
        else:
            overconfidence_rate = 0.0
        
        # Underconfidence rate: low confidence but correct
        low_conf_mask = confidences < 0.3
        if low_conf_mask.sum() > 0:
            underconfidence_rate = np.mean(correct[low_conf_mask])
        else:
            underconfidence_rate = 0.0
        
        return {
            "avg_confidence": float(avg_confidence),
            "calibration_error": float(ece),
            "overconfidence_rate": float(overconfidence_rate),
            "underconfidence_rate": float(underconfidence_rate)
        }
    
    def _calculate_ece(self, confidences: np.ndarray, 
                       correct: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        Args:
            confidences: Array of confidence values.
            correct: Array of booleans indicating whether each prediction is correct.
            n_bins: Number of bins.

        Returns:
            float: ECE value.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence_in_bin = confidences[in_bin].mean()
                avg_accuracy_in_bin = correct[in_bin].mean()
                ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    def calculate_cognitive_effort_metrics(self,
                                          tokens_used: List[int],
                                          api_calls: List[int],
                                          reasoning_steps: List[int]) -> Dict[str, float]:
        """
        Calculate cognitive effort metrics.

        Args:
            tokens_used: List of token usage counts.
            api_calls: List of API call counts.
            reasoning_steps: List of reasoning step counts.

        Returns:
            Dict: Cognitive effort metrics.
        """
        return {
            "avg_tokens_used": float(np.mean(tokens_used)) if tokens_used else 0.0,
            "total_tokens": int(sum(tokens_used)) if tokens_used else 0,
            "avg_api_calls": float(np.mean(api_calls)) if api_calls else 0.0,
            "total_api_calls": int(sum(api_calls)) if api_calls else 0,
            "avg_reasoning_steps": float(np.mean(reasoning_steps)) if reasoning_steps else 0.0
        }
    
    def calculate_response_time_metrics(self, 
                                       response_times: List[float]) -> Dict[str, float]:
        """
        Calculate response time metrics.

        Args:
            response_times: List of response times (in milliseconds).

        Returns:
            Dict: Response time metrics.
        """
        if not response_times:
            return {
                "avg_response_time_ms": 0.0,
                "median_response_time_ms": 0.0,
                "std_response_time_ms": 0.0,
                "min_response_time_ms": 0.0,
                "max_response_time_ms": 0.0
            }
        
        times = np.array(response_times)
        
        return {
            "avg_response_time_ms": float(np.mean(times)),
            "median_response_time_ms": float(np.median(times)),
            "std_response_time_ms": float(np.std(times)),
            "min_response_time_ms": float(np.min(times)),
            "max_response_time_ms": float(np.max(times))
        }
    
    def compare_systems(self, 
                       system1_results: Dict[str, Any],
                       system2_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare System 1 and System 2 performance.

        Args:
            system1_results: System 1 results.
            system2_results: System 2 results.

        Returns:
            Dict: Comparison result.
        """
        comparison = {
            "accuracy_diff": system2_results.get("accuracy", 0) - system1_results.get("accuracy", 0),
            "confidence_diff": system2_results.get("avg_confidence", 0) - system1_results.get("avg_confidence", 0),
            "response_time_ratio": (
                system2_results.get("avg_response_time_ms", 1) / 
                max(system1_results.get("avg_response_time_ms", 1), 0.001)
            ),
            "tokens_ratio": (
                system2_results.get("avg_tokens_used", 1) / 
                max(system1_results.get("avg_tokens_used", 1), 0.001)
            ),
            "reasoning_steps_diff": (
                system2_results.get("avg_reasoning_steps", 0) - 
                system1_results.get("avg_reasoning_steps", 0)
            )
        }
        
        # Statistical significance test
        if "raw_accuracies" in system1_results and "raw_accuracies" in system2_results:
            s1_acc = system1_results["raw_accuracies"]
            s2_acc = system2_results["raw_accuracies"]
            
            if len(s1_acc) > 1 and len(s2_acc) > 1:
                # McNemar's test for paired data
                # Or use t-test
                t_stat, p_value = stats.ttest_ind(s1_acc, s2_acc)
                comparison["statistical_test"] = {
                    "test": "independent_t_test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
        
        return comparison
    
    def calculate_effect_size(self, 
                             group1: List[float], 
                             group2: List[float]) -> Dict[str, float]:
        """
        Calculate effect size (Cohen's d).

        Args:
            group1: Group 1 data.
            group2: Group 2 data.

        Returns:
            Dict: Effect size metrics.
        """
        if not group1 or not group2:
            return {"cohens_d": 0.0, "interpretation": "N/A"}
        
        g1 = np.array(group1)
        g2 = np.array(group2)
        
        # Cohen's d
        pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1) + (len(g2) - 1) * np.var(g2)) / 
                            (len(g1) + len(g2) - 2))
        
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std
        
        # Interpret effect size
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            "cohens_d": float(cohens_d),
            "interpretation": interpretation
        }
    
    def compute_token_efficiency_ratio(self, accuracy: float, avg_tokens: float) -> float:
        """
        Compute Token Efficiency Ratio (TER).

        TER = accuracy / (avg_tokens / 100)

        Replaces wall-clock response time as the effort metric.
        Higher TER means more accuracy per token of computation.

        Args:
            accuracy: Proportion correct (0–1).
            avg_tokens: Mean tokens consumed per response.

        Returns:
            float: Token Efficiency Ratio. Returns 0.0 if avg_tokens is 0.
        """
        if avg_tokens == 0:
            return 0.0
        return accuracy / (avg_tokens / 100)

    def compute_marginal_accuracy_gain(
        self,
        s1_acc: float,
        s2_acc: float,
        s1_tokens: float,
        s2_tokens: float,
    ) -> float:
        """
        Compute Marginal Accuracy Gain (MAG).

        MAG = (s2_accuracy - s1_accuracy) / (s2_tokens - s1_tokens)

        Can be computed per task category to show where extra token
        expenditure yields the greatest accuracy return.

        Args:
            s1_acc: System 1 accuracy (0–1).
            s2_acc: System 2 accuracy (0–1).
            s1_tokens: Mean tokens used by System 1.
            s2_tokens: Mean tokens used by System 2.

        Returns:
            float: Marginal Accuracy Gain. Returns 0.0 if token counts are equal.
        """
        token_diff = s2_tokens - s1_tokens
        if token_diff == 0:
            return 0.0
        return (s2_acc - s1_acc) / token_diff

    def aggregate_results(self, results: List[Dict]) -> ExperimentMetrics:
        """
        Aggregate experiment results.

        Args:
            results: List of individual experiment results.

        Returns:
            ExperimentMetrics: Aggregated metrics.
        """
        if not results:
            return ExperimentMetrics(
                accuracy=0.0,
                avg_confidence=0.0,
                avg_response_time_ms=0.0,
                avg_tokens_used=0.0,
                total_api_calls=0,
                reasoning_steps_avg=0.0,
                calibration_error=0.0,
                overconfidence_rate=0.0,
                underconfidence_rate=0.0
            )
        
        correct = [r.get("is_correct", False) for r in results]
        confidences = [r.get("confidence", 0.5) for r in results]
        response_times = [r.get("response_time_ms", 0) for r in results]
        tokens = [r.get("tokens_used", 0) for r in results]
        reasoning_steps = [r.get("reasoning_steps", 0) for r in results]
        
        confidence_metrics = self.calculate_confidence_metrics(confidences, correct)
        
        return ExperimentMetrics(
            accuracy=self.calculate_accuracy(correct),
            avg_confidence=confidence_metrics["avg_confidence"],
            avg_response_time_ms=float(np.mean(response_times)) if response_times else 0.0,
            avg_tokens_used=float(np.mean(tokens)) if tokens else 0.0,
            total_api_calls=len(results),
            reasoning_steps_avg=float(np.mean(reasoning_steps)) if reasoning_steps else 0.0,
            calibration_error=confidence_metrics["calibration_error"],
            overconfidence_rate=confidence_metrics["overconfidence_rate"],
            underconfidence_rate=confidence_metrics["underconfidence_rate"]
        )
