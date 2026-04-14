"""
Dual Process Theory Evaluator

Evaluates System 1 and System 2 performance on various tasks.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class DualProcessEvaluator:
    """
    Dual Process Theory Evaluator
    
    Responsible for evaluating model response correctness and quality
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_history: List[Dict] = []
    
    def evaluate_response(self, 
                         question: str,
                         response: Any,
                         correct_answer: Any,
                         task_type: str = "general",
                         options: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single response.

        Args:
            question: Question text.
            response: Model response.
            correct_answer: Correct answer.
            task_type: Task type.
            options: Option list (if multiple choice).

        Returns:
            Dict: Evaluation result.
        """
        # Extract the answer from the response
        if hasattr(response, 'answer'):
            answer_text = response.answer
            confidence = getattr(response, 'confidence', 0.5)
            response_time = getattr(response, 'response_time_ms', 0)
            tokens_used = getattr(response, 'tokens_used', 0)
            reasoning_steps = len(getattr(response, 'reasoning_steps', []))
        elif isinstance(response, dict):
            answer_text = response.get('answer', str(response))
            confidence = response.get('confidence', 0.5)
            response_time = response.get('response_time_ms', 0)
            tokens_used = response.get('tokens_used', 0)
            reasoning_steps = len(response.get('reasoning_steps', []))
        else:
            answer_text = str(response)
            confidence = 0.5
            response_time = 0
            tokens_used = 0
            reasoning_steps = 0
        
        # Judge whether the answer is correct
        is_correct = self._check_correctness(
            answer_text, correct_answer, options, task_type
        )
        
        evaluation = {
            "question": question[:200] + "..." if len(question) > 200 else question,
            "response": answer_text,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "confidence": confidence,
            "response_time_ms": response_time,
            "tokens_used": tokens_used,
            "reasoning_steps": reasoning_steps,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_history.append(evaluation)
        
        return evaluation
    
    def _check_correctness(self, 
                          response: str, 
                          correct_answer: Any,
                          options: List[str] = None,
                          task_type: str = "general") -> bool:
        """
        Check whether a response is correct.

        Args:
            response: Model response.
            correct_answer: Correct answer.
            options: Option list.
            task_type: Task type.

        Returns:
            bool: Whether the response is correct.
        """
        if response is None or correct_answer is None:
            return False
        
        response_clean = str(response).lower().strip()
        correct_clean = str(correct_answer).lower().strip()
        
        # Direct match
        if correct_clean in response_clean:
            return True
        
        # Multiple choice match
        if options is not None:
            return self._check_multiple_choice(response_clean, correct_answer, options)
        
        # Numeric match
        if self._is_numeric_task(task_type):
            return self._check_numeric(response_clean, correct_clean)
        
        # Fuzzy match
        return self._fuzzy_match(response_clean, correct_clean)
    
    def _check_multiple_choice(self,
                               response: str,
                               correct_answer: Any,
                               options: List[str]) -> bool:
        """Check a multiple choice answer."""
        # If the correct answer is an index
        if isinstance(correct_answer, int):
            correct_letter = chr(65 + correct_answer)
            correct_text = options[correct_answer].lower() if correct_answer < len(options) else ""
        else:
            correct_letter = str(correct_answer).upper()
            correct_text = ""
            # Attempt to find the corresponding option text
            for i, opt in enumerate(options):
                if chr(65 + i) == correct_letter:
                    correct_text = opt.lower()
                    break
        
        # Check whether the response contains the correct option letter
        if correct_letter.lower() in response:
            # Ensure it is not another option
            for i in range(len(options)):
                letter = chr(65 + i).lower()
                if letter in response and letter != correct_letter.lower():
                    # Response contains multiple options, need more precise matching
                    pass
            return True
        
        # Check whether the response contains the correct option text
        if correct_text and correct_text in response:
            return True
        
        return False
    
    def _check_numeric(self, response: str, correct: str) -> bool:
        """Check a numeric answer."""
        # Extract numbers
        response_nums = re.findall(r'-?\d+\.?\d*', response)
        correct_nums = re.findall(r'-?\d+\.?\d*', correct)
        
        if not response_nums or not correct_nums:
            return False
        
        # Check whether any numbers match
        for rn in response_nums:
            for cn in correct_nums:
                try:
                    if abs(float(rn) - float(cn)) < 0.01:
                        return True
                except ValueError:
                    continue
        
        return False
    
    def _is_numeric_task(self, task_type: str) -> bool:
        """Determine whether the task is a numeric task."""
        numeric_types = [
            "math_reasoning", "arithmetic", "algebra", 
            "word_problem", "calculation", "simple_math"
        ]
        return task_type.lower() in numeric_types
    
    def _fuzzy_match(self, response: str, correct: str) -> bool:
        """Fuzzy matching."""
        # Simple containment check
        if correct in response:
            return True
        
        # Check keyword overlap
        correct_words = set(correct.split())
        response_words = set(response.split())
        
        if len(correct_words) > 0:
            overlap = len(correct_words & response_words) / len(correct_words)
            if overlap > 0.8:
                return True
        
        return False
    
    def evaluate_batch(self, 
                      tasks: List[Dict],
                      responses: List[Any],
                      system_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate a batch of responses.

        Args:
            tasks: List of tasks.
            responses: List of responses.
            system_name: System name.

        Returns:
            Dict: Batch evaluation result.
        """
        if len(tasks) != len(responses):
            raise ValueError("Tasks and responses must have the same length")
        
        results = []
        for task, response in zip(tasks, responses):
            eval_result = self.evaluate_response(
                question=task.get("question", ""),
                response=response,
                correct_answer=task.get("correct_answer"),
                task_type=task.get("task_type", "general"),
                options=task.get("options")
            )
            results.append(eval_result)
        
        # Aggregate results
        correct_count = sum(1 for r in results if r["is_correct"])
        total_count = len(results)
        
        return {
            "system": system_name,
            "total_tasks": total_count,
            "correct_count": correct_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0,
            "avg_confidence": sum(r["confidence"] for r in results) / total_count if total_count > 0 else 0,
            "avg_response_time_ms": sum(r["response_time_ms"] for r in results) / total_count if total_count > 0 else 0,
            "avg_tokens_used": sum(r["tokens_used"] for r in results) / total_count if total_count > 0 else 0,
            "avg_reasoning_steps": sum(r["reasoning_steps"] for r in results) / total_count if total_count > 0 else 0,
            "results": results,
            "raw_accuracies": [1 if r["is_correct"] else 0 for r in results]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of evaluations.

        Returns:
            Dict: Evaluation summary.
        """
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        correct_count = sum(1 for e in self.evaluation_history if e["is_correct"])
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "correct_count": correct_count,
            "accuracy": correct_count / len(self.evaluation_history),
            "avg_confidence": sum(e["confidence"] for e in self.evaluation_history) / len(self.evaluation_history),
            "task_type_breakdown": self._get_task_type_breakdown()
        }
    
    def _get_task_type_breakdown(self) -> Dict[str, Dict]:
        """Get statistics broken down by task type."""
        breakdown = {}
        
        for eval_result in self.evaluation_history:
            task_type = eval_result.get("task_type", "unknown")
            
            if task_type not in breakdown:
                breakdown[task_type] = {"total": 0, "correct": 0}
            
            breakdown[task_type]["total"] += 1
            if eval_result["is_correct"]:
                breakdown[task_type]["correct"] += 1
        
        # Calculate accuracy
        for task_type in breakdown:
            total = breakdown[task_type]["total"]
            correct = breakdown[task_type]["correct"]
            breakdown[task_type]["accuracy"] = correct / total if total > 0 else 0
        
        return breakdown
    
    def reset(self):
        """Reset evaluation history."""
        self.evaluation_history = []
