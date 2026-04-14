"""
Dual Process Theory Evaluator

Evaluates System 1 and System 2 performance on various tasks.
"""

import json
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
    
    def _extract_json_answer(self, response_text: str) -> str:
        """
        Try to extract the 'answer' field from a JSON response.
        Returns the answer string if found, otherwise returns the full response.
        """
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and 'answer' in data:
                return str(data['answer'])
        except (json.JSONDecodeError, TypeError):
            pass

        # Try to find JSON in the response (model might include text before/after)
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]*)"[^{}]*\}', response_text)
        if json_match:
            return json_match.group(1)

        # Fallback: return original response
        return response_text

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

        # Step 1: Extract answer from JSON if possible
        answer_text = self._extract_json_answer(str(response))

        # Step 2: Clean
        answer_clean = answer_text.lower().strip()
        correct_clean = str(correct_answer).lower().strip()

        # Step 3: Direct match
        if correct_clean == answer_clean:
            return True
        if correct_clean in answer_clean or answer_clean in correct_clean:
            return True

        # Step 4: Multiple choice (strict)
        if options is not None:
            return self._check_multiple_choice(answer_clean, correct_answer, options)

        # Step 5: Numeric (on extracted answer only)
        if self._is_numeric_task(task_type):
            return self._check_numeric(answer_clean, correct_clean)

        # Step 6: No fuzzy fallback — if we can't match cleanly, it's wrong
        return False
    
    def _check_multiple_choice(self,
                               answer: str,
                               correct_answer: Any,
                               options: List[str]) -> bool:
        """Check a multiple choice answer (strict matching)."""
        # Determine correct letter
        if isinstance(correct_answer, int):
            correct_letter = chr(65 + correct_answer).lower()
        else:
            correct_letter = str(correct_answer).strip().lower()

        # The answer field should contain just the letter or the option text
        # Check if answer IS the correct letter (not just contains it)
        answer_stripped = answer.strip().rstrip('.').lower()

        # Direct letter match: answer is exactly "a", "b", "c", etc.
        if answer_stripped == correct_letter:
            return True

        # Answer starts with the correct letter followed by punctuation or space
        if len(answer_stripped) > 1 and answer_stripped[0] == correct_letter and answer_stripped[1] in '.):, ':
            return True

        # Check if answer matches the correct option text
        if isinstance(correct_answer, int) and correct_answer < len(options):
            correct_text = options[correct_answer].lower().strip()
            if correct_text in answer or answer in correct_text:
                return True

        return False
    
    def _check_numeric(self, answer: str, correct: str) -> bool:
        """Check a numeric answer (matches last number in extracted answer field only)."""
        response_nums = re.findall(r'-?\d+\.?\d*', answer)
        correct_nums = re.findall(r'-?\d+\.?\d*', correct)

        if not response_nums or not correct_nums:
            return False

        # Compare the LAST number in the answer (most likely the final answer)
        # against all correct numbers
        try:
            answer_num = float(response_nums[-1])
            for cn in correct_nums:
                if abs(answer_num - float(cn)) < 0.01:
                    return True
        except ValueError:
            pass

        return False
    
    def _is_numeric_task(self, task_type: str) -> bool:
        """Determine whether the task is a numeric task."""
        numeric_types = [
            "math_reasoning", "arithmetic", "algebra", 
            "word_problem", "calculation", "simple_math"
        ]
        return task_type.lower() in numeric_types
    
    def _fuzzy_match(self, response: str, correct: str) -> bool:
        """Strict fuzzy matching (95% word overlap required). Rarely used since JSON extraction handles most cases."""
        if correct in response:
            return True

        correct_words = set(correct.split())
        response_words = set(response.split())

        if len(correct_words) > 0:
            overlap = len(correct_words & response_words) / len(correct_words)
            if overlap >= 0.95:
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
