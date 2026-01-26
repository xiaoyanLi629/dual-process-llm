"""
Math Reasoning Tasks
Mathematical Reasoning Tasks

For testingSystem 2mathematical analysis ability
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class MathReasoningItem:
    """Mathematical reasoning question data structure"""
    question: str
    correct_answer: str
    reasoning_steps: List[str]
    difficulty: str = "medium"
    math_type: str = "arithmetic"  # arithmetic, algebra, word_problem, etc.


class MathReasoningTask:
    """
    Mathematical Reasoning Tasks
    
    Contains multiple mathematical reasoning types：
    - arithmetic operations (Arithmetic)
    - algebra problems (Algebra)
    - word problems (Word Problems)
    - ProbabilityStatistics (Probability)
    """
    
    def __init__(self, task_loader=None):
        """
        InitializeMathematical Reasoning Tasks
        
        Args:
            task_loader: Task Loader（Used forLoadGSM8KetcData）
        """
        self.task_loader = task_loader
        self.custom_items: List[MathReasoningItem] = []
        
        # AddSomeExamplequestion
        self._add_sample_items()
    
    def _add_sample_items(self):
        """AddExamplemathInferencequestion"""
        self.custom_items = [
            MathReasoningItem(
                question="A train travels at 60 mph for 2 hours, then at 80 mph for 3 hours. What is the average speed for the entire journey?",
                correct_answer="72 mph",
                reasoning_steps=[
                    "Distance at 60 mph for 2 hours: 60 × 2 = 120 miles",
                    "Distance at 80 mph for 3 hours: 80 × 3 = 240 miles",
                    "Total distance: 120 + 240 = 360 miles",
                    "Total time: 2 + 3 = 5 hours",
                    "Average speed: 360 ÷ 5 = 72 mph"
                ],
                difficulty="medium",
                math_type="word_problem"
            ),
            MathReasoningItem(
                question="If 3x + 7 = 22, what is the value of x?",
                correct_answer="5",
                reasoning_steps=[
                    "3x + 7 = 22",
                    "3x = 22 - 7",
                    "3x = 15",
                    "x = 15 ÷ 3",
                    "x = 5"
                ],
                difficulty="easy",
                math_type="algebra"
            ),
            MathReasoningItem(
                question="A store offers a 20% discount on a $50 item. If there's an additional 10% tax on the discounted price, what is the final price?",
                correct_answer="$44",
                reasoning_steps=[
                    "Original price: $50",
                    "20% discount: $50 × 0.20 = $10",
                    "Discounted price: $50 - $10 = $40",
                    "10% tax on $40: $40 × 0.10 = $4",
                    "Final price: $40 + $4 = $44"
                ],
                difficulty="medium",
                math_type="word_problem"
            ),
            MathReasoningItem(
                question="In a class of 30 students, 18 play soccer and 15 play basketball. If 5 students play neither sport, how many students play both?",
                correct_answer="8",
                reasoning_steps=[
                    "Total students: 30",
                    "Students playing at least one sport: 30 - 5 = 25",
                    "Using inclusion-exclusion: Soccer + Basketball - Both = At least one",
                    "18 + 15 - Both = 25",
                    "33 - Both = 25",
                    "Both = 33 - 25 = 8"
                ],
                difficulty="hard",
                math_type="word_problem"
            ),
        ]
    
    def get_items(self, math_type: str = None, 
                  difficulty: str = None) -> List[MathReasoningItem]:
        """
        GetmathInferencequestion
        
        Args:
            math_type: mathTypeFilter
            difficulty: difficultyFilter
            
        Returns:
            List[MathReasoningItem]: questionList
        """
        items = self.custom_items.copy()
        
        if math_type:
            items = [i for i in items if i.math_type == math_type]
        
        if difficulty:
            items = [i for i in items if i.difficulty == difficulty]
        
        return items
    
    def get_from_dataset(self, n: int = 100) -> List[Dict]:
        """
        FromDatasetGetmathInferencequestion
        
        Args:
            n: questionCount
            
        Returns:
            List[Dict]: questionList
        """
        if self.task_loader is None:
            return []
        
        # FromSystem 2taskinfiltermathInferencequestion
        tasks = self.task_loader.get_system2_tasks(source_filter="GSM8K")
        return [t.to_dict() for t in tasks[:n]]
    
    def evaluate_response(self, item: MathReasoningItem, 
                         response: str) -> Dict[str, Any]:
        """
        EvaluateResponse
        
        Args:
            item: mathInferencequestion
            response: ModelResponse
            
        Returns:
            Dict: Evaluation result
        """
        # ExtractResponseindigit
        response_numbers = re.findall(r'-?\d+\.?\d*', response)
        correct_numbers = re.findall(r'-?\d+\.?\d*', item.correct_answer)
        
        is_correct = False
        extracted_answer = None
        
        if response_numbers:
            extracted_answer = response_numbers[-1]  # usuallyLastonedigitIsanswer
            
            # CheckIswhetherMatchcorrectanswer
            for cn in correct_numbers:
                if float(extracted_answer) == float(cn):
                    is_correct = True
                    break
        
        # alsoChecktextMatch
        if not is_correct and item.correct_answer.lower() in response.lower():
            is_correct = True
        
        return {
            "is_correct": is_correct,
            "extracted_answer": extracted_answer,
            "correct_answer": item.correct_answer,
            "math_type": item.math_type,
            "difficulty": item.difficulty
        }
    
    def format_for_experiment(self, item: MathReasoningItem) -> Dict[str, Any]:
        """
        FormatForexperimentFormat
        
        Args:
            item: mathInferencequestion
            
        Returns:
            Dict: experimentFormatData
        """
        return {
            "id": f"math_{self.custom_items.index(item):03d}",
            "question": item.question,
            "correct_answer": item.correct_answer,
            "task_type": "math_reasoning",
            "source": "Custom",
            "metadata": {
                "math_type": item.math_type,
                "difficulty": item.difficulty,
                "reasoning_steps": item.reasoning_steps
            }
        }
