"""
Commonsense Reasoning Tasks

For testing System 1 intuitive reasoning ability.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CommonsenseItem:
    """Commonsense reasoning question data structure"""
    question: str
    options: List[str]
    correct_answer: int  # Correct option index
    commonsense_type: str  # physical, social, temporal, etc.
    difficulty: str = "easy"


class CommonsenseTask:
    """
    Commonsense Reasoning Tasks.

    Contains multiple commonsense reasoning types:
    - Physical commonsense
    - Social commonsense
    - Temporal commonsense
    - Causal commonsense
    """
    
    def __init__(self, task_loader=None):
        """
        Initialize Commonsense Reasoning Tasks.

        Args:
            task_loader: Task loader instance.
        """
        self.task_loader = task_loader
        self.custom_items: List[CommonsenseItem] = []
        
        # Add sample questions
        self._add_sample_items()
    
    def _add_sample_items(self):
        """Add sample commonsense reasoning questions."""
        self.custom_items = [
            CommonsenseItem(
                question="What happens when you drop a glass on a hard floor?",
                options=[
                    "It bounces back up",
                    "It breaks into pieces",
                    "It floats in the air",
                    "It turns into water"
                ],
                correct_answer=1,
                commonsense_type="physical",
                difficulty="easy"
            ),
            CommonsenseItem(
                question="If someone is crying at a funeral, they are most likely feeling:",
                options=[
                    "Happy",
                    "Hungry",
                    "Sad",
                    "Sleepy"
                ],
                correct_answer=2,
                commonsense_type="social",
                difficulty="easy"
            ),
            CommonsenseItem(
                question="What typically comes after breakfast?",
                options=[
                    "Dinner",
                    "Lunch",
                    "Midnight snack",
                    "Yesterday's meal"
                ],
                correct_answer=1,
                commonsense_type="temporal",
                difficulty="easy"
            ),
            CommonsenseItem(
                question="If you leave ice cream outside on a hot day, what will happen?",
                options=[
                    "It will freeze more",
                    "It will melt",
                    "It will turn into cheese",
                    "Nothing will happen"
                ],
                correct_answer=1,
                commonsense_type="causal",
                difficulty="easy"
            ),
            CommonsenseItem(
                question="Why do people usually carry an umbrella when dark clouds appear?",
                options=[
                    "To block the sun",
                    "To catch butterflies",
                    "To prepare for rain",
                    "To wave at airplanes"
                ],
                correct_answer=2,
                commonsense_type="causal",
                difficulty="easy"
            ),
        ]
    
    def get_items(self, commonsense_type: str = None, 
                  difficulty: str = None) -> List[CommonsenseItem]:
        """
        Get commonsense reasoning questions.

        Args:
            commonsense_type: Commonsense type filter.
            difficulty: Difficulty filter.

        Returns:
            List[CommonsenseItem]: List of questions.
        """
        items = self.custom_items.copy()
        
        if commonsense_type:
            items = [i for i in items if i.commonsense_type == commonsense_type]
        
        if difficulty:
            items = [i for i in items if i.difficulty == difficulty]
        
        return items
    
    def get_from_dataset(self, n: int = 100, source: str = None) -> List[Dict]:
        """
        Get commonsense reasoning questions from the dataset.

        Args:
            n: Number of questions.
            source: Data source filter.

        Returns:
            List[Dict]: List of questions.
        """
        if self.task_loader is None:
            return []
        
        # Filter commonsense reasoning questions from System 1 tasks
        if source:
            tasks = self.task_loader.get_system1_tasks(source_filter=source)
        else:
            tasks = self.task_loader.get_system1_tasks()
        
        return [t.to_dict() for t in tasks[:n]]
    
    def evaluate_response(self, item: CommonsenseItem, 
                         response: str) -> Dict[str, Any]:
        """
        Evaluate a response.

        Args:
            item: Commonsense reasoning question.
            response: Model response.

        Returns:
            Dict: Evaluation result.
        """
        response_clean = response.upper().strip()
        
        # Attempt to extract the option letter
        correct_letter = chr(65 + item.correct_answer)
        
        is_correct = False
        selected_option = None
        
        # Check whether the response contains the correct option
        for i, opt in enumerate(item.options):
            letter = chr(65 + i)
            if letter in response_clean or opt.lower() in response.lower():
                selected_option = i
                if i == item.correct_answer:
                    is_correct = True
                break
        
        return {
            "is_correct": is_correct,
            "selected_option": selected_option,
            "correct_option": item.correct_answer,
            "correct_letter": correct_letter,
            "commonsense_type": item.commonsense_type,
            "difficulty": item.difficulty
        }
    
    def format_for_experiment(self, item: CommonsenseItem) -> Dict[str, Any]:
        """
        Format a commonsense item for the experiment.

        Args:
            item: Commonsense reasoning question.

        Returns:
            Dict: Experiment-format data.
        """
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item.options)])
        
        return {
            "id": f"commonsense_{self.custom_items.index(item):03d}",
            "question": f"{item.question}\n\n{options_text}",
            "correct_answer": chr(65 + item.correct_answer),
            "task_type": "commonsense",
            "source": "Custom",
            "options": item.options,
            "metadata": {
                "commonsense_type": item.commonsense_type,
                "difficulty": item.difficulty
            }
        }
