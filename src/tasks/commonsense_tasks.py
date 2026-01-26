"""
Commonsense Reasoning Tasks
Commonsense Reasoning Tasks

For testingSystem 1intuitive reasoning ability
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
    Commonsense Reasoning Tasks
    
    ContainsMultitypes of commonsenseReasoning type：
    - physical commonsense (Physical)
    - social commonsense (Social)
    - Timecommonsense (Temporal)
    - causal commonsense (Causal)
    """
    
    def __init__(self, task_loader=None):
        """
        InitializeCommonsense Reasoning Tasks
        
        Args:
            task_loader: Task Loader
        """
        self.task_loader = task_loader
        self.custom_items: List[CommonsenseItem] = []
        
        # AddSomeExamplequestion
        self._add_sample_items()
    
    def _add_sample_items(self):
        """AddExamplecommonsenseInferencequestion"""
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
        GetcommonsenseInferencequestion
        
        Args:
            commonsense_type: commonsenseTypeFilter
            difficulty: difficultyFilter
            
        Returns:
            List[CommonsenseItem]: questionList
        """
        items = self.custom_items.copy()
        
        if commonsense_type:
            items = [i for i in items if i.commonsense_type == commonsense_type]
        
        if difficulty:
            items = [i for i in items if i.difficulty == difficulty]
        
        return items
    
    def get_from_dataset(self, n: int = 100, source: str = None) -> List[Dict]:
        """
        FromDatasetGetcommonsenseInferencequestion
        
        Args:
            n: questionCount
            source: DatasourceFilter
            
        Returns:
            List[Dict]: questionList
        """
        if self.task_loader is None:
            return []
        
        # FromSystem 1taskinfiltercommonsenseInferencequestion
        if source:
            tasks = self.task_loader.get_system1_tasks(source_filter=source)
        else:
            tasks = self.task_loader.get_system1_tasks()
        
        return [t.to_dict() for t in tasks[:n]]
    
    def evaluate_response(self, item: CommonsenseItem, 
                         response: str) -> Dict[str, Any]:
        """
        EvaluateResponse
        
        Args:
            item: commonsenseInferencequestion
            response: ModelResponse
            
        Returns:
            Dict: Evaluation result
        """
        response_clean = response.upper().strip()
        
        # attemptExtractOptionletter
        correct_letter = chr(65 + item.correct_answer)
        
        is_correct = False
        selected_option = None
        
        # CheckResponseinIswhetherContainscorrectOption
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
        FormatForexperimentFormat
        
        Args:
            item: commonsenseInferencequestion
            
        Returns:
            Dict: experimentFormatData
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
