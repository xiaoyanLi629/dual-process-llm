"""
Logical Reasoning Tasks
Logical Reasoning Tasks

For testingSystem 2analytical reasoning ability
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LogicalReasoningItem:
    """Logical reasoning question data structure"""
    context: str
    question: str
    options: List[str]
    correct_answer: int  # Correct option index
    reasoning_type: str  # Reasoning type
    difficulty: str = "medium"


class LogicalReasoningTask:
    """
    Logical Reasoning Tasks
    
    ContainsMultitypes of logicalReasoning type：
    - deductiveInference (Deductive)
    - inductiveInference (Inductive)
    - causalInference (Causal)
    - conditionalInference (Conditional)
    """
    
    def __init__(self, task_loader=None):
        """
        InitializeLogical Reasoning Tasks
        
        Args:
            task_loader: Task Loader（Used forLoadLogiQAetcData）
        """
        self.task_loader = task_loader
        self.custom_items: List[LogicalReasoningItem] = []
        
        # AddSomeExamplequestion
        self._add_sample_items()
    
    def _add_sample_items(self):
        """AddExamplelogicInferencequestion"""
        self.custom_items = [
            LogicalReasoningItem(
                context="All mammals are warm-blooded. All whales are mammals.",
                question="Based on the above statements, which of the following must be true?",
                options=[
                    "All warm-blooded animals are mammals",
                    "All whales are warm-blooded",
                    "Some warm-blooded animals are not mammals",
                    "No whales are cold-blooded"
                ],
                correct_answer=1,
                reasoning_type="deductive",
                difficulty="easy"
            ),
            LogicalReasoningItem(
                context="If it rains, the ground gets wet. The ground is wet.",
                question="What can we conclude?",
                options=[
                    "It definitely rained",
                    "It might have rained",
                    "It did not rain",
                    "The ground is always wet"
                ],
                correct_answer=1,
                reasoning_type="conditional",
                difficulty="medium"
            ),
            LogicalReasoningItem(
                context="Every time John eats peanuts, he gets a rash. John has a rash today.",
                question="What is the most logical conclusion?",
                options=[
                    "John definitely ate peanuts today",
                    "John might have eaten peanuts today",
                    "John is allergic to all nuts",
                    "John should see a doctor"
                ],
                correct_answer=1,
                reasoning_type="causal",
                difficulty="medium"
            ),
            LogicalReasoningItem(
                context="In a group of 100 people, 70 like coffee, 80 like tea, and everyone likes at least one of these drinks.",
                question="How many people like both coffee and tea?",
                options=[
                    "10",
                    "30",
                    "50",
                    "Cannot be determined"
                ],
                correct_answer=2,
                reasoning_type="deductive",
                difficulty="hard"
            ),
        ]
    
    def get_items(self, reasoning_type: str = None, 
                  difficulty: str = None) -> List[LogicalReasoningItem]:
        """
        GetlogicInferencequestion
        
        Args:
            reasoning_type: Reasoning typeFilter
            difficulty: difficultyFilter
            
        Returns:
            List[LogicalReasoningItem]: questionList
        """
        items = self.custom_items.copy()
        
        if reasoning_type:
            items = [i for i in items if i.reasoning_type == reasoning_type]
        
        if difficulty:
            items = [i for i in items if i.difficulty == difficulty]
        
        return items
    
    def get_from_dataset(self, n: int = 100) -> List[Dict]:
        """
        FromDatasetGetlogicInferencequestion
        
        Args:
            n: questionCount
            
        Returns:
            List[Dict]: questionList
        """
        if self.task_loader is None:
            return []
        
        # FromSystem 2taskinfilterlogicInferencequestion
        tasks = self.task_loader.get_system2_tasks(source_filter="LogiQA")
        return [t.to_dict() for t in tasks[:n]]
    
    def evaluate_response(self, item: LogicalReasoningItem, 
                         response: str) -> Dict[str, Any]:
        """
        EvaluateResponse
        
        Args:
            item: logicInferencequestion
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
            "reasoning_type": item.reasoning_type,
            "difficulty": item.difficulty
        }
    
    def format_for_experiment(self, item: LogicalReasoningItem) -> Dict[str, Any]:
        """
        FormatForexperimentFormat
        
        Args:
            item: logicInferencequestion
            
        Returns:
            Dict: experimentFormatData
        """
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item.options)])
        
        return {
            "id": f"logic_{self.custom_items.index(item):03d}",
            "question": f"{item.context}\n\n{item.question}\n\n{options_text}",
            "correct_answer": chr(65 + item.correct_answer),
            "task_type": "logical_reasoning",
            "source": "Custom",
            "options": item.options,
            "metadata": {
                "reasoning_type": item.reasoning_type,
                "difficulty": item.difficulty,
                "context": item.context
            }
        }
