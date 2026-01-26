"""
Cognitive Reflection Test (CRT) Tasks
Cognitive Reflection Test Tasks

CRTis a classic task for testing dual process theory，designed to trigger intuitive errors。
Correct answers require suppressing intuitive responses and engaging in analytical thinking。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class CRTItem:
    """CRTQuestion data structure"""
    question: str
    intuitive_answer: str  # Intuitive（error）answer
    correct_answer: str    # correctanswer
    explanation: str       # explanation
    difficulty: str = "medium"


class CRTTask:
    """
    Cognitive Reflection Test Tasks
    
    classicCRTquestions and variants，For testingSystem 1andSystem 2Differential
    """
    
    # classicCRTquestion
    CLASSIC_CRT = [
        CRTItem(
            question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            intuitive_answer="$0.10",
            correct_answer="$0.05",
            explanation="If ball = x, then bat = x + 1.00. So x + (x + 1.00) = 1.10, meaning 2x = 0.10, x = 0.05",
            difficulty="easy"
        ),
        CRTItem(
            question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            intuitive_answer="100 minutes",
            correct_answer="5 minutes",
            explanation="Each machine makes 1 widget in 5 minutes. So 100 machines make 100 widgets in 5 minutes.",
            difficulty="easy"
        ),
        CRTItem(
            question="In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            intuitive_answer="24 days",
            correct_answer="47 days",
            explanation="If it doubles each day and covers the whole lake on day 48, it covered half on day 47.",
            difficulty="easy"
        ),
    ]
    
    # ExtensionCRTquestion
    EXTENDED_CRT = [
        CRTItem(
            question="A farmer had 15 sheep, and all but 8 died. How many are left?",
            intuitive_answer="7",
            correct_answer="8",
            explanation="'All but 8 died' means 8 survived.",
            difficulty="easy"
        ),
        CRTItem(
            question="Emily's father has three daughters. The first two are named April and May. What is the third daughter's name?",
            intuitive_answer="June",
            correct_answer="Emily",
            explanation="The question states Emily's father, so Emily is the third daughter.",
            difficulty="easy"
        ),
        CRTItem(
            question="A clerk at a butcher shop stands five feet ten inches tall and wears size 13 sneakers. What does he weigh?",
            intuitive_answer="About 180 pounds",
            correct_answer="Meat",
            explanation="A butcher weighs meat - it's a play on words.",
            difficulty="medium"
        ),
        CRTItem(
            question="How many cubic feet of dirt are there in a hole that is 3 feet deep, 3 feet wide, and 3 feet long?",
            intuitive_answer="27 cubic feet",
            correct_answer="0 (none)",
            explanation="A hole contains no dirt - it's empty by definition.",
            difficulty="medium"
        ),
        CRTItem(
            question="If you have a bowl with six apples and you take away four, how many do you have?",
            intuitive_answer="2",
            correct_answer="4",
            explanation="You took 4 apples, so you have 4 apples.",
            difficulty="medium"
        ),
    ]
    
    # mathInferenceCRTvariant
    MATH_CRT = [
        CRTItem(
            question="A store is having a 25% off sale. If an item originally costs $80, and you have a coupon for an additional 10% off the sale price, what is the final price?",
            intuitive_answer="$52 (thinking 35% off)",
            correct_answer="$54",
            explanation="25% off $80 = $60. Then 10% off $60 = $54. Not 35% off the original.",
            difficulty="medium"
        ),
        CRTItem(
            question="If you're running a race and you pass the person in 2nd place, what place are you in?",
            intuitive_answer="1st place",
            correct_answer="2nd place",
            explanation="You passed the person in 2nd, so you take their place (2nd).",
            difficulty="easy"
        ),
    ]
    
    def __init__(self):
        """InitializeCRTtask"""
        self.all_items = self.CLASSIC_CRT + self.EXTENDED_CRT + self.MATH_CRT
    
    def get_classic_crt(self) -> List[CRTItem]:
        """GetclassicCRTquestion"""
        return self.CLASSIC_CRT
    
    def get_all_items(self) -> List[CRTItem]:
        """GetallHasCRTquestion"""
        return self.all_items
    
    def get_by_difficulty(self, difficulty: str) -> List[CRTItem]:
        """According todifficultyGetquestion"""
        return [item for item in self.all_items if item.difficulty == difficulty]
    
    def evaluate_response(self, item: CRTItem, response: str) -> Dict[str, Any]:
        """
        EvaluateResponse
        
        Args:
            item: CRTquestion
            response: ModelResponse
            
        Returns:
            Dict: Evaluation result
        """
        # cleanResponsetext
        response_clean = response.lower().strip()
        correct_clean = item.correct_answer.lower().strip()
        intuitive_clean = item.intuitive_answer.lower().strip()
        
        # Extractdigit（IfHas）
        response_numbers = re.findall(r'\d+\.?\d*', response_clean)
        correct_numbers = re.findall(r'\d+\.?\d*', correct_clean)
        intuitive_numbers = re.findall(r'\d+\.?\d*', intuitive_clean)
        
        # judgeIswhether correct
        is_correct = False
        is_intuitive = False
        
        # CheckcompletelyMatch
        if correct_clean in response_clean:
            is_correct = True
        elif response_numbers and correct_numbers:
            # CheckdigitMatch
            if any(rn == cn for rn in response_numbers for cn in correct_numbers):
                is_correct = True
        
        # CheckIswhethergiveIntuitiveanswer
        if intuitive_clean in response_clean:
            is_intuitive = True
        elif response_numbers and intuitive_numbers:
            if any(rn == in_ for rn in response_numbers for in_ in intuitive_numbers):
                is_intuitive = True
        
        return {
            "is_correct": is_correct,
            "is_intuitive_error": is_intuitive and not is_correct,
            "response": response,
            "correct_answer": item.correct_answer,
            "intuitive_answer": item.intuitive_answer,
            "difficulty": item.difficulty
        }
    
    def format_for_experiment(self, item: CRTItem) -> Dict[str, Any]:
        """
        FormatForexperimentFormat
        
        Args:
            item: CRTquestion
            
        Returns:
            Dict: experimentFormatData
        """
        return {
            "id": f"crt_{self.all_items.index(item):03d}",
            "question": item.question,
            "correct_answer": item.correct_answer,
            "intuitive_answer": item.intuitive_answer,
            "task_type": "cognitive_reflection",
            "source": "CRT",
            "difficulty": item.difficulty,
            "metadata": {
                "explanation": item.explanation,
                "is_classic": item in self.CLASSIC_CRT
            }
        }
