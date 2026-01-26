"""
ACT-R Cognitive Architecture Buffers
ImplementACT-Rcognitive architecture buffer modules

ACT-R (Adaptive Control of Thought-Rational) is a cognitive architecture，
Contains multiple modules and buffers to simulate human cognitive processes。
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path for api_config access
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from api_config import get_openai_client, get_model_config


@dataclass
class BufferContent:
    """Buffer content data structure"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation: float = 1.0  # ACT-RActivation level


class SensoryBuffer:
    """
    Sensory Buffer (Sensory Buffer)
    
    Responsible forreceiveandpreliminaryProcessInputInfo，SimilarhumanClasssensoryMemory。
    InACT-Rin，thisIsInfoentercognitiveSystementry。
    """
    
    def __init__(self, capacity: int = 7):
        """
        InitializeSensory Buffer
        
        Args:
            capacity: BufferCapacity（Default7，conform toMiller's Law）
        """
        self.capacity = capacity
        self.buffer: List[BufferContent] = []
        self.processing_log: List[Dict] = []
    
    def receive(self, input_text: str, input_type: str = "text") -> BufferContent:
        """
        receiveInputInfo
        
        Args:
            input_text: Inputtext
            input_type: InputType（text, image, audioetc）
            
        Returns:
            BufferContent: ProcessafterBuffercontent
        """
        content = BufferContent(
            content=input_text,
            metadata={
                "input_type": input_type,
                "length": len(input_text),
                "word_count": len(input_text.split())
            }
        )
        
        # maintainCapacityLimit
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # RemovemostOlditem
        
        self.buffer.append(content)
        
        self.processing_log.append({
            "action": "receive",
            "timestamp": datetime.now().isoformat(),
            "content_preview": input_text[:100] + "..." if len(input_text) > 100 else input_text
        })
        
        return content
    
    def get_current(self) -> Optional[BufferContent]:
        """GetCurrent（latest）Buffercontent"""
        return self.buffer[-1] if self.buffer else None
    
    def clear(self):
        """ClearBuffer"""
        self.buffer = []


class GoalBuffer:
    """
    Goal Buffer (Goal Buffer)
    
    Responsible formaintainCurrentcognitivetargetandtaskStatus。
    InSystem 2in，thisModulehelpdecomposecomplexquestion andtrack solving progressdegree。
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        InitializeGoal Buffer
        
        Args:
            model_name: UsingLLMModel
            temperature: Generatetemperature
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.current_goal: Optional[Dict] = None
        self.goal_stack: List[Dict] = []
        self.processing_log: List[Dict] = []
    
    def analyze_goal(self, question: str) -> Dict[str, Any]:
        """
        Analysisquestion andsetcognitivetarget
        
        Args:
            question: Inputquestion
            
        Returns:
            Dict: targetAnalysisResult
        """
        system_prompt = """You are the Goal Buffer in an ACT-R cognitive architecture.
Your role is to analyze problems and identify cognitive goals.

For the given question, provide a structured analysis:
1. Core Question: What is being asked?
2. Reasoning Type: What type of reasoning is required? (logical, mathematical, causal, etc.)
3. Cognitive Traps: What intuitive errors should be avoided?
4. Goal State: What would a correct answer look like?
5. Subgoals: Break down into smaller steps if needed.

Respond in JSON format with keys: core_question, reasoning_type, cognitive_traps, goal_state, subgoals"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            goal_analysis = json.loads(response.choices[0].message.content)
            
            self.current_goal = {
                "question": question,
                "analysis": goal_analysis,
                "timestamp": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.goal_stack.append(self.current_goal)
            
            self.processing_log.append({
                "action": "analyze_goal",
                "timestamp": datetime.now().isoformat(),
                "goal": goal_analysis
            })
            
            return goal_analysis
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "core_question": question,
                "reasoning_type": "unknown",
                "cognitive_traps": [],
                "goal_state": "unknown",
                "subgoals": []
            }
            self.current_goal = {"question": question, "analysis": error_result, "status": "error"}
            return error_result
    
    def get_current_goal(self) -> Optional[Dict]:
        """GetCurrenttarget"""
        return self.current_goal
    
    def complete_goal(self):
        """markerCurrenttargetForComplete"""
        if self.current_goal:
            self.current_goal["status"] = "completed"
            if self.goal_stack:
                self.goal_stack[-1]["status"] = "completed"


class DeclarativeModule:
    """
    Declarative MemoryModule (Declarative Module)
    
    Responsible forStoreandRetrievefactknowledge。InSystem 2in，thisModuleprovide
    solvequestionallneed backgroundknowledgeandCorrelationExperience。
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        """
        InitializeDeclarative MemoryModule
        
        Args:
            model_name: UsingLLMModel
            temperature: Generatetemperature
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.knowledge_base: List[Dict] = []
        self.retrieval_history: List[Dict] = []
    
    def retrieve_knowledge(self, question: str, goal_analysis: Dict) -> Dict[str, Any]:
        """
        According toquestionandtargetRetrieveCorrelationknowledge
        
        Args:
            question: Inputquestion
            goal_analysis: targetAnalysisResult
            
        Returns:
            Dict: RetrieveToCorrelationknowledge
        """
        system_prompt = """You are the Declarative Module in an ACT-R cognitive architecture.
Your role is to retrieve relevant knowledge for problem-solving.

Based on the question and goal analysis, retrieve:
1. Relevant Facts: Key facts needed to solve this problem
2. Principles: Mathematical, logical, or domain principles applicable
3. Similar Examples: Analogous problems and their solutions
4. Common Mistakes: Typical errors people make on similar problems
5. Solution Strategies: Recommended approaches

Respond in JSON format with keys: relevant_facts, principles, similar_examples, common_mistakes, solution_strategies"""

        context = f"""Question: {question}

Goal Analysis:
- Core Question: {goal_analysis.get('core_question', 'N/A')}
- Reasoning Type: {goal_analysis.get('reasoning_type', 'N/A')}
- Cognitive Traps to Avoid: {goal_analysis.get('cognitive_traps', [])}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            knowledge = json.loads(response.choices[0].message.content)
            
            self.retrieval_history.append({
                "question": question,
                "knowledge": knowledge,
                "timestamp": datetime.now().isoformat()
            })
            
            return knowledge
            
        except Exception as e:
            return {
                "error": str(e),
                "relevant_facts": [],
                "principles": [],
                "similar_examples": [],
                "common_mistakes": [],
                "solution_strategies": []
            }
    
    def store_knowledge(self, knowledge: Dict):
        """StorenewknowledgeToknowledgelibrary"""
        self.knowledge_base.append({
            "knowledge": knowledge,
            "timestamp": datetime.now().isoformat(),
            "activation": 1.0
        })


class ProductionSystem:
    """
    Production System (Production System)
    
    Responsible forExecutecognitiveoperationandGenerateResponse。thisIsACT-Rin"Execute"Module，
    According toCurrentStatusandRulegenerate actionFor。
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        InitializeProduction System
        
        Args:
            model_name: UsingLLMModel
            temperature: Generatetemperature
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.production_rules: List[Dict] = []
        self.execution_history: List[Dict] = []
    
    def execute(self, question: str, goal_analysis: Dict = None, 
                knowledge: Dict = None, mode: str = "analytical") -> Dict[str, Any]:
        """
        Executecognitiveoperate andGenerateResponse
        
        Args:
            question: Inputquestion
            goal_analysis: targetAnalysisResult（Optional）
            knowledge: RetrieveToknowledge（Optional）
            mode: Executepattern ("intuitive" or "analytical")
            
        Returns:
            Dict: ExecuteResult
        """
        if mode == "intuitive":
            return self._execute_intuitive(question)
        else:
            return self._execute_analytical(question, goal_analysis, knowledge)
    
    def _execute_intuitive(self, question: str) -> Dict[str, Any]:
        """IntuitivepatternExecute（System 1style）"""
        system_prompt = """Answer immediately with your first instinct.
Give a brief, direct answer without extensive reasoning.
Format: {"answer": "your answer", "confidence": 0.0-1.0}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=0.9,  # Higher temperature for intuitive responses
                max_tokens=150,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            result["mode"] = "intuitive"
            result["tokens_used"] = response.usage.total_tokens
            
            self.execution_history.append({
                "question": question,
                "result": result,
                "mode": "intuitive",
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e), "answer": None, "confidence": 0.0, "mode": "intuitive"}
    
    def _execute_analytical(self, question: str, goal_analysis: Dict = None,
                           knowledge: Dict = None) -> Dict[str, Any]:
        """AnalysispatternExecute（System 2style）"""
        system_prompt = """You are the Production System in an ACT-R cognitive architecture.
Use the provided goal analysis and knowledge to solve the problem step by step.

Provide:
1. Step-by-step reasoning
2. Final answer
3. Confidence level (0.0-1.0)
4. Reasoning chain explanation

Format your response as JSON with keys: reasoning_steps, answer, confidence, explanation"""

        context_parts = [f"Question: {question}"]
        
        if goal_analysis:
            context_parts.append(f"\nGoal Analysis:\n{goal_analysis}")
        
        if knowledge:
            context_parts.append(f"\nRelevant Knowledge:\n{knowledge}")
        
        context = "\n".join(context_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            result["mode"] = "analytical"
            result["tokens_used"] = response.usage.total_tokens
            
            self.execution_history.append({
                "question": question,
                "result": result,
                "mode": "analytical",
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "confidence": 0.0,
                "mode": "analytical",
                "reasoning_steps": [],
                "explanation": ""
            }


class RetrievalBuffer:
    """
    Retrieval Buffer (Retrieval Buffer)
    
    workForDeclarative MemoryandOtherModulebetweenInterface，
    temporary storageRetrieveToInfoprovideOtherModuleUsing。
    """
    
    def __init__(self, capacity: int = 5):
        """
        InitializeRetrieval Buffer
        
        Args:
            capacity: BufferCapacity
        """
        self.capacity = capacity
        self.buffer: List[BufferContent] = []
    
    def store(self, content: Any, source: str = "declarative"):
        """StoreRetrieveTocontent"""
        buffer_content = BufferContent(
            content=str(content),
            metadata={"source": source}
        )
        
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        self.buffer.append(buffer_content)
    
    def get_all(self) -> List[BufferContent]:
        """GetallHasBuffercontent"""
        return self.buffer
    
    def clear(self):
        """ClearBuffer"""
        self.buffer = []
