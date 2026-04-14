"""
ACT-R Cognitive Architecture Buffers

Implements the ACT-R cognitive architecture buffer modules.

ACT-R (Adaptive Control of Thought-Rational) is a cognitive architecture
containing multiple modules and buffers to simulate human cognitive processes.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path for api_config access
sys.path.append(str(Path(__file__).parent.parent.parent))
from api_config import get_openai_client, get_model_config


@dataclass
class BufferContent:
    """Buffer content data structure"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation: float = 1.0  # ACT-R activation level


class SensoryBuffer:
    """
    Sensory Buffer.

    Receives and preprocesses input information, similar to human sensory memory.
    In ACT-R, this is the entry point for information entering the cognitive system.
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize the Sensory Buffer.

        Args:
            capacity: Buffer capacity (default 7, following Miller's Law).
        """
        self.capacity = capacity
        self.buffer: List[BufferContent] = []
        self.processing_log: List[Dict] = []
    
    def receive(self, input_text: str, input_type: str = "text") -> BufferContent:
        """
        Receive input information.

        Args:
            input_text: Input text.
            input_type: Input type (text, image, audio, etc.).

        Returns:
            BufferContent: Processed buffer content.
        """
        content = BufferContent(
            content=input_text,
            metadata={
                "input_type": input_type,
                "length": len(input_text),
                "word_count": len(input_text.split())
            }
        )
        
        # Maintain capacity limit
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the oldest item
        
        self.buffer.append(content)
        
        self.processing_log.append({
            "action": "receive",
            "timestamp": datetime.now().isoformat(),
            "content_preview": input_text[:100] + "..." if len(input_text) > 100 else input_text
        })
        
        return content
    
    def get_current(self) -> Optional[BufferContent]:
        """Get the current (latest) buffer content."""
        return self.buffer[-1] if self.buffer else None
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []


class GoalBuffer:
    """
    Goal Buffer.

    Maintains the current cognitive goal and task status.
    In System 2, this module helps decompose complex questions and track solving progress.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        Initialize the Goal Buffer.

        Args:
            model_name: LLM model to use.
            temperature: Generation temperature.
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.current_goal: Optional[Dict] = None
        self.goal_stack: List[Dict] = []
        self.processing_log: List[Dict] = []
    
    def analyze_goal(self, question: str) -> Dict[str, Any]:
        """
        Analyze the question and set the cognitive goal.

        Args:
            question: Input question.

        Returns:
            Dict: Goal analysis result.
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
        """Get the current goal."""
        return self.current_goal
    
    def complete_goal(self):
        """Mark the current goal as complete."""
        if self.current_goal:
            self.current_goal["status"] = "completed"
            if self.goal_stack:
                self.goal_stack[-1]["status"] = "completed"


class DeclarativeModule:
    """
    Declarative Memory Module.

    Stores and retrieves factual knowledge. In System 2, this module provides
    all the background knowledge and relevant experience needed to solve a question.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        """
        Initialize the Declarative Memory Module.

        Args:
            model_name: LLM model to use.
            temperature: Generation temperature.
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.knowledge_base: List[Dict] = []
        self.retrieval_history: List[Dict] = []
    
    def retrieve_knowledge(self, question: str, goal_analysis: Dict) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge based on the question and goal analysis.

        Args:
            question: Input question.
            goal_analysis: Goal analysis result.

        Returns:
            Dict: Retrieved relevant knowledge.
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
        """Store new knowledge in the knowledge base."""
        self.knowledge_base.append({
            "knowledge": knowledge,
            "timestamp": datetime.now().isoformat(),
            "activation": 1.0
        })


class ProductionSystem:
    """
    Production System.

    Executes cognitive operations and generates responses. This is the "execution"
    module in ACT-R, generating actions according to the current state and rules.
    """
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3):
        """
        Initialize the Production System.

        Args:
            model_name: LLM model to use.
            temperature: Generation temperature.
        """
        self.client = get_openai_client()
        self.model_config = get_model_config(model_name, temperature)
        self.production_rules: List[Dict] = []
        self.execution_history: List[Dict] = []
    
    def execute(self, question: str, goal_analysis: Dict = None, 
                knowledge: Dict = None, mode: str = "analytical") -> Dict[str, Any]:
        """
        Execute a cognitive operation and generate a response.

        Args:
            question: Input question.
            goal_analysis: Goal analysis result (optional).
            knowledge: Retrieved knowledge (optional).
            mode: Execution mode ("intuitive" or "analytical").

        Returns:
            Dict: Execution result.
        """
        if mode == "intuitive":
            return self._execute_intuitive(question)
        else:
            return self._execute_analytical(question, goal_analysis, knowledge)
    
    def _execute_intuitive(self, question: str) -> Dict[str, Any]:
        """Execute in intuitive mode (System 1 style)."""
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
        """Execute in analytical mode (System 2 style)."""
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
    Retrieval Buffer.

    Acts as an interface between the Declarative Memory and other modules,
    temporarily storing retrieved information for use by other modules.
    """
    
    def __init__(self, capacity: int = 5):
        """
        Initialize the Retrieval Buffer.

        Args:
            capacity: Buffer capacity.
        """
        self.capacity = capacity
        self.buffer: List[BufferContent] = []
    
    def store(self, content: Any, source: str = "declarative"):
        """Store retrieved content in the buffer."""
        buffer_content = BufferContent(
            content=str(content),
            metadata={"source": source}
        )
        
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        self.buffer.append(buffer_content)
    
    def get_all(self) -> List[BufferContent]:
        """Get all buffered content."""
        return self.buffer
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
