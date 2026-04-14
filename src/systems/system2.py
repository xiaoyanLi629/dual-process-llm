"""
System 2: Slow Analytical Thinking

Simulates the human slow, controlled, analytical thinking process.

Features:
- Requires focused attention
- High cognitive effort
- Capable of logical inference
- Can inhibit intuitive errors
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path for api_config access
sys.path.append(str(Path(__file__).parent.parent.parent))
from api_config import get_openai_client, get_model_config
from systems.actr_buffers import (
    SensoryBuffer, GoalBuffer, DeclarativeModule, 
    ProductionSystem, RetrievalBuffer
)


@dataclass
class System2Response:
    """Response data structure for System 2."""
    answer: str
    confidence: float
    response_time_ms: float
    tokens_used: int
    reasoning_steps: List[str]
    reasoning_type: str = "analytical"
    goal_analysis: Dict = field(default_factory=dict)
    retrieved_knowledge: Dict = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class System2:
    """
    System 2: Slow Analytical Thinking System

    Implementation strategy:
    - Use a larger/stronger model (e.g., GPT-4o)
    - Lower temperature to increase determinism
    - Multi-module collaborative processing
    - Chain-of-thought reasoning
    - Full cognitive path:
      Input -> Sensory Buffer -> Goal Buffer -> Declarative Module ->
      Retrieval Buffer -> Production System -> Output
    """
    
    def __init__(self,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.2,
                 max_tokens: int = 2000,
                 use_cot: bool = True):
        """
        Initialize System 2.

        Args:
            model_name: LLM model to use.
            temperature: Generation temperature (lower to increase determinism).
            max_tokens: Maximum output token count.
            use_cot: Whether to use Chain-of-Thought reasoning.
        """
        self.client = get_openai_client()
        self.model_config = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.use_cot = use_cot
        
        # Initialize ACT-R components
        self.sensory_buffer = SensoryBuffer()
        self.goal_buffer = GoalBuffer(model_name=model_name, temperature=temperature)
        self.declarative_module = DeclarativeModule(model_name=model_name, temperature=temperature)
        self.retrieval_buffer = RetrievalBuffer()
        self.production_system = ProductionSystem(model_name=model_name, temperature=temperature)
        
        # System prompt: emphasize analytical thinking
        self.system_prompt = """You are simulating System 2 thinking - slow, deliberate, analytical.

Process:
1. Carefully analyze the problem structure
2. Identify potential cognitive traps
3. Apply relevant knowledge and principles
4. Reason step by step
5. Verify your answer before responding

Be thorough and methodical. Show your reasoning process.

Respond in JSON format:
{
    "reasoning_steps": ["step 1", "step 2", ...],
    "answer": "your final answer",
    "confidence": 0.0-1.0,
    "verification": "how you verified the answer"
}"""
        
        # Record processing history
        self.processing_history: List[Dict] = []
        self.total_tokens_used: int = 0
        self.total_api_calls: int = 0
    
    def process(self, question: str, task_type: str = "general") -> System2Response:
        """
        Process an input question and generate an analytical response.

        Args:
            question: Input question.
            task_type: Task type.

        Returns:
            System2Response: System response.
        """
        start_time = time.time()
        tokens_used = 0
        
        # Step 1: Sensory Buffer receives input
        self.sensory_buffer.receive(question, input_type="text")
        
        # Step 2: Goal Buffer analyzes the question
        goal_analysis = self.goal_buffer.analyze_goal(question)
        self.total_api_calls += 1
        
        # Step 3: Declarative Memory Module retrieves relevant knowledge
        retrieved_knowledge = self.declarative_module.retrieve_knowledge(question, goal_analysis)
        self.total_api_calls += 1
        
        # Step 4: Store retrieved knowledge in Retrieval Buffer
        self.retrieval_buffer.store(retrieved_knowledge, source="declarative")
        
        # Step 5: Production System generates the final response
        try:
            # Build the complete context
            context = self._build_context(question, goal_analysis, retrieved_knowledge)
            
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            self.total_api_calls += 1
            
            response_time = (time.time() - start_time) * 1000
            
            system_response = System2Response(
                answer=result.get("answer", ""),
                confidence=float(result.get("confidence", 0.5)),
                response_time_ms=response_time,
                tokens_used=tokens_used,
                reasoning_steps=result.get("reasoning_steps", []),
                reasoning_type="analytical",
                goal_analysis=goal_analysis,
                retrieved_knowledge=retrieved_knowledge,
                metadata={
                    "task_type": task_type,
                    "model": self.model_config["model"],
                    "use_cot": self.use_cot,
                    "verification": result.get("verification", "")
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            system_response = System2Response(
                answer=f"Error: {str(e)}",
                confidence=0.0,
                response_time_ms=response_time,
                tokens_used=tokens_used,
                reasoning_steps=[],
                reasoning_type="error",
                goal_analysis=goal_analysis,
                retrieved_knowledge=retrieved_knowledge,
                metadata={"error": str(e)}
            )
        
        # Mark goal as complete
        self.goal_buffer.complete_goal()
        
        # Record processing history
        self.processing_history.append({
            "question": question,
            "response": {
                "answer": system_response.answer,
                "confidence": system_response.confidence,
                "response_time_ms": system_response.response_time_ms,
                "tokens_used": system_response.tokens_used,
                "reasoning_steps": system_response.reasoning_steps
            },
            "timestamp": datetime.now().isoformat()
        })
        
        return system_response
    
    def _build_context(self, question: str, goal_analysis: Dict, 
                       retrieved_knowledge: Dict) -> str:
        """
        Build the complete reasoning context.

        Args:
            question: Input question.
            goal_analysis: Goal analysis result.
            retrieved_knowledge: Retrieved knowledge.

        Returns:
            str: Complete context string.
        """
        context_parts = [
            f"## Question\n{question}",
            f"\n## Goal Analysis\n"
        ]
        
        # Add goal analysis
        if goal_analysis:
            context_parts.append(f"- Core Question: {goal_analysis.get('core_question', 'N/A')}")
            context_parts.append(f"- Reasoning Type: {goal_analysis.get('reasoning_type', 'N/A')}")
            
            traps = goal_analysis.get('cognitive_traps', [])
            if traps:
                context_parts.append(f"- Cognitive Traps to Avoid: {', '.join(traps) if isinstance(traps, list) else traps}")
            
            subgoals = goal_analysis.get('subgoals', [])
            if subgoals:
                context_parts.append(f"- Subgoals: {subgoals}")
        
        # Add retrieved knowledge
        context_parts.append(f"\n## Retrieved Knowledge")
        
        if retrieved_knowledge:
            facts = retrieved_knowledge.get('relevant_facts', [])
            if facts:
                context_parts.append(f"- Relevant Facts: {facts}")
            
            principles = retrieved_knowledge.get('principles', [])
            if principles:
                context_parts.append(f"- Applicable Principles: {principles}")
            
            mistakes = retrieved_knowledge.get('common_mistakes', [])
            if mistakes:
                context_parts.append(f"- Common Mistakes to Avoid: {mistakes}")
            
            strategies = retrieved_knowledge.get('solution_strategies', [])
            if strategies:
                context_parts.append(f"- Recommended Strategies: {strategies}")
        
        context_parts.append("\n## Task\nAnalyze this problem carefully and provide a well-reasoned answer.")
        
        return "\n".join(context_parts)
    
    def process_batch(self, questions: List[str], task_type: str = "general") -> List[System2Response]:
        """
        Process a batch of questions.

        Args:
            questions: List of questions.
            task_type: Task type.

        Returns:
            List[System2Response]: List of responses.
        """
        responses = []
        for question in questions:
            response = self.process(question, task_type)
            responses.append(response)
        return responses
    
    def get_cognitive_effort_metrics(self) -> Dict[str, Any]:
        """
        Get cognitive effort metrics.

        Returns:
            Dict: Cognitive effort metrics.
        """
        if not self.processing_history:
            return {
                "total_tokens": 0,
                "total_api_calls": 0,
                "avg_tokens_per_call": 0,
                "avg_response_time_ms": 0,
                "avg_reasoning_steps": 0
            }
        
        response_times = [h["response"]["response_time_ms"] for h in self.processing_history]
        reasoning_steps = [len(h["response"]["reasoning_steps"]) for h in self.processing_history]
        
        return {
            "total_tokens": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "avg_tokens_per_call": self.total_tokens_used / max(1, self.total_api_calls),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "avg_reasoning_steps": sum(reasoning_steps) / len(reasoning_steps),
            "model": self.model_config["model"],
            "use_cot": self.use_cot
        }
    
    def reset(self):
        """Reset system state."""
        self.sensory_buffer.clear()
        self.retrieval_buffer.clear()
        self.processing_history = []
        self.total_tokens_used = 0
        self.total_api_calls = 0


class System2WithMetacognition(System2):
    """
    System 2 with metacognitive ability.

    Extends the base System 2 with metacognitive monitoring:
    - Self-monitor the reasoning process
    - Detect potential errors
    - Adjust confidence
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Extend the system prompt to include metacognitive monitoring
        self.system_prompt = """You are simulating System 2 thinking with metacognitive monitoring.

Process:
1. Carefully analyze the problem structure
2. Identify potential cognitive traps
3. Apply relevant knowledge and principles
4. Reason step by step
5. MONITOR your own reasoning for errors
6. EVALUATE confidence based on reasoning quality
7. Verify your answer before responding

Be thorough and self-aware. Question your own assumptions.

Respond in JSON format:
{
    "reasoning_steps": ["step 1", "step 2", ...],
    "answer": "your final answer",
    "confidence": 0.0-1.0,
    "metacognitive_notes": ["observations about your own reasoning"],
    "potential_errors": ["possible mistakes you might be making"],
    "verification": "how you verified the answer"
}"""
    
    def process(self, question: str, task_type: str = "general") -> System2Response:
        """Process a question with metacognitive monitoring."""
        response = super().process(question, task_type)

        # Add metacognitive information to metadata
        if "metacognitive_notes" in response.metadata:
            response.metadata["metacognition_active"] = True
        
        return response
