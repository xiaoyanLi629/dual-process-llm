"""
System 1: Fast Intuitive Thinking

Simulates the human fast, automatic, intuitive thinking process.

Features:
- Fast response
- Low cognitive effort
- Susceptible to cognitive bias
- Based on pattern matching and heuristics
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
from systems.actr_buffers import SensoryBuffer, ProductionSystem


@dataclass
class System1Response:
    """Response data structure for System 1."""
    answer: str
    confidence: float
    response_time_ms: float
    tokens_used: int
    reasoning_type: str = "intuitive"
    metadata: Dict[str, Any] = field(default_factory=dict)


class System1:
    """
    System 1: Fast Intuitive Thinking System

    Implementation strategy:
    - Use a smaller/faster model (e.g., GPT-4o-mini)
    - Higher temperature to increase stochasticity
    - Limit output length
    - Zero-shot prompting, no examples provided
    - Simplified cognitive path: Input -> Sensory Buffer -> Production System -> Output
    """
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.9,
                 max_tokens: int = 150,
                 strategy: str = "model_diff"):
        """
        Initialize System 1.

        Args:
            model_name: LLM model to use.
            temperature: Generation temperature (relatively high to simulate non-deterministic intuition).
            max_tokens: Maximum output token count (limited to simulate fast response).
            strategy: Implementation strategy ("model_diff", "param_diff", "prompt_diff").
        """
        self.client = get_openai_client()
        self.strategy = strategy
        
        # Configure model parameters based on selected strategy
        if strategy == "model_diff":
            # Strategy A: Use a smaller model
            self.model_config = {
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        elif strategy == "param_diff":
            # Strategy B: Same model with different parameters
            self.model_config = {
                "model": "gpt-4o",
                "temperature": 1.0,
                "max_tokens": 50,
                "top_p": 0.3
            }
        else:
            # Strategy C: Prompt-only differentiation
            self.model_config = {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 100
            }
        
        # Initialize ACT-R components
        self.sensory_buffer = SensoryBuffer()
        self.production_system = ProductionSystem(
            model_name=self.model_config["model"],
            temperature=self.model_config["temperature"]
        )
        
        # System prompt: emphasize fast, intuitive response
        self.system_prompt = """You are simulating System 1 thinking - fast, intuitive, automatic.

Rules:
1. Answer IMMEDIATELY with your first instinct
2. Do NOT overthink or analyze deeply
3. Give a brief, direct answer
4. Trust your gut feeling
5. One or two sentences maximum

Respond in JSON format: {"answer": "your answer", "confidence": 0.0-1.0}"""
        
        # Record processing history
        self.processing_history: List[Dict] = []
        self.total_tokens_used: int = 0
        self.total_api_calls: int = 0
    
    def process(self, question: str, task_type: str = "general") -> System1Response:
        """
        Process an input question and generate an intuitive response.

        Args:
            question: Input question.
            task_type: Task type.

        Returns:
            System1Response: System response.
        """
        start_time = time.time()
        
        # Step 1: Sensory Buffer receives input
        self.sensory_buffer.receive(question, input_type="text")
        
        # Step 2: Directly generate response via Production System (skipping goal analysis and knowledge retrieval)
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model"],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            self.total_api_calls += 1
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            system_response = System1Response(
                answer=result.get("answer", ""),
                confidence=float(result.get("confidence", 0.5)),
                response_time_ms=response_time,
                tokens_used=tokens_used,
                reasoning_type="intuitive",
                metadata={
                    "task_type": task_type,
                    "strategy": self.strategy,
                    "model": self.model_config["model"]
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            system_response = System1Response(
                answer=f"Error: {str(e)}",
                confidence=0.0,
                response_time_ms=response_time,
                tokens_used=0,
                reasoning_type="error",
                metadata={"error": str(e)}
            )
        
        # Record processing history
        self.processing_history.append({
            "question": question,
            "response": system_response.__dict__,
            "timestamp": datetime.now().isoformat()
        })
        
        return system_response
    
    def process_batch(self, questions: List[str], task_type: str = "general") -> List[System1Response]:
        """
        Process a batch of questions.

        Args:
            questions: List of questions.
            task_type: Task type.

        Returns:
            List[System1Response]: List of responses.
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
                "reasoning_steps": 0  # System 1 has no explicit reasoning steps
            }
        
        response_times = [h["response"]["response_time_ms"] for h in self.processing_history]
        
        return {
            "total_tokens": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "avg_tokens_per_call": self.total_tokens_used / max(1, self.total_api_calls),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "reasoning_steps": 0,  # System 1 has no explicit reasoning steps
            "strategy": self.strategy
        }
    
    def reset(self):
        """Reset system state."""
        self.sensory_buffer.clear()
        self.processing_history = []
        self.total_tokens_used = 0
        self.total_api_calls = 0


class System1WithHeuristics(System1):
    """
    System 1 with heuristic rules.

    Extends the base System 1 with common cognitive heuristics:
    - Availability heuristic
    - Representativeness heuristic
    - Anchoring effect
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Extend the system prompt to include heuristic tendencies
        self.system_prompt = """You are simulating System 1 thinking with cognitive heuristics.

Behavioral tendencies (simulate these naturally):
1. Availability: Favor information that comes to mind easily
2. Representativeness: Judge by similarity to stereotypes
3. Anchoring: Be influenced by initial numbers/information
4. Affect: Let emotions influence judgment

Rules:
1. Answer IMMEDIATELY with your first instinct
2. Do NOT correct for biases - embrace them
3. Give a brief, direct answer
4. Trust your gut feeling

Respond in JSON format: {"answer": "your answer", "confidence": 0.0-1.0, "heuristic_used": "name of heuristic if applicable"}"""
    
    def process(self, question: str, task_type: str = "general") -> System1Response:
        """Process a question, potentially subject to heuristic bias."""
        response = super().process(question, task_type)

        # Attempt to identify which heuristic was applied
        if "heuristic_used" in response.metadata:
            response.metadata["heuristic_detected"] = True
        
        return response
