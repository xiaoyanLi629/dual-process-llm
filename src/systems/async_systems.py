"""
Async Systems for Parallel Processing
Asynchronous versions of System1 and System2 for faster experiment execution

Uses asyncio + httpx for concurrent API calls with rate limiting.
"""

import asyncio
import httpx
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add project root to path for api_config access
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from api_config import get_openai_client, get_model_config


@dataclass
class AsyncResponse:
    """Async response data structure"""
    answer: str
    confidence: float
    response_time_ms: float
    tokens_used: int
    reasoning_type: str
    reasoning_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AsyncSystem1:
    """
    Async System 1: Fast Intuitive Thinking
    Supports concurrent API calls for faster processing
    """
    
    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.9,
                 max_tokens: int = 150,
                 max_concurrent: int = 20):
        """
        Initialize Async System 1
        
        Args:
            model_name: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            max_concurrent: Maximum concurrent API calls
        """
        self.model_config = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.max_concurrent = max_concurrent
        
        # Get API configuration
        client = get_openai_client()
        self.api_key = client.api_key
        self.base_url = str(client.base_url).rstrip('/')
        
        self.system_prompt = """You are simulating System 1 thinking - fast, intuitive, automatic.

Rules:
1. Answer IMMEDIATELY with your first instinct
2. Do NOT overthink or analyze deeply
3. Give a brief, direct answer
4. Trust your gut feeling
5. One or two sentences maximum

Respond in JSON format: {"answer": "your answer", "confidence": 0.0-1.0}"""
        
        # Statistics
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.processing_history: List[Dict] = []
    
    async def process_single(self, client: httpx.AsyncClient, 
                            semaphore: asyncio.Semaphore,
                            question: str, task_type: str = "general") -> AsyncResponse:
        """Process a single question asynchronously"""
        start_time = time.time()
        
        async with semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": self.model_config["model"],
                    "temperature": self.model_config["temperature"],
                    "max_tokens": self.model_config["max_tokens"],
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": question}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                
                data = response.json()
                    
                result = json.loads(data["choices"][0]["message"]["content"])
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                self.total_tokens_used += tokens_used
                self.total_api_calls += 1
                
                response_time = (time.time() - start_time) * 1000
                
                return AsyncResponse(
                    answer=result.get("answer", ""),
                    confidence=float(result.get("confidence", 0.5)),
                    response_time_ms=response_time,
                    tokens_used=tokens_used,
                    reasoning_type="intuitive",
                    metadata={"task_type": task_type, "model": self.model_config["model"]}
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                return AsyncResponse(
                    answer="Error",
                    confidence=0.0,
                    response_time_ms=response_time,
                    tokens_used=0,
                    reasoning_type="error",
                    error=str(e)
                )
    
    async def process_batch(self, questions: List[str], 
                           task_types: List[str] = None) -> List[AsyncResponse]:
        """
        Process multiple questions concurrently
        
        Args:
            questions: List of questions
            task_types: List of task types (optional)
            
        Returns:
            List of responses
        """
        if task_types is None:
            task_types = ["general"] * len(questions)
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        limits = httpx.Limits(max_connections=self.max_concurrent, max_keepalive_connections=self.max_concurrent)
        async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
            tasks = [
                self.process_single(client, semaphore, q, t)
                for q, t in zip(questions, task_types)
            ]
            responses = await asyncio.gather(*tasks)
        
        return responses
    
    def get_cognitive_effort_metrics(self) -> Dict[str, Any]:
        """Get cognitive effort metrics"""
        return {
            "total_tokens": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "avg_tokens_per_call": self.total_tokens_used / max(1, self.total_api_calls),
            "avg_response_time_ms": 0,  # Will be calculated from responses
            "reasoning_steps": 0
        }
    
    def reset(self):
        """Reset system state"""
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.processing_history = []


class AsyncSystem2:
    """
    Async System 2: Slow Analytical Thinking
    Supports concurrent API calls for faster processing
    """
    
    def __init__(self,
                 model_name: str = "gpt-4o",
                 temperature: float = 0.2,
                 max_tokens: int = 2000,
                 use_cot: bool = True,
                 max_concurrent: int = 10):
        """
        Initialize Async System 2
        
        Args:
            model_name: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum output tokens
            use_cot: Whether to use Chain-of-Thought
            max_concurrent: Maximum concurrent API calls (lower for heavier model)
        """
        self.model_config = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        self.use_cot = use_cot
        self.max_concurrent = max_concurrent
        
        # Get API configuration
        client = get_openai_client()
        self.api_key = client.api_key
        self.base_url = str(client.base_url).rstrip('/')
        
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
        
        # Statistics
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.processing_history: List[Dict] = []
    
    async def process_single(self, client: httpx.AsyncClient,
                            semaphore: asyncio.Semaphore,
                            question: str, task_type: str = "general") -> AsyncResponse:
        """Process a single question asynchronously with full analytical pipeline"""
        start_time = time.time()
        
        async with semaphore:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                # Build context with goal analysis prompt
                context = f"""## Question
{question}

## Task
Analyze this problem carefully and provide a well-reasoned answer. Consider:
- What is the core question being asked?
- What cognitive traps should you avoid?
- What principles or knowledge apply here?
- Work through the problem step by step."""
                
                payload = {
                    "model": self.model_config["model"],
                    "temperature": self.model_config["temperature"],
                    "max_tokens": self.model_config["max_tokens"],
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": context}
                    ],
                    "response_format": {"type": "json_object"}
                }
                
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0  # Longer timeout for S2
                )
                
                if response.status_code != 200:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                
                data = response.json()
                
                result = json.loads(data["choices"][0]["message"]["content"])
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                self.total_tokens_used += tokens_used
                self.total_api_calls += 1
                
                response_time = (time.time() - start_time) * 1000
                
                return AsyncResponse(
                    answer=result.get("answer", ""),
                    confidence=float(result.get("confidence", 0.5)),
                    response_time_ms=response_time,
                    tokens_used=tokens_used,
                    reasoning_type="analytical",
                    reasoning_steps=result.get("reasoning_steps", []),
                    metadata={
                        "task_type": task_type,
                        "model": self.model_config["model"],
                        "use_cot": self.use_cot,
                        "verification": result.get("verification", "")
                    }
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                return AsyncResponse(
                    answer="Error",
                    confidence=0.0,
                    response_time_ms=response_time,
                    tokens_used=0,
                    reasoning_type="error",
                    error=str(e)
                )
    
    async def process_batch(self, questions: List[str],
                           task_types: List[str] = None) -> List[AsyncResponse]:
        """
        Process multiple questions concurrently
        
        Args:
            questions: List of questions
            task_types: List of task types (optional)
            
        Returns:
            List of responses
        """
        if task_types is None:
            task_types = ["general"] * len(questions)
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        limits = httpx.Limits(max_connections=self.max_concurrent, max_keepalive_connections=self.max_concurrent)
        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
            tasks = [
                self.process_single(client, semaphore, q, t)
                for q, t in zip(questions, task_types)
            ]
            responses = await asyncio.gather(*tasks)
        
        return responses
    
    def get_cognitive_effort_metrics(self) -> Dict[str, Any]:
        """Get cognitive effort metrics"""
        return {
            "total_tokens": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "avg_tokens_per_call": self.total_tokens_used / max(1, self.total_api_calls),
            "avg_response_time_ms": 0,
            "avg_reasoning_steps": 0,
            "model": self.model_config["model"],
            "use_cot": self.use_cot
        }
    
    def reset(self):
        """Reset system state"""
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.processing_history = []


class ParallelDualProcessRunner:
    """
    Parallel runner for dual process experiments
    Runs System 1 and System 2 concurrently on batches of tasks
    """
    
    def __init__(self,
                 system1_config: Dict = None,
                 system2_config: Dict = None,
                 s1_concurrent: int = 20,
                 s2_concurrent: int = 10):
        """
        Initialize parallel runner
        
        Args:
            system1_config: System 1 configuration
            system2_config: System 2 configuration
            s1_concurrent: Max concurrent calls for System 1
            s2_concurrent: Max concurrent calls for System 2
        """
        s1_config = system1_config or {}
        s2_config = system2_config or {}
        
        self.system1 = AsyncSystem1(
            max_concurrent=s1_concurrent,
            **s1_config
        )
        self.system2 = AsyncSystem2(
            max_concurrent=s2_concurrent,
            **s2_config
        )
    
    async def run_parallel(self, questions: List[str], 
                          task_types: List[str] = None) -> Dict[str, List[AsyncResponse]]:
        """
        Run both systems in parallel on the same questions
        
        Args:
            questions: List of questions
            task_types: List of task types
            
        Returns:
            Dict with 'system1' and 'system2' response lists
        """
        if task_types is None:
            task_types = ["general"] * len(questions)
        
        # Run both systems concurrently
        s1_task = self.system1.process_batch(questions, task_types)
        s2_task = self.system2.process_batch(questions, task_types)
        
        s1_responses, s2_responses = await asyncio.gather(s1_task, s2_task)
        
        return {
            "system1": s1_responses,
            "system2": s2_responses
        }
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get metrics from both systems"""
        return {
            "system1": self.system1.get_cognitive_effort_metrics(),
            "system2": self.system2.get_cognitive_effort_metrics()
        }
    
    def reset(self):
        """Reset both systems"""
        self.system1.reset()
        self.system2.reset()
