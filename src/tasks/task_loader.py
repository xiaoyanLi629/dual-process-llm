"""
Task Loader Module
Dataset Loading and Task Management

Responsible for loading processed datasets and providing unified task interface
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass, field
from enum import Enum


class TaskType(Enum):
    """Task type enumeration"""
    SYSTEM1 = "system1"
    SYSTEM2 = "system2"
    CONFLICT = "conflict"


@dataclass
class Task:
    """Single task data structure"""
    id: str
    question: str
    correct_answer: Any
    task_type: str
    source: str
    options: List[str] = field(default_factory=list)
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "task_type": self.task_type,
            "source": self.source,
            "options": self.options,
            "context": self.context,
            "metadata": self.metadata
        }


class TaskLoader:
    """
    Task Loader.

    Responsible for loading and managing experiment task data.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the Task Loader.

        Args:
            data_dir: Path to the data directory.
        """
        if data_dir is None:
            # Data is in data/ folder within the project
            project_root = Path(__file__).parent.parent.parent  # idea1_dual_process/
            data_dir = project_root / "data" / "processed"
        self.data_dir = Path(data_dir)
        
        self.dataset: Dict[str, List[Task]] = {
            "system1_tasks": [],
            "system2_tasks": [],
            "conflict_tasks": []
        }
        
        self.loaded = False
    
    def load(self, dataset_file: str = "dual_process_dataset.json") -> bool:
        """
        Load the dataset.

        Args:
            dataset_file: Dataset filename.

        Returns:
            bool: Whether loading succeeded.
        """
        dataset_path = self.data_dir / dataset_file
        
        if not dataset_path.exists():
            print(f"Dataset file not found: {dataset_path}")
            return False
        
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # Parse System 1 tasks
            for item in raw_data.get("system1_tasks", []):
                task = self._parse_task(item, TaskType.SYSTEM1)
                self.dataset["system1_tasks"].append(task)
            
            # Parse System 2 tasks
            for item in raw_data.get("system2_tasks", []):
                task = self._parse_task(item, TaskType.SYSTEM2)
                self.dataset["system2_tasks"].append(task)
            
            # Parse conflict tasks
            for item in raw_data.get("conflict_tasks", []):
                task = self._parse_task(item, TaskType.CONFLICT)
                self.dataset["conflict_tasks"].append(task)
            
            self.loaded = True
            print(f"Loaded dataset:")
            print(f"  - System 1 tasks: {len(self.dataset['system1_tasks'])}")
            print(f"  - System 2 tasks: {len(self.dataset['system2_tasks'])}")
            print(f"  - Conflict tasks: {len(self.dataset['conflict_tasks'])}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def _parse_task(self, item: Dict, task_type: TaskType) -> Task:
        """
        Parse a single task item.

        Args:
            item: Raw task data.
            task_type: Task type.

        Returns:
            Task: Parsed task object.
        """
        # Build the question text
        question = item.get("question", "")
        context = item.get("context", "")
        
        # If context exists, prepend it to the question
        if context:
            question = f"Context: {context}\n\nQuestion: {question}"
        
        # If options exist, append them to the question
        options = item.get("options", [])
        if options:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            question = f"{question}\n\nOptions:\n{options_text}"
        
        # Process the correct answer
        correct_answer = item.get("correct_answer", item.get("answer", ""))
        
        # If the answer is an index, convert it to the corresponding option letter
        if isinstance(correct_answer, int) and options:
            if 0 <= correct_answer < len(options):
                correct_answer = chr(65 + correct_answer)
        
        return Task(
            id=item.get("id", ""),
            question=question,
            correct_answer=correct_answer,
            task_type=item.get("task_type", task_type.value),
            source=item.get("source", "unknown"),
            options=options,
            context=context,
            metadata={
                "original_data": item,
                "category": task_type.value
            }
        )
    
    def get_tasks(self, task_category: str, 
                  n: int = None, 
                  shuffle: bool = False,
                  source_filter: str = None) -> List[Task]:
        """
        Get tasks from a specified category.

        Args:
            task_category: Task category ("system1_tasks", "system2_tasks", "conflict_tasks").
            n: Number of tasks to return (None returns all).
            shuffle: Whether to randomly shuffle.
            source_filter: Data source filter.

        Returns:
            List[Task]: List of tasks.
        """
        if not self.loaded:
            self.load()
        
        tasks = self.dataset.get(task_category, [])
        
        # Apply data source filter
        if source_filter:
            tasks = [t for t in tasks if t.source == source_filter]
        
        # Randomly shuffle
        if shuffle:
            tasks = tasks.copy()
            random.shuffle(tasks)
        
        # Limit count
        if n is not None:
            tasks = tasks[:n]
        
        return tasks
    
    def get_system1_tasks(self, n: int = None, **kwargs) -> List[Task]:
        """Get System 1 tasks."""
        return self.get_tasks("system1_tasks", n=n, **kwargs)
    
    def get_system2_tasks(self, n: int = None, **kwargs) -> List[Task]:
        """Get System 2 tasks."""
        return self.get_tasks("system2_tasks", n=n, **kwargs)
    
    def get_conflict_tasks(self, n: int = None, **kwargs) -> List[Task]:
        """Get conflict tasks."""
        return self.get_tasks("conflict_tasks", n=n, **kwargs)
    
    def get_mixed_tasks(self, n_per_category: int = 100, shuffle: bool = True) -> List[Task]:
        """
        Get a mixed task set.

        Args:
            n_per_category: Number of tasks per category.
            shuffle: Whether to randomly shuffle.

        Returns:
            List[Task]: Mixed task list.
        """
        tasks = []
        tasks.extend(self.get_system1_tasks(n=n_per_category))
        tasks.extend(self.get_system2_tasks(n=n_per_category))
        tasks.extend(self.get_conflict_tasks(n=n_per_category))
        
        if shuffle:
            random.shuffle(tasks)
        
        return tasks
    
    def iterate_tasks(self, task_category: str, batch_size: int = 10) -> Iterator[List[Task]]:
        """
        Iterate over tasks in batches (for large-scale experiments).

        Args:
            task_category: Task category.
            batch_size: Batch size.

        Yields:
            List[Task]: A batch of tasks.
        """
        tasks = self.get_tasks(task_category)
        
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dict: Statistical information.
        """
        if not self.loaded:
            self.load()
        
        stats = {
            "total_tasks": sum(len(tasks) for tasks in self.dataset.values()),
            "categories": {}
        }
        
        for category, tasks in self.dataset.items():
            sources = {}
            task_types = {}
            
            for task in tasks:
                sources[task.source] = sources.get(task.source, 0) + 1
                task_types[task.task_type] = task_types.get(task.task_type, 0) + 1
            
            stats["categories"][category] = {
                "count": len(tasks),
                "sources": sources,
                "task_types": task_types
            }
        
        return stats
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.

        Args:
            task_id: Task ID.

        Returns:
            Optional[Task]: Task object, or None if not found.
        """
        if not self.loaded:
            self.load()
        
        for tasks in self.dataset.values():
            for task in tasks:
                if task.id == task_id:
                    return task
        
        return None
