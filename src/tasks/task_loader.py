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
            project_root = Path(__file__).parent.parent.parent  # Do_LLM_Fast_Slow/
            data_dir = project_root / "data" / "processed"
        self.data_dir = Path(data_dir)
        
        self.dataset: Dict[str, List[Task]] = {
            "system1_tasks": [],
            "system2_tasks": [],
            "conflict_tasks": []
        }
        
        self.loaded = False
    
    def load(self, dataset_file: str = "bibm_dataset.json") -> bool:
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

        # Convert numeric answer to option letter. Must dispatch by source because
        # PIQA / TruthfulQA / HellaSwag are 0-indexed while SIQA / WinoGrande are
        # 1-indexed, and the two conventions overlap for idx values in
        # [1, len(options)-1].
        source = str(item.get("source", "")).lower()
        _ONE_INDEXED_SOURCES = {"siqa", "winogrande"}
        if options:
            idx = None
            if isinstance(correct_answer, int):
                idx = correct_answer
            elif isinstance(correct_answer, str) and correct_answer.strip().isdigit():
                idx = int(correct_answer.strip())

            if idx is not None:
                if source in _ONE_INDEXED_SOURCES:
                    if 1 <= idx <= len(options):
                        correct_answer = chr(65 + idx - 1)
                else:
                    if 0 <= idx < len(options):
                        correct_answer = chr(65 + idx)
                    elif 1 <= idx <= len(options):
                        correct_answer = chr(65 + idx - 1)
        
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

    @staticmethod
    def shuffle_task_options(task: Task, seed_salt: str = "opt_shuffle") -> Task:
        """
        Return a new Task with its MCQ options shuffled deterministically.

        The shuffle is seeded by hash(task.id + seed_salt), so the same task
        always gets the same shuffled order across runs and conditions. This
        removes the positional bias where e.g. TruthfulQA and novel_conjunction
        items always have the correct answer in position A.

        Args:
            task: Task to shuffle (must have options and a letter correct_answer).
            seed_salt: Salt mixed into the per-item seed; change if you need
                a different shuffle permutation.

        Returns:
            A new Task with options permuted, question text rebuilt with the
            new A/B/C/... labels, correct_answer remapped to the new letter,
            and metadata['option_permutation'] recording the mapping
            new_index -> old_index.
        """
        if not task.options:
            return task

        old_correct = str(task.correct_answer).strip().upper()
        if not (len(old_correct) == 1 and "A" <= old_correct <= "Z"):
            return task
        old_idx = ord(old_correct) - ord("A")
        if not (0 <= old_idx < len(task.options)):
            return task

        import hashlib
        seed = int(hashlib.md5(f"{task.id}|{seed_salt}".encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        permutation = list(range(len(task.options)))
        rng.shuffle(permutation)

        new_options = [task.options[i] for i in permutation]
        new_correct_idx = permutation.index(old_idx)
        new_correct_letter = chr(65 + new_correct_idx)

        base_question = task.question
        if "\n\nOptions:\n" in base_question:
            base_question = base_question.split("\n\nOptions:\n", 1)[0]
        # Some source items (e.g. novel_conjunction) embed their options inline
        # in the question stem ("...Which is more probable?\nA. ...\nB. ..."),
        # which _parse_task then duplicates under an "Options:" block. After
        # shuffling, the inline listing and the block would disagree. Strip
        # trailing inline "A. ... / B. ..." lines so only the shuffled block
        # remains.
        import re as _re
        lines = base_question.split("\n")
        while lines and _re.match(r"^[A-Z]\.\s+\S", lines[-1].strip()):
            lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
        base_question = "\n".join(lines)
        options_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(new_options))
        new_question = f"{base_question}\n\nOptions:\n{options_text}"

        new_metadata = dict(task.metadata)
        new_metadata["option_permutation"] = permutation
        new_metadata["original_correct_letter"] = old_correct

        return Task(
            id=task.id,
            question=new_question,
            correct_answer=new_correct_letter,
            task_type=task.task_type,
            source=task.source,
            options=new_options,
            context=task.context,
            metadata=new_metadata,
        )

    def get_stratified_conflict_tasks(self,
                                      n: int,
                                      force_all_novel: bool = True,
                                      seed: Optional[int] = None,
                                      randomize_options: bool = True) -> List[Task]:
        """
        Return a conflict-task sample that guarantees all novel items are included.

        Uniform random sampling of 200 out of 867 yields only ~12 novel items per
        condition in expectation, which destroys statistical power for the
        novel-vs-classic comparison. This sampler forces all 50 novel items into
        the sample and fills the remaining `n - 50` slots by random draw from
        classic (TruthfulQA) items.

        Args:
            n: Total sample size. Must be >= number of novel items.
            force_all_novel: If True, all novel items are always included.
            seed: Optional RNG seed for reproducibility of the classic-item draw.

        Returns:
            List[Task] of size n, starting with novel items (in file order) then
            classic items (shuffled).
        """
        if not self.loaded:
            self.load()

        all_conflict = self.dataset.get("conflict_tasks", [])
        novel = [t for t in all_conflict if t.source.startswith("novel")]
        classic = [t for t in all_conflict if not t.source.startswith("novel")]

        if not force_all_novel:
            return self.get_conflict_tasks(n=n, shuffle=True)

        if n < len(novel):
            raise ValueError(
                f"Stratified sample requires n >= {len(novel)} (number of novel items); got n={n}."
            )

        rng = random.Random(seed)
        classic_pool = classic.copy()
        rng.shuffle(classic_pool)
        classic_sample = classic_pool[: n - len(novel)]

        sample = list(novel) + classic_sample
        if randomize_options:
            sample = [self.shuffle_task_options(t) for t in sample]
        return sample
    
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
    
    def load_novel_conflict_tasks(self, file_path: str = None) -> bool:
        """
        Load novel conflict tasks from JSON file.

        These tasks are designed to avoid training data contamination while
        preserving the same cognitive conflict structure as classic CRT tasks.

        Args:
            file_path: Path to the novel conflict tasks JSON file.
                       Defaults to data/novel_conflict_tasks.json in the project root.

        Returns:
            bool: Whether loading succeeded.
        """
        if file_path is None:
            project_root = Path(__file__).parent.parent.parent
            file_path = project_root / "data" / "novel_conflict_tasks.json"
        else:
            file_path = Path(file_path)

        if not file_path.exists():
            print(f"Novel conflict tasks file not found: {file_path}")
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            novel_tasks = raw_data.get("novel_conflict_tasks", [])
            count = 0

            # Build set of already-loaded task IDs to avoid duplicates
            # (novel tasks may already be embedded in bibm_dataset.json)
            existing_ids = {t.id for t in self.dataset["conflict_tasks"]}

            for item in novel_tasks:
                task_id = item.get("id", "")
                if task_id in existing_ids:
                    continue  # skip — already present in the loaded dataset
                task = self._parse_task(item, TaskType.CONFLICT)
                # Store intuitive_answer and explanation in metadata
                task.metadata["intuitive_answer"] = item.get("intuitive_answer", "")
                task.metadata["explanation"] = item.get("explanation", "")
                self.dataset["conflict_tasks"].append(task)
                existing_ids.add(task_id)
                count += 1

            if count > 0:
                print(f"Loaded {count} novel conflict tasks from {file_path}")
            else:
                print(f"Novel conflict tasks already present in dataset (skipped {len(novel_tasks)} duplicates).")
            return True

        except Exception as e:
            print(f"Error loading novel conflict tasks: {e}")
            return False

    def get_novel_conflict_tasks(self, n: int = None, **kwargs) -> List[Task]:
        """
        Get only the novel (non-training-data) conflict tasks.

        These are conflict tasks whose source starts with 'novel_',
        indicating they were designed to avoid training data contamination.

        Args:
            n: Number of tasks to return (None returns all).
            **kwargs: Additional arguments passed to get_tasks (e.g., shuffle).

        Returns:
            List[Task]: List of novel conflict tasks.
        """
        all_conflict = self.get_tasks("conflict_tasks", n=None, **kwargs)
        novel = [t for t in all_conflict if t.source.startswith("novel_")]

        if n is not None:
            novel = novel[:n]

        return novel

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
