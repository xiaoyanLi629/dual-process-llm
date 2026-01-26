"""
Cognitive Tasks Module
Implement various cognitive tasks for testing dual process theory
"""

from tasks.task_loader import TaskLoader, Task
from tasks.crt_tasks import CRTTask
from tasks.logical_reasoning import LogicalReasoningTask
from tasks.math_reasoning import MathReasoningTask
from tasks.commonsense_tasks import CommonsenseTask

__all__ = [
    'TaskLoader',
    'Task',
    'CRTTask',
    'LogicalReasoningTask',
    'MathReasoningTask',
    'CommonsenseTask'
]
