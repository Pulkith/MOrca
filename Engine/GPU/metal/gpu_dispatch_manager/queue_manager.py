import queue
import heapq
import logging
from typing import Optional, Union, List
from threading import Lock
from .task import BaseTask
from .config import SchedulerType, CONFIG

logger = logging.getLogger(__name__)

class TaskQueueManager:
    """Manages the queue of pending tasks based on the selected scheduling policy."""

    def __init__(self, scheduler_type: SchedulerType, max_size: int = 0):
        self.scheduler_type = scheduler_type
        self.max_size = max_size
        self._lock = Lock() # Protects access to the queue data structure

        if self.scheduler_type == SchedulerType.FIFO or self.scheduler_type == SchedulerType.ROUND_ROBIN:
             # Use standard queue for FIFO and Round Robin (handled by scheduler logic)
            self._queue = queue.Queue(maxsize=max_size)
        elif self.scheduler_type == SchedulerType.PRIORITY:
            # Use a min-heap (priority queue) for priority scheduling
            # heapq implements a min-heap, so lower priority number = higher actual priority
            self._queue = [] # List used as heap
        # Add cases for other scheduler types if they need specific queue structures
        # elif self.scheduler_type == SchedulerType.SHORTEST_JOB_FIRST:
        #     self._queue = [] # Could also use heap based on estimated duration
        else:
            # Default or custom might use a standard queue or need specific handling
             self._queue = queue.Queue(maxsize=max_size)

        logger.info(f"Task Queue Manager initialized with type: {scheduler_type.name}, max size: {'unlimited' if max_size == 0 else max_size}")

    def put(self, task: BaseTask) -> bool:
        """Adds a task to the queue. Returns True if successful, False if full."""
        task.on_queue() # Update task status
        with self._lock:
            try:
                if self.scheduler_type == SchedulerType.PRIORITY:
                    if self.max_size > 0 and len(self._queue) >= self.max_size:
                         logger.warning(f"Priority queue is full (max size {self.max_size}). Cannot add task {task.task_id}.")
                         return False
                    heapq.heappush(self._queue, task)
                    logger.debug(f"Task {task.task_id} added to priority queue (heap size: {len(self._queue)}).")
                    return True
                else:
                    # For queue.Queue based types
                    self._queue.put(task, block=False) # Don't block if full
                    logger.debug(f"Task {task.task_id} added to queue (approx size: {self.qsize()}).")
                    return True
            except queue.Full:
                logger.warning(f"Queue is full (max size {self.max_size}). Cannot add task {task.task_id}.")
                # Optionally, implement a strategy here (e.g., drop lowest priority if applicable)
                return False
            except Exception as e:
                 logger.error(f"Error adding task {task.task_id} to queue: {e}", exc_info=True)
                 return False


    def get(self) -> Optional[BaseTask]:
        """Removes and returns the next task based on scheduling policy. Returns None if empty."""
        with self._lock:
            try:
                if self.scheduler_type == SchedulerType.PRIORITY:
                    if not self._queue:
                        return None
                    task = heapq.heappop(self._queue)
                    logger.debug(f"Task {task.task_id} retrieved from priority queue (heap size: {len(self._queue)}).")
                    return task
                else:
                     # For queue.Queue based types
                    task = self._queue.get(block=False) # Don't block if empty
                    logger.debug(f"Task {task.task_id} retrieved from queue (approx size: {self.qsize()}).")
                    return task
            except queue.Empty:
                return None
            except Exception as e:
                 logger.error(f"Error getting task from queue: {e}", exc_info=True)
                 return None

    def qsize(self) -> int:
        """Returns the approximate size of the queue."""
        with self._lock:
            if self.scheduler_type == SchedulerType.PRIORITY:
                return len(self._queue)
            else:
                # queue.Queue.qsize() is approximate
                return self._queue.qsize()

    def empty(self) -> bool:
        """Returns True if the queue is empty, False otherwise."""
        with self._lock:
            if self.scheduler_type == SchedulerType.PRIORITY:
                return len(self._queue) == 0
            else:
                return self._queue.empty()

    # --- Methods for specific scheduler needs (Optional) ---

    def peek(self) -> Optional[BaseTask]:
         """Returns the next task without removing it (if supported)."""
         with self._lock:
             if self.scheduler_type == SchedulerType.PRIORITY:
                 return self._queue[0] if self._queue else None
             else:
                 # Peeking a standard queue.Queue is tricky/not directly supported safely
                 logger.warning("Peek operation not reliably supported for this queue type.")
                 return None # Or implement carefully if needed

    def reschedule(self, task: BaseTask):
        """Adds a task back to the queue (e.g., for fairness preemption simulation)."""
        logger.debug(f"Rescheduling task {task.task_id} (priority: {task.priority}).")
        # Potentially adjust priority or other attributes before re-queuing
        # task.priority += 10 # Example: Lower priority after being preempted
        self.put(task) # Re-add to the queue respecting its structure