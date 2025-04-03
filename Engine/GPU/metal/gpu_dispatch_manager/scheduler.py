import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
from collections import deque

from .task import BaseTask
from .queue_manager import TaskQueueManager
from .config import Config, SchedulerType, FairnessPolicy

logger = logging.getLogger(__name__)

# Type alias for worker state
WorkerInfo = Dict[str, Any] # e.g., {'id': int, 'available': bool, 'active_tasks': Set[str]}

class BaseScheduler(ABC):
    """Abstract base class for scheduling algorithms."""

    def __init__(self, task_queue: TaskQueueManager, config: Config):
        self.task_queue = task_queue
        self.config = config
        self.workers: Dict[int, WorkerInfo] = {} # worker_id -> info
        self.active_tasks: Dict[str, int] = {} # task_id -> worker_id
        # Fairness related state
        self.fairness_policy = config.fairness_policy
        self.task_start_times: Dict[str, float] = {} # task_id -> start_time_ms
        self.task_quantum_ms = config.task_time_quantum_ms

    def register_worker(self, worker_id: int):
        """Adds a worker to the scheduler's pool."""
        if worker_id not in self.workers:
            self.workers[worker_id] = {'id': worker_id, 'available': True, 'active_tasks': set()}
            logger.info(f"Scheduler registered worker {worker_id}.")

    def unregister_worker(self, worker_id: int):
         """Removes a worker from the scheduler's pool."""
         if worker_id in self.workers:
             del self.workers[worker_id]
             # Need to handle tasks currently assigned to this worker (e.g., reschedule)
             tasks_to_reschedule = [tid for tid, wid in self.active_tasks.items() if wid == worker_id]
             for task_id in tasks_to_reschedule:
                 # This assumes the task object itself isn't lost
                 logger.warning(f"Worker {worker_id} unregistered. Task {task_id} needs rescheduling (mechanism TBD).")
                 # Ideally, retrieve the task object and use task_queue.reschedule(task)
                 del self.active_tasks[task_id]
             logger.info(f"Scheduler unregistered worker {worker_id}.")


    def update_worker_status(self, worker_id: int, available: Optional[bool] = None, task_completed: Optional[str] = None, task_started: Optional[str] = None):
        """Updates the status of a worker."""
        if worker_id not in self.workers:
            logger.warning(f"Attempted to update status for unknown worker {worker_id}.")
            return

        worker_info = self.workers[worker_id]

        if task_started:
            worker_info['active_tasks'].add(task_started)
            self.active_tasks[task_started] = worker_id
            if self.fairness_policy == FairnessPolicy.TASK_QUANTUM:
                self.task_start_times[task_started] = time.monotonic() * 1000
            logger.debug(f"Worker {worker_id} started task {task_started}. Active: {len(worker_info['active_tasks'])}")


        if task_completed:
            worker_info['active_tasks'].discard(task_completed)
            if task_completed in self.active_tasks:
                 del self.active_tasks[task_completed]
            if task_completed in self.task_start_times:
                 del self.task_start_times[task_completed]
            logger.debug(f"Worker {worker_id} completed task {task_completed}. Active: {len(worker_info['active_tasks'])}")

        # Update availability based on active tasks vs max allowed
        max_tasks = self.config.max_active_tasks_per_worker
        is_full = len(worker_info['active_tasks']) >= max_tasks
        
        if available is not None:
             # Explicit availability update overrides calculation
             worker_info['available'] = available and not is_full
        else:
             # Auto-update based on capacity
             worker_info['available'] = not is_full
             
        # logger.debug(f"Worker {worker_id} availability updated: {worker_info['available']}")


    @abstractmethod
    def schedule(self) -> List[Tuple[int, BaseTask]]:
        """
        Decides which tasks to assign to which available workers.
        Returns a list of (worker_id, task) tuples.
        """
        pass

    def check_fairness(self) -> List[BaseTask]:
        """
        Checks if any running tasks have exceeded their time quantum (simulation).
        Returns a list of tasks that should be preempted/rescheduled.
        NOTE: This simulates preemption. Actual GPU kernel preemption is not feasible here.
              The worker needs to cooperate (e.g., finish current small step and yield).
              For simplicity now, we just identify tasks to potentially reschedule
              when the worker becomes free next. A more complex implementation could
              send a 'yield' signal to the worker.
        """
        if self.fairness_policy != FairnessPolicy.TASK_QUANTUM:
            return []

        tasks_to_reschedule = []
        current_time_ms = time.monotonic() * 1000
        # Iterate over a copy of keys since we might modify the dict
        for task_id in list(self.task_start_times.keys()):
            start_time = self.task_start_times[task_id]
            if (current_time_ms - start_time) > self.task_quantum_ms:
                worker_id = self.active_tasks.get(task_id)
                logger.warning(f"Fairness Check: Task {task_id} on worker {worker_id} exceeded quantum ({self.task_quantum_ms}ms). Flagging for potential reschedule.")
                # Here, we'd ideally need the actual Task object to reschedule.
                # This requires the orchestrator to hold task objects or retrieve them.
                # For now, we signal the need, but don't return the object directly.
                # tasks_to_reschedule.append(task_object) # Needs task object retrieval
                
                # Reset timer to avoid constant flagging, or remove until completion
                del self.task_start_times[task_id] # Remove to avoid re-flagging immediately

        return tasks_to_reschedule # This list would contain BaseTask objects if implemented fully


class FIFOScheduler(BaseScheduler):
    """First-In, First-Out Scheduler."""
    def schedule(self) -> List[Tuple[int, BaseTask]]:
        assignments = []
        available_workers = [w_id for w_id, info in self.workers.items() if info['available']]

        for worker_id in available_workers:
            if self.task_queue.empty():
                break # No more tasks

            # Check again if worker is still available (might have changed)
            if not self.workers[worker_id]['available']:
                 continue

            task = self.task_queue.get()
            if task:
                logger.debug(f"FIFO Scheduler assigning task {task.task_id} to worker {worker_id}")
                assignments.append((worker_id, task))
                # Immediately mark worker as busy for this scheduling round
                self.update_worker_status(worker_id, task_started=task.task_id) # Mark busy
            else:
                 break # Queue became empty

        return assignments

class PriorityScheduler(BaseScheduler):
    """Priority-Based Scheduler (using priority queue)."""
    def schedule(self) -> List[Tuple[int, BaseTask]]:
        assignments = []
        # Sort workers perhaps? Or just iterate. Consider available slots per worker.
        available_workers = sorted([w_id for w_id, info in self.workers.items() if info['available']]) # Consistent order

        num_possible_assignments = sum(self.config.max_active_tasks_per_worker - len(info['active_tasks'])
                                       for info in self.workers.values() if info['available'])

        potential_tasks = []
        # Fetch as many high-priority tasks as there are slots, respecting queue order
        for _ in range(min(num_possible_assignments, self.task_queue.qsize())):
             task = self.task_queue.get()
             if task:
                 potential_tasks.append(task)
             else:
                 break # Queue empty

        # Assign tasks to available workers
        task_idx = 0
        for worker_id in available_workers:
             if task_idx >= len(potential_tasks):
                 break # No more tasks fetched

             worker_info = self.workers[worker_id]
             can_take = self.config.max_active_tasks_per_worker - len(worker_info['active_tasks'])

             for _ in range(can_take):
                 if task_idx >= len(potential_tasks):
                      break # No more tasks for this worker

                 task = potential_tasks[task_idx]
                 logger.debug(f"Priority Scheduler assigning task {task.task_id} (Prio:{task.priority}) to worker {worker_id}")
                 assignments.append((worker_id, task))
                 self.update_worker_status(worker_id, task_started=task.task_id)
                 task_idx += 1


        # Put back any fetched tasks that couldn't be assigned (shouldn't happen with logic above, but safety)
        if task_idx < len(potential_tasks):
            logger.warning("Some fetched tasks could not be assigned, rescheduling.")
            for i in range(task_idx, len(potential_tasks)):
                self.task_queue.reschedule(potential_tasks[i]) # Use reschedule method

        return assignments


class RoundRobinScheduler(BaseScheduler):
    """Round-Robin Scheduler."""
    def __init__(self, task_queue: TaskQueueManager, config: Config):
        super().__init__(task_queue, config)
        self.worker_queue = deque() # Queue of worker IDs for RR assignment

    def register_worker(self, worker_id: int):
        super().register_worker(worker_id)
        if worker_id not in self.worker_queue:
             self.worker_queue.append(worker_id)

    def unregister_worker(self, worker_id: int):
         super().unregister_worker(worker_id)
         try:
             self.worker_queue.remove(worker_id)
         except ValueError:
             pass # Worker not in queue

    def schedule(self) -> List[Tuple[int, BaseTask]]:
        assignments = []
        if not self.worker_queue or self.task_queue.empty():
            return assignments

        # Try assigning tasks in round-robin order
        workers_to_try = len(self.worker_queue)
        for _ in range(workers_to_try):
            if self.task_queue.empty():
                break

            worker_id = self.worker_queue.popleft() # Get next worker
            worker_info = self.workers.get(worker_id)

            if worker_info and worker_info['available']:
                task = self.task_queue.get()
                if task:
                    logger.debug(f"RoundRobin Scheduler assigning task {task.task_id} to worker {worker_id}")
                    assignments.append((worker_id, task))
                    self.update_worker_status(worker_id, task_started=task.task_id)
                    # Only add worker back to queue if it might take more tasks
                    if (self.config.max_active_tasks_per_worker - len(worker_info['active_tasks'])) > 0:
                         self.worker_queue.append(worker_id) # Put back at the end for next round
                    else:
                         # If worker is now full, keep it out until a task completes
                         pass
                else:
                    self.worker_queue.append(worker_id) # Put worker back if no task was available
                    break # Queue became empty
            else:
                 # Worker is busy or gone, put it back temporarily if it exists, will be skipped next time if busy
                 if worker_id in self.workers:
                      self.worker_queue.append(worker_id)


        return assignments


def get_scheduler(task_queue: TaskQueueManager, config: Config) -> BaseScheduler:
    """Factory function to get the appropriate scheduler instance."""
    scheduler_type = config.scheduler_type

    if scheduler_type == SchedulerType.FIFO:
        return FIFOScheduler(task_queue, config)
    elif scheduler_type == SchedulerType.PRIORITY:
        return PriorityScheduler(task_queue, config)
    elif scheduler_type == SchedulerType.ROUND_ROBIN:
        return RoundRobinScheduler(task_queue, config)
    elif scheduler_type == SchedulerType.CUSTOM:
        if config.custom_scheduler_class and issubclass(config.custom_scheduler_class, BaseScheduler):
            return config.custom_scheduler_class(task_queue, config)
        else:
            raise ConfigurationError("Custom scheduler class not provided or invalid.")
    else:
        logger.warning(f"Unsupported scheduler type {scheduler_type}. Defaulting to FIFO.")
        return FIFOScheduler(task_queue, config)