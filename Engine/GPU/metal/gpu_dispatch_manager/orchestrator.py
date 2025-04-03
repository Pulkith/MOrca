import logging
import multiprocessing
import time
import queue
from typing import Dict, List, Optional, Tuple, Set, Any
from threading import Lock

from .config import Config, update_global_config, CONFIG # Import global config instance
from .task import BaseTask, TaskResult
from .worker import GPUWorker, ShutdownSignal
from .scheduler import BaseScheduler, get_scheduler
from .queue_manager import TaskQueueManager
from .result_handler import ResultHandler
from .error_handler import ErrorHandler, OrchestrationError, ConfigurationError, WorkerError
from .metal_utils import METAL_AVAILABLE, get_metal_device # For initial device check

logger = logging.getLogger(__name__)

class MetalOrchestrator:
    """
    Main class to manage GPU workers, task queue, scheduling, and results for Metal.
    """
    def __init__(self, config_options: Optional[Dict[str, Any]] = None):
        """
        Initializes the orchestrator.
        :param config_options: Dictionary of configuration overrides.
        """
        # Apply config overrides FIRST
        if config_options:
             update_global_config(**config_options)

        self.config = CONFIG # Use the globally configured instance
        logger.info("Initializing MetalOrchestrator...")
        logger.info(f"Using configuration: {self.config.__dict__}")


        if not METAL_AVAILABLE and not self.config.force_cpu_as_worker:
             raise OrchestrationError("Metal is not available on this system, and force_cpu_as_worker is False.")
        elif not METAL_AVAILABLE and self.config.force_cpu_as_worker:
             logger.warning("Metal not available, but force_cpu_as_worker is True. Workers will simulate execution without GPU.")


        # --- Core Components ---
        # Using multiprocessing queues for inter-process communication
        # Maxsize can help prevent memory exhaustion if workers fall behind
        mp_context = multiprocessing.get_context('spawn') # 'spawn' is safer on macOS
        self.task_submission_queue = mp_context.Queue(maxsize=self.config.max_queue_size or 0)
        self.result_queue = mp_context.Queue()

        self.task_queue_manager = TaskQueueManager(
            scheduler_type=self.config.scheduler_type,
            max_size=self.config.max_queue_size
        )
        self.scheduler: BaseScheduler = get_scheduler(self.task_queue_manager, self.config)
        self.result_handler = ResultHandler(self.result_queue, self.config, orchestrator_callback=self._handle_task_completion)
        
        # --- State Management ---
        self.workers: Dict[int, GPUWorker] = {}
        self.worker_task_queues: Dict[int, multiprocessing.Queue] = {} # Dedicated queue per worker
        self.tasks: Dict[str, BaseTask] = {} # Store submitted tasks by ID
        self.pending_dependencies: Dict[str, Set[str]] = {} # task_id -> set of unmet dependencies
        self.task_dependents: Dict[str, Set[str]] = {} # task_id -> set of tasks that depend on it
        self._lock = Lock() # Protect shared orchestrator state (tasks, workers, etc.)
        self._running = False
        self._stop_requested = False

        # --- Initialization ---
        self._check_metal_device() # Basic check if possible

        logger.info("MetalOrchestrator initialized.")

    def _check_metal_device(self):
         """Performs a basic check for the default Metal device if Metal is available."""
         if METAL_AVAILABLE:
             logger.debug("Performing initial Metal device check...")
             device = get_metal_device(self.config.preferred_device_name)
             if device:
                 logger.info(f"Successfully accessed initial Metal device: {device.name()}")
                 # We don't keep this device reference here; workers manage their own.
             else:
                  logger.warning("Could not access the default Metal device during initial check.")
                  if not self.config.force_cpu_as_worker:
                       raise OrchestrationError("Failed initial Metal device check and CPU fallback is disabled.")


    def start(self):
        """Starts the worker processes and the result handler thread."""
        if self._running:
            logger.warning("Orchestrator is already running.")
            return

        logger.info("Starting MetalOrchestrator...")
        self._stop_requested = False
        self._running = True

        # Start Result Handler Thread
        self.result_handler.start()

        # Start Worker Processes
        logger.info(f"Starting {self.config.num_workers} GPU workers...")
        for i in range(self.config.num_workers):
            try:
                # Create a dedicated input queue for each worker
                worker_queue = multiprocessing.get_context('spawn').Queue(maxsize=self.config.max_active_tasks_per_worker * 2) # Buffer size
                
                worker = GPUWorker(
                    worker_id=i,
                    task_queue=worker_queue, # Worker reads from its dedicated queue
                    result_queue=self.result_queue, # All workers write to the same result queue
                    config=self.config
                    # Can potentially assign specific devices here if needed/possible
                )
                worker.start()
                with self._lock:
                    self.workers[i] = worker
                    self.worker_task_queues[i] = worker_queue
                    self.scheduler.register_worker(i) # Register worker with scheduler
                logger.info(f"Worker {i} started (PID: {worker.pid}).")
            except Exception as e:
                 logger.error(f"Failed to start worker {i}: {e}", exc_info=True)
                 ErrorHandler.log_error(e, f"WorkerStart-{i}")
                 # Optionally, try to clean up already started workers or raise error
                 self.shutdown() # Attempt cleanup if a worker fails to start
                 raise WorkerError(f"Failed to initialize worker {i}.") from e

        # Start the main scheduling loop in a separate thread? Or integrate into an event loop?
        # For simplicity, let's make `submit_task` add to queue, and have a scheduling loop.
        # Or, simpler: `submit_task` adds to queue_manager, scheduling happens when workers are free (via result handler)
        # Let's use the result handler callback to trigger scheduling checks.
        logger.info("Orchestrator started successfully.")


    def submit_task(self, task: BaseTask) -> str:
        """
        Submits a task to the orchestration system.
        :param task: An instance of a class derived from BaseTask.
        :return: The unique ID of the submitted task.
        :raises: OrchestrationError if orchestrator is not running or queue is full.
        """
        if not self._running or self._stop_requested:
            raise OrchestrationError("Orchestrator is not running or shutting down.")

        if not isinstance(task, BaseTask):
            raise TypeError("Submitted task must be an instance of BaseTask.")

        with self._lock:
             if task.task_id in self.tasks:
                 logger.warning(f"Task with ID {task.task_id} already submitted. Ignoring.")
                 return task.task_id # Or raise error?

             self.tasks[task.task_id] = task
             logger.info(f"Received task {task.task_id} (Priority: {task.priority}, Deps: {task.dependencies})")

             # Handle dependencies
             unmet_deps = set()
             if task.dependencies:
                 for dep_id in task.dependencies:
                     if dep_id not in self.tasks or self.tasks[dep_id].status not in ("COMPLETED",):
                          unmet_deps.add(dep_id)
                          # Register this task as a dependent of the dependency
                          if dep_id not in self.task_dependents:
                              self.task_dependents[dep_id] = set()
                          self.task_dependents[dep_id].add(task.task_id)

             if unmet_deps:
                 self.pending_dependencies[task.task_id] = unmet_deps
                 task.status = "WAITING_DEPS"
                 logger.info(f"Task {task.task_id} waiting for dependencies: {unmet_deps}")
             else:
                 # No dependencies or all met, add to the actual processing queue
                 if not self.task_queue_manager.put(task):
                      # Queue is full - clean up task state?
                      del self.tasks[task.task_id]
                      # Remove from dependents if added
                      if task.dependencies:
                          for dep_id in task.dependencies:
                               if dep_id in self.task_dependents:
                                   self.task_dependents[dep_id].discard(task.task_id)
                      raise OrchestrationError(f"Task queue is full. Cannot submit task {task.task_id}.")
                 else:
                      logger.debug(f"Task {task.task_id} added to scheduling queue.")
                      # Trigger scheduling check immediately after adding a task
                      self._schedule_tasks()


        # Don't trigger scheduling immediately from submit, let the completion handler do it?
        # Or trigger async? Let's trigger it here for now.
        # self._schedule_tasks() # NO - Potential race condition if called rapidly. Schedule on completion.

        return task.task_id

    def _schedule_tasks(self):
        """Internal method to run the scheduler and dispatch tasks."""
        # This should be called safely, ideally protected by the lock or from a single thread context
        # Called from _handle_task_completion or potentially periodically.
        
        # Check fairness first (identifies tasks exceeding quantum)
        # This current fairness check is informational; doesn't actively preempt.
        # tasks_to_preempt = self.scheduler.check_fairness()
        # if tasks_to_preempt:
             # Handle preempted tasks (e.g., signal worker, reschedule) - Complex!
             # logger.warning(f"Tasks identified for preemption (simulation): {[t.task_id for t in tasks_to_preempt]}")
             # for task in tasks_to_preempt:
             #     self.task_queue_manager.reschedule(task)

        # Run the main scheduling logic
        assignments = self.scheduler.schedule() # Gets list of (worker_id, task)

        if assignments:
             logger.debug(f"Scheduler assigned {len(assignments)} tasks.")
             for worker_id, task in assignments:
                 if worker_id in self.worker_task_queues:
                     try:
                          # Task object should already be updated (status='RUNNING', assigned_worker) by scheduler
                          # task.on_dispatch(worker_id) # Should be called by scheduler now
                          logger.debug(f"Dispatching task {task.task_id} to worker {worker_id}'s queue.")
                          self.worker_task_queues[worker_id].put(task, block=False) # Send to specific worker queue
                     except queue.Full:
                          logger.warning(f"Worker {worker_id}'s queue is full. Rescheduling task {task.task_id}.")
                          # Re-queue the task
                          task.status = "QUEUED" # Reset status
                          task.assigned_worker = None
                          self.scheduler.update_worker_status(worker_id, task_completed=task.task_id) # Undo the 'started' update in scheduler
                          self.task_queue_manager.reschedule(task) # Put back into main queue
                     except Exception as e:
                          logger.error(f"Error dispatching task {task.task_id} to worker {worker_id}: {e}", exc_info=True)
                          # Handle error - potentially reschedule or fail the task
                          task.status = "FAILED"
                          task_result = task.on_failure(OrchestrationError(f"Dispatch error: {e}"), {})
                          self._handle_task_completion(task_result) # Process failure immediately
                 else:
                     logger.error(f"Cannot dispatch task {task.task_id}: Worker {worker_id} not found or queue missing.")
                     # Reschedule the task
                     task.status = "QUEUED"
                     task.assigned_worker = None
                     self.task_queue_manager.reschedule(task)


    def _handle_task_completion(self, result: TaskResult):
        """Callback executed by ResultHandler when a task finishes."""
        with self._lock:
            task_id = result.task_id
            logger.debug(f"Orchestrator processing completion for task {task_id} (Success: {result.success})")

            task = self.tasks.get(task_id)
            if not task:
                 logger.warning(f"Received result for unknown or already processed task ID: {task_id}")
                 return

            # Update scheduler about worker availability
            if task.assigned_worker is not None:
                 self.scheduler.update_worker_status(worker_id=task.assigned_worker, task_completed=task_id)
                 logger.debug(f"Updated scheduler status for worker {task.assigned_worker} after task {task_id} completion.")
            else:
                 logger.warning(f"Task {task_id} completed but had no assigned worker recorded.")


            # Update task state based on result
            task.status = "COMPLETED" if result.success else "FAILED"
            task.end_timestamp = result.completion_timestamp
            task.error = result.error

            # --- Retry Logic ---
            if not result.success and task.attempt_count <= self.config.max_task_retries:
                 if ErrorHandler.handle_task_error(task_id, result.error, task.attempt_count, self.config.max_task_retries):
                      logger.info(f"Retrying task {task_id} (Attempt {task.attempt_count + 1}).")
                      # Reset status and re-queue
                      task.status = "PENDING"
                      task.assigned_worker = None
                      task.start_timestamp = None
                      task.end_timestamp = None
                      task.error = None
                      # Re-check dependencies before rescheduling? Assume they were met.
                      if not self.task_queue_manager.put(task):
                           logger.error(f"Failed to re-queue task {task_id} for retry (queue full?). Task permanently failed.")
                           task.status = "FAILED" # Mark as failed if re-queue fails
                           # Proceed to dependency check / callback for failure state
                      else:
                           # Successfully re-queued for retry, don't process dependents or callback yet
                           self._schedule_tasks() # Trigger scheduling as a slot is now free
                           return # Stop processing this completion, wait for retry result

            # --- Dependency Handling (if task finished successfully or failed permanently) ---
            if task.status == "COMPLETED":
                 # Check if this task unlocks any dependents
                 if task_id in self.task_dependents:
                     dependents = self.task_dependents.pop(task_id)
                     logger.debug(f"Task {task_id} completed, checking dependents: {dependents}")
                     for dependent_id in dependents:
                          if dependent_id in self.pending_dependencies:
                              self.pending_dependencies[dependent_id].discard(task_id)
                              if not self.pending_dependencies[dependent_id]:
                                   # All dependencies met for this dependent task
                                   logger.info(f"All dependencies met for task {dependent_id}. Queueing.")
                                   del self.pending_dependencies[dependent_id]
                                   dependent_task = self.tasks.get(dependent_id)
                                   if dependent_task:
                                       if not self.task_queue_manager.put(dependent_task):
                                            logger.error(f"Failed to queue dependent task {dependent_id} (queue full?).")
                                            # Mark dependent as failed?
                                            dependent_task.status = "FAILED"
                                            dep_result = dependent_task.on_failure(OrchestrationError("Failed to queue after dependencies met"), {})
                                            # Need to handle this failure recursively? Risky. Log and maybe call its callback.
                                            if dependent_task.callback:
                                                self.result_handler.execute_user_callback(dependent_task.callback, dep_result)
                                   else:
                                        logger.warning(f"Dependent task {dependent_id} not found in tasks dict.")
                          else:
                               logger.warning(f"Task {dependent_id} listed as dependent but not found in pending_dependencies.")

            # --- Execute User Callback ---
            # Done AFTER updating state and handling dependencies
            if task.callback:
                 logger.debug(f"Queueing callback execution for task {task_id}")
                 # Delegate actual execution to result_handler to manage threading
                 self.result_handler.execute_user_callback(task.callback, result)


            # --- Cleanup Task State (Optional) ---
            # If task is finished (completed/failed permanently) and callback done,
            # we could remove it from self.tasks to save memory, but maybe keep for lookup?
            # if task.status in ("COMPLETED", "FAILED"):
            #     # Consider removing from self.tasks after a delay or based on policy
            #     pass

        # --- Trigger Scheduling ---
        # A worker finished a task, so check if more tasks can be scheduled
        self._schedule_tasks()


    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
         """Retrieves the current status and basic info of a task."""
         with self._lock:
             task = self.tasks.get(task_id)
             if task:
                 return {
                     "task_id": task.task_id,
                     "status": task.status,
                     "priority": task.priority,
                     "assigned_worker": task.assigned_worker,
                     "submit_timestamp": task.submit_timestamp,
                     "start_timestamp": task.start_timestamp,
                     "end_timestamp": task.end_timestamp,
                     "attempt_count": task.attempt_count,
                     "error": str(task.error) if task.error else None,
                     "dependencies": task.dependencies,
                     "pending_dependencies": list(self.pending_dependencies.get(task_id, set()))
                 }
             else:
                 return None

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Blocks until a specific task completes or fails, or timeout occurs."""
        start_time = time.monotonic()
        while True:
            status_info = self.get_task_status(task_id)

            if not status_info:
                 raise ValueError(f"Task ID {task_id} not found.")

            if status_info["status"] in ("COMPLETED", "FAILED"):
                 # Task finished, construct and return TaskResult from stored info
                 # Need to retrieve actual result data if stored, or fetch from worker (complex)
                 # Simplification: Return status, error from orchestrator's view.
                 task = self.tasks[task_id]
                 # Result data is tricky here. The result_handler got it, but did we store it?
                 # Assume for now we only wait for completion status, not the data via this method.
                 logger.warning(f"wait_for_task returning status for {task_id}, actual result data not retrieved here.")
                 return TaskResult(task_id=task_id,
                                  success=(task.status == "COMPLETED"),
                                  error=task.error,
                                  # result_data= ... # Needs mechanism to store/retrieve
                                  )


            if timeout is not None and (time.monotonic() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id} to complete.")

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.1)


    def shutdown(self, wait_for_completion: bool = True, timeout: Optional[float] = None):
        """
        Shuts down the orchestrator, worker processes, and result handler.
        :param wait_for_completion: If True, waits for currently running tasks to finish.
                                     If False, attempts to terminate workers immediately.
        :param timeout: Maximum time in seconds to wait for shutdown.
        """
        if not self._running or self._stop_requested:
            logger.warning("Shutdown already in progress or orchestrator not running.")
            return

        logger.info(f"Initiating shutdown (wait_for_completion={wait_for_completion})...")
        self._stop_requested = True
        start_time = time.monotonic()

        with self._lock:
            # 1. Stop accepting new tasks (already handled by _stop_requested flag in submit_task)
            logger.info("Stopped accepting new tasks.")

            # 2. Signal workers to stop
            if not wait_for_completion:
                 logger.warning("Attempting immediate termination of workers.")
                 for worker_id, worker in self.workers.items():
                     if worker.is_alive():
                         logger.debug(f"Terminating worker {worker_id}...")
                         try:
                              # Send shutdown signal first? Maybe worker handles it?
                              # self.worker_task_queues[worker_id].put(ShutdownSignal())
                              worker.terminate() # Forceful termination
                         except Exception as e:
                              logger.error(f"Error terminating worker {worker_id}: {e}")
            else:
                 logger.info("Signaling workers to shut down gracefully...")
                 # Send shutdown signal to each worker's queue
                 for worker_id, q in self.worker_task_queues.items():
                      try:
                          q.put(ShutdownSignal())
                      except Exception as e:
                           logger.error(f"Error sending shutdown signal to worker {worker_id}: {e}")

            # 3. Wait for workers to exit
            active_workers = list(self.workers.items())
            for worker_id, worker in active_workers:
                if worker.is_alive():
                    join_timeout = None
                    if timeout is not None:
                         elapsed = time.monotonic() - start_time
                         join_timeout = max(0.1, timeout - elapsed) # Calculate remaining time

                    logger.debug(f"Waiting for worker {worker_id} to exit (timeout: {join_timeout})...")
                    worker.join(timeout=join_timeout)

                    if worker.is_alive():
                         logger.warning(f"Worker {worker_id} did not exit gracefully within timeout. Terminating.")
                         try:
                             worker.terminate()
                             worker.join(timeout=1.0) # Short wait after terminate
                         except Exception as e:
                             logger.error(f"Error force terminating worker {worker_id}: {e}")
                    else:
                         logger.info(f"Worker {worker_id} exited.")
                # Clean up worker entry
                # del self.workers[worker_id] # Careful modifying dict while iterating? Handled by using list copy.


            # 4. Close worker queues (after workers are confirmed down)
            for q in self.worker_task_queues.values():
                 q.close()
                 q.join_thread() # Wait for queue's internal thread
            self.worker_task_queues.clear()


            # 5. Stop the Result Handler thread
            if self.result_handler.is_alive():
                logger.info("Stopping Result Handler...")
                self.result_handler.stop()
                join_timeout = None
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    join_timeout = max(0.1, timeout - elapsed)

                self.result_handler.join(timeout=join_timeout)
                if self.result_handler.is_alive():
                    logger.warning("Result Handler did not stop within timeout.")
                else:
                     logger.info("Result Handler stopped.")

            # 6. Clear remaining state
            self.workers.clear()
            self.tasks.clear()
            self.pending_dependencies.clear()
            self.task_dependents.clear()
            # Clear task queue manager?
            # while not self.task_queue_manager.empty(): self.task_queue_manager.get()

            # Close orchestrator queues?
            self.task_submission_queue.close()
            self.result_queue.close()
            # self.task_submission_queue.join_thread() # Wait for queue threads
            # self.result_queue.join_thread()


            self._running = False
            logger.info(f"MetalOrchestrator shut down completed in {time.monotonic() - start_time:.2f}s.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Shutdown gracefully by default when used as context manager
        self.shutdown(wait_for_completion=True)