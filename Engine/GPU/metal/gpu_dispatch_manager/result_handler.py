import logging
import queue
import threading
from typing import Callable, Optional, Dict
import time

from .task import TaskResult
from .utils import StoppableThreadMixin
from .config import Config
from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class ResultHandler(StoppableThreadMixin, threading.Thread):
    """
    Handles completed tasks, triggers callbacks, and manages task state.
    Runs in a separate thread within the main orchestrator process.
    """
    def __init__(self, result_queue: queue.Queue, config: Config, orchestrator_callback: Optional[Callable] = None):
        super().__init__(name="ResultHandlerThread")
        self.result_queue = result_queue
        self.config = config
        self.orchestrator_callback = orchestrator_callback # Callback to notify orchestrator (e.g., task done)
        self.daemon = True # Allow main program to exit even if this thread is running

        # For executing callbacks in a separate pool if configured
        self.callback_executor = None
        if not self.config.callback_in_main_thread:
             # Initialize a ThreadPoolExecutor or similar if needed
             # from concurrent.futures import ThreadPoolExecutor
             # self.callback_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="CallbackExecutor")
             logger.info("Callbacks will be executed in a separate thread pool (if implemented).")
        else:
             logger.info("Callbacks will be executed synchronously in the ResultHandler thread.")


    def run(self):
        """Main loop to process results from the result queue."""
        logger.info("ResultHandler thread started.")
        while not self.stopped():
            try:
                result: TaskResult = self.result_queue.get(block=True, timeout=0.5) # Wait with timeout

                if result:
                     self._process_result(result)

            except queue.Empty:
                # Timeout, check stop event again
                continue
            except Exception as e:
                 logger.error(f"ResultHandler: Error processing result queue: {e}", exc_info=True)
                 ErrorHandler.log_error(e, "ResultHandler")
                 # Sleep briefly to avoid tight loop on persistent errors
                 time.sleep(0.1)

        self._cleanup()
        logger.info("ResultHandler thread finished.")

    def _process_result(self, result: TaskResult):
        """Process a single TaskResult."""
        task_id = result.task_id
        logger.debug(f"ResultHandler: Received result for task {task_id} (Success: {result.success})")

        # 1. Notify Orchestrator (e.g., to update scheduler)
        if self.orchestrator_callback:
            try:
                 self.orchestrator_callback(result) # Pass the whole result object
            except Exception as e:
                 logger.error(f"ResultHandler: Error calling orchestrator callback for task {task_id}: {e}", exc_info=True)


        # 2. Execute User Callback
        # Need access to the original task object to get the callback function
        # This requires the orchestrator to maintain a mapping task_id -> task_object
        # For now, assuming the orchestrator handles callback retrieval and execution
        # based on the notification above.

        # Alternative: If callback info was part of TaskResult (not ideal)
        # if result.callback:
        #     self._execute_callback(result.callback, result)

    def execute_user_callback(self, callback: Callable[[TaskResult], None], result: TaskResult):
         """Safely executes the user-defined callback."""
         if not callback:
             return

         task_id = result.task_id
         try:
             start_time = time.monotonic()
             logger.debug(f"Executing callback for task {task_id}...")

             if self.callback_executor:
                 # Submit to thread pool
                 # self.callback_executor.submit(callback, result)
                 # logger.warning("Async callback execution via ThreadPoolExecutor not yet implemented.")
                 callback(result) # Fallback to sync for now
             else:
                 # Execute synchronously in this thread
                 callback(result)

             duration = (time.monotonic() - start_time) * 1000
             logger.debug(f"Callback for task {task_id} completed in {duration:.2f}ms.")

         except Exception as e:
              logger.error(f"ResultHandler: Error executing user callback for task {task_id}: {e}", exc_info=True)
              ErrorHandler.log_error(e, f"Callback-{task_id}")

    def _cleanup(self):
        """Clean up resources like the callback executor."""
        logger.info("ResultHandler cleaning up...")
        if self.callback_executor:
             # self.callback_executor.shutdown(wait=True) # Wait for pending callbacks
             # logger.info("Callback executor shut down.")
             pass

    def stop(self):
         """Signals the handler thread to stop."""
         logger.info("ResultHandler received stop signal.")
         super().stop()
         # Optionally, add a sentinel to the queue to unblock .get() immediately
         # try:
         #    self.result_queue.put(None, block=False)
         # except queue.Full:
         #    pass