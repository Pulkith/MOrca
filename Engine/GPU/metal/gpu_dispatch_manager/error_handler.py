import logging

logger = logging.getLogger(__name__)

class OrchestrationError(Exception):
    """Base exception for orchestrator system errors."""
    pass

class MetalError(OrchestrationError):
    """Exception related to Metal API errors."""
    def __init__(self, message, metal_error_code=None):
        super().__init__(message)
        self.metal_error_code = metal_error_code

class TaskExecutionError(OrchestrationError):
    """Exception raised when a task fails during execution."""
    def __init__(self, task_id, original_exception=None):
        self.task_id = task_id
        self.original_exception = original_exception
        super().__init__(f"Task {task_id} failed execution: {original_exception}")

class WorkerError(OrchestrationError):
    """Exception related to worker process issues."""
    pass

class ConfigurationError(OrchestrationError):
    """Exception for invalid configuration."""
    pass

class ErrorHandler:
    """Centralized error handling and reporting."""

    @staticmethod
    def log_error(error: Exception, context: str = "General"):
        """Logs an error with context."""
        logger.error(f"[{context}] Error occurred: {error}", exc_info=True)

    @staticmethod
    def handle_task_error(task_id: str, error: Exception, attempt: int, max_retries: int):
        """Handles errors during task execution, deciding whether to retry."""
        logger.error(f"[Task:{task_id}] Execution failed (Attempt {attempt}/{max_retries+1}): {error}", exc_info=True)
        if attempt < max_retries:
            logger.warning(f"[Task:{task_id}] Will retry.")
            return True # Indicates retry is possible
        else:
            logger.error(f"[Task:{task_id}] Max retries reached. Task failed permanently.")
            return False # Indicates task has failed permanently

    @staticmethod
    def handle_worker_crash(worker_id: int, error: Optional[Exception] = None):
        """Handles the unexpected termination or error state of a worker."""
        if error:
            logger.critical(f"[Worker:{worker_id}] Crashed or encountered critical error: {error}", exc_info=True)
        else:
            logger.critical(f"[Worker:{worker_id}] Crashed or terminated unexpectedly.")
        # Here you might trigger logic to restart the worker or mark it as dead

    @staticmethod
    def handle_metal_error(error_details: str, task_id: Optional[str] = None):
        """Handles specific Metal API related errors."""
        context = f"Task:{task_id}" if task_id else "MetalSystem"
        logger.error(f"[{context}] Metal API Error: {error_details}")
        # Depending on the error, could raise a MetalError or take specific action