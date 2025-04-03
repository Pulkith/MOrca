import time
import numpy as np
import logging
import random

# Configure logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

from orchestrator_system.orchestrator import MetalOrchestrator
from orchestrator_system.task import SimpleVectorAdd, TaskResult
from orchestrator_system.config import Config, SchedulerType, LogLevel
from orchestrator_system.testing import SleepTask # Import for example

# --- Global result tracking ---
results_log = {}
tasks_done_count = 0
tasks_total_count = 0

def example_callback(result: TaskResult):
    """Callback function to process task results."""
    global tasks_done_count
    log_entry = {
        "success": result.success,
        "result_preview": None,
        "error": str(result.error) if result.error else None,
        "metrics": result.metrics
    }
    if result.success and result.result_data is not None:
         if isinstance(result.result_data, np.ndarray):
              log_entry["result_preview"] = f"Numpy array, shape: {result.result_data.shape}, first element: {result.result_data.flat[0]}"
         elif isinstance(result.result_data, dict):
              log_entry["result_preview"] = result.result_data # Show dict directly
         else:
             log_entry["result_preview"] = str(result.result_data)[:100] # Preview other types

    results_log[result.task_id] = log_entry
    tasks_done_count += 1
    # logger.info(f"Callback received for {result.task_id}. Progress: {tasks_done_count}/{tasks_total_count}")
    print(f"Callback received for {result.task_id}. Success: {result.success}. Progress: {tasks_done_count}/{tasks_total_count}")


def main():
    global tasks_total_count, tasks_done_count
    logger.info("Starting Metal Orchestrator Example")

    # --- Configuration ---
    # Customize configuration options here if needed
    config_options = {
        "num_workers": 4, # Adjust based on your Mac's GPU
        "scheduler_type": SchedulerType.PRIORITY,
        "log_level": LogLevel.INFO,
        # "force_cpu_as_worker": True, # Uncomment to test without GPU
        "max_active_tasks_per_worker": 2, # Allow some task pipelining
    }

    try:
        # Use context manager for automatic startup and shutdown
        with MetalOrchestrator(config_options=config_options) as orchestrator:
            logger.info("Orchestrator started.")

            # --- Submit Tasks ---
            tasks_to_submit = 50
            tasks_total_count = tasks_to_submit
            task_ids = []

            print(f"Submitting {tasks_total_count} tasks...")
            for i in range(tasks_total_count):
                try:
                     # Mix of task types
                     if i % 5 == 0: # Submit a slightly longer sleep task occasionally
                         task = SleepTask(sleep_duration_sec=random.uniform(0.1, 0.5),
                                          callback=example_callback,
                                          priority=random.randint(1, 100))
                     else: # Submit vector addition tasks
                         size = random.randint(1000, 100000)
                         vec_a = np.random.rand(size).astype(np.float32)
                         vec_b = np.random.rand(size).astype(np.float32)
                         task = SimpleVectorAdd(vec_a, vec_b,
                                                callback=example_callback,
                                                priority=random.randint(1, 100)) # Random priority

                     task_id = orchestrator.submit_task(task)
                     task_ids.append(task_id)
                     # logger.debug(f"Submitted task {task_id}")

                except Exception as e:
                     logger.error(f"Failed to submit task {i}: {e}")
                     tasks_total_count -= 1 # Adjust total count if submission fails

                # Optional: Slow down submission slightly if needed
                # time.sleep(0.01)

            print(f"All {tasks_total_count} tasks submitted. Waiting for completion...")

            # --- Wait for Completion ---
            # Option 1: Poll the count
            while tasks_done_count < tasks_total_count:
                 print(f"Progress: {tasks_done_count}/{tasks_total_count} tasks completed...")
                 time.sleep(1.0)

            # Option 2: Wait for specific tasks using wait_for_task (less efficient for many tasks)
            # for task_id in task_ids:
            #     try:
            #         orchestrator.wait_for_task(task_id, timeout=60.0) # Wait up to 60s per task
            #     except TimeoutError:
            #         logger.error(f"Timeout waiting for task {task_id}")
            #     except ValueError:
            #          logger.error(f"Task {task_id} not found during wait.") # Should not happen if submitted

            print("\nAll tasks processed.")

            # --- Analyze Results (Optional) ---
            success_count = sum(1 for r in results_log.values() if r["success"])
            failed_count = tasks_total_count - success_count
            print(f"\n--- Summary ---")
            print(f"Total Tasks Submitted: {tasks_total_count}")
            print(f"Successful: {success_count}")
            print(f"Failed: {failed_count}")

            # Print details of failed tasks
            if failed_count > 0:
                 print("\nFailed Task Details:")
                 for tid, result in results_log.items():
                     if not result["success"]:
                         print(f"  Task ID: {tid}, Error: {result['error']}")

            # Example: Calculate average execution time from metrics
            total_exec_time = 0
            valid_metrics_count = 0
            for result in results_log.values():
                 if result["success"] and result["metrics"] and 'worker_total_task_time_ms' in result["metrics"]:
                     total_exec_time += result["metrics"]['worker_total_task_time_ms']
                     valid_metrics_count +=1
            
            if valid_metrics_count > 0:
                 avg_exec_time = total_exec_time / valid_metrics_count
                 print(f"\nAverage successful task execution time (worker side): {avg_exec_time:.2f} ms")


    except Exception as e:
        logger.critical(f"An unhandled error occurred in the main application: {e}", exc_info=True)

    logger.info("Metal Orchestrator Example Finished.")


if __name__ == "__main__":
    main()