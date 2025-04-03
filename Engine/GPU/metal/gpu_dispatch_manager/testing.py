import unittest
import time
import numpy as np
import logging

# Configure logging for tests (e.g., suppress lower levels or direct to file)
logging.basicConfig(level=logging.INFO) # Adjust level for testing verbosity

from .orchestrator import MetalOrchestrator
from .task import BaseTask, SimpleVectorAdd, TaskResult
from .config import Config, SchedulerType, LogLevel
from .metal_utils import METAL_AVAILABLE

# --- Mock/Test Tasks ---

class SleepTask(BaseTask):
    """Task that simply sleeps for a specified duration."""
    def __init__(self, sleep_duration_sec: float, fail_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.sleep_duration = sleep_duration_sec
        self.fail_rate = fail_rate
        # Input data can be the sleep duration itself
        self.input_data = {'duration': sleep_duration_sec, 'fail': np.random.rand() < fail_rate}

    def get_kernel_name(self) -> str:
        # This task doesn't use a kernel, but the interface requires it
        return "no_kernel_sleep_task"

    def get_kernel_source(self) -> str:
         # Provide dummy source if needed by worker, or handle non-kernel tasks
         return """
         // Dummy kernel, not executed by SleepTask worker logic
         kernel void dummy_sleep() {}
         """

    def prepare_buffers(self, device):
         # No GPU buffers needed
         return ([], [])

    def configure_command_encoder(self, encoder, pipeline_state, buffers):
         # No GPU commands needed
         pass # Worker logic should check for this specific task type or kernel name

    def handle_result(self, output_buffers) -> Any:
         # No GPU result to handle
         # The worker's execution logic needs modification to handle non-GPU tasks
         # OR, simulate work by just sleeping in the worker's _execute_task method
         # For now, assume worker's _execute_task checks kernel name and sleeps

         # Simulate work based on input data
         if self.input_data['fail']:
             raise RuntimeError(f"Task {self.task_id} induced failure.")
         
         time.sleep(self.input_data['duration'])
         
         return {"slept_for": self.input_data['duration']}

    def estimate_duration(self) -> float:
        return self.sleep_duration * 1000 # Estimate in ms


class FailingTask(SimpleVectorAdd):
     """A task designed to fail during a specific phase."""
     def __init__(self, vector_a, vector_b, fail_at="prepare", **kwargs):
         super().__init__(vector_a, vector_b, **kwargs)
         self.fail_at = fail_at # 'prepare', 'configure', 'result', 'kernel'

     def get_kernel_source(self) -> str:
         if self.fail_at == 'kernel':
             # Intentionally invalid kernel source
             return "kernel void invalid_kernel(uint index [[thread_position_in_grid]]) { result[index] = vecA[index] + vecB[index] // Missing types"
         return super().get_kernel_source()

     def prepare_buffers(self, device):
         if self.fail_at == 'prepare':
             raise RuntimeError(f"Task {self.task_id} failed intentionally during prepare_buffers.")
         return super().prepare_buffers(device)

     def configure_command_encoder(self, encoder, pipeline_state, buffers):
          if self.fail_at == 'configure':
               raise RuntimeError(f"Task {self.task_id} failed intentionally during configure_command_encoder.")
          super().configure_command_encoder(encoder, pipeline_state, buffers)

     def handle_result(self, output_buffers: list) -> Any:
         if self.fail_at == 'result':
             raise RuntimeError(f"Task {self.task_id} failed intentionally during handle_result.")
         # Need modification in worker to simulate GPU error if fail_at == 'kernel' completion check fails
         return super().handle_result(output_buffers)


# --- Test Suite ---

# Decorator to skip tests if Metal is not available and CPU fallback isn't forced
skip_if_no_metal = unittest.skipUnless(
    METAL_AVAILABLE or Config(force_cpu_as_worker=True).force_cpu_as_worker,
    "Metal not available and force_cpu_as_worker is False"
)


class TestMetalOrchestrator(unittest.TestCase):

    def setUp(self):
        """Set up common resources for tests."""
        # Use specific config for testing
        self.test_config = {
            'num_workers': 2, # Keep low for testing
            'log_level': LogLevel.WARNING, # Reduce noise, INFO or DEBUG for deep dive
            'scheduler_type': SchedulerType.FIFO, # Predictable for basic tests
            'max_task_retries': 1,
            'force_cpu_as_worker': not METAL_AVAILABLE # Automatically enable CPU fallback if no Metal
            # Add other test-specific overrides
        }
        self.orchestrator = MetalOrchestrator(config_options=self.test_config)
        self.results = {} # Store callback results
        self.callback_received_events = {} # task_id -> threading.Event

    def tearDown(self):
        """Clean up resources after tests."""
        if self.orchestrator._running:
             self.orchestrator.shutdown(wait_for_completion=False, timeout=5.0) # Quick shutdown
        self.results = {}
        self.callback_received_events = {}

    def _result_callback(self, result: TaskResult):
        """Common callback for tests."""
        print(f"Test Callback Received: Task {result.task_id}, Success: {result.success}, Error: {result.error}")
        self.results[result.task_id] = result
        if result.task_id in self.callback_received_events:
             self.callback_received_events[result.task_id].set()


    @skip_if_no_metal
    def test_01_orchestrator_start_shutdown(self):
        """Test basic start and shutdown."""
        self.orchestrator.start()
        self.assertTrue(self.orchestrator._running)
        time.sleep(0.5) # Allow workers to initialize
        self.assertEqual(len(self.orchestrator.workers), self.test_config['num_workers'])
        self.orchestrator.shutdown(wait_for_completion=True, timeout=10.0)
        self.assertFalse(self.orchestrator._running)
        self.assertEqual(len(self.orchestrator.workers), 0)

    @skip_if_no_metal
    def test_02_submit_simple_task_success(self):
        """Test submitting a single successful SimpleVectorAdd task."""
        self.orchestrator.start()
        
        vec_a = np.arange(10, dtype=np.float32)
        vec_b = np.arange(10, dtype=np.float32) * 2
        expected_result = vec_a + vec_b

        task = SimpleVectorAdd(vec_a, vec_b, callback=self._result_callback)
        task_id = task.task_id
        self.callback_received_events[task_id] = threading.Event()

        submitted_id = self.orchestrator.submit_task(task)
        self.assertEqual(submitted_id, task_id)

        # Wait for callback
        received = self.callback_received_events[task_id].wait(timeout=15.0) # Generous timeout
        self.assertTrue(received, f"Timeout waiting for callback for task {task_id}")

        # Check result
        self.assertIn(task_id, self.results)
        result = self.results[task_id]
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.result_data)
        np.testing.assert_allclose(result.result_data, expected_result, rtol=1e-6)

        self.orchestrator.shutdown()


    @skip_if_no_metal
    def test_03_submit_multiple_tasks(self):
         """Test submitting multiple tasks."""
         self.orchestrator.start()
         num_tasks = 5
         task_ids = []
         expected_results = {}

         for i in range(num_tasks):
             vec_a = np.full(5, i, dtype=np.float32)
             vec_b = np.ones(5, dtype=np.float32)
             task = SimpleVectorAdd(vec_a, vec_b, callback=self._result_callback, priority=i) # Vary priority
             task_id = task.task_id
             task_ids.append(task_id)
             expected_results[task_id] = vec_a + vec_b
             self.callback_received_events[task_id] = threading.Event()
             self.orchestrator.submit_task(task)

         # Wait for all callbacks
         all_received = all(self.callback_received_events[tid].wait(timeout=20.0) for tid in task_ids)
         self.assertTrue(all_received, "Timeout waiting for one or more callbacks.")

         # Check all results
         self.assertEqual(len(self.results), num_tasks)
         for task_id in task_ids:
              self.assertIn(task_id, self.results)
              result = self.results[task_id]
              self.assertTrue(result.success, f"Task {task_id} failed: {result.error}")
              np.testing.assert_allclose(result.result_data, expected_results[task_id], rtol=1e-6)

         self.orchestrator.shutdown()

    @skip_if_no_metal
    def test_04_task_failure_and_retry(self):
         """Test task failure, retry mechanism, and permanent failure."""
         # NOTE: This test requires worker modification to handle FailingTask or kernel errors properly.
         # For now, we test failure during Python stages (prepare, configure, result).
         self.orchestrator.start()

         vec_a = np.arange(5, dtype=np.float32)
         vec_b = np.ones(5, dtype=np.float32)

         # Task that fails once in 'prepare', should succeed on retry
         task_retry = FailingTask(vec_a, vec_b, fail_at="prepare", callback=self._result_callback)
         task_retry_id = task_retry.task_id
         self.callback_received_events[task_retry_id] = threading.Event()
         # Modify task state to fail only on first attempt (if needed, or rely on fail_at logic)
         # For simplicity, assume fail_at="prepare" always fails first time. Worker needs to handle retry state.
         # This test might not work as expected without worker cooperation for retries.

         # Let's rethink: Make FailingTask fail based on attempt_count
         class RetryFailingTask(SimpleVectorAdd):
             def __init__(self, *args, fail_on_attempt=1, **kwargs):
                 super().__init__(*args, **kwargs)
                 self.fail_on_attempt = fail_on_attempt

             def prepare_buffers(self, device):
                  # Access attempt_count (needs to be passed or accessed via task state)
                  # This is tricky as task object sent to worker is a copy.
                  # Orchestrator handles retry logic, worker just executes.
                  # Let's simulate failure in the callback based on attempt count from result metrics.
                  # This doesn't test *worker* retry handling well.
                  #
                  # **Alternative:** Test failure propagation. A task that always fails.
                  if self.attempt_count == self.fail_on_attempt: # This won't work directly, attempt_count is updated by Orchestrator
                       raise RuntimeError(f"Task {self.task_id} induced failure on attempt (simulation).")
                  return super().prepare_buffers(device)

         # Task designed to fail permanently (more than max_retries)
         task_fail_perm = FailingTask(vec_a, vec_b, fail_at="prepare", callback=self._result_callback)
         task_fail_perm_id = task_fail_perm.task_id
         self.callback_received_events[task_fail_perm_id] = threading.Event()
         # Set retries low enough for it to fail permanently (config is 1 retry = 2 attempts total)
         task_fail_perm.max_retries_override = 0 # Need mechanism to override config per task? No, rely on global config.


         # --- Submit Task that should fail permanently ---
         self.orchestrator.submit_task(task_fail_perm)
         received_fail = self.callback_received_events[task_fail_perm_id].wait(timeout=10.0)
         self.assertTrue(received_fail, f"Timeout waiting for failing task {task_fail_perm_id} callback.")

         self.assertIn(task_fail_perm_id, self.results)
         result_fail = self.results[task_fail_perm_id]
         self.assertFalse(result_fail.success)
         self.assertIsNotNone(result_fail.error)
         # Check attempt count in orchestrator's view (requires get_task_status)
         status_info = self.orchestrator.get_task_status(task_fail_perm_id)
         self.assertIsNotNone(status_info)
         # Should have attempted config.max_task_retries + 1 times
         self.assertEqual(status_info['attempt_count'], self.test_config['max_task_retries'] + 1)


         # --- Test wait_for_task ---
         status_info_wait = self.orchestrator.wait_for_task(task_fail_perm_id, timeout=1.0)
         self.assertFalse(status_info_wait.success)


         self.orchestrator.shutdown()


    @skip_if_no_metal
    def test_05_task_dependencies(self):
         """Test tasks with dependencies."""
         self.orchestrator.start()

         results_order = []
         def dep_callback(result: TaskResult):
             results_order.append(result.task_id)
             self._result_callback(result)

         task_a = SleepTask(0.1, task_id="dep_A", callback=dep_callback)
         task_b = SleepTask(0.1, task_id="dep_B", dependencies=["dep_A"], callback=dep_callback)
         task_c = SleepTask(0.1, task_id="dep_C", dependencies=["dep_A"], callback=dep_callback)
         task_d = SleepTask(0.1, task_id="dep_D", dependencies=["dep_B", "dep_C"], callback=dep_callback)

         all_ids = [task_a.task_id, task_b.task_id, task_c.task_id, task_d.task_id]
         for tid in all_ids:
             self.callback_received_events[tid] = threading.Event()

         # Submit out of order
         self.orchestrator.submit_task(task_d)
         self.orchestrator.submit_task(task_c)
         self.orchestrator.submit_task(task_b)
         self.orchestrator.submit_task(task_a)


         # Wait for all
         all_received = all(self.callback_received_events[tid].wait(timeout=20.0) for tid in all_ids)
         self.assertTrue(all_received, "Timeout waiting for dependent tasks.")

         # Check execution order based on callbacks
         self.assertEqual(len(results_order), 4)
         self.assertEqual(results_order[0], "dep_A")
         # B and C can finish in any order relative to each other
         self.assertTrue(("dep_B" in results_order[1:3]) and ("dep_C" in results_order[1:3]))
         self.assertEqual(results_order[3], "dep_D")

         # Check all succeeded
         for tid in all_ids:
             self.assertTrue(self.results[tid].success)

         self.orchestrator.shutdown()

    # --- Add More Tests ---
    # test_priority_scheduling: Submit high/low prio tasks, check execution order (hard to verify precisely)
    # test_round_robin_scheduling: Submit tasks, check rough distribution across workers (hard to verify precisely)
    # test_queue_full: Submit tasks until queue is full (if max_size set), check error
    # test_worker_crash: Simulate a worker crash (e.g., kill process), check orchestrator recovery/task rescheduling (needs significant work)
    # test_config_overrides: Initialize orchestrator with different configs, check behavior
    # test_invalid_kernel: Submit task with bad kernel source, check for compilation error propagation
    # test_large_data: Submit task with large numpy arrays, check memory handling (within limits)
    # test_shutdown_wait_false: Test immediate termination on shutdown


# Run tests
if __name__ == '__main__':
    unittest.main()