import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict, List, Tuple, Union
import numpy as np
import logging
from .utils import generate_unique_id
from .config import CONFIG

# Forward declaration for type hinting
# class MTLBuffer: pass # Use actual import later

logger = logging.getLogger(__name__)

class TaskResult:
    """Holds the result of a completed task."""
    def __init__(self, task_id: str, success: bool, result_data: Any = None, error: Optional[Exception] = None, metrics: Optional[Dict] = None):
        self.task_id = task_id
        self.success = success
        self.result_data = result_data
        self.error = error
        self.metrics = metrics or {} # e.g., execution_time, queue_wait_time
        self.completion_timestamp = time.time()

class BaseTask(ABC):
    """Abstract base class for defining GPU tasks."""

    def __init__(self,
                 callback: Optional[Callable[[TaskResult], None]] = None,
                 priority: Optional[int] = None,
                 task_id: Optional[str] = None,
                 dependencies: Optional[List[str]] = None, # IDs of tasks that must complete first
                 metadata: Optional[Dict[str, Any]] = None):
        self.task_id: str = task_id or generate_unique_id("task")
        self.callback: Optional[Callable[[TaskResult], None]] = callback
        self.priority: int = priority if priority is not None else CONFIG.task_default_priority
        self.dependencies = dependencies or []
        self.metadata = metadata or {}

        # State managed by the orchestrator/worker
        self.status: str = "PENDING" # PENDING -> QUEUED -> RUNNING -> COMPLETED/FAILED
        self.submit_timestamp: Optional[float] = None
        self.start_timestamp: Optional[float] = None
        self.end_timestamp: Optional[float] = None
        self.assigned_worker: Optional[int] = None
        self.error: Optional[Exception] = None
        self.attempt_count: int = 0
        self.estimated_duration_ms: float = self.estimate_duration() # Used by some schedulers

        # Data (Must be pickleable to send to worker processes)
        # Users should store input data needed for prepare_buffers here
        # Example: self.input_array = np.array(...)

    # --- Methods for Orchestrator ---

    def __lt__(self, other):
        # For priority queue implementation (lower number = higher priority)
        if not isinstance(other, BaseTask):
            return NotImplemented
        return self.priority < other.priority

    # --- Methods for Worker (to be implemented by user subclasses) ---

    @abstractmethod
    def get_kernel_name(self) -> str:
        """Return the name of the Metal kernel function to execute."""
        pass

    @abstractmethod
    def get_kernel_source(self) -> Union[str, bytes]:
        """
        Return the Metal Shading Language (MSL) source code as a string
        or bytes, or a path to a .metal file.
        """
        pass

    @abstractmethod
    def prepare_buffers(self, device) -> Tuple[List[Any], List[Any]]:
        """
        Prepare and return input and output MTLBuffers.
        'device' is the metal_utils.MTLDevice object.
        This is where you'd convert numpy arrays etc. to Metal buffers.
        Return: (list_of_input_buffers, list_of_output_buffers)
        Note: The 'Any' type hint needs to be replaced with the actual
              metal-python MTLBuffer type once integrated.
        """
        # Example (pseudo-code, needs metal-python specifics):
        # input_data = self.input_array
        # input_buffer = device.newBufferWithBytes(input_data.tobytes(), length=input_data.nbytes, options=...)
        # output_buffer = device.newBufferWithLength(output_size, options=...)
        # return ([input_buffer], [output_buffer])
        pass

    @abstractmethod
    def configure_command_encoder(self, encoder, pipeline_state, buffers: Tuple[List[Any], List[Any]]):
        """
        Configure the compute command encoder.
        Set buffers, textures, threadgroup sizes etc.
        'encoder' is the metal_utils.MTLComputeCommandEncoder object.
        'pipeline_state' is the compiled metal_utils.MTLComputePipelineState.
        'buffers' is the tuple returned by prepare_buffers.
        """
        # Example (pseudo-code, needs metal-python specifics):
        # input_buffers, output_buffers = buffers
        # encoder.setComputePipelineState(pipeline_state)
        # for i, buf in enumerate(input_buffers):
        #    encoder.setBuffer_offset_atIndex(buf, 0, i)
        # for i, buf in enumerate(output_buffers):
        #    encoder.setBuffer_offset_atIndex(buf, 0, i + len(input_buffers))
        #
        # # Define grid and threadgroup sizes
        # gridSize = MTLSize(width=..., height=..., depth=...)
        # threadgroupSize = MTLSize(width=..., height=..., depth=...)
        # # Check threadgroup size limits: pipeline_state.maxTotalThreadsPerThreadgroup
        # threadgroupSize = MTLSize(width=min(w, max_threads), ...)
        # encoder.dispatchThreads_threadsPerThreadgroup(gridSize, threadgroupSize)
        pass

    @abstractmethod
    def handle_result(self, output_buffers: List[Any]) -> Any:
        """
        Process the data from the output MTLBuffers after execution.
        Return the final result data (must be pickleable).
        """
        # Example (pseudo-code, needs metal-python specifics):
        # output_buffer = output_buffers[0]
        # pointer = output_buffer.contents()
        # num_elements = output_buffer.length() // np.dtype(np.float32).itemsize
        # # Use ctypes or numpy to read data from the pointer
        # data = np.frombuffer(ctypes.string_at(pointer, output_buffer.length()), dtype=np.float32)
        # return data.copy() # Important: Copy data before buffer might be released
        pass

    # --- Optional Methods ---

    def estimate_duration(self) -> float:
        """
        Provide an estimated duration in milliseconds for this task.
        Used by some schedulers (like SJF) and fairness policies.
        Defaults to the configured quantum. Override for better accuracy.
        """
        return CONFIG.task_time_quantum_ms # Default estimate

    def on_queue(self):
        """Called when the task is added to the main queue."""
        self.status = "QUEUED"
        self.submit_timestamp = time.time()
        logger.debug(f"Task {self.task_id} queued.")

    def on_dispatch(self, worker_id: int):
        """Called when the task is assigned to a worker."""
        self.assigned_worker = worker_id
        self.status = "RUNNING"
        self.start_timestamp = time.time()
        self.attempt_count += 1
        logger.debug(f"Task {self.task_id} dispatched to worker {worker_id}.")

    def on_completion(self, result_data: Any, metrics: Dict):
        """Called internally upon successful completion on the worker."""
        self.status = "COMPLETED"
        self.end_timestamp = time.time()
        metrics['execution_time_ms'] = (self.end_timestamp - (self.start_timestamp or self.end_timestamp)) * 1000
        metrics['queue_wait_time_ms'] = ((self.start_timestamp or self.submit_timestamp or time.time()) - (self.submit_timestamp or time.time())) * 1000
        logger.debug(f"Task {self.task_id} completed successfully.")
        return TaskResult(self.task_id, True, result_data=result_data, metrics=metrics)

    def on_failure(self, error: Exception, metrics: Dict):
        """Called internally upon failure on the worker."""
        self.status = "FAILED"
        self.end_timestamp = time.time()
        self.error = error
        metrics['execution_time_ms'] = (self.end_timestamp - (self.start_timestamp or self.end_timestamp)) * 1000 if self.start_timestamp else 0
        metrics['queue_wait_time_ms'] = ((self.start_timestamp or self.submit_timestamp or time.time()) - (self.submit_timestamp or time.time())) * 1000 if self.submit_timestamp else 0
        logger.error(f"Task {self.task_id} failed: {error}")
        return TaskResult(self.task_id, False, error=error, metrics=metrics)

# --- Example Simple Task Implementation ---

class SimpleVectorAdd(BaseTask):
    """Example task: Adds two numpy arrays using Metal."""
    KERNEL_SOURCE = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void add_vectors(device const float* vecA [[buffer(0)]],
                            device const float* vecB [[buffer(1)]],
                            device float* result [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
        result[index] = vecA[index] + vecB[index];
    }
    """

    def __init__(self, vector_a: np.ndarray, vector_b: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        if vector_a.shape != vector_b.shape or vector_a.dtype != np.float32 or vector_b.dtype != np.float32:
            raise ValueError("Input vectors must have the same shape and dtype float32.")
        self.vector_a = vector_a
        self.vector_b = vector_b
        self.vector_size = vector_a.size
        self.vector_nbytes = vector_a.nbytes

    def get_kernel_name(self) -> str:
        return "add_vectors"

    def get_kernel_source(self) -> Union[str, bytes]:
        return self.KERNEL_SOURCE

    def prepare_buffers(self, device) -> Tuple[List[Any], List[Any]]:
        # This requires actual metal-python types and methods
        # from metal.objc import options # Assuming metal-python structure
        # storage_mode = options.MTLResourceStorageModeShared # Adjust based on device
        
        # Placeholder using pseudo-methods
        try:
            # Replace with actual metal-python calls:
            # buf_a = device.newBufferWithBytes(self.vector_a.tobytes(), length=self.vector_nbytes, options=storage_mode)
            # buf_b = device.newBufferWithBytes(self.vector_b.tobytes(), length=self.vector_nbytes, options=storage_mode)
            # buf_result = device.newBufferWithLength(self.vector_nbytes, options=storage_mode)

            # --- Dummy Buffer Creation for Structure ---
            print(f"[Task:{self.task_id}] Creating dummy buffers for size {self.vector_nbytes}")
            buf_a = f"dummy_buffer_a_{self.task_id}"
            buf_b = f"dummy_buffer_b_{self.task_id}"
            buf_result = f"dummy_buffer_result_{self.task_id}"
            # --- End Dummy ---

            if not buf_a or not buf_b or not buf_result:
                 raise RuntimeError("Failed to allocate Metal buffers.")
            return ([buf_a, buf_b], [buf_result])
        except Exception as e:
             logger.error(f"Error preparing buffers: {e}")
             raise # Re-raise to be caught by worker

    def configure_command_encoder(self, encoder, pipeline_state, buffers: Tuple[List[Any], List[Any]]):
        # This requires actual metal-python types and methods
        # from metal.metal import MTLSize # Assuming metal-python structure

        # Placeholder using pseudo-methods
        input_buffers, output_buffers = buffers

        # Replace with actual metal-python calls:
        # encoder.setComputePipelineState(pipeline_state)
        # encoder.setBuffer_offset_atIndex(input_buffers[0], 0, 0)
        # encoder.setBuffer_offset_atIndex(input_buffers[1], 0, 1)
        # encoder.setBuffer_offset_atIndex(output_buffers[0], 0, 2)
        print(f"[Task:{self.task_id}] Configuring dummy encoder with buffers {buffers}")


        # Determine grid and threadgroup size
        # Replace with actual metal-python calls:
        # threadgroup_width = min(pipeline_state.maxTotalThreadsPerThreadgroup(), 256) # Example limit
        # threads_per_group = MTLSize(threadgroup_width, 1, 1)
        # grid_size = MTLSize(self.vector_size, 1, 1) # Dispatch exactly enough threads
        
        # Placeholder:
        grid_size = (self.vector_size, 1, 1)
        threads_per_group = (min(self.vector_size, 256), 1, 1)

        print(f"[Task:{self.task_id}] Dispatching threads: grid={grid_size}, group={threads_per_group}")
        # encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threads_per_group)

    def handle_result(self, output_buffers: List[Any]) -> Any:
        # This requires actual metal-python types and methods
        # import ctypes

        # Placeholder using pseudo-methods
        output_buffer = output_buffers[0] # The dummy buffer name/ID
        print(f"[Task:{self.task_id}] Handling dummy result from buffer {output_buffer}")

        # Replace with actual metal-python calls:
        # try:
        #     pointer = output_buffer.contents()
        #     if not pointer:
        #         raise RuntimeError("Output buffer contents pointer is null.")
        #
        #     # Assuming float32 for this example
        #     dtype = np.float32
        #     num_elements = output_buffer.length() // np.dtype(dtype).itemsize
        #     if num_elements != self.vector_size:
        #         logger.warning(f"Output buffer size mismatch: expected {self.vector_size}, got {num_elements}")
        #
        #     # Read data safely using ctypes or numpy directly if pointer allows
        #     # Option 1: ctypes
        #     # ctype_array = (ctypes.c_float * num_elements).from_address(pointer)
        #     # result_array = np.ctypeslib.as_array(ctype_array)
        #
        #     # Option 2: numpy from buffer (requires buffer protocol support or manual copy)
        #     # This might require creating a buffer object from the raw pointer/length
        #     buffer_data = ctypes.string_at(pointer, output_buffer.length())
        #     result_array = np.frombuffer(buffer_data, dtype=dtype)
        #
        #     return result_array.copy() # IMPORTANT: Copy data out
        # except Exception as e:
        #      logger.error(f"Error reading result buffer: {e}")
        #      raise

        # --- Dummy Result Generation ---
        # Simulate the expected result without actual GPU computation
        dummy_result = self.vector_a + self.vector_b
        return dummy_result
        # --- End Dummy Result ---