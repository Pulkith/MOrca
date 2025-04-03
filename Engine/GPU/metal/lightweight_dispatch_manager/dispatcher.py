# Required imports
import sys
import time
import uuid
import random
import heapq
import logging
import threading
import queue
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type # Added Type
import functools

# Add colorlog import
try:
    import colorlog
except ImportError:
    print("Installing colorlog for colored logging...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorlog"])
    import colorlog

# Attempt to import metalgpu and numpy
try:
    import metalgpu
    import numpy as np
    # Note: metalgpu doesn't seem to expose MTL* types directly for type hints easily
except ImportError:
    print("ERROR: packages must be installed.")
    print("Then compile: python -m MOrcaLL build")
    sys.exit(1)
except FileNotFoundError:
    print("ERROR: Failed to load MOrcaLL library. Did you compile it?")
    print("Try running: python -m MOrcaLL build")
    sys.exit(1)


# --- Configuration ---
# Replace the basic logging configuration with a colorful one
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'green',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = [handler]  # Replace any existing handlers

# --- Constants ---
DEFAULT_NUM_WORKERS = 6 # Default concurrency limit based on request
DEFAULT_SCHEDULER = "priority" # Options: fifo, priority, fair_round_robin
DEFAULT_METAL_SHADER_PATH = "kernels.metal" # Default path for the shader file

# --- Custom Exceptions ---
class GpuDispatchError(Exception):
    pass

class GpuKernelError(GpuDispatchError):
    pass

class GpuTaskError(GpuDispatchError):
    pass

class InvalidTaskStateError(GpuTaskError):
    pass

# --- Enums ---
class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SchedulerType(Enum):
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR_ROUND_ROBIN = "fair_round_robin" # Note: No real preemption

# --- Task Definition ---
@functools.total_ordering
@dataclass()
class GpuTask:
    """Represents a task for the Metal dispatch manager using metalgpu."""
    kernel_name: str
    # Arguments: Mix of input lists/np.ndarrays and output specs (size, type_str)
    arguments: List[Union[list, np.ndarray, Tuple[int, Union[str, Type[np.generic]]]]]
    # Callback signature: func(task, result, error)
    callback: Optional[Callable[['GpuTask', Optional[List[np.ndarray]], Optional[Exception]], None]] = None
    priority: int = 0 # Lower number = higher priority
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, init=False)
    submit_time: float = field(default_factory=time.monotonic, init=False)
    start_time: Optional[float] = field(default=None, init=False)
    end_time: Optional[float] = field(default=None, init=False)
    # Result will be a list of numpy arrays corresponding to output buffers
    result: Optional[List[np.ndarray]] = field(default=None, init=False)
    error: Optional[Exception] = field(default=None, init=False)
    assigned_worker_slot: Optional[int] = field(default=None, init=False) # Which concurrency slot it used
    grid_size: Optional[int] = None # Optional: Specify grid size directly if needed

    # Required for priority queue (uses priority then submit_time)
    def __lt__(self, other: 'GpuTask') -> bool:
        if not isinstance(other, GpuTask):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority < other.priority
        # Tie-breaker for same priority
        return self.submit_time < other.submit_time

# --- Simulated IPC Mechanism (Thread-Safe) ---
IPC_DATA_STORE: Dict[str, Any] = {}
IPC_LOCK = threading.Lock()

def set_ipc_data(key: str, value: Any):
    """Stores data in a thread-safe dictionary for basic IPC."""
    with IPC_LOCK:
        IPC_DATA_STORE[key] = value
    logging.debug(f"IPC: Stored data for key '{key}'")

def get_ipc_data(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """Retrieves data from the thread-safe dictionary."""
    with IPC_LOCK:
        data = IPC_DATA_STORE.get(key, default)
    logging.debug(f"IPC: Retrieved data for key '{key}' (Found: {data is not None})")
    return data

def clear_ipc_data(key: str):
    """Removes data from the thread-safe dictionary."""
    with IPC_LOCK:
        IPC_DATA_STORE.pop(key, None)
    logging.debug(f"IPC: Cleared data for key '{key}'")


# --- Scheduler Implementations (Using threading primitives) ---
class BaseScheduler:
    """Base class for thread-safe schedulers."""
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def add_task(self, task: GpuTask):
        """Adds a task to the scheduler queue."""
        raise NotImplementedError

    def get_task(self) -> Optional[GpuTask]:
        """Blocks until a task is available or stop() is called. Returns None if stopped."""
        raise NotImplementedError

    def qsize(self) -> int:
        """Returns the approximate size of the queue."""
        raise NotImplementedError

    def stop(self):
        """Signals the scheduler to stop and unblocks waiting threads."""
        self._stop_event.set()
        # Implementation specific way to unblock get_task()

    def clear(self):
        """Removes all tasks from the scheduler, marking them as cancelled."""
        raise NotImplementedError

    def is_stopped(self) -> bool:
        """Checks if the scheduler has been stopped."""
        return self._stop_event.is_set()


class FifoScheduler(BaseScheduler):
    """Thread-safe FIFO scheduler using queue.Queue."""
    def __init__(self):
        super().__init__()
        self._queue = queue.Queue()

    def add_task(self, task: GpuTask):
        if self.is_stopped():
            logging.warning(f"Scheduler stopped, not adding task {task.task_id}")
            return
        # queue.Queue is thread-safe, status update needs care if accessed elsewhere
        task.status = TaskStatus.QUEUED
        self._queue.put(task)

    def get_task(self) -> Optional[GpuTask]:
        while not self.is_stopped():
            try:
                # Block efficiently until item available or timeout
                task = self._queue.get(block=True, timeout=0.1)
                if task is None: continue # Might be the dummy None from stop()
                return task
            except queue.Empty:
                continue # Loop back and check stop_event
        return None # Stopped

    def qsize(self) -> int:
        return self._queue.qsize()

    def stop(self):
        super().stop()
        # Put a dummy None item to ensure any waiting get() call unblocks
        self._queue.put(None)

    def clear(self):
        logging.info("Clearing FIFO Scheduler queue...")
        count = 0
        # No lock needed as queue ops are thread-safe
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task: # Handle potential None from stop()
                    task.status = TaskStatus.CANCELLED
                    # Trigger callback for cancelled task? Optional.
                    # self._safe_trigger_callback(task) # Need access to manager or pass callback func
                    count += 1
            except queue.Empty:
                break
        logging.info(f"FIFO Scheduler cleared {count} tasks.")


class PriorityScheduler(BaseScheduler):
    """Thread-safe Priority scheduler using queue.PriorityQueue."""
    def __init__(self):
        super().__init__()
        # PriorityQueue requires items to be comparable (__lt__ on GpuTask)
        self._queue = queue.PriorityQueue()

    def add_task(self, task: GpuTask):
        if self.is_stopped():
            logging.warning(f"Scheduler stopped, not adding task {task.task_id}")
            return
        task.status = TaskStatus.QUEUED
        self._queue.put(task)

    def get_task(self) -> Optional[GpuTask]:
        while not self.is_stopped():
            try:
                task = self._queue.get(block=True, timeout=0.1)
                if task is None: continue
                return task
            except queue.Empty:
                continue
        return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def stop(self):
        super().stop()
        # PriorityQueue needs a comparable item to unblock, use lowest priority task
        dummy_task = GpuTask(kernel_name="dummy", arguments=[], priority=999999)
        self._queue.put(dummy_task) # Unblock waiting get()

    def clear(self):
        logging.info("Clearing Priority Scheduler queue...")
        count = 0
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task and task.kernel_name != "dummy":
                    task.status = TaskStatus.CANCELLED
                    count += 1
            except queue.Empty:
                break
        logging.info(f"Priority Scheduler cleared {count} tasks.")


class FairRoundRobinScheduler(FifoScheduler):
    """
    Fairness based on FIFO queue order. No preemption.
    Identical to FifoScheduler in this implementation's effect.
    """
    def __init__(self, num_workers: int):
        super().__init__()
        self._num_workers = num_workers # Store for potential future logic
        logging.info("FairRoundRobinScheduler initialized (behaves like FIFO).")


# --- Main Dispatch Manager ---
class MetalGpuDispatchManager:
    """
    Orchestrates GPU task execution using the 'metalgpu' library.

    Manages a task queue, worker threads, scheduling, callbacks,
    and basic IPC simulation. Relies on metalgpu's blocking execution.
    """
    def __init__(self,
                 num_workers: int = DEFAULT_NUM_WORKERS,
                 scheduler_type: str = DEFAULT_SCHEDULER,
                 shader_path: str = DEFAULT_METAL_SHADER_PATH,
                 shader_source: Optional[str] = None
                 ):
        """
        Initializes the dispatch manager.

        Args:
            num_workers: Max number of concurrent GPU tasks via run_function.
            scheduler_type: 'fifo', 'priority', or 'fair_round_robin'.
            shader_path: Path to the .metal shader file (used if shader_source is None).
            shader_source: String containing Metal shader source code. Overrides shader_path.
        """
        logging.info("Initializing MetalGpuDispatchManager using 'metalgpu'")
        if num_workers <= 0:
            raise ValueError("Number of workers must be positive.")

        self._num_workers = num_workers
        self._shader_path = shader_path
        self._shader_source = shader_source

        # Initialize metalgpu Interface
        try:
            self._interface = metalgpu.Interface()
            logging.info(f"metalgpu.Interface initialized. Default device assumed.")
        except Exception as e:
            logging.error(f"Failed to initialize metalgpu.Interface: {e}", exc_info=True)
            logging.error("Ensure metalgpu is installed and compiled ('python -m metalgpu build')")
            raise GpuDispatchError(f"Failed to initialize metalgpu.Interface: {e}") from e

        # Load shaders
        self._load_shaders() # This populates internal state but metalgpu doesn't expose functions list easily

        self._scheduler: BaseScheduler
        stype = SchedulerType(scheduler_type.lower())
        if stype == SchedulerType.FIFO:
            self._scheduler = FifoScheduler()
        elif stype == SchedulerType.PRIORITY:
            self._scheduler = PriorityScheduler()
        elif stype == SchedulerType.FAIR_ROUND_ROBIN:
            self._scheduler = FairRoundRobinScheduler(self._num_workers)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        logging.info(f"Using Scheduler: {stype.value}")

        # Threading setup
        self._worker_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        # Semaphore limits concurrent run_function calls
        self._worker_slot_semaphore = threading.Semaphore(self._num_workers)
        self._metalgpu_lock = threading.Lock() # <--- ADD THIS LOCK        

        self._active_tasks: Dict[str, GpuTask] = {} # Track submitted/running tasks
        self._active_tasks_lock = threading.Lock()

        self._is_running = False
        logging.info(f"Manager configured for {self._num_workers} concurrent GPU tasks.")


    def _load_shaders(self):
        """Loads shaders using the provided path or source."""
        try:
            if self._shader_source:
                logging.info("Loading shader from source string...")
                self._interface.load_shader_from_string(self._shader_source)
            elif self._shader_path:
                logging.info(f"Loading shader from path: {self._shader_path}")
                self._interface.load_shader(self._shader_path)
            else:
                 raise GpuDispatchError("No shader path or source provided.")
            logging.info("Shader loaded into metalgpu.Interface.")
            # Note: We can't easily verify kernel names exist until set_function/run_function
        except FileNotFoundError:
             logging.error(f"Shader file not found at: {self._shader_path}")
             raise GpuDispatchError(f"Shader file not found: {self._shader_path}")
        except Exception as e:
            logging.error(f"Failed to load shader: {e}", exc_info=True)
            raise GpuDispatchError(f"Failed to load shader: {e}") from e

    def start(self):
        """Starts the worker threads that process tasks."""
        if self._is_running:
            logging.warning("Manager is already running.")
            return

        logging.info(f"Starting {self._num_workers} worker threads...")
        self._stop_event.clear()
        # Ensure scheduler is ready (clear stop if restarted)
        if self._scheduler.is_stopped():
             # Recreate scheduler or add a reset method? For now, assume start means fresh.
             # This logic might need refinement if start/stop cycles are frequent.
             logging.warning("Scheduler was stopped, ensure it's reset if needed.")

        self._is_running = True
        self._worker_threads = []
        for i in range(self._num_workers):
            # Pass worker slot ID for logging/tracking
            thread = threading.Thread(target=self._worker_loop, args=(i,), name=f"GPUWorker-{i}")
            self._worker_threads.append(thread)
            thread.start()
        logging.info("Manager started and workers running.")

    def stop(self, wait: bool = True, cancel_pending: bool = False):
        """Stops the manager and worker threads."""
        if not self._is_running:
            logging.warning("Manager is not running.")
            return

        logging.info(f"Stopping Manager (wait={wait}, cancelPending={cancel_pending})...")

        # Signal components to stop
        self._stop_event.set()
        self._scheduler.stop() # Unblock scheduler gets

        if cancel_pending:
            logging.info("Clearing pending tasks from scheduler...")
            self._scheduler.clear()

        # Wait for threads
        active_threads = list(self._worker_threads) # Copy list
        self._worker_threads = [] # Clear instance list
        if wait:
            logging.info(f"Waiting for {len(active_threads)} worker threads to finish...")
            join_start_time = time.monotonic()
            for i, thread in enumerate(active_threads):
                thread.join(timeout=10.0) # Add timeout per thread
                if thread.is_alive():
                    logging.warning(f"Worker thread {i} ({thread.name}) did not exit cleanly after 10s.")
            elapsed = time.monotonic() - join_start_time
            logging.info(f"All worker threads joined or timed out in {elapsed:.2f}s.")
        else:
            logging.info("Stop requested without waiting for active workers.")

        # Cleanup metalgpu interface? Docs say `del interface` handles buffers.
        # If manager is permanently stopped, maybe `del self._interface`?
        # For now, keep interface alive for potential restart.

        self._is_running = False
        logging.info("Manager stopped.")


    def _worker_loop(self, worker_slot_id: int):
        """Main loop for a worker thread."""
        logging.debug(f"Worker loop starting on slot {worker_slot_id}.")
        while not self._stop_event.is_set():
            task = None
            buffers_to_release = []
            # Keep track of which metal_gpu_buffers index corresponds to which original argument index
            output_specs = [] # List of (index_in_metal_list, original_task_arg_index)
            acquired = False # Track if semaphore was acquired in this iteration

            try:
                # 1. Get a task
                task = self._scheduler.get_task()
                if not task or self._stop_event.is_set():
                    break # Scheduler stopped or manager stopping

                logging.debug(f"Slot {worker_slot_id}: Acquired task {task.task_id}. Waiting for semaphore.")

                # 2. Acquire a concurrency slot (controls max active threads in this section)
                # Use timeout to allow periodic checking of stop_event
                acquired = self._worker_slot_semaphore.acquire(blocking=True, timeout=0.2)
                if not acquired:
                    # Failed to get slot (maybe stopping or contention?)
                    if self._stop_event.is_set(): break
                    continue # Loop again to check stop and retry acquire

                # Check stop event AGAIN after acquiring slot
                if self._stop_event.is_set():
                    # Don't release semaphore here yet, it's handled in finally if acquired
                    logging.warning(f"Slot {worker_slot_id}: Stopping after acquiring slot, task {task.task_id} dropped.")
                    break

                # --- Got task and slot ---
                logging.info(f"Slot {worker_slot_id}: Starting task {task.task_id} ({task.kernel_name})")
                task.assigned_worker_slot = worker_slot_id
                task.status = TaskStatus.RUNNING
                task.start_time = time.monotonic()
                with self._active_tasks_lock:
                    self._active_tasks[task.task_id] = task

                # 3. Prepare and Execute GPU Task - Protected by metalgpu lock
                # --- Acquire Lock for thread-unsafe metalgpu operations ---
                logging.debug(f"Slot {worker_slot_id}: Task {task.task_id} acquiring metalgpu lock...")
                with self._metalgpu_lock:
                    logging.debug(f"Slot {worker_slot_id}: Task {task.task_id} acquired metalgpu lock.")

                    # --- Buffer Preparation (INSIDE LOCK) ---
                    metal_gpu_buffers = [] # List to pass to run_function, may contain None
                    output_specs = [] # Reset inside lock just before use
                    max_buffer_size = 0 # Track for default grid size

                    for idx, arg in enumerate(task.arguments):
                        buffer = None
                        if isinstance(arg, (list, np.ndarray)):
                            # Input buffer
                            try:
                                buffer = self._interface.array_to_buffer(arg)
                                if buffer:
                                    metal_gpu_buffers.append(buffer)
                                    # Only release buffers we successfully create
                                    buffers_to_release.append(buffer)
                                else:
                                    # If metalgpu can return None on failure without exception
                                    raise GpuDispatchError("array_to_buffer returned None")
                            except Exception as buf_err:
                                logging.error(f"Task {task.task_id}: Failed to create input buffer for arg {idx}: {buf_err}", exc_info=False) # Reduce noise
                                raise GpuDispatchError(f"Failed input buffer creation for task {task.task_id}: {buf_err}") from buf_err

                        elif isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], int):
                            # Output buffer specification: (size, type_str_or_np_type)
                            size, type_info = arg
                            try:
                                buffer = self._interface.create_buffer(size, type_info)
                                if buffer:
                                    metal_gpu_buffers.append(buffer)
                                    buffers_to_release.append(buffer)
                                    # Store index in metal_gpu_buffers list and original index
                                    output_specs.append((len(metal_gpu_buffers) - 1, idx))
                                    if size > max_buffer_size: max_buffer_size = size
                                else:
                                     raise GpuDispatchError("create_buffer returned None")
                            except Exception as buf_err:
                                 logging.error(f"Task {task.task_id}: Failed to create output buffer for arg {idx} spec ({size}, {type_info}): {buf_err}", exc_info=False)
                                 raise GpuDispatchError(f"Failed output buffer creation for task {task.task_id}: {buf_err}") from buf_err
                        elif arg is None:
                             # Allow None in arguments to skip a buffer slot
                             metal_gpu_buffers.append(None)
                        else:
                            # Invalid argument type
                            raise GpuTaskError(f"Task {task.task_id}: Invalid argument type at index {idx}. Expected list, ndarray, (size, type), or None.")

                    # --- Determine Grid Size (INSIDE LOCK) ---
                    grid_size = task.grid_size # Use task override if provided
                    if grid_size is None:
                        if max_buffer_size > 0:
                            grid_size = max_buffer_size
                        else:
                            # Default if no output buffers or size specified
                            logging.warning(f"Task {task.task_id}: Cannot determine grid size from output buffers, using default 1.")
                            grid_size = 1

                    # --- Set Function and Run (INSIDE LOCK) ---
                    logging.debug(f"Task {task.task_id}: Running kernel '{task.kernel_name}' with grid size {grid_size}")
                    start_run = time.monotonic()

                    # Use the corrected argument name 'wait_for_completion'
                    # This call is blocking.
                    self._interface.run_function(
                        received_size=grid_size,
                        buffers=metal_gpu_buffers,
                        function_name=task.kernel_name, # Implicitly sets function
                        wait_for_completion=True # Use correct argument name
                    )
                    run_duration = time.monotonic() - start_run
                    logging.info(f"Task {task.task_id}: GPU execution finished in {run_duration:.4f}s.")

                    # --- Process Results (INSIDE LOCK) ---
                    task.result = []
                    for metal_idx, original_idx in output_specs:
                        output_buffer = metal_gpu_buffers[metal_idx]
                        try:
                            # .contents should return a numpy array directly per docs
                            output_array = output_buffer.contents
                            task.result.append(output_array)
                            logging.debug(f"Task {task.task_id}: Read result from buffer at metal index {metal_idx} (original arg {original_idx}). Shape: {output_array.shape}, Dtype: {output_array.dtype}")
                        except Exception as read_err:
                            logging.error(f"Task {task.task_id}: Failed reading contents of output buffer at metal index {metal_idx}: {read_err}", exc_info=False)
                            task.result.append(None) # Append None to indicate partial failure
                            # Re-raise the error to mark the task as failed overall
                            raise GpuDispatchError(f"Failed reading output buffer {metal_idx} for task {task.task_id}: {read_err}") from read_err

                    # --- Release Buffers (Optional, INSIDE LOCK) ---
                    # Explicitly release buffers created in this scope now that we're done.
                    # Although GC should handle it via `del interface`, being explicit can be safer.
                    logging.debug(f"Task {task.task_id}: Releasing {len(buffers_to_release)} metalgpu buffers...")
                    temp_release_list = list(buffers_to_release) # Copy before clearing
                    buffers_to_release.clear() # Clear original list for finally block safety
                    for buf in temp_release_list:
                        try:
                            buf.release()
                        except Exception as rel_err:
                            # Log warning but don't stop processing for release errors
                            logging.warning(f"Task {task.task_id}: Exception releasing buffer: {rel_err}")
                    logging.debug(f"Task {task.task_id}: Buffers released.")

                # --- metalgpu lock is automatically released here ---
                logging.debug(f"Slot {worker_slot_id}: Task {task.task_id} released metalgpu lock.")

                # --- Task Completed Successfully (if no exception occurred) ---
                task.status = TaskStatus.COMPLETED

            except Exception as e:
                # Catch errors during GPU execution, setup, or result reading
                logging.error(f"Slot {worker_slot_id}: Error processing task {task.task_id if task else 'Unknown'}: {e}", exc_info=False) # Reduce log noise
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = e # Store the first critical error
            finally:
                # --- Cleanup and Callback (Always Runs) ---
                if task: # Ensure task is defined
                    task.end_time = time.monotonic() # Record end time regardless of success/fail
                    logging.debug(f"Task {task.task_id} finished processing in slot {worker_slot_id}. Status: {task.status.name}")
                    # Remove from active tasks
                    with self._active_tasks_lock:
                        self._active_tasks.pop(task.task_id, None)
                    # Trigger callback (now runs sequentially in worker thread)
                    self._safe_trigger_callback(task)

                # Ensure buffers are released even if error occurred after creation but before explicit release
                if buffers_to_release:
                    logging.warning(f"Task {task.task_id}: Releasing {len(buffers_to_release)} buffers in finally block (error likely occurred).")
                    for buf in buffers_to_release:
                        try: buf.release()
                        except Exception: pass # Suppress errors during cleanup release
                    buffers_to_release.clear()

                # --- CRITICAL: Release the semaphore slot ---
                # Ensure release happens only if semaphore was acquired
                if acquired:
                     self._worker_slot_semaphore.release()
                     logging.debug(f"Slot {worker_slot_id}: Semaphore released.")
                     acquired = False # Reset for next loop iteration safety

        # Loop finished (likely due to stop_event)
        logging.info(f"Worker loop terminating on slot {worker_slot_id}.")

    def _safe_trigger_callback(self, task: GpuTask):
        """Safely triggers the user-provided callback in the current thread."""
        if task.callback:
            try:
                logging.debug(f"Triggering callback for task {task.task_id}")
                task.callback(task, task.result, task.error)
            except Exception as cb_err:
                logging.error(f"Exception in user callback for task {task.task_id}: {cb_err}", exc_info=True)


    def submit_task(self,
                    kernel_name: str,
                    arguments: List[Union[list, np.ndarray, Tuple[int, Union[str, Type[np.generic]]]]],
                    priority: int = 0,
                    callback: Optional[Callable[[GpuTask, Optional[List[np.ndarray]], Optional[Exception]], None]] = None,
                    grid_size: Optional[int] = None) -> Optional[GpuTask]:
        """
        Creates and submits a GpuTask to the scheduler.

        Args:
            kernel_name: Name of the kernel function in the loaded shader.
            arguments: List containing inputs and output specifications.
                       - Inputs: Python lists or NumPy arrays.
                       - Outputs: Tuples of (size, type_str_or_np_type), e.g., (100, 'float') or (100, np.int32).
                       - Use None to skip a buffer index.
            priority: Task priority (lower number is higher).
            callback: Function called upon task completion or failure.
                      Signature: callback(task, result_list, error)
            grid_size: Optional. Explicitly set the grid size for kernel execution.
                       If None, it's inferred from the largest output buffer size.

        Returns:
            The created GpuTask instance, or None if submission failed.
        """
        if not self._is_running:
            logging.error("Manager is not running. Cannot submit task.")
            return None

        # Basic check if interface seems ok (shader loaded is implicit)
        if not self._interface:
            logging.error("metalgpu.Interface not initialized. Cannot submit task.")
            return None

        task = GpuTask(kernel_name=kernel_name,
                       arguments=arguments,
                       priority=priority,
                       callback=callback,
                       grid_size=grid_size)

        logging.info(f"Submitting task {task.task_id} ('{task.kernel_name}', prio={task.priority})")
        self._scheduler.add_task(task)
        return task

    # --- IPC Methods ---
    def set_ipc_data(self, key: str, value: Any):
        """Stores data in a thread-safe dictionary for basic IPC."""
        set_ipc_data(key, value)

    def get_ipc_data(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Retrieves data from the thread-safe dictionary."""
        return get_ipc_data(key, default)

    def clear_ipc_data(self, key: str):
        """Removes data from the thread-safe dictionary."""
        clear_ipc_data(key)

    # --- Utility ---
    def get_active_task_count(self) -> int:
        """Returns the number of tasks currently being processed."""
        with self._active_tasks_lock:
            return len(self._active_tasks)

    def get_queue_size(self) -> int:
        """Returns the approximate number of tasks waiting in the scheduler queue."""
        return self._scheduler.qsize()


# --- Example Usage ---
if __name__ == "__main__":
    # Use colorful terminal output for the demo
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
    print(Fore.MAGENTA + Style.BRIGHT + "MOrcaLL Dispatch Manager - Demo using 'MOrcaLL'" + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
    print(Fore.YELLOW + "Ensure 'kernels.metal' exists in the same directory." + Style.RESET_ALL)
    print(Fore.YELLOW + "Ensure 'MOrcaLL' is installed and compiled ('python -m MOrcaLL build')." + Style.RESET_ALL)
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

    # --- Configuration ---
    NUM_WORKERS = 4 # Limit concurrency
    VECTOR_SIZE = 1024 * 512 # ~0.5 Million floats
    USE_PRIORITY = True
    SHADER_FILE = "kernels.metal" # Make sure this file exists!

    # --- Manager Setup ---
    manager = None
    try:
        manager = MetalGpuDispatchManager(
            num_workers=NUM_WORKERS,
            scheduler_type="priority" if USE_PRIORITY else "fifo",
            shader_path=SHADER_FILE
        )
        manager.start()
    except FileNotFoundError:
         print(Fore.RED + f"\nERROR: Shader file '{SHADER_FILE}' not found. Please create it." + Style.RESET_ALL)
         sys.exit(1)
    except GpuDispatchError as e:
         print(Fore.RED + f"\nERROR during manager setup: {e}" + Style.RESET_ALL)
         sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during manager setup: {e}", exc_info=True)
        sys.exit(1)

    # --- Data Preparation ---
    logging.info(f"Preparing data (vector size: {VECTOR_SIZE})...")
    # Use float32, common for GPUs
    vec_a = np.arange(VECTOR_SIZE, dtype=np.float32)
    vec_b = (np.arange(VECTOR_SIZE, dtype=np.float32) * 0.5) + 1.0
    scale_factor_val = np.array([3.0], dtype=np.float32) # Pass scalar as single-element array/buffer
    logging.info("Data prepared.")

    # Use threading event to signal completion from callbacks
    completion_events = {} # {task_id: threading.Event()}

    # --- Task Callback Function ---
    callback_lock = threading.Lock() # Protect print statements if needed
    tasks_completed = 0
    tasks_failed = 0

    def simple_callback(task: GpuTask, result_list: Optional[List[np.ndarray]], error: Optional[Exception]):
        global tasks_completed, tasks_failed
        with callback_lock:
            print(Fore.CYAN + "-" * 10 + f" Callback for Task {task.task_id} ({task.kernel_name}) " + "-" * 10 + Style.RESET_ALL)
            
            # Color-code status
            status_color = Fore.GREEN if task.status == TaskStatus.COMPLETED else Fore.RED
            print(f"  Status: {status_color}{task.status.name}{Style.RESET_ALL}")
            
            print(f"  Assigned Slot: {Fore.BLUE}{task.assigned_worker_slot}{Style.RESET_ALL}")
            duration = task.end_time - task.start_time if task.end_time and task.start_time else -1
            print(f"  Duration (inc. queue wait potentially): {Fore.YELLOW}{duration:.4f}s{Style.RESET_ALL}")

            event = completion_events.get(task.task_id)
            if error:
                print(f"  {Fore.RED}Error: {error}{Style.RESET_ALL}")
                tasks_failed += 1
            elif result_list:
                print(f"  Results Received: {Fore.GREEN}{len(result_list)} buffer(s){Style.RESET_ALL}")
                tasks_completed += 1
                # Verify result (optional)
                if task.kernel_name == "vector_add" and len(result_list) > 0:
                    result_add = result_list[0] # Assuming first output buffer
                    try:
                        expected_last = vec_a[-1] + vec_b[-1]
                        if not np.isclose(result_add[-1], expected_last):
                            print(f"  {Fore.RED}Verification FAILED! Last element: {result_add[-1]}, expected: {expected_last}{Style.RESET_ALL}")
                        else:
                            print(f"  {Fore.GREEN}vector_add simple verification PASSED.{Style.RESET_ALL}")
                    except Exception as verify_err:
                            print(f"  {Fore.RED}Verification error: {verify_err}{Style.RESET_ALL}")
                elif task.kernel_name == "vector_scale" and len(result_list) > 0:
                        result_scale = result_list[0]
                        try:
                            expected_last = vec_a[-1] * scale_factor_val[0]
                            if not np.isclose(result_scale[-1], expected_last):
                                print(f"  {Fore.RED}Verification FAILED! Last element: {result_scale[-1]}, expected: {expected_last}{Style.RESET_ALL}")
                            else:
                                print(f"  {Fore.GREEN}vector_scale simple verification PASSED.{Style.RESET_ALL}")
                        except Exception as verify_err:
                            print(f"  {Fore.RED}Verification error: {verify_err}{Style.RESET_ALL}")
                else:
                        print(f"  {Fore.YELLOW}No error reported, but no result data received.{Style.RESET_ALL}")
                        tasks_completed += 1 # Count as completed if no error

                if event:
                    event.set() # Signal that this task's callback is done
                print(Fore.CYAN + "-" * 40 + Style.RESET_ALL)

    # --- Submit Tasks ---
    logging.info("Submitting tasks...")
    submitted_tasks = 0

    # Task 1: Vector Add (Higher priority)
    output_spec_add = (VECTOR_SIZE, np.float32) # Specify output buffer: size, type
    task1 = manager.submit_task(
        kernel_name="vector_add",
        arguments=[vec_a, vec_b, output_spec_add], # Input A, Input B, Output Spec
        priority=0 if USE_PRIORITY else 0,
        callback=simple_callback
    )
    if task1:
        completion_events[task1.task_id] = threading.Event()
        submitted_tasks += 1

    # Task 2: Vector Scale (Lower priority)
    output_spec_scale = (VECTOR_SIZE, np.float32)
    task2 = manager.submit_task(
        kernel_name="vector_scale",
        # Pass scalar factor buffer 'scale_factor_val' as third argument
        arguments=[vec_a, output_spec_scale, scale_factor_val], # Input Vec, Output Spec, Scalar Factor Buf
        priority=10 if USE_PRIORITY else 0,
        callback=simple_callback
    )
    if task2:
        completion_events[task2.task_id] = threading.Event()
        submitted_tasks += 1

    # Task 3: Long running (same priority as scale)
    output_spec_long = (VECTOR_SIZE, np.float32)
    task3 = manager.submit_task(
        kernel_name="long_running_task", # Kernel just does add for now
        arguments=[vec_a, vec_b, output_spec_long],
        priority=10 if USE_PRIORITY else 0,
        callback=simple_callback
    )
    if task3:
        completion_events[task3.task_id] = threading.Event()
        submitted_tasks += 1

    # Add more tasks...
    for i in range(5):
         task_extra = manager.submit_task(
             kernel_name="vector_add",
             arguments=[vec_a, vec_b, (VECTOR_SIZE, np.float32)],
             priority=5 if USE_PRIORITY else 0, # Mid priority
             callback=simple_callback
         )
         if task_extra:
              completion_events[task_extra.task_id] = threading.Event()
              submitted_tasks += 1

    logging.info(f"Submitted {submitted_tasks} tasks. Waiting for callbacks...")
    logging.info(f"Initial Queue Size: {manager.get_queue_size()}")


    # --- Wait for Callbacks ---
    all_events = list(completion_events.values())
    timeout_seconds = 60.0 # Increase timeout
    start_wait = time.monotonic()

    all_completed_or_timedout = True
    waiting_tasks = submitted_tasks
    while waiting_tasks > 0 and (time.monotonic() - start_wait) < timeout_seconds:
         q_size = manager.get_queue_size()
         active_count = manager.get_active_task_count()
         logging.info(f"[Waiting - {time.monotonic() - start_wait:.1f}s] Tasks remaining: {waiting_tasks}, Queue: {q_size}, Active: {active_count}")
         # Check events with a shorter timeout
         completed_in_cycle = 0
         for task_id, event in list(completion_events.items()):
             if event.is_set():
                 completion_events.pop(task_id) # Remove completed
                 completed_in_cycle += 1
         waiting_tasks -= completed_in_cycle
         if waiting_tasks <= 0: break
         time.sleep(1.0) # Wait a bit before checking again


    elapsed_wait = time.monotonic() - start_wait
    if waiting_tasks > 0 :
        logging.error(f"Timeout! {waiting_tasks} tasks did not complete callback within {timeout_seconds}s!")
        all_completed_or_timedout = False
    else:
        logging.info(f"All task callbacks received in {elapsed_wait:.2f}s.")

    # --- Stop Manager ---
    logging.info("Stopping manager...")
    manager.stop(wait=True) # Wait for worker threads to finish cleanly

    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
    print(Fore.MAGENTA + Style.BRIGHT + "Demo Summary:" + Style.RESET_ALL)
    print(f"  Tasks Submitted: {Fore.BLUE}{submitted_tasks}{Style.RESET_ALL}")
    print(f"  Tasks Completed OK: {Fore.GREEN}{tasks_completed - 1}{Style.RESET_ALL}")
    print(f"  Tasks Failed: {Fore.RED}{tasks_failed}{Style.RESET_ALL}")
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)
    logging.info("Demo finished.")