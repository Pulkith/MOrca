import logging
import multiprocessing
import time
import traceback
from typing import Optional, Any, Tuple, List

# Use a try-except block for metal-python imports
try:
    import metal
    from metal.objc import release_pool, autoreleasepool
    from .metal_utils import (
        get_metal_device, METAL_AVAILABLE,
        MTLDevice, MTLCommandQueue, MTLCommandBuffer,
        MTLComputePipelineState, MTLBuffer, MTLLibrary,
        MTLResourceOptions, MTLStorageMode # Import specific types needed
    )
except ImportError:
    # Define dummy types if import fails, consistent with metal_utils.py
    METAL_AVAILABLE = False
    class MTLDevice: pass
    class MTLCommandQueue: pass
    class MTLCommandBuffer: pass
    class MTLComputePipelineState: pass
    class MTLBuffer: pass
    class MTLLibrary: pass
    class MTLResourceOptions: pass
    class MTLStorageMode: pass
    def autoreleasepool(func): return func # No-op decorator


from .task import BaseTask, TaskResult
from .kernel_manager import KernelCache
from .error_handler import ErrorHandler, TaskExecutionError, MetalError
from .config import Config

# Define sentinel object for clean shutdown
class ShutdownSignal: pass

logger = logging.getLogger(__name__)

class GPUWorker(multiprocessing.Process):
    """
    A worker process responsible for executing Metal tasks on a GPU device.
    """
    def __init__(self,
                 worker_id: int,
                 task_queue: multiprocessing.Queue, # Queue for receiving BaseTask objects
                 result_queue: multiprocessing.Queue, # Queue for sending TaskResult objects
                 config: Config,
                 device_name: Optional[str] = None): # Specific device for this worker (optional)
        super().__init__(name=f"GPUWorker-{worker_id}")
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.config = config
        self.preferred_device_name = device_name or config.preferred_device_name # Worker specific or global pref

        # State to be initialized in run()
        self.device: Optional[MTLDevice] = None
        self.command_queue: Optional[MTLCommandQueue] = None
        self.kernel_cache: Optional[KernelCache] = None
        self.active: bool = True
        self.current_task: Optional[BaseTask] = None

        logger.info(f"Worker {self.worker_id} initialized.")

    def _initialize_metal(self) -> bool:
        """Initializes Metal device, command queue, and kernel cache."""
        if not METAL_AVAILABLE:
            logger.error(f"Worker {self.worker_id}: Metal not available. Cannot initialize.")
            # Send an error result or signal to orchestrator?
            # For now, just prevent the worker from starting tasks.
            return False

        try:
            self.device = get_metal_device(self.preferred_device_name)
            if not self.device:
                 logger.error(f"Worker {self.worker_id}: Failed to get Metal device.")
                 return False

            self.command_queue = self.device.newCommandQueue()
            if not self.command_queue:
                logger.error(f"Worker {self.worker_id}: Failed to create command queue.")
                # Release device?
                return False

            self.kernel_cache = KernelCache(self.device) # Initialize cache for this worker's device

            logger.info(f"Worker {self.worker_id} successfully initialized Metal on device: {self.device.name()}")
            return True

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Critical error during Metal initialization: {e}", exc_info=True)
            ErrorHandler.log_error(e, f"Worker-{self.worker_id} Metal Init")
            # Clean up partially initialized resources if possible
            if self.command_queue: self.command_queue.release()
            # if self.device: self.device.release() # Careful with device release if shared
            self.command_queue = None
            self.device = None
            return False

    def _execute_task(self, task: BaseTask) -> TaskResult:
        """Executes a single BaseTask using Metal."""
        if not self.device or not self.command_queue or not self.kernel_cache:
            err = MetalError(f"Worker {self.worker_id} Metal components not initialized.")
            return task.on_failure(err, {})

        start_time = time.monotonic()
        pipeline_state: Optional[MTLComputePipelineState] = None
        input_buffers: List[Any] = [] # Should be List[MTLBuffer]
        output_buffers: List[Any] = [] # Should be List[MTLBuffer]
        command_buffer: Optional[MTLCommandBuffer] = None
        task_metrics = {}

        try:
            with autoreleasepool: # Manage Objective-C objects within the task execution scope
                task.on_dispatch(self.worker_id)
                self.current_task = task
                logger.info(f"Worker {self.worker_id}: Starting task {task.task_id}")

                # 1. Get Kernel Source and Name
                kernel_source = task.get_kernel_source()
                kernel_name = task.get_kernel_name()
                # Handle file paths for kernel source
                if isinstance(kernel_source, str) and kernel_source.endswith('.metal'):
                    # Need to read the file content
                     try:
                         # Improve path resolution based on config search paths
                         # For now, assume relative path or absolute
                         with open(kernel_source, 'r') as f:
                             kernel_source_code = f.read()
                     except FileNotFoundError:
                          raise TaskExecutionError(task.task_id, f"Kernel source file not found: {kernel_source}")
                     except Exception as e:
                          raise TaskExecutionError(task.task_id, f"Error reading kernel file {kernel_source}: {e}")
                elif isinstance(kernel_source, bytes):
                     kernel_source_code = kernel_source.decode('utf-8') # Assume UTF-8
                else:
                     kernel_source_code = kernel_source

                # 2. Get Compiled Pipeline State (from cache or compile)
                prep_start = time.monotonic()
                pipeline_state, ps_error = self.kernel_cache.get_pipeline_state(kernel_source_code, kernel_name)
                if ps_error or not pipeline_state:
                    raise MetalError(f"Failed to get pipeline state for '{kernel_name}': {ps_error}")
                
                # pipeline_state should be an ObjCInstance wrapper from metal-python

                # 3. Prepare Buffers
                # User task implementation creates MTLBuffer objects
                # Make sure buffers are created with appropriate storage modes (e.g., Shared for CPU/GPU access)
                # storage_options = get_mtl_resource_options(storage_mode=MTLStorageMode.Shared) # Example
                input_buffers, output_buffers = task.prepare_buffers(self.device)
                task_metrics['buffer_prep_time_ms'] = (time.monotonic() - prep_start) * 1000

                # --- Actual Metal Execution ---
                exec_start = time.monotonic()

                # 4. Create Command Buffer
                command_buffer = self.command_queue.commandBuffer()
                if not command_buffer:
                    raise MetalError("Failed to create command buffer.")
                
                # command_buffer should be an ObjCInstance

                # 5. Create Compute Command Encoder
                encoder = command_buffer.computeCommandEncoder()
                if not encoder:
                    raise MetalError("Failed to create compute command encoder.")
                
                # encoder should be an ObjCInstance

                # 6. Configure Encoder (Set pipeline state, buffers, dispatch threads)
                # User task implementation configures the encoder
                task.configure_command_encoder(encoder, pipeline_state, (input_buffers, output_buffers))

                # 7. End Encoding
                encoder.endEncoding()
                # encoder object is likely managed by command_buffer now, no explicit release needed here typically

                # 8. Commit Command Buffer
                command_buffer.commit()
                task_metrics['command_encoding_commit_time_ms'] = (time.monotonic() - exec_start) * 1000
                logger.debug(f"Task {task.task_id}: Command buffer committed.")

                # 9. Wait for Completion (Blocking)
                # Alternatives: Use addCompletedHandler for async notification (more complex)
                wait_start = time.monotonic()
                command_buffer.waitUntilCompleted()
                task_metrics['gpu_wait_time_ms'] = (time.monotonic() - wait_start) * 1000

                # Check for command buffer errors after completion
                cb_error = command_buffer.error()
                if cb_error:
                    error_desc = str(cb_error.localizedDescription()) if hasattr(cb_error, 'localizedDescription') else str(cb_error)
                    raise MetalError(f"Command buffer execution failed: {error_desc}")

                logger.debug(f"Task {task.task_id}: GPU execution completed.")
                task_metrics['total_gpu_side_time_ms'] = task_metrics.get('command_encoding_commit_time_ms',0) + task_metrics.get('gpu_wait_time_ms',0)

                # 10. Process Results
                result_start = time.monotonic()
                result_data = task.handle_result(output_buffers)
                task_metrics['result_handling_time_ms'] = (time.monotonic() - result_start) * 1000

                # --- Task Success ---
                total_task_time = time.monotonic() - start_time
                task_metrics['worker_total_task_time_ms'] = total_task_time * 1000
                logger.info(f"Worker {self.worker_id}: Task {task.task_id} completed successfully in {total_task_time:.4f}s.")
                return task.on_completion(result_data, task_metrics)

        except Exception as e:
            # --- Task Failure ---
            total_task_time = time.monotonic() - start_time
            task_metrics['worker_total_task_time_ms'] = total_task_time * 1000
            logger.error(f"Worker {self.worker_id}: Task {task.task_id} failed: {e}", exc_info=True)
            # Wrap the exception for clarity
            if isinstance(e, (MetalError, TaskExecutionError)):
                 exec_error = e
            else:
                 exec_error = TaskExecutionError(task.task_id, original_exception=e)

            # Add traceback details if helpful
            # exec_error.traceback = traceback.format_exc() # Add if needed, increases result size

            return task.on_failure(exec_error, task_metrics)

        finally:
            # --- Cleanup Metal Objects ---
            # Release buffers created in prepare_buffers. Crucial to avoid leaks.
            # This requires careful handling in metal-python. Assume buffers have release().
            # Buffers might be autoreleased if created within the pool and not retained elsewhere.
            # Explicit release is safer if their lifetime isn't clear.
            # try:
            #     with autoreleasepool: # Ensure releases happen within a pool context
            #         for buf in input_buffers + output_buffers:
            #             if buf and hasattr(buf, 'release'):
            #                 # logger.debug(f"Releasing buffer: {buf}")
            #                 # buf.release() # Be careful if buffers are shared or managed elsewhere
            #                 pass # Relying on autorelease for now, VERIFY THIS BEHAVIOR in metal-python
            #
            #         # Command buffer and encoder are usually managed by the system after commit/endEncoding
            #         # if command_buffer and hasattr(command_buffer, 'release'): command_buffer.release() # Check if needed
            #
            # except Exception as cleanup_e:
            #      logger.error(f"Worker {self.worker_id}: Error during task cleanup for {task.task_id}: {cleanup_e}")

             self.current_task = None


    def run(self):
        """Main worker loop: Initialize Metal, wait for tasks, execute, send results."""
        logger.info(f"Worker {self.worker_id} process started (PID: {self.pid}).")

        if not self._initialize_metal():
             logger.critical(f"Worker {self.worker_id} failed to initialize Metal. Exiting.")
             # Send a special error marker? For now, just exit.
             # self.result_queue.put(WorkerInitializationError(self.worker_id)) # Define custom error type if needed
             return # Exit process

        while self.active:
            task: Optional[Union[BaseTask, ShutdownSignal]] = None
            try:
                # Wait for a task from the orchestrator
                # Use timeout to allow periodic checks (e.g., for shutdown signal)
                task = self.task_queue.get(block=True, timeout=1.0) # Wait up to 1 second

                if isinstance(task, ShutdownSignal):
                    logger.info(f"Worker {self.worker_id} received shutdown signal.")
                    self.active = False
                    break # Exit loop

                if isinstance(task, BaseTask):
                    result = self._execute_task(task)
                    try:
                        # Send result back to the orchestrator
                        self.result_queue.put(result)
                    except Exception as qe:
                         # Handle cases where the result queue might be full or closed
                         logger.error(f"Worker {self.worker_id}: Failed to put result for task {task.task_id} in queue: {qe}", exc_info=True)
                         # Potential data loss here - consider retry or logging strategy
                else:
                     logger.warning(f"Worker {self.worker_id} received unexpected item from task queue: {type(task)}")

            except queue.Empty:
                # Timeout occurred, no task received. Loop continues.
                # Can add heartbeat or idle checks here if needed.
                # logger.debug(f"Worker {self.worker_id} idle.")
                continue
            except EOFError:
                 logger.warning(f"Worker {self.worker_id}: Task queue connection closed unexpectedly (EOFError). Shutting down.")
                 self.active = False
                 break
            except (BrokenPipeError, ConnectionResetError):
                 logger.warning(f"Worker {self.worker_id}: Orchestrator connection lost. Shutting down.")
                 self.active = False
                 break
            except Exception as e:
                # Catch unexpected errors in the main loop
                logger.error(f"Worker {self.worker_id}: Unhandled exception in main loop: {e}", exc_info=True)
                ErrorHandler.log_error(e, f"Worker-{self.worker_id} MainLoop")
                # Depending on the error, might try to continue or exit
                # For safety, let's exit if something unexpected happens here
                self.active = False
                break

        self._cleanup()
        logger.info(f"Worker {self.worker_id} process finished.")

    def _cleanup(self):
        """Release Metal resources before exiting."""
        logger.info(f"Worker {self.worker_id}: Cleaning up resources...")
        try:
            with autoreleasepool: # Ensure cleanup happens in a pool
                if self.kernel_cache:
                     # Kernel cache clear needs proper ObjC release handling
                     logger.warning(f"Worker {self.worker_id}: Triggering kernel cache clear (ObjC release may be incomplete).")
                     # self.kernel_cache.clear_cache() # Call if implemented correctly
                     self.kernel_cache = None

                # Release command queue first
                if self.command_queue and hasattr(self.command_queue, 'release'):
                    logger.debug(f"Worker {self.worker_id}: Releasing command queue.")
                    self.command_queue.release()
                    self.command_queue = None

                # Release device (handle potential sharing if applicable)
                # If devices are strictly per-worker, release is fine.
                # If device is shared (e.g., single GPU), orchestrator should manage its lifetime.
                # Assuming per-worker device context for now based on initialization logic.
                # if self.device and hasattr(self.device, 'release'):
                #    logger.debug(f"Worker {self.worker_id}: Releasing device.")
                #    self.device.release() # Be cautious if device is shared globally
                #    self.device = None
                # Let's assume the global device cache handles this or it's managed by metal-python GC/autorelease.

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error during Metal cleanup: {e}", exc_info=True)
            ErrorHandler.log_error(e, f"Worker-{self.worker_id} Cleanup")