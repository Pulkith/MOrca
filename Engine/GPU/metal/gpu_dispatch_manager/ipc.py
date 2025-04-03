# orchestrator_system/ipc.py

import logging
import time
import uuid
from multiprocessing import shared_memory, resource_tracker
from typing import Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_SHARED_MEM_PREFIX = "mops_shm_" # Metal Orchestrator Python System Shared Memory

# --- Error Handling ---
class IPCError(Exception):
    """Base class for IPC related errors."""
    pass

class SharedMemoryError(IPCError):
    """Errors related to shared memory operations."""
    pass

# --- Core Shared Memory Functions ---

def create_shared_memory(size: int, name: Optional[str] = None) -> shared_memory.SharedMemory:
    """
    Creates a new block of shared memory.

    Args:
        size: The size in bytes of the shared memory block to create.
        name: Optional specific name for the block. If None, a unique name is generated.

    Returns:
        A multiprocessing.shared_memory.SharedMemory instance.

    Raises:
        SharedMemoryError: If creation fails (e.g., name exists, invalid size).
    """
    if size <= 0:
        raise SharedMemoryError("Shared memory size must be positive.")

    unique_name = name or f"{DEFAULT_SHARED_MEM_PREFIX}{uuid.uuid4()}"

    try:
        logger.debug(f"Attempting to create shared memory block: name='{unique_name}', size={size} bytes")
        shm = shared_memory.SharedMemory(name=unique_name, create=True, size=size)
        logger.info(f"Created shared memory block: name='{shm.name}', size={shm.size} bytes")
        return shm
    except FileExistsError:
        logger.error(f"Shared memory block with name '{unique_name}' already exists.")
        raise SharedMemoryError(f"Shared memory block named '{unique_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to create shared memory block '{unique_name}': {e}", exc_info=True)
        raise SharedMemoryError(f"Failed to create shared memory block '{unique_name}': {e}") from e

def attach_shared_memory(name: str, read_only: bool = False) -> shared_memory.SharedMemory:
    """
    Attaches to an existing shared memory block.

    Args:
        name: The name of the shared memory block.
        read_only: Attach in read-only mode (Not directly supported by SharedMemory,
                   managed by OS permissions or careful usage).

    Returns:
        A multiprocessing.shared_memory.SharedMemory instance attached to the block.

    Raises:
        SharedMemoryError: If attaching fails (e.g., block does not exist).
    """
    # Note: SharedMemory doesn't have an explicit read_only flag on attach.
    # Access control relies on OS permissions or programming discipline.
    if read_only:
        logger.debug(f"Attaching to shared memory block '{name}' (read-only requested, but not enforced by API)")
    else:
         logger.debug(f"Attaching to shared memory block '{name}'")

    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        logger.info(f"Attached to shared memory block: name='{shm.name}', size={shm.size} bytes")
        return shm
    except FileNotFoundError:
        logger.error(f"Shared memory block named '{name}' not found.")
        raise SharedMemoryError(f"Shared memory block named '{name}' not found.")
    except Exception as e:
        logger.error(f"Failed to attach to shared memory block '{name}': {e}", exc_info=True)
        raise SharedMemoryError(f"Failed to attach to shared memory block '{name}': {e}") from e

def close_shared_memory(shm: shared_memory.SharedMemory):
    """
    Closes the mapping for the calling process to the shared memory block.
    Does NOT delete the block itself.

    Args:
        shm: The SharedMemory instance to close.
    """
    if shm:
        try:
            logger.debug(f"Closing shared memory handle: name='{shm.name}'")
            shm.close()
        except Exception as e:
            # Log error but don't necessarily raise, as cleanup might fail during shutdown
            logger.error(f"Error closing shared memory handle '{shm.name}': {e}", exc_info=True)
    else:
         logger.warning("Attempted to close an invalid or None SharedMemory object.")

def unlink_shared_memory(name: str):
    """
    Requests the underlying operating system to delete the shared memory block.
    Should typically be called only once by the creator process after all other
    processes have closed their handles.

    Args:
        name: The name of the shared memory block to unlink.
    """
    logger.info(f"Requesting unlink for shared memory block: name='{name}'")
    try:
        # Need to attach temporarily to unlink if we don't have the object
        # Using a dummy SharedMemory object just to call unlink is common practice
        # However, simply calling unlink on a known name should be sufficient if
        # the resource_tracker is working correctly across processes or if called
        # from the creator which registered it.
        # Let's try creating a temporary handle to ensure it's registered for unlinking.
        temp_shm = shared_memory.SharedMemory(name=name, create=False)
        temp_shm.unlink()
        temp_shm.close() # Close the temporary handle immediately
        logger.debug(f"Successfully unlinked shared memory block: name='{name}'")
    except FileNotFoundError:
        # This is often okay - means it was already unlinked or never created properly
        logger.warning(f"Shared memory block '{name}' not found during unlink (may already be unlinked).")
    except Exception as e:
        logger.error(f"Error unlinking shared memory block '{name}': {e}", exc_info=True)
        # Don't raise here typically, as unlink is best-effort cleanup


def cleanup_all_my_shared_memory(prefix: str = DEFAULT_SHARED_MEM_PREFIX):
    """
    Attempts to find and unlink shared memory blocks created by this *orchestrator run*
    (identified by prefix). Use with caution, especially if multiple orchestrator
    instances might run concurrently with the same prefix. Best called during
    graceful shutdown.

    Note: Relies on implementation details of where shared memory is typically stored
          (/dev/shm on Linux/macOS) and might not be perfectly reliable or portable.
          `resource_tracker` is the more robust way but operates per-process.
    """
    logger.warning(f"Attempting global cleanup of shared memory with prefix '{prefix}'. Use with caution.")
    # This is platform-dependent and fragile.
    # On Linux/macOS, shared memory often appears in /dev/shm
    import os
    shm_dir = "/dev/shm"
    count = 0
    if os.path.exists(shm_dir):
        try:
            for filename in os.listdir(shm_dir):
                if filename.startswith(prefix):
                     logger.info(f"Found potential stale shared memory: {filename}. Attempting unlink.")
                     # Use the standard unlink function which tries to attach first
                     unlink_shared_memory(filename)
                     count += 1
        except Exception as e:
             logger.error(f"Error during shared memory cleanup scan in {shm_dir}: {e}", exc_info=True)
    else:
         logger.warning(f"Shared memory directory {shm_dir} not found. Cannot perform prefix-based cleanup.")

    if count > 0:
         logger.info(f"Attempted cleanup of {count} shared memory blocks matching prefix '{prefix}'.")


# --- NumPy Integration ---

def numpy_to_shared_memory(arr: np.ndarray, name: Optional[str] = None) -> Tuple[shared_memory.SharedMemory, Dict[str, Any]]:
    """
    Copies a NumPy array into a newly created shared memory block.

    Args:
        arr: The NumPy array to share.
        name: Optional specific name for the shared memory block.

    Returns:
        A tuple containing:
        - The created SharedMemory instance (caller responsible for close/unlink).
        - A metadata dictionary: {'name': str, 'shape': tuple, 'dtype': str}

    Raises:
        SharedMemoryError: If shared memory creation or data copy fails.
    """
    try:
        shm = create_shared_memory(size=arr.nbytes, name=name)
    except SharedMemoryError as e:
        raise SharedMemoryError(f"Failed to create shared memory for NumPy array: {e}") from e

    try:
        # Create a NumPy array backed by the shared memory buffer
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

        # Copy data from the original array to the shared memory array
        start_time = time.monotonic()
        shm_arr[:] = arr[:]
        copy_duration = time.monotonic() - start_time
        logger.debug(f"Copied {arr.nbytes} bytes to shared memory '{shm.name}' in {copy_duration:.4f}s")

        metadata = {
            'name': shm.name,
            'shape': arr.shape,
            'dtype': str(arr.dtype) # Store dtype as string for serialization
        }
        return shm, metadata

    except Exception as e:
         logger.error(f"Failed to copy NumPy array to shared memory '{shm.name}': {e}", exc_info=True)
         # Attempt cleanup if copy fails after creation
         close_shared_memory(shm)
         unlink_shared_memory(shm.name)
         raise SharedMemoryError(f"Failed to copy NumPy array to shared memory '{shm.name}': {e}") from e


def numpy_from_shared_memory(metadata: Dict[str, Any]) -> Tuple[np.ndarray, shared_memory.SharedMemory]:
    """
    Attaches to an existing shared memory block and creates a NumPy array view.

    Args:
        metadata: A dictionary containing {'name': str, 'shape': tuple, 'dtype': str}.

    Returns:
        A tuple containing:
        - A NumPy array that views the shared memory data.
        - The attached SharedMemory instance (caller responsible for close()).

    Raises:
        SharedMemoryError: If attaching or creating the NumPy view fails.
        KeyError: If metadata dictionary is missing required keys.
    """
    try:
        name = metadata['name']
        shape = metadata['shape']
        dtype_str = metadata['dtype']
        dtype = np.dtype(dtype_str) # Convert string back to dtype object
    except KeyError as e:
        raise KeyError(f"Metadata dictionary is missing required key: {e}")
    except TypeError as e:
         raise TypeError(f"Invalid dtype string in metadata: '{metadata.get('dtype')}'. {e}")

    try:
        shm = attach_shared_memory(name)
    except SharedMemoryError as e:
        raise SharedMemoryError(f"Failed to attach shared memory for NumPy array: {e}") from e

    try:
        # Calculate expected size
        expected_nbytes = np.prod(shape) * dtype.itemsize
        if shm.size < expected_nbytes:
             close_shared_memory(shm) # Clean up attachment
             raise SharedMemoryError(f"Shared memory block '{name}' size ({shm.size}) is smaller than required by metadata ({expected_nbytes}).")
        elif shm.size > expected_nbytes:
            logger.warning(f"Shared memory block '{name}' size ({shm.size}) is larger than required by metadata ({expected_nbytes}). Using only required part.")


        # Create NumPy array view
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        logger.debug(f"Created NumPy array view from shared memory '{name}'")
        return arr, shm

    except Exception as e:
        logger.error(f"Failed to create NumPy array view from shared memory '{name}': {e}", exc_info=True)
        close_shared_memory(shm) # Attempt cleanup
        raise SharedMemoryError(f"Failed to create NumPy array view from shared memory '{name}': {e}") from e


# --- Context Manager (Optional Convenience) ---

class SharedMemoryContext:
    """
    Context manager for automatically attaching and closing shared memory.
    Does NOT handle creation or unlinking.
    """
    def __init__(self, name: str):
        self.name = name
        self.shm = None

    def __enter__(self) -> shared_memory.SharedMemory:
        try:
            self.shm = attach_shared_memory(self.name)
            return self.shm
        except SharedMemoryError as e:
            logger.error(f"Context manager failed to attach to '{self.name}': {e}")
            # Reraise or handle as needed, here we reraise
            raise
        except Exception as e:
            logger.error(f"Unexpected error attaching in context manager '{self.name}': {e}")
            raise SharedMemoryError(f"Unexpected error attaching in context manager '{self.name}': {e}") from e


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shm:
            close_shared_memory(self.shm)
            logger.debug(f"Context manager closed shared memory '{self.name}'")
        # Returning False (or nothing) allows exceptions to propagate
        return False


class SharedNumpyContext:
    """
    Context manager for attaching shared memory and getting a NumPy view.
    Automatically closes the shared memory handle on exit.
    """
    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata
        self.shm = None
        self.array = None

    def __enter__(self) -> np.ndarray:
        try:
            self.array, self.shm = numpy_from_shared_memory(self.metadata)
            return self.array
        except (SharedMemoryError, KeyError, TypeError) as e:
            logger.error(f"NumPy Context manager failed for '{self.metadata.get('name', 'unknown')}': {e}")
            raise # Reraise the specific error

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shm:
            close_shared_memory(self.shm)
            logger.debug(f"NumPy Context manager closed shared memory '{self.metadata.get('name', 'unknown')}'")
        self.array = None # Help with potential GC
        self.shm = None
        return False # Propagate exceptions


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("--- IPC Shared Memory Example ---")

    # --- Process A (Creator) ---
    print("\n--- Process A (Creator) ---")
    try:
        original_array = np.arange(20, dtype=np.float64) * 1.5
        print(f"Original Array (A): {original_array}")

        # Create shared memory and copy data
        shm_a, meta_a = numpy_to_shared_memory(original_array)
        print(f"Created Shared Memory: Metadata={meta_a}")

        # Metadata (meta_a) would now be sent to Process B (e.g., via queue)

        # --- Process B (User) ---
        print("\n--- Process B (User) ---")
        shm_b = None
        try:
            # Attach using metadata received from A
            received_array, shm_b = numpy_from_shared_memory(meta_a)
            print(f"Received Array (B): {received_array}")
            print(f"Data is shared: {np.shares_memory(original_array, received_array)}") # False, it's a copy in shm
            
            # Modify data in shared memory via B's view
            print("Modifying data via Process B's view...")
            received_array[0] = 999.9
            received_array[-1] = -111.1
            print(f"Modified Array (B): {received_array}")

            # --- Process A (Checking Changes) ---
            print("\n--- Process A (Checking Changes) ---")
            # Process A needs to re-create its view if it closed the original one,
            # or just access the existing view if kept open.
            # Let's re-create a view from the original shm_a object for demo
            shm_a_view = np.ndarray(meta_a['shape'], dtype=meta_a['dtype'], buffer=shm_a.buf)
            print(f"Checking Array (A's view): {shm_a_view}")
            print(f"First element match: {shm_a_view[0] == received_array[0]}")
            print(f"Last element match: {shm_a_view[-1] == received_array[-1]}")

        except Exception as e_b:
            print(f"Error in Process B: {e_b}")
        finally:
            # Process B cleans up its handle
            if shm_b:
                 print("Process B closing its handle.")
                 close_shared_memory(shm_b)

        # --- Context Manager Usage ---
        print("\n--- Process C (Context Manager User) ---")
        try:
            with SharedNumpyContext(meta_a) as context_array:
                 print(f"Array via Context Manager (C): {context_array}")
                 print(f"Context manager data[0]: {context_array[0]}")
                 # Modify again
                 context_array[1] = 777.0
            # shm handle is auto-closed here
            print("Exited context manager.")

            # Check change again from A
            print(f"Checking Array (A's view after C): {shm_a_view}")


        except Exception as e_c:
             print(f"Error in Process C: {e_c}")


    except Exception as e_a:
        print(f"Error in Process A: {e_a}")
    finally:
         # Process A cleans up its handle AND unlinks the block
         if 'shm_a' in locals() and shm_a:
             print("Process A closing its handle and unlinking.")
             close_shared_memory(shm_a)
             unlink_shared_memory(shm_a.name)

    print("\n--- IPC Example Finished ---")