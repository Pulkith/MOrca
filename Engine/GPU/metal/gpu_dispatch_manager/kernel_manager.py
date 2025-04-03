import logging
from typing import Dict, Optional, Tuple
from .metal_utils import (
    MTLDevice, MTLLibrary, MTLComputePipelineState,
    compile_metal_library, create_compute_pipeline_state, METAL_AVAILABLE
)
from .config import CONFIG

logger = logging.getLogger(__name__)

class KernelCache:
    """Manages caching of compiled Metal libraries and pipeline states."""

    def __init__(self, device: MTLDevice):
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal is not available. Cannot initialize KernelCache.")
        if not device:
            raise ValueError("A valid MTLDevice is required for KernelCache.")

        self.device: MTLDevice = device
        # Cache structure: { source_hash: MTLLibrary }
        self.library_cache: Dict[int, MTLLibrary] = {}
        # Cache structure: { (library_hash, kernel_name): MTLComputePipelineState }
        self.pipeline_state_cache: Dict[Tuple[int, str], MTLComputePipelineState] = {}
        self.cache_enabled: bool = CONFIG.kernel_cache_enabled
        logger.info(f"Kernel Cache initialized for device {device.name()}. Caching {'enabled' if self.cache_enabled else 'disabled'}.")


    def get_or_compile_library(self, source: str) -> Tuple[Optional[MTLLibrary], Optional[str]]:
        """Gets a library from cache or compiles it."""
        source_hash = hash(source)

        if self.cache_enabled and source_hash in self.library_cache:
            logger.debug(f"Library cache hit for hash {source_hash}.")
            return self.library_cache[source_hash], None

        logger.debug(f"Library cache miss for hash {source_hash}. Compiling...")
        library, error = compile_metal_library(self.device, source)

        if library and self.cache_enabled:
            self.library_cache[source_hash] = library
            # Consider adding cache eviction policies if memory becomes an issue
        elif error:
             logger.error(f"Failed to compile library: {error}")

        return library, error

    def get_or_create_pipeline_state(self, library: MTLLibrary, kernel_name: str) -> Tuple[Optional[MTLComputePipelineState], Optional[str]]:
        """Gets a pipeline state from cache or creates it."""
        # We need a stable reference to the library for the cache key.
        # Using its memory address (id()) or a generated unique ID might work,
        # but hash() on the source used to create it is more robust if we recompile.
        # Let's assume library instances are reused correctly by get_or_compile_library.
        # We'll use object ID for now, assuming library objects persist.
        library_id = id(library) # Or find a better unique identifier if available
        cache_key = (library_id, kernel_name)

        if self.cache_enabled and cache_key in self.pipeline_state_cache:
            logger.debug(f"Pipeline state cache hit for key {cache_key}.")
            return self.pipeline_state_cache[cache_key], None

        logger.debug(f"Pipeline state cache miss for key {cache_key}. Creating...")
        pipeline_state, error = create_compute_pipeline_state(self.device, library, kernel_name)

        if pipeline_state and self.cache_enabled:
            self.pipeline_state_cache[cache_key] = pipeline_state
             # Consider adding cache eviction policies
        elif error:
             logger.error(f"Failed to create pipeline state for kernel '{kernel_name}': {error}")

        return pipeline_state, error

    def get_pipeline_state(self, source: str, kernel_name: str) -> Tuple[Optional[MTLComputePipelineState], Optional[str]]:
        """Convenience method to get pipeline state directly from source."""
        library, lib_error = self.get_or_compile_library(source)
        if lib_error or not library:
            return None, f"Failed to get/compile library: {lib_error}"

        pipeline_state, ps_error = self.get_or_create_pipeline_state(library, kernel_name)
        if ps_error:
            return None, f"Failed to get/create pipeline state: {ps_error}"

        return pipeline_state, None

    def clear_cache(self):
        """Clears the kernel and pipeline state caches."""
        # Important: Need to handle Objective-C object lifetimes correctly!
        # Releasing objects from the cache requires careful management with metal-python's reference counting.
        # Simply clearing the Python dict might leak Metal objects.
        # This needs proper implementation using release() on cached objects.
        logger.warning("KernelCache.clear_cache() called - requires proper ObjC object release handling (not fully implemented).")
        
        # Placeholder for actual release logic:
        # with autoreleasepool:
        #     for state in self.pipeline_state_cache.values():
        #         state.release() # Assuming metal-python objects have release()
        #     for library in self.library_cache.values():
        #         library.release()

        self.library_cache.clear()
        self.pipeline_state_cache.clear()
        logger.info("Kernel cache cleared (Object release may be incomplete).")

    def __del__(self):
        # Attempt to clear cache on garbage collection, but proper shutdown is better.
        # Proper cleanup should happen in Orchestrator.shutdown()
        if self.cache_enabled:
             logger.debug("KernelCache instance being deleted. Triggering cache clear.")
             # self.clear_cache() # Be cautious calling this here due to potential GC order issues