import logging
from typing import Optional, List, Dict, Tuple
# --- Import metal-python ---
# Use a try-except block to handle cases where it might not be installed
try:
    import metal
    from metal.objc import ObjCClass, ObjCInstance, load_framework, release_pool, autoreleasepool
    from metal.metal import (
        MTLCreateSystemDefaultDevice, MTLDevice, MTLCommandQueue, MTLCommandBuffer,
        MTLComputePipelineState, MTLComputeCommandEncoder, MTLLibrary, MTLCompileOptions,
        MTLResourceOptions, MTLStorageMode, # Import necessary options/enums
        MTLSize # Struct for grid/threadgroup sizes
    )
    METAL_AVAILABLE = True
    load_framework('Metal')
    load_framework('Foundation') # Often needed for strings, errors etc.
    NSString = ObjCClass('NSString')
    NSError = ObjCClass('NSError')

except ImportError:
    logging.warning("metal-python library not found. Metal functionality will be disabled.")
    METAL_AVAILABLE = False
    # Define dummy classes/types so the rest of the code can type hint
    # (These match the dummy types in utils.py - ensure consistency)
    class MTLDevice: pass
    class MTLCommandQueue: pass
    class MTLCommandBuffer: pass
    class MTLComputePipelineState: pass
    class MTLComputeCommandEncoder: pass
    class MTLLibrary: pass
    class MTLCompileOptions: pass
    class MTLResourceOptions: pass
    class MTLStorageMode: pass
    class MTLSize: pass
    class ObjCInstance: pass
    class NSError: pass
    def MTLCreateSystemDefaultDevice(): return None
    def autoreleasepool(func): return func # No-op decorator

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE: Optional[MTLDevice] = None

def get_metal_device(preferred_device_name: Optional[str] = None) -> Optional[MTLDevice]:
    """Gets the default or a specific Metal device."""
    global _DEFAULT_DEVICE
    if not METAL_AVAILABLE:
        logger.warning("Metal not available, cannot get device.")
        return None

    if _DEFAULT_DEVICE and not preferred_device_name: # Cache default device
        return _DEFAULT_DEVICE

    try:
        with autoreleasepool:
            if preferred_device_name:
                # Getting device by name is more complex, requires iterating
                # For simplicity, we'll stick to default for now.
                # You might use IO Registry or other methods if needed.
                 logger.warning(f"Selecting device by name ('{preferred_device_name}') not fully implemented, using default.")
                 # Fall through to default

            device = MTLCreateSystemDefaultDevice()
            if device:
                logger.info(f"Acquired Metal Device: {device.name()}")
                if not preferred_device_name: # Cache if default
                    _DEFAULT_DEVICE = device
                return device
            else:
                logger.error("Failed to create default Metal device.")
                return None
    except Exception as e:
        logger.error(f"Error getting Metal device: {e}", exc_info=True)
        return None

def compile_metal_library(device: MTLDevice, source: str, options: Optional[MTLCompileOptions] = None) -> Tuple[Optional[MTLLibrary], Optional[str]]:
    """Compiles MSL source code into a Metal Library."""
    if not METAL_AVAILABLE or not device:
        return None, "Metal not available or device invalid."

    try:
        with autoreleasepool:
            # Prepare source string for Objective-C bridge
            ns_source = NSString.stringWithString_(source)
            error_ptr = NSError.alloc().init() # Pointer to capture error details

            library = device.newLibraryWithSource_options_error_(ns_source, options, error_ptr)

            if library:
                logger.debug("Metal library compiled successfully.")
                return library, None
            else:
                error_obj = error_ptr.autorelease()
                error_desc = str(error_obj.localizedDescription()) if error_obj else "Unknown compilation error"
                logger.error(f"Metal library compilation failed: {error_desc}")
                return None, error_desc
    except Exception as e:
        logger.error(f"Exception during Metal compilation: {e}", exc_info=True)
        return None, str(e)


def create_compute_pipeline_state(device: MTLDevice, library: MTLLibrary, kernel_name: str) -> Tuple[Optional[MTLComputePipelineState], Optional[str]]:
    """Creates a compute pipeline state for a given kernel function."""
    if not METAL_AVAILABLE or not device or not library:
        return None, "Metal device or library invalid."

    try:
        with autoreleasepool:
             ns_kernel_name = NSString.stringWithString_(kernel_name)
             kernel_function = library.newFunctionWithName_(ns_kernel_name)

             if not kernel_function:
                 err_msg = f"Kernel function '{kernel_name}' not found in the library."
                 logger.error(err_msg)
                 return None, err_msg

             error_ptr = NSError.alloc().init()
             pipeline_state = device.newComputePipelineStateWithFunction_error_(kernel_function, error_ptr)
             kernel_function.release() # Release function obj

             if pipeline_state:
                 logger.debug(f"Compute pipeline state created for kernel '{kernel_name}'.")
                 return pipeline_state, None
             else:
                 error_obj = error_ptr.autorelease()
                 error_desc = str(error_obj.localizedDescription()) if error_obj else "Unknown pipeline state error"
                 logger.error(f"Failed to create compute pipeline state for '{kernel_name}': {error_desc}")
                 return None, error_desc
    except Exception as e:
        logger.error(f"Exception creating pipeline state for '{kernel_name}': {e}", exc_info=True)
        return None, str(e)

# --- Helper to convert Python types to Metal buffer options ---
def get_mtl_resource_options(storage_mode=MTLStorageMode.Shared, cache_mode=0, hazard_tracking=0) -> int:
    """Constructs MTLResourceOptions bitmask."""
    # Note: Exact values depend on metal-python/Metal headers. These are common.
    # Check metal-python docs for MTLResourceOptions enum/constants.
    # Example: Assuming Shared Storage = 0, WriteCombined CPU Cache = (1 << 4), etc.
    options = 0
    if storage_mode == MTLStorageMode.Shared: # Typically 0 or specific enum val
        options |= 0 # Assuming Shared is the default/zero option value
    elif storage_mode == MTLStorageMode.Managed:
         options |= 16 # Placeholder - check actual value
    elif storage_mode == MTLStorageMode.Private:
         options |= 32 # Placeholder - check actual value
         
    # Add CPU Cache Mode and Hazard Tracking options similarly
    # options |= (cache_mode_value << 4)
    # options |= (hazard_tracking_value << 8) # Example bit positions
    
    return options # This needs verification against metal-python specifics

# Add more utility functions:
# - Creating buffers (newBufferWithBytes, newBufferWithLength) with correct options
# - Handling command buffer completion (addCompletedHandler, waitUntilCompleted)
# - Querying device properties (maxThreadsPerThreadgroup, etc.)