import uuid
import time
import logging
from typing import Optional

def generate_unique_id(prefix: str = "id") -> str:
    """Generates a unique ID string."""
    return f"{prefix}-{uuid.uuid4()}"

def time_now_ms() -> float:
    """Gets the current time in milliseconds."""
    return time.monotonic() * 1000.0

class StoppableThreadMixin:
    """Mixin for threads that can be gracefully stopped."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def stopped(self):
        """Check if the stop signal has been received."""
        return self._stop_event.is_set()

# Add more utility functions as needed (e.g., data serialization helpers, etc.)

# --- Dummy Metal types for structure (replace with actual metal-python imports) ---
# These are placeholders until metal-python is integrated
class MTLDevice: pass
class MTLCommandQueue: pass
class MTLCommandBuffer: pass
class MTLComputePipelineState: pass
class MTLBuffer: pass
class MTLLibrary: pass
# --- End Dummy Types ---