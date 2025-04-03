import logging
import multiprocessing
from enum import Enum, auto
from typing import Optional, Dict, Any

class SchedulerType(Enum):
    FIFO = auto()
    PRIORITY = auto()
    ROUND_ROBIN = auto()
    # Add more complex types: Shortest Job First (Estimated), Weighted Fair Queueing
    CUSTOM = auto()

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class FairnessPolicy(Enum):
    NONE = auto()
    TASK_QUANTUM = auto() # Basic time-slicing simulation
    # Add more: Resource Aware, Deadline Based

class Config:
    """Manages the configuration settings for the Metal Orchestrator."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern for global config access
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, **kwargs):
        # Initialize only once
        if self._initialized:
            return
        self._initialized = True

        # --- Core Settings ---
        # Attempt to get physical core count, fallback to logical, default to 6
        try:
            physical_cores = multiprocessing.cpu_count() // 2 # Heuristic for hyperthreading
        except NotImplementedError:
            physical_cores = multiprocessing.cpu_count() # Fallback to logical
        
        self.num_workers: int = kwargs.get('num_workers', min(physical_cores or 6, 8)) # Default, but changeable, capped reasonably

        # --- Metal Device ---
        self.preferred_device_name: Optional[str] = kwargs.get('preferred_device_name', None)
        self.force_cpu_as_worker: bool = kwargs.get('force_cpu_as_worker', False) # For testing without GPU

        # --- Scheduling ---
        self.scheduler_type: SchedulerType = kwargs.get('scheduler_type', SchedulerType.PRIORITY)
        self.custom_scheduler_class = kwargs.get('custom_scheduler_class', None)
        self.task_default_priority: int = kwargs.get('task_default_priority', 50) # 0=highest, 100=lowest

        # --- Fairness ---
        self.fairness_policy: FairnessPolicy = kwargs.get('fairness_policy', FairnessPolicy.TASK_QUANTUM)
        self.task_time_quantum_ms: float = kwargs.get('task_time_quantum_ms', 100.0) # Max estimated runtime before potential rescheduling

        # --- Task Management ---
        self.max_queue_size: int = kwargs.get('max_queue_size', 0) # 0 for unlimited
        self.max_active_tasks_per_worker: int = kwargs.get('max_active_tasks_per_worker', 2) # Allow pipelining

        # --- Kernel Management ---
        self.kernel_cache_enabled: bool = kwargs.get('kernel_cache_enabled', True)
        self.default_shader_search_paths: list[str] = kwargs.get('default_shader_search_paths', ['./shaders/'])

        # --- Callbacks ---
        self.callback_in_main_thread: bool = kwargs.get('callback_in_main_thread', False) # Execute in main event loop or separate threadpool

        # --- IPC & Data Handling ---
        # Options for potential future shared memory strategies
        self.ipc_mechanism: str = kwargs.get('ipc_mechanism', 'queue') # could be 'shared_memory' later

        # --- Logging ---
        self.log_level: LogLevel = kwargs.get('log_level', LogLevel.INFO)
        self.log_file: Optional[str] = kwargs.get('log_file', None) # None for stdout
        self.log_format: str = kwargs.get('log_format', '%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')

        # --- Error Handling ---
        self.max_task_retries: int = kwargs.get('max_task_retries', 1)
        self.worker_heartbeat_interval_sec: float = kwargs.get('worker_heartbeat_interval_sec', 10.0)
        self.worker_timeout_sec: float = kwargs.get('worker_timeout_sec', 60.0)

        # --- Resource Monitoring (Advanced/Future) ---
        self.enable_resource_monitoring: bool = kwargs.get('enable_resource_monitoring', False) # Monitor GPU temp, power, etc. (requires platform specifics)
        self.gpu_memory_safety_margin: float = kwargs.get('gpu_memory_safety_margin', 0.1) # Reserve 10% VRAM

        # --- Validation ---
        self._validate_config()

        # --- Apply Logging Immediately ---
        self._setup_logging()

    def _validate_config(self):
        """Perform basic validation of configuration options."""
        if not isinstance(self.num_workers, int) or self.num_workers <= 0:
            raise ValueError("num_workers must be a positive integer.")
        if self.scheduler_type == SchedulerType.CUSTOM and not self.custom_scheduler_class:
            raise ValueError("custom_scheduler_class must be provided for CUSTOM scheduler type.")
        # Add more validations as needed

    def _setup_logging(self):
        """Configure the root logger."""
        logging.basicConfig(
            level=self.log_level.value,
            format=self.log_format,
            filename=self.log_file,
            filemode='a' if self.log_file else None
        )
        # Silence overly verbose libraries if necessary
        # logging.getLogger("some_library").setLevel(logging.WARNING)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Safely get a configuration setting."""
        return getattr(self, key, default)

    def update_setting(self, key: str, value: Any):
        """Update a configuration setting dynamically (use with caution)."""
        if hasattr(self, key):
            setattr(self, key, value)
            # Potentially re-validate or notify components if needed
            logging.info(f"Configuration updated: {key} = {value}")
            if key == 'log_level': # Example: Reconfigure logging if level changes
                 self._setup_logging()
        else:
            logging.warning(f"Attempted to update non-existent config key: {key}")

# Global access point
CONFIG = Config()

def update_global_config(**kwargs):
    """Convenience function to update global config before initialization."""
    # This should ideally be called *before* MetalOrchestrator is created.
    global CONFIG
    # Re-initialize singleton with new kwargs if needed, or update existing
    if not CONFIG._initialized:
         CONFIG = Config(**kwargs)
    else:
        for key, value in kwargs.items():
            CONFIG.update_setting(key, value)