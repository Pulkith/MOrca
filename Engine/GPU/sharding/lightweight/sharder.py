import uuid
import time
import random
import threading
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Callable, Tuple

# --- Low-Level Simulated Hardware/Driver Primitives ---

class CUDADeviceInfo:
    """Simulates basic info retrieved about a CUDA device."""
    def __init__(self, gpu_id: int, name: str, total_memory_mb: int):
        self.id = gpu_id
        self.name = name
        self.total_memory_mb = total_memory_mb
        # Simulate other potential properties
        self.compute_capability = f"{random.randint(7, 9)}. {random.randint(0, 5)}"
        self.driver_version = "525.105.17" # Example driver version
        self.cuda_version = "12.0"         # Example CUDA version

    def __repr__(self):
        return f"CUDADeviceInfo(id={self.id}, name='{self.name}', memory={self.total_memory_mb}MB)"

class SimulatedNVML:
    """Simulates NVIDIA Management Library (NVML) interactions."""
    _instance = None
    _lock = threading.Lock()
    _devices: Dict[Tuple[str, int], CUDADeviceInfo] = {} # Key: (node_ip, gpu_id)

    def __new__(cls):
        # Singleton pattern for NVML simulation
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SimulatedNVML, cls).__new__(cls)
                cls._initialized = False
            return cls._instance

    def _initialize_devices(self, node_ip: str, num_gpus: int):
        """Initialize simulated devices for a given node."""
        if not self._initialized: # Allow initialization per node if needed
             # Only print initialization once globally (or remove if too verbose)
             # print("SimulatedNVML: Initializing library...")
             self._initialized = True # Mark as initialized globally

        # print(f"SimulatedNVML: Discovering devices on node {node_ip}...")
        for i in range(num_gpus):
            key = (node_ip, i)
            if key not in self._devices:
                # Simulate different GPU models
                model = random.choice(["NVIDIA A100", "NVIDIA H100", "NVIDIA RTX 4090", "NVIDIA V100"])
                memory = random.choice([16384, 24576, 40960, 81920]) # MB
                self._devices[key] = CUDADeviceInfo(gpu_id=i, name=model, total_memory_mb=memory)
                # print(f"SimulatedNVML: Found device {i} on {node_ip}: {self._devices[key]}")

    def get_device_count(self, node_ip: str) -> int:
        """Get the number of simulated GPUs on a specific node."""
        # Ensure devices for this node are 'discovered' if not already
        # In a real scenario, this might happen automatically or require explicit node registration
        # Here, we just count keys matching the node_ip
        count = sum(1 for ip, _ in self._devices.keys() if ip == node_ip)
        # print(f"SimulatedNVML: Node {node_ip} reports {count} devices.")
        return count


    def get_device_info(self, node_ip: str, gpu_id: int) -> Optional[CUDADeviceInfo]:
        """Get info for a specific simulated GPU on a node."""
        key = (node_ip, gpu_id)
        return self._devices.get(key)

# --- Core System Components ---

class GPUShard:
    """Represents a logical partition (shard) of a Physical GPU."""
    def __init__(self, shard_id: str, physical_gpu: 'PhysicalGPU', memory_mb: int, compute_units: int):
        self.shard_id = shard_id
        self.physical_gpu = physical_gpu
        self.node = physical_gpu.node
        self.allocated_memory_mb = memory_mb
        self.allocated_compute_units = compute_units # Abstract unit representing compute slice
        self.status = "idle"  # idle, busy, error
        self.current_task_id: Optional[str] = None
        self._lock = threading.Lock()
        # print(f"Shard {self.shard_id} created on {self.physical_gpu.unique_id} with {memory_mb}MB memory, {compute_units} CUs.")

    def get_unique_id(self) -> str:
        """Return a globally unique identifier for the shard."""
        return f"node-{self.node.node_id}_gpu-{self.physical_gpu.gpu_id}_shard-{self.shard_id}"

    def _simulate_execution(self, task: 'Task', callback: Callable):
        """Internal method to simulate running a task."""
        try:
            # Simulate acquiring CUDA context/stream for the shard
            # print(f"Shard {self.shard_id}: Acquiring context for Task {task.task_id}...")
            time.sleep(random.uniform(0.05, 0.1)) # Simulate setup time

            # Simulate data transfer (if applicable, based on task properties)
            if task.input_data_size_mb > 0:
                transfer_time = task.input_data_size_mb / 1000 # Simulate 1GB/s transfer
                # print(f"Shard {self.shard_id}: Simulating {task.input_data_size_mb}MB data H2D transfer ({transfer_time:.2f}s)...")
                time.sleep(transfer_time)

            # Simulate the actual computation
            # print(f"Shard {self.shard_id}: Executing Task {task.task_id} ({task.callable_fn.__name__})...")
            start_time = time.time()
            # --- ACTUAL (SIMULATED) WORK ---
            # In a real system, this would involve CUDA calls, kernel launches etc.
            # Here we just call the Python function.
            result = task.callable_fn(*task.args, **task.kwargs)
            # Simulate compute time based on complexity or explicitly defined duration
            compute_duration = task.estimated_duration_s or random.uniform(0.1, 1.5)
            time.sleep(max(0, compute_duration - (time.time() - start_time))) # Ensure minimum duration
            # --- END SIMULATED WORK ---
            end_time = time.time()
            # print(f"Shard {self.shard_id}: Task {task.task_id} completed in {end_time - start_time:.3f}s.")

            # Simulate data transfer back (if applicable)
            if task.output_data_size_mb > 0:
                transfer_time = task.output_data_size_mb / 800 # Simulate 0.8GB/s D2H transfer
                # print(f"Shard {self.shard_id}: Simulating {task.output_data_size_mb}MB data D2H transfer ({transfer_time:.2f}s)...")
                time.sleep(transfer_time)

            task.result = result
            task.status = "completed"
            task.error = None
        except Exception as e:
            print(f"ERROR: Shard {self.shard_id}: Task {task.task_id} failed: {e}")
            task.result = None
            task.status = "failed"
            task.error = str(e)
        finally:
            with self._lock:
                self.status = "idle"
                self.current_task_id = None
            # print(f"Shard {self.shard_id}: Released context. Status: {self.status}")
            # Notify the system the task is done (using the callback)
            if callback:
                callback(task)

    def execute_task(self, task: 'Task', callback: Callable) -> bool:
        """Assigns and starts executing a task on this shard in a new thread."""
        with self._lock:
            if self.status != "idle":
                # print(f"Warning: Shard {self.shard_id} is busy, cannot execute Task {task.task_id}.")
                return False
            self.status = "busy"
            self.current_task_id = task.task_id
            task.status = "running"
            task.assigned_shard_id = self.get_unique_id()
            # print(f"Shard {self.shard_id}: Assigned Task {task.task_id}. Starting execution thread.")

        # Run the simulation in a separate thread to avoid blocking
        thread = threading.Thread(target=self._simulate_execution, args=(task, callback))
        thread.daemon = True # Allow program to exit even if threads are running
        thread.start()
        return True

    def __repr__(self):
        return (f"GPUShard(id={self.shard_id}, gpu={self.physical_gpu.gpu_id}, "
                f"node={self.node.node_id}, mem={self.allocated_memory_mb}MB, "
                f"status={self.status})")

class PhysicalGPU:
    """Represents a physical GPU device on a Node."""
    def __init__(self, node: 'Node', gpu_id: int, device_info: CUDADeviceInfo):
        self.node = node
        self.gpu_id = gpu_id
        self.unique_id = f"node-{node.node_id}_gpu-{gpu_id}"
        self.model_name = device_info.name
        self.total_memory_mb = device_info.total_memory_mb
        self.total_compute_units = 100 # Abstract total compute capacity (e.g., percentage)
        self.available_memory_mb = self.total_memory_mb
        self.available_compute_units = self.total_compute_units
        self.shards: Dict[str, GPUShard] = {}
        self.status = "online" # online, offline, error
        self._lock = threading.Lock()
        # print(f"PhysicalGPU {self.unique_id} ({self.model_name}) initialized on Node {node.node_id}.")

    def create_shard(self, memory_mb: int, compute_units: Optional[int] = None) -> Optional[GPUShard]:
        """Creates a new GPUShard on this GPU if resources are available."""
        with self._lock:
            if self.status != "online":
                print(f"Error: Cannot create shard on offline/error GPU {self.unique_id}.")
                return None

            # Default compute units: proportional to memory if not specified
            if compute_units is None:
                compute_units = int(self.total_compute_units * (memory_mb / self.total_memory_mb))
            compute_units = max(1, compute_units) # Ensure at least 1 CU

            if memory_mb <= 0 or compute_units <=0:
                print(f"Error: Invalid resource request for shard on {self.unique_id} (Mem: {memory_mb}, CU: {compute_units}).")
                return None

            if self.available_memory_mb >= memory_mb and self.available_compute_units >= compute_units:
                self.available_memory_mb -= memory_mb
                self.available_compute_units -= compute_units
                shard_id = str(uuid.uuid4())[:8] # Short unique ID for the shard
                new_shard = GPUShard(shard_id, self, memory_mb, compute_units)
                self.shards[shard_id] = new_shard
                # print(f"Successfully created Shard {shard_id} on GPU {self.unique_id}. "
                #      f"Available Mem: {self.available_memory_mb}MB, Available CUs: {self.available_compute_units}")
                return new_shard
            else:
                print(f"Error: Insufficient resources on GPU {self.unique_id} to create shard. "
                      f"Requested Mem: {memory_mb} (Avail: {self.available_memory_mb}), "
                      f"Requested CUs: {compute_units} (Avail: {self.available_compute_units})")
                return None

    def delete_shard(self, shard_id: str) -> bool:
        """Deletes a GPUShard and frees its resources."""
        with self._lock:
            if shard_id in self.shards:
                shard = self.shards[shard_id]
                if shard.status == "busy":
                    print(f"Error: Cannot delete busy Shard {shard_id} on GPU {self.unique_id}.")
                    return False

                self.available_memory_mb += shard.allocated_memory_mb
                self.available_compute_units += shard.allocated_compute_units
                del self.shards[shard_id]
                # print(f"Successfully deleted Shard {shard_id} from GPU {self.unique_id}. "
                #       f"Available Mem: {self.available_memory_mb}MB, Available CUs: {self.available_compute_units}")
                return True
            else:
                print(f"Error: Shard {shard_id} not found on GPU {self.unique_id}.")
                return False

    def get_load(self) -> float:
        """Calculate the current load (0.0 to 1.0) based on busy shards."""
        with self._lock:
            if not self.shards:
                return 0.0
            busy_shards = sum(1 for shard in self.shards.values() if shard.status == 'busy')
            return busy_shards / len(self.shards)

    def get_resource_utilization(self) -> Dict[str, float]:
        """Returns memory and compute utilization percentages."""
        with self._lock:
            mem_used = self.total_memory_mb - self.available_memory_mb
            cu_used = self.total_compute_units - self.available_compute_units
            mem_util = (mem_used / self.total_memory_mb) * 100 if self.total_memory_mb > 0 else 0
            cu_util = (cu_used / self.total_compute_units) * 100 if self.total_compute_units > 0 else 0
            return {"memory_utilization_percent": round(mem_util, 2),
                    "compute_utilization_percent": round(cu_util, 2)}

    def __repr__(self):
        shard_count = len(self.shards)
        return (f"PhysicalGPU(id={self.gpu_id}, node={self.node.node_id}, model='{self.model_name}', "
                f"mem={self.available_memory_mb}/{self.total_memory_mb}MB, "
                f"shards={shard_count}, status={self.status})")


class Node:
    """Represents a compute node in the cluster."""
    def __init__(self, node_id: str, ip_address: str, num_gpus: int):
        self.node_id = node_id
        self.ip_address = ip_address
        self.gpus: Dict[int, PhysicalGPU] = {}
        self.status = "initializing" # initializing, online, offline, error
        self._nvml = SimulatedNVML()
        self._discover_gpus(num_gpus)
        self.status = "online" if self.gpus else "error"
        # print(f"Node {self.node_id} ({self.ip_address}) initialized with {len(self.gpus)} GPUs. Status: {self.status}")

    def _discover_gpus(self, expected_gpus: int):
        """Simulates discovering GPUs on this node using NVML."""
        # print(f"Node {self.node_id}: Discovering GPUs (expecting {expected_gpus})...")
        self._nvml._initialize_devices(self.ip_address, expected_gpus) # Tell NVML sim about these
        # count = self._nvml.get_device_count(self.ip_address) # This count might not be reliable until initialized

        for i in range(expected_gpus): # Assume we know how many to expect
            device_info = self._nvml.get_device_info(self.ip_address, i)
            if device_info:
                self.gpus[i] = PhysicalGPU(self, i, device_info)
            else:
                 # This case shouldn't happen with current NVML sim logic, but good practice
                 print(f"Warning: Node {self.node_id}: Could not get info for expected GPU {i}.")

    def get_gpu(self, gpu_id: int) -> Optional[PhysicalGPU]:
        """Get a specific GPU object by its ID."""
        return self.gpus.get(gpu_id)

    def get_node_status(self) -> Dict[str, Any]:
        """Provides a summary of the node's status and its GPUs."""
        gpu_statuses = {gpu.unique_id: gpu.get_resource_utilization() for gpu in self.gpus.values()}
        return {
            "node_id": self.node_id,
            "ip_address": self.ip_address,
            "status": self.status,
            "gpu_count": len(self.gpus),
            "gpu_details": gpu_statuses
        }

    def __repr__(self):
        return (f"Node(id={self.node_id}, ip={self.ip_address}, "
                f"gpus={len(self.gpus)}, status={self.status})")


class Task:
    """Represents a computational task to be executed on a GPUShard."""
    def __init__(self, callable_fn: Callable, args: Tuple = (), kwargs: Dict = None,
                 required_memory_mb: int = 128,
                 required_compute_units: int = 1,
                 priority: int = 0, # Lower number means higher priority
                 input_data_size_mb: int = 0, # For simulating data transfer
                 output_data_size_mb: int = 0, # For simulating data transfer
                 estimated_duration_s: Optional[float] = None): # For simulation
        self.task_id = str(uuid.uuid4())
        self.callable_fn = callable_fn
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.required_memory_mb = required_memory_mb
        self.required_compute_units = required_compute_units
        self.priority = priority
        self.input_data_size_mb = input_data_size_mb
        self.output_data_size_mb = output_data_size_mb
        self.estimated_duration_s = estimated_duration_s

        self.status = "pending"  # pending, scheduled, running, completed, failed
        self.result: Any = None
        self.error: Optional[str] = None
        self.submit_time = time.time()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.assigned_shard_id: Optional[str] = None

    def __lt__(self, other: 'Task'):
        # For priority queue implementation (lower priority number = higher priority)
        return self.priority < other.priority

    def __repr__(self):
        return (f"Task(id={self.task_id}, func={self.callable_fn.__name__}, "
                f"status={self.status}, priority={self.priority}, "
                f"mem={self.required_memory_mb}MB)")

# --- Management and Scheduling ---

class ClusterManager:
    """Manages the nodes and provides a cluster-wide view."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self._lock = threading.Lock()
        # print("ClusterManager initialized.")

    def add_node(self, node_id: str, ip_address: str, num_gpus: int):
        """Adds a new node to the cluster."""
        with self._lock:
            if node_id in self.nodes:
                print(f"Warning: Node {node_id} already exists.")
                return
            node = Node(node_id, ip_address, num_gpus)
            self.nodes[node_id] = node
            # print(f"Node {node_id} added to the cluster.")

    def remove_node(self, node_id: str):
        """Removes a node from the cluster."""
        with self._lock:
            if node_id in self.nodes:
                # In a real system, need graceful shutdown/task migration logic
                node = self.nodes.pop(node_id)
                node.status = "offline" # Mark node as offline
                # Mark all its GPUs and shards as offline/error as well
                for gpu in node.gpus.values():
                    gpu.status = "offline"
                    for shard in gpu.shards.values():
                        shard.status = "error" # Shards become unusable
                # print(f"Node {node_id} removed from the cluster.")
            else:
                print(f"Warning: Node {node_id} not found.")

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieves a node by its ID."""
        with self._lock:
            return self.nodes.get(node_id)

    def get_all_gpus(self) -> List[PhysicalGPU]:
        """Returns a list of all physical GPUs across all online nodes."""
        gpus = []
        with self._lock:
            for node in self.nodes.values():
                if node.status == "online":
                    gpus.extend(node.gpus.values())
        return gpus

    def get_all_shards(self) -> List[GPUShard]:
        """Returns a list of all shards across all online nodes and GPUs."""
        shards = []
        with self._lock:
            for node in self.nodes.values():
                if node.status == "online":
                    for gpu in node.gpus.values():
                         if gpu.status == "online":
                            shards.extend(gpu.shards.values())
        return shards

    def get_cluster_status(self) -> Dict[str, Any]:
        """Provides a summary of the entire cluster's status."""
        with self._lock:
            node_statuses = {nid: node.get_node_status() for nid, node in self.nodes.items()}
            total_gpus = sum(len(node.gpus) for node in self.nodes.values())
            online_nodes = sum(1 for node in self.nodes.values() if node.status == "online")
            return {
                "total_nodes": len(self.nodes),
                "online_nodes": online_nodes,
                "total_gpus": total_gpus,
                "nodes": node_statuses
            }

class Scheduler:
    """Handles scheduling tasks onto appropriate GPUShards."""
    def __init__(self, cluster_manager: ClusterManager, policy: str = 'least_loaded_gpu'):
        self.cluster_manager = cluster_manager
        self.set_policy(policy)
        self._task_queue = deque() # Simple FIFO queue for pending tasks
        self._lock = threading.Lock()
        self._schedule_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self._stop_event = threading.Event()
        # print(f"Scheduler initialized with policy: {self.policy_name}")

    def set_policy(self, policy: str):
        """Set the scheduling policy."""
        valid_policies = ['round_robin', 'least_loaded_gpu', 'best_fit_memory', 'random']
        if policy not in valid_policies:
            raise ValueError(f"Invalid scheduling policy. Choose from: {valid_policies}")
        self.policy = getattr(self, f"_policy_{policy}")
        self.policy_name = policy
        self._round_robin_index = 0 # For round_robin policy
        # print(f"Scheduler policy set to: {self.policy_name}")

    def submit_task(self, task: Task):
        """Adds a task to the pending queue."""
        with self._lock:
            self._task_queue.append(task)
            # print(f"Scheduler: Task {task.task_id} submitted. Queue size: {len(self._task_queue)}")

    def _scheduling_loop(self):
        """Continuously tries to schedule tasks from the queue."""
        # print("Scheduler: Starting scheduling loop...")
        while not self._stop_event.is_set():
            task_to_schedule = None
            with self._lock:
                if self._task_queue:
                    task_to_schedule = self._task_queue.popleft() # Get the next task

            if task_to_schedule:
                # print(f"Scheduler: Attempting to schedule Task {task_to_schedule.task_id}...")
                scheduled = self.schedule_task(task_to_schedule)
                if not scheduled:
                    # If scheduling failed, put it back at the front of the queue
                    # (Could implement backoff or priority adjustments here)
                    # print(f"Scheduler: Failed to schedule Task {task_to_schedule.task_id}, re-queuing.")
                    with self._lock:
                         self._task_queue.appendleft(task_to_schedule)
                    time.sleep(0.5) # Wait a bit before retrying the same task
                else:
                    pass # print(f"Scheduler: Successfully scheduled Task {task_to_schedule.task_id}.")
            else:
                # No tasks, wait a bit
                time.sleep(0.1)
        # print("Scheduler: Stopping scheduling loop.")


    def start(self):
        """Starts the scheduling loop thread."""
        if not self._schedule_thread.is_alive():
            self._stop_event.clear()
            self._schedule_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
            self._schedule_thread.start()
            # print("Scheduler: Scheduling thread started.")

    def stop(self):
        """Stops the scheduling loop thread gracefully."""
        # print("Scheduler: Signalling scheduling thread to stop...")
        self._stop_event.set()
        self._schedule_thread.join(timeout=5) # Wait for thread to finish
        if self._schedule_thread.is_alive():
             print("Warning: Scheduler thread did not stop cleanly.")
        # print("Scheduler: Scheduling thread stopped.")


    def schedule_task(self, task: Task) -> bool:
        """Finds a suitable shard using the current policy and assigns the task."""
        suitable_shard = self.policy(task)
        if suitable_shard:
            # print(f"Scheduler: Found suitable shard {suitable_shard.get_unique_id()} for Task {task.task_id}.")
            # The callback updates the task status centrally
            success = suitable_shard.execute_task(task, self._task_completion_callback)
            if success:
                task.status = "scheduled" # Mark as scheduled before it starts running
                task.start_time = time.time()
                return True
            else:
                 # Shard became busy between selection and execution attempt
                 # print(f"Scheduler: Shard {suitable_shard.get_unique_id()} became busy. Task {task.task_id} scheduling failed.")
                 return False
        else:
            # print(f"Scheduler: No suitable shard found for Task {task.task_id} with current policy {self.policy_name}.")
            task.status = "pending" # Remain pending
            return False

    def _task_completion_callback(self, task: Task):
        """Callback function passed to GPUShard upon task completion/failure."""
        task.end_time = time.time()
        duration = task.end_time - (task.start_time if task.start_time else task.submit_time)
        # print(f"Scheduler Callback: Task {task.task_id} finished with status '{task.status}'. Duration: {duration:.3f}s")
        # Here you could add logic for task chaining, notifications, logging, etc.

    # --- Scheduling Policy Implementations ---

    def _find_suitable_shards(self, task: Task) -> List[GPUShard]:
        """Helper to get all shards that meet basic task requirements."""
        available_shards = []
        all_shards = self.cluster_manager.get_all_shards()
        for shard in all_shards:
            # Check status and resource requirements
            if (shard.status == "idle" and
                shard.allocated_memory_mb >= task.required_memory_mb and
                shard.allocated_compute_units >= task.required_compute_units):
                available_shards.append(shard)
        return available_shards

    def _policy_round_robin(self, task: Task) -> Optional[GPUShard]:
        """Finds the next available shard in round-robin fashion."""
        suitable_shards = self._find_suitable_shards(task)
        if not suitable_shards:
            return None

        num_shards = len(suitable_shards)
        with self._lock: # Protect the index
            start_index = self._round_robin_index % num_shards
            # Try to find the next available one starting from the index
            for i in range(num_shards):
                current_index = (start_index + i) % num_shards
                shard = suitable_shards[current_index]
                # Double check status just in case
                if shard.status == "idle":
                     self._round_robin_index = (current_index + 1) % num_shards
                     return shard
        return None # Should ideally not happen if suitable_shards is not empty and check worked

    def _policy_least_loaded_gpu(self, task: Task) -> Optional[GPUShard]:
        """Finds an idle shard on the physical GPU with the lowest current load."""
        suitable_shards = self._find_suitable_shards(task)
        if not suitable_shards:
            return None

        # Group shards by their physical GPU and calculate GPU load
        gpu_loads = defaultdict(list)
        for shard in suitable_shards:
            gpu_loads[shard.physical_gpu.unique_id].append(shard)

        # Find the GPU with the minimum load that has suitable shards
        min_load = float('inf')
        best_gpu_id = None
        for gpu_id in gpu_loads.keys():
            # Access the PhysicalGPU object to get its load
            # This assumes shard.physical_gpu points correctly
            gpu = suitable_shards[0].physical_gpu # Get one shard to find the GPU
            if gpu_loads[gpu_id]:
                 gpu = gpu_loads[gpu_id][0].physical_gpu
                 load = gpu.get_load()
                 if load < min_load:
                     min_load = load
                     best_gpu_id = gpu_id

        if best_gpu_id:
            # Return the first suitable shard found on the least loaded GPU
            return gpu_loads[best_gpu_id][0]
        else:
            return None # No suitable GPU found

    def _policy_best_fit_memory(self, task: Task) -> Optional[GPUShard]:
        """Finds the idle shard that fits the task's memory requirement most closely."""
        suitable_shards = self._find_suitable_shards(task)
        if not suitable_shards:
            return None

        best_shard = None
        min_mem_diff = float('inf')

        for shard in suitable_shards:
            mem_diff = shard.allocated_memory_mb - task.required_memory_mb
            if mem_diff >= 0 and mem_diff < min_mem_diff:
                min_mem_diff = mem_diff
                best_shard = shard
            # Optional: If perfect match, take it immediately
            # if mem_diff == 0:
            #    return shard

        return best_shard

    def _policy_random(self, task: Task) -> Optional[GPUShard]:
        """Randomly selects an available shard that meets requirements."""
        suitable_shards = self._find_suitable_shards(task)
        if not suitable_shards:
            return None
        return random.choice(suitable_shards)


# --- User-Facing API ---

class GPUShardingSystem:
    """High-level API for interacting with the GPU sharding system."""
    def __init__(self, scheduling_policy: str = 'least_loaded_gpu'):
        print("Initializing Advanced GPU Sharding System...")
        self.cluster_manager = ClusterManager()
        self.scheduler = Scheduler(self.cluster_manager, policy=scheduling_policy)
        self._tasks: Dict[str, Task] = {} # Central tracking for all submitted tasks
        self._tasks_lock = threading.Lock()
        self.scheduler.start()
        print(f"System Ready. Scheduler Policy: {scheduling_policy}")

    def add_compute_node(self, node_id: str, ip_address: str, num_gpus: int):
        """Register a new compute node with the system."""
        print(f"API: Adding Node {node_id} ({ip_address}) with {num_gpus} GPU(s)...")
        self.cluster_manager.add_node(node_id, ip_address, num_gpus)

    def remove_compute_node(self, node_id: str):
        """De-register a compute node."""
        print(f"API: Removing Node {node_id}...")
        self.cluster_manager.remove_node(node_id)

    def create_shard_on_gpu(self, node_id: str, gpu_id: int, memory_mb: int, compute_units: Optional[int] = None) -> Optional[str]:
        """Creates a shard on a specific GPU."""
        print(f"API: Requesting shard creation on Node {node_id}, GPU {gpu_id} (Mem: {memory_mb}MB)...")
        node = self.cluster_manager.get_node(node_id)
        if not node or node.status != 'online':
            print(f"API Error: Node {node_id} not found or not online.")
            return None
        gpu = node.get_gpu(gpu_id)
        if not gpu or gpu.status != 'online':
            print(f"API Error: GPU {gpu_id} on Node {node_id} not found or not online.")
            return None

        shard = gpu.create_shard(memory_mb, compute_units)
        if shard:
            print(f"API: Shard {shard.shard_id} created successfully on {gpu.unique_id}.")
            return shard.get_unique_id()
        else:
            print(f"API Error: Failed to create shard on {gpu.unique_id}.")
            return None

    def delete_shard(self, shard_unique_id: str) -> bool:
        """Deletes a specific shard using its unique ID."""
        print(f"API: Requesting deletion of Shard {shard_unique_id}...")
        # Parse node_id, gpu_id, shard_id from unique_id (fragile parsing, improve in real system)
        try:
            parts = shard_unique_id.split('_')
            node_id = parts[0].split('-')[1]
            gpu_id = int(parts[1].split('-')[1])
            shard_id = parts[2].split('-')[1]
        except (IndexError, ValueError):
            print(f"API Error: Invalid shard unique ID format: {shard_unique_id}")
            return False

        node = self.cluster_manager.get_node(node_id)
        if not node: return False # Error already printed by get_node if missing
        gpu = node.get_gpu(gpu_id)
        if not gpu: return False

        deleted = gpu.delete_shard(shard_id)
        if deleted:
            print(f"API: Shard {shard_unique_id} deleted successfully.")
        else:
            print(f"API Error: Failed to delete shard {shard_unique_id}.")
        return deleted

    def submit_task(self, callable_fn: Callable, args: Tuple = (), kwargs: Dict = None,
                    required_memory_mb: int = 128, required_compute_units: int = 1,
                    priority: int = 0, input_data_size_mb: int = 0,
                    output_data_size_mb: int = 0, estimated_duration_s: Optional[float] = None) -> str:
        """Submits a task to the system for execution."""
        task = Task(
            callable_fn=callable_fn,
            args=args,
            kwargs=kwargs,
            required_memory_mb=required_memory_mb,
            required_compute_units=required_compute_units,
            priority=priority,
            input_data_size_mb=input_data_size_mb,
            output_data_size_mb=output_data_size_mb,
            estimated_duration_s=estimated_duration_s
        )
        with self._tasks_lock:
            self._tasks[task.task_id] = task
        self.scheduler.submit_task(task)
        print(f"API: Task {task.task_id} ({callable_fn.__name__}) submitted.")
        return task.task_id

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Gets the current status of a submitted task."""
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            return task.status if task else None

    def get_task_result(self, task_id: str, wait: bool = False, timeout: Optional[float] = None) -> Any:
        """Gets the result of a completed task. Can optionally wait."""
        start_wait = time.time()
        while True:
            with self._tasks_lock:
                task = self._tasks.get(task_id)
                if not task:
                    raise ValueError(f"Task ID {task_id} not found.")

                if task.status == "completed":
                    return task.result
                elif task.status == "failed":
                    raise RuntimeError(f"Task {task_id} failed: {task.error}")
                elif not wait:
                    return None # Indicate still running or pending
                elif timeout is not None and (time.time() - start_wait > timeout):
                     raise TimeoutError(f"Timeout waiting for task {task_id} result.")

            # If waiting, sleep briefly before checking again
            time.sleep(0.1)


    def list_gpus(self) -> List[Dict[str, Any]]:
        """Lists all detected physical GPUs and their status."""
        gpu_list = []
        for gpu in self.cluster_manager.get_all_gpus():
             gpu_list.append({
                 "unique_id": gpu.unique_id,
                 "node_id": gpu.node.node_id,
                 "gpu_id_on_node": gpu.gpu_id,
                 "model": gpu.model_name,
                 "total_memory_mb": gpu.total_memory_mb,
                 "available_memory_mb": gpu.available_memory_mb,
                 "total_compute_units": gpu.total_compute_units,
                 "available_compute_units": gpu.available_compute_units,
                 "shard_count": len(gpu.shards),
                 "status": gpu.status,
                 "utilization": gpu.get_resource_utilization()
             })
        return gpu_list

    def list_shards(self) -> List[Dict[str, Any]]:
        """Lists all created shards and their status."""
        shard_list = []
        for shard in self.cluster_manager.get_all_shards():
            shard_list.append({
                "unique_id": shard.get_unique_id(),
                "physical_gpu_id": shard.physical_gpu.unique_id,
                "node_id": shard.node.node_id,
                "shard_id_on_gpu": shard.shard_id,
                "allocated_memory_mb": shard.allocated_memory_mb,
                "allocated_compute_units": shard.allocated_compute_units,
                "status": shard.status,
                "current_task_id": shard.current_task_id
            })
        return shard_list

    def get_cluster_overview(self) -> Dict[str, Any]:
        """Provides a high-level overview of the cluster state."""
        return self.cluster_manager.get_cluster_status()

    def shutdown(self):
        """Shuts down the scheduler thread."""
        print("API: Shutting down GPUShardingSystem...")
        self.scheduler.stop()
        print("API: System shut down.")

# --- Example Usage ---

class ExampleUsage:
    """Demonstrates how to use the GPUShardingSystem."""

    def __init__(self):
        # Use the 'best_fit_memory' policy for this example
        self.system = GPUShardingSystem(scheduling_policy='best_fit_memory')

    def setup_cluster(self):
        """Simulate adding nodes to the cluster."""
        print("\n--- Setting up Cluster ---")
        self.system.add_compute_node(node_id="node-01", ip_address="192.168.1.101", num_gpus=2)
        self.system.add_compute_node(node_id="node-02", ip_address="192.168.1.102", num_gpus=4)
        print("Cluster setup complete.")
        print(f"Initial Cluster Overview: {self.system.get_cluster_overview()}")


    def create_shards(self):
        """Create some example shards."""
        print("\n--- Creating Shards ---")
        # Node 1 GPUs
        self.shard1_id = self.system.create_shard_on_gpu("node-01", 0, memory_mb=8192, compute_units=40) # ~40% CU
        self.shard2_id = self.system.create_shard_on_gpu("node-01", 0, memory_mb=8192, compute_units=40) # ~40% CU
        self.shard3_id = self.system.create_shard_on_gpu("node-01", 1, memory_mb=12288) # ~50% Mem -> ~50 CU default

        # Node 2 GPUs
        self.shard4_id = self.system.create_shard_on_gpu("node-02", 0, memory_mb=4096, compute_units=10)
        self.shard5_id = self.system.create_shard_on_gpu("node-02", 1, memory_mb=16384, compute_units=50)
        self.shard6_id = self.system.create_shard_on_gpu("node-02", 2, memory_mb=20480) # ~50% Mem -> ~50 CU default
        self.shard7_id = self.system.create_shard_on_gpu("node-02", 3, memory_mb=40960) # ~50% Mem -> ~50 CU default

        # Attempt to create a shard that likely fails (insufficient resources)
        self.system.create_shard_on_gpu("node-01", 0, memory_mb=10000) # Should fail if total is e.g. 16GB

        print("\nShard Creation Attempt Complete.")
        print("Current Shards:")
        for shard_info in self.system.list_shards():
            print(f"  {shard_info}")
        print("\nCurrent GPUs:")
        for gpu_info in self.system.list_gpus():
            print(f"  {gpu_info}")


    def submit_tasks(self):
        """Define some dummy tasks and submit them."""
        print("\n--- Submitting Tasks ---")

        # Define some simple functions to simulate workloads
        def simple_math_task(x, y):
            # print(f"Executing simple_math_task({x}, {y})...")
            result = (x * y) + (x / (y + 0.01)) # Avoid division by zero
            time.sleep(0.2) # Simulate work
            return result

        def data_processing_task(data_id):
            # print(f"Executing data_processing_task for data_id {data_id}...")
            time.sleep(0.5) # Simulate more work
            return {"status": "processed", "data_id": data_id, "result_hash": uuid.uuid4().hex}

        def high_memory_task(size_gb):
            # print(f"Executing high_memory_task requesting {size_gb}GB equivalent...")
            # Actual memory isn't used here, just simulates requirement
            time.sleep(0.8)
            return f"Processed {size_gb}GB data block"

        # Submit tasks with varying requirements
        self.task_ids = []
        self.task_ids.append(self.system.submit_task(simple_math_task, args=(10, 5), required_memory_mb=512, priority=1))
        self.task_ids.append(self.system.submit_task(data_processing_task, args=("dataset_alpha",), required_memory_mb=2048, input_data_size_mb=100, output_data_size_mb=20))
        self.task_ids.append(self.system.submit_task(simple_math_task, args=(100, 2), required_memory_mb=256))
        self.task_ids.append(self.system.submit_task(high_memory_task, args=(15,), required_memory_mb=15360)) # Requires 15GB shard
        self.task_ids.append(self.system.submit_task(data_processing_task, args=("dataset_beta",), required_memory_mb=4096, priority=0)) # High priority
        self.task_ids.append(self.system.submit_task(simple_math_task, args=(7, 3), required_memory_mb=1024))
        self.task_ids.append(self.system.submit_task(high_memory_task, args=(30,), required_memory_mb=30720)) # Requires 30GB shard


        print(f"\nSubmitted {len(self.task_ids)} tasks.")

    def monitor_and_get_results(self):
        """Monitor task progress and retrieve results."""
        print("\n--- Monitoring Tasks & Getting Results ---")
        completed_tasks = 0
        max_wait_time = 20 # seconds
        start_time = time.time()

        while completed_tasks < len(self.task_ids) and (time.time() - start_time) < max_wait_time:
            print(f"\nCluster State at T+{time.time() - start_time:.1f}s:")
            # Display concise status
            statuses = {tid: self.system.get_task_status(tid) for tid in self.task_ids}
            print(f"  Task Statuses: {statuses}")
            active_shards = [s for s in self.system.list_shards() if s['status'] == 'busy']
            print(f"  Active Shards ({len(active_shards)}): {[s['unique_id'] for s in active_shards]}")

            completed_tasks = sum(1 for status in statuses.values() if status in ["completed", "failed"])
            if completed_tasks == len(self.task_ids):
                break
            time.sleep(2) # Wait before next status check

        print("\n--- Final Results ---")
        for task_id in self.task_ids:
            try:
                # Wait briefly for potentially just-finished tasks
                result = self.system.get_task_result(task_id, wait=True, timeout=1.0)
                status = self.system.get_task_status(task_id)
                print(f"Task {task_id}: Status={status}, Result={result}")
            except (RuntimeError, TimeoutError, ValueError) as e:
                status = self.system.get_task_status(task_id)
                print(f"Task {task_id}: Status={status}, Error retrieving result: {e}")
            except Exception as e:
                 status = self.system.get_task_status(task_id)
                 print(f"Task {task_id}: Status={status}, Unexpected Error: {e}")


    def cleanup(self):
        """Clean up resources."""
        print("\n--- Cleaning Up ---")
        # Example deletion (optional) - In real use, shards might persist
        if hasattr(self, 'shard1_id') and self.shard1_id:
             self.system.delete_shard(self.shard1_id)

        self.system.shutdown()
        print("System shut down.")

    def run(self):
        """Execute the full example workflow."""
        try:
            self.setup_cluster()
            self.create_shards()
            self.submit_tasks()
            self.monitor_and_get_results()
        finally:
            # Ensure cleanup happens even if errors occur
            self.cleanup()

# --- Main Execution ---
if __name__ == "__main__":
    example = ExampleUsage()
    example.run()