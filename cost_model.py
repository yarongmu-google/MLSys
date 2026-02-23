import json
import os
from typing import List, Dict, Any, Tuple, Optional

class MLSysProblem:
    def __init__(self, data: Dict[str, Any]):
        self.widths = data["widths"]
        self.heights = data["heights"]
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]
        self.base_costs = data["base_costs"]
        self.op_types = data["op_types"]
        self.fast_memory_capacity = data["fast_memory_capacity"]
        self.slow_memory_bandwidth = data["slow_memory_bandwidth"]
        self.native_granularity = data["native_granularity"]
        
        self.num_ops = len(self.op_types)
        self.num_tensors = len(self.widths)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(data)

class MemoryManager:
    def __init__(self, capacity: int, bandwidth: float):
        self.capacity = capacity
        self.bandwidth = bandwidth
        self.resident_tensors: Dict[int, int] = {}  # tensor_id -> full_size

    def get_load_cost(self, tensor_id: int, size: int) -> float:
        """Returns 0 if tensor is resident, else calculates transfer time."""
        if tensor_id in self.resident_tensors:
            return 0.0
        return size / self.bandwidth

    def get_current_occupancy(self) -> int:
        return sum(self.resident_tensors.values())

    def update_residency(self, retain_ids: List[int], problem: 'MLSysProblem'):
        """Sets the SRAM state to only the specified retained tensors."""
        new_resident = {}
        for tid in retain_ids:
            size = problem.widths[tid] * problem.heights[tid]
            new_resident[tid] = size
        
        self.resident_tensors = new_resident
        total_size = sum(new_resident.values())
        if total_size > self.capacity:
            return False, f"Retained tensors {retain_ids} (Size: {total_size}) exceed capacity {self.capacity}"
        return True, "OK"

class CostModel:
    def __init__(self, problem: MLSysProblem):
        self.p = problem
        self.memory_manager = MemoryManager(problem.fast_memory_capacity, problem.slow_memory_bandwidth)

    def get_tensor_full_size(self, tensor_idx: int) -> int:
        return self.p.widths[tensor_idx] * self.p.heights[tensor_idx]

    def get_tensor_slice_size(self, tensor_idx: int, w: int, h: int, k: int, is_lhs: bool = False, is_rhs: bool = False, is_matmul: bool = False) -> int:
        """
        Calculates the size of a tensor slice based on granularity.
        - Pointwise: w * h
        - MatMul LHS: k * h
        - MatMul RHS: w * k
        - MatMul Output: w * h
        """
        if not is_matmul:
            return w * h
        
        if is_lhs:
            return k * h
        if is_rhs:
            return w * k
        return w * h

    def calculate_step_latency(self, 
                               subgraph_ops: List[int], 
                               granularity: Tuple[int, int, int], 
                               traversal_order: Optional[List[int]] = None) -> Tuple[float, Optional[str]]:
        """
        Calculates the latency for a single subgraph step.
        Returns (latency, error_message).
        """
        w, h, k = granularity
        native_w, native_h = self.p.native_granularity
        
        # 1. Identify Inputs and Outputs with Types for MatMul
        input_info = [] # (tensor_id, is_lhs, is_rhs, is_matmul)
        output_info = [] # (tensor_id, is_matmul)
        produced_within = set()
        
        # Identify all outputs produced within this subgraph first
        for op_idx in subgraph_ops:
            for out in self.p.outputs[op_idx]:
                produced_within.add(out)

        for op_idx in subgraph_ops:
            is_mm = (self.p.op_types[op_idx] == "MatMul")
            for i, inp in enumerate(self.p.inputs[op_idx]):
                if inp not in produced_within:
                    is_lhs = is_mm and (i == 0)
                    is_rhs = is_mm and (i == 1)
                    input_info.append((inp, is_lhs, is_rhs, is_mm))
            for out in self.p.outputs[op_idx]:
                output_info.append((out, is_mm))
        
        # 2. Tiling Logic
        first_op_out = self.p.outputs[subgraph_ops[0]][0]
        total_w = self.p.widths[first_op_out]
        total_h = self.p.heights[first_op_out]
        
        num_tiles_w = (total_w + w - 1) // w
        num_tiles_h = (total_h + h - 1) // h
        num_tiles = num_tiles_w * num_tiles_h
        
        # 3. Latency Calculation
        step_latency = 0.0
        base_compute_cost = sum(self.p.base_costs[op_idx] for op_idx in subgraph_ops)
        
        # Working Set Check
        resident_size = self.memory_manager.get_current_occupancy()
        slice_in_total = sum(self.get_tensor_slice_size(tid, w, h, k, lhs, rhs, mm) for tid, lhs, rhs, mm in input_info)
        slice_out_total = sum(self.get_tensor_slice_size(tid, w, h, k, False, False, mm) for tid, mm in output_info)
        
        if resident_size + slice_in_total + slice_out_total > self.p.fast_memory_capacity:
            return 0.0, f"OOM: Working set ({resident_size + slice_in_total + slice_out_total}) exceeds SRAM capacity ({self.p.fast_memory_capacity})"

        for _ in range(num_tiles):
            # Memory In
            mem_in = sum(self.memory_manager.get_load_cost(tid, self.get_tensor_slice_size(tid, w, h, k, lhs, rhs, mm)) 
                         for tid, lhs, rhs, mm in input_info)
            
            # Memory Out
            mem_out = sum(self.get_tensor_slice_size(tid, w, h, k, False, False, mm) / self.p.slow_memory_bandwidth 
                          for tid, mm in output_info)
            
            tile_latency = max(base_compute_cost, mem_in + mem_out)
            step_latency += tile_latency
            
        return step_latency, None

    def validate_schedule(self, schedule_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates the schedule and calculates total latency.
        """
        subgraphs = schedule_data.get("subgraphs", [])
        granularities = schedule_data.get("granularities", [])
        tensors_to_retain = schedule_data.get("tensors_to_retain", [])
        
        if not (len(subgraphs) == len(granularities) == len(tensors_to_retain)):
            return False, "Schedule lists must have same length."

        self.memory_manager.resident_tensors = {} # Reset SRAM
        executed_ops = set()
        total_calculated_latency = 0.0

        for i in range(len(subgraphs)):
            ops = subgraphs[i]
            gran = tuple(granularities[i])
            retain = tensors_to_retain[i]
            
            # 1. Calculate Latency
            latency, err = self.calculate_step_latency(ops, gran)
            if err:
                return False, f"Step {i} error: {err}"
            
            total_calculated_latency += latency
            executed_ops.update(ops)
            
            # 2. Update SRAM state
            success, msg = self.memory_manager.update_residency(retain, self.p)
            if not success:
                return False, f"Step {i} residency error: {msg}"

        if len(executed_ops) < self.p.num_ops:
            missing = set(range(self.p.num_ops)) - executed_ops
            return False, f"Missing operations: {missing}"
        
        return True, f"Valid. Total Latency: {total_calculated_latency}"

def main():
    benchmark_dir = "benchmarks"
    if not os.path.exists(benchmark_dir):
        print(f"Directory {benchmark_dir} not found.")
        return

    benchmark_files = [f for f in os.listdir(benchmark_dir) if f.endswith(".json")]
    benchmark_files.sort()

    for bf in benchmark_files:
        path = os.path.join(benchmark_dir, bf)
        print(f"Parsing {path}...")
        problem = MLSysProblem.from_json(path)
        print(f"  Ops: {problem.num_ops}, Tensors: {problem.num_tensors}")
        print(f"  Fast Memory: {problem.fast_memory_capacity}, Bandwidth: {problem.slow_memory_bandwidth}")

if __name__ == "__main__":
    main()
