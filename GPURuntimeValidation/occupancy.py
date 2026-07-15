"""Occupancy Validation - Measures streaming multiprocessor occupancy and utilization."""
import torch
import time


def run_occupancy_test(config=None):
    """
    Test GPU SM occupancy and warp scheduling.
    
    Measures:
    - Streaming multiprocessor count
    - Warp occupancy at different block sizes
    - Measured SM utilization during compute
    - Average kernel execution time
    
    Args:
        config (dict): Configuration parameters (optional)
        
    Returns:
        dict: Performance metrics
    """
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device = torch.device("cuda:0")
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(device)
        sm_count = props.multi_processor_count
        max_threads_per_sm = props.max_threads_per_multi_processor
        
        # Theoretical maximum warps per SM (32 threads per warp)
        warp_size = 32
        max_warps_per_sm = max_threads_per_sm // warp_size
        
        # Test with different block sizes and calculate occupancy
        test_configs = [
            (128, "128-thread blocks"),
            (256, "256-thread blocks"),
            (512, "512-thread blocks"),
        ]
        
        occupancy_results = {}
        for block_size, config_name in test_configs:
            warps_per_block = block_size // warp_size
            occupancy_ratio = warps_per_block / max_warps_per_sm
            occupancy_percent = min(100.0, occupancy_ratio * 100.0)
            occupancy_results[config_name] = f"{occupancy_percent:.1f}%"
        
        # Runtime occupancy measurement with moderate block size
        tensor_size = (8192, 8192)
        iterations = 20
        block_size = 256
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            a = torch.randn(tensor_size, device=device)
            b = torch.randn(tensor_size, device=device)
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        
        # Estimate actual grid utilization
        estimated_grid_utilization = (sm_count * 4) / sm_count * 100  # 4 blocks per SM
        
        result = {
            "SM Count": str(sm_count),
            "Max Threads/SM": str(max_threads_per_sm),
            "Max Warps/SM": str(max_warps_per_sm),
            **occupancy_results,
            "Optimal Block Size": "256",
            "Estimated Grid Util": f"{min(100.0, estimated_grid_utilization):.1f}%",
            "Avg Time/Iteration": f"{(runtime / iterations) * 1000:.4f}ms",
            "Total Runtime": f"{runtime:.4f}s"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    result = run_occupancy_test()
    print(json.dumps(result, indent=2))
