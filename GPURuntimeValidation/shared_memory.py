"""Shared Memory Validation - Measures memory bandwidth and L1/L2 cache efficiency."""
import torch
import time
import numpy as np


def run_shared_memory_test(config=None):
    """
    Test shared memory, L1/L2 cache, and memory coalescing.
    
    Measures:
    - Sequential vs random access patterns
    - Cache efficiency ratio
    - Memory coalescing performance
    - Effective memory bandwidth
    
    Args:
        config (dict): Configuration parameters (optional)
        
    Returns:
        dict: Performance metrics
    """
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device = torch.device("cuda:0")
        
        # Warm-up
        for _ in range(5):
            torch.randn(1000, 1000, device=device)
        torch.cuda.synchronize()
        
        # Test 1: Sequential vs Random Access (cache behavior)
        cache_test_size = 100000
        iterations = 100
        
        sequential_tensor = torch.randn(cache_test_size, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            result = sequential_tensor.sum()
        torch.cuda.synchronize()
        sequential_time = time.perf_counter() - start
        
        # Random access pattern
        random_indices = torch.randperm(cache_test_size, device=device)
        random_tensor = torch.randn(cache_test_size, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            result = random_tensor[random_indices].sum()
        torch.cuda.synchronize()
        random_time = time.perf_counter() - start
        
        cache_efficiency = (sequential_time / random_time) * 100 if random_time > 0 else 100
        
        # Test 2: Small working set (fits in shared memory)
        small_tensor = torch.randn(256, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations * 10):
            result = small_tensor.sum()
        torch.cuda.synchronize()
        small_set_time = (time.perf_counter() - start) / (iterations * 10)
        
        # Test 3: Memory coalescing efficiency
        coalesced_data = torch.randn(4096, 16, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            result = coalesced_data.sum(dim=1)
        torch.cuda.synchronize()
        coalesced_time = (time.perf_counter() - start) / 50
        
        # Test 4: Stride access patterns
        large_tensor = torch.randn(1000000, device=device)
        
        # Unit stride (coalesced)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            result = large_tensor[::1].sum()
        torch.cuda.synchronize()
        unit_stride_time = (time.perf_counter() - start) / 10
        
        # Stride-4 access
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            result = large_tensor[::4].sum()
        torch.cuda.synchronize()
        stride_4_time = (time.perf_counter() - start) / 10
        
        coalescing_efficiency = (unit_stride_time / stride_4_time) * 100 if stride_4_time > 0 else 100
        
        result = {
            "Sequential Access": f"{(sequential_time / iterations) * 1e6:.3f}µs",
            "Random Access": f"{(random_time / iterations) * 1e6:.3f}µs",
            "Cache Efficiency": f"{min(100, cache_efficiency):.1f}%",
            "Small Working Set": f"{small_set_time * 1e6:.3f}µs",
            "Coalesced Bandwidth": f"{coalesced_time * 1e6:.3f}µs/op",
            "Unit Stride Time": f"{unit_stride_time * 1e6:.3f}µs",
            "Stride-4 Time": f"{stride_4_time * 1e6:.3f}µs",
            "Coalescing Efficiency": f"{min(100, coalescing_efficiency):.1f}%"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    result = run_shared_memory_test()
    print(json.dumps(result, indent=2))
