"""Async Memcpy Overlap Validation - Measures async copy overlap with compute performance."""
import torch
import time


def run_async_memcpy_overlap_test(config=None):
    """
    Test asynchronous memory copy overlap with compute.
    
    Measures:
    - Host-to-Device (H2D) bandwidth
    - Device-to-Host (D2H) bandwidth
    - Compute-copy overlap efficiency
    - Total bidirectional bandwidth
    
    Args:
        config (dict): Configuration parameters (optional)
        
    Returns:
        dict: Performance metrics
    """
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device = torch.device("cuda:0")
        cpu_device = torch.device("cpu")
        
        # Data transfer parameters
        data_size = (4096, 4096)  # 64MB per matrix in FP32
        data_bytes = data_size[0] * data_size[1] * 4  # FP32 = 4 bytes
        iterations = 10
        
        # Warm-up
        for _ in range(3):
            test_host = torch.randn(1000, 1000, device=cpu_device)
            test_device = test_host.to(device)
            torch.cuda.synchronize()
        
        # Test 1: Sequential H2D copy then compute
        host_tensor = torch.randn(data_size, device=cpu_device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            device_tensor = host_tensor.to(device)
            result = torch.matmul(device_tensor, device_tensor)
            torch.cuda.synchronize()
        sequential_time = time.perf_counter() - start
        
        # Test 2: H2D Copy bandwidth
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            device_copy = host_tensor.to(device)
            torch.cuda.synchronize()
        h2d_time = (time.perf_counter() - start) / iterations
        h2d_bandwidth = (data_bytes / h2d_time) / 1e9  # GB/s
        
        # Test 3: D2H Copy bandwidth
        device_tensor = torch.randn(data_size, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            host_result = device_tensor.to(cpu_device)
            torch.cuda.synchronize()
        d2h_time = (time.perf_counter() - start) / iterations
        d2h_bandwidth = (data_bytes / d2h_time) / 1e9  # GB/s
        
        # Test 4: Async overlap simulation (using streams)
        stream1 = torch.cuda.Stream(device=device)
        stream2 = torch.cuda.Stream(device=device)
        
        # Create data for overlap test
        host_data1 = torch.randn(data_size, device=cpu_device)
        host_data2 = torch.randn(data_size, device=cpu_device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            # Copy in stream1
            with torch.cuda.stream(stream1):
                device_data1 = host_data1.to(device)
            
            # Compute in stream2 with different data
            with torch.cuda.stream(stream2):
                device_data2 = torch.randn(data_size, device=device)
                result = torch.matmul(device_data2, device_data2)
            
            torch.cuda.synchronize()
        
        async_time = (time.perf_counter() - start) / iterations
        
        # Calculate overlap efficiency
        overlap_efficiency = ((sequential_time / iterations - async_time) / (sequential_time / iterations)) * 100
        
        # Theoretical PCIe bandwidth
        pcie_gen = "3.0 (16GB/s)" if h2d_bandwidth < 20 else "4.0 (32GB/s)" if h2d_bandwidth < 40 else "5.0 (64GB/s)"
        
        result = {
            "H2D Copy Bandwidth": f"{h2d_bandwidth:.2f} GB/s",
            "D2H Copy Bandwidth": f"{d2h_bandwidth:.2f} GB/s",
            "Bidirectional BW": f"{(h2d_bandwidth + d2h_bandwidth):.2f} GB/s",
            "Estimated PCIe Gen": pcie_gen,
            "H2D Copy Time": f"{h2d_time * 1000:.4f}ms",
            "D2H Copy Time": f"{d2h_time * 1000:.4f}ms",
            "Sequential H2D+Compute": f"{(sequential_time / iterations) * 1000:.4f}ms",
            "Async Overlap Time": f"{async_time * 1000:.4f}ms",
            "Overlap Efficiency": f"{max(0, overlap_efficiency):.2f}%",
            "Data Transfer Size": f"{data_bytes / 1e6:.1f}MB"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    result = run_async_memcpy_overlap_test()
    print(json.dumps(result, indent=2))
