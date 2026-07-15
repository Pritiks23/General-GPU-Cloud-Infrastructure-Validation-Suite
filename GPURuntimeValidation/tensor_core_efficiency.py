"""Tensor Core Efficiency Validation - Measures Tensor Core utilization and mixed-precision performance."""
import torch
import time


def run_tensor_core_efficiency_test(config=None):
    """
    Test Tensor Core efficiency with mixed precision operations.
    
    Measures:
    - FP32 baseline performance
    - Mixed precision (BF16/FP16) performance
    - Tensor Core speedup factor
    - Hardware utilization estimation
    
    Args:
        config (dict): Configuration parameters (optional)
        
    Returns:
        dict: Performance metrics
    """
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        
        # Check for Tensor Core support (Compute Capability >= 7.0)
        has_tensor_cores = props.major >= 7
        
        if not has_tensor_cores:
            return {
                "Tensor Core Support": "Not Available",
                "Compute Capability": f"{props.major}.{props.minor}",
                "Status": f"GPU lacks Tensor Cores (CC {props.major}.{props.minor} < 7.0)",
                "Hardware": props.name
            }
        
        # Warm-up
        for _ in range(3):
            a = torch.randn(2048, 2048, dtype=torch.float32, device=device)
            b = torch.randn(2048, 2048, dtype=torch.float32, device=device)
            torch.matmul(a, b)
        torch.cuda.synchronize()
        
        matrix_size = 8192
        iterations = 15
        
        # FP32 baseline (standard ALU path)
        a_fp32 = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        b_fp32 = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            c_fp32 = torch.matmul(a_fp32, b_fp32)
        torch.cuda.synchronize()
        fp32_time = (time.perf_counter() - start) / iterations
        
        # Mixed precision test (BF16 - Tensor Cores optimized)
        mixed_tflops = 0.0
        mixed_time = fp32_time
        mixed_precision_available = False
        
        try:
            a_bf16 = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16, device=device)
            b_bf16 = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16, device=device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                c_bf16 = torch.matmul(a_bf16, b_bf16)
            torch.cuda.synchronize()
            mixed_time = (time.perf_counter() - start) / iterations
            mixed_precision_available = True
        except:
            # Fallback to FP16 if BF16 not available
            try:
                a_fp16 = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
                b_fp16 = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    c_fp16 = torch.matmul(a_fp16, b_fp16)
                torch.cuda.synchronize()
                mixed_time = (time.perf_counter() - start) / iterations
                mixed_precision_available = True
            except:
                pass
        
        # Calculate TFLOPS (2 * N^3 operations for NxN matrix multiply)
        matrix_ops = 2 * (matrix_size ** 3)
        fp32_tflops = (matrix_ops / fp32_time) / 1e12
        mixed_tflops = (matrix_ops / mixed_time) / 1e12 if mixed_precision_available else fp32_tflops
        
        # Calculate speedup
        speedup = fp32_time / mixed_time if mixed_time > 0 else 1.0
        
        # Tensor Core utilization estimation
        # Peak TFLOPS for Tensor Cores varies by architecture
        peak_tflops = {
            7: 560,   # Volta (V100)
            8: 1457,  # Ampere (A100)
            9: 1457,  # Ada
        }
        peak = peak_tflops.get(props.major, 500)
        utilization = (mixed_tflops / peak) * 100 if mixed_precision_available else 0
        
        result = {
            "Tensor Core Support": "Yes" if has_tensor_cores else "No",
            "GPU Compute Capability": f"{props.major}.{props.minor}",
            "GPU Model": props.name,
            "FP32 GEMM Time": f"{fp32_time * 1000:.4f}ms",
            "FP32 Achieved TFLOPS": f"{fp32_tflops:.2f}",
            "Mixed Precision Time": f"{mixed_time * 1000:.4f}ms",
            "Mixed Precision TFLOPS": f"{mixed_tflops:.2f}",
            "Tensor Core Speedup": f"{speedup:.3f}x",
            "Peak TFLOPS": f"{peak:.0f}",
            "Est. Utilization": f"{min(100, utilization):.1f}%",
            "Mixed Precision Available": "Yes" if mixed_precision_available else "No"
        }
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import json
    result = run_tensor_core_efficiency_test()
    print(json.dumps(result, indent=2))
