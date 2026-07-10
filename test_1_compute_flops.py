import time
import torch
import yaml

def run_compute_test():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.device("cuda:0")
    N = config["matrix_size"]
    
    # Use float16 to trigger Tensor Cores
    a = torch.randn(N, N, device=device, dtype=torch.float16)
    b = torch.randn(N, N, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(10):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Benchmark loop
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / iters
    # FLOPs formula for matrix mult: 2 * N^3
    tflops = (2 * (N ** 3)) / avg_time / (10 ** 12)
    
    return {
        "Matrix Size": f"{N}x{N}",
        "Avg GEMM Time (ms)": f"{avg_time * 1000:.2f}",
        "Achieved TFLOPs": f"{tflops:.2f}"
    }

if __name__ == "__main__":
    print(run_compute_test())
