import time
import numpy as np
import torch

def run_jitter_test():
    if not torch.cuda.is_available():
        return {"error": "CUDA unavailable"}
        
    device = torch.device("cuda:0")
    a = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.float16)
    
    latencies = []
    # Profile 50 isolated runs
    for _ in range(50):
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        
    latencies_ms = np.array(latencies) * 1000
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)
    jitter = p99 - p50 # The gap indicates multi-tenant noisy neighbor interference
    
    return {
        "Median Execution Time (ms)": f"{p50:.2f}",
        "P99 Tail Latency (ms)": f"{p99:.2f}",
        "Compute Jitter Variance": f"{jitter:.2f}ms " + ("(HIGH - Noisy Tenant Detected)" if jitter > 5 else "(EXCELLENT - Isolated)")
    }

if __name__ == "__main__":
    print(run_jitter_test())
