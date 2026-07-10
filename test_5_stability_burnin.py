import time
import torch
import yaml

def run_burn_in_test():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
        
    device = torch.device("cuda:0")
    N = config["matrix_size"]
    duration = config["burn_in_duration_sec"]
    
    a = torch.randn(N, N, device=device, dtype=torch.float16)
    b = torch.randn(N, N, device=device, dtype=torch.float16)
    
    print(f"-> Initiating loop for {duration} seconds...")
    
    start_time = time.perf_counter()
    loop_count = 0
    
    # Capture performance at start
    t0 = time.perf_counter()
    torch.matmul(a, b)
    torch.cuda.synchronize()
    initial_latency = time.perf_counter() - t0
    
    while (time.perf_counter() - start_time) < duration:
        torch.matmul(a, b)
        loop_count += 1
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Capture performance at end to detect thermal throttling degradation
    t1 = time.perf_counter()
    torch.matmul(a, b)
    torch.cuda.synchronize()
    final_latency = time.perf_counter() - t1
    
    performance_drop = ((final_latency - initial_latency) / initial_latency) * 100
    
    return {
        "Burn-In Duration": f"{end_time - start_time:.2f}s",
        "Total Core Operations": loop_count,
        "Thermal Performance Drop": f"{performance_drop:.2f}% (Lower is better)"
    }

if __name__ == "__main__":
    print(run_burn_in_test())
