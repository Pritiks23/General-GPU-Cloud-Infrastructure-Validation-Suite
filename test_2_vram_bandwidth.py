import time
import torch
import yaml

def run_vram_test():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
        
    device = torch.device("cuda:0")
    # Allocate a large tensor (~2GB at 16384x16384 float32)
    N = config["matrix_size"]
    
    start_alloc = time.perf_counter()
    x = torch.randn(N, N, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    
    # Measure memory copy speed (Read + Write)
    iters = 50
    start_copy = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()
    end_copy = time.perf_counter()
    
    # Size in Gigabytes
    tensor_gb = (x.element_size() * x.nelement()) / (1024 ** 3)
    avg_time = (end_copy - start_copy) / iters
    
    # Read + Write means 2x data moved per operation
    bandwidth = (tensor_gb * 2) / avg_time 
    
    return {
        "Allocated VRAM (GB)": f"{tensor_gb:.2f}",
        "Memory Bandwidth (GB/s)": f"{bandwidth:.2f}"
    }

if __name__ == "__main__":
    print(run_vram_test())
