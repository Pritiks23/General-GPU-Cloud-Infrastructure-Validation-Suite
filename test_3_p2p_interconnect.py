""" Validates the interconnect fabric (NVLink vs. PCIe Gen4/5) between multiple local GPUs. Vital for multi-node/distributed training infrastructure. """
import time
import torch

def run_interconnect_test():
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return {"Status": "Skipped (Requires 2+ GPUs)"}
        
    # 1 GB Payload
    size_bytes = 1024 * 1024 * 1024 
    elements = size_bytes // 4 # float32 = 4 bytes
    
    # Initialize tensors on GPU 0 and GPU 1
    tensor_src = torch.randn(elements, device="cuda:0", dtype=torch.float32)
    tensor_dst = torch.empty(elements, device="cuda:1", dtype=torch.float32)
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        tensor_dst.copy_(tensor_src)
    torch.cuda.synchronize()
    
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        tensor_dst.copy_(tensor_src)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / iters
    gb_per_sec = (size_bytes / (1024 ** 3)) / avg_time
    
    return {
        "GPUs Detected": num_gpus,
        "P2P Link Tested": "GPU 0 -> GPU 1",
        "Interconnect Speed (GB/s)": f"{gb_per_sec:.2f}"
    }

if __name__ == "__main__":
    print(run_interconnect_test())
