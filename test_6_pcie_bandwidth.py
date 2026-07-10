""" Measures PCIe bus throughput. If an infrastructure provider limits your PCIe lanes (e.g., giving you PCIe x4 instead of x16 via virtualization), moving model weights from system RAM to GPU VRAM will crawl, stalling your pipeline."""
import time
import torch

def run_pcie_test():
    if not torch.cuda.is_available():
        return {"error": "CUDA unavailable"}
        
    # Allocate 1 GB of data in Host RAM (pinned memory for maximum PCIe performance)
    size_bytes = 1024 * 1024 * 1024
    elements = size_bytes // 4
    
    host_tensor = torch.randn(elements, dtype=torch.float32).pin_memory()
    gpu_tensor = torch.empty(elements, device="cuda:0", dtype=torch.float32)
    torch.cuda.synchronize()
    
    # Measure Host-to-Device (H2D)
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        gpu_tensor.copy_(host_tensor)
    torch.cuda.synchronize()
    h2d_speed = (size_bytes / (1024**3)) * iters / (time.perf_counter() - t0)
    
    # Measure Device-to-Host (D2H)
    t1 = time.perf_counter()
    for _ in range(iters):
        host_tensor.copy_(gpu_tensor)
    torch.cuda.synchronize()
    d2h_speed = (size_bytes / (1024**3)) * iters / (time.perf_counter() - t1)
    
    return {
        "PCIe Host-to-Device (GB/s)": f"{h2d_speed:.2f}",
        "PCIe Device-to-Host (GB/s)": f"{d2h_speed:.2f}"
    }

if __name__ == "__main__":
    print(run_pcie_test())
