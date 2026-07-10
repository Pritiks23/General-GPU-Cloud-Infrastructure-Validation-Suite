""" Simulates torch.distributed NCCL (NVIDIA Collective Communications Library) collectives. If you are training across multiple nodes, the network card (InfiniBand/RoCE) speed is everything. This tests backend synchronization speed. """
import time
import torch
import torch.distributed as dist
import os

def run_nccl_mock_test():
    if not torch.cuda.is_available():
        return {"error": "CUDA unavailable"}
        
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        return {"Status": "Skipped (Requires multi-GPU local environment or multi-node cluster)"}
        
    # Setting up an in-memory single-node dummy distribution to test NCCL overhead
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    
    try:
        # Check if already initialized to prevent crashing orchestrator
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
            
        tensor = torch.randn(256 * 1024 * 1024, device="cuda:0") # 1GB
        torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        # Heavy collective operation: sum up tensor metrics across world (mocked single-node)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        
        return {
            "NCCL Init Backend": "Success",
            "1GB All-Reduce Overhead (ms)": f"{dt * 1000:.2f}"
        }
    except Exception as e:
        return {"NCCL Error": str(e)}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    print(run_nccl_mock_test())
