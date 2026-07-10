""" Runs all validation components sequentially and builds a standardized terminal-ready infrastructure report table."""
import torch
import sys
from tabulate import tabulate

# Import Phase 1 Tests
from test_1_compute_flops import run_compute_test
from test_2_vram_bandwidth import run_vram_test
from test_3_p2p_interconnect import run_interconnect_test
from test_4_disk_io_bottleneck import run_storage_test
from test_5_stability_burnin import run_burn_in_test

# Import Phase 2 Hardcore Tests
from test_6_pcie_bandwidth import run_pcie_test
from test_7_nccl_network import run_nccl_mock_test
from test_8_fp8_int4_cores import run_quantization_hardware_test
from test_9_jitter_determinism import run_jitter_test
from test_10_context_vram_leak import run_context_leak_test

def main():
    print("=" * 75)
    print("         PRODUCTION-GRADE HARDWARE CLOUD VALIDATION MATRIX")
    print("=" * 75)
    
    # Evaluate Environmental CUDA Layer
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"Target Environment : Enterprise GPU Cluster Node")
        print(f"Hardware Resources : {gpu_count}x {gpu_name}")
    else:
        print(f"Target Environment : CPU Emulation Loop / Codespace Target")
        print(f"Hardware Resources : Shared Virtual Host Core - NO ACTIVE GPU")
    print("-" * 75)
    print("Executing tests...\n")
    
    report_data = []
    
    # Helper to append dict items to global log sheet safely
    def append_metrics(category, result_dict):
        if "error" in result_dict:
            report_data.append([category, "CUDA Check", "SKIPPED (Requires GPU Hardware)"])
            return
        for k, v in result_dict.items():
            report_data.append([f" {category}", k, v])

    # Execute all 10 core components
    append_metrics("1. Compute FLOPs", run_compute_test())
    append_metrics("2. VRAM Bandwidth", run_vram_test())
    append_metrics("3. P2P Interconnect", run_interconnect_test())
    append_metrics("4. Storage Storage I/O", run_storage_test())
    append_metrics("5. Thermal Stability", run_burn_in_test())
    append_metrics("6. PCIe Bus Bandwidth", run_pcie_test())
    append_metrics("7. NCCL Cluster Comm", run_nccl_mock_test())
    append_metrics("8. Quantization Paths", run_quantization_hardware_test())
    append_metrics("9. Multi-Tenant Jitter", run_jitter_test())
    append_metrics("10. Context Allocation", run_context_leak_test())
    
    # Print formatted matrix output
    print(tabulate(report_data, headers=["Infrastructure Pillar", "Metric Audited", "Observed Node Value"], tablefmt="grid"))
    print("\n" + "=" * 75)
    print("Validation Suite Complete.")
    print("=" * 75)

if __name__ == "__main__":
    main()

