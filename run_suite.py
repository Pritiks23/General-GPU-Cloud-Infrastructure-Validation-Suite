""" Runs all validation components sequentially and builds a standardized terminal-ready infrastructure report table."""
import torch
from tabulate import tabulate

from test_1_compute_flops import run_compute_test
from test_2_vram_bandwidth import run_vram_test
from test_3_p2p_interconnect import run_interconnect_test
from test_4_disk_io_bottleneck import run_storage_test
from test_5_stability_burnin import run_burn_in_test

def main():
    print("=" * 60)
    print("HARDWARE INFRASTRUCTURE AUDIT AND VALIDATION SUITE")
    print("=" * 60)
    
    # Gather System Info
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    print(f"Target Compute Device: {gpu_name}")
    print("Running evaluation suite...\n")
    
    # Run tests
    compute_res = run_compute_test()
    vram_res = run_vram_test()
    p2p_res = run_interconnect_test()
    storage_res = run_storage_test()
    burn_res = run_burn_in_test()
    
    # Collate results into a clean table structure
    report_data = []
    
    for k, v in compute_res.items(): report_data.append(["[1] Compute FLOPs", k, v])
    for k, v in vram_res.items(): report_data.append(["[2] VRAM Speed", k, v])
    for k, v in p2p_res.items(): report_data.append(["[3] Interconnect", k, v])
    for k, v in storage_res.items(): report_data.append(["[4] Storage I/O", k, v])
    for k, v in burn_res.items(): report_data.append(["[5] Stability & Thermal", k, v])
    
    print(tabulate(report_data, headers=["Category", "Metric Analyzed", "Observed Value"], tablefmt="grid"))

if __name__ == "__main__":
    main()
