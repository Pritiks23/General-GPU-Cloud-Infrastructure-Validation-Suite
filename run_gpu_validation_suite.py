"""Runs all GPU Runtime Validation components sequentially and builds a standardized terminal-ready infrastructure report table."""
import torch
import sys
import yaml
from tabulate import tabulate

# Import GPU Runtime Validation Tests
from cuda_streams import run_cuda_streams_test
from cuda_graphs import run_cuda_graphs_test
from occupancy import run_occupancy_test
from kernel_launch import run_kernel_launch_test
from shared_memory import run_shared_memory_test
from warp_divergence import run_warp_divergence_test
from tensor_core_efficiency import run_tensor_core_efficiency_test
from async_memcpy_overlap import run_async_memcpy_overlap_test
from unified_memory_migration import run_unified_memory_migration_test
from cooperative_groups import run_cooperative_groups_test


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"Warning: Error parsing {config_path}: {e}")
        return {}


def main():
    """Execute GPU Runtime Validation suite."""
    # Load configuration
    config = load_config()
    
    print("=" * 100)
    print("         GPU RUNTIME VALIDATION SUITE - CUDA KERNEL & MEMORY ASSESSMENT")
    print("=" * 100)
    
    # Evaluate CUDA Environment
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        
        # Get detailed GPU properties
        props = torch.cuda.get_device_properties(0)
        
        print(f"Target Environment : Enterprise GPU Cluster Node")
        print(f"Hardware Resources : {gpu_count}x {gpu_name}")
        print(f"CUDA Version       : {cuda_version}")
        print(f"Compute Capability : {props.major}.{props.minor}")
        print(f"SM Count           : {props.multi_processor_count}")
        print(f"Max Threads/SM     : {props.max_threads_per_multi_processor}")
        
        # Get memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM         : {total_memory:.2f}GB")
    else:
        print(f"Target Environment : CPU Emulation Loop / Codespace Target")
        print(f"Hardware Resources : Shared Virtual Host Core - NO ACTIVE GPU")
        print("\nWARNING: GPU not available. Tests will skip GPU-specific components.")
    
    print("-" * 100)
    print("Executing GPU Runtime Validation Tests...\n")
    
    report_data = []
    
    # Helper to append dict items to global log sheet safely
    def append_metrics(category, result_dict):
        """Safely append test results to report data."""
        if "error" in result_dict:
            report_data.append([category, "GPU Check", "SKIPPED (Requires GPU Hardware)"])
            return
        for k, v in result_dict.items():
            report_data.append([f" {category}", k, v])
    
    # Execute GPU Runtime Validation Suite Tests (10 components)
    test_suites = [
        ("1. CUDA Streams", run_cuda_streams_test),
        ("2. CUDA Graphs", run_cuda_graphs_test),
        ("3. Occupancy", run_occupancy_test),
        ("4. Kernel Launch", run_kernel_launch_test),
        ("5. Shared Memory", run_shared_memory_test),
        ("6. Warp Divergence", run_warp_divergence_test),
        ("7. Tensor Core Efficiency", run_tensor_core_efficiency_test),
        ("8. Async Memcpy Overlap", run_async_memcpy_overlap_test),
        ("9. Unified Memory Migration", run_unified_memory_migration_test),
        ("10. Cooperative Groups", run_cooperative_groups_test),
    ]
    
    total_tests = len(test_suites)
    
    for idx, (test_name, test_func) in enumerate(test_suites, 1):
        print(f"  [{idx}/{total_tests}] {test_name}...", end=" ", flush=True)
        try:
            result = test_func(config)
            append_metrics(test_name, result)
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {str(e)[:50]})")
            append_metrics(test_name, {"error": str(e)})
    
    print("\n" + "-" * 100)
    
    # Print formatted matrix output
    if report_data:
        print("\nGPU RUNTIME VALIDATION RESULTS:\n")
        print(tabulate(
            report_data,
            headers=["GPU Subsystem", "Performance Metric", "Measured Value"],
            tablefmt="grid",
            maxcolwidths=[30, 35, 30]
        ))
    else:
        print("No GPU metrics collected. GPU hardware may not be available.")
    
    print("\n" + "=" * 100)
    print("GPU Runtime Validation Suite Complete.")
    print("=" * 100)
    
    # Exit status
    return 0 if cuda_available else 1


if __name__ == "__main__":
    sys.exit(main())
