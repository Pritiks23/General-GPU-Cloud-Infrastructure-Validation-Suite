<img width="2802" height="1756" alt="image" src="https://github.com/user-attachments/assets/776653a2-8889-40d5-99d4-fbf62027ca3b" />

# General GPU Cloud Infrastructure Validation Suite

A standardized, bare-metal and virtualized hardware validation harness built in Python. This suite is designed for ML Platform, MLOps, and Core Infrastructure teams to benchmark, profile, and stress-test third-party GPU cloud providers (e.g., AWS, CoreWeave, Lambda Labs, RunPod).

Instead of evaluating high-level framework abstractions, this tool probes deep-layer hardware constraints to expose virtualization overhead, noisy neighbor multi-tenancy, interconnect limits, thermal throttling, and I/O bottlenecks.

## Architectural Testing Pillars

### 1. Compute & VRAM Integrity
* **Test 1: Compute FLOPs:** Executes heavy half-precision General Matrix Multiplication (GEMM) to verify Tensor Core utilization and clock speed validation.
* **Test 2: VRAM Memory Bandwidth:** Evaluates HBM/GDDR6 read/write efficiency to ensure data buses are running at full width.
* **Test 5: Thermal Endurance Burn-In:** Loops compute loads continuously over extended timeframes to catch heat decay and power limit degradation.

### 2. Fabric & Bus Topology
* **Test 3: Peer-to-Peer Interconnect:** Profiles multi-GPU communication matrices to measure local NVLink vs. bottlenecked PCIe Gen4/5 scaling profiles.
* **Test 6: PCIe Bus Bandwidth:** Measures Host-to-Device (H2D) and Device-to-Host (D2H) memory page transfers via pinned system RAM.
* **Test 7: Distributed NCCL Collective Simulation:** Simulates `torch.distributed` all-reduce clusters to analyze networking card fabric sync overhead.

### 3. Virtualization, Kernel, & I/O Isolation
* **Test 4: Data Stream Disk I/O:** Measures sequential read/write pipelines to isolate storage network (SAN) lag spikes that starve the GPU.
* **Test 8: Quantization Core Paths:** Verifies hardware register support for modern sub-byte numeric spaces (Native FP8/INT8 matrix layouts).
* **Test 9: Compute Jitter Variance:** Assesses latency percentiles ($P_{50}$ vs. $P_{99}$ tails) to capture multi-tenant noisy neighbor cloud interference.
* **Test 10: Context Allocation Integrity:** Validates CUDA Context garbage collection across streams to detect asynchronous virtualization engine memory leaks.

## Production-Ready Error Handling
All scripts use decoupled hardware checks and safe environment fallbacks. If run inside a CPU environment like GitHub Codespaces, tests gracefully log a `SKIPPED` warning matrix rather than throwing unhandled exceptions. Storage components utilize strict `try...finally` teardowns to completely prevent dead storage leaks on host volumes.

## Getting Started

### Prerequisites & Dependencies
```bash
pip install -r requirements.txt
```

### Execution Loop
```bash
python run_suite.py
```

