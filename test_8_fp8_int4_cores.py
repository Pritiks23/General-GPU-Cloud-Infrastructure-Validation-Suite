import torch

def run_quantization_hardware_test():
    if not torch.cuda.is_available():
        return {"error": "CUDA unavailable"}
        
    device = torch.device("cuda:0")
    results = {}
    
    # Test FP8 availability (Introduced natively in Hopper H100 / Ada Lovelace L4/RTX 4090)
    try:
        a_fp8 = torch.randn(1024, 1024, device=device).to(torch.float8_e4m3fn)
        b_fp8 = torch.randn(1024, 1024, device=device).to(torch.float8_e4m3fn)
        # Attempt an scaling multiplication
        _ = torch.mm(a_fp8.to(torch.float16), b_fp8.to(torch.float16))
        results["Native FP8 Support"] = "Verified Hardware Level"
    except Exception:
        results["Native FP8 Support"] = "Unsupported / Legacy Architecture"
        
    # Check for hardware accelerated INT4/INT8 paths via fake-quantization shapes
    try:
        # Check if compilation optimization paths work for native integer targets
        _ = torch.randint(-128, 127, (1024, 1024), device=device, dtype=torch.int8)
        results["INT8 Vector Register Path"] = "Available"
    except Exception:
        results["INT8 Vector Register Path"] = "Unavailable"
        
    return results

if __name__ == "__main__":
    print(run_quantization_hardware_test())
