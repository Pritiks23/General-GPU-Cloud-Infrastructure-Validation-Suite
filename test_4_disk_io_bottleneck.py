import os
import time
import yaml

def run_storage_test():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    file_size_gb = config["storage_test_file_gb"]
    file_name = "test_infra_io.data"
    data = os.urandom(1024 * 1024) # 1MB chunk
    
    # Test Sequential Write Speed
    start_write = time.perf_counter()
    with open(file_name, "wb") as f:
        for _ in range(file_size_gb * 1024):
            f.write(data)
    end_write = time.perf_counter()
    write_time = end_write - start_write
    write_speed = file_size_gb / write_time
    
    # Test Sequential Read Speed
    start_read = time.perf_counter()
    with open(file_name, "rb") as f:
        while f.read(1024 * 1024):
            pass
    end_read = time.perf_counter()
    read_time = end_read - start_read
    read_speed = file_size_gb / read_time
    
    # Cleanup
    if os.path.exists(file_name):
        os.remove(file_name)
        
    return {
        "Test File Size": f"{file_size_gb} GB",
        "Disk Write Speed (MB/s)": f"{write_speed * 1024:.2f}",
        "Disk Read Speed (MB/s)": f"{read_speed * 1024:.2f}"
    }

if __name__ == "__main__":
    print(run_storage_test())
