import os
import sys
import subprocess
import time

mesh_sizes = [3,5,7,10,20,30]
platforms = ["cpu", "gpu"]

os.makedirs("result/bench_logs", exist_ok=True)

for size in mesh_sizes:
    for platform in platforms:
        print(f"===========================================================")
        print(f"Running Benchmark: Mesh = {size}x{size}x{size}, Platform = {platform}")
        print(f"===========================================================")
        
        log_file = f"result/bench_logs/bench_{platform}_{size}.log"
        
        # Run the command
        cmd = ["conda", "run", "-n", "dof10", "python", "Felder_ele_bench.py", "--mesh_size", str(size), "--platform", platform, "--steps", "50"]
        
        start_time = time.time()
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, 
                stdout=f, 
                stderr=subprocess.STDOUT, 
                env=dict(os.environ, JAX_PLATFORM_NAME=platform)
            )
            process.wait()
            
        end_time = time.time()
        
        print(f"Completed in {end_time - start_time:.2f} seconds.")
        print(f"Log saved to {log_file}\n")
