import glob
from pathlib import Path

log_dir = Path(__file__).resolve().parents[1] / "logs"
files = glob.glob(str(log_dir / "bench_*.log"))

data = []
for f in files:
    filename = Path(f).name
    parts = filename.split("_")
    platform = parts[1].upper()
    mesh = int(parts[2].split(".")[0])
    
    total_time = 0.0
    first_call = 0.0
    avg_comp = 0.0
    total_comm = 0.0
    assembly = 0.0
    
    with open(f, "r") as file:
        for line in file:
            if "Total" in line and "steps, Time:" in line:
                try:
                    total_time = float(line.split("Time:")[1].split("s")[0].strip())
                except: pass
            elif "JAX First Call" in line:
                first_call = float(line.split("):")[-1].replace("s", "").strip())
            elif "JAX Pure Compute (Avg per call)" in line:
                avg_comp = float(line.split(":")[-1].replace("s", "").strip())
            elif "Total Communication (H2D + D2H)" in line:
                total_comm = float(line.split(":")[-1].replace("s", "").strip())
            elif "Global Assembly & Solve (Total Approx)" in line:
                assembly = float(line.split(":")[-1].replace("s", "").strip())
                
    data.append({
        "mesh": mesh,
        "platform": platform,
        "total_time": total_time,
        "first_call": first_call,
        "avg_comp": avg_comp,
        "total_comm": total_comm,
        "assembly": assembly
    })

data.sort(key=lambda x: (x["mesh"], x["platform"]))

print("| 网格规模 (Mesh) | 平台 | 总耗时<br>(Total Time) | **初始编译开销**<br>(1st Call/JIT) | **纯本构计算 (单步平均)**<br>(Pure Compute Avg) | **数据通信 (总)**<br>(Comm H2D+D2H) | **FEM装配与求解 (总)**<br>(Assembly & Solve) |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

for row in data:
    mesh_str = f"**{row['mesh']}^3**"
    plat_str = f"**{row['platform']}**" if row['platform'] == "GPU" else row['platform']
    print(f"| {mesh_str} | {plat_str} | {row['total_time']:.2f} s | {row['first_call']:.2f} s | **{row['avg_comp']:.4f} s** | {row['total_comm']:.4f} s | {row['assembly']:.2f} s |")
