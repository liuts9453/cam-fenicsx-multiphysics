import os
import glob
import matplotlib.pyplot as plt
import numpy as np

results = {}
for file in sorted(glob.glob("result/bench_logs/bench_*.log")):
    platform = file.split("_")[2]
    mesh = int(file.split("_")[3].split(".")[0])
    
    avg_comp = 0.0
    total_comm = 0.0
    assembly_solve_total = 0.0
    steps = 1
    
    with open(file, "r") as f:
        for line in f:
            if "Total" in line and "steps, Time:" in line:
                steps = int(line.split("Total")[1].split("steps")[0].strip())
            if "JAX Pure Compute (Avg per call)" in line:
                avg_comp = float(line.split(":")[-1].replace("s", "").strip())
            if "Total Communication (H2D + D2H)" in line:
                total_comm = float(line.split(":")[-1].replace("s", "").strip())
            if "Global Assembly & Solve (Total Approx)" in line:
                assembly_solve_total = float(line.split(":")[-1].replace("s", "").strip())
                
    avg_assembly = assembly_solve_total / steps if steps > 0 else 0
    avg_comm = total_comm / max(1, (steps - 1)) if platform == "gpu" else 0.0
    
    if mesh not in results:
        results[mesh] = {}
    results[mesh][platform] = {
        "assembly": avg_assembly,
        "constitutive": avg_comp,
        "comm": avg_comm
    }

target_meshes = [10, 30] 

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
colors = ['#1f77b4', '#ff7f0e', '#d62728']
labels = ['FEM Framework Overhead (Assembly & Solve)', 'Constitutive Update (Local Newton & expm)', 'Data Transfer (Host-Device)']

for idx, mesh in enumerate(target_meshes):
    if mesh not in results or 'cpu' not in results[mesh] or 'gpu' not in results[mesh]:
        axes[idx].text(0.5, 0.5, f"Data for {mesh}^3 missing", ha='center')
        continue
        
    ax = axes[idx]
    platforms = ['CPU', 'GPU']
    
    assembly = [results[mesh]['cpu']['assembly'], results[mesh]['gpu']['assembly']]
    constitutive = [results[mesh]['cpu']['constitutive'], results[mesh]['gpu']['constitutive']]
    comm = [results[mesh]['cpu']['comm'], results[mesh]['gpu']['comm']]
    
    x = np.arange(len(platforms))
    width = 0.5
    
    p1 = ax.bar(x, assembly, width, label=labels[0], color=colors[0], edgecolor='black')
    p2 = ax.bar(x, constitutive, width, bottom=assembly, label=labels[1], color=colors[1], edgecolor='black', hatch='//')
    p3 = ax.bar(x, comm, width, bottom=np.array(assembly)+np.array(constitutive), label=labels[2], color=colors[2], edgecolor='black', hatch='\\\\')
    
    ax.set_xticks(x)
    ax.set_xticklabels(platforms)
    ax.set_title(f"Mesh: ${mesh} \\times {mesh} \\times {mesh}$")
    ax.set_ylabel("Average Time per Step (s)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i in range(2):
        total_time = assembly[i] + constitutive[i] + comm[i]
        ax.text(x[i], total_time + total_time*0.02, f"{total_time:.2f}s", ha='center', fontweight='bold')

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False)

plt.tight_layout()
plt.subplots_adjust(top=0.82)
plt.savefig("bottleneck_analysis.pdf", dpi=300, bbox_inches='tight')
