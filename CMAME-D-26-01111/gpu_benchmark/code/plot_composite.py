import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Data definition based on the latest 50-step logs
meshes = ['1$^3$', '3$^3$', '5$^3$', '7$^3$', '10$^3$', '20$^3$', '30$^3$']
cpu_tot = [15.01, 25.36, 49.83, 107.42, 301.98, 3214.96, 12207.38]
gpu_tot = [32.01, 32.59, 43.67, 62.00, 105.03, 748.52, 3258.90]
speedup = [c / g for c, g in zip(cpu_tot, gpu_tot)]

# Breakdown data (Calculated as Total - First - Comm - Remaining = Constitutive)
# 5^3 CPU
b1_cpu = [6.40, 16.542, 0.098, 26.79]
# 5^3 GPU
b1_gpu = [13.36, 5.026, 0.154, 25.13]

# 30^3 CPU
b2_cpu = [41.66, 5334.17, 21.92, 6809.63]
# 30^3 GPU
b2_gpu = [24.16, 1315.65, 6.27, 1912.82]

cat_labels = ['First-call / JIT', 'Constitutive compute', 'Communication', 'Remaining framework-level runtime']
colors = ['#b3b3b3', '#ff7f0e', '#d62728', '#1f77b4']

# Figure setup
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.35)

# --- Panel (a) ---
ax1 = fig.add_subplot(gs[0])
x = np.arange(len(meshes))
l1 = ax1.plot(x, cpu_tot, marker='o', color='#1f77b4', linewidth=2, label='CPU total time')
l2 = ax1.plot(x, gpu_tot, marker='s', color='#d62728', linewidth=2, label='GPU total time')
ax1.set_yscale('log')
ax1.set_ylabel('Total wall-clock time (s)')
ax1.set_xticks(x)
ax1.set_xticklabels(meshes)
ax1.set_xlabel('Mesh size')

ax2 = ax1.twinx()
l3 = ax2.plot(x, speedup, marker='^', color='#2ca02c', linestyle='--', linewidth=2, label='End-to-end speedup')
ax2.set_ylabel('End-to-end speedup (CPU / GPU)')
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5)
ax2.set_ylim(0, 5.5)

# Annotation for crossover
ax2.annotate('Crossover\n(between 3$^3$ and 5$^3$)', 
             xy=(1.5, 1.0), xytext=(1.5, 2.0),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5), 
             ha='center', va='bottom')

# Legend for Panel (a)
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.set_title('(a) Runtime scaling and end-to-end speedup')

# --- Panel (b) ---
def plot_stacked(ax, cpu_data, gpu_data, title):
    x_bar = np.arange(2)
    bottom_cpu = 0
    bottom_gpu = 0
    bars = []
    for i in range(4):
        b = ax.bar(x_bar, [cpu_data[i], gpu_data[i]], 
                   bottom=[bottom_cpu, bottom_gpu], 
                   width=0.5, color=colors[i], edgecolor='black', alpha=0.85)
        bottom_cpu += cpu_data[i]
        bottom_gpu += gpu_data[i]
        bars.append(b)
    ax.set_xticks(x_bar)
    ax.set_xticklabels(['CPU', 'GPU'])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Runtime contribution (s)')
    return bars

ax3 = fig.add_subplot(gs[1])
bars = plot_stacked(ax3, b1_cpu, b1_gpu, 'Mesh: 5$^3$')

ax4 = fig.add_subplot(gs[2])
plot_stacked(ax4, b2_cpu, b2_gpu, 'Mesh: 30$^3$')

# Add a text label over the subplots for Panel b
fig.text(0.68, 0.92, '(b) Runtime breakdown for representative meshes', ha='center', va='center', fontsize=12, fontweight='bold')

# Shared Legend for Panel (b)
fig.legend(reversed(bars), reversed(cat_labels), loc='center right', bbox_to_anchor=(1.12, 0.5), frameon=False)

fig.suptitle('GPU benchmarking: scaling, speedup, and runtime decomposition', fontsize=14, y=1.02)

plt.savefig('composite_gpu_benchmark.svg', bbox_inches='tight')
print("Saved composite_gpu_benchmark.svg")
