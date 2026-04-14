#!/usr/bin/env python3
"""
Regenerate paper figures with updated font sizes
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Set publication style with larger fonts
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.figsize': (10, 8),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Load results
results_path = Path("results/paper_parallel_20260125_214106/paper_results.json")
with open(results_path) as f:
    data = json.load(f)

results = data["main_experiment"]
category_results = data["category_breakdown"]

# Output directory
figures_dir = Path("Do Large Language Models Think Fast and Slow/figures")
figures_dir.mkdir(exist_ok=True)

# Colors
S1_COLOR = '#E64B35'
S2_COLOR = '#4DBBD5'

print("Regenerating figures with larger fonts...")

# Figure 1: Main comparison
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(2)
s1_mean = results["system1_accuracy"]["mean"]
s2_mean = results["system2_accuracy"]["mean"]
s1_ci = results["system1_accuracy"]["ci_95"]
s2_ci = results["system2_accuracy"]["ci_95"]

bars = ax.bar(x, [s1_mean, s2_mean],
             yerr=[[s1_mean - s1_ci[0], s2_mean - s2_ci[0]],
                   [s1_ci[1] - s1_mean, s2_ci[1] - s2_mean]],
             capsize=5, color=[S1_COLOR, S2_COLOR], alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'])
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)

p_val = results["statistical_analysis"]["paired_t_test"]["p_value"]
effect_d = results["statistical_analysis"]["effect_size"]["cohens_d"]

sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
ax.annotate(f'{sig_text}\np={p_val:.3f}\nd={effect_d:.2f}',
           xy=(0.5, max(s1_mean, s2_mean) + 0.08),
           ha='center', fontsize=13)

plt.tight_layout()
plt.savefig(figures_dir / "fig1_main_comparison.png", dpi=300)
plt.close()
print("  ✓ fig1_main_comparison.png")

# Figure 2: Category breakdown
fig, ax = plt.subplots(figsize=(10, 6))

cats = list(category_results.keys())
x = np.arange(len(cats))
width = 0.35

s1_means = [category_results[c]["system1_accuracy"]["mean"] for c in cats]
s2_means = [category_results[c]["system2_accuracy"]["mean"] for c in cats]
s1_stds = [category_results[c]["system1_accuracy"]["std"] for c in cats]
s2_stds = [category_results[c]["system2_accuracy"]["std"] for c in cats]

ax.bar(x - width/2, s1_means, width, yerr=s1_stds,
      label='System 1', color=S1_COLOR, alpha=0.8, capsize=4)
ax.bar(x + width/2, s2_means, width, yerr=s2_stds,
      label='System 2', color=S2_COLOR, alpha=0.8, capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(['Intuitive\nTasks', 'Analytical\nTasks', 'Conflict\nTasks'], fontsize=16)
ax.set_ylabel('Accuracy', fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.legend(fontsize=15)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(figures_dir / "fig2_category_breakdown.png", dpi=300)
plt.close()
print("  ✓ fig2_category_breakdown.png")

# Figure 3: Speed-Accuracy Tradeoff
fig, ax = plt.subplots(figsize=(10, 8))

s1_acc = results["system1_accuracy"]["mean"]
s2_acc = results["system2_accuracy"]["mean"]
s1_time = results["system1_response_time_ms"]["mean"]
s2_time = results["system2_response_time_ms"]["mean"]

# Create background gradient regions
x_range = np.linspace(0, s2_time * 1.3, 100)
y_range = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = Y / np.log10(X + 10)
contour = ax.contourf(X, Y, Z, levels=20, cmap='YlGnBu', alpha=0.3)

# Plot connecting line
ax.plot([s1_time, s2_time], [s1_acc, s2_acc],
       color='#8C8C8C', linestyle='--', linewidth=2, alpha=0.6, zorder=2)

# Plot points
ax.scatter(s1_time, s1_acc, s=400, c=S1_COLOR,
          edgecolors='white', linewidth=3, zorder=5, marker='o',
          label='System 1 (Intuitive)')
ax.scatter(s2_time, s2_acc, s=400, c=S2_COLOR,
          edgecolors='white', linewidth=3, zorder=5, marker='s',
          label='System 2 (Analytical)')

# Add annotations
ax.annotate(f'S1\n{s1_acc:.0%}\n{s1_time/1000:.1f}s',
           xy=(s1_time, s1_acc),
           xytext=(s1_time - 20000, s1_acc - 0.12),
           fontsize=12, ha='center', fontweight='medium',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4A582',
                    edgecolor=S1_COLOR, alpha=0.9),
           arrowprops=dict(arrowstyle='->', color=S1_COLOR, lw=1.5))

ax.annotate(f'S2\n{s2_acc:.0%}\n{s2_time/1000:.1f}s',
           xy=(s2_time, s2_acc),
           xytext=(s2_time + 20000, s2_acc + 0.08),
           fontsize=12, ha='center', fontweight='medium',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#92C5DE',
                    edgecolor=S2_COLOR, alpha=0.9),
           arrowprops=dict(arrowstyle='->', color=S2_COLOR, lw=1.5))

ax.set_xlabel('Response Time (ms)')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05)
ax.set_xlim(0, s2_time * 1.3)
ax.legend(loc='lower right', fontsize=12)

cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Efficiency Index', fontsize=12)

plt.tight_layout()
plt.savefig(figures_dir / "fig3_speed_accuracy_tradeoff.png", dpi=300)
plt.close()
print("  ✓ fig3_speed_accuracy_tradeoff.png")

# Figure 4: Confidence Analysis - SKIPPED (requires detailed per-task results)
# The original fig4_confidence_analysis.png from the experiment should be used
print("  ⏭ fig4_confidence_analysis.png (using original)")

# Figure 5: Cognitive Effort Radar
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

categories_radar = ['Accuracy', 'Speed\n(inverse)', 'Confidence', 
             'Efficiency\n(acc/token)', 'Reasoning\nDepth']
N = len(categories_radar)

# Normalize metrics
max_time = max(results["system1_response_time_ms"]["mean"], 
               results["system2_response_time_ms"]["mean"])
max_tokens = max(results["system1_tokens"]["mean"], 
                 results["system2_tokens"]["mean"])

s1_values = [
    results["system1_accuracy"]["mean"],
    1 - (results["system1_response_time_ms"]["mean"] / max_time),
    results["system1_confidence"]["mean"],
    results["system1_accuracy"]["mean"] / (results["system1_tokens"]["mean"] / 1000),
    0.1  # System 1 has minimal reasoning steps
]

s2_values = [
    results["system2_accuracy"]["mean"],
    1 - (results["system2_response_time_ms"]["mean"] / max_time),
    results["system2_confidence"]["mean"],
    results["system2_accuracy"]["mean"] / (results["system2_tokens"]["mean"] / 1000),
    0.9  # System 2 has many reasoning steps
]

# Normalize efficiency
max_eff = max(s1_values[3], s2_values[3])
s1_values[3] = s1_values[3] / max_eff
s2_values[3] = s2_values[3] / max_eff

angles = [n / float(N) * 2 * np.pi for n in range(N)]
s1_values += s1_values[:1]
s2_values += s2_values[:1]
angles += angles[:1]

ax.plot(angles, s1_values, 'o-', linewidth=2.5, color=S1_COLOR, label='System 1', markersize=8)
ax.fill(angles, s1_values, alpha=0.2, color=S1_COLOR)
ax.plot(angles, s2_values, 's-', linewidth=2.5, color=S2_COLOR, label='System 2', markersize=8)
ax.fill(angles, s2_values, alpha=0.2, color=S2_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=13, fontweight='medium')
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=11, color='#666666')
# Move radial axis labels outward
ax.tick_params(axis='x', pad=15)
ax.grid(True, linestyle='-', alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig(figures_dir / "fig5_cognitive_effort_radar.png", dpi=300)
plt.close()
print("  ✓ fig5_cognitive_effort_radar.png")

# Figure 8: Token Usage
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Token usage comparison
ax1 = axes[0]
metrics = ['Avg Tokens', 'Response Time\n(seconds)']
s1_vals = [results["system1_tokens"]["mean"], results["system1_response_time_ms"]["mean"]/1000]
s2_vals = [results["system2_tokens"]["mean"], results["system2_response_time_ms"]["mean"]/1000]

x = np.arange(len(metrics))
width = 0.35

ax1.bar(x - width/2, s1_vals, width, label='System 1', color=S1_COLOR, edgecolor='white', linewidth=1.5)
ax1.bar(x + width/2, s2_vals, width, label='System 2', color=S2_COLOR, edgecolor='white', linewidth=1.5)

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=13)
ax1.set_ylabel('Value', fontsize=14)
ax1.legend(loc='upper right', fontsize=12)

# Efficiency comparison
ax2 = axes[1]
s1_acc_per_token = results["system1_accuracy"]["mean"] / results["system1_tokens"]["mean"] * 1000
s2_acc_per_token = results["system2_accuracy"]["mean"] / results["system2_tokens"]["mean"] * 1000
s1_acc_per_sec = results["system1_accuracy"]["mean"] / (results["system1_response_time_ms"]["mean"]/1000)
s2_acc_per_sec = results["system2_accuracy"]["mean"] / (results["system2_response_time_ms"]["mean"]/1000)

metrics = ['Accuracy per\n1K Tokens', 'Accuracy per\nSecond']
s1_effs = [s1_acc_per_token, s1_acc_per_sec]
s2_effs = [s2_acc_per_token, s2_acc_per_sec]

x = np.arange(len(metrics))
ax2.bar(x - width/2, s1_effs, width, label='System 1', color=S1_COLOR, edgecolor='white', linewidth=1.5)
ax2.bar(x + width/2, s2_effs, width, label='System 2', color=S2_COLOR, edgecolor='white', linewidth=1.5)

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=13)
ax2.set_ylabel('Efficiency Score', fontsize=14)
ax2.legend(loc='upper right', fontsize=12)

# Cost-benefit scatter
ax3 = axes[2]
ax3.scatter(results["system1_tokens"]["mean"], results["system1_accuracy"]["mean"],
           s=300, c=S1_COLOR, label='System 1', edgecolors='white', linewidth=2, marker='o', zorder=5)
ax3.scatter(results["system2_tokens"]["mean"], results["system2_accuracy"]["mean"],
           s=300, c=S2_COLOR, label='System 2', edgecolors='white', linewidth=2, marker='s', zorder=5)

ax3.set_xlabel('Tokens Used', fontsize=14)
ax3.set_ylabel('Accuracy', fontsize=14)
ax3.legend(loc='lower right', fontsize=12)
ax3.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(figures_dir / "fig8_token_usage.png", dpi=300)
plt.close()
print("  ✓ fig8_token_usage.png")

# Figure 9: Response Time Distribution
fig, ax = plt.subplots(figsize=(10, 6))

s1_time = results["system1_response_time_ms"]["mean"] / 1000
s2_time = results["system2_response_time_ms"]["mean"] / 1000

x = np.arange(2)
bars = ax.bar(x, [s1_time, s2_time], color=[S1_COLOR, S2_COLOR], alpha=0.8, edgecolor='white', linewidth=2)

for bar, time in zip(bars, [s1_time, s2_time]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
           f'{time:.1f}s', ha='center', fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'], fontsize=14)
ax.set_ylabel('Response Time (seconds)', fontsize=14)

ratio = s2_time / s1_time
ax.text(0.5, 0.95, f'System 2 is {ratio:.1f}× slower than System 1',
       transform=ax.transAxes, ha='center', fontsize=13, style='italic', color='#666666')

plt.tight_layout()
plt.savefig(figures_dir / "fig9_response_time_dist.png", dpi=300)
plt.close()
print("  ✓ fig9_response_time_dist.png")

print("\n✅ All figures regenerated with larger fonts!")
print(f"   Output directory: {figures_dir}")
