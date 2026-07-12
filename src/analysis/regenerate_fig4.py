#!/usr/bin/env python3
"""
Regenerate fig4_confidence_analysis.png with larger font sizes
"""

import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
# Anchor every path to the project root so these scripts work from any cwd.
# (They used to use bare relative paths, which only resolved when run from the
#  project root -- but run_all_experiments.sh cd's into src/ first.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

results_dir = PROJECT_ROOT / "results" / "paper_parallel_20260125_214106" / "round_1"
figures_dir = PROJECT_ROOT / "IEEE_manuscript" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Load detailed results
with open(results_dir / "round_1_detailed.json", "r") as f:
    detailed = json.load(f)

s1_results = detailed["system1_detailed"]["results"]
s2_results = detailed["system2_detailed"]["results"]

# Colors
S1_COLOR = '#E07A5F'  # Terra cotta
S2_COLOR = '#81B29A'  # Sage green

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, results, name, color in zip(
    axes, [s1_results, s2_results], 
    ['System 1 (Intuitive)', 'System 2 (Analytical)'],
    [S1_COLOR, S2_COLOR]
):
    confidences = [r.get('confidence', 0.5) for r in results]
    correct = [r.get('is_correct', False) for r in results]
    
    # Bin confidences
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            bin_accs.append(np.mean([c for c, m in zip(correct, mask) if m]))
            bin_confs.append(np.mean([c for c, m in zip(confidences, mask) if m]))
            bin_counts.append(count)
        else:
            bin_accs.append(0)
            bin_confs.append((bins[i] + bins[i+1]) / 2)
            bin_counts.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, 
           label='Perfect Calibration')
    
    # Plot calibration bars
    bar_width = 0.08
    bars = ax.bar(bin_centers, bin_accs, width=bar_width, 
                 color=color, alpha=0.7, edgecolor='white', linewidth=1.5,
                 label='Observed Accuracy')
    
    # Add confidence region
    ax.fill_between([0, 1], [0, 1], [0.1, 1.1], alpha=0.1, color='gray')
    ax.fill_between([0, 1], [0, 1], [-0.1, 0.9], alpha=0.1, color='gray')
    
    # Calculate ECE (Expected Calibration Error)
    total_samples = sum(bin_counts)
    ece = sum(abs(acc - conf) * count / total_samples 
             for acc, conf, count in zip(bin_accs, bin_confs, bin_counts) if count > 0)
    
    # Add ECE annotation - increased font size
    ax.text(0.05, 0.95, f'ECE = {ece:.3f}', transform=ax.transAxes,
           fontsize=16, fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    edgecolor=color, alpha=0.9))
    
    # Axis labels - increased font size
    ax.set_xlabel('Confidence', fontsize=17, fontweight='medium')
    ax.set_ylabel('Accuracy', fontsize=17, fontweight='medium')
    
    # Subplot title - kept
    ax.set_title(f'{name}\nCalibration', fontsize=17, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=14)
    ax.set_aspect('equal')
    
    # Tick labels - increased font size
    ax.tick_params(axis='both', labelsize=15)

# No figure title (removed suptitle)
plt.tight_layout()
plt.savefig(figures_dir / "fig4_confidence_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ fig4_confidence_analysis.png regenerated with larger fonts")
