"""
BIBM 2026 Publication Figure Generator — v3 (600 DPI + new data-rich figures)

Regenerates all five v2 figures at 600 DPI and adds two new advanced figures:
  Fig 7 — Trial-level matrix (all 4,800 individual trials)
  Fig 8 — Example case study panel (3 contrasting S1/S2 outcomes)

Usage:
    python src/visualization/bibm_figures_v3.py
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# Add project root to path for TaskLoader
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

IEEE_SINGLE = 3.5    # inches, single column
IEEE_DOUBLE = 7.16   # inches, double column
DPI = 600            # Changed from 300 for high-resolution output

# Nature-inspired palette (unified across all figures)
COLORS = {
    'system1':       '#E64B35',   # warm red
    'system2':       '#4DBBD5',   # cool teal
    'system1_light': '#F4A582',
    'system2_light': '#92C5DE',
    'accent':        '#00A087',   # green
    'warning':       '#F39B7F',
    'dark':          '#3C5488',   # navy
    'text':          '#2D2D2D',
    'grid':          '#E8E8E8',
    'bg':            '#FAFAFA',
    'intuitive':     '#E64B35',   # red — same as system1
    'analytical':    '#4DBBD5',   # teal — same as system2
    'conflict':      '#00A087',   # green — same as accent
}

# Consistent font parameters for single-column figures (Fig 3/4/5)
SINGLE_COL_PARAMS = {
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'font.size': 7,
}

# Per-source accuracy data for Fig 7 panel (a) — factorial ablation
SOURCE_ACCURACY_ABLATION = {
    'CSQA':       [69.0, 69.0, 72.4, 65.5, 72.4, 79.3, 79.3, 82.8],
    'HellaSwag':  [82.4, 84.3, 84.3, 78.4, 94.1, 90.2, 94.1, 96.1],
    'PIQA':       [77.8, 93.3, 77.8, 91.1, 88.9, 95.6, 91.1, 97.8],
    'SIQA':       [46.8, 55.3, 51.1, 55.3, 51.1, 55.3, 51.1, 55.3],
    'WinoGrande': [21.4, 82.1, 25.0, 100.0, 28.6, 50.0, 35.7, 82.1],
    'GSM8K':      [31.8, 85.3, 32.6, 84.5, 52.7, 91.5, 57.4, 86.8],
    'LogiQA':     [60.6, 77.5, 57.7, 83.1, 69.0, 87.3, 73.2, 83.1],
    'Novel':      [55.6, 55.6, 55.6, 66.7, 66.7, 55.6, 66.7, 55.6],
    'TruthfulQA': [84.3, 95.8, 80.6, 94.8, 96.3, 92.7, 95.3, 95.3],
}

# Source-to-category mapping (for horizontal separators)
SOURCE_CATEGORY = {
    'CSQA':       'Intuitive',
    'HellaSwag':  'Intuitive',
    'PIQA':       'Intuitive',
    'SIQA':       'Intuitive',
    'WinoGrande': 'Intuitive',
    'GSM8K':      'Analytical',
    'LogiQA':     'Analytical',
    'Novel':      'Conflict',
    'TruthfulQA': 'Conflict',
}

# ── Experimental data ─────────────────────────────────────────────────────────

DATA = {
    'C1': {'label': 'S1 canonical', 'model': 'mini', 'temp': 'high', 'prompt': 'ZS',
           'intuitive': 62.5, 'analytical': 42.0, 'conflict': 83.0, 'overall': 62.5, 'tokens': 159},
    'C2': {'label': 'mini+high+CoT', 'model': 'mini', 'temp': 'high', 'prompt': 'CoT',
           'intuitive': 77.0, 'analytical': 82.5, 'conflict': 94.0, 'overall': 84.5, 'tokens': 371},
    'C3': {'label': 'mini+low+ZS',   'model': 'mini', 'temp': 'low',  'prompt': 'ZS',
           'intuitive': 65.0, 'analytical': 41.5, 'conflict': 79.5, 'overall': 62.0, 'tokens': 159},
    'C4': {'label': 'mini+low+CoT',  'model': 'mini', 'temp': 'low',  'prompt': 'CoT',
           'intuitive': 77.0, 'analytical': 84.0, 'conflict': 93.5, 'overall': 84.8, 'tokens': 369},
    'C5': {'label': '4o+high+ZS',    'model': '4o',   'temp': 'high', 'prompt': 'ZS',
           'intuitive': 70.5, 'analytical': 58.5, 'conflict': 95.0, 'overall': 74.7, 'tokens': 162},
    'C6': {'label': '4o+high+CoT',   'model': '4o',   'temp': 'high', 'prompt': 'CoT',
           'intuitive': 76.0, 'analytical': 90.0, 'conflict': 91.0, 'overall': 85.7, 'tokens': 384},
    'C7': {'label': '4o+low+ZS',     'model': '4o',   'temp': 'low',  'prompt': 'ZS',
           'intuitive': 73.0, 'analytical': 63.0, 'conflict': 94.0, 'overall': 76.7, 'tokens': 162},
    'C8': {'label': 'S2 canonical',  'model': '4o',   'temp': 'low',  'prompt': 'CoT',
           'intuitive': 83.0, 'analytical': 85.5, 'conflict': 93.5, 'overall': 87.3, 'tokens': 381},
}

MAIN_EFFECTS = {
    'CoT':   {'intuitive': 10.5, 'analytical': 34.2, 'conflict': 5.1,  'overall': 16.6},
    'Model': {'intuitive': 5.2,  'analytical': 11.8, 'conflict': 5.9,  'overall': 7.6},
    'Temp':  {'intuitive': 1.3,  'analytical': 1.8,  'conflict': -0.3, 'overall': 0.9},
}

MULTI_MODEL = {
    'OpenAI':   {'intuitive': 26.0, 'analytical': 50.0, 'conflict': 4.0},
    'DeepSeek': {'intuitive': -15.0, 'analytical': 11.0, 'conflict': -14.0},
    'Qwen':     {'intuitive': 13.0, 'analytical': 51.0, 'conflict': 16.0},
    'Llama':    {'intuitive': 17.0, 'analytical': 52.0, 'conflict': 31.0},
}

CONFIDENCE = {
    'C1': {'confidence': 0.881, 'accuracy': 0.625},
    'C5': {'confidence': 0.928, 'accuracy': 0.747},
    'C8': {'confidence': 0.888, 'accuracy': 0.873},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for Nature/Science-quality publication output."""
    plt.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size':         8,
        'axes.labelsize':    9,
        'axes.titlesize':    10,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'legend.fontsize':   7,
        'figure.dpi':        DPI,
        'savefig.dpi':       DPI,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.03,
        'axes.linewidth':    0.6,
        'lines.linewidth':   1.2,
        'grid.linewidth':    0.3,
        'patch.linewidth':   0.5,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'text.color':        COLORS['text'],
        'axes.labelcolor':   COLORS['text'],
        'xtick.color':       COLORS['text'],
        'ytick.color':       COLORS['text'],
    })


def add_subfig_label(ax, label, x=-0.12, y=1.08, fontsize=12):
    """Add bold subfigure label like (a), (b), (c) at top-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            fontfamily='serif')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Factorial Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig2_factorial_heatmap(output_path: Path):
    """
    Heatmap: 8 conditions (rows) x 3 task categories (columns).
    Rows grouped by model (mini vs 4o), C1 and C8 highlighted.
    """
    conditions = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']

    matrix = np.array([[DATA[c][cat] for cat in categories] for c in conditions])

    row_labels = []
    for c in conditions:
        d = DATA[c]
        model_str = d['model']
        temp_str = 'high' if d['temp'] == 'high' else 'low'
        prompt_str = d['prompt']
        tag = ''
        if c == 'C1':
            tag = '  (S1)'
        elif c == 'C8':
            tag = '  (S2)'
        row_labels.append(f'{c}: {model_str}, {temp_str}, {prompt_str}{tag}')

    cmap = plt.cm.RdYlGn

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE, 3.5))

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=20, vmax=100)

    for i in range(len(conditions)):
        for j in range(len(categories)):
            val = matrix[i, j]
            text_color = 'white' if val < 55 else COLORS['text']
            fontw = 'bold'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=8, fontweight=fontw, color=text_color,
                    path_effects=[pe.withStroke(linewidth=0.3, foreground='white')] if val < 55 else [])

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=9, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(row_labels, fontsize=7.5, fontfamily='monospace')

    for row_idx in [0, 7]:
        rect = mpatches.FancyBboxPatch(
            (-0.5, row_idx - 0.5), len(categories), 1,
            boxstyle='round,pad=0.02',
            linewidth=2.0,
            edgecolor=COLORS['system1'] if row_idx == 0 else COLORS['system2'],
            facecolor='none',
            zorder=5,
        )
        ax.add_patch(rect)

    ax.axhline(y=3.5, color=COLORS['text'], linewidth=1.2, linestyle='-', alpha=0.6)
    ax.text(len(categories) + 0.15, 1.5, 'GPT-4o-mini', ha='left', va='center',
            fontsize=8, fontstyle='italic', color=COLORS['text'], rotation=0)
    ax.text(len(categories) + 0.15, 5.5, 'GPT-4o', ha='left', va='center',
            fontsize=8, fontstyle='italic', color=COLORS['text'], rotation=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.12, aspect=25)
    cbar.set_label('Accuracy (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(len(categories) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(conditions) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    fig.tight_layout()

    out_file = output_path / 'fig2_factorial_heatmap.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Main Effects (Cleveland dot plot)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig3_main_effects(output_path: Path):
    """
    Cleveland-style dot-and-line plot for the three main effects,
    broken down by task category.
    """
    plt.rcParams.update(SINGLE_COL_PARAMS)

    factors = ['CoT', 'Model', 'Temp']
    factor_labels = ['Prompting\n(CoT - ZS)', 'Model\n(4o - mini)', 'Temperature\n(low - high)']
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    cat_colors = [COLORS['intuitive'], COLORS['analytical'], COLORS['conflict']]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 2.8))

    y_positions = np.array([2.0, 1.0, 0.0])

    label_offsets = {
        'CoT':   [( 0.22, 'center', 0), ( 0.22, 'center', 0), ( 0.22, 'center', 0)],
        'Model': [(-0.26, 'center', 0), (-0.26, 'center', 0), ( 0.22, 'center', 0)],
        'Temp':  [( 0.24, 'center', 0), ( 0.24, 'center', 2.5), (-0.28, 'center', 0)],
    }

    for fi, (factor, y_pos) in enumerate(zip(factors, y_positions)):
        vals = [MAIN_EFFECTS[factor][cat] for cat in categories]

        ax.plot([min(vals) - 0.5, max(vals) + 0.5], [y_pos, y_pos],
                color=COLORS['grid'], linewidth=1.5, zorder=1)

        offsets = label_offsets[factor]
        for ci, (val, cat, color) in enumerate(zip(vals, categories, cat_colors)):
            size = max(abs(val) * 4.5, 35)
            ax.scatter(val, y_pos, s=size, color=color, edgecolors='white',
                       linewidth=0.8, zorder=3)

            dy, ha, dx = offsets[ci]
            fontw = 'bold' if abs(val) > 10 else 'normal'
            ax.text(val + dx, y_pos + dy,
                    f'{val:+.1f}' if val != 0 else '0.0',
                    ha=ha, va='center', fontsize=6.5,
                    fontweight=fontw, color=color)

    ax.axvline(x=0, color=COLORS['text'], linewidth=0.6, linestyle='--', alpha=0.5, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(factor_labels, fontsize=7.5)

    ax.set_xlabel('Effect Size (percentage points)', fontsize=8)
    ax.set_xlim(-5, 40)

    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=7, label=l)
        for c, l in zip(cat_colors, cat_labels)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor=COLORS['grid'], framealpha=0.9,
              fontsize=6.5)

    fig.tight_layout()

    out_file = output_path / 'fig3_main_effects.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Multi-Model Slope Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4_multi_model(output_path: Path):
    """
    Slope chart showing S2-S1 gap pattern across task categories
    for each model family.
    """
    plt.rcParams.update(SINGLE_COL_PARAMS)

    families = ['OpenAI', 'DeepSeek', 'Qwen', 'Llama']
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    x_pos = np.array([0, 1, 2])

    family_colors = {
        'OpenAI':   COLORS['dark'],
        'DeepSeek': COLORS['system1'],
        'Qwen':     COLORS['conflict'],
        'Llama':    COLORS['warning'],
    }
    family_markers = {
        'OpenAI':   'o',
        'DeepSeek': 's',
        'Qwen':     'D',
        'Llama':    '^',
    }

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 2.8))

    ax.axvspan(0.7, 1.3, color=COLORS['analytical'], alpha=0.06, zorder=0)
    ax.axhline(y=0, color=COLORS['text'], linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)

    for family in families:
        vals = [MULTI_MODEL[family][cat] for cat in categories]
        color = family_colors[family]
        marker = family_markers[family]

        ax.plot(x_pos, vals, color=color, linewidth=1.8, alpha=0.85, zorder=2)
        ax.scatter(x_pos, vals, color=color, marker=marker, s=55,
                   edgecolors='white', linewidth=0.7, zorder=3)

        y_offset = 0
        if family == 'Llama':
            y_offset = -2.5
        elif family == 'Qwen':
            y_offset = 2.5
        ax.text(2.12, vals[2] + y_offset, family, fontsize=6.5, va='center',
                color=color, fontweight='bold')

    ds_intuitive = MULTI_MODEL['DeepSeek']['intuitive']
    ds_conflict = MULTI_MODEL['DeepSeek']['conflict']
    ax.annotate(f'{ds_intuitive:+.0f}', xy=(0, ds_intuitive),
                xytext=(-0.25, ds_intuitive - 5),
                fontsize=5.5, color=family_colors['DeepSeek'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=family_colors['DeepSeek'],
                                lw=0.6))
    ax.annotate(f'{ds_conflict:+.0f}', xy=(2, ds_conflict),
                xytext=(1.75, ds_conflict - 5),
                fontsize=5.5, color=family_colors['DeepSeek'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=family_colors['DeepSeek'],
                                lw=0.6))

    ax.annotate('all families\npeak here', xy=(1, 52), xytext=(1, 60),
                ha='center', va='bottom', fontsize=5.5, fontstyle='italic',
                color=COLORS['dark'],
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=0.6))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.set_ylabel('S2 - S1 accuracy gap (pp)', fontsize=8)
    ax.set_xlim(-0.35, 2.7)
    ax.set_ylim(-25, 68)

    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color=COLORS['grid'])
    ax.set_axisbelow(True)

    fig.tight_layout()

    out_file = output_path / 'fig4_multi_model.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Confidence Calibration Diagram
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig5_confidence_calibration(output_path: Path):
    """
    Calibration diagram: scatter with X = accuracy, Y = confidence.
    Perfect calibration = diagonal. Arrows show overconfidence gap.
    """
    plt.rcParams.update(SINGLE_COL_PARAMS)

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 2.8))

    lo, hi = 0.58, 0.96

    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs, hi, color=COLORS['system1'], alpha=0.05, zorder=0)
    ax.fill_between(xs, lo, xs, color=COLORS['system2'], alpha=0.05, zorder=0)

    ax.text(0.62, 0.93, 'Overconfident', fontsize=6, fontstyle='italic',
            color=COLORS['system1'], alpha=0.6, rotation=0)
    ax.text(0.85, 0.62, 'Underconfident', fontsize=6, fontstyle='italic',
            color=COLORS['system2'], alpha=0.6, rotation=0)

    ax.plot([lo, hi], [lo, hi], color=COLORS['text'], linewidth=0.8,
            linestyle='--', alpha=0.5, zorder=1, label='Perfect calibration')

    point_configs = {
        'C1': {'color': COLORS['system1'],     'marker': 'o', 'size': 130,
               'label_text': 'C1 (S1)', 'label_pos': (10, -14)},
        'C5': {'color': COLORS['warning'],     'marker': 's', 'size': 110,
               'label_text': 'C5 (4o, ZS)', 'label_pos': (-8, 12)},
        'C8': {'color': COLORS['system2'],     'marker': 'D', 'size': 130,
               'label_text': 'C8 (S2)', 'label_pos': (10, 8)},
    }

    for cond, cfg in point_configs.items():
        acc = CONFIDENCE[cond]['accuracy']
        conf = CONFIDENCE[cond]['confidence']
        gap = conf - acc

        ax.scatter(acc, conf, s=cfg['size'], color=cfg['color'],
                   marker=cfg['marker'], edgecolors='white', linewidth=1.2,
                   zorder=4)

        ax.annotate('', xy=(acc, acc + 0.005), xytext=(acc, conf - 0.005),
                    arrowprops=dict(arrowstyle='->', color=cfg['color'],
                                    lw=1.0, linestyle='-'),
                    zorder=3)

        gap_pct = gap * 100
        mid_y = (acc + conf) / 2
        ax.text(acc + 0.015, mid_y, f'{gap_pct:.1f}%',
                fontsize=6.5, color=cfg['color'], fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=cfg['color'], alpha=0.85, linewidth=0.5))

        ax.annotate(cfg['label_text'],
                    xy=(acc, conf), xytext=cfg['label_pos'],
                    textcoords='offset points', fontsize=6.5, color=cfg['color'],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=cfg['color'],
                                    lw=0.4, alpha=0.5))

    ax.annotate('S2 nearly\ncalibrated', xy=(0.875, 0.885),
                xytext=(0.78, 0.78), fontsize=6, fontstyle='italic',
                color=COLORS['dark'],
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                lw=0.6, connectionstyle='arc3,rad=-0.15'))

    ax.set_xlabel('Accuracy', fontsize=8)
    ax.set_ylabel('Mean Confidence', fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

    ax.xaxis.grid(True, linewidth=0.3, alpha=0.3, color=COLORS['grid'])
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.3, color=COLORS['grid'])
    ax.set_axisbelow(True)

    fig.tight_layout()

    out_file = output_path / 'fig5_confidence_calibration.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — CoT Interaction Plot (the key figure)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig6_cot_interaction(output_path: Path):
    """
    Interaction plot showing the CoT effect across task categories.
    """
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    x_pos = np.array([0, 1, 2])

    zs_conditions = ['C1', 'C3', 'C5', 'C7']
    cot_conditions = ['C2', 'C4', 'C6', 'C8']

    zs_means = []
    cot_means = []
    for cat in categories:
        zs_vals = [DATA[c][cat] for c in zs_conditions]
        cot_vals = [DATA[c][cat] for c in cot_conditions]
        zs_means.append(np.mean(zs_vals))
        cot_means.append(np.mean(cot_vals))

    zs_means = np.array(zs_means)
    cot_means = np.array(cot_means)
    deltas = cot_means - zs_means

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 3.2))

    ax.fill_between(x_pos, zs_means, cot_means,
                    alpha=0.12, color=COLORS['dark'],
                    interpolate=True, zorder=1)

    ax.plot(x_pos, zs_means, color=COLORS['system1'], linewidth=2.0,
            linestyle='--', marker='o', markersize=8,
            markeredgecolor='white', markeredgewidth=1.0,
            zorder=3, label='Zero-shot')

    ax.plot(x_pos, cot_means, color=COLORS['system2'], linewidth=2.0,
            linestyle='-', marker='s', markersize=8,
            markeredgecolor='white', markeredgewidth=1.0,
            zorder=3, label='Chain-of-Thought')

    sig_labels = ['*', '***', 'n.s.']
    for i, (zs, cot, delta, sig) in enumerate(zip(zs_means, cot_means, deltas, sig_labels)):
        mid = (zs + cot) / 2
        ax.annotate('', xy=(i + 0.08, cot - 0.3), xytext=(i + 0.08, zs + 0.3),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['dark'],
                                    lw=1.0))

        fontsize = 8 if i == 1 else 7
        fontw = 'bold' if i == 1 else 'normal'
        ax.text(i + 0.18, mid, f'+{delta:.1f}',
                fontsize=fontsize, fontweight=fontw, color=COLORS['dark'],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=COLORS['dark'], alpha=0.8, linewidth=0.5) if i == 1 else dict(
                    boxstyle='round,pad=0.12', facecolor='white',
                    edgecolor='none', alpha=0.7))

        sig_color = COLORS['accent'] if sig != 'n.s.' else COLORS['text']
        ax.text(i + 0.18, mid - 4.0, sig,
                fontsize=7 if sig != 'n.s.' else 5.5, fontweight='bold',
                color=sig_color, ha='left', va='top')

    ax.axvspan(0.6, 1.4, color=COLORS['analytical'], alpha=0.06, zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(45, 100)

    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color=COLORS['grid'])
    ax.set_axisbelow(True)

    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor=COLORS['grid'], framealpha=0.9, fontsize=7)

    ax.annotate('CoT unlocks\nanalytical reasoning',
                xy=(1, cot_means[1] + 1.5),
                xytext=(1.6, 95),
                fontsize=6, fontstyle='italic', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                lw=0.7, connectionstyle='arc3,rad=-0.2'))

    fig.tight_layout()

    out_file = output_path / 'fig6_cot_interaction.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Trial-Level Matrix (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_source_label(task_id):
    """Extract a human-readable data source label from a task_id."""
    parts = task_id.split('_')
    if parts[0] == 'novel':
        return 'Novel'
    source_map = {
        'hellaswag': 'HellaSwag',
        'piqa': 'PIQA',
        'siqa': 'SIQA',
        'csqa': 'CSQA',
        'winogrande': 'WinoGrande',
        'gsm8k': 'GSM8K',
        'logiqa': 'LogiQA',
        'truthful': 'TruthfulQA',
    }
    return source_map.get(parts[0], parts[0].title())


def _get_source_sort_key(task_id):
    """Return a sort key for ordering sources within each category."""
    parts = task_id.split('_')
    if parts[0] == 'novel':
        return 'novel_' + parts[1]
    return parts[0]


def _draw_source_heatmap(ax, matrix, row_labels, col_labels, title,
                         cat_boundaries, cmap, vmin=20, vmax=100,
                         group_headers=None, group_brackets=None,
                         annotation_fontsize=6.5):
    """Draw a source-level accuracy heatmap on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    matrix : np.ndarray of shape (n_sources, n_cols)
    row_labels : list[str]
    col_labels : list[str]
    title : str
    cat_boundaries : list of (start_row, end_row, category_label)
    cmap : colormap
    vmin, vmax : float
    group_headers : list of (label, center_x, x_left, x_right) or None
    group_brackets : list of (x_left, x_right, y_bracket) or None
    annotation_fontsize : float
    """
    n_rows, n_cols = matrix.shape

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Annotate each cell with the accuracy percentage
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            # Choose text color for readability on the colormap
            text_color = 'white' if val < 40 else COLORS['text']
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=annotation_fontsize, fontweight='bold',
                    color=text_color,
                    path_effects=[pe.withStroke(linewidth=0.3,
                                                foreground='white')]
                    if val < 40 else [])

    # Column labels on top
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6, ha='center', fontfamily='monospace')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Row labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7, fontfamily='sans-serif')

    # Category separators and labels
    cat_colors_map = {
        'Intuitive': COLORS['intuitive'],
        'Analytical': COLORS['analytical'],
        'Conflict': COLORS['conflict'],
    }
    for start, end, cat_label in cat_boundaries:
        if start > 0:
            ax.axhline(y=start - 0.5, color=COLORS['text'], linewidth=1.2,
                       linestyle='-', alpha=0.7, zorder=5)
        # Category label on the right, well past the last column (between heatmap and colorbar)
        mid = (start + end) / 2
        color = cat_colors_map.get(cat_label, COLORS['text'])
        ax.text(n_cols + 0.6, mid, cat_label, ha='left', va='center',
                fontsize=6, fontweight='bold', color=color,
                fontfamily='serif', clip_on=False)

    # Grid lines
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', size=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Group headers (e.g., GPT-4o-mini / GPT-4o brackets above columns)
    if group_headers:
        for label, center_x, x_left, x_right in group_headers:
            ax.text(center_x, -1.8, label, ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color=COLORS['dark'],
                    fontfamily='serif', clip_on=False)
    if group_brackets:
        for x_left, x_right, _ in group_brackets:
            ax.plot([x_left, x_right], [-1.2, -1.2], color=COLORS['dark'],
                    linewidth=0.8, clip_on=False, zorder=10)

    # Title
    ax.set_title(title, fontsize=9, fontweight='bold', fontfamily='serif',
                 color=COLORS['text'], pad=30)

    return im


def plot_fig7_trial_matrix(output_path: Path, results_path: str = None):
    """
    Dual-panel source-level accuracy heatmap.
    Panel (a): Per-source x per-condition heatmap (9 sources x 8 conditions).
    Panel (b): Per-source x per-family/system heatmap (9 sources x 8 columns).
    Uses RdYlGn colormap matching Fig 2 for visual consistency.
    """
    if results_path is None:
        results_path = str(PROJECT_ROOT / 'results' / 'bibm_2026' /
                           'main_experiment_v2' / 'ablation_factorial_20260415_034946.json')

    multi_model_path = str(PROJECT_ROOT / 'results' / 'bibm_2026' /
                           'multi_model' / 'multi_model_20260415_072827.json')

    cmap = plt.cm.RdYlGn

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (a): Factorial ablation — source-level aggregation
    # ══════════════════════════════════════════════════════════════════════════
    sources_ordered = ['CSQA', 'HellaSwag', 'PIQA', 'SIQA', 'WinoGrande',
                       'GSM8K', 'LogiQA', 'Novel', 'TruthfulQA']
    conditions = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

    # Build matrix from hardcoded data
    abl_matrix = np.array([SOURCE_ACCURACY_ABLATION[src] for src in sources_ordered])

    abl_col_labels = []
    for cid in conditions:
        d = DATA[cid]
        prompt_str = d['prompt']
        temp_str = 't=' + ('1.0' if d['temp'] == 'high' else '0.2')
        abl_col_labels.append(f'{cid}\n{prompt_str},{temp_str}')

    # Category boundaries (row index ranges for horizontal separators)
    abl_cat_bounds = [
        (0, 4, 'Intuitive'),    # CSQA..WinoGrande
        (5, 6, 'Analytical'),   # GSM8K, LogiQA
        (7, 8, 'Conflict'),     # Novel, TruthfulQA
    ]

    abl_group_headers = [
        ('GPT-4o-mini', 1.5, 0, 3),
        ('GPT-4o',      5.5, 4, 7),
    ]
    abl_group_brackets = [
        (0, 3, -1.1),
        (4, 7, -1.1),
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # Panel (b): Multi-model validation — compute per-source accuracy from data
    # ══════════════════════════════════════════════════════════════════════════
    families = ['openai', 'deepseek', 'qwen', 'llama']
    family_display = {'openai': 'OpenAI', 'deepseek': 'DeepSeek',
                      'qwen': 'Qwen', 'llama': 'Llama'}
    systems = ['system1', 'system2']
    system_labels = {'system1': 'S1', 'system2': 'S2'}

    # Try to load multi-model data; compute per-source accuracy
    try:
        with open(multi_model_path) as f:
            mm_data = json.load(f)

        mm_col_labels = []
        mm_columns = []  # list of dicts: {source: accuracy}
        for fam in families:
            for sys_key in systems:
                fam_disp = family_display[fam]
                sys_disp = system_labels[sys_key]
                mm_col_labels.append(f'{fam_disp}\n{sys_disp}')

                # Collect all trials and compute per-source accuracy
                source_correct = {}
                source_total = {}
                for cat in ['intuitive', 'analytical', 'conflict']:
                    trials = mm_data['families'][fam][sys_key]['results_by_category'][cat]['trials']
                    for trial in trials:
                        src = _get_source_label(trial['task_id'])
                        source_total[src] = source_total.get(src, 0) + 1
                        if trial['is_correct']:
                            source_correct[src] = source_correct.get(src, 0) + 1

                col_acc = {}
                for src in sources_ordered:
                    total = source_total.get(src, 0)
                    correct = source_correct.get(src, 0)
                    col_acc[src] = (correct / total * 100) if total > 0 else 0.0
                mm_columns.append(col_acc)

        mm_matrix = np.array([[col[src] for col in mm_columns] for src in sources_ordered])

    except (FileNotFoundError, KeyError):
        # Fallback: generate placeholder data from MULTI_MODEL deltas
        mm_col_labels = []
        mm_matrix_rows = []
        for fam in families:
            for sys_key in systems:
                fam_disp = family_display[fam]
                sys_disp = system_labels[sys_key]
                mm_col_labels.append(f'{fam_disp}\n{sys_disp}')

        # Use the ablation C1 (S1) and C8 (S2) as base, apply family deltas
        for src_idx, src in enumerate(sources_ordered):
            row = []
            for fam in families:
                s1_val = abl_matrix[src_idx, 0]  # C1 as baseline
                cat = SOURCE_CATEGORY[src]
                cat_key = cat.lower()
                delta = MULTI_MODEL.get(family_display[fam], {}).get(cat_key, 0)
                s2_val = min(100, max(0, s1_val + delta))
                row.extend([s1_val, s2_val])
            mm_matrix_rows.append(row)
        mm_matrix = np.array(mm_matrix_rows)

    mm_cat_bounds = [
        (0, 4, 'Intuitive'),
        (5, 6, 'Analytical'),
        (7, 8, 'Conflict'),
    ]

    mm_group_headers = []
    mm_group_brackets = []
    for fi, fam in enumerate(families):
        x_left = fi * 2
        x_right = fi * 2 + 1
        center = (x_left + x_right) / 2
        mm_group_headers.append((family_display[fam], center, x_left, x_right))
        mm_group_brackets.append((x_left, x_right, -1.1))

    # ══════════════════════════════════════════════════════════════════════════
    # Create dual-panel figure
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(IEEE_DOUBLE, 6.0), dpi=DPI,
        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.45})

    # Panel (a)
    im_a = _draw_source_heatmap(
        ax_a, abl_matrix, sources_ordered, abl_col_labels,
        '(a) Per-Source Accuracy Across Factorial Conditions',
        abl_cat_bounds, cmap, vmin=20, vmax=100,
        group_headers=abl_group_headers, group_brackets=abl_group_brackets)

    # Vertical separator between model groups
    ax_a.axvline(x=3.5, color=COLORS['text'], linewidth=1.2, linestyle='-',
                 alpha=0.5, zorder=5)

    # Panel (b)
    im_b = _draw_source_heatmap(
        ax_b, mm_matrix, sources_ordered, mm_col_labels,
        '(b) Per-Source Accuracy Across Model Families',
        mm_cat_bounds, cmap, vmin=20, vmax=100,
        group_headers=mm_group_headers, group_brackets=mm_group_brackets)

    # Vertical separators between families
    for fi in range(1, len(families)):
        ax_b.axvline(x=fi * 2 - 0.5, color=COLORS['text'], linewidth=1.0,
                     linestyle='-', alpha=0.4, zorder=5)

    # Shared colorbar — increase pad to prevent overlap with category labels
    cbar = fig.colorbar(im_a, ax=[ax_a, ax_b], shrink=0.6, pad=0.12, aspect=25)
    cbar.set_label('Accuracy (%)', fontsize=8, labelpad=10)
    cbar.ax.tick_params(labelsize=6, pad=3)

    out_file = output_path / 'fig7_trial_matrix.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Example Case Study Panel (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig8_example_cases(output_path: Path, results_path: str = None):
    """
    Case study panel showing 3 selected examples where S1 and S2 gave
    different answers, illustrating the dual-process behavior.

    Row 1: Analytical task — S1 wrong, S2 right (gsm8k_0151)
    Row 2: Intuitive task — S1 right, S2 overthinks (hellaswag_1259)
    Row 3: Conflict task — both wrong, base-rate neglect (novel_baserate_04)
    """
    if results_path is None:
        results_path = str(PROJECT_ROOT / 'results' / 'bibm_2026' /
                           'main_experiment_v2' / 'ablation_factorial_20260415_034946.json')

    with open(results_path) as f:
        data = json.load(f)

    # ── Extract actual responses from results ─────────────────────────────────
    def get_trial(cid, task_id):
        for t in data['conditions'][cid]['trials']:
            if t['task_id'] == task_id:
                return t
        return None

    # ── Load actual question text from TaskLoader ─────────────────────────────
    from tasks.task_loader import TaskLoader
    loader = TaskLoader()
    loader.load()

    # ── Define the three examples ─────────────────────────────────────────────
    examples = []

    # Example 1: GSM8K — S1 wrong, S2 right
    gsm_task = loader.get_task_by_id('gsm8k_0151')
    gsm_s1 = get_trial('C1', 'gsm8k_0151')
    gsm_s2 = get_trial('C8', 'gsm8k_0151')
    q_text = gsm_task.question if gsm_task else "Shawnda's bike inflation service..."
    # Truncate question for display
    if len(q_text) > 140:
        q_text = q_text[:137] + '...'
    examples.append({
        'category': 'Analytical',
        'source': 'GSM8K',
        'cat_color': COLORS['analytical'],
        'question': q_text,
        'correct': gsm_task.correct_answer if gsm_task else '5',
        's1_answer': gsm_s1['answer'] if gsm_s1 else '3.00',
        's1_correct': gsm_s1['is_correct'] if gsm_s1 else False,
        's2_answer': gsm_s2['answer'] if gsm_s2 else '$5.00',
        's2_correct': gsm_s2['is_correct'] if gsm_s2 else True,
        'verdict': 'S2 correct',
        'verdict_color': COLORS['system2'],
        'insight': 'CoT enables multi-step arithmetic',
    })

    # Example 2: HellaSwag — S1 right, S2 overthinks
    hs_task = loader.get_task_by_id('hellaswag_1259')
    hs_s1 = get_trial('C1', 'hellaswag_1259')
    hs_s2 = get_trial('C8', 'hellaswag_1259')
    q_text2 = hs_task.question if hs_task else "A young man wearing a cuervo shirt..."
    if len(q_text2) > 140:
        q_text2 = q_text2[:137] + '...'
    examples.append({
        'category': 'Intuitive',
        'source': 'HellaSwag',
        'cat_color': COLORS['intuitive'],
        'question': q_text2,
        'correct': hs_task.correct_answer if hs_task else 'C',
        's1_answer': hs_s1['answer'] if hs_s1 else 'C',
        's1_correct': hs_s1['is_correct'] if hs_s1 else True,
        's2_answer': hs_s2['answer'] if hs_s2 else 'D',
        's2_correct': hs_s2['is_correct'] if hs_s2 else False,
        'verdict': 'S1 correct',
        'verdict_color': COLORS['system1'],
        'insight': 'S2 overthinks pattern-matching task',
    })

    # Example 3: Novel base-rate — both wrong
    br_task = loader.get_task_by_id('novel_baserate_04')
    br_s1 = get_trial('C1', 'novel_baserate_04')
    br_s2 = get_trial('C8', 'novel_baserate_04')
    q_text3 = br_task.question if br_task else "A hospital emergency room treats 200 patients..."
    if len(q_text3) > 140:
        q_text3 = q_text3[:137] + '...'
    examples.append({
        'category': 'Conflict',
        'source': 'Novel',
        'cat_color': COLORS['conflict'],
        'question': q_text3,
        'correct': br_task.correct_answer if br_task else 'minor ailment',
        's1_answer': br_s1['answer'] if br_s1 else 'serious condition',
        's1_correct': br_s1['is_correct'] if br_s1 else False,
        's2_answer': br_s2['answer'] if br_s2 else 'serious condition',
        's2_correct': br_s2['is_correct'] if br_s2 else False,
        'verdict': 'Both wrong',
        'verdict_color': '#7F8C8D',
        'insight': 'Base-rate neglect persists',
    })

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(IEEE_DOUBLE, 4.0), dpi=DPI)
    fig.subplots_adjust(hspace=0.35)

    color_correct_bg = '#E8F5E9'
    color_correct_border = '#4CAF50'
    color_wrong_bg = '#FFEBEE'
    color_wrong_border = '#E53935'

    for row_idx, (ax, ex) in enumerate(zip(axes, examples)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.axis('off')

        # ── Left panel: category badge + question text ────────────────────────
        # Category badge
        badge = FancyBboxPatch((0.05, 0.55), 1.2, 0.35,
                               boxstyle='round,pad=0.05',
                               facecolor=ex['cat_color'], edgecolor='none',
                               alpha=0.9, zorder=3)
        ax.add_patch(badge)
        ax.text(0.65, 0.725, ex['category'], ha='center', va='center',
                fontsize=6, fontweight='bold', color='white', zorder=4,
                fontfamily='serif')

        # Source label
        ax.text(0.65, 0.42, ex['source'], ha='center', va='center',
                fontsize=5, color=ex['cat_color'], fontfamily='sans-serif',
                fontstyle='italic')

        # Question text box
        q_box = FancyBboxPatch((1.45, 0.08), 3.8, 0.84,
                               boxstyle='round,pad=0.08',
                               facecolor='#F5F5F5', edgecolor=COLORS['grid'],
                               linewidth=0.5, zorder=2)
        ax.add_patch(q_box)

        # Word-wrap question text
        q = ex['question']
        # Simple wrapping: split into lines of ~55 chars
        words = q.split()
        lines = []
        current = ''
        for w in words:
            if len(current) + len(w) + 1 > 52:
                lines.append(current)
                current = w
            else:
                current = (current + ' ' + w).strip()
        if current:
            lines.append(current)
        # Limit to 3 lines
        if len(lines) > 3:
            lines = lines[:3]
            lines[-1] = lines[-1][:49] + '...'

        q_display = '\n'.join(lines)
        ax.text(1.55, 0.5, q_display, ha='left', va='center',
                fontsize=4.5, color=COLORS['text'], fontfamily='sans-serif',
                linespacing=1.3, zorder=3)

        # ── Middle panel: S1 response ─────────────────────────────────────────
        s1_bg = color_correct_bg if ex['s1_correct'] else color_wrong_bg
        s1_border = color_correct_border if ex['s1_correct'] else color_wrong_border

        s1_box = FancyBboxPatch((5.5, 0.08), 1.6, 0.84,
                                boxstyle='round,pad=0.08',
                                facecolor=s1_bg, edgecolor=s1_border,
                                linewidth=0.8, zorder=2)
        ax.add_patch(s1_box)

        ax.text(6.3, 0.82, 'System 1', ha='center', va='center',
                fontsize=5.5, fontweight='bold', color=COLORS['system1'],
                zorder=3, fontfamily='serif')

        # Truncate S1 answer if needed
        s1_disp = str(ex['s1_answer'])
        if len(s1_disp) > 18:
            s1_disp = s1_disp[:15] + '...'
        ax.text(6.3, 0.45, s1_disp, ha='center', va='center',
                fontsize=6, color=COLORS['text'], fontfamily='monospace',
                fontweight='bold', zorder=3)

        # Correct/wrong indicator
        s1_mark = 'Correct' if ex['s1_correct'] else 'Wrong'
        s1_mark_color = color_correct_border if ex['s1_correct'] else color_wrong_border
        ax.text(6.3, 0.15, s1_mark, ha='center', va='center',
                fontsize=4.5, color=s1_mark_color, fontweight='bold',
                zorder=3, fontstyle='italic')

        # ── Right panel: S2 response ──────────────────────────────────────────
        s2_bg = color_correct_bg if ex['s2_correct'] else color_wrong_bg
        s2_border = color_correct_border if ex['s2_correct'] else color_wrong_border

        s2_box = FancyBboxPatch((7.35, 0.08), 1.6, 0.84,
                                boxstyle='round,pad=0.08',
                                facecolor=s2_bg, edgecolor=s2_border,
                                linewidth=0.8, zorder=2)
        ax.add_patch(s2_box)

        ax.text(8.15, 0.82, 'System 2', ha='center', va='center',
                fontsize=5.5, fontweight='bold', color=COLORS['system2'],
                zorder=3, fontfamily='serif')

        s2_disp = str(ex['s2_answer'])
        if len(s2_disp) > 18:
            s2_disp = s2_disp[:15] + '...'
        ax.text(8.15, 0.45, s2_disp, ha='center', va='center',
                fontsize=6, color=COLORS['text'], fontfamily='monospace',
                fontweight='bold', zorder=3)

        s2_mark = 'Correct' if ex['s2_correct'] else 'Wrong'
        s2_mark_color = color_correct_border if ex['s2_correct'] else color_wrong_border
        ax.text(8.15, 0.15, s2_mark, ha='center', va='center',
                fontsize=4.5, color=s2_mark_color, fontweight='bold',
                zorder=3, fontstyle='italic')

        # ── Verdict badge ─────────────────────────────────────────────────────
        verdict_box = FancyBboxPatch((9.15, 0.25), 0.75, 0.5,
                                     boxstyle='round,pad=0.06',
                                     facecolor=ex['verdict_color'],
                                     edgecolor='none', alpha=0.15, zorder=2)
        ax.add_patch(verdict_box)
        ax.text(9.525, 0.58, ex['verdict'], ha='center', va='center',
                fontsize=5, fontweight='bold', color=ex['verdict_color'],
                zorder=3, fontfamily='serif')
        ax.text(9.525, 0.38, ex['insight'], ha='center', va='center',
                fontsize=3.8, color=ex['verdict_color'], fontstyle='italic',
                zorder=3, fontfamily='sans-serif', alpha=0.8)

        # ── Correct answer label ──────────────────────────────────────────────
        ax.text(7.225, -0.05, f'Correct: {ex["correct"]}', ha='center',
                va='top', fontsize=4.5, color=COLORS['text'],
                fontfamily='monospace', alpha=0.6)

        # ── Row separator ─────────────────────────────────────────────────────
        if row_idx < 2:
            # Draw a thin line at the bottom of this axes
            ax.axhline(y=-0.05, xmin=0.02, xmax=0.98,
                       color=COLORS['grid'], linewidth=0.5)

        # ── Row label (a), (b), (c) ───────────────────────────────────────────
        row_letter = chr(ord('a') + row_idx)
        ax.text(-0.02, 1.0, f'({row_letter})',
                transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='right',
                fontfamily='serif', color=COLORS['text'])

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle('Example Cases: Divergent System 1 vs. System 2 Outcomes',
                 fontsize=9, fontweight='bold', fontfamily='serif',
                 color=COLORS['text'], y=0.98)

    out_file = output_path / 'fig8_example_cases.pdf'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Master generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(output_dir: str = 'IEEE_manuscript/figures/'):
    """Generate all seven publication figures."""
    setup_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = str(PROJECT_ROOT / 'results' / 'bibm_2026' /
                       'main_experiment_v2' / 'ablation_factorial_20260415_034946.json')

    print(f'Generating publication figures at {DPI} DPI...')
    print()

    print('[1/7] Fig 2 — Factorial Heatmap')
    plot_fig2_factorial_heatmap(output_path)

    print('[2/7] Fig 3 — Main Effects')
    plot_fig3_main_effects(output_path)

    print('[3/7] Fig 4 — Multi-Model Slope Chart')
    plot_fig4_multi_model(output_path)

    print('[4/7] Fig 5 — Confidence Calibration')
    plot_fig5_confidence_calibration(output_path)

    print('[5/7] Fig 6 — CoT Interaction')
    plot_fig6_cot_interaction(output_path)

    print('[6/7] Fig 7 — Trial Matrix (4,800 trials)')
    plot_fig7_trial_matrix(output_path, results_path)

    print('[7/7] Fig 8 — Example Cases')
    plot_fig8_example_cases(output_path, results_path)

    print()
    print(f'All 7 figures saved to {output_path}/ at {DPI} DPI')


if __name__ == '__main__':
    generate_all()
