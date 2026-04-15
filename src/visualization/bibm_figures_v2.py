"""
BIBM 2026 Publication Figure Generator — v2 (Nature/Science aesthetic)

Generates five high-quality figures for the dual-process theory in LLMs paper:
  Fig 2 — Factorial heatmap (8 conditions x 3 task categories)
  Fig 3 — Main effects dot-and-line plot (Cleveland style)
  Fig 4 — Multi-model slope chart (S2-S1 gap across task categories)
  Fig 5 — Confidence calibration diagram
  Fig 6 — CoT interaction plot (the paper's key figure)

Usage:
    python src/visualization/bibm_figures_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

IEEE_SINGLE = 3.5    # inches, single column
IEEE_DOUBLE = 7.16   # inches, double column
DPI = 300

# Nature-inspired palette
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
    'intuitive':     '#E64B35',
    'analytical':    '#4DBBD5',
    'conflict':      '#00A087',
}

# ── Experimental data ──────────────────────────────────────────────────────────

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

    # Build the data matrix
    matrix = np.array([[DATA[c][cat] for cat in categories] for c in conditions])

    # Row labels with factor levels
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

    # Custom colormap: red -> yellow -> green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'accuracy',
        ['#D73027', '#FC8D59', '#FEE08B', '#D9EF8B', '#66BD63', '#1A9850'],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE, 3.5))

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=40, vmax=100)

    # Cell annotations
    for i in range(len(conditions)):
        for j in range(len(categories)):
            val = matrix[i, j]
            # Choose text color for contrast
            text_color = 'white' if val < 55 else COLORS['text']
            fontw = 'bold'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=8, fontweight=fontw, color=text_color,
                    path_effects=[pe.withStroke(linewidth=0.3, foreground='white')] if val < 55 else [])

    # Axis setup
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=9, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(row_labels, fontsize=7.5, fontfamily='monospace')

    # Highlight S1 (C1) and S2 (C8) rows
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

    # Model group divider line
    ax.axhline(y=3.5, color=COLORS['text'], linewidth=1.2, linestyle='-', alpha=0.6)
    # Group labels on the right
    ax.text(len(categories) + 0.15, 1.5, 'GPT-4o-mini', ha='left', va='center',
            fontsize=8, fontstyle='italic', color=COLORS['text'], rotation=0)
    ax.text(len(categories) + 0.15, 5.5, 'GPT-4o', ha='left', va='center',
            fontsize=8, fontstyle='italic', color=COLORS['text'], rotation=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.12, aspect=25)
    cbar.set_label('Accuracy (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)

    # Remove default frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Grid lines between cells
    ax.set_xticks(np.arange(len(categories) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(conditions) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    fig.tight_layout()

    out_file = output_path / 'fig2_factorial_heatmap.png'
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
    factors = ['CoT', 'Model', 'Temp']
    factor_labels = ['Prompting\n(CoT - ZS)', 'Model\n(4o - mini)', 'Temperature\n(low - high)']
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    cat_colors = [COLORS['intuitive'], COLORS['analytical'], COLORS['conflict']]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 3.2))

    y_positions = np.array([2.4, 1.2, 0.0])  # top to bottom: CoT, Model, Temp

    # Pre-define label offsets to avoid overlaps per factor
    # Each entry: (y_offset, ha, x_nudge) for [intuitive, analytical, conflict]
    # Model row: intuitive=+5.2, analytical=+11.8, conflict=+5.9 — first & last close
    # Temp row: intuitive=+1.3, analytical=+1.8, conflict=-0.3 — all very close
    label_offsets = {
        'CoT':   [( 0.28, 'center', 0), ( 0.28, 'center', 0), ( 0.28, 'center', 0)],
        'Model': [(-0.32, 'center', 0), (-0.32, 'center', 0), ( 0.28, 'center', 0)],
        'Temp':  [( 0.30, 'center', 0), ( 0.30, 'center', 2.5), (-0.35, 'center', 0)],
    }

    for fi, (factor, y_pos) in enumerate(zip(factors, y_positions)):
        vals = [MAIN_EFFECTS[factor][cat] for cat in categories]

        # Draw thin connecting line spanning the range
        ax.plot([min(vals) - 0.5, max(vals) + 0.5], [y_pos, y_pos],
                color=COLORS['grid'], linewidth=1.5, zorder=1)

        # Dots
        offsets = label_offsets[factor]
        for ci, (val, cat, color) in enumerate(zip(vals, categories, cat_colors)):
            size = max(abs(val) * 4.5, 35)  # proportional to magnitude, min 35
            ax.scatter(val, y_pos, s=size, color=color, edgecolors='white',
                       linewidth=0.8, zorder=3)

            # Value labels — use pre-computed offsets to avoid overlaps
            dy, ha, dx = offsets[ci]
            fontw = 'bold' if abs(val) > 10 else 'normal'
            ax.text(val + dx, y_pos + dy,
                    f'{val:+.1f}' if val != 0 else '0.0',
                    ha=ha, va='center', fontsize=6.5,
                    fontweight=fontw, color=color)

    # Vertical dashed line at x=0
    ax.axvline(x=0, color=COLORS['text'], linewidth=0.6, linestyle='--', alpha=0.5, zorder=0)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(factor_labels, fontsize=7.5)

    ax.set_xlabel('Effect Size (percentage points)', fontsize=8)
    ax.set_xlim(-5, 40)

    # Clean up
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=7, label=l)
        for c, l in zip(cat_colors, cat_labels)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              fancybox=False, edgecolor=COLORS['grid'], framealpha=0.9,
              fontsize=6.5)

    # Bracket / emphasis for the prompting row
    ax.annotate('', xy=(34.2, 2.65), xytext=(10.5, 2.65),
                arrowprops=dict(arrowstyle='<->', color=COLORS['dark'],
                                lw=0.8, connectionstyle='arc3,rad=0'))
    ax.text(22.35, 2.82, 'largest spread', ha='center', va='bottom',
            fontsize=5.5, fontstyle='italic', color=COLORS['dark'])

    fig.tight_layout()

    out_file = output_path / 'fig3_main_effects.png'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Multi-Model Slope Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig4_multi_model(output_path: Path):
    """
    Slope chart showing S2-S1 gap pattern across task categories
    for each model family. X = task category, Y = S2-S1 gap (pp).
    """
    families = ['OpenAI', 'DeepSeek', 'Qwen', 'Llama']
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    x_pos = np.array([0, 1, 2])

    family_colors = {
        'OpenAI':   '#3C5488',
        'DeepSeek': '#E64B35',
        'Qwen':     '#00A087',
        'Llama':    '#F39B7F',
    }
    family_markers = {
        'OpenAI':   'o',
        'DeepSeek': 's',
        'Qwen':     'D',
        'Llama':    '^',
    }

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 3.0))

    # Shade background to emphasize the analytical peak
    ax.axvspan(0.7, 1.3, color=COLORS['analytical'], alpha=0.06, zorder=0)

    # Zero line
    ax.axhline(y=0, color=COLORS['text'], linewidth=0.5, linestyle='--', alpha=0.4, zorder=0)

    for family in families:
        vals = [MULTI_MODEL[family][cat] for cat in categories]
        color = family_colors[family]
        marker = family_markers[family]

        # Line
        ax.plot(x_pos, vals, color=color, linewidth=1.8, alpha=0.85, zorder=2)
        # Markers
        ax.scatter(x_pos, vals, color=color, marker=marker, s=55,
                   edgecolors='white', linewidth=0.7, zorder=3)

        # Label at the right end
        y_offset = 0
        if family == 'Llama':
            y_offset = -2.5
        elif family == 'Qwen':
            y_offset = 2.5
        ax.text(2.12, vals[2] + y_offset, family, fontsize=6.5, va='center',
                color=color, fontweight='bold')

    # Annotate DeepSeek's negative values
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

    # Annotation: "All families peak here"
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

    out_file = output_path / 'fig4_multi_model.png'
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
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE, 3.5))

    # Axis range
    lo, hi = 0.58, 0.96

    # Shade regions
    # Overconfident region (above diagonal): light red
    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs, hi, color=COLORS['system1'], alpha=0.05, zorder=0)
    ax.fill_between(xs, lo, xs, color=COLORS['system2'], alpha=0.05, zorder=0)

    # Region labels
    ax.text(0.62, 0.93, 'Overconfident', fontsize=6, fontstyle='italic',
            color=COLORS['system1'], alpha=0.6, rotation=0)
    ax.text(0.85, 0.62, 'Underconfident', fontsize=6, fontstyle='italic',
            color=COLORS['system2'], alpha=0.6, rotation=0)

    # Perfect calibration line
    ax.plot([lo, hi], [lo, hi], color=COLORS['text'], linewidth=0.8,
            linestyle='--', alpha=0.5, zorder=1, label='Perfect calibration')

    # Data points
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

        # Point
        ax.scatter(acc, conf, s=cfg['size'], color=cfg['color'],
                   marker=cfg['marker'], edgecolors='white', linewidth=1.2,
                   zorder=4)

        # Arrow from point down to diagonal (showing overconfidence)
        ax.annotate('', xy=(acc, acc + 0.005), xytext=(acc, conf - 0.005),
                    arrowprops=dict(arrowstyle='->', color=cfg['color'],
                                    lw=1.0, linestyle='-'),
                    zorder=3)

        # Gap label — position to the right of the arrow
        gap_pct = gap * 100
        mid_y = (acc + conf) / 2
        ax.text(acc + 0.015, mid_y, f'{gap_pct:.1f}%',
                fontsize=6.5, color=cfg['color'], fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=cfg['color'], alpha=0.85, linewidth=0.5))

        # Point label
        ax.annotate(cfg['label_text'],
                    xy=(acc, conf), xytext=cfg['label_pos'],
                    textcoords='offset points', fontsize=6.5, color=cfg['color'],
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=cfg['color'],
                                    lw=0.4, alpha=0.5))

    # Key insight annotation
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

    # Grid
    ax.xaxis.grid(True, linewidth=0.3, alpha=0.3, color=COLORS['grid'])
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.3, color=COLORS['grid'])
    ax.set_axisbelow(True)

    fig.tight_layout()

    out_file = output_path / 'fig5_confidence_calibration.png'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — CoT Interaction Plot (the key figure)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig6_cot_interaction(output_path: Path):
    """
    Interaction plot showing the CoT effect across task categories.
    Two lines (Zero-shot vs CoT), with the gap filled and annotated.
    The analytical category shows the largest divergence.
    """
    categories = ['intuitive', 'analytical', 'conflict']
    cat_labels = ['Intuitive', 'Analytical', 'Conflict']
    x_pos = np.array([0, 1, 2])

    # Compute ZS and CoT means across all conditions
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

    # Fill between the two lines with a gradient effect
    ax.fill_between(x_pos, zs_means, cot_means,
                    alpha=0.12, color=COLORS['dark'],
                    interpolate=True, zorder=1)

    # Zero-shot line (dashed, warm)
    ax.plot(x_pos, zs_means, color=COLORS['system1'], linewidth=2.0,
            linestyle='--', marker='o', markersize=8,
            markeredgecolor='white', markeredgewidth=1.0,
            zorder=3, label='Zero-shot')

    # CoT line (solid, cool)
    ax.plot(x_pos, cot_means, color=COLORS['system2'], linewidth=2.0,
            linestyle='-', marker='s', markersize=8,
            markeredgecolor='white', markeredgewidth=1.0,
            zorder=3, label='Chain-of-Thought')

    # Delta annotations with arrows and significance
    sig_labels = ['*', '***', 'n.s.']
    for i, (zs, cot, delta, sig) in enumerate(zip(zs_means, cot_means, deltas, sig_labels)):
        mid = (zs + cot) / 2
        # Connecting bracket / arrow
        ax.annotate('', xy=(i + 0.08, cot - 0.3), xytext=(i + 0.08, zs + 0.3),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['dark'],
                                    lw=1.0))

        # Delta text
        fontsize = 8 if i == 1 else 7  # analytical is larger
        fontw = 'bold' if i == 1 else 'normal'
        ax.text(i + 0.18, mid, f'+{delta:.1f}',
                fontsize=fontsize, fontweight=fontw, color=COLORS['dark'],
                ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor=COLORS['dark'], alpha=0.8, linewidth=0.5) if i == 1 else dict(
                    boxstyle='round,pad=0.12', facecolor='white',
                    edgecolor='none', alpha=0.7))

        # Significance stars
        sig_color = COLORS['accent'] if sig != 'n.s.' else COLORS['text']
        ax.text(i + 0.18, mid - 4.0, sig,
                fontsize=7 if sig != 'n.s.' else 5.5, fontweight='bold',
                color=sig_color, ha='left', va='top')

    # Highlight the analytical column
    ax.axvspan(0.6, 1.4, color=COLORS['analytical'], alpha=0.06, zorder=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(45, 100)

    ax.yaxis.grid(True, linewidth=0.3, alpha=0.4, color=COLORS['grid'])
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=False,
              edgecolor=COLORS['grid'], framealpha=0.9, fontsize=7)

    # Annotation callout for the key finding
    ax.annotate('CoT unlocks\nanalytical reasoning',
                xy=(1, cot_means[1] + 1.5),
                xytext=(1.6, 95),
                fontsize=6, fontstyle='italic', color=COLORS['dark'],
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                lw=0.7, connectionstyle='arc3,rad=-0.2'))

    fig.tight_layout()

    out_file = output_path / 'fig6_cot_interaction.png'
    fig.savefig(out_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved {out_file}')


# ═══════════════════════════════════════════════════════════════════════════════
# Master generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all(output_dir: str = 'IEEE_manuscript/figures/'):
    """Generate all five publication figures."""
    setup_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print('Generating Nature-quality publication figures...')
    plot_fig2_factorial_heatmap(output_path)
    plot_fig3_main_effects(output_path)
    plot_fig4_multi_model(output_path)
    plot_fig5_confidence_calibration(output_path)
    plot_fig6_cot_interaction(output_path)
    print(f'All 5 figures saved to {output_path}/')


if __name__ == '__main__':
    generate_all()
