"""
BIBM 2026 Publication Figure Generator

Generates IEEE-compliant figures for the dual-process theory paper.

Usage:
    # From actual results
    python -m src.visualization.bibm_figures --input results/bibm_2026/ --output IEEE_manuscript/figures/

    # Generate with sample data (for layout/design testing)
    python -m src.visualization.bibm_figures --sample --output IEEE_manuscript/figures/
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional

# IEEE formatting constants
IEEE_SINGLE_COL = 3.5   # inches
IEEE_DOUBLE_COL = 7.16  # inches
IEEE_DPI = 300

# Professional, colorblind-safe palette
COLORS = {
    'system1': '#E64B35',       # Red for intuitive/fast
    'system2': '#4DBBD5',       # Cyan for analytical/slow
    'system1_light': '#F4A582',
    'system2_light': '#92C5DE',
    'human': '#00A087',         # Green for human data
    'neutral': '#8C8C8C',
    'accent': '#3C5488',
}

# Sample data for layout / design testing
SAMPLE_FACTORIAL = {
    "C1": {"intuitive": 0.50, "analytical": 0.72, "conflict": 0.60},  # mini, high-T, zero-shot (S1)
    "C2": {"intuitive": 0.52, "analytical": 0.80, "conflict": 0.65},  # mini, high-T, CoT
    "C3": {"intuitive": 0.51, "analytical": 0.74, "conflict": 0.62},  # mini, low-T, zero-shot
    "C4": {"intuitive": 0.53, "analytical": 0.82, "conflict": 0.68},  # mini, low-T, CoT
    "C5": {"intuitive": 0.55, "analytical": 0.78, "conflict": 0.70},  # 4o, high-T, zero-shot
    "C6": {"intuitive": 0.56, "analytical": 0.88, "conflict": 0.80},  # 4o, high-T, CoT
    "C7": {"intuitive": 0.54, "analytical": 0.80, "conflict": 0.72},  # 4o, low-T, zero-shot
    "C8": {"intuitive": 0.56, "analytical": 0.93, "conflict": 0.85},  # 4o, low-T, CoT (S2)
}

SAMPLE_MULTI_MODEL = {
    "OpenAI": {"intuitive": 0.063, "analytical": 0.213, "conflict": 0.017},
    "DeepSeek": {"intuitive": 0.045, "analytical": 0.185, "conflict": 0.032},
    "Qwen": {"intuitive": 0.055, "analytical": 0.195, "conflict": 0.025},
    "Llama": {"intuitive": 0.040, "analytical": 0.170, "conflict": 0.028},
}

SAMPLE_HUMAN_COMPARISON = {
    "Human": {"intuitive": 0.030, "analytical": 0.150, "conflict": 0.080},
    "LLM (avg)": {"intuitive": 0.051, "analytical": 0.191, "conflict": 0.026},
}

SAMPLE_TOKEN_EFFICIENCY = {
    "intuitive": {"S1": 0.50, "S2": 0.52, "S1_tokens": 45, "S2_tokens": 310},
    "analytical": {"S1": 0.72, "S2": 0.93, "S1_tokens": 48, "S2_tokens": 320},
    "conflict":   {"S1": 0.65, "S2": 0.80, "S1_tokens": 46, "S2_tokens": 315},
}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_ieee_style():
    """Configure matplotlib for IEEE publication quality."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': IEEE_DPI,
        'savefig.dpi': IEEE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'grid.linewidth': 0.3,
        'patch.linewidth': 0.5,
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(input_dir: str) -> Dict[str, Any]:
    """Load experiment results from a directory of JSON files."""
    input_path = Path(input_dir)
    data: Dict[str, Any] = {}

    factorial_candidates = [
        input_path / "factorial_ablation.json",
        input_path / "factorial.json",
    ]
    for p in factorial_candidates:
        if p.exists():
            with open(p) as f:
                data["factorial"] = json.load(f)
            break

    multi_candidates = [
        input_path / "multi_model.json",
        input_path / "multi_model_validation.json",
    ]
    for p in multi_candidates:
        if p.exists():
            with open(p) as f:
                data["multi_model"] = json.load(f)
            break

    human_candidates = [
        input_path / "human_comparison.json",
        input_path / "human.json",
    ]
    for p in human_candidates:
        if p.exists():
            with open(p) as f:
                data["human_comparison"] = json.load(f)
            break

    token_candidates = [
        input_path / "token_efficiency.json",
        input_path / "tokens.json",
    ]
    for p in token_candidates:
        if p.exists():
            with open(p) as f:
                data["token_efficiency"] = json.load(f)
            break

    return data


def get_sample_data() -> Dict[str, Any]:
    """Return all sample datasets."""
    return {
        "factorial": SAMPLE_FACTORIAL,
        "multi_model": SAMPLE_MULTI_MODEL,
        "human_comparison": SAMPLE_HUMAN_COMPARISON,
        "token_efficiency": SAMPLE_TOKEN_EFFICIENCY,
    }


# ---------------------------------------------------------------------------
# Figure 2 — Factorial ablation (double column)
# ---------------------------------------------------------------------------

def plot_factorial_ablation(
    factorial_data: Optional[Dict],
    output_path: Path,
) -> plt.Figure:
    """
    Fig 2: 2x3 panel showing main effects of the 2x2x2 factorial design.

    Top row  — main effect of each factor (model, temperature, prompt) on accuracy.
    Bottom row — key interactions (Prompt x Task Category).
    """
    if factorial_data is None:
        factorial_data = SAMPLE_FACTORIAL

    categories = ["intuitive", "analytical", "conflict"]
    cat_labels = ["Intuitive", "Analytical", "Conflict"]

    # Decode conditions: C1-C4 = mini, C5-C8 = 4o
    #                    odd = high-T, even... actually C1,C2=high-T, C3,C4=low-T (per docstring)
    # Factor mappings (0-indexed from C1):
    # model:  C1-C4 = mini (0), C5-C8 = 4o (1)
    # temp:   C1,C2,C5,C6 = high (0), C3,C4,C7,C8 = low (1)
    # prompt: C1,C3,C5,C7 = zero-shot (0), C2,C4,C6,C8 = CoT (1)

    conditions = [f"C{i}" for i in range(1, 9)]
    model_idx  = [0, 0, 0, 0, 1, 1, 1, 1]   # 0=mini, 1=4o
    temp_idx   = [0, 0, 1, 1, 0, 0, 1, 1]   # 0=high, 1=low
    prompt_idx = [0, 1, 0, 1, 0, 1, 0, 1]   # 0=zero-shot, 1=CoT

    def mean_by_factor(factor_list, value, cat):
        vals = [factorial_data[c][cat] for c, f in zip(conditions, factor_list) if f == value]
        return np.mean(vals)

    fig, axes = plt.subplots(2, 3, figsize=(IEEE_DOUBLE_COL, 3.8))

    # ---- Top row: main effects ----
    factor_configs = [
        ("Model",       model_idx,  ["mini", "4o"],        [COLORS['system1_light'], COLORS['system1']]),
        ("Temperature", temp_idx,   ["High", "Low"],       [COLORS['system2_light'], COLORS['system2']]),
        ("Prompting",   prompt_idx, ["Zero-shot", "CoT"],  [COLORS['neutral'],       COLORS['accent']]),
    ]

    for col, (factor_name, factor_list, level_labels, bar_colors) in enumerate(factor_configs):
        ax = axes[0, col]
        x = np.arange(len(categories))
        width = 0.35

        for lv, (label, color) in enumerate(zip(level_labels, bar_colors)):
            vals = [mean_by_factor(factor_list, lv, c) for c in categories]
            bars = ax.bar(x + (lv - 0.5) * width, vals, width,
                          label=label, color=color, edgecolor='black',
                          linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("Accuracy" if col == 0 else "")
        ax.set_title(f"Effect of {factor_name}", fontsize=9)
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=6, loc='lower right', frameon=False)

    # ---- Bottom row: interactions (Prompt x Category for each model) ----
    model_names = ["mini", "4o"]
    model_conditions = [["C1", "C2", "C3", "C4"], ["C5", "C6", "C7", "C8"]]

    for col, (mname, mconds) in enumerate(zip(model_names, model_conditions)):
        ax = axes[1, col]
        x = np.arange(len(categories))
        width = 0.35

        zs_vals = [np.mean([factorial_data[c][cat] for c, pi in zip(mconds, [0,1,0,1]) if pi == 0])
                   for cat in categories]
        cot_vals = [np.mean([factorial_data[c][cat] for c, pi in zip(mconds, [0,1,0,1]) if pi == 1])
                    for cat in categories]

        ax.bar(x - width/2, zs_vals,  width, label="Zero-shot", color=COLORS['neutral'],
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, cot_vals, width, label="CoT",       color=COLORS['accent'],
               edgecolor='black', linewidth=0.5)

        # Annotate CoT gain
        for xi, (z, c) in enumerate(zip(zs_vals, cot_vals)):
            gain = c - z
            ax.annotate(f'+{gain:.2f}', xy=(xi + width/2, c + 0.01),
                        ha='center', va='bottom', fontsize=5.5, color=COLORS['accent'])

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=7)
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel("Accuracy" if col == 0 else "")
        ax.set_title(f"Prompt Effect ({mname})", fontsize=9)
        ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=6, loc='lower right', frameon=False)

    # Third bottom subplot: difference (CoT gain) comparison across models
    ax = axes[1, 2]
    x = np.arange(len(categories))
    width = 0.35

    for mi, (mname, mconds, color) in enumerate(zip(
            model_names, model_conditions,
            [COLORS['system1'], COLORS['system2']])):
        zs_vals = [np.mean([factorial_data[c][cat] for c, pi in zip(mconds, [0,1,0,1]) if pi == 0])
                   for cat in categories]
        cot_vals = [np.mean([factorial_data[c][cat] for c, pi in zip(mconds, [0,1,0,1]) if pi == 1])
                    for cat in categories]
        gains = [c - z for z, c in zip(zs_vals, cot_vals)]
        ax.bar(x + (mi - 0.5) * width, gains, width, label=mname,
               color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=7)
    ax.set_ylabel("CoT Accuracy Gain")
    ax.set_title("CoT Gain by Model", fontsize=9)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper left', frameon=False)

    fig.suptitle("Fig. 2: Factorial Ablation — Main Effects and Interactions",
                 fontsize=9, y=1.01)
    fig.tight_layout()

    out_file = output_path / "fig2_factorial_ablation.png"
    fig.savefig(out_file, dpi=IEEE_DPI, bbox_inches='tight')
    print(f"Saved {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Main effects bar chart (single column)
# ---------------------------------------------------------------------------

def plot_main_effects(
    factorial_data: Optional[Dict],
    output_path: Path,
) -> plt.Figure:
    """
    Fig 3: Bar chart of the 3 main effects (model, temperature, prompt)
    as accuracy *differences*, broken down by task category.
    """
    if factorial_data is None:
        factorial_data = SAMPLE_FACTORIAL

    categories = ["intuitive", "analytical", "conflict"]
    cat_labels  = ["Intuitive", "Analytical", "Conflict"]
    conditions  = [f"C{i}" for i in range(1, 9)]
    model_idx   = [0, 0, 0, 0, 1, 1, 1, 1]
    temp_idx    = [0, 0, 1, 1, 0, 0, 1, 1]
    prompt_idx  = [0, 1, 0, 1, 0, 1, 0, 1]

    def delta(factor_list, cat):
        lo = np.mean([factorial_data[c][cat] for c, f in zip(conditions, factor_list) if f == 0])
        hi = np.mean([factorial_data[c][cat] for c, f in zip(conditions, factor_list) if f == 1])
        return hi - lo

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))

    factors = [
        ("Model\n(4o−mini)", model_idx,  COLORS['system1']),
        ("Temp\n(low−high)", temp_idx,   COLORS['system2']),
        ("Prompt\n(CoT−ZS)", prompt_idx, COLORS['accent']),
    ]

    n_factors = len(factors)
    n_cats    = len(categories)
    group_w   = 0.8
    bar_w     = group_w / n_cats
    x_base    = np.arange(n_factors)

    for ci, (cat, cat_label) in enumerate(zip(categories, cat_labels)):
        deltas = [delta(fl, cat) for _, fl, _ in factors]
        offsets = (ci - (n_cats - 1) / 2) * bar_w
        bars = ax.bar(x_base + offsets, deltas, bar_w,
                      label=cat_label,
                      color=[COLORS['system1_light'], COLORS['neutral'], COLORS['system2_light']][ci],
                      edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xticks(x_base)
    ax.set_xticklabels([f[0] for f in factors], fontsize=7)
    ax.set_ylabel("Accuracy Difference")
    ax.set_title("Fig. 3: Main Effects of Experimental Factors", fontsize=9)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper left', frameon=False, ncol=1)

    fig.tight_layout()
    out_file = output_path / "fig3_main_effects.png"
    fig.savefig(out_file, dpi=IEEE_DPI, bbox_inches='tight')
    print(f"Saved {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Multi-model S2−S1 gap (single column)
# ---------------------------------------------------------------------------

def plot_multi_model(
    multi_model_data: Optional[Dict],
    output_path: Path,
) -> plt.Figure:
    """
    Fig 4: Grouped bar chart showing S2−S1 accuracy gap for each model
    family, by task category.
    """
    if multi_model_data is None:
        multi_model_data = SAMPLE_MULTI_MODEL

    models     = list(multi_model_data.keys())
    categories = ["intuitive", "analytical", "conflict"]
    cat_labels  = ["Intuitive", "Analytical", "Conflict"]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))

    n_models = len(models)
    n_cats   = len(categories)
    group_w  = 0.8
    bar_w    = group_w / n_models
    x_base   = np.arange(n_cats)

    model_colors = [COLORS['system1'], COLORS['system2'], COLORS['human'], COLORS['accent']]

    for mi, (model, color) in enumerate(zip(models, model_colors)):
        vals = [multi_model_data[model][cat] for cat in categories]
        offsets = (mi - (n_models - 1) / 2) * bar_w
        ax.bar(x_base + offsets, vals, bar_w,
               label=model, color=color, edgecolor='black', linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels(cat_labels, fontsize=7)
    ax.set_ylabel("S2 − S1 Accuracy Gap")
    ax.set_title("Fig. 4: Multi-Model Consistency", fontsize=9)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper right', frameon=False)

    fig.tight_layout()
    out_file = output_path / "fig4_multi_model.png"
    fig.savefig(out_file, dpi=IEEE_DPI, bbox_inches='tight')
    print(f"Saved {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Human vs LLM comparison (single column)
# ---------------------------------------------------------------------------

def plot_human_comparison(
    human_data: Optional[Dict],
    output_path: Path,
) -> plt.Figure:
    """
    Fig 5: Side-by-side comparison of S2−S1 gaps between humans and LLMs,
    by task category.
    """
    if human_data is None:
        human_data = SAMPLE_HUMAN_COMPARISON

    groups     = list(human_data.keys())
    categories = ["intuitive", "analytical", "conflict"]
    cat_labels  = ["Intuitive", "Analytical", "Conflict"]

    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 2.8))

    n_groups = len(groups)
    group_w  = 0.7
    bar_w    = group_w / n_groups
    x_base   = np.arange(len(categories))

    group_colors = [COLORS['human'], COLORS['system2']]

    for gi, (group, color) in enumerate(zip(groups, group_colors)):
        vals = [human_data[group][cat] for cat in categories]
        offsets = (gi - (n_groups - 1) / 2) * bar_w
        ax.bar(x_base + offsets, vals, bar_w,
               label=group, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels(cat_labels, fontsize=7)
    ax.set_ylabel("S2 − S1 Accuracy Gap")
    ax.set_title("Fig. 5: Human vs. LLM Dual-Process Gap", fontsize=9)
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=6, loc='upper right', frameon=False)

    fig.tight_layout()
    out_file = output_path / "fig5_human_comparison.png"
    fig.savefig(out_file, dpi=IEEE_DPI, bbox_inches='tight')
    print(f"Saved {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Token efficiency (single column)
# ---------------------------------------------------------------------------

def plot_token_efficiency(
    token_data: Optional[Dict],
    output_path: Path,
) -> plt.Figure:
    """
    Fig 6: Scatter + bar hybrid showing Token Efficiency Ratio
    (accuracy per 100 tokens) for S1 vs S2 by task category.
    """
    if token_data is None:
        token_data = SAMPLE_TOKEN_EFFICIENCY

    categories = list(token_data.keys())
    cat_labels  = [c.capitalize() for c in categories]

    s1_eff = [token_data[c]["S1"] / (token_data[c]["S1_tokens"] / 100) for c in categories]
    s2_eff = [token_data[c]["S2"] / (token_data[c]["S2_tokens"] / 100) for c in categories]
    s1_acc = [token_data[c]["S1"] for c in categories]
    s2_acc = [token_data[c]["S2"] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_SINGLE_COL, 2.6))

    # Left: efficiency ratio bars
    x      = np.arange(len(categories))
    width  = 0.35
    ax1.bar(x - width/2, s1_eff, width, label="S1 (fast)", color=COLORS['system1'],
            edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, s2_eff, width, label="S2 (slow)", color=COLORS['system2'],
            edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=6.5)
    ax1.set_ylabel("Accuracy / 100 tokens")
    ax1.set_title("Token Efficiency", fontsize=8)
    ax1.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=5.5, loc='upper right', frameon=False)

    # Right: scatter — accuracy vs tokens
    s1_tok_vals = [token_data[c]["S1_tokens"] for c in categories]
    s2_tok_vals = [token_data[c]["S2_tokens"] for c in categories]
    scatter_colors = [COLORS['system1_light'], COLORS['neutral'], COLORS['system2_light']]

    for ci, (cat, color) in enumerate(zip(cat_labels, scatter_colors)):
        ax2.scatter(s1_tok_vals[ci], s1_acc[ci], color=COLORS['system1'],
                    marker='o', s=30, zorder=3)
        ax2.scatter(s2_tok_vals[ci], s2_acc[ci], color=COLORS['system2'],
                    marker='s', s=30, zorder=3)
        ax2.annotate(cat, (s2_tok_vals[ci], s2_acc[ci]),
                     textcoords="offset points", xytext=(4, 2), fontsize=5.5)

    # Arrow from S1 to S2 per category
    for ci in range(len(categories)):
        ax2.annotate("", xy=(s2_tok_vals[ci], s2_acc[ci]),
                     xytext=(s1_tok_vals[ci], s1_acc[ci]),
                     arrowprops=dict(arrowstyle='->', color=COLORS['neutral'],
                                     lw=0.7))

    s1_patch = mpatches.Patch(color=COLORS['system1'], label='S1')
    s2_patch = mpatches.Patch(color=COLORS['system2'], label='S2')
    ax2.legend(handles=[s1_patch, s2_patch], fontsize=5.5, loc='lower right', frameon=False)
    ax2.set_xlabel("Avg. Tokens Used")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs. Tokens", fontsize=8)
    ax2.yaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax2.xaxis.grid(True, linewidth=0.3, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle("Fig. 6: Token Efficiency Analysis", fontsize=8, y=1.01)
    fig.tight_layout()

    out_file = output_path / "fig6_token_efficiency.png"
    fig.savefig(out_file, dpi=IEEE_DPI, bbox_inches='tight')
    print(f"Saved {out_file}")
    return fig


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

def generate_all_figures(
    input_dir: Optional[str] = None,
    output_dir: str = "IEEE_manuscript/figures/",
    use_sample: bool = False,
) -> None:
    """Generate all figures for the paper."""
    setup_ieee_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_sample:
        data = get_sample_data()
    else:
        if input_dir is None:
            print("ERROR: --input is required when not using --sample.", file=sys.stderr)
            sys.exit(1)
        data = load_results(input_dir)

    plot_factorial_ablation(data.get("factorial"), output_path)
    plot_main_effects(data.get("factorial"), output_path)
    plot_multi_model(data.get("multi_model"), output_path)
    plot_human_comparison(data.get("human_comparison"), output_path)
    plot_token_efficiency(data.get("token_efficiency"), output_path)

    print(f"All figures saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate IEEE-compliant publication figures for BIBM 2026 paper."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Directory containing experiment result JSON files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="IEEE_manuscript/figures/",
        help="Directory where figures will be saved (default: IEEE_manuscript/figures/).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use built-in sample data instead of loading from --input.",
    )
    args = parser.parse_args()

    generate_all_figures(
        input_dir=args.input,
        output_dir=args.output,
        use_sample=args.sample,
    )


if __name__ == "__main__":
    main()
