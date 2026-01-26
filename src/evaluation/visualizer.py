"""
Dual Process Theory Visualizer - Publication Quality
Professional, aesthetically pleasing scientific visualizations

Design Philosophy:
- Clean, minimalist design inspired by Nature/Science publications
- Carefully curated color palettes with scientific credibility
- Proper statistical annotations and effect size reporting
- High-contrast, accessible color schemes
- Elegant typography and spacing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ==================== Professional Color Palettes ====================

# Nature-inspired scientific palette
NATURE_PALETTE = {
    "system1": "#E64B35",      # Nature Red - Intuitive/Fast
    "system2": "#4DBBD5",      # Nature Cyan - Analytical/Slow  
    "system1_light": "#F4A582",
    "system2_light": "#92C5DE",
    "accent": "#00A087",       # Nature Green
    "warning": "#F39B7F",      # Soft coral
    "neutral": "#8C8C8C",      # Neutral gray
    "dark": "#3C5488",         # Deep blue
    "correct": "#91D1C2",      # Soft teal
    "incorrect": "#E64B35",    # Matching red
    "background": "#FAFAFA",   # Off-white
    "grid": "#E5E5E5",         # Light grid
    "text": "#2D2D2D",         # Dark text
    "text_light": "#666666"    # Secondary text
}

# Alternative: Cell/Science palette
SCIENCE_PALETTE = {
    "system1": "#DC3912",      # Google Red
    "system2": "#3366CC",      # Google Blue
    "system1_light": "#F4A582",
    "system2_light": "#92C5DE", 
    "accent": "#109618",       # Google Green
    "warning": "#FF9900",      # Orange
    "neutral": "#999999",
    "dark": "#22313F",
    "correct": "#109618",
    "incorrect": "#DC3912",
    "background": "#FFFFFF",
    "grid": "#EEEEEE",
    "text": "#333333",
    "text_light": "#777777"
}

# Elegant gradient colors for advanced visualizations
GRADIENT_SYSTEM1 = ["#FFEAA7", "#FDCB6E", "#F39C12", "#E74C3C", "#C0392B"]
GRADIENT_SYSTEM2 = ["#DFE6E9", "#B2BEC3", "#74B9FF", "#0984E3", "#2D3436"]


def create_custom_cmap(colors: List[str], name: str = "custom") -> LinearSegmentedColormap:
    """Create custom colormap from color list"""
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def setup_publication_style():
    """Setup matplotlib style for high-quality scientific publications"""
    
    # Reset to default first
    plt.rcdefaults()
    
    # Use a clean base style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Publication-quality settings
    plt.rcParams.update({
        # Figure
        'figure.figsize': (10, 7),
        'figure.dpi': 150,
        'figure.facecolor': '#FAFAFA',
        'figure.edgecolor': 'none',
        'figure.autolayout': False,
        
        # Saving
        'savefig.dpi': 300,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font - Use professional fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'font.weight': 'normal',
        
        # Axes
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelsize': 12,
        'axes.labelweight': 'medium',
        'axes.labelpad': 8,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#CCCCCC',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.7,
        
        # Ticks
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.major.width': 1,
        'ytick.major.width': 1,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        'legend.borderpad': 0.5,
        'legend.labelspacing': 0.5,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        
        # Patches (bars, etc.)
        'patch.linewidth': 1.0,
        'patch.edgecolor': 'white',
    })


def add_significance_annotation(ax, x1, x2, y, p_value, height=0.02):
    """Add significance bracket and stars to plot"""
    
    # Determine significance level
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."
    
    # Draw bracket
    bracket_height = height * (ax.get_ylim()[1] - ax.get_ylim()[0])
    y_bracket = y + bracket_height * 0.5
    
    ax.plot([x1, x1, x2, x2], 
            [y, y_bracket, y_bracket, y], 
            color='#333333', linewidth=1.2, clip_on=False)
    
    # Add text
    ax.text((x1 + x2) / 2, y_bracket + bracket_height * 0.3, 
            sig_text, ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='#333333')


def add_effect_size_annotation(ax, cohens_d, x, y):
    """Add effect size annotation"""
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
        color = "#999999"
    elif abs_d < 0.5:
        interpretation = "small"
        color = "#F39B7F"
    elif abs_d < 0.8:
        interpretation = "medium"
        color = "#F39C12"
    else:
        interpretation = "large"
        color = "#E64B35"
    
    text = f"d = {cohens_d:.2f}\n({interpretation})"
    ax.annotate(text, xy=(x, y), fontsize=9, ha='center', va='bottom',
                color=color, fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, alpha=0.9))


def save_figure(fig, filename: str, output_dir: Path, formats: List[str] = None):
    """Save figure in PNG format with high quality"""
    if formats is None:
        formats = ["png"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', dpi=300,
                   pad_inches=0.15)
    
    print(f"  ✓ Saved: {filename}")


class DualProcessVisualizer:
    """
    Publication-quality visualizer for Dual Process Theory experiments
    
    Generates beautiful, scientifically rigorous figures suitable for
    top-tier venues like Nature, Science, ICLR, NeurIPS.
    """
    
    def __init__(self, output_dir: str = None, palette: str = "nature"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Output directory for figures
            palette: Color palette ("nature", "science")
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "figures"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup style
        setup_publication_style()
        
        # Select palette
        self.colors = NATURE_PALETTE if palette == "nature" else SCIENCE_PALETTE
        self.generated_figures = []
    
    def generate_all_figures(self, experiment_results: Dict, save: bool = True) -> List[plt.Figure]:
        """
        Generate all visualization figures from experiment results
        
        Args:
            experiment_results: Complete experiment results dictionary
            save: Whether to save figures to disk
            
        Returns:
            List of generated figures
        """
        print("\n" + "="*50)
        print("🎨 Generating Publication-Quality Visualizations")
        print("="*50)
        
        figures = []
        
        # Extract data
        s1 = experiment_results.get('system1_results', {})
        s2 = experiment_results.get('system2_results', {})
        categories = experiment_results.get('category_analysis', {})
        comparison = experiment_results.get('comparison', {})
        
        # 1. Main Accuracy Comparison (Hero Figure)
        fig1 = self.plot_accuracy_comparison_hero(s1, s2, categories, comparison, save)
        figures.append(fig1)
        
        # 2. Response Time Distribution
        if 'results' in s1 and 'results' in s2:
            s1_times = [r.get('response_time_ms', 0) for r in s1['results']]
            s2_times = [r.get('response_time_ms', 0) for r in s2['results']]
            if any(s1_times) and any(s2_times):
                fig2 = self.plot_response_time_analysis(s1_times, s2_times, save)
                figures.append(fig2)
        
        # 3. Speed-Accuracy Tradeoff
        fig3 = self.plot_speed_accuracy_elegant(s1, s2, save)
        figures.append(fig3)
        
        # 4. Cognitive Profile Radar
        fig4 = self.plot_cognitive_profile_radar(s1, s2, save)
        figures.append(fig4)
        
        # 5. Category Performance Heatmap
        if categories:
            fig5 = self.plot_category_heatmap_elegant(categories, save)
            figures.append(fig5)
        
        # 6. Confidence Calibration
        if 'results' in s1 and 'results' in s2:
            fig6 = self.plot_confidence_calibration(s1['results'], s2['results'], save)
            figures.append(fig6)
        
        # 7. Error Analysis
        if 'results' in s1 and 'results' in s2:
            fig7 = self.plot_error_analysis_elegant(s1['results'], s2['results'], categories, save)
            figures.append(fig7)
        
        # Note: Statistical Summary (fig8) removed - not needed
        
        # 8. Token Efficiency Analysis (renumbered from 9)
        fig8 = self.plot_efficiency_analysis(s1, s2, save)
        figures.append(fig8)
        
        # Note: Dashboard (fig10) removed - not needed
        
        self.generated_figures = figures
        print(f"\n✅ Generated {len(figures)} publication-quality figures")
        
        return figures
    
    def plot_accuracy_comparison_hero(self, s1: Dict, s2: Dict, 
                                      categories: Dict, comparison: Dict,
                                      save: bool = True) -> plt.Figure:
        """
        Hero figure: Main accuracy comparison with statistical annotations
        """
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], 
                     hspace=0.35, wspace=0.3)
        
        # === Main comparison (top, spans 2 columns) ===
        ax_main = fig.add_subplot(gs[0, :2])
        
        s1_acc = s1.get('accuracy', 0)
        s2_acc = s2.get('accuracy', 0)
        
        # Create elegant bars with gradient effect
        x = np.array([0, 1])
        bars = ax_main.bar(x, [s1_acc, s2_acc], width=0.5,
                          color=[self.colors['system1'], self.colors['system2']],
                          edgecolor='white', linewidth=2.5,
                          zorder=3)
        
        # Add subtle shadow effect
        for bar in bars:
            shadow = mpatches.FancyBboxPatch(
                (bar.get_x() + 0.02, 0), bar.get_width(), bar.get_height(),
                boxstyle="round,pad=0.01", facecolor='gray', alpha=0.15,
                zorder=1
            )
            ax_main.add_patch(shadow)
        
        # Add value labels with elegant styling
        for bar, acc, color in zip(bars, [s1_acc, s2_acc], 
                                   [self.colors['system1'], self.colors['system2']]):
            height = bar.get_height()
            ax_main.annotate(f'{acc:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 8), textcoords="offset points",
                           ha='center', va='bottom', fontsize=16, fontweight='bold',
                           color=self.colors['text'])
        
        # Styling
        ax_main.set_xticks(x)
        ax_main.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'],
                               fontsize=12, fontweight='medium')
        ax_main.set_ylabel('Accuracy', fontsize=12, fontweight='medium')
        ax_main.set_ylim(0, 1.18)
        ax_main.set_xlim(-0.5, 1.5)
        
        # Add chance level line
        ax_main.axhline(y=0.5, color=self.colors['neutral'], linestyle='--', 
                       alpha=0.5, linewidth=1.5, label='Chance Level', zorder=2)
        
        # Add significance annotation if available
        p_val = comparison.get('statistical_test', {}).get('p_value', 1.0)
        if p_val < 1.0:
            max_height = max(s1_acc, s2_acc)
            add_significance_annotation(ax_main, 0, 1, max_height + 0.05, p_val)
        
        ax_main.set_title('Overall Performance Comparison', 
                         fontsize=14, fontweight='bold', pad=15)
        ax_main.legend(loc='upper right', framealpha=0.9)
        
        # === Effect size indicator (top right) ===
        ax_effect = fig.add_subplot(gs[0, 2])
        
        # Calculate effect size
        acc_diff = s2_acc - s1_acc
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        ax_effect.plot(r * np.cos(theta), r * np.sin(theta), 
                      color=self.colors['grid'], linewidth=20, solid_capstyle='round')
        
        # Colored arc based on difference
        if acc_diff != 0:
            fill_ratio = min(abs(acc_diff) / 0.3, 1)  # Normalize to 30% max
            theta_fill = np.linspace(0, np.pi * fill_ratio, 50)
            color = self.colors['correct'] if acc_diff > 0 else self.colors['incorrect']
            ax_effect.plot(r * np.cos(theta_fill), r * np.sin(theta_fill),
                          color=color, linewidth=18, solid_capstyle='round')
        
        # Center text
        ax_effect.text(0, 0.3, f'{acc_diff:+.1%}', ha='center', va='center',
                      fontsize=24, fontweight='bold', color=self.colors['text'])
        ax_effect.text(0, -0.1, 'Accuracy\nDifference', ha='center', va='center',
                      fontsize=10, color=self.colors['text_light'])
        
        ax_effect.set_xlim(-1.3, 1.3)
        ax_effect.set_ylim(-0.3, 1.3)
        ax_effect.axis('off')
        ax_effect.set_title('S2 vs S1', fontsize=12, fontweight='bold', pad=10)
        
        # === Category breakdown (bottom) ===
        ax_cat = fig.add_subplot(gs[1, :])
        
        if categories:
            cat_names = list(categories.keys())
            x = np.arange(len(cat_names))
            width = 0.35
            
            s1_accs = [categories[cat].get('system1', {}).get('accuracy', 0) for cat in cat_names]
            s2_accs = [categories[cat].get('system2', {}).get('accuracy', 0) for cat in cat_names]
            
            bars1 = ax_cat.bar(x - width/2, s1_accs, width, label='System 1',
                              color=self.colors['system1'], edgecolor='white', 
                              linewidth=1.5, zorder=3)
            bars2 = ax_cat.bar(x + width/2, s2_accs, width, label='System 2',
                              color=self.colors['system2'], edgecolor='white',
                              linewidth=1.5, zorder=3)
            
            # Add value labels
            for bars, accs in [(bars1, s1_accs), (bars2, s2_accs)]:
                for bar, acc in zip(bars, accs):
                    if acc > 0:
                        ax_cat.annotate(f'{acc:.0%}',
                                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                                       xytext=(0, 3), textcoords="offset points",
                                       ha='center', va='bottom', fontsize=9,
                                       fontweight='medium', color=self.colors['text'])
            
            # Clean category names
            clean_names = []
            for c in cat_names:
                name = c.replace('_tasks', '').replace('_', ' ').title()
                clean_names.append(name)
            
            ax_cat.set_xticks(x)
            ax_cat.set_xticklabels(clean_names, fontsize=11)
            ax_cat.set_ylabel('Accuracy', fontsize=11)
            ax_cat.set_ylim(0, 1.15)
            ax_cat.set_title('Performance by Task Category', fontsize=12, fontweight='bold', pad=10)
            ax_cat.legend(loc='upper right', ncol=2)
            ax_cat.axhline(y=0.5, color=self.colors['neutral'], linestyle='--', 
                          alpha=0.4, linewidth=1, zorder=2)
        
        plt.suptitle('Dual Process Theory: System Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        if save:
            save_figure(fig, "01_accuracy_comparison", self.output_dir)
        
        return fig
    
    def plot_response_time_analysis(self, s1_times: List[float], 
                                    s2_times: List[float], save: bool = True) -> plt.Figure:
        """
        Elegant response time analysis with multiple views
        """
        # Filter valid times
        s1_times = [t for t in s1_times if t > 0]
        s2_times = [t for t in s2_times if t > 0]
        
        if not s1_times or not s2_times:
            return None
        
        fig = plt.figure(figsize=(16, 6))
        gs = GridSpec(1, 3, figure=fig, wspace=0.3)
        
        # === Violin + Strip plot ===
        ax1 = fig.add_subplot(gs[0])
        
        # Prepare data for seaborn
        import pandas as pd
        data = pd.DataFrame({
            'Response Time (ms)': s1_times + s2_times,
            'System': ['System 1'] * len(s1_times) + ['System 2'] * len(s2_times)
        })
        
        # Create violin plot
        palette = {'System 1': self.colors['system1'], 'System 2': self.colors['system2']}
        
        parts = sns.violinplot(data=data, x='System', y='Response Time (ms)',
                              palette=palette, ax=ax1, inner=None, alpha=0.7)
        
        # Add strip plot for individual points
        sns.stripplot(data=data, x='System', y='Response Time (ms)',
                     palette=palette, ax=ax1, alpha=0.4, size=3, jitter=0.2)
        
        # Add mean markers
        means = [np.mean(s1_times), np.mean(s2_times)]
        ax1.scatter([0, 1], means, color='white', s=100, zorder=5, 
                   edgecolor='black', linewidth=2, marker='D')
        
        ax1.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('')
        
        # === Histogram with KDE ===
        ax2 = fig.add_subplot(gs[1])
        
        # Plot histograms
        bins = np.linspace(0, max(max(s1_times), max(s2_times)) * 1.1, 40)
        
        ax2.hist(s1_times, bins=bins, alpha=0.6, color=self.colors['system1'],
                label=f'S1 (μ={np.mean(s1_times):.0f}ms)', density=True, edgecolor='white')
        ax2.hist(s2_times, bins=bins, alpha=0.6, color=self.colors['system2'],
                label=f'S2 (μ={np.mean(s2_times):.0f}ms)', density=True, edgecolor='white')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        if len(s1_times) > 2:
            kde1 = gaussian_kde(s1_times)
            x_range = np.linspace(min(s1_times), max(s1_times), 100)
            ax2.plot(x_range, kde1(x_range), color=self.colors['system1'], 
                    linewidth=2.5, linestyle='-')
        if len(s2_times) > 2:
            kde2 = gaussian_kde(s2_times)
            x_range = np.linspace(min(s2_times), max(s2_times), 100)
            ax2.plot(x_range, kde2(x_range), color=self.colors['system2'],
                    linewidth=2.5, linestyle='-')
        
        ax2.set_xlabel('Response Time (ms)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Response Time Histogram', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', framealpha=0.9)
        
        # === CDF comparison ===
        ax3 = fig.add_subplot(gs[2])
        
        for times, color, label in [(s1_times, self.colors['system1'], 'System 1'),
                                    (s2_times, self.colors['system2'], 'System 2')]:
            sorted_times = np.sort(times)
            cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            ax3.plot(sorted_times, cumulative, color=color, linewidth=2.5, label=label)
            ax3.fill_between(sorted_times, cumulative, alpha=0.15, color=color)
        
        # Add median lines
        ax3.axhline(y=0.5, color=self.colors['neutral'], linestyle=':', alpha=0.5)
        ax3.axvline(x=np.median(s1_times), color=self.colors['system1'], 
                   linestyle='--', alpha=0.7, linewidth=1.5)
        ax3.axvline(x=np.median(s2_times), color=self.colors['system2'],
                   linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax3.set_xlabel('Response Time (ms)', fontsize=11)
        ax3.set_ylabel('Cumulative Probability', fontsize=11)
        ax3.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', framealpha=0.9)
        
        # Add ratio annotation
        ratio = np.mean(s2_times) / np.mean(s1_times) if np.mean(s1_times) > 0 else 0
        fig.text(0.5, 0.02, f'System 2 is {ratio:.1f}× slower than System 1',
                ha='center', fontsize=11, style='italic', color=self.colors['text_light'])
        
        plt.suptitle('Response Time Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            save_figure(fig, "02_response_time_distribution", self.output_dir)
        
        return fig
    
    def plot_speed_accuracy_elegant(self, s1: Dict, s2: Dict, save: bool = True) -> plt.Figure:
        """
        Elegant speed-accuracy tradeoff visualization
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        s1_acc = s1.get('accuracy', 0)
        s2_acc = s2.get('accuracy', 0)
        s1_time = s1.get('avg_response_time_ms', 100)
        s2_time = s2.get('avg_response_time_ms', 200)
        
        # Ensure non-zero times
        if s1_time == 0: s1_time = 100
        if s2_time == 0: s2_time = 200
        
        # Create background gradient regions
        x_range = np.linspace(0, max(s1_time, s2_time) * 1.5, 100)
        y_range = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Efficiency = Accuracy / log(Time) - higher is better
        Z = Y / np.log10(X + 10)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=20, cmap='YlGnBu', alpha=0.3)
        
        # Plot connecting line with gradient
        ax.plot([s1_time, s2_time], [s1_acc, s2_acc],
               color=self.colors['neutral'], linestyle='--', linewidth=2, 
               alpha=0.6, zorder=2)
        
        # Plot System 1 point
        ax.scatter(s1_time, s1_acc, s=400, c=self.colors['system1'],
                  edgecolors='white', linewidth=3, zorder=5, marker='o',
                  label='System 1 (Intuitive)')
        
        # Plot System 2 point
        ax.scatter(s2_time, s2_acc, s=400, c=self.colors['system2'],
                  edgecolors='white', linewidth=3, zorder=5, marker='s',
                  label='System 2 (Analytical)')
        
        # Add annotations with elegant boxes
        ax.annotate(f'S1\n{s1_acc:.0%}\n{s1_time:.0f}ms',
                   xy=(s1_time, s1_acc),
                   xytext=(s1_time - (s2_time - s1_time) * 0.3, s1_acc - 0.12),
                   fontsize=10, ha='center', fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=self.colors['system1_light'],
                            edgecolor=self.colors['system1'], alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color=self.colors['system1'], lw=1.5))
        
        ax.annotate(f'S2\n{s2_acc:.0%}\n{s2_time:.0f}ms',
                   xy=(s2_time, s2_acc),
                   xytext=(s2_time + (s2_time - s1_time) * 0.2, s2_acc + 0.08),
                   fontsize=10, ha='center', fontweight='medium',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=self.colors['system2_light'],
                            edgecolor=self.colors['system2'], alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color=self.colors['system2'], lw=1.5))
        
        # Add quadrant labels
        mid_time = (s1_time + s2_time) / 2
        ax.text(s1_time * 0.5, 0.85, 'Fast &\nAccurate', ha='center', va='center',
               fontsize=9, color=self.colors['correct'], alpha=0.8, fontweight='medium')
        ax.text(s2_time * 1.2, 0.85, 'Slow &\nAccurate', ha='center', va='center',
               fontsize=9, color=self.colors['accent'], alpha=0.8, fontweight='medium')
        ax.text(s1_time * 0.5, 0.15, 'Fast &\nInaccurate', ha='center', va='center',
               fontsize=9, color=self.colors['warning'], alpha=0.8, fontweight='medium')
        ax.text(s2_time * 1.2, 0.15, 'Slow &\nInaccurate', ha='center', va='center',
               fontsize=9, color=self.colors['incorrect'], alpha=0.8, fontweight='medium')
        
        # Styling
        ax.set_xlabel('Response Time (ms)', fontsize=12, fontweight='medium')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='medium')
        ax.set_title('Speed-Accuracy Tradeoff\n', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(s1_time, s2_time) * 1.4)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        
        # Add colorbar for efficiency
        cbar = plt.colorbar(contour, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Efficiency Index', fontsize=10)
        
        if save:
            save_figure(fig, "03_speed_accuracy_tradeoff", self.output_dir)
        
        return fig
    
    def plot_cognitive_profile_radar(self, s1: Dict, s2: Dict, save: bool = True) -> plt.Figure:
        """
        Elegant radar chart for cognitive profile comparison
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        categories = ['Accuracy', 'Speed\n(inverse)', 'Confidence', 
                     'Efficiency\n(acc/token)', 'Reasoning\nDepth']
        N = len(categories)
        
        # Normalize metrics
        max_time = max(s1.get('avg_response_time_ms', 1), s2.get('avg_response_time_ms', 1), 1)
        max_tokens = max(s1.get('avg_tokens_used', 1), s2.get('avg_tokens_used', 1), 1)
        max_steps = max(s1.get('avg_reasoning_steps', 1), s2.get('avg_reasoning_steps', 1), 1)
        
        s1_values = [
            s1.get('accuracy', 0),
            1 - (s1.get('avg_response_time_ms', 0) / max_time),  # Inverse for speed
            s1.get('avg_confidence', 0.5),
            s1.get('accuracy', 0) / max(s1.get('avg_tokens_used', 1) / 1000, 0.1),
            s1.get('avg_reasoning_steps', 0) / max(max_steps, 1)
        ]
        
        s2_values = [
            s2.get('accuracy', 0),
            1 - (s2.get('avg_response_time_ms', 0) / max_time),
            s2.get('avg_confidence', 0.5),
            s2.get('accuracy', 0) / max(s2.get('avg_tokens_used', 1) / 1000, 0.1),
            s2.get('avg_reasoning_steps', 0) / max(max_steps, 1)
        ]
        
        # Normalize efficiency to 0-1
        max_eff = max(s1_values[3], s2_values[3], 0.1)
        s1_values[3] = s1_values[3] / max_eff
        s2_values[3] = s2_values[3] / max_eff
        
        # Angles
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        
        # Close the polygon
        s1_values += s1_values[:1]
        s2_values += s2_values[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, s1_values, 'o-', linewidth=2.5, 
               color=self.colors['system1'], label='System 1', markersize=8)
        ax.fill(angles, s1_values, alpha=0.2, color=self.colors['system1'])
        
        ax.plot(angles, s2_values, 's-', linewidth=2.5,
               color=self.colors['system2'], label='System 2', markersize=8)
        ax.fill(angles, s2_values, alpha=0.2, color=self.colors['system2'])
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight='medium')
        
        # Set radial limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color=self.colors['text_light'])
        
        # Style the grid
        ax.grid(True, linestyle='-', alpha=0.3)
        
        # Title and legend
        ax.set_title('Cognitive Profile Comparison\n', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
        
        if save:
            save_figure(fig, "04_cognitive_effort_radar", self.output_dir)
        
        return fig
    
    def plot_category_heatmap_elegant(self, categories: Dict, save: bool = True) -> plt.Figure:
        """
        Elegant heatmap for category performance
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        task_types = list(categories.keys())
        systems = ['System 1\n(Intuitive)', 'System 2\n(Analytical)']
        
        # Create data matrix
        data = np.array([
            [categories[t].get('system1', {}).get('accuracy', 0) for t in task_types],
            [categories[t].get('system2', {}).get('accuracy', 0) for t in task_types]
        ])
        
        # Create custom colormap
        cmap = create_custom_cmap(['#FFFFFF', '#C8E6C9', '#81C784', '#4CAF50', '#2E7D32'], 'accuracy')
        
        # Plot heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Add cell annotations with adaptive text color
        for i in range(len(systems)):
            for j in range(len(task_types)):
                value = data[i, j]
                text_color = 'white' if value > 0.6 else self.colors['text']
                ax.text(j, i, f'{value:.0%}',
                       ha='center', va='center', color=text_color,
                       fontsize=14, fontweight='bold')
        
        # Clean category names
        clean_names = [t.replace('_tasks', '').replace('_', ' ').title() for t in task_types]
        
        # Set ticks
        ax.set_xticks(np.arange(len(task_types)))
        ax.set_yticks(np.arange(len(systems)))
        ax.set_xticklabels(clean_names, fontsize=11, fontweight='medium')
        ax.set_yticklabels(systems, fontsize=11, fontweight='medium')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Accuracy', fontsize=11, fontweight='medium')
        cbar.ax.tick_params(labelsize=10)
        
        # Add grid lines
        ax.set_xticks(np.arange(len(task_types) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(systems) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        ax.set_title('Performance Heatmap by Task Category', fontsize=14, fontweight='bold', pad=15)
        
        # Add difference annotation below
        diffs = data[1] - data[0]
        diff_text = "S2-S1 Difference: " + " | ".join([f"{clean_names[i]}: {d:+.0%}" for i, d in enumerate(diffs)])
        fig.text(0.5, 0.02, diff_text, ha='center', fontsize=10, 
                style='italic', color=self.colors['text_light'])
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, "05_task_type_heatmap", self.output_dir)
        
        return fig
    
    def plot_confidence_calibration(self, s1_results: List[Dict], 
                                    s2_results: List[Dict], save: bool = True) -> plt.Figure:
        """
        Elegant confidence calibration plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for ax, results, name, color, light_color in zip(
            axes, [s1_results, s2_results], 
            ['System 1 (Intuitive)', 'System 2 (Analytical)'],
            [self.colors['system1'], self.colors['system2']],
            [self.colors['system1_light'], self.colors['system2_light']]
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
            
            # Add ECE annotation
            ax.text(0.05, 0.95, f'ECE = {ece:.3f}', transform=ax.transAxes,
                   fontsize=11, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.9))
            
            ax.set_xlabel('Confidence', fontsize=11, fontweight='medium')
            ax.set_ylabel('Accuracy', fontsize=11, fontweight='medium')
            ax.set_title(f'{name}\nCalibration', fontsize=12, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(loc='lower right', fontsize=9)
            ax.set_aspect('equal')
        
        plt.suptitle('Confidence Calibration Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            save_figure(fig, "06_confidence_accuracy_analysis", self.output_dir)
        
        return fig
    
    def plot_error_analysis_elegant(self, s1_results: List[Dict], s2_results: List[Dict],
                                    categories: Dict, save: bool = True) -> plt.Figure:
        """
        Elegant error analysis visualization
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Calculate error statistics
        s1_errors = sum(1 for r in s1_results if not r.get('is_correct', True))
        s2_errors = sum(1 for r in s2_results if not r.get('is_correct', True))
        s1_correct = len(s1_results) - s1_errors
        s2_correct = len(s2_results) - s2_errors
        
        # === 1. Correct vs Incorrect (Stacked) ===
        ax1 = fig.add_subplot(gs[0, 0])
        
        x = np.array([0, 1])
        width = 0.5
        
        # Stacked bars
        ax1.bar(x, [s1_correct, s2_correct], width, label='Correct',
               color=self.colors['correct'], edgecolor='white', linewidth=1.5)
        ax1.bar(x, [s1_errors, s2_errors], width, bottom=[s1_correct, s2_correct],
               label='Incorrect', color=self.colors['incorrect'], edgecolor='white', linewidth=1.5)
        
        # Add percentage labels
        for i, (correct, error) in enumerate([(s1_correct, s1_errors), (s2_correct, s2_errors)]):
            total = correct + error
            if total > 0:
                ax1.text(i, correct/2, f'{correct}\n({correct/total:.0%})', 
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
                if error > 0:
                    ax1.text(i, correct + error/2, f'{error}\n({error/total:.0%})',
                            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(['System 1', 'System 2'], fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Response Outcomes', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        
        # === 2. Error Rate by Category ===
        ax2 = fig.add_subplot(gs[0, 1])
        
        if categories:
            cat_names = list(categories.keys())
            clean_names = [c.replace('_tasks', '').replace('_', ' ').title() for c in cat_names]
            
            s1_error_rates = [1 - categories[c].get('system1', {}).get('accuracy', 0) for c in cat_names]
            s2_error_rates = [1 - categories[c].get('system2', {}).get('accuracy', 0) for c in cat_names]
            
            x = np.arange(len(cat_names))
            width = 0.35
            
            ax2.bar(x - width/2, s1_error_rates, width, label='System 1',
                   color=self.colors['system1'], edgecolor='white', linewidth=1.5)
            ax2.bar(x + width/2, s2_error_rates, width, label='System 2',
                   color=self.colors['system2'], edgecolor='white', linewidth=1.5)
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(clean_names, rotation=15, ha='right', fontsize=10)
            ax2.set_ylabel('Error Rate', fontsize=11)
            ax2.set_title('Error Rate by Category', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, max(max(s1_error_rates), max(s2_error_rates)) * 1.2 + 0.05)
        
        # === 3. Confidence Distribution for Errors ===
        ax3 = fig.add_subplot(gs[0, 2])
        
        s1_error_conf = [r.get('confidence', 0.5) for r in s1_results if not r.get('is_correct', True)]
        s2_error_conf = [r.get('confidence', 0.5) for r in s2_results if not r.get('is_correct', True)]
        
        if s1_error_conf or s2_error_conf:
            data_to_plot = []
            labels = []
            colors_box = []
            
            if s1_error_conf:
                data_to_plot.append(s1_error_conf)
                labels.append('S1 Errors')
                colors_box.append(self.colors['system1'])
            if s2_error_conf:
                data_to_plot.append(s2_error_conf)
                labels.append('S2 Errors')
                colors_box.append(self.colors['system2'])
            
            bp = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax3.set_ylabel('Confidence', fontsize=11)
            ax3.set_title('Error Confidence Distribution', fontsize=12, fontweight='bold')
            ax3.axhline(y=0.5, color=self.colors['neutral'], linestyle='--', alpha=0.5)
        
        # === 4. Error Type Breakdown (Donut Chart) ===
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Categorize errors by confidence level
        s1_overconf = sum(1 for r in s1_results if not r.get('is_correct', True) and r.get('confidence', 0.5) > 0.7)
        s1_underconf = sum(1 for r in s1_results if not r.get('is_correct', True) and r.get('confidence', 0.5) < 0.3)
        s1_calibrated = max(0, s1_errors - s1_overconf - s1_underconf)
        
        if s1_errors > 0:
            sizes = [s1_overconf, s1_calibrated, s1_underconf]
            labels = ['Overconfident', 'Calibrated', 'Underconfident']
            colors_pie = [self.colors['incorrect'], self.colors['neutral'], self.colors['system2']]
            explode = (0.05, 0, 0)
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie,
                                               autopct='%1.0f%%', startangle=90, explode=explode,
                                               wedgeprops=dict(width=0.6, edgecolor='white'))
            
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            ax4.set_title('System 1 Error Types', fontsize=12, fontweight='bold')
        
        # === 5. Error Type Breakdown S2 ===
        ax5 = fig.add_subplot(gs[1, 1])
        
        s2_overconf = sum(1 for r in s2_results if not r.get('is_correct', True) and r.get('confidence', 0.5) > 0.7)
        s2_underconf = sum(1 for r in s2_results if not r.get('is_correct', True) and r.get('confidence', 0.5) < 0.3)
        s2_calibrated = max(0, s2_errors - s2_overconf - s2_underconf)
        
        if s2_errors > 0:
            sizes = [s2_overconf, s2_calibrated, s2_underconf]
            labels = ['Overconfident', 'Calibrated', 'Underconfident']
            colors_pie = [self.colors['incorrect'], self.colors['neutral'], self.colors['system2']]
            explode = (0.05, 0, 0)
            
            wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors_pie,
                                               autopct='%1.0f%%', startangle=90, explode=explode,
                                               wedgeprops=dict(width=0.6, edgecolor='white'))
            
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            ax5.set_title('System 2 Error Types', fontsize=12, fontweight='bold')
        
        # === 6. Error Comparison Summary ===
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create summary text
        s1_error_rate = s1_errors / len(s1_results) if s1_results else 0
        s2_error_rate = s2_errors / len(s2_results) if s2_results else 0
        
        summary = f"""
        Error Analysis Summary
        ─────────────────────────
        
        System 1:
          • Total Errors: {s1_errors} / {len(s1_results)}
          • Error Rate: {s1_error_rate:.1%}
          • Overconfident: {s1_overconf} ({s1_overconf/max(s1_errors,1):.0%})
        
        System 2:
          • Total Errors: {s2_errors} / {len(s2_results)}
          • Error Rate: {s2_error_rate:.1%}
          • Overconfident: {s2_overconf} ({s2_overconf/max(s2_errors,1):.0%})
        
        Key Finding:
          {'S2 makes fewer errors' if s2_error_rate < s1_error_rate else 'S1 makes fewer errors'}
          (Δ = {abs(s2_error_rate - s1_error_rate):.1%})
        """
        
        ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'],
                         edgecolor=self.colors['grid']))
        
        plt.suptitle('Comprehensive Error Analysis', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            save_figure(fig, "07_error_analysis", self.output_dir)
        
        return fig
    
    def plot_statistical_summary(self, comparison: Dict, s1: Dict, s2: Dict, 
                                 save: bool = True) -> plt.Figure:
        """
        Elegant statistical summary visualization
        """
        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)
        
        # === 1. Accuracy Difference with CI ===
        ax1 = fig.add_subplot(gs[0, 0])
        
        acc_diff = comparison.get('accuracy_diff', 0)
        color = self.colors['correct'] if acc_diff > 0 else self.colors['incorrect']
        
        ax1.barh(['Accuracy\nDifference'], [acc_diff * 100], color=color,
                edgecolor='white', linewidth=2, height=0.5)
        ax1.axvline(x=0, color='black', linewidth=1)
        ax1.set_xlabel('Percentage Points')
        ax1.set_title('S2 - S1 Accuracy', fontsize=12, fontweight='bold')
        ax1.text(acc_diff * 100 + (2 if acc_diff > 0 else -2), 0, 
                f'{acc_diff:+.1%}', va='center', fontsize=12, fontweight='bold')
        
        # === 2. Response Time Ratio ===
        ax2 = fig.add_subplot(gs[0, 1])
        
        time_ratio = comparison.get('response_time_ratio', 1)
        
        # Create a gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        ax2.plot(r * np.cos(theta), r * np.sin(theta), 
                color=self.colors['grid'], linewidth=15, solid_capstyle='round')
        
        # Fill based on ratio (normalized to 10x max)
        fill_ratio = min(time_ratio / 10, 1)
        theta_fill = np.linspace(0, np.pi * fill_ratio, 50)
        ax2.plot(r * np.cos(theta_fill), r * np.sin(theta_fill),
                color=self.colors['accent'], linewidth=13, solid_capstyle='round')
        
        ax2.text(0, 0.3, f'{time_ratio:.1f}×', ha='center', va='center',
                fontsize=20, fontweight='bold', color=self.colors['text'])
        ax2.text(0, -0.1, 'Time Ratio\n(S2/S1)', ha='center', va='center',
                fontsize=9, color=self.colors['text_light'])
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.3, 1.2)
        ax2.axis('off')
        ax2.set_title('Response Time', fontsize=12, fontweight='bold')
        
        # === 3. Token Ratio ===
        ax3 = fig.add_subplot(gs[0, 2])
        
        token_ratio = comparison.get('tokens_ratio', 1)
        
        ax3.plot(r * np.cos(theta), r * np.sin(theta),
                color=self.colors['grid'], linewidth=15, solid_capstyle='round')
        
        fill_ratio = min(token_ratio / 10, 1)
        theta_fill = np.linspace(0, np.pi * fill_ratio, 50)
        ax3.plot(r * np.cos(theta_fill), r * np.sin(theta_fill),
                color=self.colors['warning'], linewidth=13, solid_capstyle='round')
        
        ax3.text(0, 0.3, f'{token_ratio:.1f}×', ha='center', va='center',
                fontsize=20, fontweight='bold', color=self.colors['text'])
        ax3.text(0, -0.1, 'Token Ratio\n(S2/S1)', ha='center', va='center',
                fontsize=9, color=self.colors['text_light'])
        
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-0.3, 1.2)
        ax3.axis('off')
        ax3.set_title('Token Usage', fontsize=12, fontweight='bold')
        
        # === 4. Statistical Test Result ===
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        stat_test = comparison.get('statistical_test', {})
        p_value = stat_test.get('p_value', 1.0)
        significant = stat_test.get('significant', False)
        
        # Significance indicator
        if p_value < 0.001:
            sig_text = "***"
            sig_color = self.colors['correct']
        elif p_value < 0.01:
            sig_text = "**"
            sig_color = self.colors['correct']
        elif p_value < 0.05:
            sig_text = "*"
            sig_color = self.colors['warning']
        else:
            sig_text = "n.s."
            sig_color = self.colors['neutral']
        
        ax4.text(0.5, 0.7, sig_text, ha='center', va='center', fontsize=36,
                fontweight='bold', color=sig_color, transform=ax4.transAxes)
        ax4.text(0.5, 0.4, f'p = {p_value:.4f}', ha='center', va='center',
                fontsize=12, color=self.colors['text'], transform=ax4.transAxes)
        ax4.text(0.5, 0.2, 'Significant' if significant else 'Not Significant',
                ha='center', va='center', fontsize=10, color=self.colors['text_light'],
                transform=ax4.transAxes)
        ax4.set_title('Statistical Test', fontsize=12, fontweight='bold')
        
        # === 5. Metrics Comparison Bar Chart ===
        ax5 = fig.add_subplot(gs[1, :2])
        
        metrics = ['Accuracy', 'Confidence', 'Response Time\n(normalized)', 'Tokens\n(normalized)']
        
        # Normalize time and tokens
        max_time = max(s1.get('avg_response_time_ms', 1), s2.get('avg_response_time_ms', 1))
        max_tokens = max(s1.get('avg_tokens_used', 1), s2.get('avg_tokens_used', 1))
        
        s1_vals = [
            s1.get('accuracy', 0),
            s1.get('avg_confidence', 0),
            s1.get('avg_response_time_ms', 0) / max_time,
            s1.get('avg_tokens_used', 0) / max_tokens
        ]
        
        s2_vals = [
            s2.get('accuracy', 0),
            s2.get('avg_confidence', 0),
            s2.get('avg_response_time_ms', 0) / max_time,
            s2.get('avg_tokens_used', 0) / max_tokens
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax5.bar(x - width/2, s1_vals, width, label='System 1',
               color=self.colors['system1'], edgecolor='white', linewidth=1.5)
        ax5.bar(x + width/2, s2_vals, width, label='System 2',
               color=self.colors['system2'], edgecolor='white', linewidth=1.5)
        
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics, fontsize=10)
        ax5.set_ylabel('Normalized Value', fontsize=11)
        ax5.set_title('Metrics Comparison', fontsize=12, fontweight='bold')
        ax5.legend(loc='upper right')
        ax5.set_ylim(0, 1.15)
        
        # === 6. Summary Statistics Table ===
        ax6 = fig.add_subplot(gs[1, 2:])
        ax6.axis('off')
        
        # Create table data
        table_data = [
            ['Metric', 'System 1', 'System 2', 'Difference'],
            ['Accuracy', f"{s1.get('accuracy', 0):.1%}", f"{s2.get('accuracy', 0):.1%}", 
             f"{comparison.get('accuracy_diff', 0):+.1%}"],
            ['Confidence', f"{s1.get('avg_confidence', 0):.2f}", f"{s2.get('avg_confidence', 0):.2f}",
             f"{s2.get('avg_confidence', 0) - s1.get('avg_confidence', 0):+.2f}"],
            ['Response Time', f"{s1.get('avg_response_time_ms', 0):.0f}ms", 
             f"{s2.get('avg_response_time_ms', 0):.0f}ms",
             f"{comparison.get('response_time_ratio', 1):.1f}×"],
            ['Tokens Used', f"{s1.get('avg_tokens_used', 0):.0f}", 
             f"{s2.get('avg_tokens_used', 0):.0f}",
             f"{comparison.get('tokens_ratio', 1):.1f}×"]
        ]
        
        table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                         colWidths=[0.25, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor(self.colors['dark'])
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        # Alternate row colors
        for i in range(1, 5):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor(self.colors['background'])
        
        ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Statistical Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
        
        if save:
            save_figure(fig, "08_statistical_comparison", self.output_dir)
        
        return fig
    
    def plot_efficiency_analysis(self, s1: Dict, s2: Dict, save: bool = True) -> plt.Figure:
        """
        Token and computational efficiency analysis
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # === 1. Token Usage Comparison ===
        ax1 = axes[0]
        
        metrics = ['Avg Tokens', 'Total Calls']
        s1_vals = [s1.get('avg_tokens_used', 0), s1.get('total_api_calls', 0)]
        s2_vals = [s2.get('avg_tokens_used', 0), s2.get('total_api_calls', 0)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, s1_vals, width, label='System 1',
               color=self.colors['system1'], edgecolor='white', linewidth=1.5)
        ax1.bar(x + width/2, s2_vals, width, label='System 2',
               color=self.colors['system2'], edgecolor='white', linewidth=1.5)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Resource Usage', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        
        # === 2. Efficiency Metrics ===
        ax2 = axes[1]
        
        # Calculate efficiency metrics
        s1_acc_per_token = s1.get('accuracy', 0) / max(s1.get('avg_tokens_used', 1), 1) * 1000
        s2_acc_per_token = s2.get('accuracy', 0) / max(s2.get('avg_tokens_used', 1), 1) * 1000
        
        s1_acc_per_ms = s1.get('accuracy', 0) / max(s1.get('avg_response_time_ms', 1), 1) * 1000
        s2_acc_per_ms = s2.get('accuracy', 0) / max(s2.get('avg_response_time_ms', 1), 1) * 1000
        
        metrics = ['Accuracy per\n1K Tokens', 'Accuracy per\nSecond']
        s1_effs = [s1_acc_per_token, s1_acc_per_ms]
        s2_effs = [s2_acc_per_token, s2_acc_per_ms]
        
        x = np.arange(len(metrics))
        
        ax2.bar(x - width/2, s1_effs, width, label='System 1',
               color=self.colors['system1'], edgecolor='white', linewidth=1.5)
        ax2.bar(x + width/2, s2_effs, width, label='System 2',
               color=self.colors['system2'], edgecolor='white', linewidth=1.5)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, fontsize=11)
        ax2.set_ylabel('Efficiency Score', fontsize=11)
        ax2.set_title('Efficiency Comparison', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        
        # === 3. Cost-Benefit Analysis ===
        ax3 = axes[2]
        
        # Scatter plot: X = cost (tokens), Y = benefit (accuracy)
        ax3.scatter(s1.get('avg_tokens_used', 0), s1.get('accuracy', 0),
                   s=300, c=self.colors['system1'], label='System 1',
                   edgecolors='white', linewidth=2, marker='o', zorder=5)
        ax3.scatter(s2.get('avg_tokens_used', 0), s2.get('accuracy', 0),
                   s=300, c=self.colors['system2'], label='System 2',
                   edgecolors='white', linewidth=2, marker='s', zorder=5)
        
        # Add iso-efficiency lines
        max_tokens = max(s1.get('avg_tokens_used', 1), s2.get('avg_tokens_used', 1)) * 1.3
        for eff in [0.5, 1.0, 2.0]:
            x_line = np.linspace(1, max_tokens, 100)
            y_line = eff * x_line / 1000
            y_line = np.clip(y_line, 0, 1)
            ax3.plot(x_line, y_line, '--', color=self.colors['neutral'], alpha=0.3, linewidth=1)
            ax3.text(max_tokens * 0.95, min(eff * max_tokens / 1000, 0.95), 
                    f'eff={eff}', fontsize=8, color=self.colors['text_light'], alpha=0.7)
        
        ax3.set_xlabel('Tokens Used', fontsize=11)
        ax3.set_ylabel('Accuracy', fontsize=11)
        ax3.set_title('Cost-Benefit Analysis', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.set_ylim(0, 1.05)
        ax3.set_xlim(0, max_tokens)
        
        plt.suptitle('Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            save_figure(fig, "08_token_usage_analysis", self.output_dir)
        
        return fig
    
    def create_executive_dashboard(self, results: Dict, save: bool = True) -> plt.Figure:
        """
        Executive summary dashboard with key findings
        """
        fig = plt.figure(figsize=(20, 14))
        
        # Create custom grid
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3,
                     height_ratios=[0.8, 1, 1, 0.6])
        
        s1 = results.get('system1_results', {})
        s2 = results.get('system2_results', {})
        comparison = results.get('comparison', {})
        categories = results.get('category_analysis', {})
        
        # === Header ===
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        header_text = f"""
        DUAL PROCESS THEORY EXPERIMENT
        ══════════════════════════════════════════════════════════════════════════════════════════
        Experiment: {results.get('experiment_name', 'N/A')}  |  Tasks: {results.get('config', {}).get('total_tasks', 'N/A')}  |  Duration: {results.get('duration_seconds', 0):.1f}s
        """
        
        ax_header.text(0.5, 0.5, header_text, transform=ax_header.transAxes,
                      fontsize=14, ha='center', va='center', fontfamily='monospace',
                      fontweight='bold', color=self.colors['dark'])
        
        # === Key Metrics Row ===
        # Metric 1: S1 Accuracy
        ax_m1 = fig.add_subplot(gs[1, 0])
        self._draw_metric_card(ax_m1, 'System 1', f"{s1.get('accuracy', 0):.1%}",
                              'Accuracy', self.colors['system1'])
        
        # Metric 2: S2 Accuracy
        ax_m2 = fig.add_subplot(gs[1, 1])
        self._draw_metric_card(ax_m2, 'System 2', f"{s2.get('accuracy', 0):.1%}",
                              'Accuracy', self.colors['system2'])
        
        # Metric 3: Time Ratio
        ax_m3 = fig.add_subplot(gs[1, 2])
        time_ratio = comparison.get('response_time_ratio', 1)
        self._draw_metric_card(ax_m3, 'Speed Ratio', f"{time_ratio:.1f}×",
                              'S2/S1 Time', self.colors['accent'])
        
        # Metric 4: Significance
        ax_m4 = fig.add_subplot(gs[1, 3])
        p_val = comparison.get('statistical_test', {}).get('p_value', 1.0)
        sig_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        self._draw_metric_card(ax_m4, 'Significance', sig_text,
                              f'p={p_val:.4f}', self.colors['correct'] if p_val < 0.05 else self.colors['neutral'])
        
        # === Charts Row ===
        # Chart 1: Accuracy comparison
        ax_c1 = fig.add_subplot(gs[2, :2])
        
        x = np.array([0, 1])
        bars = ax_c1.bar(x, [s1.get('accuracy', 0), s2.get('accuracy', 0)], width=0.5,
                        color=[self.colors['system1'], self.colors['system2']],
                        edgecolor='white', linewidth=2)
        
        for bar, acc in zip(bars, [s1.get('accuracy', 0), s2.get('accuracy', 0)]):
            ax_c1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')
        
        ax_c1.set_xticks(x)
        ax_c1.set_xticklabels(['System 1\n(Intuitive)', 'System 2\n(Analytical)'])
        ax_c1.set_ylabel('Accuracy')
        ax_c1.set_ylim(0, 1.15)
        ax_c1.set_title('Overall Performance', fontsize=12, fontweight='bold')
        ax_c1.axhline(y=0.5, color=self.colors['neutral'], linestyle='--', alpha=0.5)
        
        # Chart 2: Category breakdown
        ax_c2 = fig.add_subplot(gs[2, 2:])
        
        if categories:
            cat_names = list(categories.keys())
            clean_names = [c.replace('_tasks', '').title() for c in cat_names]
            x = np.arange(len(cat_names))
            width = 0.35
            
            s1_accs = [categories[c].get('system1', {}).get('accuracy', 0) for c in cat_names]
            s2_accs = [categories[c].get('system2', {}).get('accuracy', 0) for c in cat_names]
            
            ax_c2.bar(x - width/2, s1_accs, width, label='S1', color=self.colors['system1'])
            ax_c2.bar(x + width/2, s2_accs, width, label='S2', color=self.colors['system2'])
            
            ax_c2.set_xticks(x)
            ax_c2.set_xticklabels(clean_names, fontsize=10)
            ax_c2.set_ylabel('Accuracy')
            ax_c2.set_ylim(0, 1.15)
            ax_c2.set_title('By Category', fontsize=12, fontweight='bold')
            ax_c2.legend(loc='upper right')
        
        # === Footer: Key Findings ===
        ax_footer = fig.add_subplot(gs[3, :])
        ax_footer.axis('off')
        
        s1_acc = s1.get('accuracy', 0)
        s2_acc = s2.get('accuracy', 0)
        acc_diff = s2_acc - s1_acc
        
        findings = f"""
        KEY FINDINGS
        ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        • System 2 (Analytical) {'outperforms' if acc_diff > 0 else 'underperforms'} System 1 (Intuitive) by {abs(acc_diff):.1%}
        • System 2 takes {time_ratio:.1f}× longer to respond, demonstrating the classic speed-accuracy tradeoff
        • Statistical significance: {'Confirmed (p<0.05)' if p_val < 0.05 else 'Not confirmed (p≥0.05)'}
        • This aligns with Dual Process Theory predictions (Kahneman, 2011)
        """
        
        ax_footer.text(0.5, 0.5, findings, transform=ax_footer.transAxes,
                      fontsize=11, ha='center', va='center', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'],
                               edgecolor=self.colors['grid']))
        
        if save:
            save_figure(fig, "10_experiment_dashboard", self.output_dir)
        
        return fig
    
    def _draw_metric_card(self, ax, title: str, value: str, subtitle: str, color: str):
        """Draw a metric card"""
        ax.axis('off')
        
        # Background
        rect = mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9,
                                       boxstyle="round,pad=0.02,rounding_size=0.05",
                                       facecolor='white', edgecolor=color,
                                       linewidth=3, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Title
        ax.text(0.5, 0.85, title, transform=ax.transAxes, ha='center', va='top',
               fontsize=11, fontweight='medium', color=self.colors['text_light'])
        
        # Value
        ax.text(0.5, 0.5, value, transform=ax.transAxes, ha='center', va='center',
               fontsize=28, fontweight='bold', color=color)
        
        # Subtitle
        ax.text(0.5, 0.15, subtitle, transform=ax.transAxes, ha='center', va='bottom',
               fontsize=9, color=self.colors['text_light'])
    
    def close_all(self):
        """Close all generated figures"""
        for fig in self.generated_figures:
            plt.close(fig)
        self.generated_figures = []
