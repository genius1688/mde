"""
Puiseux-Newton-Riemann Visualizations
=====================================
Publication-quality figures for the mathematical report on ODE reducibility analysis.

Author: Generated for Mathematics Report
Date: 2025-11-13
Output: repo-relative `figure/section*` folders
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, Rectangle
from mpl_toolkits.mplot3d import proj3d
from fractions import Fraction
import os
from pathlib import Path
from matplotlib.ticker import MaxNLocator

# ============================================================================
# COLOR PALETTE CONFIGURATION
# ============================================================================

# Typography and sizing constants
AXIS_FONT = 11
TICK_FONT = 8
LEGEND_FS = 7
PANEL_FS = 13
ANNO_FS = 7
FIGSIZE_SINGLE = (3.4, 3.4)

# Natural Vintage Palette (warm, earthy)
PALETTE_VINTAGE = {
    'light_blue': '#acd2d6',
    'sage_green': '#91a5a1', 
    'cream': '#fff8db',
    'gold': '#e6b451',
    'dark_green': '#5e6e54'
}

# Contrast Palette (cool, professional)
PALETTE_CONTRAST = {
    'dark_teal': '#1F3A3D',
    'navy': '#2C3E50',
    'medium_blue': '#4B8DA3',
    'light_cyan': '#A8DADC',
    'pale_white': '#F1FAEE'
}

# Academic color scheme combining both
COLORS = {
    'primary': PALETTE_CONTRAST['navy'],
    'secondary': PALETTE_CONTRAST['medium_blue'],
    'accent1': PALETTE_VINTAGE['gold'],
    'accent2': PALETTE_VINTAGE['dark_green'],
    'light_gray': '#b0b0b0',
    'light1': PALETTE_CONTRAST['light_cyan'],
    'light2': PALETTE_VINTAGE['light_blue'],
    'background': PALETTE_CONTRAST['pale_white'],
    'neutral': PALETTE_VINTAGE['sage_green'],
    'highlight': PALETTE_VINTAGE['cream']
}

# Global matplotlib settings for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': AXIS_FONT,
    'axes.labelsize': AXIS_FONT,
    'axes.labelweight': 'bold',
    'axes.titlesize': AXIS_FONT + 2,
    'xtick.labelsize': TICK_FONT,
    'ytick.labelsize': TICK_FONT,
    'legend.fontsize': LEGEND_FS,
    'figure.titlesize': AXIS_FONT + 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True
})

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================

# Resolve output relative to the script location (not caller CWD)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(fig, filename, subfolder=None):
    """Save figure with consistent formatting."""
    if subfolder:
        save_path = OUTPUT_DIR / subfolder
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = OUTPUT_DIR
    
    full_path = save_path / filename
    fig.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {full_path}")
    return full_path

# ============================================================================
# STYLING HELPERS
# ============================================================================

def apply_axis_styling(ax, xlabel=None, ylabel=None, title=None, rotate_xticks=True,
                       y_locator=None, x_locator=None, tight_y=True, title_fs=None):
    """Apply shared typography, ticks, and aspect styling to an Axes."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_FONT, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_FONT, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=title_fs or (AXIS_FONT + 2), fontweight='bold', pad=10)

    if x_locator is not None:
        ax.xaxis.set_major_locator(x_locator)
    if y_locator is not None:
        ax.yaxis.set_major_locator(y_locator)

    if rotate_xticks:
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha('right')

    ax.tick_params(labelsize=TICK_FONT, width=0.8)
    ax.margins(x=0.0, y=0.05 if tight_y else 0.08)

    fig = ax.get_figure()
    fig.canvas.draw_idle()
    # Ensure square panels after layout computation
    fig.canvas.draw()
    ax.set_box_aspect(1)

def add_headroom(ax, top_pad=0.08, bottom_pad=0.05):
    """Add 5–10% vertical padding to avoid cramped annotations."""
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    if span == 0:
        return
    ax.set_ylim(ymin - span * bottom_pad, ymax + span * top_pad)

def styled_legend(ax, loc='upper right', ncol=1, bbox_to_anchor=None, handles=None, **legend_kwargs):
    """Standard legend with thin black frame and small font.

    The legacy ``linewidth`` keyword (sometimes passed by callers) is stripped
    to avoid Matplotlib ``Legend`` init errors, and the frame linewidth is set
    after creation to keep the thin border styling consistent across versions.
    """
    frame_linewidth = legend_kwargs.pop('linewidth', 0.5)
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fontsize=LEGEND_FS,
        handles=handles,
        **legend_kwargs,
    )
    if legend and legend.get_frame() is not None:
        frame = legend.get_frame()
        frame.set_linewidth(frame_linewidth)
        frame.set_edgecolor('black')
        frame.set_alpha(0.95)
    return legend

# ============================================================================
# FIGURE 1a: SOLVABILITY MAP - (α,β) plane heatmap
# ============================================================================

def gcd_rational(a, b, tol=1e-6):
    """Compute GCD of two rational numbers expressed as fractions."""
    try:
        frac_a = Fraction(a).limit_denominator(1000)
        frac_b = Fraction(b).limit_denominator(1000)
        from math import gcd as int_gcd
        g = int_gcd(frac_a.numerator * frac_b.denominator, 
                    frac_b.numerator * frac_a.denominator)
        d = frac_a.denominator * frac_b.denominator
        result = Fraction(g, d)
        return float(result)
    except:
        return np.nan

def compute_puiseux_step(alpha, beta):
    """Compute p = gcd_Q{2-α, 1-β}."""
    diff1 = 2 - alpha
    diff2 = 1 - beta
    
    if abs(diff1) < 1e-10 or abs(diff2) < 1e-10:
        return np.inf
    
    return gcd_rational(diff1, diff2)

# ============================================================================
# FIGURE 1a: SOLVABILITY MAP - (α,β) plane heatmap (REDESIGNED)
# ============================================================================

def create_fig1a_solvability_map():
    """Fig 1a: Solvability map with DISCRETE colormap and dual colorbar."""
    print("\n" + "="*60)
    print("Creating Figure 1a: Solvability Map (Optimized)")
    print("="*60)
    
    # Create grid
    alpha_range = np.linspace(-1, 5, 400)
    beta_range = np.linspace(-1, 4, 400)
    Alpha, Beta = np.meshgrid(alpha_range, beta_range)
    
    # Compute Puiseux step for each point
    P = np.zeros_like(Alpha)
    for i in range(Alpha.shape[0]):
        for j in range(Alpha.shape[1]):
            P[i, j] = compute_puiseux_step(Alpha[i, j], Beta[i, j])
    
    # Cap infinite/large values for visualization
    P = np.minimum(P, 5)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    
    # DISCRETE colormap (6 bins) to reinforce gcd quantization
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    colors_map = [
        COLORS['highlight'],      # Light cream/yellow
        COLORS['light2'],         # Light blue
        COLORS['light1'],         # Light cyan
        COLORS['secondary'],      # Medium blue
        COLORS['neutral'],        # Sage green
        COLORS['accent2']         # Dark green
    ]
    n_bins = 6  # Discrete bins
    cmap_custom = LinearSegmentedColormap.from_list('academic', colors_map, N=n_bins)
    bounds = np.linspace(0, 5, n_bins + 1)
    norm = BoundaryNorm(bounds, cmap_custom.N)
    
    # Create heatmap with DISCRETE colormap
    im = ax.contourf(Alpha, Beta, P, levels=bounds, cmap=cmap_custom, norm=norm, alpha=0.85)
    
    # Add discrete contour lines for specific p values
    contour_levels = [1/3, 1/2, 2/3, 1, 3/2, 2, 3]
    cs = ax.contour(Alpha, Beta, P, levels=contour_levels, colors='white', 
                    linewidths=1.2, alpha=0.5, linestyles='--')
    ax.clabel(cs, inline=True, fontsize=ANNO_FS, fmt='p=%.2f', 
              inline_spacing=8, use_clabeltext=True)
    
    # Highlight constant-coefficient line with CONTRASTING DASHED stroke
    alpha_cc = np.linspace(-1, 5, 100)
    beta_cc = 1 - (2 - alpha_cc)
    ax.plot(alpha_cc, beta_cc, 'r--', linewidth=2.5, dashes=(8, 4),
            label=r'Constant-coeff: $2-\alpha = 1-\beta$', zorder=5, alpha=0.9)
    
    # Mark the mother problem point as GLYPH with call-out
    alpha_mother, beta_mother = 4/3, 1/3
    ax.plot(alpha_mother, beta_mother, 'rD', markersize=10, 
            label=r'Mother: $(\alpha,\beta)=(4/3,1/3)$', zorder=6, 
            markeredgecolor='darkred', markeredgewidth=1.5, markerfacecolor='red')
    
    # Add call-out box for mother problem (visually anchored)
    ax.annotate(r'$p=2/3, m=3$', xy=(alpha_mother, beta_mother), 
               xytext=(alpha_mother + 0.7, beta_mother + 0.7),
               fontsize=ANNO_FS, color='darkred', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                        edgecolor='darkred', linewidth=1.0),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2, shrinkA=6))
    
    # Add key regions annotations
    annotations = [
        {'pos': (1.5, 0.5), 'text': r'$p = 2-\alpha = 1-\beta$', 
         'desc': 'Reducible region'},
        {'pos': (3.5, 3), 'text': r'Large $p$', 
         'desc': 'Higher ramification'},
        {'pos': (0.5, 2.5), 'text': r'Small $p$', 
         'desc': 'Lower ramification'}
    ]
    
    for anno in annotations:
        ax.annotate(anno['text'] + '\n' + f"({anno['desc']})", 
                   xy=anno['pos'], fontsize=ANNO_FS, ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=COLORS['background'], 
                            alpha=0.9, edgecolor=COLORS['neutral'], linewidth=1.0),
                   style='italic')
    
    # Enhanced explanation box
    explanation = (
        r'$\mathbf{Puiseux\ Step:}\ p = \gcd_{\mathbb{Q}}\{2-\alpha,\,1-\beta\}$' + '\n' +
        r'Determines ramification index $m = 1/p$ for substitution $x = w^m$'
    )
    ax.text(0.5, 0.02, explanation, transform=ax.transAxes, 
           fontsize=AXIS_FONT, ha='center', va='bottom',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    alpha=0.95, edgecolor=COLORS['primary'], linewidth=1.2))
    
    apply_axis_styling(
        ax,
        xlabel=r'$\alpha$ (exponent of $x$ in $y^{\prime\prime}$ term)',
        ylabel=r'$\beta$ (exponent of $x$ in $y^{\prime}$ term)',
        title=r'Solvability Map: Puiseux Step $p = \gcd_{\mathbb{Q}}\{2-\alpha,\,1-\beta\}$',
        rotate_xticks=True
    )
    ax.grid(True, alpha=0.25, linestyle=':', color='gray', linewidth=0.6)
    
    # DUAL COLORBAR: Primary for p, secondary for m=1/p
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="2.5%", pad=0.1)
    cax2 = divider.append_axes("right", size="2.5%", pad=0.5)
    
    # Primary colorbar: Puiseux step p
    cbar1 = plt.colorbar(im, cax=cax1)
    cbar1.set_label('Puiseux step $p$', fontsize=AXIS_FONT, fontweight='bold',
                    rotation=270, labelpad=16)
    cbar1.ax.tick_params(labelsize=TICK_FONT)
    
    # Secondary colorbar: sheets m = 1/p at integer ticks
    # Create discrete m values
    m_ticks = [1, 2, 3, 5, 10]
    p_from_m = [1/m for m in m_ticks]
    cbar2 = fig.colorbar(im, cax=cax2, ticks=p_from_m)
    cbar2.ax.set_yticklabels([f'{m}' for m in m_ticks])
    cbar2.set_label('Sheets $m = 1/p$', fontsize=AXIS_FONT, fontweight='bold',
                    rotation=270, labelpad=14)
    cbar2.ax.tick_params(labelsize=TICK_FONT)
    
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    add_headroom(ax)
    
    # Add grid reference lines
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3, linestyle='-')
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3, linestyle='-')

    styled_legend(ax, loc='upper right')
    
    save_figure(fig, 'fig1a_solvability_map.pdf', 'section1')
    save_figure(fig, 'fig1a_solvability_map.png', 'section1')
    plt.close()

# ============================================================================
# FIGURE 1b: POWER-LINE BUNDLE - Affine lines showing exponent alignment
# ============================================================================

def create_fig1b_powerline_bundle():
    """Fig 1b: Three affine lines with UNIFIED HUE and n-shift braces."""
    print("\n" + "="*60)
    print("Creating Figure 1b: Power-line Bundle (Optimized)")
    print("="*60)
    
    # Parameters for mother problem
    alpha, beta = 4/3, 1/3
    p = 2/3  # Puiseux step
    r = 0    # Starting exponent
    
    n_range = np.arange(-2, 8)
    
    # Three exponent sequences
    E1 = alpha - 2 + p * n_range + r  # y'' term
    E2 = beta - 1 + p * n_range + r   # y' term
    E3 = p * n_range + r               # y term
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIGSIZE_SINGLE[0] * 2, FIGSIZE_SINGLE[1]), constrained_layout=True
    )
    
    # UNIFIED HUE with increasing luminance
    from matplotlib.colors import to_rgb
    import colorsys
    base_color = to_rgb(COLORS['secondary'])
    h, l, s = colorsys.rgb_to_hls(*base_color)
    colors_lines = [
        colorsys.hls_to_rgb(h, l * 0.6, s),  # E1: darker
        colorsys.hls_to_rgb(h, l * 0.8, s),  # E2: medium
        colorsys.hls_to_rgb(h, l * 1.0, s)   # E3: lighter
    ]

    # Left panel: All three lines with unified hue
    ax1.plot(n_range, E1, 'o-',
             color=COLORS['light_gray'], linewidth=4.0, markersize=9,
             label=r"$E_1(\alpha-2+pn+r)$: $y''$ term",
             zorder=2, clip_on=False)
    ax1.plot(n_range, E2, 's--',
             color=COLORS['secondary'], linewidth=2.0, markersize=6,
             label=r"$E_2(\beta-1+pn+r)$: $y'$ term",
             zorder=3, clip_on=False)
    ax1.plot(n_range, E3, '^-', linewidth=2.5, markersize=8,
             label=r"$E_3(pn+r)$: $y$ term", color=colors_lines[2],
             zorder=3, clip_on=False)
    
    # ADD BRACES showing n -> n +/- 1 shift
    for n_val in [1, 2]:
        ax1.axvline(n_val, color='gray', linestyle=':', alpha=0.3)
    
    # Compact annotation for n-shift alignment
    ax1.annotate('', xy=(2, E1[4]), xytext=(1, E2[3]),
                arrowprops=dict(arrowstyle='<->', color=COLORS['accent1'], lw=2))
    ax1.text(1.5, (E1[4] + E2[3])/2 + 0.1, r'$n \mapsto n+1$',
            fontsize=ANNO_FS, ha='center', color=COLORS['accent1'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, linewidth=0.8))
    
    apply_axis_styling(ax1, xlabel=r'Index $n$', ylabel=r'Exponent of $x$',
                       title='Affine Power Lines: n-shift enforces alignment',
                       rotate_xticks=True, title_fs=AXIS_FONT + 1)
    styled_legend(ax1, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    y_min_1 = np.min([E1.min(), E2.min(), E3.min()])
    y_max_1 = np.max([E1.max(), E2.max(), E3.max()])
    ax1.set_xlim(n_range[0], n_range[-1])
    ax1.set_ylim(y_min_1, y_max_1)

    # Right panel: Zoomed with p as CONTRASTING ACCENT color
    n_zoom = np.arange(0, 5)
    E1_zoom = alpha - 2 + p * n_zoom + r
    E2_zoom = beta - 1 + p * n_zoom + r
    E3_zoom = p * n_zoom + r

    ax2.plot(n_zoom, E1_zoom, 'o-',
             color=COLORS['light_gray'], linewidth=4.0, markersize=9,
             label=r"$E_1$", zorder=2, clip_on=False)
    ax2.plot(n_zoom, E2_zoom, 's--',
             color=COLORS['secondary'], linewidth=2.0, markersize=6,
             label=r"$E_2$", zorder=3, clip_on=False)
    ax2.plot(n_zoom, E3_zoom, '^-', linewidth=2.5, markersize=10,
             label=r"$E_3$", color=colors_lines[2], zorder=3, clip_on=False)
    
    # Highlight p step as contrasting accent
    ax2.annotate('', xy=(1, E3_zoom[1]), xytext=(0, E3_zoom[0]),
                arrowprops=dict(arrowstyle='<->', color=COLORS['accent1'], lw=3))
    ax2.text(0.5, (E3_zoom[0] + E3_zoom[1])/2 - 0.15, r'Step $p=2/3$',
            fontsize=ANNO_FS, ha='center', color=COLORS['accent1'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.9, linewidth=0.8))
    
    # Annotate alignment
    for i, n_val in enumerate(n_zoom):
        y_pos = E3_zoom[i]
        ax2.annotate(f'$n={n_val}$', xy=(n_val, y_pos), 
                    xytext=(n_val + 0.2, y_pos - 0.3),
                    fontsize=ANNO_FS, color=COLORS['accent2'])
    
    apply_axis_styling(ax2, xlabel=r'Index $n$', ylabel=r'Exponent of $x$',
                       title='Power Alignment Detail (Mother Problem)',
                       rotate_xticks=True, title_fs=AXIS_FONT + 1)
    handles, labels = ax2.get_legend_handles_labels()
    if len(labels) >= 2:
        labels[0] = r"$E_1 \equiv E_2$ (degenerate)"
        del handles[1]
        del labels[1]
    styled_legend(ax2, loc='upper left', handles=handles, labels=labels)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    y_min_2 = np.min([E1_zoom.min(), E2_zoom.min(), E3_zoom.min()])
    y_max_2 = np.max([E1_zoom.max(), E2_zoom.max(), E3_zoom.max()])
    ax2.set_xlim(n_zoom[0], n_zoom[-1])
    ax2.set_ylim(y_min_2, y_max_2)
    save_figure(fig, 'fig1b_powerline_bundle.pdf', 'section1')
    save_figure(fig, 'fig1b_powerline_bundle.png', 'section1')
    plt.close()

# ============================================================================
# FIGURE NP-1: NEWTON POLYGON - General cases (Dashboard with 4 scenarios)
# ============================================================================

def create_figNP1_newton_polygon_general():
    """Fig NP-1: Newton polygon classification (optimized for clarity)."""
    print("\n" + "="*60)
    print("Creating Figure NP-1: Newton Polygon (Optimized)")
    print("="*60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    fig.suptitle(
        'Newton Polygon Classification: Lower Convex Hull Selection\n'
        r'Rule: Only the steepest segment determines the Puiseux step $p = 1/|\sigma|$',
        fontsize=AXIS_FONT + 3,
        fontweight='bold',
        y=1.02,
    )

    cases = [
        {
            'e_vals': [0, 0.6, 1.2],
            'hull_edges': [(0, 2, 'main')],
            'title': 'Case I: Single Edge (Collinear)',
            'desc': r'Points collinear $\to$ Single slope $\sigma$',
            'color': '#2E8B57',  # SeaGreen
        },
        {
            'e_vals': [0, 0.8, 1.2],
            'hull_edges': [(0, 1, 'main'), (1, 2, 'secondary')],
            'title': 'Case II: Two-Edge Convex Break',
            'desc': r'Break in hull: Steepest $\sigma$ wins',
            'color': '#E69F00',  # Orange
        },
        {
            'e_vals': [0, 1.4, 1.2],
            'hull_edges': [(0, 2, 'main')],
            'extra_points': [(1, 1.4)],
            'title': 'Case III: Point Above Hull',
            'desc': r'Point above hull is ignored',
            'color': '#56B4E9',  # SkyBlue
        },
        {
            'e_vals': [0, 0.67, 0.67],
            'hull_edges': [(0, 1, 'main'), (1, 2, 'horizontal')],
            'title': 'Case IV: Horizontal Edge',
            'desc': r'Horizontal edge $\sigma=0 \to$ No step',
            'color': '#CC79A7',  # Reddish Purple
        },
    ]

    for idx, (ax, case) in enumerate(zip(axes.flat, cases)):
        k_vals = np.array([0, 1, 2])
        e_vals = np.array(case['e_vals'])
        main_color = case['color']

        # Compute hull baseline for shading
        # Case III ignores the middle point, others follow the full path
        if idx == 2:
            hull_k = np.array([k_vals[0], k_vals[-1]])
            hull_y = np.array([e_vals[0], e_vals[-1]])
        else:
            hull_k = k_vals
            hull_y = e_vals

        ax.fill_between(hull_k, hull_y, -0.5, color=main_color, alpha=0.12, zorder=1)

        # Plot points with hollow styling; ignored points pushed back
        extra_pts = case.get('extra_points', [])
        ignored_mask = {(k, e) for k, e in extra_pts}
        for k, e in zip(k_vals, e_vals):
            is_ignored = (k, e) in ignored_mask
            pt_face = 'white'
            edge_col = 'gray' if is_ignored else main_color
            inner_col = 'lightgray' if is_ignored else main_color
            z_pt = 3 if is_ignored else 5

            ax.plot(
                k,
                e,
                'o',
                markersize=9,
                color=pt_face,
                markeredgecolor=edge_col,
                markeredgewidth=2,
                zorder=z_pt,
                clip_on=False,
            )
            if not is_ignored:
                ax.plot(
                    k,
                    e,
                    'o',
                    markersize=5,
                    color=inner_col,
                    zorder=z_pt,
                    clip_on=False,
                )

            ax.text(
                k,
                e + 0.12,
                f'({k}, {e:.2g})',
                ha='center',
                fontsize=AXIS_FONT - 2,
                color='#333',
                zorder=6,
            )

        # Draw hull and secondary edges with clear hierarchy
        legend_handles = []
        for edge in case['hull_edges']:
            i, j, role = edge
            slope = (e_vals[j] - e_vals[i]) / (k_vals[j] - k_vals[i]) if k_vals[j] != k_vals[i] else 0.0
            mid_k = (k_vals[i] + k_vals[j]) / 2
            mid_e = (e_vals[i] + e_vals[j]) / 2

            if role == 'horizontal':
                line = ax.plot(
                    [k_vals[i], k_vals[j]],
                    [e_vals[i], e_vals[j]],
                    '--',
                    color='gray',
                    linewidth=2,
                    alpha=0.7,
                    zorder=2,
                    label=r'$\sigma=0$ (no $p$)',
                    clip_on=False,
                )[0]
                ax.text(mid_k, mid_e - 0.18, r'$\sigma=0$', ha='center',
                        color='gray', fontsize=AXIS_FONT - 2)
                legend_handles.append(line)
                continue

            is_main = role == 'main'
            line_color = main_color if is_main else COLORS['light_gray']
            line_width = 3.5 if is_main else 1.8
            line_style = '-' if is_main else '--'
            line_alpha = 0.95 if is_main else 0.55

            line = ax.plot(
                [k_vals[i], k_vals[j]],
                [e_vals[i], e_vals[j]],
                line_style,
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=2 if is_main else 2.5,
                label='Main slope' if is_main else 'Secondary (ignored)',
                clip_on=False,
            )[0]
            if is_main:
                p_val = 1.0 / abs(slope) if abs(slope) > 1e-6 else np.inf
                offset = (-20, 20) if idx == 1 else (-10, 20)
                ax.annotate(
                    f'$\\sigma={slope:.2f}$\n$p={p_val:.2f}$',
                    xy=(mid_k, mid_e),
                    xytext=offset,
                    textcoords='offset points',
                    ha='right',
                    fontsize=AXIS_FONT - 1,
                    color='white',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.35', fc=line_color, ec='none', alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color=line_color, lw=1.5),
                )
            else:
                ax.annotate(
                    f'$\\sigma={slope:.2f}$\n(ignored)',
                    xy=(mid_k, mid_e),
                    xytext=(10, -30),
                    textcoords='offset points',
                    ha='left',
                    fontsize=AXIS_FONT - 2,
                    color='gray',
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                )

            legend_handles.append(line)

        # Axes styling
        ax.set_title(case['title'], fontsize=AXIS_FONT + 1, fontweight='bold', pad=8)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, e_vals.max())
        ax.set_xticks(k_vals)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=TICK_FONT)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle=':', alpha=0.25, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Derivative order $k$', fontsize=AXIS_FONT, fontweight='bold')
        ax.set_ylabel('Power offset $e_k$', fontsize=AXIS_FONT, fontweight='bold')
        ax.text(
            0.5,
            0.05,
            case['desc'],
            transform=ax.transAxes,
            fontsize=AXIS_FONT,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8, edgecolor='none'),
        )
        ax.set_box_aspect(1)

        if idx == 1:
            styled_legend(
                ax,
                loc='upper left',
                handles=legend_handles,
                edgecolor='gray',
                framealpha=0.9,
                ncol=1,
                borderpad=0.5,
            )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    save_figure(fig, 'figNP1_newton_polygon_general.pdf', 'section3')
    save_figure(fig, 'figNP1_newton_polygon_general.png', 'section3')
    plt.close()

# ============================================================================
# FIGURE NP-2: NEWTON POLYGON - Mother problem specific (CORRECTED)
# ============================================================================

def create_figNP2_newton_polygon_mother():
    """Fig NP-2: Newton polygon for 9x^(4/3)y'' + 6x^(1/3)y' + y = 0 (CORRECTED)."""
    print("\n" + "="*60)
    print("Creating Figure NP-2: Newton Polygon (Mother Problem - CORRECTED)")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    
    # Mother problem: α=4/3, β=1/3
    alpha, beta = 4/3, 1/3
    k_vals = np.array([0, 1, 2])
    e_vals = np.array([0, 1 - beta, 2 - alpha])  # [0, 2/3, 2/3]
    
    # Plot points with different styles
    # P0 and P2 are on the hull (dark), P1 is NOT on the lower hull edge
    colors_points = [COLORS['primary'], 'gray', COLORS['primary']]
    sizes = [10, 8, 10]
    alphas = [1.0, 0.5, 1.0]

    for k, e, c, s, a in zip(k_vals, e_vals, colors_points, sizes, alphas):
        ax.plot(k, e, 'o', markersize=s, color=c, zorder=4, alpha=a, clip_on=False)
    
    # Annotate with exact values - ADJUSTED POSITIONS
    labels = [
        r'$P_0=(0,0)$',
        r'$P_1=(1,\frac{2}{3})$',
        r'$P_2=(2,\frac{2}{3})$'
    ]
    # Adjusted offsets to avoid overlap
    offsets = [(-0.25, -0.14), (-0.45, 0.08), (0.18, 0.08)]
    for k, e, label, offset in zip(k_vals, e_vals, labels, offsets):
        alpha_text = 0.5 if label == labels[1] else 1.0
        ax.annotate(label, xy=(k, e), xytext=(k + offset[0], e + offset[1]),
                   fontsize=AXIS_FONT, fontweight='bold', alpha=alpha_text,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, linewidth=1.2),
                   clip_on=False)

    # CORRECT LOWER HULL: P0 to P2 directly (slope = 2/3 / 2 = 1/3)
    ax.plot([0, 2], [0, 2/3], '-', linewidth=3, color=COLORS['primary'],
            label='Lower edge: $P_0 \\to P_2$', zorder=3, alpha=0.9, clip_on=False)

    # Shade region below the CORRECT hull
    ax.fill_between([0, 2], [0, 2/3], -0.12, alpha=0.18, color=COLORS['primary'])

    # SLOPE TRIANGLE (inset removed): visual rise/run cue under the main edge
    ax.plot([0, 2], [0, 0], '--', color=COLORS['secondary'], linewidth=1.6, zorder=1)
    ax.plot([2, 2], [0, 2/3], '--', color=COLORS['secondary'], linewidth=1.6, zorder=1)
    ax.text(1.0, -0.05, r'$\Delta k = 2$', ha='center', va='top', fontsize=AXIS_FONT,
            color=COLORS['secondary'], fontweight='bold', clip_on=False)
    ax.text(2.05, 1/3, r'$\Delta e = \frac{2}{3}$', ha='left', va='center', fontsize=AXIS_FONT,
            color=COLORS['secondary'], fontweight='bold', clip_on=False)

    # Simplified slope annotation on hypotenuse
    ax.text(1.05, 0.32, r'$\sigma = \frac{1}{3}$', fontsize=AXIS_FONT, fontweight='bold',
            color=COLORS['primary'], ha='left', va='center', clip_on=False)
    
    # Result box - MOVED TO BOTTOM RIGHT as requested
    textstr = r'$\sigma = \frac{1}{3} \quad \Rightarrow \quad p = \frac{1}{|\sigma|} = 3$'
    props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['accent1'],
                alpha=0.95, edgecolor=COLORS['primary'], linewidth=2)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=PANEL_FS,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, clip_on=False)
    
    # Add explanation box at top
    explanation = r'Lower convex hull: direct connection $P_0 \to P_2$ (ignores $P_1$)'
    ax.text(0.5, 0.97, explanation, transform=ax.transAxes, fontsize=AXIS_FONT,
            verticalalignment='top', horizontalalignment='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.8),
            clip_on=False)
    
    apply_axis_styling(
        ax,
        xlabel=r'Derivative order $k$',
        ylabel=r'Power offset $e_k$',
        title=r'Newton Polygon for $9x^{4/3}y^{\prime\prime} + 6x^{1/3}y^{\prime} + y = 0$',
        rotate_xticks=True,
        x_locator=MaxNLocator(integer=True, nbins=4)
    )
    styled_legend(ax, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax.set_xlim(-0.1, 2.2)
    ax.set_ylim(-0.1, 0.9)
    
    save_figure(fig, 'figNP2_newton_polygon_mother.pdf', 'section3')
    save_figure(fig, 'figNP2_newton_polygon_mother.png', 'section3')
    plt.close()

# ============================================================================
# FIGURE RIEMANN LIFT: Multi-valued to single-valued lifting
# ============================================================================

def create_fig_riemann_lift():
    """Conceptual visualization of lifting from x-plane to w-plane."""
    print("\n" + "="*60)
    print("Creating Figure: Riemann Surface Lift")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIGSIZE_SINGLE[0] * 2, FIGSIZE_SINGLE[1]), constrained_layout=True
    )
    
    # Left: x-plane with multi-valued branches
    theta = np.linspace(0, 4*np.pi, 200)
    r = np.exp(theta / (4*np.pi))
    
    # Three branches spiraling
    for i in range(3):
        theta_branch = theta + i * 2 * np.pi / 3
        x_real = r * np.cos(theta_branch)
        x_imag = r * np.sin(theta_branch)
        color = [COLORS['primary'], COLORS['accent1'], COLORS['accent2']][i]
        ax1.plot(x_real, x_imag, linewidth=2.0, alpha=0.8, 
                label=f'Branch {i+1}', color=color)
    
    ax1.plot(0, 0, 'ko', markersize=10, label='Branch point $x=0$', zorder=5)
    
    # Add branch cut on negative real axis (ochre color)
    ax1.plot([-2.5, 0], [0, 0], color='#CC8800', linewidth=3, linestyle='--', 
            label='Branch cut', alpha=0.7, zorder=4)
    
    apply_axis_styling(
        ax1,
        xlabel=r'$\mathrm{Re}(x)$',
        ylabel=r'$\mathrm{Im}(x)$',
        title=r'Multi-valued function $y(x)$ in $x$-plane',
        rotate_xticks=True
    )
    styled_legend(ax1, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax1.set_aspect('equal')
    add_headroom(ax1)
    
    # Add covering map annotation
    ax1.text(0.5, 0.97, r'$\pi: \mathcal{R} \to \mathbb{C}_x$ (covering map)', 
            transform=ax1.transAxes, fontsize=AXIS_FONT, ha='center', va='top',
            style='italic', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.9))
    
    # Right: w-plane with single-valued function
    theta_w = np.linspace(0, 4*np.pi, 200)
    r_w = np.exp(theta_w / (4*np.pi))
    
    x_real_w = r_w * np.cos(theta_w)
    x_imag_w = r_w * np.sin(theta_w)
    
    # Single spiral in w-plane
    ax2.plot(x_real_w, x_imag_w, linewidth=2.6, color=COLORS['secondary'], 
            label=r'Single-valued $y(w)$', alpha=0.9)
    ax2.plot(0, 0, 'ko', markersize=10, label='Regular point $w=0$')
    
    apply_axis_styling(
        ax2,
        xlabel=r'$\mathrm{Re}(w)$',
        ylabel=r'$\mathrm{Im}(w)$',
        title=r'Single-valued after $w = x^{1/m}$ substitution ($\mathcal{R}$ becomes simply-connected)$',
        rotate_xticks=True
    )
    styled_legend(ax2, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    ax2.set_aspect('equal')
    add_headroom(ax2)
    
    # Add central annotation
    fig.text(0.5, 0.02, r'Lifting: $x = w^m$ unwraps multi-valuedness', 
             ha='center', fontsize=AXIS_FONT + 2, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.8))
    
    save_figure(fig, 'fig_riemann_lift.pdf', 'section4')
    save_figure(fig, 'fig_riemann_lift.png', 'section4')
    plt.close()

# ============================================================================
# FIGURE THREE-LEAF: 3-sheeted Riemann surface structure
# ============================================================================

def create_fig_threeleaf_structure():
    """3D visualization of three-sheeted Riemann surface with 2D sheet-index inset."""
    print("\n" + "="*60)
    print("Creating Figure: Three-Leaf Riemann Surface (3D + 2D Inset)")
    print("="*60)
    
    fig = plt.figure(figsize=(FIGSIZE_SINGLE[0] * 2, FIGSIZE_SINGLE[1]), constrained_layout=True)
    ax = fig.add_subplot(121, projection='3d')
    
    # Create three sheets
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0.1, 2, 50)
    Theta, R = np.meshgrid(theta, r)
    
    colors_sheets = [COLORS['primary'], COLORS['accent1'], COLORS['accent2']]
    alphas = [0.6, 0.5, 0.4]
    labels = ['Sheet 0: $w_0 = x^{1/3}$', 
              r'Sheet 1: $w_1 = e^{2\pi i/3}x^{1/3}$',
              r'Sheet 2: $w_2 = e^{4\pi i/3}x^{1/3}$']
    
    for i, (color, alpha, label) in enumerate(zip(colors_sheets, alphas, labels)):
        Z = i * 0.5 * np.ones_like(R)  # Height offset for each sheet
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        surf = ax.plot_surface(X, Y, Z, alpha=alpha, color=color, 
                              edgecolor='none', label=label)
        
        # Add branch cut
        branch_cut_r = np.linspace(0.1, 2, 20)
        branch_cut_x = -branch_cut_r
        branch_cut_y = np.zeros_like(branch_cut_r)
        branch_cut_z = i * 0.5 * np.ones_like(branch_cut_r)
        ax.plot(branch_cut_x, branch_cut_y, branch_cut_z, 
               'k--', linewidth=2, alpha=0.7)
    
    # Add connecting curves showing sheet transitions
    theta_connect = np.linspace(np.pi, 3*np.pi, 50)
    r_connect = 1.5
    for i in range(3):
        theta_start = np.pi + i * 2*np.pi/3
        theta_segment = theta_start + np.linspace(0, 2*np.pi/3, 30)
        x_conn = r_connect * np.cos(theta_segment)
        y_conn = r_connect * np.sin(theta_segment)
        z_conn = np.linspace(i * 0.5, ((i+1) % 3) * 0.5, 30)
        ax.plot(x_conn, y_conn, z_conn, 'r-', linewidth=2.5, alpha=0.8)
    
    # Mark branch point
    ax.scatter([0], [0], [0], color='black', s=90, marker='o', 
              label='Branch point $x=0$')
    
    ax.set_xlabel(r'$\mathrm{Re}(x)$', fontsize=AXIS_FONT, fontweight='bold')
    ax.set_ylabel(r'$\mathrm{Im}(x)$', fontsize=AXIS_FONT, fontweight='bold')
    ax.set_zlabel('Sheet index', fontsize=AXIS_FONT, fontweight='bold')
    ax.tick_params(labelsize=TICK_FONT)
    ax.set_title('Three-Sheeted Riemann Surface for $x^{1/3}$ (3D Ribbon View)', fontsize=AXIS_FONT + 3, pad=12, fontweight='bold')
    ax.set_box_aspect((1, 1, 0.6))
    
    # Manual legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors_sheets[0], lw=4, alpha=0.7, label=labels[0]),
        plt.Line2D([0], [0], color=colors_sheets[1], lw=4, alpha=0.7, label=labels[1]),
        plt.Line2D([0], [0], color=colors_sheets[2], lw=4, alpha=0.7, label=labels[2]),
        plt.Line2D([0], [0], color='k', lw=2, linestyle='--', label='Branch cut'),
        plt.Line2D([0], [0], color='r', lw=2, label='Sheet connection')
    ]
    styled_legend(ax, loc='upper left', handles=legend_elements)
    
    # Add deck map annotation
    ax.text2D(0.5, 0.97, r'Deck map: $g(w) = e^{2\pi i/3} w$ (rotation by $120^\circ$)', 
             transform=ax.transAxes, fontsize=AXIS_FONT, ha='center', va='top',
             style='italic', bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.9))
    
    ax.view_init(elev=25, azim=45)
    
    # RIGHT: 2D sheet-index inset (top-down view)
    ax2 = fig.add_subplot(122)
    angles = np.linspace(0, 2*np.pi, 100)
    radius = 1.8
    for i in range(3):
        start_angle = i * 2 * np.pi / 3
        end_angle = (i + 1) * 2 * np.pi / 3
        theta_sector = np.linspace(start_angle, end_angle, 50)
        x_sector = np.append([0], radius * np.cos(theta_sector))
        y_sector = np.append([0], radius * np.sin(theta_sector))
        x_sector = np.append(x_sector, [0])
        y_sector = np.append(y_sector, [0])
        color = [COLORS['primary'], COLORS['accent1'], COLORS['accent2']][i]
        ax2.fill(x_sector, y_sector, color=color, alpha=0.6, edgecolor='black', linewidth=1.5)
        # Sheet labels
        mid_angle = (start_angle + end_angle) / 2
        label_x = 0.8 * radius * np.cos(mid_angle)
        label_y = 0.8 * radius * np.sin(mid_angle)
        ax2.text(label_x, label_y, f'Sheet {i}', fontsize=AXIS_FONT + 2, ha='center', va='center',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, linewidth=1.5))
    
    # Branch point
    ax2.plot(0, 0, 'ko', markersize=13, zorder=5)
    ax2.text(0, -0.35, r'$x=0$ (branch point)', fontsize=AXIS_FONT, ha='center', fontweight='bold')
    
    # Add radial lines showing boundaries
    for i in range(3):
        angle = i * 2 * np.pi / 3
        ax2.plot([0, radius * np.cos(angle)], [0, radius * np.sin(angle)], 
                'k-', linewidth=1.6, alpha=0.7, zorder=4)
    
    ax2.set_xlim(-2.3, 2.3)
    ax2.set_ylim(-2.3, 2.3)
    ax2.set_box_aspect(1)
    ax2.margins(x=0.02, y=0.02)
    ax2.axis('off')
    ax2.set_title(r'2D Sheet-Index Map (Top-Down View)', fontsize=AXIS_FONT + 2, fontweight='bold', pad=8)
    ax2.text(0.5, 0.03, r'Each sheet covers a $120^\circ$ sector in $\mathrm{arg}(x)$', 
            transform=ax2.transAxes, fontsize=AXIS_FONT, ha='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['highlight'], alpha=0.9))
    
    save_figure(fig, 'fig_threeleaf_structure.pdf', 'section4')
    save_figure(fig, 'fig_threeleaf_structure.png', 'section4')
    plt.close()

# ============================================================================
# FIGURE MONODROMY: Path visualization around branch point
# ============================================================================

def create_fig_monodromy_path():
    """Visualization of monodromy: 1, 2, 3 loops around x=0."""
    print("\n" + "="*60)
    print("Creating Figure: Monodromy Path")
    print("="*60)
    
    fig, axes = plt.subplots(
        1, 3, figsize=(FIGSIZE_SINGLE[0] * 3, FIGSIZE_SINGLE[1]), constrained_layout=True
    )
    
    loop_counts = [1, 2, 3]
    titles = [
        r'After 1 loop ($2\pi$): Sheet $0 \to 1$',
        r'After 2 loops ($4\pi$): Sheet $0 \to 2$',
        r'After 3 loops ($6\pi$): Sheet $0 \to 0$'
    ]
    
    for idx, (ax, n_loops, title) in enumerate(zip(axes, loop_counts, titles)):
        # Draw circle at radius 1
        theta_circle = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta_circle)
        y_circle = np.sin(theta_circle)
        ax.plot(x_circle, y_circle, 'k--', linewidth=1, alpha=0.3)
        
        # Draw path
        theta_path = np.linspace(0, 2*np.pi * n_loops, 200)
        x_path = np.cos(theta_path)
        y_path = np.sin(theta_path)
        
        # Color gradient along path
        colors_gradient = plt.cm.viridis(np.linspace(0, 1, len(theta_path)))
        for i in range(len(theta_path)-1):
            ax.plot(x_path[i:i+2], y_path[i:i+2], 
                   color=colors_gradient[i], linewidth=2.5)
        
        # Add direction arrows at key points
        arrow_positions = [int(len(theta_path) * p) for p in [0.15, 0.4, 0.65, 0.9]]
        for pos in arrow_positions:
            if pos < len(theta_path) - 5:
                dx = x_path[pos+5] - x_path[pos]
                dy = y_path[pos+5] - y_path[pos]
                ax.arrow(x_path[pos], y_path[pos], dx*0.8, dy*0.8, 
                        head_width=0.12, head_length=0.08, fc=COLORS['accent1'], 
                        ec=COLORS['accent1'], linewidth=2, zorder=4)
        
        # Mark start and end
        ax.plot(1, 0, 'go', markersize=12, label='Start', zorder=5)
        ax.plot(x_path[-1], y_path[-1], 'ro', markersize=12, label='End', zorder=5)
        
        # Branch point
        ax.plot(0, 0, 'k*', markersize=15, label='Branch point', zorder=6)
        
        # Annotate sheet transitions
        sheet_start = 0
        sheet_end = n_loops % 3
        ax.text(0.5, -1.6, f'Sheet: ${sheet_start} \\to {sheet_end}$', 
               fontsize=12, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.8))
        
        apply_axis_styling(
            ax,
            xlabel=r'$\mathrm{Re}(x)$',
            ylabel=r'$\mathrm{Im}(x)$',
            title=title,
            rotate_xticks=True
        )
        styled_legend(ax, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.8, 1.5)
        add_headroom(ax)
    
    plt.suptitle('Monodromy: Cyclic Sheet Transitions Around Branch Point', 
                fontsize=AXIS_FONT + 3, fontweight='bold', y=1.02)
    
    # Add global deck map explanation
    fig.text(0.5, 0.01, r'Deck map: After one loop $\gamma$, sheet permutation given by $g(w) = e^{2\pi i/m} w$ (rotation by $2\pi/m$)', 
            fontsize=AXIS_FONT, ha='center', style='italic', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['highlight'], alpha=0.95))
    
    save_figure(fig, 'fig_monodromy_path.pdf', 'section4')
    save_figure(fig, 'fig_monodromy_path.png', 'section4')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def create_all_figures():
    """Generate all figures for the report."""
    print("\n" + "="*70)
    print(" PUISEUX-NEWTON-RIEMANN VISUALIZATION SUITE")
    print(" Publication-Quality Figure Generation")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nColor palette:")
    print(f"  Primary: {COLORS['primary']}")
    print(f"  Secondary: {COLORS['secondary']}")
    print(f"  Accents: {COLORS['accent1']}, {COLORS['accent2']}")
    print("\n" + "="*70)
    
    # Section 1 figures
    create_fig1a_solvability_map()
    create_fig1b_powerline_bundle()
    
    # Section 3 figures  
    create_figNP1_newton_polygon_general()
    create_figNP2_newton_polygon_mother()
    
    # Section 4 figures
    create_fig_riemann_lift()
    create_fig_threeleaf_structure()
    create_fig_monodromy_path()
    
    print("\n" + "="*70)
    print(" ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nTotal output files: {len(list(OUTPUT_DIR.rglob('*.*')))}")
    print(f"Location: {OUTPUT_DIR}")
    
    # Create summary README
    create_readme()

def create_readme():
    """Generate README documentation for the figures."""
    readme_content = """# Puiseux-Newton-Riemann Visualizations

## Overview
Publication-quality figures for the mathematical report:
*"From Puiseux Series to Newton Polygons and Riemann Surfaces:
A Unified Framework for Reducibility of ODEs near x=0"*

Generated: 2025-11-13

## Color Palette

### Natural Vintage (Warm)
- Light Blue: #acd2d6
- Sage Green: #91a5a1
- Cream: #fff8db
- Gold: #e6b451
- Dark Green: #5e6e54

### Contrast Palette (Cool)
- Dark Teal: #1F3A3D
- Navy: #2C3E50
- Medium Blue: #4B8DA3
- Light Cyan: #A8DADC
- Pale White: #F1FAEE

## Figure Inventory

### Section 1: Introduction
- **fig1a_solvability_map**: (α,β) plane heatmap showing Puiseux step p = gcd_Q{2-α, 1-β}
- **fig1b_powerline_bundle**: Three affine lines demonstrating exponent alignment

### Section 3: Newton Polygon
- **figNP1_newton_polygon_general**: General construction with three points and lower hull
- **figNP2_newton_polygon_mother**: Specific example for 9x^(4/3)y'' + 6x^(1/3)y' + y = 0

### Section 4: Riemann Surfaces
- **fig_riemann_lift**: Conceptual lifting from multi-valued to single-valued functions
- **fig_threeleaf_structure**: 3D three-sheeted Riemann surface for x^(1/3)
- **fig_monodromy_path**: Monodromy visualization showing 1, 2, 3 loops

## File Formats
All figures provided in both:
- PDF (vector graphics, recommended for LaTeX)
- PNG (raster graphics, 300 DPI)

## Usage in LaTeX
```latex
\\includegraphics[width=0.8\\textwidth]{figure/section1/fig1a_solvability_map.pdf}
```

## Technical Details
- Resolution: 300 DPI
- Font: Times New Roman (serif)
- Grid: Enabled with 30% alpha
- Aspect ratios: Optimized for A4 paper

## Contact
For questions or modifications, refer to the generating script:
`Puiseux-Newton-Riemann Visualizations.py`
"""
    
    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"\n✓ Created documentation: {readme_path}")

# ============================================================================
# RUN ALL
# ============================================================================

if __name__ == "__main__":
    create_all_figures()
    print("\n✓ Visualization suite complete!\n")
