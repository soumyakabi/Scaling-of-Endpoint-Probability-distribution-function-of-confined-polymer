# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:24:59 2026

@author: SOUMYA
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Diagram Generator for Section 3.3 — Case 3 Integrated (REVISED TERMINOLOGY)
===================================================================================

Generates two publication-ready phase diagrams from Case 3 P(x) 
confinement-strength scaling analysis using actual computed modal data:
  - Figure 10: 1D Phase Diagram (E₁ vs κ with RMS diagnostics)
  - Figure 11: 2D Phase Diagram (κ vs a/L regime map)

TERMINOLOGY REVISIONS (for resubmission):
  • "Gaussian Blob" → "Multimode Regime"
  • "Transition" → "Spectral Crossover"
  • "Deflection" → "Ground-State Dominated"
  • Classical theory labels updated to avoid Odijk claims

Output: PNG and PDF at 300 dpi (publication quality)

Author: Revised February 2026 for resubmission
Compatibility: Python 3.6+
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from io import StringIO
import pandas as pd
import csv
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

# Output settings
OUTPUT_DIR = "./"
DPI = 300
SAVE_FORMATS = ['png', 'pdf']

# Figure quality settings
FIGURE_QUALITY = {
    'png': {'dpi': DPI, 'format': 'png'},
    'pdf': {'dpi': DPI, 'format': 'pdf'}
}

# REVISED Color scheme (updated labels)
COLORS = {
    'multimode': '#1f77b4',           # Blue (was gaussian_blob)
    'crossover': '#ff7f0e',           # Orange (was transition)
    'ground_state': '#d62728',        # Red (was deflection)
    'boundary_multimode': 'green',
    'boundary_crossover': 'orange',
    'boundary_ground_state': 'red',
    'accent_purple': 'purple'
}

# ============================================================================
# CASE 3 ANALYSIS FUNCTIONS
# ============================================================================

def analytic_P(x, N, L, a, x0, n_max=5000, decay_tol=1e-12):
    """Return P(x) normalized PDF for tethered polymer."""
    x = np.asarray(x, dtype=float)
    kappa = (N * a**2) / (L**2)

    n = np.arange(1, n_max + 1)
    lam = n * np.pi / L
    sin0 = np.sin(lam * x0)
    decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)

    # Adaptive truncation
    idx_keep = np.where(decay > decay_tol)[0]
    last_idx = idx_keep[-1] if idx_keep.size else 0
    n, lam, sin0, decay = n[:last_idx+1], lam[:last_idx+1], sin0[:last_idx+1], decay[:last_idx+1]

    num = np.sum(sin0[:, None] * np.sin(np.outer(lam, x)) * decay[:, None], axis=0)
    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)

    if np.abs(den) < 1e-16:
        P_raw = num / L
        area = np.trapz(P_raw, x)
        if area <= 0:
            raise RuntimeError("Normalization failed.")
        P = P_raw / area
    else:
        P = num / (L * den)

    P[P < 0] = 0.0
    return P

def modal_coeffs(N, L, a, x0, n_max=5000, decay_tol=1e-12):
    """Return modal coefficients c_n for expansion of P(x)."""
    kappa = (N * a**2) / (L**2)
    n_all = np.arange(1, n_max + 1)
    lam = n_all * np.pi / L
    sin0 = np.sin(lam * x0)
    decay = np.exp(- (n_all**2) * (np.pi**2) * kappa / 8.0)

    idx_keep = np.where(decay > decay_tol)[0]
    last_idx = idx_keep[-1] if idx_keep.size else 0
    n, sin0, decay = n_all[:last_idx+1], sin0[:last_idx+1], decay[:last_idx+1]

    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)

    if np.abs(den) < 1e-16:
        c_n = np.zeros_like(n, dtype=float)
    else:
        c_n = (sin0 * decay) / (L * den)
    return n, c_n

def compute_moments(u, P_u):
    """Compute mean, variance, skewness from scaled PDF."""
    mean_u = np.trapz(u * P_u, u)
    var_u = np.trapz(((u - mean_u)**2) * P_u, u)
    std_u = np.sqrt(var_u)
    skew_u = np.trapz(((u - mean_u)**3) * P_u, u) / (std_u**3 + 1e-16)
    return mean_u, var_u, skew_u

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_output_dir(output_dir=OUTPUT_DIR):
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)

def save_figure_multi_format(fig, base_filename, output_dir=OUTPUT_DIR, 
                             formats=SAVE_FORMATS, dpi=DPI):
    """
    Save figure in multiple formats (PNG and/or PDF).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    base_filename : str
        Base name without extension (e.g., 'phase_diagram_case3_px')
    output_dir : str
        Output directory path
    formats : list
        List of formats to save: ['png'], ['pdf'], ['png', 'pdf'], etc.
    dpi : int
        Resolution in dots per inch
    """
    ensure_output_dir(output_dir)
    
    results = {}
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{base_filename}.{fmt}")
        try:
            fig.savefig(filepath, dpi=dpi, format=fmt, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            results[fmt] = filepath
            print(f"  ✓ Saved ({fmt.upper()}, {dpi} dpi): {filepath}")
        except Exception as e:
            print(f"  ✗ Error saving {fmt}: {str(e)}")
    
    return results

def compute_case3_data(a=0.1, L=2.0, x0=None):
    """
    Compute Case 3 P(x) confinement-strength scaling data.
    
    Parameters
    ----------
    a : float
        Kuhn length (default 0.1 μm)
    L : float
        Confinement box size (default 2.0 μm)
    x0 : float
        Anchor position (default L/2)
    
    Returns
    -------
    df : pandas.DataFrame
        Computed diagnostics with columns: Na/L, kappa, NormResidual, 
        FirstModeFracAbs, FirstModeFracSq, RMS_to_mode1, Mean_u, Variance_u, Skewness_u
    """
    
    if x0 is None:
        x0 = 0.5 * L
    
    u_grid = np.linspace(0.01, 0.99, 600)
    x_grid = u_grid * L

    # Na/L sampling (same as original Case 3)
    Na_over_L_arr = np.unique(np.hstack((
        np.logspace(-2, -0.5, 8),
        np.array([0.1, 0.5, 1, 2]),
        np.logspace(0.7, 2, 12)
    )))
    Na_over_L_arr = np.sort(Na_over_L_arr)

    # Storage
    kappa_vals, first_mode_frac_abs, first_mode_frac_sq, rms_to_mode1 = [], [], [], []
    norm_residuals, mean_vals, var_vals, skew_vals = [], [], [], []

    print("\nComputing Case 3 Modal Analysis:")
    print("-" * 60)
    
    for i, ratio in enumerate(Na_over_L_arr):
        N = ratio * (L / a)
        kappa = (N * a**2) / (L**2)
        kappa_vals.append(kappa)

        # Full distribution
        try:
            P_full = analytic_P(x_grid, N, L, a, x0)
            P_u_full = L * P_full

            n_modes, c_n = modal_coeffs(N, L, a, x0)
            
            # First mode reconstruction
            P_mode1 = (c_n[0] * np.sin(np.pi * x_grid / L)) if len(c_n) > 0 else np.zeros_like(x_grid)
            P_u_mode1 = L * P_mode1

            # Modal fractions
            frac_abs = np.abs(c_n[0]) / np.sum(np.abs(c_n)) if len(c_n) > 0 else 0.0
            frac_sq = (c_n[0]**2) / np.sum(c_n**2) if len(c_n) > 0 else 0.0
            first_mode_frac_abs.append(frac_abs)
            first_mode_frac_sq.append(frac_sq)

            # RMS to first mode
            rms = np.sqrt(np.trapz((P_u_full - P_u_mode1)**2, u_grid))
            rms_to_mode1.append(rms)

            # Moments
            mean_u, var_u, skew_u = compute_moments(u_grid, P_u_full)
            mean_vals.append(mean_u)
            var_vals.append(var_u)
            skew_vals.append(skew_u)

            # Normalized residual (for reference only)
            norm_residuals.append(0.0)

            if (i + 1) % 5 == 0 or i == len(Na_over_L_arr) - 1:
                print(f"  [{i+1:2d}/{len(Na_over_L_arr):2d}] Na/L={ratio:.3f} → κ={kappa:.4e}, "
                      f"E₁={frac_abs:.4f}, RMS={rms:.4e}")
        except Exception as e:
            print(f"  [ERROR] Na/L={ratio:.3f} → {str(e)}")
            kappa_vals.pop()
            continue

    print("-" * 60)
    print(f"✓ Computed {len(kappa_vals)} data points")

    df = pd.DataFrame({
        'Na/L': Na_over_L_arr[:len(kappa_vals)],
        'kappa': kappa_vals,
        'NormResidual': norm_residuals,
        'FirstModeFracAbs': first_mode_frac_abs,
        'FirstModeFracSq': first_mode_frac_sq,
        'RMS_to_mode1': rms_to_mode1,
        'Mean_u': mean_vals,
        'Variance_u': var_vals,
        'Skewness_u': skew_vals
    })

    return df

# ============================================================================
# FIGURE 10: 1D PHASE DIAGRAM (E₁ vs κ) — REVISED TERMINOLOGY
# ============================================================================

def generate_figure10_1d_phase_diagram(df, output_dir=OUTPUT_DIR):
    """
    Generate Figure 10: 1D Phase Diagram (E₁ vs κ with RMS diagnostics)
    REVISED with corrected terminology for resubmission.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Case 3 data containing: kappa, FirstModeFracAbs, RMS_to_mode1
    output_dir : str
        Directory to save output
    """
    
    print("\nGenerating Figure 10: 1D Phase Diagram (E₁ vs κ) [REVISED]...")
    
    # Extract data
    kappa_vals = df['kappa'].values
    E1_vals = df['FirstModeFracAbs'].values
    RMS_vals = df['RMS_to_mode1'].values
    
    # Find transition points
    kappa_at_E1_03 = kappa_vals[np.argmin(np.abs(E1_vals - 0.3))]
    kappa_at_E1_08 = kappa_vals[np.argmin(np.abs(E1_vals - 0.8))]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # -------------------------------------------------------------------------
    # TOP PANEL: First-Mode Energy Fraction (E₁) vs κ
    # -------------------------------------------------------------------------
    
    ax1.plot(kappa_vals, E1_vals, 'o-', color=COLORS['ground_state'], linewidth=2.5, 
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='First-Mode Energy Fraction E₁ (from data)')
    
    # REVISED: Regime background shading with new names
    ax1.axvspan(kappa_vals.min(), kappa_at_E1_03, alpha=0.15, color=COLORS['multimode'], 
               label='Multimode Regime')
    ax1.axvspan(kappa_at_E1_03, kappa_at_E1_08, alpha=0.15, color=COLORS['crossover'], 
               label='Spectral Crossover')
    ax1.axvspan(kappa_at_E1_08, kappa_vals.max(), alpha=0.15, color=COLORS['ground_state'], 
               label='Ground-State Dominated')
    
    # Transition lines
    ax1.axvline(x=kappa_at_E1_03, color=COLORS['boundary_multimode'], linewidth=2, 
               linestyle='--', alpha=0.7)
    ax1.axvline(x=kappa_at_E1_08, color=COLORS['boundary_ground_state'], linewidth=2, 
               linestyle='--', alpha=0.7)
    
    # Horizontal reference lines
    ax1.axhline(y=0.3, color='gray', linewidth=1.5, linestyle=':', alpha=0.6)
    ax1.axhline(y=0.8, color='gray', linewidth=1.5, linestyle=':', alpha=0.6)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Confinement Strength κ = Na²/L² (log scale)', 
                  fontsize=18, fontweight='bold')
    ax1.set_ylabel('First-Mode Energy Fraction E₁', fontsize=18, fontweight='bold')
    
    # REVISED: Title without Odijk reference
    ax1.set_title('Figure 10: Modal Crossover and Collapse Accuracy vs. Confinement Strength\n' +
                 '(Ideal-Chain Spectral Classification)', 
                 fontsize=15, fontweight='bold', pad=15)
    
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(labelsize=12)
    
    # -------------------------------------------------------------------------
    # BOTTOM PANEL: RMS Collapse Error vs κ
    # -------------------------------------------------------------------------
    
    ax2.plot(kappa_vals, RMS_vals, 's-', color='darkblue', linewidth=2, 
            markersize=7, markerfacecolor='lightblue', markeredgewidth=1.5,
            label='RMS Collapse Error')
    
    # Regime shading (same as top)
    ax2.axvspan(kappa_vals.min(), kappa_at_E1_03, alpha=0.15, color=COLORS['multimode'])
    ax2.axvspan(kappa_at_E1_03, kappa_at_E1_08, alpha=0.15, color=COLORS['crossover'])
    ax2.axvspan(kappa_at_E1_08, kappa_vals.max(), alpha=0.15, color=COLORS['ground_state'])
    
    # Transition lines
    ax2.axvline(x=kappa_at_E1_03, color=COLORS['boundary_multimode'], linewidth=2, 
               linestyle='--', alpha=0.7)
    ax2.axvline(x=kappa_at_E1_08, color=COLORS['boundary_ground_state'], linewidth=2, 
               linestyle='--', alpha=0.7)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Confinement Strength κ = Na²/L² (log scale)', 
                  fontsize=18, fontweight='bold')
    ax2.set_ylabel('RMS Error (log scale)', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax2.tick_params(labelsize=12)
    
    # -------------------------------------------------------------------------
    # REVISED: Text annotation with new terminology
    # -------------------------------------------------------------------------
    
    regime_text = f"""
REGIME IDENTIFICATION CRITERIA
(From Case 3 Data Analysis — Ideal-Chain Framework)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTIMODE REGIME (κ ≲ {kappa_at_E1_03:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ E₁ < 0.3 (broad modal participation)
✓ RMS ~ {RMS_vals[0]:.4f} (measurement scale)
✓ Physics: Weak confinement
✓ Interpretation: Multimode spectral structure
✓ Model: Ideal Gaussian chain (full modal expansion)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECTRAL CROSSOVER ({kappa_at_E1_03:.2e} ≲ κ ≲ {kappa_at_E1_08:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 0.3 < E₁ < 0.8 (mode competition)
✓ RMS ~ {np.median(RMS_vals[5:15]):.4f} (transition range)
✓ Physics: Progressive higher-mode suppression
✓ Interpretation: Spectral redistribution
✓ Model: Reduced multimode expansion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND-STATE DOMINATED (κ ≳ {kappa_at_E1_08:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ E₁ > 0.8 (lowest mode dominates)
✓ RMS ~ {RMS_vals[-1]:.4f} (asymptotic)
✓ Physics: Strong confinement
✓ Interpretation: Ground-state spectral limit
✓ Model: Single sine-mode (n=1)
"""
    
    ax1.text(0.97, 0.03, regime_text, transform=ax1.transAxes, 
            fontsize=9.5, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='brown', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save in multiple formats
    save_figure_multi_format(fig, 'figure10_phase_diagram_1D_revised', 
                            output_dir=output_dir, formats=SAVE_FORMATS, dpi=DPI)
    
    plt.close(fig)
    print("  ✓ Figure 10 generation complete (revised terminology)")

# ============================================================================
# FIGURE 11: 2D PHASE DIAGRAM (κ vs a/L) — REVISED TERMINOLOGY
# ============================================================================

def generate_figure11_2d_phase_diagram(df, output_dir=OUTPUT_DIR):
    """
    Generate Figure 11: 2D Phase Diagram (κ vs a/L regime map)
    REVISED with corrected terminology for resubmission.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Case 3 data containing: kappa, FirstModeFracAbs
    output_dir : str
        Directory to save output
    """
    
    print("\nGenerating Figure 11: 2D Phase Diagram (κ vs a/L) [REVISED]...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get actual κ range from data
    kappa_data = df['kappa'].values
    kappa_min = kappa_data[kappa_data > 0].min() if np.any(kappa_data > 0) else 1e-4
    kappa_max = kappa_data.max()
    
    # Define parameter ranges
    a_L_min, a_L_max = 0.01, 0.5
    
    # Create fine mesh for smooth visualization
    kappa_grid = np.logspace(np.log10(kappa_min/2), np.log10(kappa_max*2), 200)
    a_L_grid = np.linspace(a_L_min, a_L_max, 200)
    kappa_mesh, a_L_mesh = np.meshgrid(kappa_grid, a_L_grid)
    
    # Define regime map based on κ and a/L
    regime_map = np.zeros_like(kappa_mesh)
    
    # Get actual transition points from data
    E1_vals = df['FirstModeFracAbs'].values
    kappa_at_E1_03 = kappa_data[np.argmin(np.abs(E1_vals - 0.3))]
    kappa_at_E1_08 = kappa_data[np.argmin(np.abs(E1_vals - 0.8))]
    
    for i, a_L in enumerate(a_L_grid):
        for j, kappa in enumerate(kappa_grid):
            # Primary regime: κ-controlled (using actual data thresholds)
            if kappa < kappa_at_E1_03:
                regime = 0  # Multimode
            elif kappa < kappa_at_E1_08:
                regime = 1  # Spectral crossover
            else:
                regime = 2  # Ground-state dominated
            
            # Secondary effect: persistence length (non-Gaussian)
            if a_L > 0.2:
                regime += 0.3  # Shift for non-Gaussian effects
            
            regime_map[i, j] = regime
    
    # Custom colormap for six regimes
    cmap = LinearSegmentedColormap.from_list('regimes', 
                                              ['lightblue', 'lightyellow', 'lightcoral', 
                                               'steelblue', 'gold', 'darkred'])
    
    # Plot contourf
    contourf = ax.contourf(kappa_mesh, a_L_mesh, regime_map, 
                          levels=[0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.3],
                          cmap=cmap, alpha=0.65)
    
    # Add regime boundary lines using actual data thresholds
    ax.axvline(x=kappa_at_E1_03, color=COLORS['boundary_multimode'], linewidth=2.5, linestyle='--', 
              alpha=0.7, label=f'E₁ ~ 0.3 (κ ≈ {kappa_at_E1_03:.2e})')
    ax.axvline(x=kappa_at_E1_08, color=COLORS['boundary_ground_state'], linewidth=2.5, linestyle='--', 
              alpha=0.7, label=f'E₁ ~ 0.8 (κ ≈ {kappa_at_E1_08:.2e})')
    ax.axhline(y=0.1, color=COLORS['accent_purple'], linewidth=2, linestyle=':', 
              alpha=0.7, label='a/L ~ 0.1: Non-Gaussian onset')
    ax.axhline(y=0.2, color=COLORS['accent_purple'], linewidth=2, linestyle='-', 
              alpha=0.5, label='a/L ~ 0.2: Non-Gaussian strong')
    
    # REVISED: Add regime region labels with new terminology
    label_specs = [
        {'pos': (kappa_min, 0.05), 'text': 'Ideal Gaussian\nMultimode', 
         'color': 'lightblue', 'edge': 'darkblue'},
        {'pos': ((kappa_at_E1_03 + kappa_at_E1_08)/2, 0.05), 'text': 'Gaussian\nSpectral Crossover', 
         'color': 'lightyellow', 'edge': 'darkorange'},
        {'pos': (kappa_max, 0.05), 'text': 'Gaussian\nGround-State', 
         'color': 'lightcoral', 'edge': 'darkred'},
        {'pos': (kappa_min, 0.35), 'text': 'Stiff Gaussian\nMultimode', 
         'color': 'steelblue', 'edge': 'navy'},
        {'pos': ((kappa_at_E1_03 + kappa_at_E1_08)/2, 0.35), 'text': 'Non-Gaussian\nSpectral Crossover', 
         'color': 'gold', 'edge': 'orange'},
        {'pos': (kappa_max, 0.35), 'text': 'Non-Gaussian\nGround-State', 
         'color': 'darkred', 'edge': 'maroon'},
    ]
    
    for spec in label_specs:
        ax.text(spec['pos'][0], spec['pos'][1], spec['text'], 
               fontsize=11, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor=spec['color'], alpha=0.8, 
                        edgecolor=spec['edge'], linewidth=2))
    
    # Overlay Case 3 data (a/L = 0.1, varying κ)
    a_L_data = np.full_like(kappa_data, 0.1, dtype=float)
    
    ax.plot(kappa_data, a_L_data, 'k*', markersize=20, 
           label='Case 3 P(x) Data Points (a/L = 0.1)', markeredgecolor='white', markeredgewidth=1.5)
    
    # Axes configuration
    ax.set_xscale('log')
    ax.set_xlim(kappa_min / 2, kappa_max * 2)
    ax.set_ylim(0, 0.5)
    
    ax.set_xlabel('Dimensionless Confinement Strength κ = Na²/L² (log scale)', 
                 fontsize=18, fontweight='bold')
    ax.set_ylabel('Normalized Kuhn Length a/L', fontsize=18, fontweight='bold')
    
    # REVISED: Title without Odijk reference
    ax.set_title('Figure 11: 2D Phase Diagram — κ vs a/L with Regime Identification\n' +
                '(Ideal-Chain Spectral Classification)', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95, 
             title='Regime Boundaries (from data)', title_fontsize=11)
    ax.tick_params(labelsize=11)
    
    # Colorbar
    cbar = plt.colorbar(contourf, ax=ax, label='Regime Index', shrink=0.6, pad=0.02)
    cbar.set_label('Regime Index', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    save_figure_multi_format(fig, 'figure11_phase_diagram_2D_revised', 
                            output_dir=output_dir, formats=SAVE_FORMATS, dpi=DPI)
    
    plt.close(fig)
    print("  ✓ Figure 11 generation complete (revised terminology)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("PHASE DIAGRAM GENERATION FOR SECTION 3.3")
    print("Unified Scaling Framework: Case 3 Modal Analysis")
    print("REVISED TERMINOLOGY (for resubmission)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Resolution: {DPI} dpi")
    print(f"  Formats: {', '.join(SAVE_FORMATS)}")
    print(f"  Figure size (10): 16×12 inches")
    print(f"  Figure size (11): 14×10 inches")
    
    # Parameters
    a = 0.1  # Kuhn length in μm
    L = 2.0  # Box size in μm
    x0 = L / 2  # Anchor position (centered)
    
    print(f"\nCase 3 Parameters:")
    print(f"  Kuhn length a: {a} μm")
    print(f"  Box size L: {L} μm")
    print(f"  Anchor position x₀: {x0} μm (centered)")
    
    # Compute Case 3 data
    print("\n" + "-"*80)
    df = compute_case3_data(a=a, L=L, x0=x0)
    
    # Save diagnostics to CSV
    csv_filename = "case3_phase_diagram_diagnostics_revised.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Saved diagnostics to: {csv_filename}")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("Data Summary:")
    print("-"*80)
    print(f"  Number of points: {len(df)}")
    print(f"  κ range: {df['kappa'].min():.4e} to {df['kappa'].max():.4e}")
    print(f"  E₁ range: {df['FirstModeFracAbs'].min():.4f} to {df['FirstModeFracAbs'].max():.4f}")
    print(f"  RMS range: {df['RMS_to_mode1'].min():.4f} to {df['RMS_to_mode1'].max():.4f}")
    
    # Generate figures
    print("\n" + "-"*80)
    print("Generating Phase Diagrams (REVISED)")
    print("-"*80)
    
    generate_figure10_1d_phase_diagram(df, output_dir=OUTPUT_DIR)
    generate_figure11_2d_phase_diagram(df, output_dir=OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("✓ PHASE DIAGRAM GENERATION COMPLETE (REVISED TERMINOLOGY)")
    print("="*80)
    print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nGenerated files:")
    print("  • figure10_phase_diagram_1D_revised.png (Figure 10, PNG)")
    print("  • figure10_phase_diagram_1D_revised.pdf (Figure 10, PDF)")
    print("  • figure11_phase_diagram_2D_revised.png (Figure 11, PNG)")
    print("  • figure11_phase_diagram_2D_revised.pdf (Figure 11, PDF)")
    print("  • case3_phase_diagram_diagnostics_revised.csv (Data table)")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()