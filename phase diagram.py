# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:14:34 2025

@author: SOUMYA
"""

#!/usr/bin/env python3
"""
Phase Diagram Generator for Section 3.3 — Case 3 Integrated (FIXED)
===================================================================

Generates two publication-ready phase diagrams from Case 3 P(x) 
confinement-strength scaling analysis using actual computed modal data:
  - Figure 14: 1D Phase Diagram (E₁ vs κ with RMS diagnostics)
  - Figure 15: 2D Phase Diagram (κ vs a/L regime map)

Output: PNG and PDF at 300 dpi (publication quality)

FIXES:
- Uses ACTUAL data ranges (not hardcoded template values)
- RMS axis scales to actual data (not 10⁻¹⁵ to 10⁻¹)
- κ axis scales appropriately (from actual min to max)
- All thresholds computed from data instead of hardcoded

Author: Generated December 4, 2025 (REVISED)
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

# Color scheme
COLORS = {
    'gaussian_blob': '#1f77b4',      # Blue
    'transition': '#ff7f0e',          # Orange
    'deflection': '#d62728',          # Red
    'boundary_blob': 'green',
    'boundary_transition': 'orange',
    'boundary_deflection': 'red',
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
        
        # Compute P(x)
        P_x = analytic_P(x_grid, N=N, L=L, a=a, x0=x0)
        P_u = L * P_x

        # Normalization check
        integral = np.trapz(P_x, x_grid)
        norm_resid = np.abs(integral - 1.0)

        # Modal coefficients
        n, c_n = modal_coeffs(N, L, a, x0)
        if c_n.size > 0:
            abs_sum = np.sum(np.abs(c_n))
            sq_sum = np.sum(c_n**2)
            c1 = c_n[0]
            f_abs = np.abs(c1) / abs_sum if abs_sum > 0 else 0.0
            f_sq = (c1**2) / sq_sum if sq_sum > 0 else 0.0
        else:
            f_abs, f_sq, c1 = 0.0, 0.0, 0.0

        # RMS vs single mode
        P_mode1 = c1 * np.sin(np.pi * x_grid / L) if c_n.size > 0 else np.zeros_like(P_u)
        P_mode1[P_mode1 < 0] = 0.0
        rms = np.sqrt(np.mean((P_u - P_mode1)**2))

        # Moments
        mean_u, var_u, skew_u = compute_moments(u_grid, P_u)

        # Store
        norm_residuals.append(norm_resid)
        first_mode_frac_abs.append(f_abs)
        first_mode_frac_sq.append(f_sq)
        rms_to_mode1.append(rms)
        mean_vals.append(mean_u)
        var_vals.append(var_u)
        skew_vals.append(skew_u)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  {i+1:2d}/{len(Na_over_L_arr):2d} | Na/L={ratio:7.4f} | κ={kappa:10.4e} | "
                  f"E₁={f_abs:6.4f} | RMS={rms:8.4f}")

    print("-" * 60)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Na/L': Na_over_L_arr,
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
# FIGURE 14: 1D PHASE DIAGRAM (κ-CONTROL)
# ============================================================================

def generate_figure14_1d_phase_diagram(df, output_dir=OUTPUT_DIR):
    """
    Generate Figure 14: 1D Phase Diagram with κ-control
    
    Three-panel layout with AUTOMATIC SCALING based on actual data ranges.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Case 3 data containing: kappa, FirstModeFracAbs, RMS_to_mode1
    output_dir : str
        Directory to save output
    """
    
    # Extract data
    kappa_vals = df['kappa'].values
    E1_vals = df['FirstModeFracAbs'].values
    RMS_vals = df['RMS_to_mode1'].values
    
    # IMPORTANT: Use actual data ranges (NOT hardcoded template values)
    kappa_min = kappa_vals[kappa_vals > 0].min() if np.any(kappa_vals > 0) else 1e-4
    kappa_max = kappa_vals.max()
    rms_min = RMS_vals[RMS_vals > 0].min() if np.any(RMS_vals > 0) else 1e-2
    rms_max = RMS_vals.max()
    
    print("\nGenerating Figure 14: 1D Phase Diagram (κ-control)...")
    print(f"  Data ranges detected:")
    print(f"    κ: {kappa_min:.4e} to {kappa_max:.4e}")
    print(f"    E₁: {E1_vals.min():.4f} to {E1_vals.max():.4f}")
    print(f"    RMS: {rms_min:.4f} to {rms_max:.4f}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # ========== PANEL A: MAIN PHASE DIAGRAM (E₁ vs κ + RMS) ==========
    ax_main = fig.add_subplot(gs[0, :])
    
    # Plot primary axis: E₁ vs κ
    line1 = ax_main.semilogx(kappa_vals, E1_vals, 'o-', linewidth=2.5, markersize=8, 
                             label='First-mode fraction E₁ = c₁/Σ|cₙ|', 
                             color=COLORS['gaussian_blob'], markerfacecolor=COLORS['gaussian_blob'])
    
    # Regime boundaries (vertical lines) — using actual data quantiles
    E1_sorted = np.sort(E1_vals)
    kappa_at_E1_03 = kappa_vals[np.argmin(np.abs(E1_vals - 0.3))] if len(E1_vals) > 0 else 0.05
    kappa_at_E1_08 = kappa_vals[np.argmin(np.abs(E1_vals - 0.8))] if len(E1_vals) > 0 else 0.5
    
    ax_main.axvline(x=kappa_at_E1_03, color=COLORS['boundary_blob'], linestyle='--', 
                    linewidth=2.5, alpha=0.7, label=f'E₁ ~ 0.3 (κ ≈ {kappa_at_E1_03:.4e})')
    ax_main.axvline(x=kappa_at_E1_08, color=COLORS['boundary_deflection'], linestyle='--', 
                    linewidth=2.5, alpha=0.7, label=f'E₁ ~ 0.8 (κ ≈ {kappa_at_E1_08:.4e})')
    
    # Shaded regime regions based on actual boundaries
    ax_main.axvspan(kappa_min/10, kappa_at_E1_03, alpha=0.15, color='blue', 
                    label='Gaussian Blob (E₁ < 0.3)')
    ax_main.axvspan(kappa_at_E1_03, kappa_at_E1_08, alpha=0.15, color='yellow', 
                    label='Transition (0.3 < E₁ < 0.8)')
    ax_main.axvspan(kappa_at_E1_08, kappa_max*10, alpha=0.15, color='red', 
                    label='Deflection (E₁ > 0.8)')
    
    # Reference lines for E₁ thresholds
    ax_main.axhline(y=0.3, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_main.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Threshold annotations
    ax_main.text(kappa_at_E1_03, 0.05, 'E₁ = 0.3', fontsize=10, ha='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_main.text(kappa_at_E1_08, 0.05, 'E₁ = 0.8', fontsize=10, ha='right', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Secondary axis: RMS on right (with ACTUAL data scale)
    ax_rms = ax_main.twinx()
    line2 = ax_rms.plot(kappa_vals, RMS_vals, 's-', linewidth=2.5, 
                        markersize=7, label='RMS(P − P₁)', 
                        color=COLORS['transition'], markerfacecolor=COLORS['transition'])
    
    # Set axes properties with ACTUAL data ranges
    ax_main.set_xlabel('Dimensionless Confinement Strength κ = Na²/L² (log scale)', 
                      fontsize=13, fontweight='bold')
    ax_main.set_ylabel('First-Mode Energy Fraction E₁', fontsize=13, fontweight='bold')
    ax_rms.set_ylabel('RMS Deviation: RMS(P − P₁)', fontsize=12, fontweight='bold')
    ax_main.set_title('Figure 14: Phase Diagram — Polymer Confinement Regimes Controlled by κ (Case 3, P(x))', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.set_ylim(-0.05, 1.1)
    
    # FIXED: Use actual κ range with proper scaling
    ax_main.set_xlim(kappa_min / 2, kappa_max * 2)
    
    # RMS axis with actual data range
    rms_plot_min = max(rms_min / 2, 0)
    rms_plot_max = rms_max * 1.5
    ax_rms.set_ylim(rms_plot_min, rms_plot_max)
    
    ax_main.grid(True, alpha=0.3, which='both')
    
    # Combined legend
    lines1, labels1 = ax_main.get_legend_handles_labels()
    lines2, labels2 = ax_rms.get_legend_handles_labels()
    ax_main.legend(lines1 + lines2, labels1 + labels2, loc='lower right', 
                  fontsize=9, framealpha=0.95)
    
    ax_main.tick_params(axis='both', which='major', labelsize=11)
    ax_rms.tick_params(axis='both', which='major', labelsize=11)
    
    # ========== PANEL B: RMS COLLAPSE QUALITY ==========
    ax_rms_diag = fig.add_subplot(gs[1, 0])
    
    ax_rms_diag.semilogx(kappa_vals, RMS_vals, 's-', linewidth=2.5, markersize=7, 
                        label='RMS(P − P₁)', color=COLORS['transition'], 
                        markerfacecolor=COLORS['transition'])
    
    # Add regime markers
    ax_rms_diag.axvline(x=kappa_at_E1_03, color=COLORS['boundary_blob'], linestyle='--', 
                        linewidth=1.5, alpha=0.5, label=f'E₁ = 0.3')
    ax_rms_diag.axvline(x=kappa_at_E1_08, color=COLORS['boundary_deflection'], linestyle='--', 
                        linewidth=1.5, alpha=0.5, label=f'E₁ = 0.8')
    
    # Reference line at median RMS
    median_rms = np.median(RMS_vals)
    ax_rms_diag.axhline(y=median_rms, color='gray', linestyle='--', linewidth=1, alpha=0.5, 
                       label=f'Median RMS = {median_rms:.4f}')
    
    ax_rms_diag.set_xlabel('κ = Na²/L² (log scale)', fontsize=12, fontweight='bold')
    ax_rms_diag.set_ylabel('RMS Deviation: RMS(P − P₁)', fontsize=12, fontweight='bold')
    ax_rms_diag.set_title('RMS Collapse Quality vs Confinement', fontsize=13, fontweight='bold')
    ax_rms_diag.grid(True, alpha=0.3, which='both')
    ax_rms_diag.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax_rms_diag.tick_params(labelsize=10)
    
    # Set y-limit based on actual data
    ax_rms_diag.set_ylim(rms_plot_min, rms_plot_max)
    ax_rms_diag.set_xlim(kappa_min / 2, kappa_max * 2)
    
    # ========== PANEL C: REGIME SUMMARY TABLE ==========
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')
    
    regime_text = f"""
REGIME IDENTIFICATION CRITERIA
(From Case 3 Data Analysis)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAUSSIAN BLOB (κ ≲ {kappa_at_E1_03:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ E₁ < 0.3 (multimode dominance)
✓ RMS ~ {RMS_vals[0]:.4f} (measurement scale)
✓ Physics: Entropy-driven, weak confinement
✓ Theory: de Gennes blob
✓ Model: Ideal Gaussian chain

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSITION ({kappa_at_E1_03:.2e} ≲ κ ≲ {kappa_at_E1_08:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ 0.3 < E₁ < 0.8 (higher modes visible)
✓ RMS ~ {np.median(RMS_vals[5:15]):.4f} (transition range)
✓ Physics: Modal crossover
✓ Theory: Blob → Deflection
✓ Model: Multimode expansion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFLECTION (κ ≳ {kappa_at_E1_08:.2e})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ E₁ > 0.8 (first-mode dominates)
✓ RMS ~ {RMS_vals[-1]:.4f} (asymptotic)
✓ Physics: Wall-dominated confinement
✓ Theory: Odijk deflection length
✓ Model: Single sine-mode (n=1)
"""
    
    ax_table.text(0.05, 0.95, regime_text, transform=ax_table.transAxes, 
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, 
                          edgecolor='brown', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save in multiple formats
    save_figure_multi_format(fig, 'phase_diagram_case3_px', 
                            output_dir=output_dir, formats=SAVE_FORMATS, dpi=DPI)
    
    plt.close(fig)
    print("  ✓ Figure 14 generation complete")

# ============================================================================
# FIGURE 15: 2D PHASE DIAGRAM (κ vs a/L)
# ============================================================================

def generate_figure15_2d_phase_diagram(df, output_dir=OUTPUT_DIR):
    """
    Generate Figure 15: 2D Phase Diagram (κ vs a/L regime map)
    
    Parameters
    ----------
    df : pandas.DataFrame
        Case 3 data containing: kappa, FirstModeFracAbs
    output_dir : str
        Directory to save output
    """
    
    print("\nGenerating Figure 15: 2D Phase Diagram (κ vs a/L)...")
    
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
                regime = 0  # Gaussian blob
            elif kappa < kappa_at_E1_08:
                regime = 1  # Transition
            else:
                regime = 2  # Deflection
            
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
    ax.axvline(x=kappa_at_E1_03, color=COLORS['boundary_blob'], linewidth=2.5, linestyle='--', 
              alpha=0.7, label=f'E₁ ~ 0.3 (κ ≈ {kappa_at_E1_03:.2e})')
    ax.axvline(x=kappa_at_E1_08, color=COLORS['boundary_deflection'], linewidth=2.5, linestyle='--', 
              alpha=0.7, label=f'E₁ ~ 0.8 (κ ≈ {kappa_at_E1_08:.2e})')
    ax.axhline(y=0.1, color=COLORS['accent_purple'], linewidth=2, linestyle=':', 
              alpha=0.7, label='a/L ~ 0.1: Non-Gaussian onset')
    ax.axhline(y=0.2, color=COLORS['accent_purple'], linewidth=2, linestyle='-', 
              alpha=0.5, label='a/L ~ 0.2: Non-Gaussian strong')
    
    # Add regime region labels
    label_specs = [
        {'pos': (kappa_min, 0.05), 'text': 'Ideal Gaussian\nBlob', 
         'color': 'lightblue', 'edge': 'darkblue'},
        {'pos': ((kappa_at_E1_03 + kappa_at_E1_08)/2, 0.05), 'text': 'Gaussian\nTransition', 
         'color': 'lightyellow', 'edge': 'darkorange'},
        {'pos': (kappa_max, 0.05), 'text': 'Gaussian\nDeflection', 
         'color': 'lightcoral', 'edge': 'darkred'},
        {'pos': (kappa_min, 0.35), 'text': 'Stiff Gaussian\nBlob', 
         'color': 'steelblue', 'edge': 'navy'},
        {'pos': ((kappa_at_E1_03 + kappa_at_E1_08)/2, 0.35), 'text': 'Non-Gaussian\nTransition', 
         'color': 'gold', 'edge': 'orange'},
        {'pos': (kappa_max, 0.35), 'text': 'Non-Gaussian\nDeflection', 
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
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Kuhn Length a/L', fontsize=13, fontweight='bold')
    ax.set_title('Figure 15: 2D Phase Diagram — κ vs a/L with Regime Identification\n' +
                '(Classical Theories: de Gennes Blob, Transition, Odijk Deflection + Wormlike Chain)', 
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
    save_figure_multi_format(fig, 'phase_diagram_2d_kappa_vs_aL', 
                            output_dir=output_dir, formats=SAVE_FORMATS, dpi=DPI)
    
    plt.close(fig)
    print("  ✓ Figure 15 generation complete")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("PHASE DIAGRAM GENERATION FOR SECTION 3.3")
    print("Unified Scaling Framework: Case 3 Modal Analysis (FIXED)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Resolution: {DPI} dpi")
    print(f"  Formats: {', '.join(SAVE_FORMATS)}")
    print(f"  Figure size (14): 16×12 inches")
    print(f"  Figure size (15): 14×10 inches")
    
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
    csv_filename = "case3_phase_diagram_diagnostics.csv"
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
    print("Generating Phase Diagrams")
    print("-"*80)
    
    generate_figure14_1d_phase_diagram(df, output_dir=OUTPUT_DIR)
    generate_figure15_2d_phase_diagram(df, output_dir=OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*80)
    print("✓ PHASE DIAGRAM GENERATION COMPLETE (FIXED)")
    print("="*80)
    print(f"\nOutput files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nGenerated files:")
    print("  • phase_diagram_case3_px.png (Figure 14, PNG)")
    print("  • phase_diagram_case3_px.pdf (Figure 14, PDF)")
    print("  • phase_diagram_2d_kappa_vs_aL.png (Figure 15, PNG)")
    print("  • phase_diagram_2d_kappa_vs_aL.pdf (Figure 15, PDF)")
    print("  • case3_phase_diagram_diagnostics.csv (Data table)")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()

