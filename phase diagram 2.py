# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:25:21 2026

@author: SOUMYA
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import pandas as pd
import os
from matplotlib.lines import Line2D

OUTPUT_DIR = "./"
DPI = 300
SAVE_FORMATS = ['png']

COLORS = {
    'multimode': '#1f77b4',
    'crossover': '#ff7f0e',
    'ground_state': '#d62728',
    'boundary_multimode': 'green',
    'boundary_crossover': 'orange',
    'boundary_ground_state': 'red',
    'accent_purple': 'purple',
    'bio_marker': 'black'
}

# Each system: kappa, display name, annotation text x offset (in log space multiplier), annotation y
BIOLOGICAL_SYSTEMS = {
    'systems': [
        {
            'name': '500 bp DNA\n(1 μm nucleus)',
            'kappa': 1.25e-4,
            'color': COLORS['multimode'],
            'marker_y': 0.75,
            'ann_x_mult': 3.5,   # multiply kappa for annotation x
            'ann_y': 0.62,
            'ann_ha': 'left',
            'rad': 0.25
        },
        {
            'name': '5 kb chromatin\n(10 μm nucleus)',
            'kappa': 4.5e-4,
            'color': COLORS['multimode'],
            'marker_y': 0.75,
            'ann_x_mult': 0.18,  # left side
            'ann_y': 0.88,
            'ann_ha': 'right',
            'rad': -0.25
        },
        {
            'name': 'V. cholerae ori1\n(3 μm cell)',
            'kappa': 2.8e-3,
            'color': COLORS['crossover'],
            'marker_y': 0.52,
            'ann_x_mult': 5.0,
            'ann_y': 0.39,
            'ann_ha': 'left',
            'rad': 0.2
        },
        {
            'name': 'E. coli chr.\n(1 μm nucleoid)',
            'kappa': 1.25,
            'color': COLORS['ground_state'],
            'marker_y': 0.92,
            'ann_x_mult': 0.15,
            'ann_y': 0.80,
            'ann_ha': 'right',
            'rad': -0.2
        },
    ]
}

def analytic_P(x, N, L, a, x0, n_max=5000, decay_tol=1e-12):
    x = np.asarray(x, dtype=float)
    kappa = (N * a**2) / (L**2)
    n = np.arange(1, n_max + 1)
    lam = n * np.pi / L
    sin0 = np.sin(lam * x0)
    decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)
    idx_keep = np.where(decay > decay_tol)[0]
    last_idx = idx_keep[-1] if idx_keep.size else 0
    n, lam, sin0, decay = n[:last_idx+1], lam[:last_idx+1], sin0[:last_idx+1], decay[:last_idx+1]
    num = np.sum(sin0[:, None] * np.sin(np.outer(lam, x)) * decay[:, None], axis=0)
    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)
    if np.abs(den) < 1e-16:
        P_raw = num / L
        area = np.trapezoid(P_raw, x)
        if area <= 0:
            raise RuntimeError("Normalization failed.")
        P = P_raw / area
    else:
        P = num / (L * den)
    P[P < 0] = 0.0
    return P

def modal_coeffs(N, L, a, x0, n_max=5000, decay_tol=1e-12):
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
    mean_u = np.trapezoid(u * P_u, u)
    var_u = np.trapezoid(((u - mean_u)**2) * P_u, u)
    std_u = np.sqrt(var_u)
    skew_u = np.trapezoid(((u - mean_u)**3) * P_u, u) / (std_u**3 + 1e-16)
    return mean_u, var_u, skew_u

def ensure_output_dir(output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

def save_figure_multi_format(fig, base_filename, output_dir=OUTPUT_DIR, formats=SAVE_FORMATS, dpi=DPI):
    ensure_output_dir(output_dir)
    results = {}
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{base_filename}.{fmt}")
        try:
            fig.savefig(filepath, dpi=dpi, format=fmt, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            results[fmt] = filepath
            print(f"  Saved ({fmt.upper()}, {dpi} dpi): {filepath}")
        except Exception as e:
            print(f"  Error saving {fmt}: {str(e)}")
    return results

def compute_case3_data(a=0.1, L=2.0, x0=None):
    if x0 is None:
        x0 = 0.5 * L
    u_grid = np.linspace(0.01, 0.99, 600)
    x_grid = u_grid * L
    Na_over_L_arr = np.unique(np.hstack((
        np.logspace(-2, -0.5, 8),
        np.array([0.1, 0.5, 1, 2]),
        np.logspace(0.7, 2, 12)
    )))
    Na_over_L_arr = np.sort(Na_over_L_arr)
    kappa_vals, first_mode_frac_abs, first_mode_frac_sq, rms_to_mode1 = [], [], [], []
    norm_residuals, mean_vals, var_vals, skew_vals = [], [], [], []
    print("\nComputing Case 3 Modal Analysis:")
    print("-" * 60)
    for i, ratio in enumerate(Na_over_L_arr):
        N = ratio * (L / a)
        kappa = (N * a**2) / (L**2)
        kappa_vals.append(kappa)
        try:
            P_full = analytic_P(x_grid, N, L, a, x0)
            P_u_full = L * P_full
            n_modes, c_n = modal_coeffs(N, L, a, x0)
            P_mode1 = (c_n[0] * np.sin(np.pi * x_grid / L)) if len(c_n) > 0 else np.zeros_like(x_grid)
            P_u_mode1 = L * P_mode1
            frac_abs = np.abs(c_n[0]) / np.sum(np.abs(c_n)) if len(c_n) > 0 else 0.0
            frac_sq = (c_n[0]**2) / np.sum(c_n**2) if len(c_n) > 0 else 0.0
            first_mode_frac_abs.append(frac_abs)
            first_mode_frac_sq.append(frac_sq)
            rms = np.sqrt(np.trapezoid((P_u_full - P_u_mode1)**2, u_grid))
            rms_to_mode1.append(rms)
            mean_u, var_u, skew_u = compute_moments(u_grid, P_u_full)
            mean_vals.append(mean_u)
            var_vals.append(var_u)
            skew_vals.append(skew_u)
            norm_residuals.append(0.0)
            if (i + 1) % 5 == 0 or i == len(Na_over_L_arr) - 1:
                print(f"  [{i+1:2d}/{len(Na_over_L_arr):2d}] Na/L={ratio:.3f} -> kappa={kappa:.4e}, "
                      f"E1={frac_abs:.4f}, RMS={rms:.4e}")
        except Exception as e:
            print(f"  [ERROR] Na/L={ratio:.3f} -> {str(e)}")
            kappa_vals.pop()
            continue
    print("-" * 60)
    print(f"Computed {len(kappa_vals)} data points")
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


def generate_figure10_with_bio_markers(df, output_dir=OUTPUT_DIR):
    print("\nGenerating Figure 10 with Biological System Markers...")

    kappa_vals = df['kappa'].values
    E1_vals = df['FirstModeFracAbs'].values
    RMS_vals = df['RMS_to_mode1'].values

    kappa_at_E1_03 = kappa_vals[np.argmin(np.abs(E1_vals - 0.3))]
    kappa_at_E1_08 = kappa_vals[np.argmin(np.abs(E1_vals - 0.8))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12),
                                    gridspec_kw={'height_ratios': [2, 1]})

    # ----- TOP PANEL -----
    ax1.plot(kappa_vals, E1_vals, 'o-', color=COLORS['ground_state'], linewidth=2.5,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='First-Mode Energy Fraction E₁ (from data)', zorder=5)

    ax1.axvspan(kappa_vals.min(), kappa_at_E1_03, alpha=0.15, color=COLORS['multimode'],
               label='Multimode Regime')
    ax1.axvspan(kappa_at_E1_03, kappa_at_E1_08, alpha=0.15, color=COLORS['crossover'],
               label='Spectral Crossover')
    ax1.axvspan(kappa_at_E1_08, kappa_vals.max(), alpha=0.15, color=COLORS['ground_state'],
               label='Ground-State Dominated')

    ax1.axvline(x=kappa_at_E1_03, color=COLORS['boundary_multimode'], linewidth=2,
               linestyle='--', alpha=0.7)
    ax1.axvline(x=kappa_at_E1_08, color=COLORS['boundary_ground_state'], linewidth=2,
               linestyle='--', alpha=0.7)
    ax1.axhline(y=0.3, color='gray', linewidth=1.5, linestyle=':', alpha=0.6)
    ax1.axhline(y=0.8, color='gray', linewidth=1.5, linestyle=':', alpha=0.6)

    # =========================================================
    # BIOLOGICAL SYSTEM MARKERS — carefully positioned
    # Each marker placed at its actual kappa on the E1 curve,
    # with a label box offset to avoid overlap.
    # =========================================================

    # Interpolate E1 at each biological kappa for accurate y placement
    def interp_E1(kappa_bio):
        return float(np.interp(np.log10(kappa_bio),
                               np.log10(kappa_vals), E1_vals))

    # Define label offsets manually for clean layout:
    # (label_kappa_mult, label_y_abs, arrow_rad, ha)
    bio_label_cfg = {
        '500 bp DNA\n(1 μm nucleus)':      (8.0,  0.20,  0.30, 'left'),
        '5 kb chromatin\n(10 μm nucleus)': (5.0,  0.72, -0.25, 'left'),
        'V. cholerae ori1\n(3 μm cell)':   (6.0,  0.38,  0.20, 'left'),
        'E. coli chr.\n(1 μm nucleoid)':   (0.12, 0.82, -0.20, 'right'),
    }

    for system in BIOLOGICAL_SYSTEMS['systems']:
        kappa_bio = system['kappa']
        name = system['name']
        e1_bio = interp_E1(kappa_bio)
        cfg = bio_label_cfg[name]
        lbl_kappa = kappa_bio * cfg[0]
        lbl_y = cfg[1]
        rad = cfg[2]
        ha = cfg[3]

        # Star at actual kappa, at actual E1 curve value
        ax1.plot(kappa_bio, e1_bio, marker='*',
                markersize=22, color='black',
                markeredgecolor='white', markeredgewidth=1.0,
                zorder=15)

        # Label box with arrow
        ax1.annotate(
            name,
            xy=(kappa_bio, e1_bio),
            xytext=(lbl_kappa, lbl_y),
            fontsize=14,fontweight='bold', ha=ha, va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
                     edgecolor=system['color'], linewidth=1.5, alpha=0.95),
            arrowprops=dict(arrowstyle='->', lw=1.5,
                           color=system['color'],
                           connectionstyle=f'arc3,rad={rad}'),
            zorder=20
        )

    # Legend
    bio_legend_handle = Line2D([0], [0], marker='*', color='w',
                               markerfacecolor='black', markersize=14,
                               markeredgecolor='white',
                               label='Biological Systems (Table 1)')
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(bio_legend_handle)
    labels.append('Biological Systems (Table 1)')

    ax1.set_xscale('log')
    ax1.set_xlabel('Confinement Strength κ = Na²/L² (log scale)',
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('First-Mode Energy Fraction E₁', fontsize=16, fontweight='bold')
    ax1.set_title('Figure 10: Modal Crossover and Collapse Accuracy vs. Confinement Strength\n'
                 '(Ideal-Chain Spectral Classification with Biological System Mapping)',
                 fontsize=15, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    # Transparent legend so stars behind it remain visible
    ax1.legend(handles=handles, labels=labels, loc='upper left', fontsize=12, framealpha=0.0)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(labelsize=12)

    # ----- BOTTOM PANEL -----
    ax2.plot(kappa_vals, RMS_vals, 's-', color='darkblue', linewidth=2,
            markersize=7, markerfacecolor='lightblue', markeredgewidth=1.5,
            label='RMS Collapse Error')

    ax2.axvspan(kappa_vals.min(), kappa_at_E1_03, alpha=0.15, color=COLORS['multimode'])
    ax2.axvspan(kappa_at_E1_03, kappa_at_E1_08, alpha=0.15, color=COLORS['crossover'])
    ax2.axvspan(kappa_at_E1_08, kappa_vals.max(), alpha=0.15, color=COLORS['ground_state'])
    ax2.axvline(x=kappa_at_E1_03, color=COLORS['boundary_multimode'], linewidth=2,
               linestyle='--', alpha=0.7)
    ax2.axvline(x=kappa_at_E1_08, color=COLORS['boundary_ground_state'], linewidth=2,
               linestyle='--', alpha=0.7)

    for system in BIOLOGICAL_SYSTEMS['systems']:
        ax2.axvline(x=system['kappa'], color=COLORS['bio_marker'],
                   linewidth=0.8, linestyle=':', alpha=0.4)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Confinement Strength κ = Na²/L² (log scale)',
                  fontsize=16, fontweight='bold')
    ax2.set_ylabel('RMS Error (log scale)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper left', fontsize=14, framealpha=0.95)
    ax2.tick_params(labelsize=12)

    # ----- Regime text box -----
    regime_text = (
        "REGIME IDENTIFICATION CRITERIA\n"
        "(From Case 3 Data Analysis — Ideal-Chain Framework)\n\n"
        f"MULTIMODE REGIME (κ ≲ {kappa_at_E1_03:.2e})\n"
        f"  ✓ E₁ < 0.3 (broad modal participation)\n"
        f"  ✓ RMS ~ {RMS_vals[0]:.4f}\n"
        "  ✓ Physics: Weak confinement\n"
        "  ✓ Model: Full modal expansion\n\n"
        f"SPECTRAL CROSSOVER ({kappa_at_E1_03:.2e} ≲ κ ≲ {kappa_at_E1_08:.2e})\n"
        f"  ✓ 0.3 < E₁ < 0.8 (mode competition)\n"
        f"  ✓ RMS ~ {np.median(RMS_vals[5:15]):.4f}\n"
        "  ✓ Physics: Progressive mode suppression\n"
        "  ✓ Model: Reduced multimode expansion\n\n"
        f"GROUND-STATE DOMINATED (κ ≳ {kappa_at_E1_08:.2e})\n"
        f"  ✓ E₁ > 0.8 (lowest mode dominates)\n"
        f"  ✓ RMS ~ {RMS_vals[-1]:.4e} (asymptotic)\n"
        "  ✓ Physics: Strong confinement\n"
        "  ✓ Model: Single sine-mode (n=1)"
    )

    ax1.text(0.97, 0.03, regime_text, transform=ax1.transAxes,
            fontsize=14, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                     edgecolor='brown', linewidth=1.5),
            zorder=25)

    plt.tight_layout()
    save_figure_multi_format(fig, 'figure10_with_biological_markers',
                            output_dir=output_dir, formats=SAVE_FORMATS, dpi=DPI)
    plt.close(fig)
    print("  Figure 10 with biological markers complete")


def main():
    a, L = 0.1, 2.0
    x0 = L / 2
    df = compute_case3_data(a=a, L=L, x0=x0)
    generate_figure10_with_bio_markers(df, output_dir=OUTPUT_DIR)

if __name__ == "__main__":
    main()