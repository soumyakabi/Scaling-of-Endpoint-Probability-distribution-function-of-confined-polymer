# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:42:17 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

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

if __name__ == "__main__":
    # Parameters
    a = 0.1
    L = 2.0
    x0 = 0.5 * L
    u_grid = np.linspace(0.01, 0.99, 600)
    x_grid = u_grid * L

    # Na/L sampling
    Na_over_L_arr = np.unique(np.hstack((
        np.logspace(-2, -0.5, 8),
        np.array([0.1, 0.5, 1, 2]),
        np.logspace(0.7, 2, 12)
    )))
    Na_over_L_arr = np.sort(Na_over_L_arr)

    # Storage
    kappa_vals, first_mode_frac_abs, first_mode_frac_sq, rms_to_mode1 = [], [], [], []
    norm_residuals, mean_vals, var_vals, skew_vals = [], [], [], []
    rep_ratios = [0.1, 1.0, 5.0, 10.0, 50.0]
    rep_curves = {}

    for ratio in Na_over_L_arr:
        N = ratio * (L / a)
        kappa = (N * a**2) / (L**2)
        kappa_vals.append(kappa)
        
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

        if any(np.isclose(ratio, r, rtol=1e-2) for r in rep_ratios):
            rep_curves[ratio] = (P_u, P_mode1, N, ratio)

    # Save kappa diagnostics to a separate CSV file
    with open("kappa_diagnostics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kappa", "Na/L", "FirstModeFracAbs", "FirstModeFracSq", "RMS_to_mode1"])
        for i, (kappa, ratio) in enumerate(zip(kappa_vals, Na_over_L_arr)):
            writer.writerow([kappa, ratio, first_mode_frac_abs[i], 
                           first_mode_frac_sq[i], rms_to_mode1[i]])

    # Save original diagnostics to CSV
    with open("case3_diagnostics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Na/L", "kappa", "NormResidual", "FirstModeFracAbs", "FirstModeFracSq",
                         "RMS_to_mode1", "Mean_u", "Variance_u", "Skewness_u"])
        for i, ratio in enumerate(Na_over_L_arr):
            writer.writerow([ratio, kappa_vals[i], norm_residuals[i], first_mode_frac_abs[i],
                             first_mode_frac_sq[i], rms_to_mode1[i],
                             mean_vals[i], var_vals[i], skew_vals[i]])

    # Create a new figure for kappa vs first mode fraction and RMS
    fig_kappa, ax_kappa = plt.subplots(figsize=(10, 6))
    
    ax1 = ax_kappa
    l1, = ax1.plot(kappa_vals, first_mode_frac_abs, label='|c1| / Σ|c_n|', marker='o', linewidth=2)
    l2, = ax1.plot(kappa_vals, first_mode_frac_sq, label='c1^2 / Σ c_n^2', marker='s', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('κ = Na²/L² (log scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('First-mode fraction', fontsize=14, fontweight='bold')
    ax1.set_title('Modal dominance & RMS to single-mode vs κ', fontsize=15, fontweight='bold')
    ax1.grid(alpha=0.3, which='both')
    
    ax2 = ax1.twinx()
    l3, = ax2.plot(kappa_vals, rms_to_mode1, label='RMS(P - mode1)', color='C3', marker='^', linewidth=2)
    ax2.set_ylabel('RMS difference (abs)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    
    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='lower right', fontsize=11, framealpha=0.9)
    
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()

    # --- Save high-resolution (600 dpi) PNG and PDF for kappa diagnostics ---
    fig_kappa.savefig('kappa_diagnostics_600dpi.png', dpi=600, bbox_inches='tight')
    fig_kappa.savefig('kappa_diagnostics_600dpi.pdf', dpi=600, bbox_inches='tight')
    # ------------------------------------------------------------------------

    plt.show()

    # Generalized collapse residual check for all high Na/L >= 5
    high_ratios = [r for r in rep_curves.keys() if r >= 5]
    if high_ratios:
        ref_ratio = max(high_ratios)  # use largest ratio as reference (≈50)
        P_ref, _, _, _ = rep_curves[ref_ratio]
        print("\nResidual collapse diagnostics (vs Na/L≈{:.1f}):".format(ref_ratio))
        print("Ratio   RMS residual")
        for r in sorted(high_ratios):
            P_u, _, _, _ = rep_curves[r]
            rms_resid = np.sqrt(np.mean((P_u - P_ref)**2))
            print(f"{r:5.1f}   {rms_resid:.3e}")

    # Original plotting: representative PDFs and diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios':[2,1]})
    ax_pdf, ax_diag = axes

    cmap = plt.get_cmap('tab10')
    for i, ratio in enumerate(sorted(rep_curves.keys())):
        P_u, P_mode1, N_used, r = rep_curves[ratio]
        ax_pdf.plot(u_grid, P_u, label=f'Na/L={ratio:.1g}', linewidth=2, color=cmap(i))
        ax_pdf.plot(u_grid, P_mode1, linestyle='--', linewidth=1, color=cmap(i), alpha=0.8)

    ax_pdf.plot([], [], linestyle='--', color='k', linewidth=1, label='single-mode (dashed)')
    ax_pdf.set_xlabel('x / L', fontsize=14, fontweight='bold')
    ax_pdf.set_ylabel('L · P(x) (scaled density)', fontsize=14, fontweight='bold')
    ax_pdf.set_title('Case 3 — Confinement-strength scaling (representative PDFs)', fontsize=15, fontweight='bold')
    ax_pdf.grid(alpha=0.3)
    ax_pdf.legend(fontsize=10, title='Representative Na/L', loc='upper right')

    ax1 = ax_diag
    l1, = ax1.plot(Na_over_L_arr, first_mode_frac_abs, label='|c1| / Σ|c_n|', marker='o', linewidth=1)
    l2, = ax1.plot(Na_over_L_arr, first_mode_frac_sq, label='c1^2 / Σ c_n^2', marker='s', linewidth=1)
    ax1.set_xscale('log')
    ax1.set_xlabel('Na / L (log scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('First-mode fraction', fontsize=14, fontweight='bold')
    ax1.set_title('Modal dominance & RMS to single-mode', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, which='both')

    ax1b = ax1.twinx()
    l3, = ax1b.plot(Na_over_L_arr, rms_to_mode1, label='RMS(P - mode1)', color='C3', marker='^', linewidth=1)
    ax1b.set_ylabel('RMS difference (abs)', fontsize=14, fontweight='bold')
    ax1b.set_yscale('log')

    lines = [l1, l2, l3]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc='lower right', bbox_to_anchor=(0.98, 0.25), fontsize=9, framealpha=0.9)

    ax_pdf.tick_params(axis='both', which='major', labelsize=11)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1b.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()

    # --- Save high-resolution (600 dpi) PNG and PDF for case 3 results ---
    fig.savefig('case3_results_600dpi.png', dpi=600, bbox_inches='tight')
    fig.savefig('case3_results_600dpi.pdf', dpi=600, bbox_inches='tight')
    # ---------------------------------------------------------------------

    plt.show()