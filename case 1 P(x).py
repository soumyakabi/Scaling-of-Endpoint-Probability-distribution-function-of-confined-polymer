# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:39:05 2025

@author: SOUMYA
"""

import numpy as np 
import matplotlib.pyplot as plt

def analytic_P(x, N, L, a, x0, n_max=5000, decay_tol=1e-12):
    x = np.asarray(x, dtype=float)
    kappa = (N * a**2) / (L**2)

    n = np.arange(1, n_max + 1)
    lam = n * np.pi / L
    sin0 = np.sin(lam * x0)
    decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)

    idx_keep = np.where(decay > decay_tol)[0]
    last_idx = idx_keep[-1] if idx_keep.size else 0

    n = n[: last_idx + 1]
    lam = lam[: last_idx + 1]
    sin0 = sin0[: last_idx + 1]
    decay = decay[: last_idx + 1]

    num = np.sum(sin0[:, None] * np.sin(np.outer(lam, x)) * decay[:, None], axis=0)
    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)

    if np.abs(den) < 1e-16:
        P_raw = num / L
        area = np.trapz(P_raw, x)
        P = P_raw / area
    else:
        P = num / (L * den)

    P[P < 0] = 0.0
    return P

if __name__ == "__main__":
    # Fixed parameters
    a = 0.1     # Kuhn length (um)
    kappa_star = 0.05  # choose target kappa to enforce collapse
    L_values = [1.0, 2.0, 3.0]  # different cell lengths (um)

    u_grid = np.linspace(0.01, 0.99, 300)

    # Create two-panel figure: top for distributions, bottom for residuals
    fig, (ax_top, ax_resid) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(9, 8)
    )

    # Store scaled distributions for each L to compute residuals
    Pu_dict = {}

    for L in L_values:
        x0 = 0.5 * L  # tether at center
        # Adjust N(L) to enforce fixed kappa
        N_L = kappa_star * (L**2) / (a**2)

        sigma = np.sqrt(N_L) * a
        print(f"L={L:.2f} μm -> N(L)={N_L:.4f}, sigma/L={sigma/L:.4f} (target {np.sqrt(kappa_star):.4f})")

        x = u_grid * L
        P_x = analytic_P(x, N=N_L, L=L, a=a, x0=x0)
        P_u = L * P_x
        Pu_dict[L] = P_u

        area = np.trapz(P_x, x)
        print(f"  integral P(x) = {area:.6e}")

        ax_top.plot(u_grid, P_u, label=f"L={L} μm", linewidth=2)

    # Reference distribution for residuals: take largest L (here 3.0 μm)
    L_ref = max(L_values)
    P_ref = Pu_dict[L_ref]

    # Compute and plot residuals for other L against the reference
    for L in L_values:
        if L == L_ref:
            continue
        residual = Pu_dict[L] - P_ref
        ax_resid.plot(u_grid, residual, label=f"L={L:.1f} μm vs {L_ref:.1f} μm")

    # Formatting top panel
    ax_top.set_ylabel('L · P(x) (scaled density)', fontsize=14,fontweight='bold')
    ax_top.set_title('Scaled distributions and residuals — Confined tethered polymer', fontsize=15,fontweight='bold')
    ax_top.legend(fontsize=11)
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, None)

    # Formatting residual panel
    ax_resid.set_xlabel('x / L', fontsize=14,fontweight='bold')
    ax_resid.set_ylabel('Residual', fontsize=14,fontweight='bold')
    ax_resid.axhline(0.0, linestyle='--', linewidth=1)  # dashed zero line
    ax_resid.grid(alpha=0.3)
    ax_resid.legend(fontsize=10)

    fig.tight_layout()

    # --- Save high-resolution figures (600 dpi) ---
    fig.savefig("tethered_polymer_scaled_distribution_residuals_600dpi.png",
                dpi=600, bbox_inches="tight")
    fig.savefig("tethered_polymer_scaled_distribution_residuals_600dpi.pdf",
                dpi=600, bbox_inches="tight")
    # ---------------------------------------------

    plt.show()
