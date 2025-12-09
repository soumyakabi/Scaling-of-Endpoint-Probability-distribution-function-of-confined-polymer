# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:40:58 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analytic_P(x, N, L, a, x0, n_max=5000, decay_tol=1e-12):
    """Compute endpoint distribution P(x) using sine-mode expansion."""
    x = np.asarray(x, dtype=float)
    kappa = (N * a**2) / (L**2)

    n = np.arange(1, n_max + 1)
    lam = n * np.pi / L
    sin0 = np.sin(lam * x0)

    decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)
    idx_keep = np.where(decay > decay_tol)[0]
    if idx_keep.size == 0:
        # fallback: return small nonzero array to avoid division by zero later
        return np.full_like(x, 1.0 / x.size)
    last_idx = idx_keep[-1]

    n = n[: last_idx + 1]
    lam = lam[: last_idx + 1]
    sin0 = sin0[: last_idx + 1]
    decay = decay[: last_idx + 1]

    # numerator: modal sum
    num = np.sum(sin0[:, None] * np.sin(np.outer(lam, x)) * decay[:, None], axis=0)

    # analytic denominator (accounts for odd-only factor)
    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)

    if np.abs(den) < 1e-16:
        # numeric fallback normalization
        P_raw = num / L
        area = np.trapz(P_raw, x)
        P = P_raw / area if (area > 0 and np.isfinite(area)) else np.full_like(x, 1.0 / x.size)
    else:
        P = num / (L * den)

    # clip tiny negative numerical noise
    P = np.where(P < 0, 0.0, P)
    return P

def compute_moments(u, P_u):
    """Compute mean and skewness for distribution in scaled variable u."""
    # Ensure P_u integrates to 1 (numerical safety)
    Z = np.trapz(P_u, u)
    if Z <= 0 or not np.isfinite(Z):
        return np.nan, np.nan
    P_norm = P_u / Z
    mu = np.trapz(u * P_norm, u)
    var = np.trapz((u - mu)**2 * P_norm, u)
    sigma = np.sqrt(var) if var > 0 else 0.0
    if sigma > 0:
        skew = np.trapz(((u - mu)**3) * P_norm, u) / (sigma**3)
    else:
        skew = np.nan
    return mu, skew

if __name__ == "__main__":
    # === User parameters ===
    a = 0.1                    # Kuhn length (um)
    L_main = 2.0               # reference L for left panel (um)
    kappa_ref = 0.025          # enforced kappa for all runs (right panel)

    tether_ratios = np.linspace(0.1, 0.9, 9)   # left panel: 9 curves
    overlay_Ls = [1.0, 2.0, 3.0]               # right panel: 3 curves
    overlay_ratio = 0.1                        # x0/L for right panel

    u_grid = np.linspace(0.01, 0.99, 500)      # common scaled grid (u = x/L)

    # Prepare plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_left, ax_right = axes

    # === LEFT PANEL: tether-position sweep at L_main ===
    left_records = []
    N_left = kappa_ref * (L_main**2) / (a**2)   # N that enforces kappa_ref at L_main
    x_left = u_grid * L_main

    # plot 9 coloured solid curves
    cmap_left = plt.cm.viridis(np.linspace(0, 1, len(tether_ratios)))
    for i, ratio in enumerate(tether_ratios):
        x0 = ratio * L_main
        P_x = analytic_P(x_left, N_left, L_main, a, x0)
        # integral over x (should be ~1)
        integral = np.trapz(P_x, x_left)
        P_u = L_main * P_x               # scaled density L * P(x) evaluated vs u
        mu, skew = compute_moments(u_grid, P_u)
        left_records.append({
            "Panel": "Left",
            "x0/L": ratio,
            "Integral_Px": integral,
            "<u>": mu,
            "Skewness": skew
        })
        ax_left.plot(u_grid, P_u, color=cmap_left[i], linestyle='solid', linewidth=2,
                     label=f"x0/L={ratio:.1f}")

    ax_left.set_xlabel("x / L", fontsize=18, fontweight='bold')
    ax_left.set_ylabel("L · P(x)", fontsize=18, fontweight='bold')
    ax_left.set_title(f"Case 2 — Tether position (L={L_main:.1f} μm, N≈{N_left:.3g}, a={a:.2f} μm)", fontsize=14)
    leg_left = ax_left.legend(title="Tether positions", fontsize=9, title_fontsize=11)
    # improve ticks
    ax_left.tick_params(axis='both', labelsize=11)

    # === RIGHT PANEL: overlay across L with kappa enforced ===
    right_records = []
    P_u_list = []
    for L_val in overlay_Ls:
        N_eff = kappa_ref * (L_val**2) / (a**2)      # enforce fixed kappa
        x_vals = u_grid * L_val
        x0_val = overlay_ratio * L_val
        P_x = analytic_P(x_vals, N_eff, L_val, a, x0_val)
        integral = np.trapz(P_x, x_vals)            # should be ~1
        P_u = L_val * P_x                           # scaled density on u grid
        P_u_list.append(P_u)
        mu, skew = compute_moments(u_grid, P_u)
        right_records.append({
            "Panel": "Right",
            "L": L_val,
            "N_eff": N_eff,
            "Integral_Px": integral,
            "<u>": mu,
            "Skewness": skew,
            "max_residual": np.nan,   # placeholder, will fill after ref known
            "RMS_residual": np.nan
        })

    # Now compute residuals vs reference (choose L=3.0)
    try:
        ref_index = overlay_Ls.index(3.0)
    except ValueError:
        # fall back to last element if 3.0 not present
        ref_index = len(overlay_Ls) - 1
    ref_Pu = P_u_list[ref_index]

    # compute residuals & plot with requested linestyles/colors
    linestyles = ["solid", "dashed", "dotted"]
    colors = plt.cm.plasma(np.linspace(0, 1, len(overlay_Ls)))
    for j, L_val in enumerate(overlay_Ls):
        P_u = P_u_list[j]
        resid = P_u - ref_Pu
        max_res = np.max(np.abs(resid))
        rms_res = np.sqrt(np.mean(resid**2))
        right_records[j]["max_residual"] = float(max_res)
        right_records[j]["RMS_residual"] = float(rms_res)
        # plot with requested style
        ax_right.plot(u_grid, P_u, color=colors[j], linestyle=linestyles[j], linewidth=2,
                      label=f"L={L_val:.0f} μm")

    ax_right.set_xlabel("x / L", fontsize=18, fontweight='bold')
    ax_right.set_ylabel("L · P(x)", fontsize=18, fontweight='bold')
    ax_right.set_title(f"Overlay x0/L={overlay_ratio:.2f} across L (κ={kappa_ref:.3f})", fontsize=14)
    leg_right = ax_right.legend(title="Cell length", fontsize=9, title_fontsize=11)
    ax_right.tick_params(axis='both', labelsize=11)

    plt.tight_layout()

    # --- Save high-resolution figures (PNG + PDF, 600 dpi) ---
    fig.savefig("case2_tether_overlay_600dpi.png",
                dpi=600, bbox_inches="tight")
    fig.savefig("case2_tether_overlay_600dpi.pdf",
                dpi=600, bbox_inches="tight")
    # ----------------------------------------------------------

    plt.show()

    # === Combine stats and save to CSV ===
    # Normalize dictionaries to have same set of columns for DataFrame
    # Left entries won't have L, N_eff, max_residual, RMS_residual keys — fill with NaN
    combined = []
    for r in left_records:
        rec = {
            "Panel": r["Panel"],
            "x0/L": r["x0/L"],
            "L": np.nan,
            "N_eff": np.nan,
            "Integral_Px": r["Integral_Px"],
            "<u>": r["<u>"],
            "Skewness": r["Skewness"],
            "max_residual": np.nan,
            "RMS_residual": np.nan
        }
        combined.append(rec)
    for r in right_records:
        rec = {
            "Panel": r["Panel"],
            "x0/L": np.nan,
            "L": r["L"],
            "N_eff": r["N_eff"],
            "Integral_Px": r["Integral_Px"],
            "<u>": r["<u>"],
            "Skewness": r["Skewness"],
            "max_residual": r["max_residual"],
            "RMS_residual": r["RMS_residual"]
        }
        combined.append(rec)

    df = pd.DataFrame(combined)
    out_fn = "case2_statistics.csv"
    df.to_csv(out_fn, index=False)
    print(f"\nStatistical results saved to '{out_fn}'")
    print(df)
    # === Combine stats and save to CSV ===
    # Normalize dictionaries to have same set of columns for DataFrame
    # Left entries won't have L, N_eff, max_residual, RMS_residual keys — fill with NaN
    combined = []
    for r in left_records:
        rec = {
            "Panel": r["Panel"],
            "x0/L": r["x0/L"],
            "L": np.nan,
            "N_eff": np.nan,
            "Integral_Px": r["Integral_Px"],
            "<u>": r["<u>"],
            "Skewness": r["Skewness"],
            "max_residual": np.nan,
            "RMS_residual": np.nan
        }
        combined.append(rec)
    for r in right_records:
        rec = {
            "Panel": r["Panel"],
            "x0/L": np.nan,
            "L": r["L"],
            "N_eff": r["N_eff"],
            "Integral_Px": r["Integral_Px"],
            "<u>": r["<u>"],
            "Skewness": r["Skewness"],
            "max_residual": r["max_residual"],
            "RMS_residual": r["RMS_residual"]
        }
        combined.append(rec)

    df = pd.DataFrame(combined)
    out_fn = "case2_statistics.csv"
    df.to_csv(out_fn, index=False)
    print(f"\nStatistical results saved to '{out_fn}'")
    print(df) 