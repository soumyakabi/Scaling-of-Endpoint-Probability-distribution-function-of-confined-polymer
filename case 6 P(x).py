# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:55:53 2025

@author: SOUMYA
"""

import math
import csv
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# -------------------------------
# Core modal and PDF routines
# -------------------------------
def P_x_fourier(x: np.ndarray, x0: float, N: float, a: float, L: float,
               max_modes: int = 20000, tol: float = 1e-15) -> Tuple[np.ndarray, int]:
    """Eigenfunction expansion for P(x)."""
    x = np.asarray(x, dtype=np.float64)
    kappa = (N * a**2) / (L**2)
    P = np.zeros_like(x, dtype=np.float64)
    modes_used = 0
    for n in range(1, max_modes + 1):
        lam = n * np.pi / L
        sin_x = np.sin(lam * x)
        sin_x0 = math.sin(lam * x0)
        decay = math.exp(- (n**2) * (np.pi**2) * kappa / 8.0)
        term = (2.0 / L) * sin_x * sin_x0 * decay
        P += term
        modes_used = n
        if np.max(np.abs(term)) < tol:
            break
    P[P < 0] = 0.0
    integral = np.trapz(P, x)
    if not np.isfinite(integral) or integral <= 0:
        raise RuntimeError("Bad normalization in P_x_fourier.")
    P /= integral
    return P, modes_used


def modal_coeffs_analytic(N: float, a: float, L: float, x0: float,
                          n_max: int = 20000, decay_tol: float = 1e-16) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic modal coefficients c_n."""
    n_all = np.arange(1, n_max + 1)
    lam = n_all * np.pi / L
    sin0 = np.sin(lam * x0)
    kappa = (N * a**2) / (L**2)
    decay = np.exp(- (n_all**2) * (np.pi**2) * kappa / 8.0)
    keep_idx = np.where(decay > decay_tol)[0]
    if keep_idx.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    last = keep_idx[-1]
    n = n_all[: last + 1]
    sin0 = sin0[: last + 1]
    decay = decay[: last + 1]
    cos_term = 1 - (-1)**n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)
    if np.abs(den) < 1e-16:
        return n, np.zeros_like(n, dtype=float)
    c_n = (sin0 * decay) / (L * den)
    return n, c_n


# -------------------------------
# Utilities
# -------------------------------
def trapz_integral(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def save_diagnostics_csv(filename: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filename, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# -------------------------------
# Build cases for a fixed kappa (varying L)
# -------------------------------
def build_fixed_kappa_cases(kappa: float,
                            a: float = 1.0,
                            L_values: Optional[List[float]] = None,
                            x0_fraction: float = 0.5,
                            y_points: int = 1201,
                            max_modes: int = 5000,
                            tol: float = 1e-14,
                            eps: float = 1e-8) -> List[Dict[str, Any]]:

    if L_values is None:
        L_values = [2.0, 4.0, 6.0]
    cases = []

    for L in L_values:
        N = kappa * (L**2) / (a**2)
        x0 = x0_fraction * L
        x = np.linspace(eps * L, (1.0 - eps) * L, y_points)
        P, modes_used = P_x_fourier(x, x0, N, a, L, max_modes=max_modes, tol=tol)

        sigma = math.sqrt(N * a**2)
        s = (x - x0) / sigma
        Ptilde = sigma * P

        n_arr, c_n = modal_coeffs_analytic(N, a, L, x0, n_max=max_modes)
        if c_n.size > 0:
            abs_sum = np.sum(np.abs(c_n))
            sq_sum = np.sum(c_n**2)
            f_abs = float(abs(c_n[0]) / abs_sum) if abs_sum > 0 else float('nan')
            f_sq = float((c_n[0]**2) / sq_sum) if sq_sum > 0 else float('nan')
        else:
            f_abs = float('nan')
            f_sq = float('nan')

        cases.append({
            "L": float(L), "N": float(N), "kappa": float(kappa),
            "x0": float(x0), "sigma": float(sigma),
            "x": x, "P": P, "s": s, "Ptilde": Ptilde,
            "modes": int(modes_used),
            "first_mode_frac_abs": f_abs,
            "first_mode_frac_sq": f_sq,
            "label": f"L={L:.3g}"
        })

    return cases


# -------------------------------
# Build negative-test (fixed L, varying kappa)
# -------------------------------
def build_fixed_L_vary_kappa(L_fixed: float,
                             kappas: List[float],
                             a: float = 1.0,
                             x0_fraction: float = 0.5,
                             y_points: int = 1201,
                             max_modes: int = 5000,
                             tol: float = 1e-14,
                             eps: float = 1e-8) -> List[Dict[str, Any]]:

    cases = []
    for kappa in kappas:
        N = kappa * (L_fixed**2) / (a**2)
        x0 = x0_fraction * L_fixed
        x = np.linspace(eps * L_fixed, (1.0 - eps) * L_fixed, y_points)
        P, modes_used = P_x_fourier(x, x0, N, a, L_fixed, max_modes=max_modes, tol=tol)

        sigma = math.sqrt(N * a**2)
        s = (x - x0) / sigma
        Ptilde = sigma * P

        cases.append({
            "kappa": float(kappa), "L": float(L_fixed), "N": float(N),
            "x": x, "P": P, "s": s, "Ptilde": Ptilde,
            "sigma": float(sigma),
            "label": f"kappa={kappa:.3g}"
        })
    return cases


# -------------------------------
# Main plotting (3 rows × 2 columns)
# -------------------------------
def plot_three_kappas_with_negative_test(
        kappa_list=[0.1, 0.5, 2.0],
        L_values=[2.0, 4.0, 6.0],
        out_figure="tether_sigma_scaling_compare_combined",
        L_fixed_for_negtest=4.0,
        kappas_negtest=[0.1, 0.5, 2.0, 4.0],
        a=1.0,
        x0_fraction=0.5,
        y_points=1201,
        max_modes=5000,
        tol=1e-14,
        eps=1e-8):

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(14, 12),
        gridspec_kw={'height_ratios': [1, 1, 1]}
    )

    colors_L = plt.get_cmap("tab10")(np.linspace(0, 1, len(L_values)))
    all_diag_rows = []

    for i, kappa in enumerate(kappa_list):

        cases = build_fixed_kappa_cases(
            kappa=kappa, a=a, L_values=L_values,
            x0_fraction=x0_fraction, y_points=y_points,
            max_modes=max_modes, tol=tol, eps=eps
        )

        s_all = np.hstack([c["s"] for c in cases])
        s_common = np.linspace(np.min(s_all), np.max(s_all), 1601)
        ref_interp = np.interp(
            s_common, cases[0]["s"], cases[0]["Ptilde"],
            left=np.nan, right=np.nan
        )

        ax_top = axes[i, 0]
        ax_bot = axes[i, 1]

        # ---------------------------
        # Plot curves
        # ---------------------------
        for j, c in enumerate(cases):

            # Main curve
            ax_top.plot(
                c["s"], c["Ptilde"],
                color=colors_L[j], lw=1.6, label=c["label"]
            )

            # Residuals
            P_interp = np.interp(
                s_common, c["s"], c["Ptilde"],
                left=np.nan, right=np.nan
            )

            valid = np.isfinite(P_interp) & np.isfinite(ref_interp)
            if np.any(valid):
                resid = P_interp[valid] - ref_interp[valid]
                ax_bot.plot(
                    s_common[valid], resid,
                    color=colors_L[j], lw=1.2
                )
                overlap = float(np.sum(valid) / s_common.size)
                rms_resid = float(np.sqrt(np.nanmean(resid**2)))
                max_resid = float(np.nanmax(np.abs(resid)))
            else:
                overlap = float('nan')
                rms_resid = float('nan')
                max_resid = float('nan')

            all_diag_rows.append({
                "panel_kappa": float(kappa),
                "L": float(c["L"]),
                "sigma": float(c["sigma"]),
                "modes": int(c["modes"]),
                "first_mode_frac_abs": float(c["first_mode_frac_abs"]),
                "first_mode_frac_sq": float(c["first_mode_frac_sq"]),
                "overlap_fraction_vs_ref": overlap,
                "rms_resid_vs_ref": rms_resid,
                "max_abs_resid_vs_ref": max_resid
            })

        # ---------------------------
        # TITLES & LABELING
        # ---------------------------
        ax_top.set_title(
            f"Fixed κ = {kappa} (positive test across L)",
            fontsize=14, fontweight="bold"
        )

        ax_top.set_ylabel(
            r"$\sigma P(x)$",
            fontsize=18, fontweight="bold"
        )

        ax_bot.set_title(
            "Residual vs L=%.3g reference" % L_values[0],
            fontsize=12, fontweight="bold"
        )

        # ---------------------------
        # OPTION A: X-labels for BOTH bottom panels (every row)
        # ---------------------------
        axes[i, 0].set_xlabel(
            r"$s = (x-x_0)/\sigma$",
            fontsize=18, fontweight="bold"
        )
        axes[i, 1].set_xlabel(
            r"$s = (x-x_0)/\sigma$",
            fontsize=18, fontweight="bold"
        )

        # Aesthetics
        ax_top.tick_params(labelsize=12)
        ax_bot.tick_params(labelsize=12)
        ax_bot.axhline(0.0, color='k', lw=0.6)

        if i == 0:
            ax_top.legend(
                loc='upper right', fontsize=11, frameon=False
            )

        # ---------------------------
        # INSET: NEGATIVE TEST
        # ---------------------------
        axins = inset_axes(
            ax_top, width="36%", height="36%",
            loc='upper right', borderpad=1
        )

        neg_cases = build_fixed_L_vary_kappa(
            L_fixed=L_fixed_for_negtest,
            kappas=kappas_negtest,
            a=a,
            x0_fraction=x0_fraction,
            y_points=y_points,
            max_modes=max_modes,
            tol=tol,
            eps=eps
        )

        s_all2 = np.hstack([nc["s"] for nc in neg_cases])
        s_common2 = np.linspace(np.min(s_all2), np.max(s_all2), 801)

        for idx_nc, nc in enumerate(neg_cases):
            Pn = np.interp(
                s_common2, nc["s"], nc["Ptilde"],
                left=np.nan, right=np.nan
            )
            axins.plot(
                s_common2, Pn, lw=1.2,
                color=plt.cm.tab10(idx_nc), label=nc["label"]
            )

        axins.set_xlim(0.0, min(1.0, np.max(s_all2)))
        axins.set_title(
            f"Negative test: L={L_fixed_for_negtest}",
            fontsize=8
        )
        axins.tick_params(labelsize=7)
        axins.legend(fontsize=7, frameon=False)

    plt.tight_layout()

    # -------------------------------
    # Saving Figures (600 dpi)
    # -------------------------------
    plt.savefig(out_figure + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(out_figure + ".pdf", dpi=600, bbox_inches="tight")
    print(f"[INFO] Saved {out_figure}.png and .pdf at 600 dpi.")

    save_diagnostics_csv(
        "tether_sigma_scaling_compare_combined_diag.csv",
        all_diag_rows
    )


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    kappas_main = [0.1, 0.5, 2.0]
    kappas_negtest = [0.1, 0.5, 2.0, 4.0]
    L_values = [2.0, 4.0, 6.0]

    plot_three_kappas_with_negative_test(
        kappa_list=kappas_main,
        L_values=L_values,
        out_figure="tether_sigma_scaling_compare_combined",
        L_fixed_for_negtest=4.0,
        kappas_negtest=kappas_negtest
    )