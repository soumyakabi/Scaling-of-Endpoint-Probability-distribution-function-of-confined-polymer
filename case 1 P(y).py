# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:58:59 2025

@author: SOUMYA
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

# -------------------------
# Helper: estimate how many image terms are needed
# -------------------------
def estimate_M_from_tol(kappa, L, eps=1e-12, M_min=3, M_max=500):
    if kappa <= 0:
        raise ValueError("kappa must be positive for estimate_M_from_tol")
    sigma = np.sqrt(kappa) * L
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    ratio = max(eps / pref, 1e-300)
    rhs = -np.log(ratio)
    if rhs <= 0:
        return M_min
    m_needed = int(np.ceil(np.sqrt(2.0 * sigma**2 * rhs) / (2.0 * L)))
    return max(M_min, min(m_needed, M_max))


# -------------------------
# Core: absorbing image-sum P(y)
# -------------------------
def P_y_image_absorbing(y, kappa, L=1.0, M=None, eps=1e-12, M_max=500):
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    sigma = np.sqrt(kappa) * L
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    if M is None:
        M_used = estimate_M_from_tol(kappa, L, eps=eps, M_max=M_max)
    else:
        M_used = int(M)

    s = np.zeros_like(y, dtype=float)
    for m in range(-M_used, M_used + 1):
        s += (-1)**m * np.exp(-(y - 2.0*m*L)**2 / (2.0 * sigma**2))

    P = pref * s
    outside = np.abs(y) > L
    if np.any(outside):
        P[outside] = 0.0
    P[P < 0.0] = 0.0

    S = np.trapz(P, y)
    return P, S, M_used


# -------------------------
# Case 1 driver (fixed kappa, varying L)
# -------------------------
def case1_fixed_kappa(
    kappa=0.1,
    Lvals=(1.0, 2.0, 4.0),
    M=None,
    eps=1e-12,
    conditional=False,
    ny_scaled=2000,
    ny_unscaled=2000,
    L_ref=None,
    diagnostics_csv="case1_diagnostics.csv"
):
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    Lvals = list(Lvals)
    if L_ref is None:
        L_ref = Lvals[0]

    u_grid = np.linspace(-1.0, 1.0, ny_scaled)

    results_scaled = {}
    results_unscaled = {}
    diag_rows = []

    for L in Lvals:
        y_scaled_grid = u_grid * L
        P_uncond, S, M_used = P_y_image_absorbing(
            y_scaled_grid, kappa, L=L, M=M, eps=eps
        )
        note = "unconditional"
        if conditional:
            if S <= 0:
                raise RuntimeError("Survival S<=0; cannot form conditional PDF.")
            P_store = P_uncond / S
            note = "conditional"
        else:
            P_store = P_uncond

        P_scaled = L * P_store
        results_scaled[L] = (u_grid.copy(), P_scaled.copy())

        y_native = np.linspace(-L, L, ny_unscaled)
        P_native_uncond, S_native, M_used_native = P_y_image_absorbing(
            y_native, kappa, L=L, M=M, eps=eps
        )
        if conditional:
            P_native = P_native_uncond / S_native
        else:
            P_native = P_native_uncond
        results_unscaled[L] = (y_native.copy(), P_native.copy())

        diag_rows.append({
            "L": L,
            "M_used": M_used,
            "S": S,
            "note": note,
            "max_abs_residual": np.nan,
            "rms_residual": np.nan
        })

        print(f"[diagnostic] L={L:.6g}, M_used={M_used}, survival S={S:.6e}, mode={note}")

    if L_ref not in results_scaled:
        raise ValueError("L_ref must be one of the Lvals")

    u_ref, P_ref = results_scaled[L_ref]

    for row in diag_rows:
        L = row["L"]
        u_curr, P_curr = results_scaled[L]
        residual = P_curr - P_ref
        max_abs = float(np.max(np.abs(residual)))
        rms = float(np.sqrt(np.mean(residual**2)))
        row["max_abs_residual"] = max_abs
        row["rms_residual"] = rms
        print(f"[residual] L={L} vs ref={L_ref}: max={max_abs:.3e}, rms={rms:.3e}")

    df_diag = pd.DataFrame(
        diag_rows,
        columns=["L","M_used","S","max_abs_residual","rms_residual","note"]
    )
    df_diag.sort_values("L", inplace=True)
    df_diag.to_csv(diagnostics_csv, index=False)

    print(f"[output] diagnostics written to: {os.path.abspath(diagnostics_csv)}")
    return results_scaled, results_unscaled, df_diag


# -------------------------
# Negative control
# -------------------------
def case1_negative_control_sigma_fixed(sigma=1.0, Lvals=(1.0,2.0,4.0), **kwargs):
    results = {}
    for L in Lvals:
        kappa = (sigma**2)/L**2
        y = np.linspace(-L, L, kwargs.get("ny_unscaled", 2000))
        P_uncond, S, M_used = P_y_image_absorbing(
            y, kappa, L=L, M=kwargs.get("M",None), eps=kwargs.get("eps",1e-12)
        )
        results[L] = (y, P_uncond, S, M_used)
        print(f"[negctrl] L={L}, kappa={kappa:.3e}, M_used={M_used}, S={S:.6e}")
    return results


# -------------------------
# Plotting (modified)
# -------------------------
if __name__ == "__main__":

    kappa = 0.1
    Lvals = [1.0, 2.0, 4.0]

    results_scaled, results_unscaled, df_diag = case1_fixed_kappa(
        kappa=kappa,
        Lvals=Lvals,
        M=None,
        eps=1e-12,
        conditional=False,
        ny_scaled=2000,
        ny_unscaled=2000,
        L_ref=1.0,
        diagnostics_csv="case1_diagnostics.csv"
    )

    # ---------------------------
    # FIGURE 1: scaled + residuals
    # ---------------------------
    plt.figure(figsize=(7, 6))
    ax_top = plt.subplot2grid((3,1),(0,0), rowspan=2)
    ax_bot = plt.subplot2grid((3,1),(2,0), rowspan=1, sharex=ax_top)

    for L, (u, Pscaled) in results_scaled.items():
        ax_top.plot(u, Pscaled, label=f"L={L}")
    ax_top.set_ylabel(r"$L\,P(y)$", fontsize=18, fontweight='bold')
    ax_top.set_title(f"Case 1 (scaled): fixed kappa = {kappa}")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    L_ref = Lvals[0]
    _, Pref = results_scaled[L_ref]
    for L, (u, Pscaled) in results_scaled.items():
        res = Pscaled - Pref
        ax_bot.plot(u, res, label=f"res(L={L}-Lref={L_ref})")

    ax_bot.set_xlabel(r"$y/L$", fontsize=18, fontweight='bold')
    ax_bot.set_ylabel("Residual", fontsize=18, fontweight='bold')
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig("case1_scaled_residuals.png", dpi=600, bbox_inches="tight")
    plt.savefig("case1_scaled_residuals.pdf", dpi=600, bbox_inches="tight")
    plt.close()

    # ---------------------------
    # FIGURE 2: unscaled
    # ---------------------------
    plt.figure(figsize=(6, 4))
    for L, (y, P) in results_unscaled.items():
        plt.plot(y, P, label=f"L={L}")

    plt.xlabel(r"$y$", fontsize=18, fontweight='bold')
    plt.ylabel(r"$P(y)$", fontsize=18, fontweight='bold')
    plt.title(f"Case 1 (unscaled): fixed kappa = {kappa}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("case1_unscaled.png", dpi=600, bbox_inches="tight")
    plt.savefig("case1_unscaled.pdf", dpi=600, bbox_inches="tight")
    plt.close()

    print("\nDiagnostics summary:")
    print(df_diag.to_string(index=False))
