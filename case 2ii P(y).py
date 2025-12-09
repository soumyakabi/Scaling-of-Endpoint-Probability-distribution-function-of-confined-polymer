# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:01:23 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ----------------------------
# Vectorized method-of-images PDF (absorbing walls at y = +/- R)
# ----------------------------
def P_image_vectorized(y, sigma, R, m_max=400):
    m = np.arange(-m_max, m_max + 1)
    m_col = m[:, None]
    arg = (y[None, :] - 2.0 * m_col * R)
    expo = np.exp(-0.5 * (arg / sigma)**2)
    signs = (-1.0)**m_col
    sum_over_m = np.sum(signs * expo, axis=0)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    P = prefactor * sum_over_m
    return np.where(np.abs(y) <= R, P, 0.0)

# ----------------------------
# sigma(kappa)
# ----------------------------
def compute_sigma_from_kappa(kappa):
    return np.sqrt(kappa)

# ----------------------------
# Diagnostics
# ----------------------------
def unit_gaussian(s):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s**2)

def diagnostics_on_sgrid(s_grid, P_coil, R_over_sigma):
    mask = np.abs(s_grid) <= (R_over_sigma + 1e-12)
    if mask.sum() == 0:
        return {'rms': np.nan, 'rms_center': np.nan,
                'ks_like': np.nan, 'linf': np.nan}

    G = unit_gaussian(s_grid)
    diff = P_coil[mask] - G[mask]
    rms = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(G[mask]**2))

    mask_center = (np.abs(s_grid) <= 1.0) & mask
    if mask_center.sum() == 0:
        rms_center = np.nan
    else:
        diff_c = P_coil[mask_center] - G[mask_center]
        rms_center = np.sqrt(np.mean(diff_c**2)) / np.sqrt(np.mean(G[mask_center]**2))

    s_mask = s_grid[mask]
    P_mask = P_coil[mask].copy()
    G_mask = G[mask].copy()
    dx = s_mask[1] - s_mask[0] if len(s_mask) > 1 else 1.0
    P_mask = np.where(P_mask < 0.0, 0.0, P_mask)
    intP = np.trapz(P_mask, s_mask)
    intG = np.trapz(G_mask, s_mask)

    if intP <= 0 or intG <= 0:
        ks_like = np.nan
    else:
        cdfP = np.cumsum(P_mask) * dx / intP
        cdfG = np.cumsum(G_mask) * dx / intG
        ks_like = np.max(np.abs(cdfP - cdfG))

    linf = np.max(np.abs(diff))
    return {'rms': float(rms), 'rms_center': float(rms_center),
            'ks_like': float(ks_like), 'linf': float(linf)}

# ----------------------------
# Sweep configuration
# ----------------------------
kappa_list = [0.30, 0.50, 0.75]
R_over_sigma_values = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0])
Ns = 801
m_max_default = 400
output_dir = "case2_Rsweep_R_is_radius"
os.makedirs(output_dir, exist_ok=True)

records = []

# ----------------------------
# Grid sweep
# ----------------------------
for kappa in kappa_list:
    sigma = compute_sigma_from_kappa(kappa)
    for Rrs in R_over_sigma_values:
        s_grid = np.linspace(-Rrs, Rrs, Ns)
        y = sigma * s_grid
        R = Rrs * sigma

        P = P_image_vectorized(y, sigma, R, m_max=m_max_default)
        S = np.trapz(P, y)

        if S <= 0:
            P_cond = np.zeros_like(P)
        else:
            P_cond = P / S
            P_cond = np.where(P_cond < 0.0, 0.0, P_cond)
            if np.trapz(P_cond, y) > 0:
                P_cond = P_cond / np.trapz(P_cond, y)

        var_cond = np.trapz((y**2) * P_cond, y) if np.any(P_cond) else np.nan
        P_coil = sigma * P_cond
        diag = diagnostics_on_sgrid(s_grid, P_coil, Rrs)

        records.append({
            'kappa': float(kappa),
            'sigma': float(sigma),
            'R_over_sigma': float(Rrs),
            'R': float(R),
            'm_max_used': int(m_max_default),
            'survival_S': float(S),
            'var_cond': float(var_cond),
            'rms': diag['rms'],
            'rms_center': diag['rms_center'],
            'ks_like': diag['ks_like'],
            'linf': diag['linf']
        })

df = pd.DataFrame(records)
csv_path = os.path.join(output_dir, "case2_Rsweep_R_as_radius_diagnostics.csv")
df.to_csv(csv_path, index=False)
print("Saved diagnostics CSV to:", csv_path)

# ----------------------------
# Heatmap
# ----------------------------
rms_mat = np.zeros((len(kappa_list), len(R_over_sigma_values)))
for i, kappa in enumerate(kappa_list):
    for j, Rrs in enumerate(R_over_sigma_values):
        row = df[(df.kappa == kappa) & (np.isclose(df.R_over_sigma, Rrs))]
        rms_mat[i, j] = float(row['rms'].values[0]) if len(row) == 1 else np.nan

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
X, Y = np.meshgrid(R_over_sigma_values, kappa_list)
pcm = ax.pcolormesh(X, Y, rms_mat, shading='auto')
cb = fig.colorbar(pcm, ax=ax)

cb.set_label("Normalized RMS to unit Gaussian", fontsize=18, fontweight="bold")
ax.set_xlabel(r"$R/\sigma$", fontsize=18, fontweight="bold")
ax.set_ylabel(r"$\kappa$", fontsize=18, fontweight="bold")
ax.set_title(r"Deviation from Gaussian (R sweep at fixed $\kappa$)",
             fontsize=14, fontweight="bold")

ax.axvline(3.0, color='white', linestyle='--', linewidth=1.2)
ax.set_xticks(R_over_sigma_values)

plt.tight_layout()
png = os.path.join(output_dir, "case2_Rsweep_heatmap_Rradius.png")
pdf = os.path.join(output_dir, "case2_Rsweep_heatmap_Rradius.pdf")
plt.savefig(png, dpi=600, bbox_inches="tight")
plt.savefig(pdf, dpi=600, bbox_inches="tight")
print("Saved:", png, "and PDF")
plt.show()

# ----------------------------
# Representative coil-scaled traces
# ----------------------------
s_plot = np.linspace(-4.0, 4.0, 1601)
rep_Rs = [0.7, 1.4, 3.0]

fig2, axs2 = plt.subplots(1, len(kappa_list),
                          figsize=(4 * len(kappa_list), 3.6),
                          squeeze=False)

for i, kappa in enumerate(kappa_list):
    sigma = compute_sigma_from_kappa(kappa)
    ax = axs2[0, i]

    for Rrs in rep_Rs:
        s_grid = np.linspace(-Rrs, Rrs, Ns)
        y = sigma * s_grid
        R = Rrs * sigma

        P = P_image_vectorized(y, sigma, R, m_max=m_max_default)
        S = np.trapz(P, y)
        P_cond = P / S if S > 0 else np.zeros_like(P)
        P_cond = np.where(P_cond < 0.0, 0.0, P_cond)

        if np.trapz(P_cond, y) > 0:
            P_cond = P_cond / np.trapz(P_cond, y)

        P_coil = sigma * P_cond
        P_interp = np.interp(s_plot, s_grid, P_coil, left=0, right=0)
        style = "-" if Rrs >= 3 else "--"
        ax.plot(s_plot, P_interp, linestyle=style, label=fr"$R/\sigma={Rrs}$")

    ax.plot(s_plot, unit_gaussian(s_plot), ":k", label="unit Gaussian")

    ax.set_xlim(-4, 4)
    ax.set_xlabel(r"$s = y/\sigma$", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"$\widehat{P}(s)=\sigma P_{\mathrm{cond}}(y)$",
                  fontsize=18, fontweight="bold")
    ax.set_title(fr"$\kappa={kappa}$", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.2)
    if i == 0:
        ax.legend(fontsize=10)

plt.tight_layout()
png2 = os.path.join(output_dir, "case2_Rsweep_coilscaled_Rradius.png")
pdf2 = os.path.join(output_dir, "case2_Rsweep_coilscaled_Rradius.pdf")
plt.savefig(png2, dpi=600, bbox_inches="tight")
plt.savefig(pdf2, dpi=600, bbox_inches="tight")
print("Saved:", png2, "and PDF")
plt.show()

# ----------------------------
# Geometry-scaled traces
# ----------------------------
fig3, axs3 = plt.subplots(1, len(kappa_list),
                          figsize=(4 * len(kappa_list), 3.6),
                          squeeze=False)

for i, kappa in enumerate(kappa_list):
    sigma = compute_sigma_from_kappa(kappa)
    ax = axs3[0, i]

    for Rrs in rep_Rs:
        s_grid = np.linspace(-Rrs, Rrs, Ns)
        y = sigma * s_grid
        R = Rrs * sigma

        P = P_image_vectorized(y, sigma, R, m_max=m_max_default)
        S = np.trapz(P, y)
        P_cond = P / S if S > 0 else np.zeros_like(P)
        P_cond = np.where(P_cond < 0.0, 0.0, P_cond)

        if np.trapz(P_cond, y) > 0:
            P_cond = P_cond / np.trapz(P_cond, y)

        u = y / R
        ax.plot(u, R * P_cond, label=fr"$R/\sigma={Rrs}$")

    ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$u = y/R$", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"$\mathcal{P}_{\mathrm{cond}}(u)=R\,P_{\mathrm{cond}}(y)$",
                  fontsize=18, fontweight="bold")
    ax.set_title(fr"$\kappa={kappa}$ (geometry-scaled)",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.2)

    if i == 0:
        ax.legend(fontsize=10)

plt.tight_layout()
png3 = os.path.join(output_dir, "case2_Rsweep_geomscaled_Rradius.png")
pdf3 = os.path.join(output_dir, "case2_Rsweep_geomscaled_Rradius.pdf")
plt.savefig(png3, dpi=600, bbox_inches="tight")
plt.savefig(pdf3, dpi=600, bbox_inches="tight")
print("Saved:", png3, "and PDF")
plt.show()

# ----------------------------
# Convergence plots
# ----------------------------
conv_points = [(0.50, 0.7), (0.50, 1.4), (0.50, 3.0)]
m_list = [50, 100, 200, 400, 800]

fig4, axs4 = plt.subplots(1, len(conv_points),
                          figsize=(4 * len(conv_points), 3.6),
                          squeeze=False)

for idx, (kappa, Rrs) in enumerate(conv_points):
    sigma = compute_sigma_from_kappa(kappa)
    s_grid = np.linspace(-Rrs, Rrs, Ns)
    y = sigma * s_grid

    rms_vals = []
    for m_try in m_list:
        P = P_image_vectorized(y, sigma, Rrs * sigma, m_max=m_try)
        S = np.trapz(P, y)
        P_cond = P / S if S > 0 else np.zeros_like(P)
        P_cond = np.where(P_cond < 0.0, 0.0, P_cond)

        if np.trapz(P_cond, y) > 0:
            P_cond = P_cond / np.trapz(P_cond, y)

        P_coil = sigma * P_cond
        diag = diagnostics_on_sgrid(s_grid, P_coil, Rrs)
        rms_vals.append(diag['rms'])

    ax = axs4[0, idx]
    ax.plot(m_list, rms_vals, "o-")
    ax.set_xscale("log")

    ax.set_xlabel("m_max (images)", fontsize=18, fontweight="bold")
    ax.set_ylabel("normalized RMS", fontsize=18, fontweight="bold")
    ax.set_title(fr"$\kappa={kappa},\ R/\sigma={Rrs}$",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.2)

plt.tight_layout()
png4 = os.path.join(output_dir, "case2_Rsweep_convergence_Rradius.png")
pdf4 = os.path.join(output_dir, "case2_Rsweep_convergence_Rradius.pdf")
plt.savefig(png4, dpi=600, bbox_inches="tight")
plt.savefig(pdf4, dpi=600, bbox_inches="tight")
print("Saved:", png4, "and PDF")
plt.show()

print("Finished. All outputs (CSV + PNG + PDF) are in folder:", output_dir)

