# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:48:39 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats

# ---------- analytic_P definition (same correct modal damping) ----------
def analytic_P(x, N, L, a, x0, n_max=20000, decay_tol=1e-16):
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
        if area <= 0:
            raise RuntimeError("Normalization failed.")
        P = P_raw / area
    else:
        P = num / (L * den)
    P[P < 0] = 0.0
    return P

# ---------------- user parameters ----------------
N = 10.0
L = 2.0
x0 = 0.5 * L
a_values = [0.05, 0.10, 0.20, 0.50, 1.00]

# numerical resolution
n_x = 8000
tiny = 1e-9
x_grid = np.linspace(tiny * L, (1 - tiny) * L, n_x)

# thresholds
EPS_SIG = 1e-6
# ------------------------------------------------

# storage
data = []
colors = plt.cm.viridis(np.linspace(0, 1, len(a_values)))

for (a, c) in zip(a_values, colors):
    P_x = analytic_P(x_grid, N=N, L=L, a=a, x0=x0, n_max=20000, decay_tol=1e-16)

    integral = np.trapz(P_x, x_grid)
    mean_x = np.trapz(x_grid * P_x, x_grid)
    mean_x2 = np.trapz((x_grid**2) * P_x, x_grid)
    sigma = np.sqrt(max(0.0, mean_x2 - mean_x**2))

    y = (x_grid - mean_x) / sigma if sigma > 0 else np.zeros_like(x_grid)
    P_y = sigma * P_x

    sort_idx = np.argsort(y)
    y_sorted = y[sort_idx]
    P_y_sorted = P_y[sort_idx]
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (P_y_sorted[:-1] + P_y_sorted[1:]) * np.diff(y_sorted))))
    cdf_interp = interpolate.interp1d(y_sorted, cdf, bounds_error=False, fill_value=(0.0,1.0))
    pdf_interp = interpolate.interp1d(y_sorted, P_y_sorted, bounds_error=False, fill_value=0.0)

    peak = P_y_sorted.max()
    mask_sig = P_y_sorted > (EPS_SIG * peak)
    if not np.any(mask_sig):
        mask_sig = P_y_sorted > (1e-8 * peak)
    y_sig_min, y_sig_max = y_sorted[mask_sig].min(), y_sorted[mask_sig].max()

    data.append({
        'a': a, 'color': c, 'x': x_grid, 'P_x': P_x,
        'y_sorted': y_sorted, 'P_y_sorted': P_y_sorted, 'cdf_interp': cdf_interp,
        'pdf_interp': pdf_interp, 'mean': mean_x, 'sigma': sigma,
        'peak': peak, 'y_sig_min': y_sig_min, 'y_sig_max': y_sig_max
    })

    print(f"a={a:.3f}: integral={integral:.6e}, mean={mean_x:.6e}, sigma={sigma:.6e}, "
          f"sig_y=[{y_sig_min:.3f}, {y_sig_max:.3f}]")

# ---------------- build common grids ----------------
y_inter_low = max(d['y_sig_min'] for d in data)
y_inter_high = min(d['y_sig_max'] for d in data)
if y_inter_high <= y_inter_low:
    widths = [d['y_sig_max'] - d['y_sig_min'] for d in data]
    median_width = np.median(widths)
    half = max(2.0, 0.5 * median_width)
    y_inter_low, y_inter_high = -half, half
    print("No full intersection; using fallback symmetric central window:", (y_inter_low, y_inter_high))

y_common = np.linspace(y_inter_low, y_inter_high, 1200)
p_grid = np.linspace(1e-6, 1 - 1e-6, 1000)
z_ref = stats.norm.ppf(p_grid)

# ---------------- PLOTTING ----------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
ax_cdf, ax_qq, ax_ratio = axes

# (1) Standardized CDFs
for d in data:
    cdf_vals = d['cdf_interp'](y_common)
    ax_cdf.plot(y_common, cdf_vals, color=d['color'], linewidth=2, label=f"a={d['a']:.2f}")
ax_cdf.set_xlabel('(x - <x>) / σ', fontsize=20, fontweight='bold')
ax_cdf.set_ylabel('CDF', fontsize=20, fontweight='bold')
ax_cdf.set_title('Standardized CDFs (common central window)', fontsize=16, fontweight='bold')
ax_cdf.legend(title='Kuhn length', fontsize=12, title_fontsize=13)
ax_cdf.tick_params(axis='both', which='major', labelsize=14)

# (2) QQ-plot
for d in data:
    P_sorted = d['P_y_sorted']
    y_sorted = d['y_sorted']
    cdf_array = np.concatenate(([0.0], np.cumsum(0.5 * (P_sorted[:-1] + P_sorted[1:]) * np.diff(y_sorted))))
    cdf_array = np.clip(cdf_array, 0.0, 1.0)
    inv_cdf = interpolate.interp1d(
        cdf_array,
        np.concatenate(([y_sorted[0]], y_sorted[1:])),
        bounds_error=False,
        fill_value=(y_sorted[0], y_sorted[-1])
    )
    y_q = inv_cdf(p_grid)
    ax_qq.plot(z_ref, y_q, color=d['color'], linewidth=1.8, label=f"a={d['a']:.2f}")
ax_qq.plot(z_ref, z_ref, 'k--', linewidth=1.4, label='y = z (normal)')
ax_qq.set_xlabel('Standard normal quantile z_p', fontsize=20, fontweight='bold')
ax_qq.set_ylabel('Empirical quantile y_p', fontsize=20, fontweight='bold')
ax_qq.set_title('QQ-plot: empirical standardized quantiles vs Normal', fontsize=16, fontweight='bold')
ax_qq.legend(title='Kuhn length', fontsize=12, title_fontsize=13)
ax_qq.tick_params(axis='both', which='major', labelsize=14)

# (3) Ratio P_y / φ(y)
phi_common = stats.norm.pdf(y_common)
for d in data:
    pdf_vals = d['pdf_interp'](y_common)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(phi_common > 0, pdf_vals / phi_common, np.nan)
    ax_ratio.plot(y_common, ratio, color=d['color'], linewidth=1.8, label=f"a={d['a']:.2f}")
    mask = np.isfinite(pdf_vals) & np.isfinite(phi_common)
    rms = np.sqrt(np.mean((pdf_vals[mask] - phi_common[mask])**2))
    print(f"a={d['a']:.2f}: RMS density error vs normal on intersection = {rms:.3e}")

ax_ratio.set_xlabel('(x - <x>) / σ', fontsize=20, fontweight='bold')
ax_ratio.set_ylabel('P_y / φ (ratio)', fontsize=20, fontweight='bold')
ax_ratio.set_title('Ratio to standard normal on intersection', fontsize=16, fontweight='bold')
ax_ratio.legend(title='Kuhn length', fontsize=12, title_fontsize=13)
ax_ratio.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# --- Save high-resolution PNG and PDF (600 dpi) ---
fig.savefig("case_normality_checks_600dpi.png", dpi=600, bbox_inches="tight")
fig.savefig("case_normality_checks_600dpi.pdf", dpi=600, bbox_inches="tight")
# --------------------------------------------------

plt.show()
