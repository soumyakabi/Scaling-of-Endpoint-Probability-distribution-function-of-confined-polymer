# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:49:48 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------- core modal expansion ----------------
def P_x_fourier(x, x0, N, a, L, max_modes=20000, tol=1e-16):
    x = np.asarray(x, dtype=np.float64)
    P = np.zeros_like(x)
    kappa = (N * a**2) / (L**2)
    modes_used = 0
    for n in range(1, max_modes + 1):
        lam = n * np.pi / L
        sin_x = np.sin(lam * x)
        sin_x0 = np.sin(lam * x0)
        decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)
        term = (2.0 / L) * sin_x * sin_x0 * decay
        P += term
        modes_used = n
        if np.max(np.abs(term)) < tol:
            break
    P[P < 0] = 0.0
    integral = np.trapz(P, x)
    if not np.isfinite(integral) or integral <= 0:
        raise RuntimeError("P_x_fourier produced nonpositive or non-finite integral.")
    P /= integral
    return P, modes_used

# ---------------- helpers ----------------
def first_mode_slope_theory(x0, L, N, a):
    kappa = (N * a**2) / (L**2)
    pref = np.exp(- (np.pi**2) * kappa / 8.0)
    return (2.0 * np.pi * np.sin(np.pi * x0 / L) / (L**2)) * pref

def compute_delta_nearest_wall(x, L):
    return np.minimum(x, L - x)

def fit_linear_slope(eta, y, eta_max_fit=0.15):
    mask = (eta >= 0) & (eta <= eta_max_fit)
    if mask.sum() < 3:
        return np.nan, np.nan
    A = np.vstack([eta[mask], np.ones(mask.sum())]).T
    m, c = np.linalg.lstsq(A, y[mask], rcond=None)[0]
    return m, c

def finite_diff_slope(eta, y, window=10):
    order = np.argsort(eta)
    e = eta[order][:window]
    v = y[order][:window]
    A = np.vstack([e, np.ones(len(e))]).T
    m, c = np.linalg.lstsq(A, v, rcond=None)[0]
    return m

def rms_on_common_grid(curves_vals, n_points=300):
    valid = [(np.asarray(eta), np.asarray(vals)) for eta, vals in curves_vals if len(eta) >= 2]
    if len(valid) < 2:
        return np.nan
    mins = [eta.min() for eta, _ in valid]
    maxs = [eta.max() for eta, _ in valid]
    eta_min = max(mins)
    eta_max = min(maxs)
    if eta_max <= eta_min:
        return np.nan
    eta_common = np.linspace(eta_min, eta_max, n_points)
    M = len(valid)
    mat = np.zeros((M, n_points), dtype=float)
    for i, (eta, vals) in enumerate(valid):
        mat[i, :] = np.interp(eta_common, eta, vals)
    mean = np.mean(mat, axis=0)
    rms = np.sqrt(np.mean((mat - mean[None, :])**2))
    return rms

# ---------------- parameters ----------------
a = 1.0
L = 4.0
x0 = 0.5 * L
kappa_list = [0.05, 0.1, 0.5, 1.0, 2.0]

eps = 1e-8
n_x = 4000
x = np.linspace(eps * L, (1 - eps) * L, n_x)

eta_max_display = 2.0
eta_fit = 0.15

# ---------------- compute P(x) ----------------
cases = []
for kappa in kappa_list:
    N = kappa * (L**2) / (a**2)
    P, modes_used = P_x_fourier(x, x0, N, a, L)
    sigma = np.sqrt(N * a**2)
    delta = compute_delta_nearest_wall(x, L)
    cases.append({
        'kappa': kappa, 'N': N, 'x': x, 'P': P,
        'modes_used': modes_used, 'sigma': sigma, 'delta': delta
    })
    print(f"Computed κ={kappa:.3g}, N={N:.6g}, sigma={sigma:.6g}, modes_used={modes_used}, integral={np.trapz(P,x):.6g}")

# ---------------- optimized alpha ----------------
def objective_alpha(a0):
    if a0 <= 0:
        return 1e9
    curves = [(c['delta']/(a0 * c['sigma']), (a0 * c['sigma'] * c['P'])) for c in cases]
    val = rms_on_common_grid(curves, n_points=400)
    return float(1e6) if np.isnan(val) else float(val)

res_alpha = minimize_scalar(objective_alpha, bounds=(0.05, 10.0), method='bounded')
alpha_best = res_alpha.x
rms_best = res_alpha.fun
print(f"\nBest alpha for ell = alpha*sigma -> alpha = {alpha_best:.6f}, RMS = {rms_best:.6e}")

# ---------------- compare simple choices ----------------
ell_sigma_curves = [(c['delta']/c['sigma'], (c['sigma'] * c['P'])) for c in cases]
ell_Lpi_curves   = [(c['delta']/(L/np.pi), ((L/np.pi) * c['P'])) for c in cases]
rms_sigma = rms_on_common_grid(ell_sigma_curves, n_points=400)
rms_Lpi   = rms_on_common_grid(ell_Lpi_curves, n_points=400)
print(f"RMS (ell = sigma) = {rms_sigma:.6e}")
print(f"RMS (ell = L/pi) = {rms_Lpi:.6e}")

# ---------------- slope diagnostics ----------------
print("\nSlope diagnostics (measured vs theoretical):")
for c in cases:
    ell = c['sigma']
    eta = c['delta'] / ell
    yvals = ell * c['P']
    order = np.argsort(eta)
    eta_s, y_s = eta[order], yvals[order]
    mask = (eta_s >= 0) & (eta_s <= eta_fit)
    npt = mask.sum()
    if npt >= 3:
        m_meas, _ = fit_linear_slope(eta_s, y_s, eta_max_fit=eta_fit)
        m_fd = finite_diff_slope(eta_s, y_s, window=min(10, npt))
    else:
        m_meas, m_fd = np.nan, np.nan
    m_th = first_mode_slope_theory(x0, L, c['N'], a) * (ell**2)
    ratio = m_meas / m_th if m_th != 0 else np.nan
    print(f"ℓ=σ: κ={c['kappa']:.3g}, npt={npt}, "
          f"measured={m_meas:.4e}, FD={m_fd:.4e}, theory={m_th:.4e}, ratio={ratio:.3f}")

    ell2 = L / np.pi
    eta2 = c['delta'] / ell2
    yvals2 = ell2 * c['P']
    order2 = np.argsort(eta2)
    eta2_s, y2_s = eta2[order2], yvals2[order2]
    mask2 = (eta2_s >= 0) & (eta2_s <= eta_fit)
    npt2 = mask2.sum()
    if npt2 >= 3:
        m_meas2, _ = fit_linear_slope(eta2_s, y2_s, eta_max_fit=eta_fit)
        m_fd2 = finite_diff_slope(eta2_s, y2_s, window=min(10, npt2))
    else:
        m_meas2, m_fd2 = np.nan, np.nan
    m_th2 = first_mode_slope_theory(x0, L, c['N'], a) * (ell2**2)
    ratio2 = m_meas2 / m_th2 if m_th2 != 0 else np.nan
    print(f"ℓ=L/π: κ={c['kappa']:.3g}, npt={npt2}, "
          f"measured={m_meas2:.4e}, FD={m_fd2:.4e}, theory={m_th2:.4e}, ratio={ratio2:.3f}")

# ---------------- RMS collapse ----------------
curves_Lpi = []
for c in cases:
    ell = L / np.pi
    eta = c['delta'] / ell
    yvals = ell * c['P']
    curves_Lpi.append((eta, yvals))
rms_val_Lpi = rms_on_common_grid(curves_Lpi, n_points=300)
print(f"\nRMS residual across ℓ=L/π collapsed curves = {rms_val_Lpi:.6e}")

# ---------------- plotting ----------------
plt.rcParams.update({'font.size': 12})
colors = plt.cm.viridis(np.linspace(0, 1, len(cases)))
markers = ['o', 's', '^', 'D', 'v']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax1, ax2 = axes

# left: ell = sigma
for i, c in enumerate(cases):
    ell = c['sigma']
    eta = c['delta'] / ell
    yvals = ell * c['P']
    mask = (eta >= 0) & (eta <= eta_max_display)
    ax1.plot(eta[mask], yvals[mask], color=colors[i], lw=1.8)
    ax1.scatter(eta[mask][::120], yvals[mask][::120], marker=markers[i], color=colors[i], s=28)
    C = first_mode_slope_theory(x0, L, c['N'], a)
    m_th = C * (ell**2)
    xlin = np.linspace(0, min(0.6, eta[mask].max()), 50)
    ax1.plot(xlin, m_th * xlin, color=colors[i], lw=1.0, linestyle='--')

# right: ell = L/pi
ell_Lpi = L / np.pi
for i, c in enumerate(cases):
    ell = ell_Lpi
    eta = c['delta'] / ell
    yvals = ell * c['P']
    mask = (eta >= 0) & (eta <= eta_max_display)
    ax2.plot(eta[mask], yvals[mask], color=colors[i], lw=1.8)
    ax2.scatter(eta[mask][::120], yvals[mask][::120], marker=markers[i], color=colors[i], s=28)
    C = first_mode_slope_theory(x0, L, c['N'], a)
    m_th = C * (ell**2)
    xlin = np.linspace(0, min(0.6, eta[mask].max()), 50)
    ax2.plot(xlin, m_th * xlin, color=colors[i], lw=1.0, linestyle='--')

# insets
axins1 = inset_axes(ax1, width="30%", height="28%", loc='upper right',
                    bbox_to_anchor=(0, -0.05, 1, 1), bbox_transform=ax1.transAxes)
axins2 = inset_axes(ax2, width="30%", height="28%", loc='lower right',
                    bbox_to_anchor=(0, 0.07, 1, 1), bbox_transform=ax2.transAxes)

ymax_sigma = 0.0
ymax_Lpi = 0.0
for c in cases:
    eta1 = c['delta'] / c['sigma']
    y1 = c['sigma'] * c['P']
    mask1 = (eta1 >= 0) & (eta1 <= 0.2)
    if np.any(mask1):
        ymax_sigma = max(ymax_sigma, np.nanmax(y1[mask1]))

    eta2 = c['delta'] / (L / np.pi)
    y2 = (L / np.pi) * c['P']
    mask2 = (eta2 >= 0) & (eta2 <= 0.2)
    if np.any(mask2):
        ymax_Lpi = max(ymax_Lpi, np.nanmax(y2[mask2]))

if ymax_sigma == 0.0: ymax_sigma = 0.2
if ymax_Lpi == 0.0: ymax_Lpi = 0.05

for i, c in enumerate(cases):
    eta1 = c['delta'] / c['sigma']
    y1 = c['sigma'] * c['P']
    mask1 = (eta1 >= 0) & (eta1 <= 0.2)
    if np.any(mask1):
        axins1.plot(eta1[mask1], y1[mask1], color=colors[i], lw=1.2)

    eta2 = c['delta'] / (L / np.pi)
    y2 = (L / np.pi) * c['P']
    mask2 = (eta2 >= 0) & (eta2 <= 0.2)
    if np.any(mask2):
        axins2.plot(eta2[mask2], y2[mask2], color=colors[i], lw=1.2)

axins1.set_xlim(0, 0.2); axins1.set_ylim(0, ymax_sigma * 1.05)
axins2.set_xlim(0, 0.2); axins2.set_ylim(0, ymax_Lpi * 1.05)
axins1.set_title(r'zoom $\eta\in[0,0.2]$', fontsize=10)
axins2.set_title(r'zoom $\eta\in[0,0.2]$', fontsize=10)

# labels
ax1.set_xlabel(r'$\eta=\delta/\ell,\ \ell=\sigma$', fontsize=18, fontweight='bold')
ax1.set_ylabel(r'$\ell\,P(x)$', fontsize=18, fontweight='bold')
ax1.set_title(r'Inner scaling: $\ell=\sigma$', fontsize=14)

ax2.set_xlabel(r'$\eta=\delta/\ell,\ \ell=L/\pi$', fontsize=18, fontweight='bold')
ax2.set_title(r'Inner scaling: $\ell=L/\pi$', fontsize=14)

labels = [f'κ={c["kappa"]}' for c in cases]
ax1.legend(labels, title='κ (left)', fontsize=10, title_fontsize=11, loc='upper left')
ax2.legend(labels, title='κ (right)', fontsize=10, title_fontsize=11, loc='upper left')

plt.suptitle(f'Boundary-layer inner scaling near walls (L={L}, x0/L={x0/L:.2f})', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# ---------------- SAVE FIGURES ----------------
fig.savefig("boundary_layer_scaling.png", dpi=600, bbox_inches='tight')
fig.savefig("boundary_layer_scaling.pdf", bbox_inches='tight')

plt.show()