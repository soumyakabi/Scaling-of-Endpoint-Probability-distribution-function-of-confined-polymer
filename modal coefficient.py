# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:10:49 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------- font size controls ----------------
label_fontsize = 18      # axis label font size (bold)
tick_fontsize  = 14      # tick label font size (bold)

# ---------------- stable Fourier routine (with renormalization) ----------------
def P_x_fourier(x, x0, N, a, L, max_modes=20000, tol=1e-15):
    x = np.asarray(x, dtype=np.float64)
    P = np.zeros_like(x)
    kappa = (N * a**2) / (L**2)
    modes_used = 0

    for n in range(1, max_modes + 1):
        lam = n * np.pi / L
        sin_x = np.sin(lam * x)
        sin_x0 = np.sin(lam * x0)
        decay = np.exp(-(n**2) * np.pi**2 * kappa / 8.0)
        term = (2.0 / L) * sin_x * sin_x0 * decay
        P += term
        modes_used = n
        if np.max(np.abs(term)) < tol:
            break

    P[P < 0] = 0
    integral = np.trapz(P, x)
    if integral <= 0 or not np.isfinite(integral):
        raise RuntimeError("Nonpositive or non-finite integral in P_x_fourier.")
    P /= integral
    return P, modes_used

# ---------------- analytic modal coefficient ----------------
def analytic_a_n(n_array, x0, L, kappa):
    n = np.asarray(n_array, dtype=int)
    pref = np.sqrt(2.0 / L)
    return pref * np.sin(n * np.pi * x0 / L) * np.exp(-(n**2) * np.pi**2 * kappa / 8.0)

# ---------------- numerical projection ----------------
def numerical_project(x, P, L, nmax):
    x = np.asarray(x, np.float64)
    P = np.asarray(P, np.float64)
    a_num = np.zeros(nmax)
    phi_norm = np.sqrt(2.0 / L)
    for n in range(1, nmax + 1):
        phi = phi_norm * np.sin(n * np.pi * x / L)
        a_num[n-1] = np.trapz(phi * P, x)
    return a_num

# ---------------- parameters ----------------
a = 1.0
L = 4.0
x0_fraction = 0.5
x0 = x0_fraction * L
kappa_list = [0.05, 0.1, 0.5, 1.0, 2.0]

eps = 1e-9
x_points = 3001
x = np.linspace(eps*L, (1-eps)*L, x_points)

nmax = 240
tol = 1e-15
max_modes_compute = 20000
n_display = 60
M_recon_max = 200
plot_floor = 1e-12

colors = plt.cm.tab10(np.linspace(0, 1, len(kappa_list)))
markers = ['o','s','D','^','v','<','>']

# ---------------- compute numeric + analytic coefficients ----------------
results = []
for kappa in kappa_list:
    N = kappa * L**2 / a**2
    P, modes_used = P_x_fourier(x, x0, N, a, L, max_modes_compute, tol)
    integral = np.trapz(P, x)

    a_num = numerical_project(x, P, L, nmax)
    n_arr = np.arange(1, nmax+1)
    a_analytic = analytic_a_n(n_arr, x0, L, kappa)
    energy = np.abs(a_analytic)**2
    total_energy = np.sum(energy)

    rel_err = np.linalg.norm(a_num - a_analytic) / (np.linalg.norm(a_analytic) + 1e-30)

    cum = np.cumsum(energy)
    idx90 = np.searchsorted(cum/total_energy, 0.90) + 1
    idx99 = np.searchsorted(cum/total_energy, 0.99) + 1

    results.append(dict(
        kappa=kappa, N=N, P=P, modes_used=modes_used, integral=integral,
        a_num=a_num, a_analytic=a_analytic,
        energy_analytic=energy, total_energy_analytic=total_energy,
        rel_err=rel_err, idx90=idx90, idx99=idx99
    ))

# ---------------- diagnostics print ----------------
print("Diagnostics:")
for r in results:
    print(f" κ={r['kappa']:.3g} | N={r['N']:.6g} | modes_used={r['modes_used']} | "
          f"integral={r['integral']:.6g} | rel_err={r['rel_err']:.3e} | "
          f"90%={r['idx90']} | 99%={r['idx99']}")

# ---------------- prepare mode index selection ----------------
n_all = np.arange(1, nmax+1)
is_central = abs(x0/L - 0.5) < 1e-12
odd_mask = (n_all % 2 == 1) if is_central else np.ones_like(n_all, bool)
n_plot_all = n_all[odd_mask]
n_plot = n_plot_all[:n_display]

# compute adaptive floor
min_pos = np.inf
for r in results:
    v = np.abs(r['a_analytic'][odd_mask][:n_display])
    pos = v[v > 0]
    if pos.size > 0:
        min_pos = min(min_pos, pos.min())
adaptive_floor = max(plot_floor, min_pos*1e-6) if min_pos < np.inf else plot_floor

# ---------------- plotting ----------------
fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# semilog modal amplitudes
for i, r in enumerate(results):
    aA = np.abs(r['a_analytic'][odd_mask][:n_display])
    aN = np.abs(r['a_num'][odd_mask][:n_display])

    a_plot = np.where(aN > 0, aN, np.nan)
    a_plot = np.where(a_plot < adaptive_floor, adaptive_floor, a_plot)

    ax1.semilogy(n_plot, a_plot, lw=1.4, color=colors[i], label=fr'κ={r["kappa"]}')
    ax1.plot(n_plot, aA[:len(n_plot)], marker=markers[i], linestyle='None',
             color=colors[i], ms=6)

ax1.set_xlabel(f'Mode index n (first {len(n_plot)} {"odd" if is_central else ""} modes)',
               fontsize=label_fontsize, fontweight='bold')
ax1.set_ylabel(r'$|c_n|$', fontsize=label_fontsize, fontweight='bold')
ax1.set_title('Modal coefficients |c_n| vs n', fontsize=16, fontweight='bold')
ax1.grid(alpha=0.25)
ax1.legend(fontsize=9)
ax1.tick_params(axis='both', labelsize=tick_fontsize)
for t in ax1.get_xticklabels() + ax1.get_yticklabels():
    t.set_fontweight("bold")

# cumulative analytic energy
for i, r in enumerate(results):
    cum = np.cumsum(r['energy_analytic'])
    cum_frac = cum / r['total_energy_analytic']
    ax2.plot(n_all, cum_frac, lw=1.6, color=colors[i], label=fr'κ={r["kappa"]}')

ax2.set_xlabel('mode index n', fontsize=label_fontsize, fontweight='bold')
ax2.set_ylabel('cumulative modal energy', fontsize=label_fontsize, fontweight='bold')
ax2.set_title('Cumulative modal energy', fontsize=16, fontweight='bold')
ax2.grid(alpha=0.25)
ax2.legend(fontsize=8)
ax2.tick_params(axis='both', labelsize=tick_fontsize)
for t in ax2.get_xticklabels() + ax2.get_yticklabels():
    t.set_fontweight("bold")

# inset
axins = inset_axes(ax2, width="45%", height="45%", loc='upper left', borderpad=2)
for i, r in enumerate(results):
    cum = np.cumsum(r['energy_analytic']) / r['total_energy_analytic']
    axins.plot(n_all[:60], cum[:60], lw=1.4, color=colors[i])
axins.set_xlim(1, 40); axins.set_ylim(0,1.02)
axins.set_title("Zoom n=1..40", fontsize=9)
axins.grid(alpha=0.25)
axins.xaxis.set_major_locator(ticker.MaxNLocator(5))

# low-mode amplitudes vs kappa
modes_show = [1,3,5,7,9] if is_central else [1,2,3,4,5]
kaps = [r['kappa'] for r in results]
for j,m in enumerate(modes_show):
    vals = [np.abs(r['a_analytic'][m-1]) for r in results]
    ax3.plot(kaps, vals, marker=markers[j], lw=1.6, label=f'|c_{m}|')

ax3.set_xscale('log')
ax3.set_xlabel(r'$\kappa$', fontsize=label_fontsize, fontweight='bold')
ax3.set_ylabel('low-mode amplitudes |c_n|', fontsize=label_fontsize, fontweight='bold')
ax3.set_title('Low modes vs κ', fontsize=16, fontweight='bold')
ax3.grid(alpha=0.25)
ax3.legend(fontsize=9)
ax3.tick_params(axis='both', labelsize=tick_fontsize)
for t in ax3.get_xticklabels() + ax3.get_yticklabels():
    t.set_fontweight("bold")

# reconstruction error vs M
Ms = np.arange(1, M_recon_max+1)
phi_norm = np.sqrt(2.0 / L)
S = np.sin(np.pi * np.outer(np.arange(1, M_recon_max+1), x/L))

for i, r in enumerate(results):
    rec_err = []
    aA = r['a_analytic']
    for M in Ms:
        P_M = np.dot(aA[:M], phi_norm * S[:M])
        err = np.sqrt(np.trapz((r['P'] - P_M)**2, x))
        rec_err.append(err)
    ax4.plot(Ms, rec_err, lw=1.3, color=colors[i], label=fr'κ={r["kappa"]}')

ax4.set_xlabel('Number of modes M', fontsize=label_fontsize, fontweight='bold')
ax4.set_ylabel('L2 reconstruction error', fontsize=label_fontsize, fontweight='bold')
ax4.set_title('Reconstruction error vs M', fontsize=16, fontweight='bold')
ax4.grid(alpha=0.25)
ax4.legend(fontsize=8)
ax4.tick_params(axis='both', labelsize=tick_fontsize)
for t in ax4.get_xticklabels() + ax4.get_yticklabels():
    t.set_fontweight("bold")

# ---------- finalize layout + saving ----------
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("modal_coeff_analysis_clean_fixed.pdf", dpi=600, bbox_inches='tight')
plt.savefig("modal_coeff_analysis_clean_fixed.png", dpi=600, bbox_inches='tight')

plt.show()

print("\nSaved: modal_coeff_analysis_clean_fixed.pdf and .png (600 dpi)")
