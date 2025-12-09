# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:05:08 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# ---------- style ----------
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "figure.figsize": (14, 10)
})
palette = plt.get_cmap('tab10')

# ---------- font size control ----------
label_fontsize = 18     # changed to 18 (bold required)
tick_fontsize = 12

# ---------- math routines: method of images ----------
def P_image(y, sigma, L, m_max=200):
    P = np.zeros_like(y, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for m in range(-m_max, m_max + 1):
        term = (-1) ** m * np.exp(-(y - 2.0 * m * L) ** 2 / (2.0 * sigma ** 2))
        P += term
    P = prefactor * P
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

def P_tilde_image(s, lam, m_max=200):
    sigma = 1.0
    y = s * sigma
    L = float(lam)
    return P_image(y, sigma, L, m_max=m_max)

# ---------- parameters ----------
kappa_list = [0.01, 0.1, 0.5, 2.0, 8.0]
lambda_list = [1.0 / np.sqrt(k) for k in kappa_list]
colors = [palette(i) for i in range(len(kappa_list))]
max_lambda = max(lambda_list)

m_max = 300
density_per_unit = 1200
Npoints = int(max(20001, density_per_unit * int(np.ceil(2.0 * max_lambda))))
s_common = np.linspace(-1.05 * max_lambda, 1.05 * max_lambda, Npoints)

# ---------- compute PDFs ----------
P_uncond_list = []
P_cond_list = []
integrals = []

for lam in lambda_list:
    P_on_common = P_tilde_image(s_common, lam, m_max=m_max)
    P_on_common[P_on_common < 0.0] = 0.0
    integral = np.trapz(P_on_common, s_common)

    if integral <= 0.0:
        P_cond = np.zeros_like(P_on_common)
    else:
        P_cond = P_on_common / integral

    P_uncond_list.append(P_on_common)
    P_cond_list.append(P_cond)
    integrals.append(integral)

# ---------- theoretical Gaussian ----------
s_th = np.linspace(-max_lambda * 1.05, max_lambda * 1.05, 3001)
gauss_th = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s_th ** 2)

# ---------- plotting ----------
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig,
                       height_ratios=[1, 0.5], hspace=0.25, wspace=0.3)

ax_uncond = fig.add_subplot(gs[0, 0])
ax_cond = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])

# ---------- Unconditional panel ----------
for P_on_common, lam, kappa, col in zip(P_uncond_list, lambda_list, kappa_list, colors):
    ax_uncond.plot(s_common, P_on_common, color=col, lw=1.6,
                   label=f'κ={kappa} (λ={lam:.3g})', alpha=0.95)

ax_uncond.plot(s_th, gauss_th, 'k--', lw=1.2, label='Free Gaussian (theory)')
ax_uncond.set_xlim(-max_lambda * 1.02, max_lambda * 1.02)

ax_uncond.set_xlabel(r'$s = y/\sigma$', fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_ylabel(r'$\widetilde{P}(s)=\sigma P(y)$ (unconditional)',
                     fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_title('Unconditional scaled PDFs (area = survival probability)',
                    fontweight='bold')

ax_uncond.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_uncond.get_xticklabels() + ax_uncond.get_yticklabels():
    label.set_fontweight('bold')

ax_uncond.grid(alpha=0.25)
ax_uncond.legend(loc='upper right', fontsize=9, framealpha=0.9)

# inset
zoom_range = 0.6 * (2.0 if max_lambda > 2.0 else max_lambda)
zoom_range = max(0.1, float(zoom_range))
axins_u = inset_axes(ax_uncond, width="38%", height="38%", loc='lower left',
                     bbox_to_anchor=(0.05, 0.05, 0.5, 0.5), bbox_transform=ax_uncond.transAxes)
axins_u.set_xlim(-zoom_range, zoom_range)
center_mask = np.abs(s_common) <= zoom_range
if not np.any(center_mask):
    center_mask = np.abs(s_common) <= (0.02 * max_lambda + 1e-6)

max_u = max([P_uncond_list[i][center_mask].max() for i in range(len(P_uncond_list))])
axins_u.set_ylim(0, 1.05 * max_u)

for P_on_common, col in zip(P_uncond_list, colors):
    axins_u.plot(s_common, P_on_common, color=col, lw=1.4, alpha=0.95)

axins_u.plot(s_th, gauss_th, 'k--', lw=1.0)
axins_u.grid(alpha=0.2)
mark_inset(ax_uncond, axins_u, loc1=2, loc2=4, fc="none", ec="0.5")

# ---------- Conditional panel ----------
for P_cond, lam, kappa, col in zip(P_cond_list, lambda_list, kappa_list, colors):
    ax_cond.plot(s_common, P_cond, color=col, lw=1.6,
                 label=f'κ={kappa} (λ={lam:.3g})', alpha=0.95)

ax_cond.set_xlim(-max_lambda * 1.02, max_lambda * 1.02)

ax_cond.set_xlabel(r'$s = y/\sigma$', fontsize=label_fontsize, fontweight='bold')
ax_cond.set_ylabel(r'Conditional $\widetilde{P}(s)$ (normalized)',
                   fontsize=label_fontsize, fontweight='bold')
ax_cond.set_title('Conditional scaled PDFs (normalized)', fontweight='bold')

ax_cond.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_cond.get_xticklabels() + ax_cond.get_yticklabels():
    label.set_fontweight('bold')

ax_cond.grid(alpha=0.25)
ax_cond.legend(loc='upper right', fontsize=9, framealpha=0.9)

# inset
axins_c = inset_axes(ax_cond, width="38%", height="38%", loc='lower left',
                     bbox_to_anchor=(0.05, 0.05, 0.5, 0.5), bbox_transform=ax_cond.transAxes)
axins_c.set_xlim(-zoom_range, zoom_range)

if not np.any(center_mask):
    center_mask = np.abs(s_common) <= (0.02 * max_lambda + 1e-6)

max_c = max([P_cond_list[i][center_mask].max() for i in range(len(P_cond_list))])
axins_c.set_ylim(0, 1.05 * max_c)

for P_cond, col in zip(P_cond_list, colors):
    axins_c.plot(s_common, P_cond, color=col, lw=1.4, alpha=0.95)

axins_c.grid(alpha=0.2)
mark_inset(ax_cond, axins_c, loc1=2, loc2=4, fc="none", ec="0.5")

# ---------- Bottom panel ----------
S_vals = integrals
cond_var = []

for P_cond in P_cond_list:
    mean_s2 = np.trapz((s_common**2) * P_cond, s_common)
    cond_var.append(mean_s2)

ax_bottom.plot(kappa_list, S_vals, 'o-b', lw=1.8, ms=7, label=r'Survival $S(\kappa)$')
ax_bottom.plot(kappa_list, cond_var, 's--', color='orange', lw=1.8, ms=7,
               label=r'Cond. Var $\langle s^2\rangle_{cond}$')

ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')

ax_bottom.set_xlabel(r'$\kappa$ (confinement parameter)',
                     fontsize=label_fontsize, fontweight='bold')
ax_bottom.set_ylabel(r'Survival prob. / Cond. variance',
                     fontsize=14, fontweight='bold')
ax_bottom.set_title(r'Survival probability and conditional variance vs $\kappa$',
                    fontweight='bold')

ax_bottom.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_bottom.get_xticklabels() + ax_bottom.get_yticklabels():
    label.set_fontweight('bold')

ax_bottom.grid(True, which="both", ls="--", alpha=0.5)
ax_bottom.legend(fontsize=10, loc='best', framealpha=0.9)

# diagnostics annotation
diag_lines = []
for i, kappa in enumerate(kappa_list):
    diag_lines.append(f"κ={kappa:.3g}, λ={lambda_list[i]:.3f}, "
                      f"S={S_vals[i]:.3g}, Var={cond_var[i]:.3g}")
diag_text = "\n".join(diag_lines)

ax_bottom.text(0.99, 0.01, diag_text,
               transform=ax_bottom.transAxes,
               fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

# ---------- final adjustments + save ----------
plt.subplots_adjust(top=0.95, bottom=0.08, left=0.06, right=0.92,
                    hspace=0.28, wspace=0.28)

png_name = "Case3_ImageMethod.png"
pdf_name = "Case3_ImageMethod.pdf"

plt.savefig(pdf_name, dpi=600, bbox_inches='tight')
plt.savefig(png_name, dpi=600, bbox_inches='tight')

print("Saved:", pdf_name, "and", png_name)

plt.show()
