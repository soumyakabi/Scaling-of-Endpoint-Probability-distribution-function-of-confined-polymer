# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:08:05 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ---------------- font size controls ----------------
label_fontsize = 18   # axis labels bold 18
tick_fontsize = 14    # tick labels

# ---------------- image method routine ----------------
def P_y_image(y, N, a, L, m_max=200):
    """
    Endpoint distribution using method of images (absorbing walls).
    """
    sigma = np.sqrt(N * (a**2))
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    P = np.zeros_like(y, dtype=float)

    for m in range(-m_max, m_max + 1):
        P += (-1.0)**m * np.exp(-(y - 2.0*m*L)**2 / (2.0 * sigma**2))

    P = prefactor * P
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

# ---------------- parameters ----------------
a = 1.0
L_fixed = 4.0
kappa_values = [0.05, 0.1, 0.5, 1.0, 2.0]
y_points = 2001
m_max = 300

colors = plt.cm.tab10.colors
linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1))]
markers = ['o','s','^','D','v']

# ---------------- compute curves ----------------
curves = []
for i, kappa in enumerate(kappa_values):
    N = kappa * (L_fixed**2) / (a**2)
    y = np.linspace(-L_fixed, L_fixed, y_points)

    P = P_y_image(y, N, a, L_fixed, m_max=m_max)
    surv = np.trapz(P, y)

    u = y / L_fixed
    mathcalP = L_fixed * P

    if surv > 0:
        mathcalP_cond = mathcalP / surv
    else:
        mathcalP_cond = mathcalP * 0.0

    cond_var = np.trapz(u**2 * mathcalP_cond, u)

    peak_val = np.max(mathcalP_cond) if np.max(mathcalP_cond) > 0 else 1.0
    mathcalP_peak = mathcalP_cond / peak_val

    curves.append({
        'kappa': kappa,
        'N': N,
        'y': y,
        'u': u,
        'P': P,
        'mathcalP': mathcalP,
        'surv': surv,
        'mathcalP_cond': mathcalP_cond,
        'mathcalP_peak': mathcalP_peak,
        'cond_var': cond_var
    })

# ---------------- plotting ----------------
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
ax1, ax2, ax_diag, ax3 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

for i, c in enumerate(curves):
    label = fr'$\kappa={c["kappa"]}$, $N={c["N"]:.3g}$'
    style = dict(color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 lw=1.6)

    ax1.plot(c['u'], c['mathcalP'], label=label, **style)
    ax1.plot(c['u'][::200], c['mathcalP'][::200], marker=markers[i%len(markers)],
             color=colors[i%len(colors)], linestyle='None')

    ax2.plot(c['u'], c['mathcalP_cond'], label=label, **style)
    ax2.plot(c['u'][::200], c['mathcalP_cond'][::200], marker=markers[i%len(markers)],
             color=colors[i%len(colors)], linestyle='None')

    ax3.plot(c['u'], c['mathcalP_peak'], label=label, **style)
    ax3.plot(c['u'][::200], c['mathcalP_peak'][::200], marker=markers[i%len(markers)],
             color=colors[i%len(colors)], linestyle='None')

# inset zoom for conditional
axins = inset_axes(ax2, width="40%", height="40%", loc='upper left',
                   bbox_to_anchor=(0.05,0.45,0.5,0.5),
                   bbox_transform=ax2.transAxes)
zoom_range = 0.4
for i, c in enumerate(curves):
    axins.plot(c['u'], c['mathcalP_cond'], color=colors[i%len(colors)], lw=1.4)

axins.set_xlim(-zoom_range, zoom_range)
axins.set_ylim(0, max(c['mathcalP_cond'].max() for c in curves)*1.05)
axins.grid(alpha=0.2)
mark_inset(ax2, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# ---------------- decorate ----------------
def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(alpha=0.25)

style_axes(ax1, r'$y/L$', r'$\mathcal{P}(u)$', 'Unconditional: $\\mathcal{P}(u)=L P(y)$')
style_axes(ax2, r'$y/L$', r'$\mathcal{P}_{cond}(u)$', 'Conditional (normalized)')
style_axes(ax3, r'$y/L$', 'Peak-normalized', 'Peak-normalized (shape comparison)')

# diagnostics
kappa_list = [c['kappa'] for c in curves]
surv_list = [c['surv'] for c in curves]
cond_var_list = [c['cond_var'] for c in curves]

ax_diag.set_xscale('log')
ax_diag.set_yscale('log')

ax_diag.plot(kappa_list, surv_list, 'o-b', label='Survival $S(\\kappa)$')
ax_diag.plot(kappa_list, cond_var_list, 's--', color='orange',
             label='Cond. Var$(u^2)$')

style_axes(ax_diag,
           r'$\kappa$ (confinement parameter)',
           'Survival / Variance (log-log)',
           'Diagnostics: $S(\\kappa)$ and conditional variance')

ax_diag.legend()

# legends outside
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
ax3.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)

plt.tight_layout(rect=[0,0,0.85,1.0])

# ---------------- SAVE FIGURES (600 dpi PNG + PDF) ----------------
plt.savefig('Case4_with_diagnostics.pdf', dpi=600, bbox_inches='tight')
plt.savefig('Case4_with_diagnostics.png', dpi=600, bbox_inches='tight')

plt.show()

# ---------------- print diagnostics ----------------
print("kappa     N        survival_prob    cond_var")
for c in curves:
    print(f"{c['kappa']:8g} {c['N']:10.4g} {c['surv']:14.6g} {c['cond_var']:14.6g}")
