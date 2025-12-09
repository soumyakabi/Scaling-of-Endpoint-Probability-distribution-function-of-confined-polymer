# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:08:37 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Font size controls ----------------
label_fontsize = 18   # axis label font size (bold)
tick_fontsize = 14    # tick label font size

# ---------------- Parameters ----------------
kappa_values = [0.05, 0.1, 0.5, 1.0, 2.0]
y_points = 801
L = 1.0
ell = 1.0

# Optimized vs reference exponents
alpha_opt, alpha_ref = 0.5, 3.0
beta_opt, beta_ref   = 0.5, 1.0

# ---------------- Mock-up probability function ----------------
def P_y(y, kappa):
    """ Placeholder for P(y). """
    return np.exp(-kappa * (y**2)) * (1 - (np.abs(y)/L)**2)

y = np.linspace(-L, L, y_points)

# ---------------- Compute transformed variables ----------------
results = []
for kappa in kappa_values:
    P = P_y(y, kappa)
    mu = 1.0

    # Scaling variable
    eta = (L - np.abs(y)) / ell

    alpha_opt_curve = mu * P * eta**alpha_opt
    alpha_ref_curve = mu * P * eta**alpha_ref
    beta_opt_curve  = mu * P * eta**beta_opt
    beta_ref_curve  = mu * P * eta**beta_ref

    results.append({
        'kappa': kappa,
        'eta': eta,
        'alpha_opt': alpha_opt_curve,
        'alpha_ref': alpha_ref_curve,
        'beta_opt': beta_opt_curve,
        'beta_ref': beta_ref_curve
    })

# ---------------- Plotting ----------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
ax_aopt, ax_aref, ax_res_alpha = axes[0]
ax_bopt, ax_bref, ax_res_beta  = axes[1]

colors = plt.cm.tab10.colors
markers = ['o','s','^','D','v']
linestyles = ['-','--','-.',':',(0,(3,1,1,1))]

# Use kappa = 1 curve as reference for residuals and overlay
kappa_ref_index = kappa_values.index(1.0)
ref_alpha_opt = results[kappa_ref_index]['alpha_opt']
ref_beta_opt  = results[kappa_ref_index]['beta_opt']
eta_ref       = results[kappa_ref_index]['eta']

# ---------------- Draw curves ----------------
for i, res in enumerate(results):
    label = fr'$\kappa={res["kappa"]}$'
    style = dict(color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)],
                 lw=1.6)

    # α optimized
    ax_aopt.plot(res['eta'], res['alpha_opt'], label=label, **style)
    ax_aopt.plot(res['eta'][::80], res['alpha_opt'][::80], marker=markers[i%len(markers)],
                 linestyle='None', color=colors[i%len(colors)])

    # α reference
    ax_aref.plot(res['eta'], res['alpha_ref'], label=label, **style)
    ax_aref.plot(res['eta'][::80], res['alpha_ref'][::80], marker=markers[i%len(markers)],
                 linestyle='None', color=colors[i%len(colors)])

    # β optimized
    ax_bopt.plot(res['eta'], res['beta_opt'], label=label, **style)
    ax_bopt.plot(res['eta'][::80], res['beta_opt'][::80], marker=markers[i%len(markers)],
                 linestyle='None', color=colors[i%len(colors)])

    # β reference
    ax_bref.plot(res['eta'], res['beta_ref'], label=label, **style)
    ax_bref.plot(res['eta'][::80], res['beta_ref'][::80], marker=markers[i%len(markers)],
                 linestyle='None', color=colors[i%len(colors)])

    # Residuals α
    ax_res_alpha.plot(res['eta'], res['alpha_opt'] - ref_alpha_opt, **style)

    # Residuals β
    ax_res_beta.plot(res['eta'], res['beta_opt'] - ref_beta_opt, **style)

# Reference overlays
ax_aopt.plot(eta_ref, ref_alpha_opt, 'k--', lw=1.0, label="ref κ=1")
ax_bopt.plot(eta_ref, ref_beta_opt, 'k--', lw=1.0, label="ref κ=1")

# ---------------- Titles ----------------
ax_aopt.set_title(fr'Optimized scaling ($\alpha={alpha_opt}$)')
ax_aref.set_title(fr'Reference scaling ($\alpha={alpha_ref}$)')
ax_bopt.set_title(fr'Optimized scaling ($\beta={beta_opt}$)')
ax_bref.set_title(fr'Reference scaling ($\beta={beta_ref}$)')
ax_res_alpha.set_title(r'Residuals (optimized $\alpha$)')
ax_res_beta.set_title(r'Residuals (optimized $\beta$)')

# ---------------- Decoration helper ----------------
def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    ax.grid(alpha=0.25)

for ax in [ax_aopt, ax_aref, ax_bopt, ax_bref, ax_res_alpha, ax_res_beta]:
    style_axes(ax, r'$\eta = (L-|y|)/\ell$', r'$\mu P(y)$')

# ---------------- External legend ----------------
handles, labels = ax_aopt.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right',
           bbox_to_anchor=(0.96, 0.5), frameon=False)

plt.tight_layout(rect=[0,0,0.88,1.0])

# ---------------- Save High-Resolution Outputs ----------------
plt.savefig("Case5_scalings.pdf", dpi=600, bbox_inches='tight')
plt.savefig("Case5_scalings.png", dpi=600, bbox_inches='tight')

plt.show()
