# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:00:33 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Image method
# ----------------------------
def P_image(y, sigma, L, m_max=200):
    """Endpoint distribution with absorbing walls via method of images."""
    P = np.zeros_like(y, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for m in range(-m_max, m_max+1):
        term = (-1)**m * np.exp(-(y - 2*m*L)**2 / (2.0 * sigma**2))
        P += term
    P = prefactor * P
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

# ----------------------------
# Parameters
# ----------------------------
L = 1.0
kappa_list = [0.02, 0.05, 0.1, 0.5, 1.0, 2.0]
colors = plt.cm.tab10.colors
ny = 2001
y = np.linspace(-L, L, ny)

label_fontsize = 18   # axis label font size (bold)

# Storage
results = {}
for kappa in kappa_list:
    sigma = np.sqrt(kappa) * L
    P = P_image(y, sigma, L, m_max=400)
    S = np.trapz(P, y)   
    P_cond = P / S if S > 0 else P*0
    var_cond = np.trapz((y**2) * P_cond, y)
    
    results[kappa] = dict(
        sigma=sigma, P=P, P_cond=P_cond,
        S=S, var_cond=var_cond, L_over_sigma=L/sigma
    )

# ----------------------------
# Plotting
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# --- (a) Unconditional scaled ---
ax = axes[0,0]
for i, kappa in enumerate(kappa_list):
    u = y/L
    mathcalP = L*results[kappa]['P']
    ax.plot(u, mathcalP, label=fr"$\kappa={kappa}$", color=colors[i%10])

ax.set_xlabel(r"$u = y/L$", fontsize=label_fontsize, fontweight="bold")
ax.set_ylabel(r"$\mathcal{P}(u) = L P(y)$", fontsize=label_fontsize, fontweight="bold")
ax.set_title("Case 2: unconditional scaled distributions (varying κ)",fontsize=12)
ax.legend(); ax.grid(alpha=0.3)

# --- (b) Conditional scaled ---
ax = axes[0,1]
for i, kappa in enumerate(kappa_list):
    u = y/L
    mathcalP_cond = L*results[kappa]['P_cond']
    ax.plot(u, mathcalP_cond, label=fr"$\kappa={kappa}$", color=colors[i%10])

ax.set_xlabel(r"$u = y/L$", fontsize=label_fontsize, fontweight="bold")
ax.set_ylabel(r"$\mathcal{P}_{\mathrm{cond}}(u) = L P_{\mathrm{cond}}(y)$",
              fontsize=label_fontsize, fontweight="bold")
ax.set_title("Conditional (survivors) scaled distributions",fontsize=12)
ax.legend(); ax.grid(alpha=0.3)

# --- (c) Coil-scaled conditional densities ---
ax = axes[1,0]
for i, kappa in enumerate(kappa_list):
    sigma = results[kappa]['sigma']
    s = y/sigma
    P_cond = results[kappa]['P_cond']
    P_coil = sigma * P_cond
    style = '-' if results[kappa]['L_over_sigma'] >= 3 else '--'
    
    ax.plot(s, P_coil, label=fr"$\kappa={kappa}$",
            color=colors[i%10], linestyle=style)
    
    if results[kappa]['L_over_sigma'] < 3:
        x_pos = 0.5 * results[kappa]['L_over_sigma']
        y_pos = max(P_coil) * 0.7
        ax.text(x_pos, y_pos,
                fr"$L/\sigma={results[kappa]['L_over_sigma']:.2f}$",
                color=colors[i%10], fontsize=9, weight='bold')

ax.set_xlabel(r"$s = y/\sigma$", fontsize=label_fontsize, fontweight="bold")
ax.set_ylabel(r"$\widehat{P}(s) = \sigma P_{\mathrm{cond}}(y)$",
              fontsize=label_fontsize, fontweight="bold")
ax.set_title("Coil-scaled conditional densities (small κ)",fontsize=12)
ax.legend(); ax.grid(alpha=0.3)

# --- (d) Diagnostics ---
ax = axes[1,1]
kap = np.array(kappa_list)
Svals = [results[k]['S'] for k in kappa_list]
Vvals = [results[k]['var_cond'] for k in kappa_list]

ax2 = ax.twinx()
ax.plot(kap, Svals, 'o-b', label="Survival $S(\\kappa)$")
ax2.plot(kap, Vvals, 's--', color='orange', label="Conditional variance")

ax.set_xscale('log')
ax.set_xlabel(r"$\kappa$", fontsize=label_fontsize, fontweight="bold")
ax.set_ylabel(r"$S(\kappa)$", fontsize=label_fontsize, fontweight="bold")
ax2.set_ylabel(r"Conditional variance $\langle y^2 \rangle_{\mathrm{cond}}$",
               fontsize=14, fontweight="bold")

ax.set_title("Diagnostics: Survival and conditional variance",fontsize=12)

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="best")

plt.tight_layout()

# ----------------------------
# SAVEFIG (added as requested)
# ----------------------------
plt.savefig("case2_vary_kappa_fixedL.png", dpi=600, bbox_inches="tight")
plt.savefig("case2_vary_kappa_fixedL.pdf", dpi=600, bbox_inches="tight")

plt.show()
