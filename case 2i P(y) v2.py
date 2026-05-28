import numpy as np
import matplotlib.pyplot as plt

def P_image(y, sigma, L, m_max=200):
    P = np.zeros_like(y, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for m in range(-m_max, m_max + 1):
        P += (-1)**m * np.exp(-(y - 2*m*L)**2 / (2.0 * sigma**2))
    P *= prefactor
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

L = 1.0
kappa_list = [0.02, 0.05, 0.1, 0.5, 1.0, 2.0]
colors = plt.cm.tab10.colors
ny = 2001
y = np.linspace(-L, L, ny)
label_fontsize = 18

results = {}
for kappa in kappa_list:
    sigma = np.sqrt(kappa) * L
    P = P_image(y, sigma, L, m_max=400)
    S = np.trapz(P, y)
    P_cond = P / S if S > 0 else P * 0
    var_cond = np.trapz((y**2) * P_cond, y)
    results[kappa] = dict(sigma=sigma, P=P, P_cond=P_cond, S=S, var_cond=var_cond, L_over_sigma=L/sigma)

fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.32)

ax_uncond = fig.add_subplot(gs[0, 0])
ax_cond   = fig.add_subplot(gs[0, 1])
ax_coil   = fig.add_subplot(gs[1, 0])
ax_diag   = fig.add_subplot(gs[1, 1])
ax_peak   = fig.add_subplot(gs[2, :])

for i, kappa in enumerate(kappa_list):
    u = y/L
    ax_uncond.plot(u, L*results[kappa]['P'], label=fr'$\kappa={kappa}$', color=colors[i%10])
ax_uncond.set_xlabel(r'$u = y/L$', fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_ylabel(r'$\mathcal{P}(u) = L P(y)$', fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_title(r'(a) Unconditional scaled distributions (varying $\kappa$)', fontsize=12)
ax_uncond.legend(); ax_uncond.grid(alpha=0.3)

for i, kappa in enumerate(kappa_list):
    u = y/L
    ax_cond.plot(u, L*results[kappa]['P_cond'], label=fr'$\kappa={kappa}$', color=colors[i%10])
ax_cond.set_xlabel(r'$u = y/L$', fontsize=label_fontsize, fontweight='bold')
ax_cond.set_ylabel(r'$\mathcal{P}_{\mathrm{cond}}(u) = L P_{\mathrm{cond}}(y)$', fontsize=label_fontsize, fontweight='bold')
ax_cond.set_title(r'(b) Conditional (survivors) scaled distributions', fontsize=12)
ax_cond.legend(); ax_cond.grid(alpha=0.3)

for i, kappa in enumerate(kappa_list):
    sigma = results[kappa]['sigma']
    s = y/sigma
    P_cond = results[kappa]['P_cond']
    P_coil = sigma * P_cond
    style = '-' if results[kappa]['L_over_sigma'] >= 3 else '--'
    ax_coil.plot(s, P_coil, label=fr'$\kappa={kappa}$', color=colors[i%10], linestyle=style)
    if results[kappa]['L_over_sigma'] < 3:
        x_pos = 0.5 * results[kappa]['L_over_sigma']
        y_pos = max(P_coil) * 0.7
        ax_coil.text(x_pos, y_pos, fr'$L/\sigma={results[kappa]["L_over_sigma"]:.2f}$', color=colors[i%10], fontsize=9, weight='bold')
ax_coil.set_xlabel(r'$s = y/\sigma$', fontsize=label_fontsize, fontweight='bold')
ax_coil.set_ylabel(r'$\widehat{P}(s) = \sigma P_{\mathrm{cond}}(y)$', fontsize=label_fontsize, fontweight='bold')
ax_coil.set_title(r'(c) Coil-scaled conditional densities (small $\kappa$)', fontsize=12)
ax_coil.legend(); ax_coil.grid(alpha=0.3)

kap = np.array(kappa_list)
Svals = [results[k]['S'] for k in kappa_list]
Vvals = [results[k]['var_cond'] for k in kappa_list]
ax2 = ax_diag.twinx()
ax_diag.plot(kap, Svals, 'o-b', label=r'Survival $S(\kappa)$')
ax2.plot(kap, Vvals, 's--', color='orange', label='Conditional variance')
ax_diag.set_xscale('log')
ax_diag.set_xlabel(r'$\kappa$', fontsize=label_fontsize, fontweight='bold')
ax_diag.set_ylabel(r'$S(\kappa)$', fontsize=label_fontsize, fontweight='bold')
ax2.set_ylabel(r'Conditional variance $\langle y^2 \rangle_{\mathrm{cond}}$', fontsize=14, fontweight='bold')
ax_diag.set_title(r'(d) Diagnostics: Survival and conditional variance', fontsize=12)
lines, labels = ax_diag.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left', bbox_to_anchor=(0.0, 0.02), fontsize=9, frameon=True)
ax_diag.grid(alpha=0.3)

for i, kappa in enumerate(kappa_list):
    mathcalP_cond = L * results[kappa]['P_cond']
    peak_val = np.max(mathcalP_cond) if np.max(mathcalP_cond) > 0 else 1.0
    ax_peak.plot(u, mathcalP_cond/peak_val, label=fr'$\kappa={kappa}$', color=colors[i%10], lw=1.6)
ax_peak.set_xlabel(r'$u = y/L$', fontsize=label_fontsize, fontweight='bold')
ax_peak.set_ylabel(r'$\mathcal{P}_{\mathrm{cond}}(u)\,/\,\mathcal{P}_{\mathrm{cond}}^{\mathrm{peak}}$', fontsize=label_fontsize, fontweight='bold')
ax_peak.set_title(r'(e) Peak-normalized conditional distributions (pure shape comparison)', fontsize=12)
ax_peak.legend(loc='upper right')
ax_peak.grid(alpha=0.3)

plt.savefig('case2_vary_kappa_fixedL_5panel.png', dpi=600, bbox_inches='tight')
plt.savefig('case2_vary_kappa_fixedL_5panel.pdf', dpi=600, bbox_inches='tight')
print('saved')