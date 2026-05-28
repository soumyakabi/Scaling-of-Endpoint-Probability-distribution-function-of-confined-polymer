import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib as mpl
import matplotlib.gridspec as gridspec

mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'figure.figsize': (15.5, 10)
})
palette = plt.get_cmap('tab10')
label_fontsize = 18
tick_fontsize = 12

def P_image(y, sigma, L, m_max=200):
    P = np.zeros_like(y, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for m in range(-m_max, m_max + 1):
        P += (-1) ** m * np.exp(-(y - 2.0 * m * L) ** 2 / (2.0 * sigma ** 2))
    P = prefactor * P
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

def P_tilde_image(s, lam, m_max=200):
    sigma = 1.0
    y = s * sigma
    L = float(lam)
    return P_image(y, sigma, L, m_max=m_max)

kappa_list = [0.01, 0.1, 0.5, 2.0, 8.0]
lambda_list = [1.0 / np.sqrt(k) for k in kappa_list]
colors = [palette(i) for i in range(len(kappa_list))]
max_lambda = max(lambda_list)

m_max = 300
density_per_unit = 1200
Npoints = int(max(20001, density_per_unit * int(np.ceil(2.0 * max_lambda))))
s_common = np.linspace(-1.05 * max_lambda, 1.05 * max_lambda, Npoints)

P_uncond_list, P_cond_list, integrals = [], [], []
for lam in lambda_list:
    P_on_common = P_tilde_image(s_common, lam, m_max=m_max)
    P_on_common[P_on_common < 0.0] = 0.0
    integral = np.trapz(P_on_common, s_common)
    P_cond = P_on_common / integral if integral > 0.0 else np.zeros_like(P_on_common)
    P_uncond_list.append(P_on_common)
    P_cond_list.append(P_cond)
    integrals.append(integral)

s_th = np.linspace(-max_lambda * 1.05, max_lambda * 1.05, 3001)
gauss_th = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * s_th ** 2)

fig = plt.figure(figsize=(15.5, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 0.5], width_ratios=[1.35, 0.75], hspace=0.28, wspace=0.18)

ax_uncond = fig.add_subplot(gs[0, 0])
ax_cond = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, :])

for P_on_common, lam, kappa, col in zip(P_uncond_list, lambda_list, kappa_list, colors):
    ax_uncond.plot(s_common, P_on_common, color=col, lw=1.6, label=f'κ={kappa} (λ={lam:.3g})', alpha=0.95)
ax_uncond.plot(s_th, gauss_th, 'k--', lw=1.2, label='Free Gaussian (theory)')
ax_uncond.set_xlim(-max_lambda * 1.02, max_lambda * 1.02)
ax_uncond.set_xlabel(r'$s = y/\sigma$', fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_ylabel(r'$\widetilde{P}(s)=\sigma P(y)$ (unconditional)', fontsize=label_fontsize, fontweight='bold')
ax_uncond.set_title('(a) Unconditional scaled PDFs (area = survival probability)', fontweight='bold')
ax_uncond.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_uncond.get_xticklabels() + ax_uncond.get_yticklabels(): label.set_fontweight('bold')
ax_uncond.grid(alpha=0.25)
ax_uncond.legend(loc='upper right', bbox_to_anchor=(0.995, 0.995), fontsize=11.0, framealpha=0.9,
                 borderaxespad=0.25, handlelength=1.6, labelspacing=0.25, borderpad=0.25, columnspacing=0.6)

zoom_range = 0.6 * (2.0 if max_lambda > 2.0 else max_lambda)
zoom_range = max(0.1, float(zoom_range))
axins_u = inset_axes(ax_uncond, width='34%', height='34%', loc='lower left',
                     bbox_to_anchor=(0.05, 0.06, 0.5, 0.5), bbox_transform=ax_uncond.transAxes)
axins_u.set_xlim(-zoom_range, zoom_range)
center_mask = np.abs(s_common) <= zoom_range
if not np.any(center_mask):
    center_mask = np.abs(s_common) <= (0.02 * max_lambda + 1e-6)
max_u = max([P_uncond_list[i][center_mask].max() for i in range(len(P_uncond_list))])
axins_u.set_ylim(0, 1.05 * max_u)
for P_on_common, col in zip(P_uncond_list, colors):
    axins_u.plot(s_common, P_on_common, color=col, lw=1.2, alpha=0.95)
axins_u.plot(s_th, gauss_th, 'k--', lw=1.0)
axins_u.grid(alpha=0.2)
mark_inset(ax_uncond, axins_u, loc1=2, loc2=4, fc='none', ec='0.5')

for P_cond, lam, kappa, col in zip(P_cond_list, lambda_list, kappa_list, colors):
    ax_cond.plot(s_common, P_cond, color=col, lw=1.6, label=f'κ={kappa} (λ={lam:.3g})', alpha=0.95)
ax_cond.set_xlim(-max_lambda * 1.02, max_lambda * 1.02)
ax_cond.set_xlabel(r'$s = y/\sigma$', fontsize=label_fontsize, fontweight='bold')
ax_cond.set_ylabel(r'Conditional $\widetilde{P}(s)$ (normalized)', fontsize=label_fontsize, fontweight='bold')
ax_cond.set_title('(b) Conditional scaled PDFs (normalized)', fontweight='bold')
ax_cond.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_cond.get_xticklabels() + ax_cond.get_yticklabels(): label.set_fontweight('bold')
ax_cond.grid(alpha=0.25)
ax_cond.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=11.0, framealpha=0.9,
               borderaxespad=0.2, handlelength=1.6, labelspacing=0.25, borderpad=0.25, columnspacing=0.6)

axins_c = inset_axes(ax_cond, width='34%', height='34%', loc='lower left',
                     bbox_to_anchor=(0.05, 0.06, 0.5, 0.5), bbox_transform=ax_cond.transAxes)
axins_c.set_xlim(-zoom_range, zoom_range)
if not np.any(center_mask):
    center_mask = np.abs(s_common) <= (0.02 * max_lambda + 1e-6)
max_c = max([P_cond_list[i][center_mask].max() for i in range(len(P_cond_list))])
axins_c.set_ylim(0, 1.05 * max_c)
for P_cond, col in zip(P_cond_list, colors):
    axins_c.plot(s_common, P_cond, color=col, lw=1.2, alpha=0.95)
axins_c.grid(alpha=0.2)
mark_inset(ax_cond, axins_c, loc1=2, loc2=4, fc='none', ec='0.5')

S_vals = integrals
cond_var = [np.trapz((s_common**2) * P_cond, s_common) for P_cond in P_cond_list]
ax2 = ax_bottom.twinx()
ax_bottom.plot(kappa_list, S_vals, 'o-b', lw=1.8, ms=7, label=r'Survival $S(\kappa)$')
ax2.plot(kappa_list, cond_var, 's--', color='orange', lw=1.8, ms=7, label=r'Cond. Var $\langle s^2\rangle_{cond}$')
ax_bottom.set_xscale('log')
ax_bottom.set_yscale('log')
ax_bottom.set_xlabel(r'$\kappa$ (confinement parameter)', fontsize=label_fontsize, fontweight='bold')
ax_bottom.set_ylabel(r'Survival prob. / Cond. variance', fontsize=14, fontweight='bold')
ax_bottom.set_title(r'(c) Survival probability and conditional variance vs $\kappa$', fontweight='bold')
ax_bottom.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for label in ax_bottom.get_xticklabels() + ax_bottom.get_yticklabels() + ax2.get_yticklabels(): label.set_fontweight('bold')
ax_bottom.grid(True, which='both', ls='--', alpha=0.5)
lines, labels = ax_bottom.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='best', framealpha=0.9)
ax_bottom.text(0.02, 0.02, '\n'.join([f'κ={kappa:.3g}, λ={lambda_list[i]:.3f}, S={S_vals[i]:.3g}, Var={cond_var[i]:.3g}' for i, kappa in enumerate(kappa_list)]),
               transform=ax_bottom.transAxes, fontsize=12, va='bottom', ha='left', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.savefig('Case3_ImageMethod.png', dpi=600, bbox_inches='tight')
plt.savefig('Case3_ImageMethod.pdf', dpi=600, bbox_inches='tight')
print('Saved: Case3_ImageMethod.pdf and Case3_ImageMethod.png')
plt.show()