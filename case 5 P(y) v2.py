import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = 1.0
kappa = 0.1
sigma = np.sqrt(kappa) * R
eta_list = [0.0, 0.25, 0.50, 0.70]
ny = 2001
y = np.linspace(-R, R, ny)
u = y / R
colors = plt.cm.tab10.colors
linestyles = ['-', '--', '-.', ':']

def P_image_shifted(y, y0, sigma, R, m_max=400):
    P = np.zeros_like(y, dtype=float)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for m in range(-m_max, m_max + 1):
        P += (-1) ** m * np.exp(-(y - y0 - 2 * m * R) ** 2 / (2.0 * sigma ** 2))
    P *= prefactor
    P[np.abs(y) > R] = 0.0
    P[P < 0.0] = 0.0
    return P

def adaptive_M(sigma, R, y0, eps=1e-12):
    for m in range(1, 600):
        dist = abs(y0 + 2 * m * R) - 4 * sigma
        if dist > 0 and np.exp(-dist ** 2 / (2 * sigma ** 2)) < eps:
            return m
    return 400

def unit_gaussian_shifted(s, s0):
    return np.exp(-0.5 * (s - s0) ** 2) / np.sqrt(2 * np.pi)

results = {}
for eta in eta_list:
    y0 = eta * R
    M = adaptive_M(sigma, R, y0)
    P = P_image_shifted(y, y0, sigma, R, m_max=M)
    S = np.trapz(P, y)
    P_cond = P / S if S > 0 else np.zeros_like(P)
    mean_cond = np.trapz(u * P_cond, u)
    var_cond = np.trapz(u ** 2 * P_cond, u) - mean_cond ** 2
    peak_val = np.max(P_cond) if np.max(P_cond) > 0 else 1.0
    P_peak = P_cond / peak_val
    s = (y - y0) / sigma
    P_coil = sigma * P_cond
    results[eta] = dict(
        y0=y0, sigma=sigma, M_used=M,
        P=P, P_cond=P_cond, P_peak=P_peak,
        P_coil=P_coil, s=s,
        S=S, mean_cond=mean_cond, var_cond=var_cond
    )

fig = plt.figure(figsize=(13, 15))
gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.32)

ax_uncond = fig.add_subplot(gs[0, 0])
ax_cond = fig.add_subplot(gs[0, 1])
ax_coil = fig.add_subplot(gs[1, 0])
ax_peak = fig.add_subplot(gs[1, 1])
ax_diag = fig.add_subplot(gs[2, :])

for i, eta in enumerate(eta_list):
    r = results[eta]
    lbl = fr'$\eta={eta:.2f}$'
    ls = linestyles[i % 4]
    col = colors[i % 10]

    ax_uncond.plot(u, R * r['P'], label=lbl, color=col, ls=ls, lw=1.7)
    ax_cond.plot(u, R * r['P_cond'], label=lbl, color=col, ls=ls, lw=1.7)
    ax_coil.plot(r['s'], r['P_coil'], label=lbl, color=col, ls=ls, lw=1.7)
    ax_peak.plot(u, r['P_peak'], label=lbl, color=col, ls=ls, lw=1.7)

s_ref = np.linspace(-5, 5, 800)
ax_coil.plot(s_ref, unit_gaussian_shifted(s_ref, 0.0), 'k:', lw=1.4, label=r'$G(s)$ (ref.)')
ax_coil.set_xlim(-5, 5)

eta_arr = np.array(eta_list)
S_arr = np.array([results[e]['S'] for e in eta_list])
var_arr = np.array([results[e]['var_cond'] for e in eta_list])
mean_arr = np.array([results[e]['mean_cond'] for e in eta_list])

ax_diag2 = ax_diag.twinx()
ax_diag.plot(eta_arr, S_arr, 'o-b', lw=1.8, ms=7, label=r'Survival $S(\eta)$')
ax_diag2.plot(eta_arr, var_arr, 's--', color='orange', lw=1.8, ms=7,
              label=r'Cond. variance $\langle u^2\rangle_\mathrm{cond}$')
ax_diag2.plot(eta_arr, mean_arr, 'D:', color='green', lw=1.8, ms=7,
              label=r'Cond. mean $\langle u\rangle_\mathrm{cond}$')

for ax in (ax_uncond, ax_cond, ax_coil, ax_peak):
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

ax_uncond.set_xlabel(r'$u = y/R$', fontsize=18, fontweight='bold')
ax_uncond.set_ylabel(r'$\mathcal{P}(u) = R\,P(y)$', fontsize=18, fontweight='bold')
ax_uncond.set_title(r'(a) Unconditional scaled distributions (varying $\eta = y_0/R$)', fontsize=12)

ax_cond.set_xlabel(r'$u = y/R$', fontsize=18, fontweight='bold')
ax_cond.set_ylabel(r'$\mathcal{P}_\mathrm{cond}(u) = R\,P_\mathrm{cond}(y)$', fontsize=18, fontweight='bold')
ax_cond.set_title(r'(b) Conditional scaled distributions', fontsize=12)

ax_coil.set_xlabel(r'$s = (y-y_0)/\sigma$', fontsize=18, fontweight='bold')
ax_coil.set_ylabel(r'$\widehat{P}(s) = \sigma\,P_\mathrm{cond}(y)$', fontsize=18, fontweight='bold')
ax_coil.set_title(r'(c) Coil-scaled conditional densities', fontsize=12)

ax_peak.set_xlabel(r'$u = y/R$', fontsize=18, fontweight='bold')
ax_peak.set_ylabel(r'$\mathcal{P}_\mathrm{cond}(u)/\mathcal{P}_\mathrm{cond}^{\mathrm{peak}}$', fontsize=18, fontweight='bold')
ax_peak.set_title(r'(d) Peak-normalized conditional distributions', fontsize=12)

for ax in (ax_uncond, ax_cond, ax_coil, ax_peak):
    ax.legend(fontsize=10, loc='best', frameon=True)

ax_diag.set_xlabel(r'$\eta = y_0/R$  (dimensionless tether offset)', fontsize=18, fontweight='bold')
ax_diag.set_ylabel(r'$S(\eta)$', fontsize=18, fontweight='bold')
ax_diag2.set_ylabel(r'Cond. variance / mean', fontsize=14, fontweight='bold')
ax_diag.set_title(r'(e) Diagnostics: survival, conditional variance, and conditional mean vs $\eta$', fontsize=12)
ax_diag.grid(alpha=0.3)
ax_diag.tick_params(axis='both', which='major', labelsize=13)

lines1, lab1 = ax_diag.get_legend_handles_labels()
lines2, lab2 = ax_diag2.get_legend_handles_labels()
ax_diag2.legend(lines1 + lines2, lab1 + lab2, loc='best', fontsize=11)

fig.suptitle(
    r'Case 5: Off-centre tether — broken symmetry scaling'
    '\n'
    r'($\kappa = 0.1$, $R/\sigma = %.2f$, absorbing walls at $y=\pm R$)' % (R / sigma),
    fontsize=13, fontweight='bold', y=1.005
)

plt.savefig('case5_offcentre_tether.png', dpi=600, bbox_inches='tight')
plt.savefig('case5_offcentre_tether.pdf', dpi=600, bbox_inches='tight')
plt.show()

print(f"{'eta':>6} {'y0':>6} {'M_used':>8} {'S':>12} {'mean_cond':>14} {'var_cond':>12}")
for eta in eta_list:
    r = results[eta]
    print(f"{eta:6.2f} {r['y0']:6.3f} {r['M_used']:8d} {r['S']:12.6f} "
          f"{r['mean_cond']:14.6f} {r['var_cond']:12.6f}")