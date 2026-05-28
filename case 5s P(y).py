import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

label_fontsize = 18
tick_fontsize  = 13

# ── image method (vectorised) ─────────────────────────────────────────────────
def P_image(y, kappa, L, m_max=400):
    sigma = np.sqrt(kappa) * L
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    m = np.arange(-m_max, m_max + 1)
    P = np.sum(
        ((-1.0) ** m) * np.exp(-(y[:, None] - 2.0 * m * L) ** 2 / (2.0 * sigma ** 2)),
        axis=1
    )
    P *= prefactor
    P[np.abs(y) > L] = 0.0
    P[P < 0.0] = 0.0
    return P

def unit_gaussian(s):
    return np.exp(-s ** 2 / 2.0) / np.sqrt(2.0 * np.pi)

# ── analytical references ─────────────────────────────────────────────────────
# Normalised first Dirichlet eigenmode on [-1,1]:
#   P_inf(u) = (pi/4) * cos(pi*u/2)
# Conditional variance at large kappa (analytical):
#   <u^2>_inf = (pi^2 - 8) / pi^2 ~ 0.18943
def eigenmode(u):
    phi = np.cos(np.pi * u / 2.0)
    return phi / np.trapz(phi, u)

var_plateau = (np.pi ** 2 - 8.0) / np.pi ** 2

# ── parameters ────────────────────────────────────────────────────────────────
kappa_values = [0.05, 0.1, 0.5, 1.0, 2.0]
L = 1.0
y = np.linspace(-L, L, 4001)
u = y / L

colors     = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
markers    = ['o', 's', '^', 'D', 'v']

eigen_ref = eigenmode(u)

# ── compute ───────────────────────────────────────────────────────────────────
results = []
for kappa in kappa_values:
    P     = P_image(y, kappa, L)
    surv  = np.trapz(P, y)
    sigma = np.sqrt(kappa) * L
    P_cond    = P / surv if surv > 0 else np.zeros_like(P)
    geom_cond = L * P_cond
    coil_cond = sigma * P_cond
    var_cond  = np.trapz(u ** 2 * geom_cond, u)
    s      = y / sigma
    mask_s = np.abs(s) <= (L / sigma)
    G_s    = unit_gaussian(s)
    rms_gauss    = (np.sqrt(np.mean((coil_cond[mask_s] - G_s[mask_s]) ** 2))
                    / G_s[0])
    rms_eigenmode = (np.sqrt(np.mean((geom_cond - eigen_ref) ** 2))
                     / np.max(eigen_ref))
    results.append(dict(
        kappa=kappa, sigma=sigma, surv=surv,
        u=u, s=s, P_cond=P_cond,
        geom_cond=geom_cond, coil_cond=coil_cond,
        var_cond=var_cond, rms_gauss=rms_gauss,
        rms_eigenmode=rms_eigenmode,
        L_over_sigma=L / sigma,
    ))

kap_arr  = np.array([r['kappa']         for r in results])
surv_arr = np.array([r['surv']          for r in results])
rms_g    = np.array([r['rms_gauss']     for r in results])
rms_e    = np.array([r['rms_eigenmode'] for r in results])
var_arr  = np.array([r['var_cond']      for r in results])

# ── crossing kappa* ────────────────────────────────────────────────────────────
log_kap  = np.log10(kap_arr)
f_g      = interp1d(log_kap, np.log10(rms_g), kind='linear')
f_e      = interp1d(log_kap, np.log10(rms_e), kind='linear')
kap_fine = np.linspace(log_kap[0], log_kap[-1], 10000)
diff     = f_g(kap_fine) - f_e(kap_fine)
sign_change = np.where(np.diff(np.sign(diff)))[0]

if len(sign_change) > 0:
    kap_cross = 10 ** kap_fine[sign_change[0]]
    rms_cross = 10 ** f_g(kap_fine[sign_change[0]])
else:
    kap_cross = np.nan
    rms_cross = np.nan
    print("Warning: no crossing found in the kappa range provided.")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 14))
gs  = GridSpec(3, 2, figure=fig, hspace=0.44, wspace=0.33)

ax_geom = fig.add_subplot(gs[0, 0])
ax_coil = fig.add_subplot(gs[0, 1])
ax_rms  = fig.add_subplot(gs[1, 0])
ax_eig  = fig.add_subplot(gs[1, 1])
ax_diag = fig.add_subplot(gs[2, :])

def style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')
    ax.grid(alpha=0.25)
    if title:
        ax.set_title(title, fontsize=11)

s_dense = np.linspace(-5, 5, 4000)
G_ref   = unit_gaussian(s_dense)

# (a) Geometry-scaled conditional
for i, res in enumerate(results):
    in_conf = res['L_over_sigma'] <= 1.5
    lw   = 2.2 if in_conf else 1.2
    alph = 1.0 if in_conf else 0.50
    lab  = (fr"$\kappa={res['kappa']}$, $L/\sigma={res['L_over_sigma']:.2f}$"
            + (r" $\checkmark$" if in_conf else ""))
    ax_geom.plot(u, res['geom_cond'], color=colors[i], ls=linestyles[i],
                 lw=lw, alpha=alph, label=lab)
    ax_geom.plot(u[::250], res['geom_cond'][::250],
                 marker=markers[i], color=colors[i], ls='None', alpha=alph, ms=5)
ax_geom.plot(u, eigen_ref, 'k:', lw=2.2, zorder=10,
             label=r'$\frac{\pi}{4}\cos\!\left(\frac{\pi u}{2}\right)$ (eigenmode)')
ax_geom.text(0.03, 0.53,
             r"$\checkmark$ confinement-dominated" + "\n" + r"($L/\sigma \leq 1.5$)",
             transform=ax_geom.transAxes, fontsize=8, color='k',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
style_ax(ax_geom, r"$u = y/L$", r"$L\,P_{\mathrm{cond}}(y)$",
         r"(a) Geometry-scaled $P_\mathrm{cond}$: confinement-regime collapse")
ax_geom.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

# (b) Coil-scaled conditional
ax_coil.plot(s_dense, G_ref, 'k:', lw=2.2, zorder=10, label=r'Unit Gaussian $G(s)$')
for i, res in enumerate(results):
    in_coil = res['L_over_sigma'] >= 3.0
    lw   = 2.2 if in_coil else 1.2
    alph = 1.0 if in_coil else 0.50
    lab  = (fr"$\kappa={res['kappa']}$, $L/\sigma={res['L_over_sigma']:.2f}$"
            + (r" $\checkmark$" if in_coil else ""))
    ax_coil.plot(res['s'], res['coil_cond'], color=colors[i], ls=linestyles[i],
                 lw=lw, alpha=alph, label=lab)
    ax_coil.plot(res['s'][::250], res['coil_cond'][::250],
                 marker=markers[i], color=colors[i], ls='None', alpha=alph, ms=5)
ax_coil.set_xlim(-5, 5)
ax_coil.text(0.03, 0.53,
             r"$\checkmark$ free-coil regime" + "\n" + r"($L/\sigma \geq 3$)",
             transform=ax_coil.transAxes, fontsize=8, color='k',
             bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
style_ax(ax_coil, r"$s = y/\sigma$", r"$\sigma\,P_{\mathrm{cond}}(y)$",
         r"(b) Coil-scaled $P_\mathrm{cond}$: free-coil regime collapse")
ax_coil.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

# (c) Dual RMS crossover
ax_rms.loglog(kap_arr, rms_g, 'o-',  color='royalblue', lw=2.2, ms=8,
              label=r'$\delta_G(\kappa)$: RMS from Gaussian')
ax_rms.loglog(kap_arr, rms_e, 's--', color='crimson',   lw=2.2, ms=8,
              label=r'$\delta_E(\kappa)$: RMS from eigenmode')
if not np.isnan(kap_cross):
    ax_rms.axvline(kap_cross, color='gray', lw=1.4, ls='--', alpha=0.8)
    ax_rms.plot(kap_cross, rms_cross, '*', color='darkorange', ms=14, zorder=10,
                label=fr'Crossing $\kappa^*\approx{kap_cross:.2f}$')
    ax_rms.text(kap_cross * 1.15, rms_cross * 1.2,
                fr'$\kappa^*\approx{kap_cross:.2f}$' + '\n'
                + fr'$L/\sigma\approx{1/np.sqrt(kap_cross):.2f}$',
                fontsize=9, color='gray')
ax_rms.text(0.05, 0.25, 'free-coil\nregime',
            transform=ax_rms.transAxes, fontsize=8.5, color='royalblue',
            ha='left', style='italic')
ax_rms.text(0.72, 0.25, 'confinement\nregime',
            transform=ax_rms.transAxes, fontsize=8.5, color='crimson',
            ha='left', style='italic')
style_ax(ax_rms, r"$\kappa = \sigma^2/L^2$", r"Normalised RMS deviation",
         r"(c) Dual RMS crossover: quantitative definition of $\kappa^*$")
ax_rms.legend(fontsize=8.5, loc='center', framealpha=0.9)

# (d) Eigenmode convergence
ax_eig.plot(u, eigen_ref, 'k-', lw=3.0, zorder=10,
            label=r'$\frac{\pi}{4}\cos\!\left(\frac{\pi u}{2}\right)$  (theory, $\kappa\to\infty$)')
for i, res in enumerate(results):
    lw   = 2.2 if res['L_over_sigma'] <= 1.5 else 1.2
    alph = 1.0 if res['L_over_sigma'] <= 1.5 else 0.45
    ax_eig.plot(u, res['geom_cond'], color=colors[i], ls=linestyles[i],
                lw=lw, alpha=alph,
                label=fr"$\kappa={res['kappa']}$, $\delta_E={res['rms_eigenmode']:.4f}$")
style_ax(ax_eig, r"$u = y/L$", r"$L\,P_{\mathrm{cond}}(y)$",
         r"(d) Convergence of $P_\mathrm{cond}$ to first Dirichlet eigenmode")
ax_eig.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

# (e) Diagnostics
ax2 = ax_diag.twinx()
ax_diag.loglog(kap_arr, surv_arr, 'o-b', lw=2.2, ms=8,
               label=r'Survival $S(\kappa)$')
ax2.semilogx(kap_arr, var_arr, 's--', color='darkorange', lw=2.2, ms=8,
             label=r'Cond. variance $\langle u^2\rangle_\mathrm{cond}$')
ax2.axhline(var_plateau, color='darkorange', lw=1.4, ls=':', alpha=0.85)
ax2.text(kap_arr[-1] * 1.02, var_plateau * 1.015,
         r'$\frac{\pi^2-8}{\pi^2}\approx0.1894$',
         fontsize=9, color='darkorange', va='bottom')
if not np.isnan(kap_cross):
    ax_diag.axvline(kap_cross, color='gray', lw=1.4, ls='--', alpha=0.8)
    ax_diag.text(kap_cross * 1.12, surv_arr.min() * 2,
                 fr'$\kappa^*\approx{kap_cross:.2f}$', fontsize=9, color='gray')
ax_diag.set_xlabel(r'$\kappa = \sigma^2/L^2$',
                   fontsize=label_fontsize, fontweight='bold')
ax_diag.set_ylabel(r'$S(\kappa)$', fontsize=label_fontsize, fontweight='bold')
ax2.set_ylabel(r'$\langle u^2\rangle_\mathrm{cond}$',
               fontsize=label_fontsize, fontweight='bold')
ax_diag.set_title(
    r"(e) Crossover diagnostics: survival $S(\kappa)$ and conditional variance"
    r" with theoretical plateau $(\pi^2-8)/\pi^2$", fontsize=11)
ax_diag.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
for lab in (ax_diag.get_xticklabels() + ax_diag.get_yticklabels()
            + ax2.get_yticklabels()):
    lab.set_fontweight('bold')
ax_diag.grid(alpha=0.25)

lines1, labs1 = ax_diag.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
plateau_handle = Line2D([0], [0], color='darkorange', lw=1.4, ls=':',
                        label=r'Plateau $(\pi^2-8)/\pi^2$')
ax2.legend(lines1 + lines2 + [plateau_handle],
           labs1 + labs2 + [r'Plateau $(\pi^2-8)/\pi^2$'],
           fontsize=9.5, loc='center left', framealpha=0.9)

fig.suptitle(
    r'Case 5 (Supplementary): Two-regime scaling and eigenmode convergence of $P_\mathrm{cond}$',
    fontsize=13, fontweight='bold', y=1.002
)

plt.savefig('Case5_S_supplementary.png', dpi=600, bbox_inches='tight')
plt.savefig('Case5_S_supplementary.pdf', dpi=600, bbox_inches='tight')
print("Saved: Case5_S_supplementary.png / .pdf")