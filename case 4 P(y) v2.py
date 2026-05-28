# Supplementary Case S-4: Image-method distributions with varying chain length N
# at fixed slit width L — universality check for Case 2.
# Panels:
#   (S4a) Unconditional scaled distributions
#   (S4b) Conditional (normalized survivors) with inset zoom
#   (S4c) Diagnostics: Survival and conditional variance (log-log)
#   (S4d) Peak-normalized shape comparison

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

label_fontsize = 18
tick_fontsize  = 14

# ── image method (vectorised) ─────────────────────────────────────────────────
def Py_image(y, N, a, L, m_max=300):
    sigma     = np.sqrt(N) * a
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    m         = np.arange(-m_max, m_max + 1)
    P         = np.sum(
        ((-1.0) ** m) * np.exp(-(y[:, None] - 2.0 * m * L) ** 2 / (2.0 * sigma ** 2)),
        axis=1
    )
    P *= prefactor
    P[np.abs(y) > L] = 0.0
    P[P < 0.0]       = 0.0
    return P

# ── parameters ────────────────────────────────────────────────────────────────
a            = 1.0
Lfixed       = 4.0
kappa_values = [0.05, 0.1, 0.5, 1.0, 2.0]
ypoints      = 2001
colors       = plt.cm.tab10.colors
linestyles   = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
markers      = ['o', 's', '^', 'D', 'v']

# ── compute curves ────────────────────────────────────────────────────────────
curves = []
for kappa in kappa_values:
    N    = kappa * Lfixed ** 2 / a ** 2
    y    = np.linspace(-Lfixed, Lfixed, ypoints)
    P    = Py_image(y, N, a, Lfixed)
    surv = np.trapz(P, y)
    u    = y / Lfixed
    mathcalP = Lfixed * P
    mathcalPcond = mathcalP / surv if surv > 0 else np.zeros_like(mathcalP)
    condvar  = np.trapz(u ** 2 * mathcalPcond, u)
    peak_val = np.max(mathcalPcond) if np.max(mathcalPcond) > 0 else 1.0
    mathcalPpeak = mathcalPcond / peak_val
    curves.append(dict(
        kappa=kappa, N=N, y=y, u=u, P=P,
        mathcalP=mathcalP, surv=surv,
        mathcalPcond=mathcalPcond,
        mathcalPpeak=mathcalPpeak,
        condvar=condvar
    ))

# ── figure ────────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    r"Fig. S-4: Image-method distributions with varying chain length $N$"
    "\n"
    r"at fixed slit width $L$ (universality check for Case 2)",
    fontsize=12, fontweight='bold', y=1.02
)

ax1   = axs[0, 0]   # (S4a) unconditional
ax2   = axs[0, 1]   # (S4b) conditional
axc   = axs[1, 0]   # (S4c) diagnostics
ax3   = axs[1, 1]   # (S4d) peak-normalized

# ── plot panels a, b, d ───────────────────────────────────────────────────────
for i, c in enumerate(curves):
    label = fr"$\kappa={c['kappa']},\ N={c['N']:.3g}$"
    style = dict(color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)], lw=1.8)
    ax1.plot(c['u'], c['mathcalP'],     label=label, **style)
    ax1.plot(c['u'][200], c['mathcalP'][200],
             marker=markers[i], color=colors[i], linestyle='None', ms=6)
    ax2.plot(c['u'], c['mathcalPcond'], label=label, **style)
    ax2.plot(c['u'][200], c['mathcalPcond'][200],
             marker=markers[i], color=colors[i], linestyle='None', ms=6)
    ax3.plot(c['u'], c['mathcalPpeak'], label=label, **style)
    ax3.plot(c['u'][200], c['mathcalPpeak'][200],
             marker=markers[i], color=colors[i], linestyle='None', ms=6)

# ── inset zoom for panel (b) ──────────────────────────────────────────────────
axins = inset_axes(ax2, width="40%", height="40%", loc='upper right', borderpad=1)
for i, c in enumerate(curves):
    axins.plot(c['u'], c['mathcalPcond'],
               color=colors[i], ls=linestyles[i], lw=1.2)
axins.set_xlim(-0.35, 0.35)
axins.set_ylim(0, max(c['mathcalPcond'].max() for c in curves) * 1.05)
axins.grid(alpha=0.2)
axins.tick_params(labelsize=8)
for spine in axins.spines.values():
    spine.set_linewidth(0.8)
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

# ── panel (c) diagnostics ─────────────────────────────────────────────────────
kappa_arr   = np.array([c['kappa']   for c in curves])
surv_arr    = np.array([c['surv']    for c in curves])
condvar_arr = np.array([c['condvar'] for c in curves])

axc.set_xscale('log')
axc.set_yscale('log')
axc.plot(kappa_arr, surv_arr,    'o-',  color='royalblue',  lw=1.8, ms=7,
         label=r"Survival $S(\kappa)$")
axc.plot(kappa_arr, condvar_arr, 's--', color='darkorange', lw=1.8, ms=7,
         label=r"Cond. var. $\langle u^2\rangle_{\mathrm{cond}}$")
axc.legend(fontsize=10, loc='best', framealpha=0.9)

# ── axis decoration ───────────────────────────────────────────────────────────
def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight='bold')
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=1.2)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(alpha=0.25)

style_axes(ax1,
           r"$u = y/L$",
           r"$\mathcal{P}(u) = L P(y)$",
           r"(S4a) Unconditional $\mathcal{P}(u)$  [area $= S$]")
style_axes(ax2,
           r"$u = y/L$",
           r"$\mathcal{P}_{\mathrm{cond}}(u)$",
           r"(S4b) Conditional (normalized survivors)")
style_axes(axc,
           r"$\kappa$  (confinement parameter)",
           r"$S(\kappa)$ / Cond. variance",
           r"(S4c) Diagnostics: survival and conditional variance")
style_axes(ax3,
           r"$u = y/L$",
           r"$\mathcal{P}_{\mathrm{cond}}(u)/\mathcal{P}_{\mathrm{cond}}^{\mathrm{peak}}$",
           r"(S4d) Peak-normalized shape comparison")

for ax in (ax1, ax2, ax3):
    ax.legend(fontsize=9, loc='best', frameon=True)

plt.tight_layout()
plt.savefig("FigS_Case4_vary_kappa_fixedN.png", dpi=600, bbox_inches="tight")
plt.savefig("FigS_Case4_vary_kappa_fixedN.pdf", dpi=600, bbox_inches="tight")

# ── print diagnostics (Supplementary Table S3) ───────────────────────────────
print(f"{'kappa':>8}  {'N':>10}  {'survival':>14}  {'condvar':>14}")
for c in curves:
    print(f"{c['kappa']:8g}  {c['N']:10.4g}  {c['surv']:14.6g}  {c['condvar']:14.6g}")
print("Saved: FigS_Case4_vary_kappa_fixedN.png / .pdf")