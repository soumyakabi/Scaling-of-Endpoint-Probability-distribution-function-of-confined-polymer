import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

R = 1.0
D = 0.5
x0_list = [0.05, 0.2, 0.5, 0.8]
t = np.linspace(1e-4, 6, 2500)

nmax = 800
n = np.arange(nmax)
k = (2 * n + 1) * np.pi / (2 * R)
coef = (4 / np.pi) * ((-1) ** n)

def fpt(t, x0):
    c = coef * np.cos(k * x0)
    return np.sum(c[:, None] * np.exp(-D * (k[:, None] ** 2) * t[None, :]), axis=0)

fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.22)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(x0_list)))
metrics = []

for c, x0 in zip(cols, x0_list):
    p = fpt(t, x0)
    p[p < 0] = 0
    area = np.trapz(p, t)
    p /= area
    mt = np.trapz(t * p, t)
    vt = np.trapz((t - mt) ** 2 * p, t)
    metrics.append((x0, area, mt, vt))
    ax1.plot(t, p, color=c, lw=2.2, label=fr'$x_0/R={x0:.2f}$')
    ax2.plot(t / mt, mt * p, color=c, lw=2.2, label=fr'$x_0/R={x0:.2f}$')
    ax3.plot(t, np.cumsum(p) * (t[1] - t[0]), color=c, lw=2.2, label=fr'$x_0/R={x0:.2f}$')

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

ax1.set_xlabel(r'$\tau$', fontsize=18, fontweight='bold')
ax1.set_ylabel(r'$f(\tau)$', fontsize=18, fontweight='bold')
ax1.set_title('(a) First-passage-time density', fontsize=14, fontweight='bold')

ax2.set_xlabel(r'$\tau/\langle\tau\rangle$', fontsize=18, fontweight='bold')
ax2.set_ylabel(r'$\langle\tau\rangle f(\tau)$', fontsize=18, fontweight='bold')
ax2.set_title('(b) Mean-scaled collapse', fontsize=14, fontweight='bold')

ax3.set_xlabel(r'$\tau$', fontsize=18, fontweight='bold')
ax3.set_ylabel(r'$F(\tau)$', fontsize=18, fontweight='bold')
ax3.set_title('(c) First-passage cumulative distribution', fontsize=14, fontweight='bold')

ax1.legend(fontsize=10, frameon=True, loc='upper right')
ax2.legend(fontsize=10, frameon=True, loc='upper right')
ax3.legend(fontsize=10, frameon=True, loc='lower right')

ax1.set_xlim(0, 6)
ax2.set_xlim(0, 4)
ax3.set_xlim(0, 6)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(0, 1.02)

axins = inset_axes(ax1, width="42%", height="42%", loc='upper left', borderpad=1)
for c, x0 in zip(cols, x0_list):
    p = fpt(t, x0)
    p[p < 0] = 0
    p /= np.trapz(p, t)
    axins.plot(t, p, color=c, lw=1.4)

axins.set_xlim(0, 0.8)
ref = fpt(t, 0.8)
ref[ref < 0] = 0
ref /= np.trapz(ref, t)
axins.set_ylim(0, ref.max() * 1.1)
axins.grid(alpha=0.2)
axins.tick_params(labelsize=8)
for spine in axins.spines.values():
    spine.set_linewidth(0.9)

fig.suptitle('Case 7: First-passage dynamics from the confined propagator',
             fontsize=16, fontweight='bold', y=0.99)

plt.savefig('case7_fpt_refined.png', dpi=600, bbox_inches='tight')
plt.savefig('case7_fpt_refined.pdf', dpi=600, bbox_inches='tight')

print(metrics)
print('saved')