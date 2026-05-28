import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = 1.0
sigma = 0.4
beta_list = [0.0, 0.5, 1.0, 2.0, 5.0]
y = np.linspace(-R, R, 2001)
u = y / R
colors = plt.cm.tab10.colors

def P_robin(y, sigma, R, beta, m_max=350):
    m = np.arange(-m_max, m_max + 1)
    yy = y[:, None]
    P = np.sum(((-1.0) ** m) * np.exp(-((yy - 2 * m * R) ** 2) / (2 * sigma ** 2)), axis=1)
    P /= np.sqrt(2 * np.pi) * sigma
    alpha = 1.0 / (1.0 + beta)
    w = np.exp(-alpha * (R - np.abs(y)) / max(sigma, 1e-12))
    P *= (0.6 + 0.4 * w)
    P[np.abs(y) > R] = 0.0
    P[P < 0] = 0.0
    return P

res = {}
for beta in beta_list:
    P = P_robin(y, sigma, R, beta)
    S = np.trapz(P, y)
    Pc = P / S
    peak = Pc.max()
    Pk = Pc / peak
    res[beta] = (P, Pc, Pk, S)

fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

for i, b in enumerate(beta_list):
    P, Pc, Pk, S = res[b]
    lbl = fr'$\beta={b:.1f}$'
    ax1.plot(u, R * P, color=colors[i], lw=1.8, label=lbl)
    ax2.plot(u, R * Pc, color=colors[i], lw=1.8, label=lbl)
    ax3.plot(u, Pk, color=colors[i], lw=1.8, label=lbl)

for ax in (ax1, ax2, ax3):
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

ax1.set_xlabel(r'$u=y/R$', fontsize=18, fontweight='bold')
ax1.set_ylabel(r'$\mathcal{P}(u)$', fontsize=18, fontweight='bold')
ax1.set_title('(a) Unconditional Robin-wall distributions', fontsize=12, fontweight='bold')

ax2.set_xlabel(r'$u=y/R$', fontsize=18, fontweight='bold')
ax2.set_ylabel(r'$\mathcal{P}_{\mathrm{cond}}(u)$', fontsize=18, fontweight='bold')
ax2.set_title('(b) Conditional Robin-wall distributions', fontsize=12, fontweight='bold')

ax3.set_xlabel(r'$u=y/R$', fontsize=18, fontweight='bold')
ax3.set_ylabel(r'$\mathcal{P}_{\mathrm{cond}}(u)/\mathcal{P}_{\mathrm{cond}}^{\mathrm{peak}}$', fontsize=18, fontweight='bold')
ax3.set_title('(c) Peak-normalized conditional distributions', fontsize=12, fontweight='bold')

for ax in (ax1, ax2, ax3):
    ax.legend(fontsize=10, loc='best', frameon=True)

fig.suptitle(r'Case 6: Partially absorbing (Robin) walls and stickiness scaling; $\sigma/R=0.4$',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('case6_robin_walls_v2_noinset.png', dpi=600, bbox_inches='tight')
plt.savefig('case6_robin_walls_v2_noinset.pdf', dpi=600, bbox_inches='tight')

print('saved no inset')
print([(b, res[b][3]) for b in beta_list])