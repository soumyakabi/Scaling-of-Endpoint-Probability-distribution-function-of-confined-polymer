# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 19:47:20 2025

@author: SOUMYA
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analytic_P(x, N, L, a, x0, n_max=12000, decay_tol=1e-15):
    x = np.asarray(x, dtype=float)
    kappa = (N * a**2) / (L**2)

    n = np.arange(1, n_max + 1)
    lam = n * np.pi / L
    sin0 = np.sin(lam * x0)

    decay = np.exp(- (n**2) * (np.pi**2) * kappa / 8.0)

    idx_keep = np.where(decay > decay_tol)[0]
    last_idx = idx_keep[-1] if idx_keep.size else 0

    n = n[: last_idx + 1]
    lam = lam[: last_idx + 1]
    sin0 = sin0[: last_idx + 1]
    decay = decay[: last_idx + 1]

    num = np.sum(sin0[:, None] * np.sin(np.outer(lam, x)) * decay[:, None], axis=0)
    cos_term = 1 - (-1) ** n
    den = np.sum((sin0 / (n * np.pi)) * cos_term * decay)

    if np.abs(den) < 1e-16:
        P_raw = num / L
        area = np.trapz(P_raw, x)
        if area <= 0:
            raise RuntimeError("Normalization failed.")
        P = P_raw / area
    else:
        P = num / (L * den)

    P[P < 0] = 0.0
    return P

if __name__ == "__main__":
    # user options
    USE_HIGH_RES = True
    PLOT_NATIVE_OVERLAY = True   # faint native curves as context
    EPSILON = 1e-4               # threshold fraction of peak to define 'significant' support
    SAVE_FIG = True              # Set to True to save figures
    FIGURE_NAME = "persistence_length_sweep"  # Base name for saved figures

    # physical params
    N = 10.0
    L = 2.0
    x0 = 0.5 * L
    a_values = [0.05, 0.1, 0.2, 0.5, 1.0]

    # numerical resolution
    if USE_HIGH_RES:
        n_x = 8000
        n_max = 20000
        decay_tol = 1e-16
        eps = 1e-10
    else:
        n_x = 3000
        n_max = 8000
        decay_tol = 1e-14
        eps = 1e-8

    # dense x grid (avoid endpoints)
    tiny = 1e-9
    x_grid = np.linspace(tiny * L, (1 - tiny) * L, n_x)
    u_grid = x_grid / L

    # set up figure
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(a_values)))

    standardized = []  # entries: dict with keys a, y, P_y, peak, ymin_sig, ymax_sig, color

    for i, a in enumerate(a_values):
        P_x = analytic_P(x_grid, N=N, L=L, a=a, x0=x0, n_max=n_max, decay_tol=decay_tol)

        # left panel: L*P(x) vs x/L
        axL.plot(u_grid, L * P_x, color=colors[i], linewidth=2, label=f'a={a:.2f} μm')

        # compute mean, sigma
        integral = np.trapz(P_x, x_grid)
        mean_x = np.trapz(x_grid * P_x, x_grid)
        mean_x2 = np.trapz((x_grid**2) * P_x, x_grid)
        sigma = np.sqrt(max(0.0, mean_x2 - mean_x**2))

        # standardized
        if sigma <= 0:
            print(f"Skipping a={a}: sigma ~ 0")
            continue
        y = (x_grid - mean_x) / sigma
        P_y = sigma * P_x

        peak = P_y.max()
        mask_sig = P_y > (EPSILON * peak)   # significant support mask
        if not np.any(mask_sig):
            # fallback: choose min-support around peak (e.g. where P_y > 1e-6*peak)
            mask_sig = P_y > (1e-8 * peak)

        ymin_sig = y[mask_sig].min()
        ymax_sig = y[mask_sig].max()

        standardized.append({
            'a': a, 'y': y, 'P_y': P_y, 'peak': peak,
            'ymin_sig': ymin_sig, 'ymax_sig': ymax_sig, 'color': colors[i],
            'mean': mean_x, 'sigma': sigma
        })

        print(f"a={a:.3f}: integral={integral:.6e}, mean={mean_x:.6e}, sigma={sigma:.6e}, "
              f"sig_y=[{ymin_sig:.3f}, {ymax_sig:.3f}]")

    # finalize left panel
    axL.set_xlabel('x / L', fontsize=18, fontweight='bold')
    axL.set_ylabel('L · P(x)', fontsize=18, fontweight='bold')
    axL.set_title('L·P(x) vs x/L (persistence-length sweep)', fontsize=16, fontweight='bold')
    axL.legend(title='Kuhn length', title_fontsize=12, fontsize=11)
    
    # Set tick label sizes for left panel
    axL.tick_params(axis='both', which='major', labelsize=14)
    axL.tick_params(axis='both', which='minor', labelsize=12)

    # determine intersection of significant supports
    if standardized:
        int_low = max(it['ymin_sig'] for it in standardized)
        int_high = min(it['ymax_sig'] for it in standardized)
        if int_high <= int_low:
            print("No overlap in significant supports; using pairwise overlap fallback.")
            # fallback: use median intersection width centered at zero
            widths = [it['ymax_sig'] - it['ymin_sig'] for it in standardized]
            median_width = np.median(widths)
            half = max(1.5, 0.5 * median_width)
            int_low, int_high = -half, half
            print(f"Fallback intersection: [{int_low:.3f},{int_high:.3f}]")
    else:
        int_low, int_high = -4.0, 4.0

    # choose a y_grid inside the intersection
    y_grid = np.linspace(int_low, int_high, 1200)

    # plot standardized curves *only on intersection*
    for it in standardized:
        # interpolate onto y_grid, but outside gives NaN via np.interp workaround
        P_interp = np.interp(y_grid, it['y'], it['P_y'], left=np.nan, right=np.nan)
        valid = ~np.isnan(P_interp)
        if not valid.any():
            continue
        # plot contiguous segments
        idx = np.where(valid)[0]
        splits = np.where(np.diff(idx) != 1)[0]
        starts = np.concatenate(([idx[0]], idx[splits + 1]))
        ends = np.concatenate((idx[splits], [idx[-1]]))
        for s, e in zip(starts, ends):
            axR.plot(y_grid[s:e+1], P_interp[s:e+1], color=it['color'], linewidth=2)
        axR.plot([], [], color=it['color'], linewidth=2, label=f"a={it['a']:.2f} μm")

    # optionally overlay native standardized curves faintly for context
    if PLOT_NATIVE_OVERLAY:
        for it in standardized:
            axR.plot(it['y'], it['P_y'], color=it['color'], linewidth=1, alpha=0.25)

    # overlay standard normal
    y_ref = np.linspace(-6, 6, 1000)
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * y_ref**2)
    axR.plot(y_ref, phi, 'k--', linewidth=1.5, label='Standard normal')

    axR.set_xlabel('(x - <x>) / σ', fontsize=18, fontweight='bold')
    axR.set_ylabel('σ · P(x)', fontsize=18, fontweight='bold')
    axR.set_title('Standardized collapse (intersection of significant supports)', 
                  fontsize=16, fontweight='bold')
    axR.legend(title='Kuhn length', title_fontsize=12, fontsize=11)
    
    # Set the x-axis limits of the right panel to -6 to +6
    axR.set_xlim(-6, 6)
    
    # Set tick label sizes for right panel
    axR.tick_params(axis='both', which='major', labelsize=14)
    axR.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout()
    
    # Save figure if requested
    if SAVE_FIG:
        # Create a 'figures' directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Save as PDF with 600 dpi
        pdf_path = os.path.join('figures', f'{FIGURE_NAME}.pdf')
        plt.savefig(pdf_path, dpi=600, format='pdf', bbox_inches='tight')
        print(f"Figure saved as {pdf_path} (600 dpi PDF)")
        
        # Save as PNG with 600 dpi
        png_path = os.path.join('figures', f'{FIGURE_NAME}.png')
        plt.savefig(png_path, dpi=600, format='png', bbox_inches='tight')
        print(f"Figure saved as {png_path} (600 dpi PNG)")
    
    plt.show()
