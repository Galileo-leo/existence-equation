#!/usr/bin/env python3
"""
EP7 Figures 2 and 3
  Fig 2: Goldstone + Higgs dispersion relations
  Fig 3: epsilon-regime map across physical systems
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

# ============================================================
# FIGURE 2: Dispersion relations
# ============================================================
# ED parameters (natural units: c = 1, lambda = 1)
c = 1.0
lam = 1.0
mu_ED = c * np.sqrt(2 * lam)  # = sqrt(2)

k = np.linspace(0, 3.5, 500)
omega_goldstone = c * k
omega_higgs = np.sqrt(c**2 * k**2 + 2 * lam * c**2)

fig, ax = plt.subplots(figsize=(8, 6))

# GP-valid region (shaded)
k_gp = 0.5  # k where epsilon ~ 0.03 (GP valid up to here)
ax.axvspan(0, k_gp, alpha=0.12, color='blue', label=r'GP regime ($\varepsilon \ll 1$)')

# ED-only region (Higgs active)
k_ed_start = 1.0  # k where epsilon ~ 0.25
ax.axvspan(k_ed_start, 3.5, alpha=0.12, color='red',
           label=r'ED regime ($\varepsilon \gtrsim \mathcal{O}(0.1)$)')

# Light cone reference
ax.plot(k, c * k, 'k:', lw=1, alpha=0.5, label=r'Light cone $\omega = ck$')

# Dispersion curves
ax.plot(k, omega_goldstone, 'b-', lw=2.5,
        label=r'Goldstone (phase): $\omega^2 = c^2 k^2$')
ax.plot(k, omega_higgs, 'r-', lw=2.5,
        label=r'Higgs (amplitude): $\omega^2 = c^2 k^2 + 2\lambda c^2$')

# Higgs gap annotation
ax.axhline(mu_ED, color='red', ls='--', lw=1, alpha=0.6)
ax.annotate(r'$\mu_{\mathrm{ED}} = c\sqrt{2\lambda}$',
            xy=(0.05, mu_ED), xytext=(0.3, mu_ED + 0.2),
            fontsize=12, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1))

# Annotation: condensate point
ax.plot(0, 0, 'ko', ms=8)
ax.annotate('condensate\nvacuum $A_0$', xy=(0, 0), xytext=(0.15, 0.35),
            fontsize=10, ha='left')

# Epsilon markers on x-axis
for eps_val, label in [(0.1, r'$\varepsilon=0.1$'), (0.5, r'$\varepsilon=0.5$'),
                        (1.0, r'$\varepsilon=1$')]:
    k_eps = np.sqrt(4 * lam * eps_val)
    if k_eps < 3.5:
        ax.plot([k_eps, k_eps], [0, 0.1], 'k-', lw=1)
        ax.text(k_eps, -0.25, label, ha='center', fontsize=9, color='gray')

ax.set_xlabel(r'Wavenumber $k$ (units of $\sqrt{\lambda}$)', fontsize=12)
ax.set_ylabel(r'Frequency $\omega$ (units of $c\sqrt{\lambda}$)', fontsize=12)
ax.set_title('Dispersion Relations of the Existence Equation\n'
             r'around the Mexican-hat vacuum $A_0 = \sqrt{\lambda/\alpha}$',
             fontsize=13)
ax.set_xlim(0, 3.5)
ax.set_ylim(-0.4, 4.2)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Text: GP freezes the Higgs
ax.text(2.5, 0.6, 'GP reduction:\nfreeze Higgs branch,\nretain low-$k$ Goldstone',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.savefig('/sessions/elegant-wizardly-hopper/mnt/ed8/ep7_fig2_dispersion.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved: ep7_fig2_dispersion.png")

# ============================================================
# FIGURE 3: epsilon-regime map across physical systems
# ============================================================
systems = [
    # (name, epsilon_low, epsilon_high, regime, color)
    (r'ULDM / fuzzy DM',              1e-42, 1e-38, 'GP exact',        'steelblue'),
    (r'SC Josephson, $V \ll \Delta$', 1e-7,  1e-5,  'GP exact',        'steelblue'),
    (r'BEC Bragg, low $k$',           1e-4,  1e-2,  'GP exact',        'steelblue'),
    (r'BEC Bragg, high $k$',          5e-2,  2e-1,  'ED correction',   'darkorange'),
    (r'SC Josephson, $V \sim \Delta$', 1e-1,  1.0,   'ED regime',       'crimson'),
    (r'Neutron-star superfluid',      1e-1,  1.0,   'ED required',     'crimson'),
    (r'QCD chiral condensate',        5e-1,  3.0,   'ED fundamental',  'darkred'),
]

fig, ax = plt.subplots(figsize=(10, 6.5))

# Background regime bands
ax.axvspan(1e-44, 1e-2, alpha=0.08, color='blue')
ax.axvspan(1e-2, 1e-1, alpha=0.10, color='orange')
ax.axvspan(1e-1, 1e1, alpha=0.12, color='red')

# Critical vertical lines
ax.axvline(1e-2, color='blue', ls='--', lw=1, alpha=0.5)
ax.axvline(1e-1, color='orange', ls='--', lw=1, alpha=0.6)
ax.axvline(1.0, color='red', ls='--', lw=1, alpha=0.6)

# Plot each system as horizontal bar
for i, (name, e_lo, e_hi, regime, color) in enumerate(systems):
    y = len(systems) - 1 - i
    ax.plot([e_lo, e_hi], [y, y], '-', color=color, lw=6, solid_capstyle='round')
    ax.plot([e_lo, e_hi], [y, y], 'o', color=color, ms=8)
    ax.text(e_hi * 2.5, y, f'  {regime}', va='center', fontsize=9,
            color=color, fontweight='bold')

# System names on y-axis
ax.set_yticks(range(len(systems)))
ax.set_yticklabels([s[0] for s in reversed(systems)], fontsize=10)

# Regime labels at top
ax.text(1e-22, len(systems) - 0.3, 'GP exact', ha='center', fontsize=11,
        color='steelblue', fontweight='bold')
ax.text(3e-2, len(systems) - 0.3, 'transition', ha='center', fontsize=11,
        color='darkorange', fontweight='bold')
ax.text(3, len(systems) - 0.3, 'ED required', ha='center', fontsize=11,
        color='crimson', fontweight='bold')

ax.set_xscale('log')
ax.set_xlim(1e-44, 1e2)
ax.set_ylim(-0.7, len(systems) - 0.5)
ax.set_xlabel(r'$\varepsilon \equiv k^2 / (4\lambda) \;=\; c^2 k^2 / (2\mu_{\mathrm{ED}}^2)$',
              fontsize=12)
ax.set_title(r'Experimental $\varepsilon$-Regime Map: where GP holds, '
             r'where ED becomes necessary', fontsize=12)
ax.grid(True, which='major', axis='x', alpha=0.3)

# Critical threshold annotations
ax.text(1e-2, -0.5, r'$\varepsilon = 10^{-2}$', ha='center', fontsize=9, color='blue')
ax.text(1e-1, -0.5, r'$\varepsilon = 10^{-1}$', ha='center', fontsize=9, color='darkorange')
ax.text(1.0, -0.5, r'$\varepsilon = 1$', ha='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('/sessions/elegant-wizardly-hopper/mnt/ed8/ep7_fig3_regime_map.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved: ep7_fig3_regime_map.png")

print("\nDone. Two figures ready for upload.")
