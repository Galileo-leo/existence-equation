"""
analyze_v5_hydrogen_orbit.py  v1.1
===================================
Hydrogen orbital analysis: radial distribution and motion from the ED v5 electron track.

Reads the saved proton–electron relative trajectory, splits time into an early transient
and a late stable segment (t >= T_CUT), and compares radial PDF P(r) for ALL vs STABLE data.
Fits three reference radial models to the histogram (nonlinear least squares on binned P(r)):
  - H 1s:  P(r) ~ r^2 exp(-2r/a0), the hydrogen ground-state |psi|^2 radial factor in 3D.
  - Rayleigh: P(r) ~ r exp(-r^2/(2 sigma^2)), 2D radial PDF for isotropic Gaussian displacement.
  - Maxwell-3D: P(r) ~ r^2 exp(-r^2/(2 sigma^2)), 3D speed PDF for isotropic Gaussian velocity.

Also estimates angular velocity from delta(theta)/dt with branch cuts folded to (-pi, pi],
plots the stable 2D orbit in the proton frame, and writes numeric summaries including R^2
per model for ALL vs STABLE.

v1.1: Transient removal (T_CUT), fit on stable region only, ALL vs STABLE comparison panel.
v1.0: Initial version.

Input:  v5_hydrogen_proton2_track.json
Output: v5_hydrogen_orbit_analysis.png, v5_hydrogen_orbit_stats.json

Jae-Ahn Shin & Kael · 2026-03-23
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

T_CUT = 20.0

with open('v5_hydrogen_proton2_track.json') as f:
    d = json.load(f)

track = d['track']
dt_sim = 0.022553

rx_all = np.array([t['rx'] for t in track if t['found']])
ry_all = np.array([t['ry'] for t in track if t['found']])
r_all  = np.array([t['dist'] for t in track if t['found']])
th_all = np.array([t['theta'] for t in track if t['found']])
steps_all = np.array([t['step'] for t in track if t['found']])
t_all = steps_all * dt_sim

stable = t_all >= T_CUT
rx = rx_all[stable]
ry = ry_all[stable]
r  = r_all[stable]
th = th_all[stable]
t_phys = t_all[stable]

print(f"  Total points: {len(r_all)}")
print(f"  Stable points (t >= {T_CUT}): {len(r)}")
print(f"  ALL   r: mean={np.mean(r_all):.3f}, std={np.std(r_all):.3f}")
print(f"  STABLE r: mean={np.mean(r):.3f}, std={np.std(r):.3f}")

# ══════════════════════════════════════════════════════════════
# Radial PDF shapes (unnormalized; A is a vertical scale in the fit)
# ══════════════════════════════════════════════════════════════

def hydrogen_1s(r, a0, A):
    return A * r**2 * np.exp(-2*r / a0)

def rayleigh(r, sig, A):
    return A * r * np.exp(-r**2 / (2 * sig**2))

def maxwell_3d(r, sig, A):
    return A * r**2 * np.exp(-r**2 / (2 * sig**2))

def fit_models(r_data, label):
    r_max_hist = min(np.percentile(r_data, 99.5) * 1.3, np.max(r_data) + 0.5)
    bins = np.linspace(0, r_max_hist, 80)
    counts, edges = np.histogram(r_data, bins=bins, density=True)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    r_fit = np.linspace(0.01, r_max_hist, 300)

    fits = {}
    for name, func, p0 in [
        ('H 1s', hydrogen_1s, [2.0, 1.0]),
        ('Rayleigh', rayleigh, [2.0, 1.0]),
        ('Maxwell-3D', maxwell_3d, [2.0, 1.0]),
    ]:
        try:
            popt, _ = curve_fit(func, r_mid, counts, p0=p0, maxfev=5000)
            y_pred = func(r_mid, *popt)
            ss_res = np.sum((counts - y_pred)**2)
            ss_tot = np.sum((counts - np.mean(counts))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fits[name] = {'popt': popt.tolist(), 'R2': r2, 'func': func}
            print(f"  [{label}] {name:12s}: params={[f'{p:.4f}' for p in popt]}, R2={r2:.4f}")
        except Exception as e:
            print(f"  [{label}] {name:12s}: fit failed ({e})")

    return fits, r_mid, counts, edges, r_fit

fits_all, rm_a, cnt_a, edg_a, rf_a = fit_models(r_all, "ALL")
fits_stb, rm_s, cnt_s, edg_s, rf_s = fit_models(r, "STABLE")

# ══════════════════════════════════════════════════════════════
# Angular velocity (stable segment): unwrap theta jumps across +/-pi
# ══════════════════════════════════════════════════════════════

dth = np.diff(th)
dth = np.where(dth > np.pi, dth - 2*np.pi, dth)
dth = np.where(dth < -np.pi, dth + 2*np.pi, dth)
dt_arr = np.diff(t_phys)
omega = dth / np.where(dt_arr > 0, dt_arr, 1e-10)
mean_omega = float(np.mean(np.abs(omega)))
T_orbit = 2*np.pi / mean_omega if mean_omega > 0 else 0
print(f"  <|omega|> = {mean_omega:.3f} rad/t, T_orbit ~ {T_orbit:.3f}")

# ══════════════════════════════════════════════════════════════
# Figure: 2x3 panels
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors_m = {'H 1s': 'red', 'Rayleigh': 'green', 'Maxwell-3D': 'blue'}

# (a) r(t) full timeline
ax = axes[0, 0]
ax.plot(t_all, r_all, lw=0.3, alpha=0.6, color='steelblue')
ax.axvline(T_CUT, color='orange', ls='--', lw=2, label=f'T_cut={T_CUT}')
ax.axhline(np.mean(r), color='red', ls='--', lw=1, label=f'stable mean={np.mean(r):.2f}')
ax.set_xlabel('time (phys)')
ax.set_ylabel('r (phys)')
ax.set_title('(a) r(t) full timeline')
ax.legend(fontsize=8)

# (b) P(r) ALL
ax = axes[0, 1]
ax.bar(rm_a, cnt_a, width=edg_a[1]-edg_a[0], alpha=0.35, color='gray', label='ALL data')
for name, info in fits_all.items():
    y = info['func'](rf_a, *info['popt'])
    ax.plot(rf_a, y, color=colors_m[name], lw=2,
            label=f"{name} R2={info['R2']:.3f}")
ax.set_xlabel('r (phys)')
ax.set_ylabel('P(r)')
ax.set_title(f'(b) P(r) ALL (n={len(r_all)})')
ax.legend(fontsize=8)

# (c) P(r) STABLE
ax = axes[0, 2]
ax.bar(rm_s, cnt_s, width=edg_s[1]-edg_s[0], alpha=0.35, color='steelblue', label='STABLE data')
for name, info in fits_stb.items():
    y = info['func'](rf_s, *info['popt'])
    ax.plot(rf_s, y, color=colors_m[name], lw=2,
            label=f"{name} R2={info['R2']:.3f}")
ax.set_xlabel('r (phys)')
ax.set_ylabel('P(r)')
ax.set_title(f'(c) P(r) STABLE t>{T_CUT} (n={len(r)})')
ax.legend(fontsize=8)

# (d) 2D trajectory (stable)
ax = axes[1, 0]
ax.plot(rx, ry, lw=0.2, alpha=0.4, color='steelblue')
ax.scatter([0], [0], c='red', s=120, marker='^', zorder=5, label='Proton')
ax.set_xlabel('rx (phys)')
ax.set_ylabel('ry (phys)')
ax.set_title(f'(d) 2D orbit (stable, t>{T_CUT})')
ax.set_aspect('equal')
ax.legend()

# (e) omega(t)
ax = axes[1, 1]
ax.plot(t_phys[1:], omega, lw=0.2, alpha=0.4, color='purple')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axhline(mean_omega, color='red', ls='--', lw=1, label=f'<|w|>={mean_omega:.1f}')
ax.axhline(-mean_omega, color='red', ls='--', lw=1)
ax.set_xlabel('time (phys)')
ax.set_ylabel('omega (rad/t)')
ax.set_title(f'(e) Angular velocity (T_orb ~ {T_orbit:.2f})')
ax.legend(fontsize=8)

# (f) R2 comparison bar chart
ax = axes[1, 2]
model_names = ['H 1s', 'Rayleigh', 'Maxwell-3D']
r2_all_vals = [fits_all.get(n, {}).get('R2', 0) for n in model_names]
r2_stb_vals = [fits_stb.get(n, {}).get('R2', 0) for n in model_names]
x_pos = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x_pos - w/2, r2_all_vals, w, label='ALL', color='gray', alpha=0.7)
bars2 = ax.bar(x_pos + w/2, r2_stb_vals, w, label='STABLE', color='steelblue', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.set_ylabel('R^2')
ax.set_title('(f) Model comparison: ALL vs STABLE')
ax.set_ylim(0, 1.05)
ax.legend()
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
            f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle(f'ED v5 Hydrogen: Orbit analysis (T_cut={T_CUT})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('v5_hydrogen_orbit_analysis.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"\n  Saved: v5_hydrogen_orbit_analysis.png")

# ══════════════════════════════════════════════════════════════
# Stats JSON
# ══════════════════════════════════════════════════════════════

best_all = max(fits_all.items(), key=lambda x: x[1]['R2'])[0] if fits_all else "?"
best_stb = max(fits_stb.items(), key=lambda x: x[1]['R2'])[0] if fits_stb else "?"

stats = {
    'T_CUT': T_CUT,
    'n_total': len(r_all),
    'n_stable': len(r),
    'all': {
        'r_mean': round(float(np.mean(r_all)), 4),
        'r_std': round(float(np.std(r_all)), 4),
        'best_model': best_all,
        'fits': {n: {'params': info['popt'], 'R2': round(info['R2'], 4)}
                 for n, info in fits_all.items()},
    },
    'stable': {
        'r_mean': round(float(np.mean(r)), 4),
        'r_std': round(float(np.std(r)), 4),
        'best_model': best_stb,
        'fits': {n: {'params': info['popt'], 'R2': round(info['R2'], 4)}
                 for n, info in fits_stb.items()},
    },
    'omega_mean': round(mean_omega, 4),
    'T_orbit': round(T_orbit, 4),
    'bound_pct': round(float(np.sum(r < 10) / len(r) * 100), 2),
}

with open('v5_hydrogen_orbit_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  Saved: v5_hydrogen_orbit_stats.json")
