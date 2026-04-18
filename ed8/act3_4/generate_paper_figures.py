"""
generate_paper_figures.py
==========================
Publication-quality figure generator (300 dpi PNG) for the ED strong-force paper.

Run from the folder that contains the input files (paths are relative to CWD).
Matplotlib `savefig.dpi` is set to 300 so every `plt.savefig` writes print resolution.

Outputs (PNG):
  1. ed_strong_yjunction.png   -- Vortex map + Y-junction triplet structure
  2. ed_strong_confinement.png -- Linear confinement (flux tube E vs distance)
  3. ed_strong_charge.png      -- Charge quantization (winding vs R)
  4. ed_strong_orbit.png       -- Electron binding: r(t), P(r), 2D orbit, model comparison
  5. ed_strong_tail.png        -- Tail analysis: H 1s vs Rayleigh (uses fits from Fig 4)

Inputs:
  v5_frozen_state.npz
  v5_analysis.json
  v5_hydrogen_proton2_track.json

Jae-Ahn Shin & Kael · 2026-03-24
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import curve_fit

# On-screen DPI 100; saved figures use 300 dpi and tight bounding boxes for publication.
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Output directory for PNG files (relative to current working directory).
SAVE_DIR = '.'

# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════

print("Loading data...")

with open('v5_analysis.json') as f:
    analysis = json.load(f)

with open('v5_hydrogen_proton2_track.json') as f:
    track_data = json.load(f)

data = np.load('v5_frozen_state.npz')
psi_np = data['psi']
dx = float(data['dx'])
L = float(data['L'])
N = int(data['N'])
best_z = int(data['best_z'])

print(f"  N={N}, dx={dx:.4f}, L={L:.2f}, best_z={best_z}")

# Vortex detection: discrete phase curl on the best z-slice -> winding density; |w|>0.5 marks cores.
phase_sl = np.angle(psi_np[best_z])
amp_sl = np.abs(psi_np[best_z])

dp_x = np.angle(np.exp(1j * (np.roll(phase_sl, -1, axis=1) - phase_sl)))
dp_y = np.angle(np.exp(1j * (np.roll(phase_sl, -1, axis=0) - phase_sl)))
curl = dp_x + np.roll(dp_y, -1, axis=1) - np.roll(dp_x, -1, axis=0) - dp_y
winding_map = curl / (2 * np.pi)

yp, xp = np.where(winding_map > 0.5)
yn, xn = np.where(winding_map < -0.5)

coords_phys = np.arange(N) * dx - L/2

PROTON_X = -3.5
PROTON_Y = -4.2
pc_ix = int((PROTON_X + L/2) / dx) % N
pc_iy = int((PROTON_Y + L/2) / dx) % N

print(f"  Vortices: +{len(xp)} / -{len(xn)}")


# ══════════════════════════════════════════════════════════════
# Fig 1: Y-junction vortex map
# ══════════════════════════════════════════════════════════════

print("Generating ed_strong_yjunction.png ...")

xp_phys = xp * dx - L/2
yp_phys = yp * dx - L/2
xn_phys = xn * dx - L/2
yn_phys = yn * dx - L/2

# Identify core triplet (+1 vortices within R<1.5 of Proton #2)
CORE_R = 1.5
dp_core = np.sqrt((xp_phys - PROTON_X)**2 + (yp_phys - PROTON_Y)**2)
dn_core = np.sqrt((xn_phys - PROTON_X)**2 + (yn_phys - PROTON_Y)**2)
core_p_mask = dp_core < CORE_R
core_n_mask = dn_core < CORE_R

core_px = xp_phys[core_p_mask]
core_py = yp_phys[core_p_mask]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# (a) Full vortex map
ax = axes[0]
ax.imshow(amp_sl, extent=[-L/2, L/2, -L/2, L/2], cmap='gray', origin='lower', alpha=0.5)
ax.scatter(xp_phys, yp_phys, c='red', s=3, alpha=0.6, label=f'+1 vortex (n={len(xp)})')
ax.scatter(xn_phys, yn_phys, c='blue', s=3, alpha=0.6, label=f'-1 vortex (n={len(xn)})')
ax.set_xlabel('x (phys)')
ax.set_ylabel('y (phys)')
ax.set_title('(a) Vortex map (2D slice at $z=z_{best}$)')
ax.legend(loc='upper right', markerscale=3)

# (b) Zoom on Proton #2 with core triplet highlighted
zoom_r = 3.5
ax = axes[1]
ax.imshow(amp_sl, extent=[-L/2, L/2, -L/2, L/2], cmap='gray', origin='lower', alpha=0.5)

# Outer vortices: faded
ax.scatter(xp_phys, yp_phys, c='red', s=15, alpha=0.15, zorder=2)
ax.scatter(xn_phys, yn_phys, c='blue', s=15, alpha=0.15, zorder=2)

# -1 vortices near core: visible but marked as slice artifacts
near_nx = xn_phys[core_n_mask]
near_ny = yn_phys[core_n_mask]
ax.scatter(near_nx, near_ny, c='blue', s=50, alpha=0.5, zorder=3,
           marker='x', linewidths=2, label='-1 (slice artifact)')

# Core +1 triplet: bright, large, connected by triangle
ax.scatter(core_px, core_py, c='red', s=120, alpha=1.0, zorder=5,
           edgecolors='darkred', linewidths=1.5, label='+1 core (triplet)')

if len(core_px) >= 3:
    tri_x = list(core_px[:3]) + [core_px[0]]
    tri_y = list(core_py[:3]) + [core_py[0]]
    ax.plot(tri_x, tri_y, 'r-', lw=2.5, alpha=0.8, zorder=4)

circle_inner = Circle((PROTON_X, PROTON_Y), CORE_R, fill=False,
                       ec='orange', lw=2, ls='-', zorder=4)
circle_outer = Circle((PROTON_X, PROTON_Y), 3.0, fill=False,
                       ec='orange', lw=1.2, ls=':', zorder=4)
ax.add_patch(circle_inner)
ax.add_patch(circle_outer)

ax.set_xlim(PROTON_X - zoom_r, PROTON_X + zoom_r)
ax.set_ylim(PROTON_Y - zoom_r, PROTON_Y + zoom_r)
ax.set_xlabel('x (phys)')
ax.set_ylabel('y (phys)')
ax.set_title('(b) Y-junction core (Proton #2)')
ax.legend(loc='upper right', fontsize=8)

# (c) NET charge vs radius table/bar
radii = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
net_charges = []
for rr in radii:
    n_p = np.sum(np.sqrt((xp_phys - PROTON_X)**2 + (yp_phys - PROTON_Y)**2) < rr)
    n_n = np.sum(np.sqrt((xn_phys - PROTON_X)**2 + (yn_phys - PROTON_Y)**2) < rr)
    net_charges.append(n_p - n_n)

ax = axes[2]
colors_bar = ['red' if v > 0 else ('blue' if v < 0 else 'gray') for v in net_charges]
bars = ax.bar(radii, net_charges, width=0.4, color=colors_bar, alpha=0.7, edgecolor='black', lw=0.8)
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axhline(1, color='red', ls='--', lw=1, alpha=0.5, label='Q = +1 (proton)')

for bar, val in zip(bars, net_charges):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.08 * np.sign(val + 0.01),
            f'{val:+d}', ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=11, fontweight='bold')

ax.set_xlabel('Measurement radius R (phys)')
ax.set_ylabel('NET winding (charge)')
ax.set_title('(c) Enclosed charge vs R')
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(radii)
ax.legend(fontsize=9)

ax.text(0.5, -0.15,
        '2D slice: flux tubes crossing the plane create\n'
        'apparent $\\pm$1 pairs that cancel at larger R.',
        transform=ax.transAxes, ha='center', fontsize=8,
        style='italic', color='gray')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ed_strong_yjunction.png')
plt.close()
print("  Saved: ed_strong_yjunction.png")


# ══════════════════════════════════════════════════════════════
# Fig 2: Flux tube confinement (3D vortex-line measurement)
# ══════════════════════════════════════════════════════════════

print("Generating ed_strong_confinement.png ...")

import os
conf3d_path = 'v5_confinement_3d.json'
if os.path.exists(conf3d_path):
    with open(conf3d_path) as f:
        conf3d = json.load(f)
    pairs = conf3d['pairs']
    dists = np.array([p['dist_3d'] for p in pairs])
    e_path = np.array([p['tube_E_path'] for p in pairs])
    ft_ratio = np.array([p['FT_ratio'] for p in pairs])
else:
    conf = analysis['confinement']
    dists = np.array([c['dist'] for c in conf])
    e_path = np.array([c['tube_E'] for c in conf])
    ft_ratio = np.array([c['FT_ratio'] for c in conf])

ft_mask = ft_ratio > 1.5
n_ft = int(np.sum(ft_mask))

sigma_per_d = e_path / np.maximum(dists, 0.01)
if os.path.exists(conf3d_path):
    sigma_bg = float(conf3d.get('sigma_bg', np.median(sigma_per_d[~ft_mask])))
    sigma_ft_mean = float(conf3d.get('sigma_mean_ft', np.mean(sigma_per_d[ft_mask]) if n_ft > 0 else 0))
else:
    sigma_bg = float(np.median(sigma_per_d[~ft_mask])) if np.sum(~ft_mask) >= 3 else float(np.median(sigma_per_d))
    sigma_ft_mean = float(np.mean(sigma_per_d[ft_mask])) if n_ft > 0 else 0
ratio_ft_bg = sigma_ft_mean / sigma_bg if sigma_bg > 0 else 0

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# (a) E_path vs d — geometric baseline
ax = axes[0]
if n_ft > 0:
    ax.scatter(dists[~ft_mask], e_path[~ft_mask], c='gray', s=20, alpha=0.35, zorder=2,
               label=f'FT<1.5 (n={np.sum(~ft_mask)})')
    ax.scatter(dists[ft_mask], e_path[ft_mask], c='#D32F2F', s=40, alpha=0.8, zorder=4,
               edgecolors='darkred', linewidths=0.5,
               label=f'FT>1.5 (n={n_ft})')
xf = np.linspace(0, max(dists)*1.1, 100)
ax.plot(xf, sigma_bg * xf, 'k--', lw=1.5, alpha=0.5,
        label=f'Background $\\sigma_{{bg}}$={sigma_bg:.0f}')
ax.set_xlabel('Vortex-line separation d (phys)')
ax.set_ylabel('Path-integrated energy $E_{{path}}$')
ax.set_title('(a) Tube energy vs distance')
ax.legend(fontsize=8)

# (b) σ = E/d vs d — KEY panel
ax = axes[1]
if n_ft > 0:
    ax.scatter(dists[~ft_mask], sigma_per_d[~ft_mask], c='gray', s=20, alpha=0.35, zorder=2,
               label=f'FT<1.5 (n={np.sum(~ft_mask)})')
    ax.scatter(dists[ft_mask], sigma_per_d[ft_mask], c='#D32F2F', s=40, alpha=0.8, zorder=4,
               edgecolors='darkred', linewidths=0.5,
               label=f'FT>1.5 (n={n_ft})')
ax.axhline(sigma_bg, color='gray', ls='--', lw=1.5, alpha=0.5,
           label=f'$\\sigma_{{bg}}$={sigma_bg:.0f}')
if n_ft > 0:
    ax.axhline(sigma_ft_mean, color='red', ls='-', lw=1.5, alpha=0.5,
               label=f'FT>1.5 mean={sigma_ft_mean:.0f}')
ax.set_xlabel('Vortex-line separation d (phys)')
ax.set_ylabel('$\\sigma$ = E/d (string tension)')
ax.set_title(f'(b) String tension: FT/bg = {ratio_ft_bg:.2f}×')
ax.legend(fontsize=8)

# (c) FT ratio vs d
ax = axes[2]
if n_ft > 0:
    ax.scatter(dists[ft_mask], ft_ratio[ft_mask], c='#D32F2F', s=40, alpha=0.7,
               label=f'FT>1.5 (n={n_ft})', zorder=3)
    ax.scatter(dists[~ft_mask], ft_ratio[~ft_mask], c='gray', s=20, alpha=0.4)
ax.axhline(1.5, color='orange', ls='--', lw=1.5, label='FT threshold')
ax.set_xlabel('Vortex-line separation d (phys)')
ax.set_ylabel('Flux tube / background ratio')
ax.set_title(f'(c) Signal quality ({n_ft}/{len(dists)} pass)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ed_strong_confinement.png')
plt.close()
print(f"  Saved: ed_strong_confinement.png (σ_FT={sigma_ft_mean:.0f}, σ_bg={sigma_bg:.0f}, ratio={ratio_ft_bg:.2f})")


# ══════════════════════════════════════════════════════════════
# Fig 3: Charge quantization
# ══════════════════════════════════════════════════════════════

print("Generating ed_strong_charge.png ...")

# Enclosed phase winding vs measurement radius (charge quantization around triplet center).
cvr = analysis['charge_vs_r']
R_phys = np.array([c['R_phys'] for c in cvr])
windings = np.array([c['winding'] for c in cvr])

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(R_phys, windings, 'o-', color='steelblue', ms=6, lw=1.5, zorder=3)
ax.axhline(0, color='gray', ls='-', lw=0.5)

for q_val, q_label, q_color in [(1, '+1', 'red'), (-1, '-1', 'blue'), (0, '0', 'gray')]:
    ax.axhline(q_val, color=q_color, ls=':', lw=1, alpha=0.5)

ax.fill_between(R_phys, -0.1, 0.1, alpha=0.1, color='green', label='Charge ~ 0 band')
ax.set_xlabel('Measurement radius R (phys)')
ax.set_ylabel('Total winding number (charge)')
ax.set_title(f'Charge quantization: mean={analysis["mean_charge"]:.3f}, std={analysis["std_charge"]:.3f}')
ax.legend(fontsize=10)
ax.set_ylim(-2, 2)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ed_strong_charge.png')
plt.close()
print("  Saved: ed_strong_charge.png")


# ══════════════════════════════════════════════════════════════
# Fig 4: Electron orbit + P(r)
# ══════════════════════════════════════════════════════════════

print("Generating ed_strong_orbit.png ...")

track = track_data['track']
dt_sim = 0.022553

rx_all = np.array([t['rx'] for t in track if t['found']])
ry_all = np.array([t['ry'] for t in track if t['found']])
r_all = np.array([t['dist'] for t in track if t['found']])
th_all = np.array([t['theta'] for t in track if t['found']])
steps = np.array([t['step'] for t in track if t['found']])
t_all = steps * dt_sim

# Drop early times so histograms and orbit plots reflect post-transient bound motion only.
T_CUT = 20.0
stable = t_all >= T_CUT
rx = rx_all[stable]
ry = ry_all[stable]
r = r_all[stable]
th = th_all[stable]

def hydrogen_1s(r, a0, A):
    return A * r**2 * np.exp(-2*r / a0)

def rayleigh(r, sig, A):
    return A * r * np.exp(-r**2 / (2 * sig**2))

def maxwell_3d(r, sig, A):
    return A * r**2 * np.exp(-r**2 / (2 * sig**2))

r_max_hist = min(np.percentile(r, 99.5) * 1.3, np.max(r) + 0.5)
bins = np.linspace(0, r_max_hist, 80)
counts, edges = np.histogram(r, bins=bins, density=True)
r_mid = 0.5 * (edges[:-1] + edges[1:])
r_fit = np.linspace(0.01, r_max_hist, 300)

# Nonlinear least-squares fits to the radial PDF (stable segment); R^2 for legend and Fig 5 tail.
fits = {}
for name, func, p0 in [('H 1s', hydrogen_1s, [2.0, 1.0]),
                         ('Rayleigh', rayleigh, [2.0, 1.0]),
                         ('Maxwell-3D', maxwell_3d, [2.0, 1.0])]:
    try:
        popt, _ = curve_fit(func, r_mid, counts, p0=p0, maxfev=5000)
        y_pred = func(r_mid, *popt)
        ss_res = np.sum((counts - y_pred)**2)
        ss_tot = np.sum((counts - np.mean(counts))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits[name] = {'popt': popt, 'R2': r2, 'func': func}
    except:
        pass

colors_m = {'H 1s': 'red', 'Rayleigh': 'green', 'Maxwell-3D': 'blue'}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) r(t) timeline
ax = axes[0, 0]
ax.plot(t_all, r_all, lw=0.3, alpha=0.5, color='steelblue')
ax.axvline(T_CUT, color='orange', ls='--', lw=2, label=f'$T_{{cut}}$={T_CUT}')
ax.axhline(np.mean(r), color='red', ls='--', lw=1, label=f'$\\langle r \\rangle$={np.mean(r):.2f}')
ax.set_xlabel('Time (phys)')
ax.set_ylabel('r (phys)')
ax.set_title(f'(a) Electron-proton distance (n={len(r_all)})')
ax.legend(fontsize=9)

# (b) P(r) with fits
ax = axes[0, 1]
ax.bar(r_mid, counts, width=edges[1]-edges[0], alpha=0.3, color='steelblue', label='Data')
for name, info in fits.items():
    y = info['func'](r_fit, *info['popt'])
    ax.plot(r_fit, y, color=colors_m[name], lw=2,
            label=f"{name} $R^2$={info['R2']:.3f}")
ax.set_xlabel('r (phys)')
ax.set_ylabel('P(r)')
ax.set_title(f'(b) Radial probability (stable, n={len(r)})')
ax.legend(fontsize=9)

# (c) 2D orbit
ax = axes[1, 0]
ax.scatter(rx, ry, c=np.arange(len(rx)), cmap='viridis', s=0.5, alpha=0.3)
ax.plot(0, 0, 'r^', ms=10, zorder=5, label='Proton')
ax.set_xlabel('$r_x$ (phys)')
ax.set_ylabel('$r_y$ (phys)')
ax.set_title('(c) 2D orbit (stable)')
ax.set_aspect('equal')
ax.legend(fontsize=9)

# (d) R^2 comparison
ax = axes[1, 1]
names = list(fits.keys())
r2_vals = [fits[n]['R2'] for n in names]
colors_bar = [colors_m[n] for n in names]
bars = ax.bar(names, r2_vals, color=colors_bar, alpha=0.7)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('$R^2$')
ax.set_title('(d) Model comparison')
ax.set_ylim(0.7, 1.02)
ax.axhline(0.95, color='gray', ls=':', lw=0.8)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ed_strong_orbit.png')
plt.close()
print("  Saved: ed_strong_orbit.png")


# ══════════════════════════════════════════════════════════════
# Fig 5: Tail analysis
# ══════════════════════════════════════════════════════════════

print("Generating ed_strong_tail.png ...")

# Compare H 1s vs Rayleigh in the large-r tail using parameters from Fig 4 fits.
popt_h = fits['H 1s']['popt']
popt_r = fits['Rayleigh']['popt']

# Radial bins for mean model vs normalized histogram in each tail segment.
segments = [(4, 5, '4-5'), (5, 6, '5-6'), (6, 7, '6-7'), (7, 14, '7+')]
seg_results = []

for rlo, rhi, label in segments:
    mask = (r >= rlo) & (r < rhi)
    n_pts = int(np.sum(mask))
    if n_pts < 10:
        continue
    bins_seg = np.linspace(rlo, rhi, max(int((rhi-rlo)/0.2), 5))
    cnt_seg, edg_seg = np.histogram(r[mask], bins=bins_seg, density=False)
    cnt_seg_norm = cnt_seg / (len(r) * (edg_seg[1] - edg_seg[0]))
    rm_seg = 0.5 * (edg_seg[:-1] + edg_seg[1:])

    h1s_mean = float(np.mean(hydrogen_1s(rm_seg, *popt_h)))
    ray_mean = float(np.mean(rayleigh(rm_seg, *popt_r)))
    data_mean = float(np.mean(cnt_seg_norm[cnt_seg_norm > 0])) if np.any(cnt_seg_norm > 0) else 0

    seg_results.append({
        'label': label, 'n': n_pts,
        'h1s_ratio': h1s_mean / data_mean if data_mean > 0 else 0,
        'ray_ratio': ray_mean / data_mean if data_mean > 0 else 0,
    })

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# (a) Log-scale tail zoom
ax = axes[0]
tail_mask = r_mid >= 3.5
valid = tail_mask & (counts > 0)
ax.semilogy(r_mid[valid], counts[valid], 'ko', ms=5, alpha=0.7, label='Data', zorder=3)
rf_tail = np.linspace(3.5, r_max_hist, 300)
ax.semilogy(rf_tail, hydrogen_1s(rf_tail, *popt_h), 'r-', lw=2.5,
            label=f'H 1s ($a_0$={popt_h[0]:.2f})')
ray_y = rayleigh(rf_tail, *popt_r)
ray_y[ray_y < 1e-30] = 1e-30
ax.semilogy(rf_tail, ray_y, 'g--', lw=2.5,
            label=f'Rayleigh ($\\sigma$={popt_r[0]:.2f})')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r)  [log scale]', fontsize=12)
ax.set_title('(a) Tail region P(r): log scale', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(bottom=5e-5)

# (b) Segment ratio bar chart
ax = axes[1]
if seg_results:
    labels = [s['label'] for s in seg_results]
    h_ratios = [s['h1s_ratio'] for s in seg_results]
    r_ratios = [s['ray_ratio'] for s in seg_results]

    x = np.arange(len(labels))
    w = 0.35
    bars_h = ax.bar(x - w/2, h_ratios, w, color='red', alpha=0.7, label='H 1s / Data')
    bars_r = ax.bar(x + w/2, r_ratios, w, color='green', alpha=0.7, label='Rayleigh / Data')
    ax.axhline(1.0, color='black', ls='-', lw=1.5)

    for bar, val in zip(bars_h, h_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='darkred', fontweight='bold')
    for bar, val in zip(bars_r, r_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, color='darkgreen', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'r = {l}' for l in labels], fontsize=11)
    ax.set_ylabel('Model / Data ratio', fontsize=12)
    ax.set_title('(b) Tail accuracy: H 1s vs Rayleigh', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(h_ratios), 1.2) + 0.15)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/ed_strong_tail.png')
plt.close()
print("  Saved: ed_strong_tail.png")


# ══════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print("  All figures generated (300 dpi):")
print("    1. ed_strong_yjunction.png")
print("    2. ed_strong_confinement.png")
print("    3. ed_strong_charge.png")
print("    4. ed_strong_orbit.png")
print("    5. ed_strong_tail.png")
print(f"{'='*50}")
