#!/usr/bin/env python3
"""
==============================================================================
  EP V Figure 1 Generator — FINAL
==============================================================================
  Panel (a): Single vortex rotation curve vs 1/r
  Panel (b): Radial profile of |Psi| before/after (amplitude-phase decoupling)
  Panel (c): Distribution-dependent rotation curves (uniform vs inverse_r)
  Panel (d): Vortex position maps

  All data comes from JSON files — no simulations are run here.
  Robust to Step 5 key renaming (config-based run detection).

  Output:
    ed_topological_phase_result.png  (300 dpi, 4-panel composite)

  Run:
    python ep5_make_figure1_final.py

  Author: Jae-Ahn Shin  (Existence Dynamics framework)
==============================================================================
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
#  CONFIG
# ============================================================
JSON_STEP2 = 'ep5_step2_single_vortex.json'
JSON_STEP3 = 'ep5_step3_two_vortex.json'
JSON_STEP5 = 'ep5_step5_distributions.json'

OUTPUT_PNG = 'ed_topological_phase_result.png'
DPI = 300

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'savefig.dpi': DPI,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

COL_SIM     = '#1f77b4'
COL_THEORY  = '#d62728'
COL_INITIAL = '#1f77b4'
COL_FINAL   = '#d62728'
COL_UNIFORM = '#1f77b4'
COL_INVRAD  = '#d62728'
COL_CLEAN   = '#2ca02c'
COL_VVAC    = '#555555'


# ============================================================
#  LOAD
# ============================================================
print(f"Loading {JSON_STEP2}...")
with open(JSON_STEP2) as f:
    d2 = json.load(f)

print(f"Loading {JSON_STEP3}...")
with open(JSON_STEP3) as f:
    d3 = json.load(f)

print(f"Loading {JSON_STEP5}...")
with open(JSON_STEP5) as f:
    d5 = json.load(f)

print("All JSON loaded.\n")


# ============================================================
#  HELPERS
# ============================================================
def find_run_by_config(runs_dict, **match):
    for key, run in runs_dict.items():
        cfg = run.get('config', {})
        if all(cfg.get(k) == v for k, v in match.items()):
            return run, key
    first_key = next(iter(runs_dict))
    print(f"  [warn] no match for {match}; falling back to '{first_key}'")
    return runs_dict[first_key], first_key


def clean_profile(profile):
    rs, vs = [], []
    for p in profile:
        v = p.get('v_theta')
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        rs.append(p['r'])
        vs.append(v)
    return np.array(rs), np.array(vs)


def compute_flatness(r_arr, v_arr, r1=3.0, r2=9.0):
    mask = (r_arr >= r1) & (r_arr <= r2)
    if mask.sum() < 2:
        return None
    vs = np.abs(v_arr[mask])
    if vs.mean() < 1e-9:
        return None
    return 1.0 - vs.std() / vs.mean()


def extract_flatness(run, r_arr, v_arr):
    for key_path in (('flatness_statistics', 'flatness'),
                     ('flatness', 'flatness'),
                     ('statistics', 'flatness')):
        obj = run
        try:
            for k in key_path:
                obj = obj[k]
            if isinstance(obj, (int, float)):
                return float(obj)
        except (KeyError, TypeError):
            pass
    return compute_flatness(r_arr, v_arr)


def radial_profile(field_2d, extent, n_bins=80, r_max=None):
    """
    Azimuthally averaged radial profile of a 2D field.
    field_2d shape: (nx, ny)
    extent: [xmin, xmax, ymin, ymax]
    """
    xmin, xmax, ymin, ymax = extent
    nx, ny = field_2d.shape

    x = np.linspace(xmin, xmax, nx, endpoint=False)
    y = np.linspace(ymin, ymax, ny, endpoint=False)
    dx_pix = x[1] - x[0]
    dy_pix = y[1] - y[0]
    x = x + dx_pix / 2
    y = y + dy_pix / 2

    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)

    if r_max is None:
        r_max = min(abs(xmin), abs(xmax), abs(ymin), abs(ymax))

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    prof = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (R >= r_edges[i]) & (R < r_edges[i+1])
        if mask.sum() > 10:
            prof[i] = float(field_2d[mask].mean())

    return r_centers, prof


# ============================================================
#  RUN LOOKUP
# ============================================================
run5u, key5u = find_run_by_config(d5['runs'], distribution='uniform_disk')
run5i, key5i = find_run_by_config(d5['runs'], distribution='inverse_r')

print(f"  Step 5 uniform key: '{key5u}'")
print(f"  Step 5 inverse key: '{key5i}'\n")


# ============================================================
#  FIGURE
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
ax_a, ax_b = axes[0, 0], axes[0, 1]
ax_c, ax_d = axes[1, 0], axes[1, 1]


# ============================================================
#  PANEL (a) — Single vortex rotation curve
# ============================================================
print("Building Panel (a)...")

run2 = d2['runs']['single_vortex']
profile2 = run2['rotation_curve']
r2, v2 = clean_profile(profile2)

mask = (r2 > 0.5) & (r2 < 25)
r2p, v2p = r2[mask], v2[mask]

r_theory = np.linspace(0.5, 25, 300)
v_theory = 1.0 / r_theory

ax_a.plot(r_theory, v_theory, '-', color=COL_THEORY, linewidth=2,
          label=r'theory: $1/r$', zorder=2)
ax_a.plot(r2p, v2p, 'o', color=COL_SIM, markersize=3.5,
          markevery=2, label='simulation', zorder=3)

clean_stats = run2.get('clean_regime_statistics', {})
r_range = clean_stats.get('r_range', [3, 15])
ax_a.axvspan(r_range[0], r_range[1], alpha=0.12, color=COL_CLEAN,
             zorder=1, label='clean regime')

# Precision annotation — LOWER LEFT this time
mean_vr = clean_stats.get('vtheta_times_r_mean', float('nan'))
std_vr  = clean_stats.get('vtheta_times_r_std',  float('nan'))
if not np.isnan(mean_vr):
    ax_a.text(0.97, 0.50,
              rf'$\langle v_\theta \cdot r \rangle$' + '\n' +
              rf'$= {mean_vr:.4f} \pm {std_vr:.4f}$',
              transform=ax_a.transAxes, ha='right', va='center', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='gray', alpha=0.9))

ax_a.set_xlabel(r'$r$')
ax_a.set_ylabel(r'$v_\theta(r)$')
ax_a.set_title(r'(a) Single vortex: $v_\theta = 1/r$')
ax_a.set_xlim(0, 20)
ax_a.set_ylim(0, 1.4)
ax_a.legend(loc='upper right', framealpha=0.9)
ax_a.grid(True, alpha=0.3)


# ============================================================
#  PANEL (b) — Amplitude-phase decoupling (RADIAL PROFILE)
# ============================================================
print("Building Panel (b): radial profile of |Psi| before/after...")

run3b = d3['runs']['run_3B_same_sign']
snap_init  = run3b['field_snapshots']['initial']
snap_final = run3b['field_snapshots']['final']

amp_init  = np.array(snap_init['psi_abs'],  dtype=np.float32)
amp_final = np.array(snap_final['psi_abs'], dtype=np.float32)
extent    = snap_init['extent']

# Radial profiles
r_init,  prof_init  = radial_profile(amp_init,  extent, n_bins=60, r_max=15)
r_final, prof_final = radial_profile(amp_final, extent, n_bins=60, r_max=15)

v_vac = d3['params']['v_vac']

ax_b.plot(r_init, prof_init, '-', color=COL_INITIAL, linewidth=2.2,
          label='initial (two cores)', zorder=3)
ax_b.plot(r_final, prof_final, '-', color=COL_FINAL, linewidth=2.2,
          label='final (smoothed)', zorder=3)
ax_b.axhline(v_vac, color=COL_VVAC, linestyle='--', linewidth=1.2,
             label=rf'$v_{{\rm vac}} = \sqrt{{\lambda/\alpha}}$',
             zorder=2)

# Mark vortex radial position (|x| = 2 → r = 2)
ax_b.axvline(2.0, color='gray', linestyle=':', linewidth=0.9, alpha=0.7,
             zorder=1)
ax_b.text(2.0, v_vac * 0.15, r'  $r_{\rm vortex}=2$',
          color='gray', fontsize=9, ha='left', va='center')

# Winding preservation annotation
windings = run3b.get('windings', {}).get('final', {})
w_r10 = windings.get('w_global_r10', None)
if w_r10 is not None:
    ax_b.text(0.97, 0.35,
              rf'$w(r{{=}}10) = {w_r10:+.4f}$' + '\n' +
              r'phase preserved',
              transform=ax_b.transAxes, ha='right', va='center', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='gray', alpha=0.9))

ax_b.set_xlabel(r'$r$')
ax_b.set_ylabel(r'$\langle |\Psi| \rangle(r)$')
ax_b.set_title(r'(b) Amplitude-phase decoupling')
ax_b.set_xlim(0, 15)
ax_b.set_ylim(0, v_vac * 1.18)
ax_b.legend(loc='lower right', framealpha=0.9)
ax_b.grid(True, alpha=0.3)


# ============================================================
#  PANEL (c) — Distribution-dependent rotation curves
# ============================================================
print("Building Panel (c)...")

r5u, v5u = clean_profile(run5u['rotation_curve'])
r5i, v5i = clean_profile(run5i['rotation_curve'])

ax_c.plot(r5u, v5u, '-', color=COL_UNIFORM, linewidth=2,
          label='uniform disk')
ax_c.plot(r5i, v5i, '-', color=COL_INVRAD, linewidth=2,
          label=r'$\sigma \propto 1/r$')

R_gal = 10.0
ax_c.axvline(R_gal, color='gray', linestyle=':', linewidth=1.2,
             label=rf'$R_{{\rm gal}}={R_gal}$')

flat_u = extract_flatness(run5u, r5u, v5u)
flat_i = extract_flatness(run5i, r5i, v5i)

if flat_u is not None and flat_i is not None:
    ax_c.text(0.97, 0.12,
              (rf'$\mathcal{{F}}_{{\rm uniform}} = {flat_u:.2f}$' + '\n' +
               rf'$\mathcal{{F}}_{{\rm inv}} = {flat_i:.2f}$'),
              transform=ax_c.transAxes, ha='right', va='bottom', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='gray', alpha=0.9))

ax_c.axvspan(3.0, 9.0, alpha=0.08, color=COL_CLEAN)

ax_c.set_xlabel(r'$r$')
ax_c.set_ylabel(r'$v_\theta(r)$')
ax_c.set_title(r'(c) Distribution determines rotation curve')
ax_c.set_xlim(0, 20)
ax_c.set_ylim(-0.1, 1.4)
ax_c.legend(loc='upper right', framealpha=0.9)
ax_c.grid(True, alpha=0.3)


# ============================================================
#  PANEL (d) — Vortex positions
# ============================================================
print("Building Panel (d)...")

pos_u = np.array(run5u['positions'])
pos_i = np.array(run5i['positions'])

ax_d.scatter(pos_u[:, 0], pos_u[:, 1], s=110,
             c=COL_UNIFORM, marker='o', alpha=0.85,
             edgecolors='black', linewidths=1.0,
             label='uniform disk', zorder=3)
ax_d.scatter(pos_i[:, 0], pos_i[:, 1], s=110,
             c=COL_INVRAD, marker='^', alpha=0.85,
             edgecolors='black', linewidths=1.0,
             label=r'$\sigma \propto 1/r$', zorder=3)

circle = Circle((0, 0), R_gal, fill=False,
                color='gray', linestyle='--', linewidth=1.3, zorder=2)
ax_d.add_patch(circle)
ax_d.text(R_gal * 0.72, R_gal * 0.72,
          rf'$R_{{\rm gal}}={R_gal}$',
          color='gray', fontsize=9, ha='center')

ax_d.plot(0, 0, '+', color='black', markersize=10,
          markeredgewidth=1.5, zorder=4)

ax_d.set_xlim(-12, 12)
ax_d.set_ylim(-12, 12)
ax_d.set_aspect('equal')
ax_d.set_xlabel(r'$x$')
ax_d.set_ylabel(r'$y$')
ax_d.set_title(r'(d) Vortex distributions ($N = 10$)')
ax_d.legend(loc='upper right', framealpha=0.9)
ax_d.grid(True, alpha=0.3)


# ============================================================
#  SAVE
# ============================================================
plt.tight_layout(pad=1.2)
fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close(fig)

print(f"\n{'='*62}")
print(f"  Figure saved: {OUTPUT_PNG}  ({DPI} dpi)")
print(f"{'='*62}")
print("Done.")
