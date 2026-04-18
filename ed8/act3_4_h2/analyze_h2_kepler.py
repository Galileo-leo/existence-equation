"""
analyze_h2_kepler.py — N-body Kepler analysis for all H-atom pairs
====================================================================
Focused analysis beyond P_B-P_C:
  1. Per-pair angle sweep in sliding windows (orbital periods)
  2. Radial oscillation spectrum (bound orbit signature)
  3. 3-body subsystem analysis (H2 molecule + satellite H)
  4. Proton drift tracking (do protons move toward each other?)
  5. Individual pair orbit plots

Author: Jae-Ahn Shin
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

print("=" * 65)
print("  H2 Kepler & N-body Analysis")
print("=" * 65)

with open('v5_h2_multi_track.json', 'r') as f:
    data = json.load(f)

pairs_data = data['pairs']
n_steps = data['n_steps']
labels = [p['label'] for p in pairs_data]

# ══════════════════════════════════════════════════════════════
# Build trajectory arrays
# ══════════════════════════════════════════════════════════════

traj = {}
for p in pairs_data:
    lab = p['label']
    track = p['track']
    traj[lab] = {
        'steps':  np.array([t['step'] for t in track]),
        'dist':   np.array([t['dist'] for t in track]),
        'e_x':    np.array([t.get('e_x', 0) for t in track]),
        'e_y':    np.array([t.get('e_y', 0) for t in track]),
        'cm_x':   np.array([t.get('cm_x', 0) for t in track]),
        'cm_y':   np.array([t.get('cm_y', 0) for t in track]),
        'nearest': [t.get('nearest_p', lab) for t in track],
        'px0':    p['proton_phys'][0],
        'py0':    p['proton_phys'][1],
    }
    pt = p.get('proton_track', [])
    if pt and isinstance(pt[0], dict):
        traj[lab]['p_x'] = np.array([r['px'] for r in pt])
        traj[lab]['p_y'] = np.array([r['py'] for r in pt])
    elif pt and isinstance(pt[0], (list, tuple)):
        traj[lab]['p_x'] = np.array([r[0] for r in pt])
        traj[lab]['p_y'] = np.array([r[1] for r in pt])
    else:
        traj[lab]['p_x'] = np.full(len(track), p['proton_phys'][0])
        traj[lab]['p_y'] = np.full(len(track), p['proton_phys'][1])

n_pts = min(len(traj[lab]['steps']) for lab in labels)
print(f"  Data points per atom: {n_pts}")
print(f"  Labels: {labels}")

# ══════════════════════════════════════════════════════════════
# 1. All-pairs detailed Kepler metrics
# ══════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print(f"  (1) ALL PAIRS — Kepler Metrics")
print(f"{'─'*65}")
print(f"  {'Pair':8s} {'p_dist':>6s} {'cm_d':>9s} {'sweep':>8s} {'|dθ/dt|':>8s} "
      f"{'Δr/r':>6s} {'T_est':>8s} {'Verdict':>12s}")
print(f"  {'-'*72}")

pair_results = []
for la, lb in combinations(labels, 2):
    ta, tb = traj[la], traj[lb]
    d_proton = np.sqrt((ta['px0'] - tb['px0'])**2 + (ta['py0'] - tb['py0'])**2)

    n = min(len(ta['cm_x']), len(tb['cm_x']), n_pts)
    rel_x = ta['cm_x'][:n] - tb['cm_x'][:n]
    rel_y = ta['cm_y'][:n] - tb['cm_y'][:n]
    d_cm = np.sqrt(rel_x**2 + rel_y**2)

    angles = np.unwrap(np.arctan2(rel_y, rel_x))
    total_sweep_deg = np.degrees(angles[-1] - angles[0])

    dtheta = np.diff(angles)
    omega_mean = np.mean(np.abs(dtheta))
    omega_std = np.std(np.abs(dtheta))

    d_mean = np.mean(d_cm)
    d_std = np.std(d_cm)
    radial_var = d_std / max(d_mean, 0.01)

    if omega_mean > 0:
        T_est = 2 * np.pi / omega_mean
    else:
        T_est = np.inf

    n_orbits = abs(total_sweep_deg) / 360.0

    verdict = ""
    if abs(total_sweep_deg) > 360 and d_proton < 5:
        verdict = "H2+ORBIT"
    elif abs(total_sweep_deg) > 180:
        verdict = "ORBITAL"
    elif abs(total_sweep_deg) > 45:
        verdict = "partial"
    else:
        verdict = "static"

    T_str = f"{T_est:.0f}" if T_est < 1e6 else "inf"
    print(f"  {la}-{lb:2s}  {d_proton:6.1f} {d_mean:5.1f}±{d_std:3.1f} "
          f"{total_sweep_deg:+7.0f}° {omega_mean:8.5f} "
          f"{radial_var:6.2f} {T_str:>8s} {verdict:>12s}")

    pair_results.append({
        'pair': f"{la}-{lb}", 'd_proton': d_proton,
        'sweep': total_sweep_deg, 'n_orbits': n_orbits,
        'omega_mean': omega_mean, 'omega_std': omega_std,
        'T_est': T_est, 'd_mean': d_mean, 'd_std': d_std,
        'radial_var': radial_var, 'verdict': verdict,
        'rel_x': rel_x, 'rel_y': rel_y, 'd_cm': d_cm,
        'angles': angles,
    })

# ══════════════════════════════════════════════════════════════
# 2. Proton drift analysis (do protons approach each other?)
# ══════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print(f"  (2) PROTON DRIFT — Are protons moving?")
print(f"{'─'*65}")

for la, lb in combinations(labels, 2):
    ta, tb = traj[la], traj[lb]
    n = min(len(ta['p_x']), len(tb['p_x']))
    if n < 10:
        continue
    d_pp = np.sqrt((ta['p_x'][:n] - tb['p_x'][:n])**2 +
                   (ta['p_y'][:n] - tb['p_y'][:n])**2)
    d_init = d_pp[0] if d_pp[0] > 0 else 1.0
    d_final = d_pp[-1]
    d_min = np.min(d_pp)
    d_max = np.max(d_pp)
    delta = d_final - d_init
    tag = ""
    if delta < -1.0:
        tag = " ← APPROACHING"
    elif delta > 1.0:
        tag = " → SEPARATING"
    print(f"  {la}-{lb}: init={d_init:.1f}, final={d_final:.1f}, "
          f"Δ={delta:+.1f}, range=[{d_min:.1f}, {d_max:.1f}]{tag}")

# ══════════════════════════════════════════════════════════════
# 3. Sliding-window angular velocity (detect orbital periods)
# ══════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print(f"  (3) ANGULAR VELOCITY — Sliding Window (top pairs)")
print(f"{'─'*65}")

orbital_pairs = [r for r in pair_results if abs(r['sweep']) > 45]
orbital_pairs.sort(key=lambda r: -abs(r['sweep']))

for pr in orbital_pairs[:5]:
    angles = pr['angles']
    win = min(2000, len(angles) // 5)
    if win < 10:
        continue
    omega_t = np.convolve(np.abs(np.diff(angles)),
                          np.ones(win)/win, mode='valid')
    omega_max = np.max(omega_t)
    omega_min = np.min(omega_t)
    omega_avg = np.mean(omega_t)

    d_cm = pr['d_cm']
    d_early = np.mean(d_cm[:len(d_cm)//3])
    d_mid = np.mean(d_cm[len(d_cm)//3:2*len(d_cm)//3])
    d_late = np.mean(d_cm[2*len(d_cm)//3:])

    print(f"\n  {pr['pair']} (sweep={pr['sweep']:+.0f}°):")
    print(f"    ω: mean={omega_avg:.5f}, min={omega_min:.5f}, max={omega_max:.5f}")
    print(f"    T_orbit ~ {2*np.pi/max(omega_avg,1e-10):.0f} steps")
    print(f"    CM dist: early={d_early:.1f}, mid={d_mid:.1f}, late={d_late:.1f}")
    if d_late < d_early * 0.8:
        print(f"    ★ INSPIRAL detected (d shrinking)")
    elif d_late > d_early * 1.2:
        print(f"    ↗ EXPANDING orbit")
    else:
        print(f"    ≈ STABLE orbit radius")

# ══════════════════════════════════════════════════════════════
# 4. 3-body subsystem: H2(B-C) + satellite
# ══════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print(f"  (4) 3-BODY: H₂(B-C) molecule + satellite atoms")
print(f"{'─'*65}")

bc_cm_x = (traj['P_B']['cm_x'][:n_pts] + traj['P_C']['cm_x'][:n_pts]) / 2
bc_cm_y = (traj['P_B']['cm_y'][:n_pts] + traj['P_C']['cm_y'][:n_pts]) / 2

for sat in ['P_A', 'P_D', 'P_E']:
    ts = traj[sat]
    n = min(len(ts['cm_x']), len(bc_cm_x))
    rel_x = ts['cm_x'][:n] - bc_cm_x[:n]
    rel_y = ts['cm_y'][:n] - bc_cm_y[:n]
    d = np.sqrt(rel_x**2 + rel_y**2)
    angles = np.unwrap(np.arctan2(rel_y, rel_x))
    sweep = np.degrees(angles[-1] - angles[0])
    d_mean = np.mean(d)

    tag = ""
    if abs(sweep) > 90:
        tag = " ★ ORBITAL around H₂!"
    elif abs(sweep) > 30:
        tag = " ~ partial sweep"

    print(f"  {sat} → H₂(B-C): d={d_mean:.1f}±{np.std(d):.1f}, "
          f"sweep={sweep:+.0f}°{tag}")

# ══════════════════════════════════════════════════════════════
# 5. Figure — per-pair orbital plots (top 6 pairs by |sweep|)
# ══════════════════════════════════════════════════════════════

pair_results.sort(key=lambda r: -abs(r['sweep']))
n_show = min(6, len(pair_results))

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('H-atom Pair Orbits — All Pairs (sorted by |sweep|)',
             fontsize=13, fontweight='bold')

for idx in range(n_show):
    ax = axes[idx // 3][idx % 3]
    pr = pair_results[idx]
    sub = slice(None, None, max(1, len(pr['rel_x']) // 3000))
    rx, ry = pr['rel_x'][sub], pr['rel_y'][sub]

    n_seg = len(rx)
    colors_seg = plt.cm.viridis(np.linspace(0, 1, n_seg))
    for k in range(n_seg - 1):
        ax.plot([rx[k], rx[k+1]], [ry[k], ry[k+1]],
                color=colors_seg[k], lw=0.6)
    ax.plot(rx[0], ry[0], 'go', ms=8, zorder=5, label='Start')
    ax.plot(rx[-1], ry[-1], 'r*', ms=12, zorder=5, label='End')
    ax.plot(0, 0, 'k+', ms=15, mew=2)

    ax.set_title(f'{pr["pair"]}  p_d={pr["d_proton"]:.1f}  '
                 f'sweep={pr["sweep"]:+.0f}°\n'
                 f'{pr["verdict"]}  T~{pr["T_est"]:.0f}  '
                 f'n_orb={pr["n_orbits"]:.1f}',
                 fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=7, loc='upper right')

for idx in range(n_show, 6):
    axes[idx // 3][idx % 3].axis('off')

plt.tight_layout()
plt.savefig('h2_kepler_orbits.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: h2_kepler_orbits.png")

# ══════════════════════════════════════════════════════════════
# 6. Radial oscillation figure for orbital pairs
# ══════════════════════════════════════════════════════════════

orbital = [r for r in pair_results if abs(r['sweep']) > 45]
if orbital:
    fig2, axes2 = plt.subplots(len(orbital), 2, figsize=(16, 4*len(orbital)))
    if len(orbital) == 1:
        axes2 = axes2.reshape(1, -1)
    fig2.suptitle('Radial Oscillation & Angular Velocity', fontsize=12, fontweight='bold')

    for i, pr in enumerate(orbital):
        sub = slice(None, None, max(1, len(pr['d_cm']) // 5000))
        steps_sub = np.arange(len(pr['d_cm']))[sub]

        ax_r = axes2[i, 0]
        ax_r.plot(steps_sub, pr['d_cm'][sub], 'k-', lw=0.5)
        ax_r.axhline(pr['d_mean'], color='r', ls='--', lw=1, alpha=0.5)
        ax_r.set_ylabel('CM distance')
        ax_r.set_title(f'{pr["pair"]}: radial oscillation (d̄={pr["d_mean"]:.1f})')
        ax_r.grid(True, alpha=0.2)

        ax_w = axes2[i, 1]
        dtheta = np.abs(np.diff(pr['angles']))
        win = min(500, len(dtheta) // 5)
        if win > 1:
            omega_smooth = np.convolve(dtheta, np.ones(win)/win, mode='valid')
            ax_w.plot(np.arange(len(omega_smooth)), omega_smooth, 'b-', lw=0.5)
        ax_w.set_ylabel('|dθ/dt| (rad/step)')
        ax_w.set_title(f'{pr["pair"]}: angular velocity')
        ax_w.grid(True, alpha=0.2)

    axes2[-1, 0].set_xlabel('Step')
    axes2[-1, 1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig('h2_radial_omega.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: h2_radial_omega.png")

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  KEPLER ANALYSIS SUMMARY")
print(f"{'='*65}")
print(f"\n  {'Pair':8s} {'|sweep|':>8s} {'n_orb':>6s} {'verdict':>12s}")
print(f"  {'-'*40}")
for pr in pair_results:
    sw = abs(pr['sweep'])
    if sw > 30:
        print(f"  {pr['pair']:8s} {sw:7.0f}° {pr['n_orbits']:6.1f} {pr['verdict']:>12s}")

print(f"\n  Total pairs analyzed: {len(pair_results)}")
print(f"  Orbital (>180°):      {sum(1 for r in pair_results if abs(r['sweep']) > 180)}")
print(f"  Partial (>45°):       {sum(1 for r in pair_results if 45 < abs(r['sweep']) <= 180)}")
print(f"  Static (<45°):        {sum(1 for r in pair_results if abs(r['sweep']) <= 45)}")
print(f"{'='*65}")
