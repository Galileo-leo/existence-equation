"""
analyze_v5_proton.py  v1.1
===========================
Proton structure analysis on a frozen ED complex-amplitude field (no time evolution).

Loads v5_frozen_state.npz (3D psi on a periodic grid) and post-processes it to find and
characterize proton-like defects: Y-shaped junctions of vortex cores, confinement along
flux tubes, and effective charge from phase winding.

Pipeline (measurement only):
  (0) Plaquette curl of the phase field -> approximate vortex-density / winding map.
  (1) Enumerate triplets of nearby + or - winding centers as triangular candidates;
      score ~ equilateral quality (shortest/longest side).
  (2) For top candidates passing a flux-tube gate: 3D ROI energy split into gradient-of-
      amplitude, phase-gradient (kinetic), double-well, and quartic terms.
  (3) Circle integral of phase gradient vs radius -> winding number vs R; amplitude
      floor rejects radii that cut through the core (unreliable phase).
  (4) Opening angles between the three legs from the triplet center (Y-junction ~ 120 deg).
  (5) Flux tube gate: average 2D |grad psi|^2 along rays from center; require >=2 legs
      above background to keep tube-like structures.

v1.0: Initial version.
v1.1: MAX_SIDE 15→6, amp_cut 0.3*mean, FT gate (>=2 legs), dense R (2.3..5.5, step 0.15).

Jae-Ahn Shin & Kael · 2026-03-23
"""

import numpy as np
import json
import time as timer
from itertools import combinations

VERSION = "1.1"
print("=" * 65)
print(f"  Proton Candidate Analysis  v{VERSION}")
print("=" * 65)

print("\n  Loading frozen state...")
data = np.load('v5_frozen_state.npz')
psi_np = data['psi']
dx = float(data['dx'])
L = float(data['L'])
a = float(data['a'])
N = int(data['N'])
best_z = int(data['best_z'])

lam = 1.0
alpha = 1.0
v_vac = 1.0

print(f"  N={N}, dx={dx:.4f}, L={L:.2f}, a={a:.3f}")
print(f"  Best z={best_z}")

phase_np = np.angle(psi_np)
amp_np = np.abs(psi_np)


def wrap_ph(dp):
    return (dp + np.pi) % (2*np.pi) - np.pi


def measure_winding_circle(phase_slice, cx, cy, radius, n_sample=120):
    angles = np.linspace(0, 2*np.pi, n_sample, endpoint=False)
    phases = []
    N_s = phase_slice.shape[0]
    for ang in angles:
        ix = int(round(cx + radius * np.cos(ang))) % N_s
        iy = int(round(cy + radius * np.sin(ang))) % N_s
        phases.append(phase_slice[iy, ix])
    phases = np.array(phases)
    dp = np.diff(phases)
    dp = np.where(dp > np.pi, dp - 2*np.pi, dp)
    dp = np.where(dp < -np.pi, dp + 2*np.pi, dp)
    last_dp = phases[0] - phases[-1]
    last_dp = last_dp - 2*np.pi * round(last_dp / (2*np.pi))
    return float((np.sum(dp) + last_dp) / (2*np.pi))


def circle_amp_min(amp_slice, cx, cy, radius, n_sample=120):
    """Minimum |Psi| on a circle. Used to skip core-crossing radii."""
    angles = np.linspace(0, 2*np.pi, n_sample, endpoint=False)
    N_s = amp_slice.shape[0]
    amps = []
    for ang in angles:
        ix = int(round(cx + radius * np.cos(ang))) % N_s
        iy = int(round(cy + radius * np.sin(ang))) % N_s
        amps.append(amp_slice[iy, ix])
    return float(np.min(amps))


# ══════════════════════════════════════════════════════════════
# (0) Winding map: discrete curl of phase links (plaquette sum) / 2pi
#     Large positive/negative values locate vortex-like phase singularities.
# ══════════════════════════════════════════════════════════════

print("\n  Computing winding map...")
phase_slice = np.angle(psi_np[best_z])
dp_x = np.angle(np.exp(1j * (np.roll(phase_slice, -1, axis=1) - phase_slice)))
dp_y = np.angle(np.exp(1j * (np.roll(phase_slice, -1, axis=0) - phase_slice)))
curl = dp_x + np.roll(dp_y, -1, axis=1) - np.roll(dp_x, -1, axis=0) - dp_y
winding_map = curl / (2 * np.pi)

yp, xp = np.where(winding_map > 0.5)
yn, xn = np.where(winding_map < -0.5)
print(f"  z={best_z}: +{len(xp)} / -{len(xn)} = {len(xp)+len(xn)}")


# ══════════════════════════════════════════════════════════════
# (1) Proton candidate search: same-sign vortex triplets with periodic min-image distances
# ══════════════════════════════════════════════════════════════

print(f"\n{'─'*65}")
print("  (1) Proton Candidate Search")
print(f"{'─'*65}")

SCORE_THRESHOLD = 0.5
MAX_SIDE = 6.0
MAX_RATIO = 3.0
MIN_SIDE = 0.5

wall_start = timer.time()

all_candidates = []

for sign_label, xs, ys in [('+', xp, yp), ('-', xn, yn)]:
    if len(xs) < 3:
        continue

    pos = np.column_stack([xs.astype(float), ys.astype(float)])
    pos_phys = pos * dx - L/2
    n_vort = len(pos)

    close_pairs = {}
    for i in range(n_vort):
        close_pairs[i] = []

    for i in range(n_vort):
        for j in range(i+1, n_vort):
            diff = pos[i] - pos[j]
            diff = diff - N * np.round(diff / N)
            d = np.sqrt(np.sum(diff**2)) * dx
            if MIN_SIDE < d < MAX_SIDE:
                close_pairs[i].append(j)
                close_pairs[j].append(i)

    checked = set()
    for i in range(n_vort):
        for j in close_pairs[i]:
            if j <= i:
                continue
            for k in close_pairs[j]:
                if k <= j:
                    continue
                if k not in close_pairs[i]:
                    continue

                tri_key = (i, j, k)
                if tri_key in checked:
                    continue
                checked.add(tri_key)

                p1, p2, p3 = pos_phys[i], pos_phys[j], pos_phys[k]
                d12 = np.sqrt(np.sum((p1-p2)**2))
                d23 = np.sqrt(np.sum((p2-p3)**2))
                d13 = np.sqrt(np.sum((p1-p3)**2))
                sides = sorted([d12, d23, d13])

                if sides[0] < MIN_SIDE or sides[2] > MAX_SIDE:
                    continue
                if sides[2] / max(sides[0], 0.01) > MAX_RATIO:
                    continue

                score = sides[0] / max(sides[2], 0.01)
                center = (p1 + p2 + p3) / 3.0
                center_grid = ((center + L/2) / dx).astype(int) % N

                all_candidates.append({
                    'sign': sign_label,
                    'positions': [p1.tolist(), p2.tolist(), p3.tolist()],
                    'positions_grid': [pos[i].tolist(), pos[j].tolist(), pos[k].tolist()],
                    'sides': [round(s, 3) for s in sides],
                    'center': center.tolist(),
                    'center_grid': center_grid.tolist(),
                    'score': round(score, 4),
                })

elapsed = timer.time() - wall_start
print(f"  Found {len(all_candidates)} triplet candidates ({elapsed:.1f}s)")

all_candidates.sort(key=lambda x: -x['score'])
good = [c for c in all_candidates if c['score'] >= SCORE_THRESHOLD]
print(f"  Score > {SCORE_THRESHOLD}: {len(good)} candidates")

n_show = min(20, len(good))
print(f"\n  Top {n_show} candidates:")
print(f"  {'#':>3s}  {'sign':>4s}  {'score':>6s}  {'sides':>24s}  {'center':>16s}")
print(f"  {'─'*60}")
for i, c in enumerate(good[:n_show]):
    sides_str = "[%.1f, %.1f, %.1f]" % (c['sides'][0], c['sides'][1], c['sides'][2])
    center_str = "(%.1f, %.1f)" % (c['center'][0], c['center'][1])
    print(f"  {i+1:>3d}  {c['sign']:>4s}  {c['score']:>6.3f}  {sides_str:>24s}  {center_str:>16s}")


# ══════════════════════════════════════════════════════════════
# (2)~(5) Fine-grained measurement per top candidate
# ══════════════════════════════════════════════════════════════

n_analyze = min(10, len(good))
print(f"\n{'─'*65}")
print(f"  (2)~(5) Detailed Analysis: Top {n_analyze}")
print(f"{'─'*65}")

phase_slice_z = phase_np[best_z]
amp_slice_z = amp_np[best_z]
amp_mean_slice = float(np.mean(amp_slice_z))
AMP_CUT = 0.3 * amp_mean_slice

psi_slice = psi_np[best_z]
grad_x = (np.roll(psi_slice, -1, axis=1) - np.roll(psi_slice, 1, axis=1)) / (2*dx)
grad_y = (np.roll(psi_slice, -1, axis=0) - np.roll(psi_slice, 1, axis=0)) / (2*dx)
E_grad_2d = 0.5 * (np.abs(grad_x)**2 + np.abs(grad_y)**2)
bg_E = float(np.median(E_grad_2d))

print(f"  amp_mean={amp_mean_slice:.2f}, amp_cut={AMP_CUT:.2f}")

R_phys_list = list(np.arange(2.3, 5.5, 0.15))

MIN_FT_LEGS = 2
FT_THRESHOLD = 1.5

def angle_between(a, b):
    cos_ang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    cos_ang = np.clip(cos_ang, -1, 1)
    return np.degrees(np.arccos(cos_ang))

detailed_results = []
n_skipped_ft = 0
n_scanned = 0

for rank, cand in enumerate(good):
    if len(detailed_results) >= n_analyze:
        break
    n_scanned += 1

    p1, p2, p3 = [np.array(p) for p in cand['positions']]
    center = np.array(cand['center'])

    # ── (5) Flux tube gate: energy density along legs vs median background (must pass first)
    ft_ratios = []
    for pi_phys in cand['positions']:
        pi = np.array(pi_phys)
        direction = pi - center
        d_leg = np.linalg.norm(direction)
        if d_leg < 0.1:
            ft_ratios.append(0.0)
            continue
        samples = []
        for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pt = center + frac * direction
            ix = int((pt[0] + L/2) / dx) % N
            iy = int((pt[1] + L/2) / dx) % N
            samples.append(E_grad_2d[iy, ix])
        line_mean = float(np.mean(samples))
        ft_r = line_mean / max(bg_E, 1e-10)
        ft_ratios.append(round(ft_r, 3))

    n_strong_legs = sum(1 for ft in ft_ratios if ft > FT_THRESHOLD)
    if n_strong_legs < MIN_FT_LEGS:
        n_skipped_ft += 1
        continue

    real_rank = len(detailed_results) + 1
    print(f"\n  --- Proton #{real_rank} (sign={cand['sign']}, score={cand['score']:.3f}) ---")

    cx_phys, cy_phys = cand['center']
    cx_grid = int((cx_phys + L/2) / dx) % N
    cy_grid = int((cy_phys + L/2) / dx) % N

    mean_ft = np.mean(ft_ratios)
    print(f"    FT: {ft_ratios}, mean={mean_ft:.2f}, strong_legs={n_strong_legs}")

    result = {'rank': real_rank, 'sign': cand['sign'], 'score': cand['score'],
              'sides': cand['sides'], 'center': cand['center'],
              'ft_ratios': ft_ratios, 'mean_ft': round(float(mean_ft), 3),
              'n_strong_legs': n_strong_legs}

    # ── (4) Y-junction geometry: three angles at center; ideal 3-fold ~ 120 deg each
    v1 = p1 - center
    v2 = p2 - center
    v3 = p3 - center

    ang12 = angle_between(v1, v2)
    ang23 = angle_between(v2, v3)
    ang13 = angle_between(v1, v3)
    angles = sorted([ang12, ang23, ang13])
    dev_120 = max(abs(a - 120.0) for a in angles)

    result['angles'] = [round(a, 1) for a in angles]
    result['dev_120'] = round(dev_120, 1)

    print(f"    Angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}] deg")
    print(f"    120 dev: {dev_120:.1f} deg")

    # ── (3) Topological charge: phase winding on circles; skip R if min |psi| on circle is low
    winding_vs_r = []
    plateau_val = None
    plateau_count = 0

    for rp in R_phys_list:
        R = int(round(rp / dx))
        if R >= N//2 - 2 or R < 3:
            continue
        n_samp = max(int(8*R), 100)
        amp_min = circle_amp_min(amp_slice_z, cx_grid, cy_grid, R, n_samp)
        if amp_min < AMP_CUT:
            continue

        w = measure_winding_circle(phase_slice_z, cx_grid, cy_grid, R, n_sample=n_samp)
        winding_vs_r.append({'R': R, 'R_phys': round(R*dx, 2), 'winding': round(w, 4),
                             'amp_min': round(amp_min, 3)})

        w_int = round(w)
        if abs(w - w_int) < 0.2 and abs(w_int) >= 1:
            plateau_count += 1
            plateau_val = w_int

    result['winding_vs_r'] = winding_vs_r
    result['plateau'] = plateau_val
    result['plateau_count'] = plateau_count

    w_str = ", ".join([f"R={wr['R']}:{wr['winding']:+.3f}" for wr in winding_vs_r])
    print(f"    Winding: {w_str}")
    if plateau_val is not None:
        print(f"    Plateau: Q={plateau_val} ({plateau_count} hits)")
    else:
        print(f"    Plateau: none")

    # ── (2) ED energy density integrated in a 3D cube (ROI_BOX) centered on candidate
    ROI_BOX = 10
    cz = best_z
    cy_r = cy_grid
    cx_r = cx_grid
    mid = N // 2

    shifted = np.roll(psi_np, (mid-cz, mid-cy_r, mid-cx_r), axis=(0,1,2))
    roi = shifted[mid-ROI_BOX:mid+ROI_BOX,
                  mid-ROI_BOX:mid+ROI_BOX,
                  mid-ROI_BOX:mid+ROI_BOX]

    roi_amp = np.abs(roi)
    roi_phase = np.angle(roi)
    dV = dx**3
    n_cells = roi.size

    E_grad_A = 0.0
    E_phase = 0.0
    for ax in range(3):
        dA = (np.roll(roi_amp,-1,axis=ax) - np.roll(roi_amp,1,axis=ax)) / (2*dx)
        E_grad_A += float(np.sum(0.5 * dA**2)) * dV
        dphi = wrap_ph(np.roll(roi_phase,-1,axis=ax) - roi_phase) / dx
        E_phase += float(np.sum(0.5 * roi_amp**2 * dphi**2)) * dV

    E_rest = float(np.sum(0.5 * lam * roi_amp**2)) * dV
    E_cond = float(np.sum(0.25 * alpha * roi_amp**4)) * dV
    E_rest -= (0.5 * lam * v_vac**2) * n_cells * dV
    E_cond -= (0.25 * alpha * v_vac**4) * n_cells * dV
    E_total = E_grad_A + E_phase + E_rest + E_cond
    phase_frac = E_phase / max(abs(E_total), 1e-10) * 100

    result['energy'] = {
        'E_grad_A': round(E_grad_A, 2),
        'E_phase': round(E_phase, 2),
        'E_total': round(E_total, 2),
        'phase_frac': round(phase_frac, 1),
        'mean_amp': round(float(np.mean(roi_amp)), 4),
    }

    print(f"    Energy: phase={phase_frac:.1f}%, E_total={E_total:.1f}, <A>={np.mean(roi_amp):.2f}")

    detailed_results.append(result)

print(f"\n  FT gate: scanned {n_scanned}, skipped {n_skipped_ft} (no flux tube), passed {len(detailed_results)}")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("  SUMMARY TABLE")
print(f"{'='*65}")
print(f"  {'#':>3s}  {'sign':>4s}  {'score':>6s}  {'dev120':>7s}  "
      f"{'Q':>4s}  {'phase%':>7s}  {'FT':>6s}  {'sides':>20s}")
print(f"  {'─'*62}")

for r in detailed_results:
    q_str = "%+d" % r['plateau'] if r['plateau'] is not None else "?"
    print(f"  {r['rank']:>3d}  {r['sign']:>4s}  {r['score']:>6.3f}  "
          f"{r['dev_120']:>6.1f}   {q_str:>4s}  "
          f"{r['energy']['phase_frac']:>6.1f}%  {r['mean_ft']:>6.2f}  "
          f"[{r['sides'][0]:.1f},{r['sides'][1]:.1f},{r['sides'][2]:.1f}]")

if detailed_results:
    best = detailed_results[0]
    print(f"\n  Best: #{best['rank']} sign={best['sign']} score={best['score']:.3f}")
    print(f"    Sides={best['sides']}, Angles={best['angles']}, dev={best['dev_120']:.1f}")
    print(f"    Q={best['plateau']}, phase={best['energy']['phase_frac']:.1f}%, FT={best['mean_ft']:.2f}")

output = {
    'params': {'N': N, 'dx': dx, 'L': L, 'a': a, 'best_z': best_z},
    'n_total': len(all_candidates),
    'n_good': len(good),
    'top20': good[:20],
    'detailed': detailed_results,
}
with open('v5_proton_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: v5_proton_analysis.json")

elapsed = timer.time() - wall_start
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*65}")
