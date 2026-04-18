"""
ed_v5_hydrogen_multi.py — Multi-electron binding on multiple proton candidates
================================================================================

Loads frozen ED field (v5_frozen_state.npz) and places one electron (winding=-1)
around each POSITIVE-sign proton candidate, then evolves the full system.

If H2-like covalent bonding emerges spontaneously, nearby hydrogen atoms will
share electrons. We don't force anything — the equation decides.

Proton candidates (sign=+, from analyze_v5_proton.py output):
  P_A: (-9.6, -17.6)  score=0.995  (original Proton #2)
  P_B: (-18.1, -5.0)  score=0.995  (original Proton #3)
  P_C: (-17.1, -8.3)  score=0.991  (original Proton #4)
  P_D: (-8.7,  +3.7)  score=0.988  (original Proton #9)
  P_E: (-2.3,  -9.0)  score=0.986  (original Proton #14)

Each gets one electron at offset 2.5 phys → 5 hydrogen atoms total.
N_STEPS=30000, tracking all 5 electron-proton pairs.

Author: Jae-Ahn Shin
"""

import numpy as np
import time as timer
import json

try:
    import cupy as cp
    xp = cp
    USE_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
except ImportError:
    xp = np
    USE_GPU = False
    print("CPU mode")

def to_np(arr):
    return cp.asnumpy(arr) if USE_GPU else arr

def to_xp(arr):
    return xp.asarray(arr) if USE_GPU else arr

def laplacian_3d(f, dx):
    return (xp.roll(f,1,0) + xp.roll(f,-1,0) +
            xp.roll(f,1,1) + xp.roll(f,-1,1) +
            xp.roll(f,1,2) + xp.roll(f,-1,2) - 6*f) / dx**2

def count_vortices_2d(psi_slice):
    phase = xp.angle(psi_slice)
    dp_x = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=1) - phase)))
    dp_y = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=0) - phase)))
    curl = dp_x + xp.roll(dp_y,-1,axis=1) - xp.roll(dp_x,-1,axis=0) - dp_y
    return curl / (2*xp.pi)


VERSION = "1.0"
print(f"\n{'='*65}")
print(f"  ED v5 H2 Experiment: Multi-Hydrogen  v{VERSION}")
print(f"  5 proton candidates (sign=+) × 1 electron each")
print(f"{'='*65}")

# ══════════════════════════════════════════════════════════════
# Load frozen state
# ══════════════════════════════════════════════════════════════

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
c = 1.0
dt = 0.25 * dx / (c * np.sqrt(3))

print(f"  N={N}, dx={dx:.4f}, L={L:.2f}, a={a:.3f}")
print(f"  dt={dt:.6f}, best_z={best_z}")

# ══════════════════════════════════════════════════════════════
# Proton candidates (sign=+) from analyze_v5_proton.py
# ══════════════════════════════════════════════════════════════

PROTONS = [
    {'label': 'P_A', 'x': -9.6,  'y': -17.6, 'score': 0.995, 'orig': '#2'},
    {'label': 'P_B', 'x': -18.1, 'y': -5.0,  'score': 0.995, 'orig': '#3'},
    {'label': 'P_C', 'x': -17.1, 'y': -8.3,  'score': 0.991, 'orig': '#4'},
    {'label': 'P_D', 'x': -8.7,  'y': 3.7,   'score': 0.988, 'orig': '#9'},
    {'label': 'P_E', 'x': -2.3,  'y': -9.0,  'score': 0.986, 'orig': '#14'},
]

E_OFFSET_PHYS = 2.5
E_WINDING = -1
XI = 0.5

print(f"\n  Proton-Electron pairs (offset={E_OFFSET_PHYS} phys, e_wind={E_WINDING}):")
print(f"  {'Label':6s} {'Proton (x,y)':>16s} {'score':>6s} {'Electron (x,y)':>18s} {'Grid (px,py)':>14s}")
print(f"  {'-'*65}")

coords = np.arange(N) * dx - L/2
Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')

pairs = []
for p in PROTONS:
    px, py = p['x'], p['y']
    pc_ix = int((px + L/2) / dx) % N
    pc_iy = int((py + L/2) / dx) % N
    pc_iz = best_z

    e_offset_grid = int(round(E_OFFSET_PHYS / dx))
    e_ix = (pc_ix + e_offset_grid) % N
    e_iy = pc_iy
    ex_phys = e_ix * dx - L/2
    ey_phys = e_iy * dx - L/2

    pairs.append({
        'label': p['label'], 'orig': p['orig'], 'score': p['score'],
        'px': px, 'py': py, 'pc_ix': pc_ix, 'pc_iy': pc_iy, 'pc_iz': pc_iz,
        'ex_phys': ex_phys, 'ey_phys': ey_phys,
        'e_ix': e_ix, 'e_iy': e_iy,
    })

    print(f"  {p['label']:6s} ({px:+6.1f}, {py:+6.1f})  {p['score']:.3f}  "
          f"({ex_phys:+6.2f}, {ey_phys:+6.2f})  ({pc_ix:3d},{pc_iy:3d})")

# ══════════════════════════════════════════════════════════════
# Insert electrons — one vortex per proton candidate
# ══════════════════════════════════════════════════════════════

print(f"\n  Inserting {len(pairs)} electrons...")

for pair in pairs:
    ex_phys = pair['ex_phys']
    ey_phys = pair['ey_phys']

    r_e = np.sqrt((X - ex_phys)**2 + (Y - ey_phys)**2)
    theta_e = np.arctan2(Y - ey_phys, X - ex_phys)
    f_e = r_e / np.sqrt(r_e**2 + XI**2)
    vort_e = f_e * np.exp(1j * E_WINDING * theta_e)
    vort_e /= (np.abs(vort_e) + 1e-10)

    psi_np = psi_np * vort_e

del Z, Y, X, r_e, theta_e, f_e, vort_e

psi_np *= v_vac / (np.abs(psi_np) + 1e-10)

psi = to_xp(psi_np)
psi_dot = xp.zeros_like(psi)
del psi_np

w_check = count_vortices_2d(psi[best_z])
w_np_c = to_np(w_check)
vp0 = int(np.sum(w_np_c > 0.5))
vm0 = int(np.sum(w_np_c < -0.5))
print(f"  After insertion: +{vp0} / -{vm0} (added {len(pairs)} electrons)")

# ══════════════════════════════════════════════════════════════
# Distance table between protons
# ══════════════════════════════════════════════════════════════

print(f"\n  Inter-proton distances:")
print(f"  {'':6s}", end='')
for p in pairs:
    print(f" {p['label']:>6s}", end='')
print()
for i, pi in enumerate(pairs):
    print(f"  {pi['label']:6s}", end='')
    for j, pj in enumerate(pairs):
        if j <= i:
            print(f" {'---':>6s}", end='')
        else:
            d = np.sqrt((pi['px'] - pj['px'])**2 + (pi['py'] - pj['py'])**2)
            print(f" {d:6.1f}", end='')
    print()

# ══════════════════════════════════════════════════════════════
# Evolve + track all electron-proton pairs
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  Evolving {len(pairs)} H atoms, N_STEPS=30000")
print(f"{'='*65}\n")

N_STEPS = 30000
MEASURE_INTERVAL = 1
PRINT_INTERVAL = 500

proton_positions = np.array([[p['pc_ix'], p['pc_iy']] for p in pairs])
proton_labels = [p['label'] for p in pairs]
n_protons = len(pairs)

for pair in pairs:
    pair['track'] = []
    pair['bound_count'] = 0
    pair['total_checks'] = 0

global_track = []

wall_start = timer.time()

for step in range(N_STEPS):
    lap = laplacian_3d(psi, dx)
    accel = c**2 * (lap + lam*psi - alpha*xp.abs(psi)**2*psi)
    psi_dot += accel * dt
    psi += psi_dot * dt

    if step % MEASURE_INTERVAL == 0:
        psi_now = to_np(psi)
        w_now = count_vortices_2d(to_xp(psi_now)[best_z])
        w_np = to_np(w_now)

        # Locate + vortices (proton cores) and - vortices (electrons)
        ey_neg, ex_neg = np.where(w_np < -0.5)
        ey_pos, ex_pos = np.where(w_np > 0.5)
        ex_all, ey_all = ex_neg, ey_neg
        n_electrons = len(ex_all)

        step_record = {'step': step, 'n_electrons': n_electrons,
                       'n_plus': len(ex_pos), 'n_minus': n_electrons}

        # Track proton positions (nearest + vortex cluster center)
        if len(ex_pos) > 0:
            ppos = np.column_stack([ex_pos, ey_pos])
            for pair in pairs:
                diff_p = ppos - np.array([pair['pc_ix'], pair['pc_iy']])
                diff_p = diff_p - N * np.round(diff_p / N)
                d_p = np.sqrt(np.sum(diff_p**2, axis=1)) * dx
                nearby = d_p < 4.0
                if np.sum(nearby) >= 3:
                    cx = np.mean(diff_p[nearby, 0]) * dx + pair['px']
                    cy = np.mean(diff_p[nearby, 1]) * dx + pair['py']
                    pair.setdefault('proton_track', []).append({
                        'step': step,
                        'px': round(float(cx), 4),
                        'py': round(float(cy), 4),
                        'n_nearby': int(np.sum(nearby)),
                    })
                else:
                    pair.setdefault('proton_track', []).append({
                        'step': step, 'px': pair['px'], 'py': pair['py'],
                        'n_nearby': int(np.sum(nearby)),
                    })

        # For each electron vortex, compute distance to ALL protons
        electron_assignments = {}
        if n_electrons > 0:
            epos = np.column_stack([ex_all, ey_all])
            for ei in range(n_electrons):
                diff_all = epos[ei] - proton_positions
                diff_all = diff_all - N * np.round(diff_all / N)
                dists_all = np.sqrt(np.sum(diff_all**2, axis=1)) * dx
                nearest_idx = np.argmin(dists_all)
                nearest_label = proton_labels[nearest_idx]
                nearest_dist = float(dists_all[nearest_idx])

                sorted_idx = np.argsort(dists_all)
                second_idx = sorted_idx[1]
                second_label = proton_labels[second_idx]
                second_dist = float(dists_all[second_idx])

                if nearest_label not in electron_assignments:
                    electron_assignments[nearest_label] = []
                electron_assignments[nearest_label].append({
                    'ei': ei, 'd1': nearest_dist,
                    'd2_label': second_label, 'd2': second_dist,
                })

        # H2 detection: any proton pair sharing electrons?
        sharing_pairs = []
        for li in range(n_protons):
            for lj in range(li+1, n_protons):
                la, lb = proton_labels[li], proton_labels[lj]
                ea = electron_assignments.get(la, [])
                eb = electron_assignments.get(lb, [])
                shared = 0
                for e in ea:
                    if e['d2_label'] == lb and e['d2'] < 8.0:
                        shared += 1
                for e in eb:
                    if e['d2_label'] == la and e['d2'] < 8.0:
                        shared += 1
                if shared > 0:
                    sharing_pairs.append((la, lb, shared))

        step_record['sharing'] = [(a, b, s) for a, b, s in sharing_pairs]

        # Per-proton tracking (nearest electron)
        for pair in pairs:
            pc_ix = pair['pc_ix']
            pc_iy = pair['pc_iy']
            label = pair['label']

            e_found = False
            e_dist = 999.0
            rx_phys, ry_phys = 0.0, 0.0
            nearest_p = label
            second_d = 999.0

            if n_electrons > 0:
                diff = epos - np.array([pc_ix, pc_iy])
                diff = diff - N * np.round(diff / N)
                dists = np.sqrt(np.sum(diff**2, axis=1)) * dx

                ci = np.argmin(dists)
                e_dist = float(dists[ci])
                rx_phys = float(diff[ci, 0]) * dx
                ry_phys = float(diff[ci, 1]) * dx
                e_found = True

                # Which proton is this electron actually closest to?
                e_grid = epos[ci]
                diff_to_all = e_grid - proton_positions
                diff_to_all = diff_to_all - N * np.round(diff_to_all / N)
                d_to_all = np.sqrt(np.sum(diff_to_all**2, axis=1)) * dx
                true_nearest = np.argmin(d_to_all)
                nearest_p = proton_labels[true_nearest]

                sorted_p = np.argsort(d_to_all)
                second_d = float(d_to_all[sorted_p[1]])

            # Electron absolute position
            e_abs_x = pair['px'] + rx_phys
            e_abs_y = pair['py'] + ry_phys

            # H atom center of mass (proton + electron midpoint)
            pt = pair.get('proton_track', [])
            if pt and len(pt) > 0 and pt[-1]['step'] == step:
                p_now_x = pt[-1]['px']
                p_now_y = pt[-1]['py']
            else:
                p_now_x = pair['px']
                p_now_y = pair['py']
            cm_x = (p_now_x + e_abs_x) / 2.0
            cm_y = (p_now_y + e_abs_y) / 2.0

            pair['track'].append({
                'step': step, 'found': e_found,
                'dist': round(e_dist, 4),
                'rx': round(rx_phys, 4), 'ry': round(ry_phys, 4),
                'e_x': round(e_abs_x, 4), 'e_y': round(e_abs_y, 4),
                'cm_x': round(cm_x, 4), 'cm_y': round(cm_y, 4),
                'nearest_p': nearest_p,
                'second_d': round(second_d, 4),
            })

            pair['total_checks'] += 1
            if e_dist < 10.0:
                pair['bound_count'] += 1

        global_track.append(step_record)

    if step % PRINT_INTERVAL == 0:
        el = timer.time() - wall_start
        line = f"  Step {step:5d} ({el:6.0f}s) |"
        for pair in pairs:
            t = pair['track'][-1] if pair['track'] else {'dist': 999}
            d = t['dist']
            np_label = t.get('nearest_p', '?')
            tag = '*' if np_label != pair['label'] else ' '
            line += f" {pair['label']}:{d:5.1f}{tag}"
        # Show sharing
        rec = global_track[-1] if global_track else {}
        sh = rec.get('sharing', [])
        if sh:
            line += f" | H2:{','.join(f'{a}-{b}' for a,b,_ in sh)}"
        print(line)

# ══════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════

elapsed = timer.time() - wall_start

print(f"\n{'='*65}")
print(f"  RESULTS — Multi-Hydrogen ({len(pairs)} atoms)")
print(f"{'='*65}")

psi_f = to_np(psi)
w_f = count_vortices_2d(to_xp(psi_f)[best_z])
w_fn = to_np(w_f)
vp_final = int(np.sum(w_fn > 0.5))
vm_final = int(np.sum(w_fn < -0.5))
print(f"  Vortices: +{vp_final}/{-vm_final} (initial: +{vp0}/{-vm0})")

print(f"\n  ── Per-atom binding ──")
print(f"  {'Label':6s} {'Orig':>5s} {'Bound%':>7s} {'Mean_d':>7s} {'Min_d':>7s} {'Max_d':>7s} {'Std_d':>7s}")
print(f"  {'-'*52}")

results = {}
for pair in pairs:
    dists = np.array([t['dist'] for t in pair['track'] if t['found']])
    bp = pair['bound_count'] / max(pair['total_checks'], 1) * 100

    if len(dists) > 0:
        mean_d = np.mean(dists)
        min_d = np.min(dists)
        max_d = np.max(dists)
        std_d = np.std(dists)
    else:
        mean_d = min_d = max_d = std_d = 999.0

    print(f"  {pair['label']:6s} {pair['orig']:>5s} {bp:6.1f}% {mean_d:7.2f} {min_d:7.2f} {max_d:7.2f} {std_d:7.2f}")

    results[pair['label']] = {
        'proton': {'x': pair['px'], 'y': pair['py'], 'score': pair['score']},
        'bound_pct': round(bp, 2),
        'mean_dist': round(float(mean_d), 4),
        'min_dist': round(float(min_d), 4),
        'max_dist': round(float(max_d), 4),
        'std_dist': round(float(std_d), 4),
    }

# ── H2 DETECTION ──
print(f"\n  ══ H2 DETECTION ══")

# 1) Electron loyalty: how often does nearest electron actually belong to this proton?
print(f"\n  1) Electron loyalty (nearest_p == self):")
for pair in pairs:
    loyal = sum(1 for t in pair['track'] if t.get('nearest_p') == pair['label'])
    total = len(pair['track'])
    pct = loyal / max(total, 1) * 100
    print(f"     {pair['label']}: {pct:.1f}% loyal ({loyal}/{total})")
    results[pair['label']]['loyalty_pct'] = round(pct, 2)

# 2) Electron hopping: count how many times nearest_p changes
print(f"\n  2) Electron hopping (nearest proton switches):")
for pair in pairs:
    hops = 0
    prev = None
    for t in pair['track']:
        cur = t.get('nearest_p', pair['label'])
        if prev is not None and cur != prev:
            hops += 1
        prev = cur
    print(f"     {pair['label']}: {hops} hops")
    results[pair['label']]['hops'] = hops

# 3) Sharing frequency: for each proton pair, how often were electrons shared?
print(f"\n  3) Sharing frequency (electron's 2nd-nearest proton < 8 phys):")
sharing_count = {}
for rec in global_track:
    for a, b, s in rec.get('sharing', []):
        key = f"{a}-{b}"
        sharing_count[key] = sharing_count.get(key, 0) + 1

total_steps = len(global_track)
if sharing_count:
    for key, cnt in sorted(sharing_count.items(), key=lambda x: -x[1]):
        pct = cnt / max(total_steps, 1) * 100
        print(f"     {key}: {pct:.1f}% of time ({cnt}/{total_steps})")
else:
    print(f"     No sharing detected")

# 4) Final electron positions relative to all protons
print(f"\n  4) Final state — electron-proton distances:")
for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):
        pi, pj = pairs[i], pairs[j]
        d_proton = np.sqrt((pi['px'] - pj['px'])**2 + (pi['py'] - pj['py'])**2)
        ti = pi['track'][-1] if pi['track'] else None
        tj = pj['track'][-1] if pj['track'] else None
        if ti and tj and ti['found'] and tj['found']:
            ei = np.array([ti['rx'] + pi['px'], ti['ry'] + pi['py']])
            ej = np.array([tj['rx'] + pj['px'], tj['ry'] + pj['py']])
            d_electron = np.sqrt(np.sum((ei - ej)**2))
            tag = " ★★ H2?" if d_proton < 10 and d_electron < d_proton else ""
            print(f"     {pi['label']}-{pj['label']}: p_d={d_proton:.1f}, e_d={d_electron:.1f}{tag}")

# 5) H atom center-of-mass motion (Kepler check)
print(f"\n  5) H atom center-of-mass drift:")
for pair in pairs:
    cms = [(t['cm_x'], t['cm_y']) for t in pair['track'] if t['found']]
    if len(cms) > 100:
        cms = np.array(cms)
        drift = np.sqrt((cms[-1, 0] - cms[0, 0])**2 + (cms[-1, 1] - cms[0, 1])**2)
        std_x = np.std(cms[:, 0])
        std_y = np.std(cms[:, 1])
        print(f"     {pair['label']}: drift={drift:.2f}, std=({std_x:.2f},{std_y:.2f})")

# 6) Inter-H-atom CM distances over time (Kepler orbit detection)
print(f"\n  6) H-atom pair CM trajectory (Kepler candidates):")
for i in range(len(pairs)):
    for j in range(i+1, len(pairs)):
        pi, pj = pairs[i], pairs[j]
        d_proton = np.sqrt((pi['px'] - pj['px'])**2 + (pi['py'] - pj['py'])**2)
        if d_proton > 20.0:
            continue
        cmi = np.array([(t['cm_x'], t['cm_y']) for t in pi['track'] if t['found']])
        cmj = np.array([(t['cm_x'], t['cm_y']) for t in pj['track'] if t['found']])
        min_len = min(len(cmi), len(cmj))
        if min_len > 100:
            cmi, cmj = cmi[:min_len], cmj[:min_len]
            d_cm = np.sqrt(np.sum((cmi - cmj)**2, axis=1))
            rel_x = cmi[:, 0] - cmj[:, 0]
            rel_y = cmi[:, 1] - cmj[:, 1]
            angles = np.unwrap(np.arctan2(rel_y, rel_x))
            total_angle = np.degrees(angles[-1] - angles[0])
            print(f"     {pi['label']}-{pj['label']}: p_d={d_proton:.1f}, "
                  f"cm_d: {np.mean(d_cm):.1f}+/-{np.std(d_cm):.1f}, "
                  f"angle_sweep={total_angle:+.0f}°")
            if abs(total_angle) > 90:
                print(f"       ★ Significant orbital motion!")

# 7) Verdict
print(f"\n  ══ VERDICT ══")
total_meas = len(global_track)
h2_candidates = []
for key, cnt in sharing_count.items():
    pct = cnt / max(total_meas, 1) * 100
    if pct > 10.0:
        h2_candidates.append((key, pct))

if h2_candidates:
    print(f"  H2 CANDIDATES FOUND:")
    for key, pct in h2_candidates:
        print(f"    {key}: sharing {pct:.1f}% of simulation time")
else:
    all_loyal = all(
        sum(1 for t in p['track'] if t.get('nearest_p') == p['label']) / max(len(p['track']), 1) > 0.9
        for p in pairs
    )
    if all_loyal:
        print(f"  No H2 formed — all electrons stayed loyal to their proton")
        print(f"  (5 independent H atoms, no covalent bonding)")
    else:
        print(f"  Inconclusive — some electron migration but no sustained sharing")

with open('v5_h2_multi_track.json', 'w') as f:
    json.dump({
        'version': VERSION,
        'n_protons': len(pairs),
        'n_steps': N_STEPS,
        'e_offset': E_OFFSET_PHYS,
        'e_winding': E_WINDING,
        'vortices_initial': {'plus': vp0, 'minus': vm0},
        'vortices_final': {'plus': vp_final, 'minus': vm_final},
        'results': results,
        'sharing_counts': {k: v for k, v in sharing_count.items()},
        'h2_candidates': [(k, round(p, 2)) for k, p in h2_candidates],
        'pairs': [
            {
                'label': p['label'], 'orig': p['orig'],
                'proton_phys': [p['px'], p['py']],
                'proton_track': p.get('proton_track', []),
                'track': p['track'],
            }
            for p in pairs
        ],
    }, f, indent=2)

print(f"\n  Saved: v5_h2_multi_track.json")
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*65}")
