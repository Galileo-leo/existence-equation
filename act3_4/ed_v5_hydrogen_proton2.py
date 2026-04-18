"""
Secondary proton analysis script (ED v5 hydrogen binding) — v1.3
====================================================================

Loads a frozen ED (complex scalar field) checkpoint and studies interaction with
proton candidate #2: the proton grid site is fixed; over time the script tracks
electron–proton separation (binding proxy), angular offset (rx, ry, theta),
phase winding around the tracked electron, bound-state fraction, and radial
oscillation count. Final output includes vortex counts and optional proton-
vs-electron winding comparison.

Pipeline: ``v5_frozen_state.npz`` → multiply in an electron vortex at a fixed
offset from proton #2 → time evolution on a fixed grid (no expansion) → JSON
trajectory log.

Proton #2 (hardcoded target): center=(-3.5,-4.2), sign=+, score=0.992,
    sides=[2.8,2.8,2.8], Q=+1 (triplet).
Electron: winding=-1, offset 2.5 physical units (just outside triplet).

Version history
---------------
v1.3: N_STEPS=30000 (30k trajectory, ~25min) — tail statistics
v1.2: N_STEPS=10000 (10k trajectory, ~5min)
v1.1: MEASURE_INTERVAL=1 (3000pt trajectory), rx/ry physical coords, separate winding

Jae-Ahn Shin & Kael · 2026-03-23
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
    # Standard 7-point discrete Laplacian on a periodic cubic grid (dx spacing).
    return (xp.roll(f,1,0) + xp.roll(f,-1,0) +
            xp.roll(f,1,1) + xp.roll(f,-1,1) +
            xp.roll(f,1,2) + xp.roll(f,-1,2) - 6*f) / dx**2

def count_vortices_2d(psi_slice):
    # Vortex (winding) density in 2D from discrete curl of the phase gradient.
    phase = xp.angle(psi_slice)
    dp_x = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=1) - phase)))
    dp_y = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=0) - phase)))
    curl = dp_x + xp.roll(dp_y,-1,axis=1) - xp.roll(dp_x,-1,axis=0) - dp_y
    return curl / (2*xp.pi)

def measure_winding_circle(phase_slice, cx, cy, radius, n_sample=120):
    # Total phase winding number around (cx, cy) by integrating dφ on a circle (unwrapped).
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


VERSION = "1.3"

print(f"\n{'='*65}")
print(f"  ED v5 Phase 3: Hydrogen Binding  v{VERSION}")
print("  Proton #2 + Electron")
print(f"{'='*65}")

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
print(f"  dt={dt:.6f}")


# ══════════════════════════════════════════════════════════════
# Proton #2 — fixed spatial target (secondary-proton candidate from prior scan)
# ══════════════════════════════════════════════════════════════

PROTON_X_PHYS = -9.6
PROTON_Y_PHYS = -17.6
PROTON_SIGN = '+'
PROTON_Z = best_z

pc_ix = int((PROTON_X_PHYS + L/2) / dx) % N
pc_iy = int((PROTON_Y_PHYS + L/2) / dx) % N
pc_iz = PROTON_Z

print(f"\n  Proton #2 (hardcoded):")
print(f"    Physical: ({PROTON_X_PHYS}, {PROTON_Y_PHYS})")
print(f"    Grid: ({pc_ix}, {pc_iy}, {pc_iz})")
print(f"    Sign: {PROTON_SIGN}, Score: 0.995")
print(f"    Sides: [3.0, 3.0, 3.0]")


# ══════════════════════════════════════════════════════════════
# Electron — vortex seed: phase winding and radial profile at fixed offset
# ══════════════════════════════════════════════════════════════

E_OFFSET_PHYS = 2.5
e_offset_grid = int(round(E_OFFSET_PHYS / dx))

e_winding = -1
e_label = "-"

e_ix = (pc_ix + e_offset_grid) % N
e_iy = pc_iy
e_iz = pc_iz
e_x_phys = e_ix * dx - L/2
e_y_phys = e_iy * dx - L/2

print(f"\n  Electron [{e_label}] (hardcoded):")
print(f"    Physical: ({e_x_phys:.2f}, {e_y_phys:.2f})")
print(f"    Grid: ({e_ix}, {e_iy}, {e_iz})")
print(f"    Offset: {E_OFFSET_PHYS} phys = {e_offset_grid} grid")

xi = 0.5
coords = np.arange(N) * dx - L/2
Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')

# Multiply frozen ψ by a 2D vortex factor (core scale xi), then renormalize |ψ|→v_vac.
r_e = np.sqrt((X - e_x_phys)**2 + (Y - e_y_phys)**2)
theta_e = np.arctan2(Y - e_y_phys, X - e_x_phys)
f_e = r_e / np.sqrt(r_e**2 + xi**2)
vort_e = f_e * np.exp(1j * e_winding * theta_e)
vort_e /= (np.abs(vort_e) + 1e-10)

psi_np = psi_np * vort_e
psi_np *= v_vac / (np.abs(psi_np) + 1e-10)

psi = to_xp(psi_np)
psi_dot = xp.zeros_like(psi)
del Z, Y, X, r_e, theta_e, f_e, vort_e

w_check = count_vortices_2d(psi[best_z])
w_np_c = to_np(w_check)
vp0 = int(np.sum(w_np_c > 0.5))
vm0 = int(np.sum(w_np_c < -0.5))
print(f"\n  After insertion: +{vp0} / -{vm0}")


# ══════════════════════════════════════════════════════════════
# Evolve + track — leapfrog step on nonlinear wave equation (c²∇² − λψ − α|ψ|²ψ).
# Fixed dx; binding proxy = periodic distance from proton #2 to nearest −1 vortex.
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print("  Evolving (no expansion, dx fixed)...")
print(f"{'='*65}\n")

N_STEPS = 30000
MEASURE_INTERVAL = 1
WINDING_INTERVAL = 10
PRINT_INTERVAL = 200

E_WINDING_R_PHYS = 2.3
R_e_grid = int(round(E_WINDING_R_PHYS / dx))

track_log = []
bound_count = 0
total_checks = 0
n_osc = 0
prev_dist = None
prev_dir = None

wall_start = timer.time()

for step in range(N_STEPS):
    # ψ̈ = c²(∇²ψ + λψ − α|ψ|²ψ); ψ += ψ̇ dt, ψ̇ += accel dt.
    lap = laplacian_3d(psi, dx)
    accel = c**2 * (lap + lam*psi - alpha*xp.abs(psi)**2*psi)
    psi_dot += accel * dt
    psi += psi_dot * dt

    if step % MEASURE_INTERVAL == 0:
        psi_now = to_np(psi)

        w_now = count_vortices_2d(to_xp(psi_now)[best_z])
        w_np = to_np(w_now)

        # Locate electron as negative-winding vortex cores on the best_z slice.
        ey, ex = np.where(w_np < -0.5)

        e_found = False
        e_dist = 999.0
        e_cix, e_ciy = 0, 0
        e_th = 0.0
        rx_phys, ry_phys = 0.0, 0.0
        w_e = 0.0

        if len(ex) > 0:
            epos = np.column_stack([ex, ey])
            diff = epos - np.array([pc_ix, pc_iy])
            diff = diff - N * np.round(diff / N)
            dists = np.sqrt(np.sum(diff**2, axis=1)) * dx

            ci = np.argmin(dists)
            e_dist = float(dists[ci])
            e_cix = int(epos[ci, 0])
            e_ciy = int(epos[ci, 1])
            e_found = True

            rx_phys = float(diff[ci, 0]) * dx
            ry_phys = float(diff[ci, 1]) * dx
            e_th = float(np.arctan2(ry_phys, rx_phys))

            if step % WINDING_INTERVAL == 0 and R_e_grid < N//2 - 2:
                phase_sl = np.angle(psi_now[best_z])
                w_e = measure_winding_circle(phase_sl, e_cix, e_ciy, R_e_grid,
                                              n_sample=max(int(8*R_e_grid), 60))

        track_log.append({
            'step': step, 'found': e_found,
            'ix': int(e_cix), 'iy': int(e_ciy),
            'rx': round(rx_phys, 4), 'ry': round(ry_phys, 4),
            'dist': round(e_dist, 4),
            'theta': round(e_th, 4),
            'w_e': round(w_e, 4),
        })

        total_checks += 1
        if e_dist < 10.0:
            bound_count += 1

        if prev_dist is not None:
            d = "out" if e_dist > prev_dist else "in"
            if prev_dir is not None and d != prev_dir:
                n_osc += 1
            prev_dir = d
        prev_dist = e_dist

        if step % PRINT_INTERVAL == 0:
            el = timer.time() - wall_start
            bp = bound_count / max(total_checks, 1) * 100
            s = "OK" if e_found else "X"
            print(f"  {step:5d} | {s} r={e_dist:6.2f} | "
                  f"th={np.degrees(e_th):+6.1f} | "
                  f"osc={n_osc:4d} | bd={bp:5.1f}% | "
                  f"w={w_e:+.3f} | {el:.0f}s")


# ══════════════════════════════════════════════════════════════
# Results — aggregate binding stats, vortex balance, optional P/E winding ratio
# ══════════════════════════════════════════════════════════════

elapsed = timer.time() - wall_start
dists = np.array([t['dist'] for t in track_log if t['found']])
bp = bound_count / max(total_checks, 1) * 100

print(f"\n{'='*65}")
print(f"  RESULTS")
print(f"{'='*65}")
print(f"  Proton: #2 (score=0.992, sides=[2.8,2.8,2.8], Q=+1)")
print(f"  Electron: winding=-1, offset={E_OFFSET_PHYS} phys")
print(f"  Steps: {N_STEPS}, dt={dt:.6f}")
print(f"")
print(f"  Bound (<10 phys): {bp:.1f}% ({bound_count}/{total_checks})")
print(f"  Oscillations: {n_osc}")
if len(dists) > 0:
    print(f"  Distance: mean={np.mean(dists):.2f}, min={np.min(dists):.2f}, "
          f"max={np.max(dists):.2f}, std={np.std(dists):.2f}")

psi_f = to_np(psi)
w_f = count_vortices_2d(to_xp(psi_f)[best_z])
w_fn = to_np(w_f)
vp = int(np.sum(w_fn > 0.5))
vm = int(np.sum(w_fn < -0.5))
print(f"  V+/V-: +{vp}/{-vm} (initial: +{vp0}/{-vm0})")
print(f"  Conservation: {'PERFECT' if vp==vm else 'BROKEN'}")

valid_we = [t['w_e'] for t in track_log if t['found'] and abs(t['w_e']) > 0.3]
if valid_we:
    mean_we = np.mean(np.abs(valid_we))
    phase_final = np.angle(psi_f)
    phase_sl = phase_final[best_z]
    wp_list = []
    for R in range(16, 30):
        if R < N//2 - 2:
            wp = measure_winding_circle(phase_sl, pc_ix, pc_iy, R,
                                         n_sample=max(int(8*R), 60))
            if abs(wp) > 0.3:
                wp_list.append(abs(wp))
    if wp_list:
        mean_wp = np.mean(wp_list)
        pe = mean_wp / mean_we
        print(f"\n  P winding: {mean_wp:.4f} (R=16~29)")
        print(f"  E winding: {mean_we:.4f}")
        print(f"  P/E ratio = {pe:.4f}")

print(f"\n  {'-'*45}")
print(f"  {'':18s}  {'v21(128)':>10s}  {'v5(256)':>10s}")
print(f"  {'Proton score':18s}  {'0.911':>10s}  {'0.992':>10s}")
print(f"  {'120 dev':18s}  {'5.2':>10s}  {'0.5':>10s}")
print(f"  {'Bound':18s}  {'100%':>10s}  {f'{bp:.1f}%':>10s}")
print(f"  {'Oscillations':18s}  {'464':>10s}  {f'{n_osc}':>10s}")
if len(dists) > 0:
    print(f"  {'Mean dist':18s}  {'1.51':>10s}  {f'{np.mean(dists):.2f}':>10s}")

with open('v5_hydrogen_proton2_track.json', 'w') as f:
    json.dump({
        'proton': {
            'candidate': 2,
            'center_phys': [PROTON_X_PHYS, PROTON_Y_PHYS],
            'grid': [pc_ix, pc_iy, pc_iz],
            'sign': PROTON_SIGN, 'score': 0.995,
            'sides': [3.0, 3.0, 3.0], 'Q': 1,
        },
        'electron': {
            'offset_phys': E_OFFSET_PHYS,
            'offset_grid': e_offset_grid,
            'winding': e_winding,
            'grid': [e_ix, e_iy, e_iz],
        },
        'summary': {
            'bound_pct': round(bp, 2),
            'oscillations': n_osc,
            'mean_dist': round(float(np.mean(dists)), 4) if len(dists) > 0 else None,
            'std_dist': round(float(np.std(dists)), 4) if len(dists) > 0 else None,
        },
        'track': track_log,
    }, f, indent=2)
print(f"\n  Saved: v5_hydrogen_proton2_track.json")
print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*65}")
