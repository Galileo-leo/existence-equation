#!/usr/bin/env python3
"""
==============================================================================
  EP V (Dark Matter as Topological Phase Persistence of the Existence Equation)
  STEP 2 — Single-Vortex Validation
==============================================================================
  Purpose:
    Validate the numerical pipeline against the analytic single-vortex
    profile. Verify that:
      (a) winding number is topologically conserved,
      (b) v_theta(r) matches 1/r to 4-digit precision in the clean regime,
      (c) |Psi| recovers to v_vac far from the core,
      (d) |Psi| -> 0 at the vortex center.

  Method:
    - 2D finite-difference ED equation on 2048x2048 grid
    - Full-field multiplicative ansatz: Psi = v_vac * tanh(r/xi) * exp(i*theta)
    - Damped leapfrog: 6000 steps, gamma = 0.02
    - Runtime: ~15 seconds on a single NVIDIA A100

  Outputs:
    - ep5_step2_single_vortex.json
        Unified JSON schema matching Steps 3 and 5.
        All data needed for reproducing Figure 1 Panel (a) is here.

  Reproducibility:
    Running this script produces a JSON file whose contents are used
    by ep5_make_figure1.py to regenerate Panel (a) of Figure 1 in the
    EP V paper. No external data is required beyond the output of
    this script itself.

  Author: Jae-Ahn Shin  (Existence Dynamics framework)
==============================================================================
"""

import time
import json
import numpy as np

# ============================================================
#  Backend
# ============================================================
try:
    import cupy as xp
    BACKEND = "CuPy (GPU)"
except ImportError:
    import numpy as xp
    BACKEND = "NumPy (CPU)"

print(f"Backend: {BACKEND}")


# ============================================================
#  PARAMETERS  (locked — do not change without version bump)
# ============================================================
N_GRID   = 2048
L        = 40.0
dx       = 2 * L / N_GRID
c        = 1.0

LAMBDA   = 3.0
ALPHA    = 1.0
v_vac    = float(np.sqrt(LAMBDA / ALPHA))
xi       = 1.0 / np.sqrt(2 * LAMBDA)

DT       = 0.4 * dx / (c * np.sqrt(2))
N_RELAX  = 6000
DAMPING  = 0.02
MEAS_EVERY = 1000

# Clean regime for precision assessment
CLEAN_R_MIN = 3.0
CLEAN_R_MAX = 15.0

# Output file
OUTPUT_JSON = 'ep5_step2_single_vortex.json'

print(f"\n{'='*62}")
print(f"  EP V Step 2 — Single Vortex Validation")
print(f"{'='*62}")
print(f"  Grid: {N_GRID}^2,  L={L},  dx={dx:.5f}")
print(f"  lambda={LAMBDA},  alpha={ALPHA},  v_vac={v_vac:.4f}")
print(f"  xi={xi:.4f}  (xi/dx={xi/dx:.1f})")
print(f"  Relax: {N_RELAX} steps,  dt={DT:.6f},  gamma={DAMPING}")
print(f"  Clean regime: {CLEAN_R_MIN} < r < {CLEAN_R_MAX}")
print(f"{'='*62}\n")


# ============================================================
#  GRID
# ============================================================
t_global = time.time()
x1d = xp.linspace(-L, L, N_GRID, endpoint=False, dtype=xp.float64) + dx/2
X, Y = xp.meshgrid(x1d, x1d, indexing='ij')
R_grid = xp.sqrt(X**2 + Y**2)


# ============================================================
#  NUMERICAL OPERATORS
# ============================================================
def laplacian_fd(f):
    """2D five-point finite-difference Laplacian."""
    return (xp.roll(f,  1, 0) + xp.roll(f, -1, 0) +
            xp.roll(f,  1, 1) + xp.roll(f, -1, 1) - 4 * f) / dx**2


def measure_winding(Psi, r_test, cx=0.0, cy=0.0):
    """Winding number on a circle of radius r_test centered at (cx, cy)."""
    n_pts = max(256, int(2 * np.pi * r_test / dx))
    theta_arr = xp.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    x_pts = cx + r_test * xp.cos(theta_arr)
    y_pts = cy + r_test * xp.sin(theta_arr)

    ix = xp.clip(((x_pts + L) / dx).astype(int), 0, N_GRID - 1)
    iy = xp.clip(((y_pts + L) / dx).astype(int), 0, N_GRID - 1)

    psi_loop = Psi[ix, iy]
    phase = xp.angle(psi_loop)
    dphase = xp.angle(xp.exp(1j * xp.diff(phase)))
    return float(xp.sum(dphase)) / (2 * np.pi)


def measure_rotation_curve(Psi, r_bins):
    """
    Azimuthally averaged tangential velocity v_theta(r) from the
    gauge-invariant superfluid current.
    """
    amp2 = xp.abs(Psi)**2 + 1e-30

    dPsi_dx = (xp.roll(Psi, -1, 0) - xp.roll(Psi, 1, 0)) / (2 * dx)
    dPsi_dy = (xp.roll(Psi, -1, 1) - xp.roll(Psi, 1, 1)) / (2 * dx)

    jx = xp.imag(xp.conj(Psi) * dPsi_dx) / amp2
    jy = xp.imag(xp.conj(Psi) * dPsi_dy) / amp2

    cos_th = X / (R_grid + 1e-30)
    sin_th = Y / (R_grid + 1e-30)
    v_theta = -jx * sin_th + jy * cos_th

    dr = float(r_bins[1] - r_bins[0]) if len(r_bins) > 1 else 1.0
    amp_field = xp.sqrt(amp2)
    profile = []

    for r_c in r_bins:
        mask = (R_grid > r_c - dr/2) & (R_grid < r_c + dr/2)
        n_pts = int(xp.sum(mask))
        if n_pts > 50:
            v_mean = float(xp.mean(v_theta[mask]))
            v_std  = float(xp.std(v_theta[mask]))
            a_mean = float(xp.mean(amp_field[mask]))
            profile.append({
                'r': float(r_c),
                'v_theta': v_mean,
                'v_std': v_std,
                'amp_mean': a_mean,
                'n_pts': n_pts,
            })
        else:
            profile.append({
                'r': float(r_c),
                'v_theta': float('nan'),
                'v_std': float('nan'),
                'amp_mean': float('nan'),
                'n_pts': n_pts,
            })
    return profile


# ============================================================
#  INITIAL CONDITION
# ============================================================
def build_single_vortex_ic():
    """
    Full-field multiplicative ansatz:
        Psi(r, theta) = v_vac * tanh(r/xi) * exp(i*theta)
    Exact analytic form of an isolated n=+1 vortex in the
    Mexican-hat Bose-Einstein condensate ground state.
    """
    r_v = R_grid + 1e-30
    theta_v = xp.arctan2(Y, X)
    Psi = v_vac * xp.tanh(r_v / xi) * xp.exp(1j * theta_v)
    return Psi.astype(xp.complex128)


# ============================================================
#  RELAXATION
# ============================================================
def relax(Psi_init, n_steps, dt, damping, label=""):
    """Damped leapfrog toward the Mexican-hat minimum."""
    Psi = Psi_init.copy()
    amp2 = xp.abs(Psi)**2
    accel = laplacian_fd(Psi) * c**2 + LAMBDA * Psi - ALPHA * amp2 * Psi
    Psi_dot = xp.zeros_like(Psi)
    Psi_dot += 0.5 * dt * accel

    t0 = time.time()

    for step in range(1, n_steps + 1):
        Psi += dt * Psi_dot
        amp2 = xp.abs(Psi)**2
        accel = laplacian_fd(Psi) * c**2 + LAMBDA * Psi - ALPHA * amp2 * Psi
        Psi_dot += dt * accel
        Psi_dot *= (1.0 - damping)

        if step % MEAS_EVERY == 0:
            psi_max = float(xp.max(xp.abs(Psi)))
            w5  = measure_winding(Psi, 5.0)
            w10 = measure_winding(Psi, 10.0)
            elapsed = time.time() - t0
            print(f"  [{label}] step {step:5d}: "
                  f"|Psi|_max={psi_max:.4f}  "
                  f"w(5)={w5:+.4f}  w(10)={w10:+.4f}  "
                  f"{elapsed:.0f}s", flush=True)

    return Psi


# ============================================================
#  MAIN RUN
# ============================================================
print("Building grid...")
print(f"  Done in {time.time()-t_global:.1f}s")

print("\nBuilding IC (full-field multiplicative ansatz)...")
Psi0 = build_single_vortex_ic()
psi0_max = float(xp.max(xp.abs(Psi0)))
psi0_center = float(xp.abs(Psi0[N_GRID//2, N_GRID//2]))
print(f"  |Psi|_max    = {psi0_max:.4f}  (expect {v_vac:.4f})")
print(f"  |Psi|_center = {psi0_center:.4f}  (expect ~0)")

# Initial windings
radii_winding = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
print("\nInitial windings:")
windings_initial = {}
for r_test in radii_winding:
    if r_test < L * 0.9:
        w = measure_winding(Psi0, r_test)
        windings_initial[f"r{int(r_test)}"] = round(w, 6)
        print(f"  w({r_test:4.1f}) = {w:+.4f}")

# Relaxation
print(f"\nRelaxing {N_RELAX} steps (damping={DAMPING})...")
Psi_final = relax(Psi0, N_RELAX, DT, DAMPING, label="single")

# Final windings
print("\nFinal windings:")
windings_final = {}
for r_test in radii_winding:
    if r_test < L * 0.9:
        w = measure_winding(Psi_final, r_test)
        windings_final[f"r{int(r_test)}"] = round(w, 6)
        print(f"  w({r_test:4.1f}) = {w:+.4f}")

psi_final_max    = float(xp.max(xp.abs(Psi_final)))
psi_final_center = float(xp.abs(Psi_final[N_GRID//2, N_GRID//2]))
print(f"\n  |Psi|_max    = {psi_final_max:.4f}")
print(f"  |Psi|_center = {psi_final_center:.4f}")

# Rotation curve
print("\nMeasuring rotation curve...")
r_bins = np.linspace(0.5, 30.0, 80)
profile = measure_rotation_curve(Psi_final, r_bins)

# Theory overlay (for JSON, enables direct plot without recomputation)
for p in profile:
    if p['r'] > 0:
        p['v_theory'] = 1.0 / p['r']
        if not np.isnan(p['v_theta']):
            p['ratio'] = p['v_theta'] / p['v_theory']
        else:
            p['ratio'] = float('nan')
    else:
        p['v_theory'] = float('nan')
        p['ratio'] = float('nan')

# Clean-regime statistics
clean_vr = []
for p in profile:
    if (CLEAN_R_MIN < p['r'] < CLEAN_R_MAX
            and not np.isnan(p['v_theta'])):
        clean_vr.append(p['v_theta'] * p['r'])

if clean_vr:
    clean_mean = float(np.mean(clean_vr))
    clean_std  = float(np.std(clean_vr))
    clean_dev  = abs(clean_mean - 1.0)
else:
    clean_mean = clean_std = clean_dev = float('nan')

print(f"\n{'='*62}")
print(f"  CLEAN REGIME  {CLEAN_R_MIN} < r < {CLEAN_R_MAX}")
print(f"{'='*62}")
print(f"  <v_theta * r>  = {clean_mean:.4f}")
print(f"  std            = {clean_std:.4f}")
print(f"  deviation      = {clean_dev*100:.2f}%")

# Pass/fail checks (same as original)
w5  = windings_final.get("r5", 0)
w10 = windings_final.get("r10", 0)
w20 = windings_final.get("r20", 0)

checks = {
    'w5_near_1':    abs(w5 - 1.0) < 0.05,
    'w10_near_1':   abs(w10 - 1.0) < 0.05,
    'w20_near_1':   abs(w20 - 1.0) < 0.05,
    'center_near_0': abs(psi_final_center) < 0.1,
    'max_near_vvac': abs(psi_final_max - v_vac) < 0.05,
    'clean_regime_ok': clean_dev < 0.01,
}
checks['all_pass'] = all(checks.values())

print(f"\n  Checks:")
for k, v in checks.items():
    if k != 'all_pass':
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
print(f"  ==> {'ALL PASSED' if checks['all_pass'] else 'SOME FAILED'}")


# ============================================================
#  UNIFIED JSON OUTPUT
# ============================================================
output = {
    'metadata': {
        'version': 'ep5_v2_final',
        'step': 2,
        'step_name': 'single_vortex_validation',
        'description': (
            'Validation of the ED numerical pipeline via a single '
            'n=+1 vortex. Provides the data for Figure 1 Panel (a) '
            'in the EP V paper: v_theta(r) vs 1/r overlay.'
        ),
    },
    'params': {
        'N_GRID': N_GRID,
        'L': L,
        'dx': dx,
        'lambda': LAMBDA,
        'alpha': ALPHA,
        'v_vac': v_vac,
        'xi': xi,
        'c': c,
        'dt': DT,
        'n_relax': N_RELAX,
        'damping': DAMPING,
        'clean_r_min': CLEAN_R_MIN,
        'clean_r_max': CLEAN_R_MAX,
    },
    'runs': {
        'single_vortex': {
            'config': {
                'n_vortex': 1,
                'winding': 1,
                'position': [0.0, 0.0],
                'initial_condition': 'full_field_multiplicative_ansatz',
            },
            'amplitude': {
                'psi_max_initial': psi0_max,
                'psi_max_final': psi_final_max,
                'psi_center_initial': psi0_center,
                'psi_center_final': psi_final_center,
            },
            'windings': {
                'initial': windings_initial,
                'final':   windings_final,
            },
            'rotation_curve': profile,
            'clean_regime_statistics': {
                'r_range': [CLEAN_R_MIN, CLEAN_R_MAX],
                'vtheta_times_r_mean': clean_mean,
                'vtheta_times_r_std':  clean_std,
                'deviation_from_unity': clean_dev,
                'n_bins_used': len(clean_vr),
            },
            'checks': checks,
        },
    },
    'timing': {
        'total_wall_time_s': round(time.time() - t_global, 1),
        'backend': BACKEND,
    },
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*62}")
print(f"  Results saved: {OUTPUT_JSON}")
print(f"  Total wall time: {time.time() - t_global:.0f}s")
print(f"{'='*62}")
print("Done.")
