#!/usr/bin/env python3
"""
==============================================================================
  EP V (Dark Matter as Topological Phase Persistence of the Existence Equation)
  STEP 5 — Distribution-Dependent Rotation Curves
==============================================================================
  Purpose:
    Demonstrate that the shape of the rotation curve is a direct function
    of the spatial distribution of topological charge, via the kinematic
    relation v_theta(r) = N_enc(r) / r.

    Two N=10 same-sign vortex distributions are compared:
      (Step 4)  Uniform disk:   sigma(r) = const,       R_gal = 10
      (Step 5)  Inverse radial: sigma(r) ~ 1/r,         R_gal = 10

    Both configurations carry the same total topological charge
    (N_enc(infty) = 10) but distribute it differently. The inverse_r
    distribution yields N_enc(r) ~ r in the inner disk and hence a
    FLAT rotation curve; the uniform distribution yields a rising-then-
    declining profile. Both converge to v_theta = 10/r in the far field.

  Method:
    - 2D finite-difference ED equation on 2048x2048 grid
    - Multi-vortex multiplicative ansatz
    - Minimal stabilization: 50 steps, gamma = 0.001
      (Short damping preserves the prescribed spatial distribution.
       Longer relaxation would let same-sign vortices migrate outward
       under Magnus repulsion, destroying the initial configuration.)
    - Runtime: ~2 seconds on a single NVIDIA A100

  Outputs:
    - ep5_step5_distributions.json
        Unified JSON schema matching Steps 2 and 3.
        Contains both uniform and inverse_r runs with full rotation
        curves, winding measurements, position lists, and flatness
        statistics. Provides data for Figure 1 Panels (c) and (d).

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
#  PARAMETERS  (locked)
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

# Minimal stabilization (key choice — preserves distribution)
N_RELAX  = 50
DAMPING  = 0.001
MEAS_EVERY = 10

# Vortex distribution parameters
R_GAL      = 10.0
MIN_SEP    = 4.0 * xi    # minimum pairwise separation
N_VORTEX   = 10
SEED       = 42

# Flatness measurement window
FLATNESS_R_MIN = 3.0
FLATNESS_R_MAX = 9.0

# Output file
OUTPUT_JSON = 'ep5_step5_distributions.json'

print(f"\n{'='*62}")
print(f"  EP V Step 5 — Distribution-Dependent Rotation Curves")
print(f"  Uniform disk vs Inverse-radial, N={N_VORTEX}")
print(f"{'='*62}")
print(f"  Grid: {N_GRID}^2,  L={L},  dx={dx:.5f}")
print(f"  lambda={LAMBDA},  alpha={ALPHA},  v_vac={v_vac:.4f}")
print(f"  xi={xi:.4f}  (xi/dx={xi/dx:.1f})")
print(f"  R_gal={R_GAL},  min_sep={MIN_SEP:.3f}")
print(f"  Relax: {N_RELAX} steps (minimal),  gamma={DAMPING}")
print(f"  Flatness window: {FLATNESS_R_MIN} <= r <= {FLATNESS_R_MAX}")
print(f"  Seed: {SEED}")
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
    """Azimuthally averaged v_theta(r) about the origin."""
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
#  VORTEX DISTRIBUTION GENERATORS
# ============================================================
def generate_positions(n_vortex, distribution, r_gal, min_sep, seed=42):
    """
    Generate n_vortex positions inside disk of radius r_gal
    with minimum pairwise separation enforced by rejection sampling.
    """
    rng = np.random.RandomState(seed)
    positions = []

    def too_close(x, y):
        for (px, py) in positions:
            if (x - px)**2 + (y - py)**2 < min_sep**2:
                return True
        return False

    r_min = max(2.0 * xi, 0.5)
    max_tries = 500

    for i in range(n_vortex):
        tries = 0
        while True:
            if distribution == 'uniform_disk':
                # Uniform on disk: r ~ sqrt(U)
                r = r_gal * np.sqrt(rng.uniform(0, 1))
            elif distribution == 'inverse_r':
                # sigma(r) ~ 1/r:  r ~ U on [r_min, r_gal]
                r = r_min + (r_gal - r_min) * rng.uniform(0, 1)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

            theta = rng.uniform(0, 2*np.pi)
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            if not too_close(x, y):
                positions.append((float(x), float(y)))
                break
            tries += 1
            if tries > max_tries:
                # Accept whatever we got after max_tries attempts
                positions.append((float(x), float(y)))
                break

    return np.array(positions)


def build_multi_vortex_ic(positions):
    """
    Multi-vortex multiplicative ansatz (all windings = +1):
        Psi = v_vac * Prod_k tanh(r_k / xi) * exp(i * sum_k theta_k)
    """
    amp = xp.ones_like(X, dtype=xp.float64) * v_vac
    phase = xp.zeros_like(X, dtype=xp.float64)

    for (x0, y0) in positions:
        dx1 = X - float(x0)
        dy1 = Y - float(y0)
        r1  = xp.sqrt(dx1**2 + dy1**2) + 1e-30
        th1 = xp.arctan2(dy1, dx1)

        amp   = amp * xp.tanh(r1 / xi)
        phase = phase + th1

    return (amp * xp.exp(1j * phase)).astype(xp.complex128)


def analytic_v_theta(positions, r_bins):
    """
    Analytic prediction from Eq. (v-kinematic):
        v_theta(r) = N_enc(r) / r
    where N_enc(r) = number of vortices with distance-from-origin < r.
    """
    radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    result = []
    for r in r_bins:
        n_enc = int(np.sum(radii < r))
        v = n_enc / r if r > 0 else 0
        result.append({
            'r': float(r),
            'n_enc': n_enc,
            'v_analytic': float(v),
        })
    return result


# ============================================================
#  RELAXATION
# ============================================================
def relax(Psi_init, n_steps, dt, damping, label=""):
    """Damped leapfrog — minimal stabilization."""
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
            w15 = measure_winding(Psi, 15.0)
            elapsed = time.time() - t0
            print(f"  [{label}] step {step:3d}: "
                  f"|Psi|_max={psi_max:.4f}  "
                  f"w(5)={w5:+.2f}  w(10)={w10:+.2f}  w(15)={w15:+.2f}  "
                  f"{elapsed:.0f}s", flush=True)

    return Psi


# ============================================================
#  SINGLE DISTRIBUTION RUN
# ============================================================
def run_distribution(label, distribution):
    print(f"\n{'='*62}")
    print(f"  {label}: N={N_VORTEX}, distribution={distribution}")
    print(f"{'='*62}")

    # --- Generate positions ---
    positions = generate_positions(
        N_VORTEX, distribution, R_GAL, MIN_SEP, seed=SEED)
    radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    radii_sorted = np.sort(radii)
    print(f"\n  Vortex radii: {[round(r, 2) for r in radii_sorted.tolist()]}")

    # --- Build IC ---
    print(f"\n  Building IC (multi-vortex multiplicative ansatz)...")
    Psi0 = build_multi_vortex_ic(positions)
    psi0_max = float(xp.max(xp.abs(Psi0)))
    print(f"  Initial |Psi|_max = {psi0_max:.4f}")

    # --- Initial windings ---
    print(f"\n  Initial windings:")
    radii_winding = [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    windings_initial = {}
    for r_test in radii_winding:
        w = measure_winding(Psi0, r_test)
        n_enc = int(np.sum(radii < r_test))
        windings_initial[f"r{int(r_test)}"] = {
            'w': round(w, 4),
            'n_enc': n_enc,
        }
        print(f"    w({r_test:4.1f}) = {w:+.4f}   (N_enclosed = {n_enc})")

    # --- Relax ---
    print(f"\n  Relaxing {N_RELAX} steps (damping={DAMPING})...")
    Psi_final = relax(Psi0, N_RELAX, DT, DAMPING, label)

    # --- Final windings ---
    print(f"\n  Final windings:")
    windings_final = {}
    for r_test in radii_winding:
        w = measure_winding(Psi_final, r_test)
        n_enc = int(np.sum(radii < r_test))
        windings_final[f"r{int(r_test)}"] = {
            'w': round(w, 4),
            'n_enc': n_enc,
        }
        print(f"    w({r_test:4.1f}) = {w:+.4f}   (N_enclosed = {n_enc})")

    psi_final_max = float(xp.max(xp.abs(Psi_final)))

    # --- Rotation curve ---
    print(f"\n  Measuring rotation curve...")
    r_bins = np.linspace(0.5, 25.0, 70)
    profile = measure_rotation_curve(Psi_final, r_bins)
    v_analytic = analytic_v_theta(positions, r_bins)

    # Store theory + ratio in each bin
    for p, va in zip(profile, v_analytic):
        p['n_enc']      = va['n_enc']
        p['v_analytic'] = va['v_analytic']
        if abs(va['v_analytic']) > 1e-10 and not np.isnan(p['v_theta']):
            p['ratio'] = p['v_theta'] / va['v_analytic']
        else:
            p['ratio'] = float('nan')

    # Print
    print(f"\n  {'r':>6s}  {'v_th(sim)':>10s}  {'N_enc/r':>10s}  {'ratio':>8s}")
    print(f"  {'-'*40}")
    for p in profile:
        if not np.isnan(p['v_theta']):
            ratio_str = f"{p['ratio']:+8.4f}" if not np.isnan(p['ratio']) else "    ----"
            print(f"  {p['r']:6.2f}  {p['v_theta']:+10.5f}  "
                  f"{p['v_analytic']:10.5f}  {ratio_str}")

    # --- Flatness statistics ---
    flat_vals = []
    for p in profile:
        if (FLATNESS_R_MIN <= p['r'] <= FLATNESS_R_MAX
                and not np.isnan(p['v_theta'])):
            flat_vals.append(p['v_theta'])

    if len(flat_vals) >= 3:
        mean_v = float(np.mean(np.abs(flat_vals)))
        std_v  = float(np.std(flat_vals))
        flatness = 1.0 - std_v / mean_v if mean_v > 1e-6 else float('nan')
    else:
        mean_v = std_v = flatness = float('nan')

    print(f"\n  Flatness [{FLATNESS_R_MIN}, {FLATNESS_R_MAX}]: "
          f"{flatness:.4f}  (mean v={mean_v:.4f}, std={std_v:.4f})")

    # --- Far-field statistics (r > 12) ---
    far_ratios = []
    for p in profile:
        if p['r'] > 12.0 and not np.isnan(p['ratio']):
            far_ratios.append(p['ratio'])

    if far_ratios:
        far_mean = float(np.mean(far_ratios))
        far_std  = float(np.std(far_ratios))
    else:
        far_mean = far_std = float('nan')

    print(f"  Far-field (r > 12): <sim/analytic> = {far_mean:.4f} ± {far_std:.4f}")

    return {
        'config': {
            'n_vortex': N_VORTEX,
            'distribution': distribution,
            'r_gal': R_GAL,
            'min_sep': MIN_SEP,
            'seed': SEED,
            'windings_all': [1] * N_VORTEX,
            'initial_condition': 'multi_vortex_multiplicative_ansatz',
        },
        'positions': positions.tolist(),
        'vortex_radii_sorted': [round(r, 4) for r in radii_sorted.tolist()],
        'amplitude': {
            'psi_max_initial': psi0_max,
            'psi_max_final':   psi_final_max,
        },
        'windings': {
            'initial': windings_initial,
            'final':   windings_final,
        },
        'rotation_curve': profile,
        'flatness_statistics': {
            'r_range': [FLATNESS_R_MIN, FLATNESS_R_MAX],
            'mean_v': mean_v,
            'std_v':  std_v,
            'flatness': flatness,
            'n_bins_used': len(flat_vals),
        },
        'far_field_statistics': {
            'r_min_far': 12.0,
            'ratio_mean': far_mean,
            'ratio_std':  far_std,
            'n_bins_used': len(far_ratios),
        },
    }


# ============================================================
#  MAIN RUNS
# ============================================================
results = {}
results['step4_uniform']   = run_distribution(
    "Step 4 uniform_disk",  "uniform_disk")
results['step5_inverse_r'] = run_distribution(
    "Step 5 inverse_r",     "inverse_r")


# ============================================================
#  PHYSICS CHECKS
# ============================================================
# The central claim of this experiment is that the INVERSE_R
# distribution produces a flatter rotation curve than UNIFORM,
# consistent with the kinematic prediction
#   v_theta(r) = N_enc(r) / r
# where N_enc(r) grows linearly for sigma(r) ~ 1/r
# and quadratically for sigma = const.
# ------------------------------------------------------------
flatness_uniform = results['step4_uniform']['flatness_statistics']['flatness']
flatness_inverse = results['step5_inverse_r']['flatness_statistics']['flatness']

far_uniform = results['step4_uniform']['far_field_statistics']['ratio_mean']
far_inverse = results['step5_inverse_r']['far_field_statistics']['ratio_mean']

p1 = flatness_inverse > flatness_uniform
p2 = abs(far_uniform - 1.0) < 0.10 if not np.isnan(far_uniform) else False
p3 = abs(far_inverse - 1.0) < 0.10 if not np.isnan(far_inverse) else False
p4 = abs(flatness_inverse - flatness_uniform) > 0.05

physics_checks = {
    'inverse_r_flatter_than_uniform':   p1,
    'uniform_far_field_matches_N_over_r': p2,
    'inverse_r_far_field_matches_N_over_r': p3,
    'flatness_difference_significant':  p4,
    'distribution_determines_curve':    p1 and p2 and p3 and p4,
}

print(f"\n{'='*62}")
print(f"  STEP 5 PHYSICS CHECKS")
print(f"{'='*62}")
print(f"  Uniform  flatness [{FLATNESS_R_MIN}, {FLATNESS_R_MAX}] = {flatness_uniform:.4f}")
print(f"  Inverse  flatness [{FLATNESS_R_MIN}, {FLATNESS_R_MAX}] = {flatness_inverse:.4f}")
print(f"  Uniform  far-field ratio                   = {far_uniform:.4f}")
print(f"  Inverse  far-field ratio                   = {far_inverse:.4f}")
print(f"\n  [{'PASS' if p1 else 'FAIL'}] inverse_r flatter than uniform")
print(f"  [{'PASS' if p2 else 'FAIL'}] uniform far-field matches N_enc/r")
print(f"  [{'PASS' if p3 else 'FAIL'}] inverse_r far-field matches N_enc/r")
print(f"  [{'PASS' if p4 else 'FAIL'}] flatness difference > 0.05")
print(f"  ==> {'DISTRIBUTION DETERMINES CURVE' if physics_checks['distribution_determines_curve'] else 'INVESTIGATE'}")


# ============================================================
#  UNIFIED JSON OUTPUT
# ============================================================
output = {
    'metadata': {
        'version': 'ep5_v2_final',
        'step': 5,
        'step_name': 'distribution_dependent_rotation_curves',
        'description': (
            'Distribution-dependent rotation curves: N=10 same-sign '
            'vortices under two spatial densities (uniform disk vs '
            'sigma ~ 1/r). Verifies the kinematic relation '
            'v_theta(r) = N_enc(r)/r and demonstrates that the shape '
            'of the rotation curve is uniquely determined by the '
            'spatial distribution of topological charge. Provides '
            'data for Figure 1 Panels (c) and (d) in the EP V paper.'
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
        'r_gal': R_GAL,
        'min_sep': MIN_SEP,
        'n_vortex': N_VORTEX,
        'seed': SEED,
        'flatness_r_min': FLATNESS_R_MIN,
        'flatness_r_max': FLATNESS_R_MAX,
    },
    'runs': results,
    'physics_checks': physics_checks,
    'overall_pass': physics_checks['distribution_determines_curve'],
    'timing': {
        'total_wall_time_s': round(time.time() - t_global, 1),
        'backend': BACKEND,
    },
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

file_size_mb = 0
try:
    import os
    file_size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024
except Exception:
    pass

print(f"\n{'='*62}")
print(f"  Results saved: {OUTPUT_JSON}")
if file_size_mb > 0:
    print(f"  File size: {file_size_mb:.2f} MB")
print(f"  Total wall time: {time.time() - t_global:.0f}s")
print(f"{'='*62}")
print("Done.")
