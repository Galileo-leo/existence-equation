"""
vortex_line_3d.py
=================
3D test: single vortex LINE along z-axis — λ sweep.

Setup: vortex line along z, winding in xy-plane (n=1).
On 256³ grid, sweeping λ = 1.0, 2.0, 3.0, 4.0 (α=10 fixed).

Goal:
  Verify that a 3D vortex line reproduces the same 1/r tangential
  phase gradient structure we saw in 2D, and that the radial
  component remains zero. This establishes that the 2D result
  was not an artifact of dimensional reduction.

  The λ sweep shows how the condensate stiffness affects
  the vortex core structure while preserving the universal
  1/r tangential gradient (topologically protected).

Decomposition (in xy-plane perpendicular to vortex line):
  - radial component (∂Φ/∂r in xy)  → "electric" candidate
  - tangential component             → "magnetic vector potential" candidate
  - z component (∂Φ/∂z)              → should be ~0 (translation invariance)

What 3D adds:
  - We can check translation invariance along z
  - We can later test bent / closed vortex lines (real magnetic flux tubes)
  - The full B = ∇×A becomes a real 3-vector

Output:
  vortex_line_3d.json — all 4 λ results in one file

Run on Colab GPU (cupy). ~8-10 minutes expected (4 × 256³).

Author: Jae-Ahn Shin
"""

import numpy as np
import json
import time as timer

try:
    import cupy as xp
    GPU = True
    print("Backend: CuPy (GPU)")
except ImportError:
    import numpy as xp
    GPU = False
    print("Backend: NumPy (CPU)")


# ═══════════════════════════════════════════════
# FIXED PARAMETERS
# ═══════════════════════════════════════════════
N = 256
L = 12.5
dx = L / N

dt = 0.0003
c = 1.0
alpha = 10.0

A0 = 4.0
sigma_r = 0.04
n_wind = 1
relax_steps = 10000

LAMBDA_VALUES = [1.0, 2.0, 3.0, 4.0]


# ═══════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════
def laplacian_3d(f, dx_):
    return (xp.roll(f,1,0) + xp.roll(f,-1,0) +
            xp.roll(f,1,1) + xp.roll(f,-1,1) +
            xp.roll(f,1,2) + xp.roll(f,-1,2) - 6*f) / dx_**2


# ═══════════════════════════════════════════════
# GRID (shared across all λ runs)
# ═══════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  3D Vortex Line — λ Sweep")
print(f"  λ values: {LAMBDA_VALUES}")
print(f"  Grid: {N}³ = {N**3:,}, L={L}, dx={dx:.5f}")
print(f"  α={alpha}, n={n_wind}")
print(f"  Relax: {relax_steps} steps, dt={dt}")
print(f"  Memory per run: ~{2 * 16 * N**3 / 1e6:.0f} MB")
print(f"{'═'*65}\n")

print("  Building grid...", flush=True)
t_global = timer.time()

x = xp.linspace(-0.5, 0.5, N, dtype=xp.float64)
y = xp.linspace(-0.5, 0.5, N, dtype=xp.float64)
z = xp.linspace(-0.5, 0.5, N, dtype=xp.float64)
X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')

R_xy = xp.sqrt(X**2 + Y**2) + 1e-10
Theta = xp.arctan2(Y, X)
R_xy_phys = R_xy * L

cos_t = X / R_xy
sin_t = Y / R_xy

core_envelope = A0 * xp.exp(-R_xy**2 / (2 * sigma_r**2))

r_bins = np.linspace(0.5, L * 0.4, 20)
theory_t = [n_wind / float(r) for r in r_bins]

ref_r = 1.5
z_indices = [N//4, 3*N//8, N//2, 5*N//8, 3*N//4]

print(f"  Grid ready.\n")


# ═══════════════════════════════════════════════
# MAIN SWEEP FUNCTION
# ═══════════════════════════════════════════════
def run_vortex_line(lam):
    """Run a single 3D vortex line simulation for given λ."""

    v_vac = float(np.sqrt(lam / alpha))

    print(f"\n{'─'*65}")
    print(f"  λ = {lam}  (v_vac = {v_vac:.4f})")
    print(f"{'─'*65}")

    # Initial condition
    Psi = v_vac * xp.ones((N, N, N), dtype=xp.complex128)
    Psi += core_envelope * xp.exp(1j * n_wind * Theta)
    Psi_dot = xp.zeros_like(Psi)

    print(f"  Initial: max|Ψ|={float(xp.max(xp.abs(Psi))):.3f}")

    # Relax
    print(f"  Relaxing {relax_steps} steps...", flush=True)
    t0 = timer.time()

    for step in range(relax_steps):
        lap = laplacian_3d(Psi, dx)
        rho = xp.abs(Psi)**2
        accel = c**2 * (lap + lam*Psi - alpha*rho*Psi)
        Psi_dot += dt * accel
        Psi += dt * Psi_dot

        if (step + 1) % 1000 == 0:
            elapsed = timer.time() - t0
            max_amp = float(xp.max(xp.abs(Psi)))
            center_amp = float(xp.abs(Psi[N//2, N//2, N//2]))
            print(f"    step {step+1:5d}: max|Ψ|={max_amp:.3f}, "
                  f"center|Ψ|={center_amp:.3f}, {elapsed:.0f}s", flush=True)

    t_relax = timer.time() - t0
    print(f"  Relaxed in {t_relax:.1f}s")

    # Decomposition
    phase = xp.angle(Psi)
    gpx = xp.angle(xp.exp(1j * (xp.roll(phase,-1,0) - xp.roll(phase,1,0)))) / (2*dx)
    gpy = xp.angle(xp.exp(1j * (xp.roll(phase,-1,1) - xp.roll(phase,1,1)))) / (2*dx)
    gpz = xp.angle(xp.exp(1j * (xp.roll(phase,-1,2) - xp.roll(phase,1,2)))) / (2*dx)

    grad_r = gpx * cos_t + gpy * sin_t
    grad_t = -gpx * sin_t + gpy * cos_t
    grad_z = gpz

    # Shell averages (central z slab)
    z_slab = slice(N//2 - 8, N//2 + 8)
    amp = xp.abs(Psi)
    amp_slab = amp[:, :, z_slab]
    grad_r_slab = grad_r[:, :, z_slab]
    grad_t_slab = grad_t[:, :, z_slab]
    grad_z_slab = grad_z[:, :, z_slab]
    R_slab = R_xy_phys[:, :, z_slab]

    amp_profile = []
    grad_r_profile = []
    grad_t_profile = []
    grad_z_profile = []

    for rb in r_bins:
        dr_bin = (r_bins[1] - r_bins[0]) if len(r_bins) > 1 else 1.0
        mask = (R_slab > rb - dr_bin/2) & (R_slab < rb + dr_bin/2)
        n_pts = int(xp.sum(mask))
        if n_pts > 50:
            amp_profile.append(float(xp.mean(amp_slab[mask])))
            grad_r_profile.append(float(xp.mean(grad_r_slab[mask])))
            grad_t_profile.append(float(xp.mean(grad_t_slab[mask])))
            grad_z_profile.append(float(xp.mean(grad_z_slab[mask])))
        else:
            amp_profile.append(float('nan'))
            grad_r_profile.append(float('nan'))
            grad_t_profile.append(float('nan'))
            grad_z_profile.append(float('nan'))

    # Z-invariance check
    z_inv = []
    for zi in z_indices:
        slab_2d = grad_t[:, :, zi]
        R_2d = R_xy_phys[:, :, zi]
        mask = (R_2d > ref_r - 0.2) & (R_2d < ref_r + 0.2)
        if int(xp.sum(mask)) > 50:
            val = float(xp.mean(slab_2d[mask]))
        else:
            val = float('nan')
        z_inv.append({'z_idx': int(zi), 'grad_t_at_r1.5': val})

    # Print
    center_amp = float(xp.abs(Psi[N//2, N//2, N//2]))
    max_amp = float(xp.max(xp.abs(Psi)))

    print(f"\n  === Results: λ={lam}, n={n_wind} ===")
    print(f"  Center |Ψ|: {center_amp:.4f},  max |Ψ|: {max_amp:.3f}")
    print(f"  v_vac (theory): {v_vac:.4f}")
    print()
    print(f"  {'r':>6s} {'amp':>8s} {'grad_r':>10s} {'grad_t':>10s} "
          f"{'grad_z':>10s} {'theory_t':>10s} {'ratio':>8s}")
    print(f"  {'-'*70}")
    for i, r in enumerate(r_bins):
        if not np.isnan(grad_t_profile[i]) and theory_t[i] != 0:
            ratio = grad_t_profile[i] / theory_t[i]
        else:
            ratio = float('nan')
        print(f"  {r:6.2f} {amp_profile[i]:8.4f} {grad_r_profile[i]:+10.4f} "
              f"{grad_t_profile[i]:+10.4f} {grad_z_profile[i]:+10.4f} "
              f"{theory_t[i]:+10.4f} {ratio:8.4f}")

    print(f"\n  Z-invariance (grad_t at r={ref_r}):")
    for entry in z_inv:
        print(f"    z={entry['z_idx']:3d}: {entry['grad_t_at_r1.5']:+.4f}")

    # Clean up GPU memory
    del Psi, Psi_dot, phase, gpx, gpy, gpz, grad_r, grad_t, grad_z
    del amp, amp_slab, grad_r_slab, grad_t_slab, grad_z_slab, R_slab
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()

    return {
        'lam': lam,
        'v_vac': v_vac,
        'center_amp': center_amp,
        'max_amp': max_amp,
        'relax_time_s': round(t_relax, 1),
        'profile': {
            'r_bins': r_bins.tolist(),
            'amp': amp_profile,
            'grad_r': grad_r_profile,
            'grad_t': grad_t_profile,
            'grad_z': grad_z_profile,
            'theory_t': theory_t,
        },
        'z_invariance': z_inv,
    }


# ═══════════════════════════════════════════════
# RUN ALL λ VALUES
# ═══════════════════════════════════════════════
all_results = []

for lam_val in LAMBDA_VALUES:
    res = run_vortex_line(lam_val)
    all_results.append(res)

total_time = timer.time() - t_global


# ═══════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  COMPARISON SUMMARY — λ sweep")
print(f"{'═'*65}")
print(f"  {'λ':>5s} {'v_vac':>8s} {'center':>8s} {'max|Ψ|':>8s} "
      f"{'<ratio>':>8s} {'<|grad_r|>':>10s} {'<|grad_z|>':>10s} {'time':>6s}")
print(f"  {'-'*72}")

for res in all_results:
    prof = res['profile']
    ratios = []
    abs_gr = []
    abs_gz = []
    for i in range(len(prof['r_bins'])):
        gt = prof['grad_t'][i]
        th = prof['theory_t'][i]
        gr = prof['grad_r'][i]
        gz = prof['grad_z'][i]
        if not np.isnan(gt) and th != 0:
            ratios.append(gt / th)
        if not np.isnan(gr):
            abs_gr.append(abs(gr))
        if not np.isnan(gz):
            abs_gz.append(abs(gz))

    mean_ratio = np.mean(ratios) if ratios else float('nan')
    mean_gr = np.mean(abs_gr) if abs_gr else float('nan')
    mean_gz = np.mean(abs_gz) if abs_gz else float('nan')

    print(f"  {res['lam']:5.1f} {res['v_vac']:8.4f} {res['center_amp']:8.4f} "
          f"{res['max_amp']:8.3f} {mean_ratio:8.4f} {mean_gr:10.4f} "
          f"{mean_gz:10.4f} {res['relax_time_s']:5.0f}s")

print(f"\n  Key predictions:")
print(f"    • grad_t / theory ≈ 1.0 for ALL λ → 1/r is topologically universal")
print(f"    • |grad_r| ≈ 0 → no radial component (pure azimuthal)")
print(f"    • |grad_z| ≈ 0 → translation invariance along vortex line")
print(f"    • v_vac = √(λ/α) changes → core structure changes, but 1/r survives")


# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
output = {
    'params_fixed': {
        'N': N, 'L': L, 'dx': dx, 'alpha': alpha,
        'A0': A0, 'sigma_r': sigma_r,
        'n_wind': n_wind, 'relax_steps': relax_steps,
        'dt': dt, 'c': c,
    },
    'lambda_values': LAMBDA_VALUES,
    'runs': all_results,
    'total_time_s': round(total_time, 1),
}

with open('vortex_line_3d.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved: vortex_line_3d.json")
print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
print(f"\n{'═'*65}")
print(f"  DONE — {len(LAMBDA_VALUES)} λ values completed")
print(f"{'═'*65}")
