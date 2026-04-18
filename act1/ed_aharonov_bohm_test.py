"""
aharonov_bohm_test.py
=====================
3D test of the Aharonov-Bohm effect in ED.

The Aharonov-Bohm effect is the most famous demonstration that
electromagnetic phase is a nonlocal, topological quantity:
a charged particle traveling through a region with B=0 still
acquires a phase shift if its path encloses magnetic flux.
In standard QM this is mysterious — the particle "feels" the
vector potential without ever touching the field.

In ED there is no mystery. The vector potential IS the phase
gradient of the deviation field. The Aharonov-Bohm phase is
just the closed-loop integral of ∇Φ around enclosed winding —
i.e. Stokes' theorem applied to phase circulation.

This script verifies the effect directly by:
  - placing two parallel vortex lines (winding n=+1 each) on a
    256³ lattice
  - measuring closed-loop integrals of ∇Φ · dℓ around different
    loops in the xy-plane
  - confirming that:
      (1) loops enclosing both vortices give exactly 2 × (2π)
          [phase additivity, Aharonov-Bohm]
      (2) loops in vacuum enclosing neither give exactly 0
          [the local |∇Φ| can be nonzero, but topology is what matters]
      (3) the result is independent of loop size, position, and
          z-slice [it depends only on enclosed winding]

If all three hold, the famous "nonlocal action of vector potential
on point particles" is just Stokes' theorem on a complex field.

Run on Colab GPU (cupy). ~60 seconds expected.

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
# PARAMETERS
# ═══════════════════════════════════════════════
N = 256
L = 12.5
dx = L / N

dt = 0.0003
c = 1.0
lam = 2.0
alpha = 10.0
v_vac = float(np.sqrt(lam / alpha))

# Two vortex lines along z, separated in x
d_separation = 2.0       # distance between two vortices in physical units
x_A = -d_separation / 2  # vortex A position
x_B = +d_separation / 2  # vortex B position
sigma = 0.04             # core width (same as before)
A0 = 4.0                 # initial amplitude depression
n_A = 1                  # winding of vortex A
n_B = 1                  # winding of vortex B

relax_steps = 5000       # short to keep vortices in place

print(f"\n=== Aharonov-Bohm Test (Two Parallel Vortex Lines) ===")
print(f"  Grid: {N}³ = {N**3:,} points, ~{2*16*N**3/1e6:.0f} MB")
print(f"  L={L}, dx={dx:.5f}")
print(f"  λ={lam}, α={alpha}, v_vac={v_vac:.4f}")
print(f"  Vortex A: x={x_A:+.2f}, n={n_A}")
print(f"  Vortex B: x={x_B:+.2f}, n={n_B}")
print(f"  Separation d={d_separation}")
print(f"  Healing length ξ = {1/np.sqrt(2*lam):.3f}")
print(f"  d/ξ = {d_separation*np.sqrt(2*lam):.2f}")
print(f"  Relax: {relax_steps} steps, dt={dt}")
print()


# ═══════════════════════════════════════════════
# 3D GRID
# ═══════════════════════════════════════════════
print("  Building grid...", flush=True)
t0 = timer.time()

x = xp.linspace(-L/2, L/2, N, dtype=xp.float64)
y = xp.linspace(-L/2, L/2, N, dtype=xp.float64)
z = xp.linspace(-L/2, L/2, N, dtype=xp.float64)
X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')

# Distance from each vortex axis (in xy plane, ignoring z)
R_A = xp.sqrt((X - x_A)**2 + Y**2) + 1e-10
R_B = xp.sqrt((X - x_B)**2 + Y**2) + 1e-10

# Phase angle around each vortex
theta_A = xp.arctan2(Y, X - x_A)
theta_B = xp.arctan2(Y, X - x_B)

print(f"  Grid built in {timer.time()-t0:.1f}s")


# ═══════════════════════════════════════════════
# INITIAL CONDITION: superposition of two vortex phases
# ═══════════════════════════════════════════════
print("  Initializing two vortex lines...", flush=True)

# Amplitude: depression at each vortex core
core_A = A0 * xp.exp(-R_A**2 / (2 * sigma**2))
core_B = A0 * xp.exp(-R_B**2 / (2 * sigma**2))

# Phase: sum of two winding phases
# Ψ = (v_vac + cores) * exp(i (n_A θ_A + n_B θ_B))
total_phase = n_A * theta_A + n_B * theta_B
total_amp_dep = core_A + core_B

Psi = (v_vac + total_amp_dep) * xp.exp(1j * total_phase)
Psi_dot = xp.zeros_like(Psi)

print(f"  Initial: max|Ψ|={float(xp.max(xp.abs(Psi))):.3f}")


# ═══════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════
def laplacian_3d(f, dx_):
    return (xp.roll(f,1,0) + xp.roll(f,-1,0) +
            xp.roll(f,1,1) + xp.roll(f,-1,1) +
            xp.roll(f,1,2) + xp.roll(f,-1,2) - 6*f) / dx_**2


# ═══════════════════════════════════════════════
# RELAX
# ═══════════════════════════════════════════════
print(f"\n  Relaxing for {relax_steps} steps...", flush=True)
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
        # Track core amplitudes at vortex positions
        ix_A = int((x_A + L/2) / dx)
        ix_B = int((x_B + L/2) / dx)
        iy0, iz0 = N//2, N//2
        amp_A = float(xp.abs(Psi[ix_A, iy0, iz0]))
        amp_B = float(xp.abs(Psi[ix_B, iy0, iz0]))
        print(f"    step {step+1:5d}: max|Ψ|={max_amp:.3f}, "
              f"|Ψ|@A={amp_A:.3f}, |Ψ|@B={amp_B:.3f}, "
              f"{elapsed:.0f}s", flush=True)

t_relax = timer.time() - t0
print(f"  Relaxed in {t_relax:.1f}s")


# ═══════════════════════════════════════════════
# CLOSED LOOP INTEGRAL FUNCTION
# ═══════════════════════════════════════════════
def circulation(Psi_2d, center_x, center_y, radius, n_samples=128):
    """Compute ∮ ∇Φ · dℓ around a circular loop in xy plane.
    
    Returns (circulation / 2π).
    
    Psi_2d: 2D complex field (single z slice)
    center_x, center_y: physical coordinates of loop center
    radius: physical radius of loop
    n_samples: number of points around the loop
    """
    angles = xp.linspace(0, 2*np.pi, n_samples, endpoint=False)
    
    # Sample positions along the loop
    xs = center_x + radius * xp.cos(angles)
    ys = center_y + radius * xp.sin(angles)
    
    # Convert to grid indices (with bilinear-like rounding)
    ixs = ((xs + L/2) / dx).astype(xp.int32)
    iys = ((ys + L/2) / dx).astype(xp.int32)
    
    # Clip to valid range
    ixs = xp.clip(ixs, 0, N-1)
    iys = xp.clip(iys, 0, N-1)
    
    # Get phases at sample points
    if GPU:
        ixs_np = xp.asnumpy(ixs)
        iys_np = xp.asnumpy(iys)
        Psi_np = xp.asnumpy(Psi_2d)
    else:
        ixs_np = ixs
        iys_np = iys
        Psi_np = Psi_2d
    
    phases = np.angle(Psi_np[ixs_np, iys_np])
    
    # Compute phase differences along the loop, with 2π wrap handling
    dphases = np.diff(phases)
    dphases = (dphases + np.pi) % (2*np.pi) - np.pi
    
    # Add the wraparound (last to first)
    dphi_last = phases[0] - phases[-1]
    dphi_last = (dphi_last + np.pi) % (2*np.pi) - np.pi
    
    total = np.sum(dphases) + dphi_last
    return total / (2*np.pi)


# ═══════════════════════════════════════════════
# MEASUREMENTS
# ═══════════════════════════════════════════════
print(f"\n  Measuring closed-loop integrals...", flush=True)

# Use the central z slice for 2D measurements
Psi_slice = Psi[:, :, N//2]


# Loop 1: around vortex A only
loop1_radii = [0.5, 0.7, 0.9]
print(f"\n  --- Loop 1: around vortex A only ---")
print(f"  {'radius':>8s} {'circulation/2π':>16s} {'expected':>10s}")
print(f"  {'-'*40}")
for r in loop1_radii:
    circ = circulation(Psi_slice, x_A, 0.0, r)
    print(f"  {r:8.2f} {circ:16.4f} {n_A:10.0f}")


# Loop 2: around vortex B only
print(f"\n  --- Loop 2: around vortex B only ---")
print(f"  {'radius':>8s} {'circulation/2π':>16s} {'expected':>10s}")
print(f"  {'-'*40}")
for r in loop1_radii:
    circ = circulation(Psi_slice, x_B, 0.0, r)
    print(f"  {r:8.2f} {circ:16.4f} {n_B:10.0f}")


# Loop 3: around both vortices (centered between them)
loop3_radii = [2.0, 2.5, 3.0, 3.5, 4.0]
print(f"\n  --- Loop 3: around BOTH vortices (phase additivity) ---")
print(f"  {'radius':>8s} {'circulation/2π':>16s} {'expected':>10s}")
print(f"  {'-'*40}")
for r in loop3_radii:
    circ = circulation(Psi_slice, 0.0, 0.0, r)
    print(f"  {r:8.2f} {circ:16.4f} {n_A + n_B:10.0f}")


# Loop 4: small loop in the gap (between vortices but not enclosing either)
# Place it at (0, 1.5) — above the midpoint, away from both vortices
print(f"\n  --- Loop 4: in vacuum, NOT enclosing either vortex ---")
print(f"  {'center':>14s} {'radius':>8s} {'circulation/2π':>16s} {'expected':>10s}")
print(f"  {'-'*55}")
test_loops = [
    (0.0, 2.5, 0.5),   # above the midpoint
    (0.0, -2.5, 0.5),  # below the midpoint
    (3.0, 0.0, 0.5),   # right of both vortices
    (-3.0, 0.0, 0.5),  # left of both vortices
]
for cx, cy, r in test_loops:
    circ = circulation(Psi_slice, cx, cy, r)
    print(f"  ({cx:+.1f},{cy:+.1f}) {r:8.2f} {circ:16.4f} {0:10.0f}")


# Linearity check
print(f"\n  --- Linearity check (phase additivity) ---")
print(f"  Loop3(r) - Loop1(r=0.7) - Loop2(r=0.7) should equal 0")
print(f"  {'r3':>6s} {'L3':>10s} {'L1':>10s} {'L2':>10s} {'L3-L1-L2':>10s}")
print(f"  {'-'*55}")
L1 = circulation(Psi_slice, x_A, 0.0, 0.7)
L2 = circulation(Psi_slice, x_B, 0.0, 0.7)
for r in loop3_radii:
    L3 = circulation(Psi_slice, 0.0, 0.0, r)
    diff = L3 - L1 - L2
    print(f"  {r:6.2f} {L3:10.4f} {L1:10.4f} {L2:10.4f} {diff:+10.4f}")


# Z-invariance check (verify line defects, not point defects)
print(f"\n  --- Z-invariance: Loop 3 (r=2.5) at multiple z-slices ---")
print(f"  {'z_idx':>8s} {'circulation/2π':>16s}")
print(f"  {'-'*30}")
for zi in [N//4, 3*N//8, N//2, 5*N//8, 3*N//4]:
    Psi_z = Psi[:, :, zi]
    circ = circulation(Psi_z, 0.0, 0.0, 2.5)
    print(f"  {zi:8d} {circ:16.4f}")


# Save
results = {
    'params': {
        'N': N, 'L': L, 'dx': dx,
        'lam': lam, 'alpha': alpha, 'v_vac': v_vac,
        'd_separation': d_separation,
        'x_A': x_A, 'x_B': x_B,
        'n_A': n_A, 'n_B': n_B,
        'sigma': sigma, 'A0': A0,
        'relax_steps': relax_steps,
    },
    'L1_at_r0.7': L1,
    'L2_at_r0.7': L2,
}

with open('aharonov_bohm.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved: aharonov_bohm.json")
print(f"  Total time: {timer.time()-t0+t_relax:.0f}s")
