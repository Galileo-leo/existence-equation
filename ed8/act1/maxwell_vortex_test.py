"""
maxwell_vortex_test.py
======================
Demonstrates Maxwell-type electromagnetic behavior emerging from ED (existence-
equation) vortex dynamics on a 2D grid. A complex order parameter Phi with a
single vortex (unit winding) is relaxed via a Ginzburg–Landau-type equation with
Dirichlet boundary conditions. From the relaxed Phi we extract the gauge-invariant
phase-gradient field A (analogous to a vector potential / supercurrent), test for
|A| ~ 1/r decay away from the core (Coulomb / Biot–Savart scaling), and compute
curl(A) as an out-of-plane magnetic-field analogue B_z peaked at the vortex
core (effective point-like flux / charge density).

Pipeline:
1. Initialize a vortex with winding number n=1 at the center.
2. Relax Phi toward a lower-energy configuration.
3. Measure A = grad(theta) from Phi.
4. Fit log|A| vs log r; expect slope ~ -1 (1/r law).
5. Evaluate curl(A); expect a sharp peak at the core (delta-like concentration).

Interpretation: successful checks support viewing effective charge and EM-like
fields as arising from topology and phase structure, not from separate ad hoc
sources.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Configuration — numerical domain and relaxation schedule
N = 128          # Grid points per axis (resolution)
L = 40.0         # Physical box size (same units as x, y)
dx = L / N       # Spatial grid spacing
dt = 0.001       # Explicit time step for relaxation
steps = 5000     # Number of relaxation iterations

# 2. Coordinates — Cartesian grid centered at the origin
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# 3. Initial field — vortex ansatz for the complex order parameter
# Phi(r, theta) = f(r) * exp(i * n * theta); f(r) sets the core profile.
n_winding = 1
f_r = np.tanh(R / 1.0)  # Smooth radial profile regularizing the origin
Phi = f_r * np.exp(1j * n_winding * Theta)

# 4. Relaxation — Ginzburg–Landau dynamics with fixed Dirichlet boundaries
print(f"Initializing Vortex (n={n_winding})...")
print(f"Relaxing field for {steps} steps with Dirichlet BC...")

# Boundary mask: True on outer grid lines (Dirichlet sites)
mask_boundary = np.zeros((N, N), dtype=bool)
mask_boundary[0, :] = True
mask_boundary[-1, :] = True
mask_boundary[:, 0] = True
mask_boundary[:, -1] = True

# Frozen boundary values of Phi (held fixed during relaxation)
Phi_boundary = Phi[mask_boundary].copy()

for i in range(steps):
    # Discrete Laplacian via nearest neighbors; interior uses periodic shifts in roll
    lap = (np.roll(Phi, 1, axis=0) + np.roll(Phi, -1, axis=0) +
           np.roll(Phi, 1, axis=1) + np.roll(Phi, -1, axis=1) - 4*Phi) / (dx**2)
    
    # Ginzburg–Landau-type RHS: drives |Phi| -> 1 in bulk while allowing phase windings
    rho = np.abs(Phi)**2
    dPhi = lap + (1.0 - rho) * Phi
    
    Phi += dt * dPhi
    
    # Reapply Dirichlet data after each step
    Phi[mask_boundary] = Phi_boundary
    
    if i % 500 == 0:
        # Residual “force” norm in the interior (proxy for relaxation progress)
        energy = np.sum(np.abs(dPhi[~mask_boundary])**2)
        print(f"  Step {i}: Residual Energy = {energy:.4e}")

print("Relaxation Complete.")

# 5. Phase-gradient field A = grad(theta) — gauge-invariant from Phi
# A_x = Im(conj(Phi) * d_x Phi) / |Phi|^2, A_y analogous (supercurrent / effective vector potential).

print("Measuring Field A = grad(theta)...")
grad_x, grad_y = np.gradient(Phi, dx)
rho = np.abs(Phi)**2
# Floor |Phi|^2 so A stays finite at the vortex core where rho -> 0
rho_safe = np.maximum(rho, 1e-6)

A_x = np.imag(np.conj(Phi) * grad_x) / rho_safe
A_y = np.imag(np.conj(Phi) * grad_y) / rho_safe
A_mag = np.sqrt(A_x**2 + A_y**2)

# 6. Maxwell check — |A| vs r along the positive x-axis (y = 0)
mid = N // 2
r_vals = x[mid+2:]  # Exclude origin and first interior points
A_vals = A_mag[mid, mid+2:]

# Linear fit in log–log space: log|A| = slope * log(r) + const; expect slope ~ -1 (1/r)
log_r = np.log(r_vals)
log_A = np.log(A_vals)

# Use an annular band away from core and outer boundary to reduce finite-size artifacts
mask = (r_vals > 2.0) & (r_vals < L/2 - 2.0)
popt, pcov = curve_fit(lambda x, a, b: a * x + b, log_r[mask], log_A[mask])
slope, intercept = popt

print(f"Maxwell Check: Slope of log(A) vs log(r) = {slope:.4f}")
print(f"Expected: -1.0000 (1/r decay)")

# 7. Curl of A — 2D analogue of B_z = d_x A_y - d_y A_x (normal to the plane)
dA_y_dx = np.gradient(A_y, dx, axis=1)
dA_x_dy = np.gradient(A_x, dx, axis=0)
B_z = dA_y_dx - dA_x_dy

# 8. Figures — |Phi|^2, |A| scaling, and curl(A) at the vortex
plt.figure(figsize=(15, 5))

# Panel 1: Quiver plot of A overlaid on density |Phi|^2
plt.subplot(1, 3, 1)
st = 8
plt.quiver(X[::st, ::st], Y[::st, ::st], A_x[::st, ::st], A_y[::st, ::st], 
           pivot='mid', color='cyan', scale=20)
plt.imshow(rho, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='inferno', alpha=0.6)
plt.title('Vector Field A (Phase Gradient)')
plt.xlabel('x')
plt.ylabel('y')

# Panel 2: log–log comparison of measured |A| to a 1/r power law
plt.subplot(1, 3, 2)
plt.loglog(r_vals, A_vals, 'b.', label='Measured |A|')
plt.loglog(r_vals, np.exp(intercept) * r_vals**(-1), 'r--', label='1/r Fit')
plt.title(f'Maxwell Check: Slope = {slope:.4f}')
plt.xlabel('Distance r')
plt.ylabel('Field Strength |A|')
plt.legend()
plt.grid(True, which="both", ls="-")

# Panel 3: curl(A); localized structure at the core mimics concentrated flux / effective charge
plt.subplot(1, 3, 3)
plt.imshow(B_z, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='bwr', vmin=-2, vmax=2)
plt.colorbar(label='Curl(A)')
plt.title('Curl(A) ~ Charge Density')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig('maxwell_vortex_result.png')
print("Result saved to maxwell_vortex_result.png")
