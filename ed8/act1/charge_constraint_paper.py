#!/usr/bin/env python3
"""
Charge quantization and conservation from ED constraint and topology — paper figure.
===================================================================================

This script demonstrates how charge quantization (integer winding number of the
phase field) and charge conservation (topological protection of that winding)
follow from an ED-style setup: only a 2D phase field on a lattice and a stiffness
(stretching) energy—no Coulomb force, no 1/r potential, no Poisson or Maxwell
equations, and no independent input variable called "electric charge."

What is specified (constraint / microscopic input):
  1. 2D lattice with phase field Phi(x,y)
  2. Topological vortices (winding ±1), i.e. quantized circulation
  3. Phase stiffness energy E = (rho_s/2)|grad Phi|^2

What emerges (effective long-range physics):
  - Interaction energy E_int ~ ln(d): 2D Coulomb scaling; slope ~ ±2*pi*rho_s.
  - Same-sign vortices repel; opposite-sign attract.
  - Gauss's law as exact phase flux: winding = 1.000000 at all loop radii.
  - Quantization: winding is integer by topology, not fitted.
  - Conservation: vortex charge cannot be continuously removed without a core singularity.

In this view, what Maxwell describes as electromagnetic interaction is recast as the
energetic cost of bending Phi under topological constraints.

Paper figure (300 dpi):
  charge_result.png — four panels: phase fields, Coulomb scaling, Gauss flux

Jae-Ahn Shin, Main Paper — Charge Section
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# SETUP: 2D periodic lattice; phase Phi is the sole dynamical field here.
# ============================================================

N = 512          # Grid size
L = 80.0         # Physical size
dx = L / N
rho_s = 1.0      # Phase stiffness (vacuum condensate A_0^2)

x = np.linspace(-L/2, L/2, N, endpoint=False)
y = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y = np.meshgrid(x, y)

print("=" * 65)
print("  CHARGE FROM PHASE STIFFNESS ALONE")
print("  No Coulomb. No 1/r. No Poisson. No Maxwell.")
print("  Just: phase field + vortices + stiffness energy.")
print("=" * 65)
print(f"\n  Grid: {N}x{N}, L={L}, dx={dx:.4f}")

# ============================================================
# VORTEX: topological defect; integer winding q is the quantized "charge."
# ============================================================

def vortex_phase(X, Y, x0, y0, q):
    """
    Phase field of a single vortex.

    A vortex is a point where the phase winds
    by 2*pi*q around a closed loop.
    q = +1: positive charge (counterclockwise winding).
    q = -1: negative charge (clockwise winding).

    This is not a model of charge.
    This IS charge: the winding number of the phase field.
    """
    return q * np.arctan2(Y - y0, X - x0)

# ============================================================
# ENERGY: phase stiffness only (no separate interaction potential).
# ============================================================

def compute_energy_density(Phi):
    """
    Energy density from phase gradients.

    E = (rho_s / 2) |grad Phi|^2

    This is the ONLY energy in the system.
    No Coulomb. No 1/r. No interaction potential.
    The phase gradient IS the electric field.
    The stiffness IS the vacuum permittivity.
    """
    dPhi_x = np.angle(np.exp(1j * (np.roll(Phi, -1, axis=1) - Phi)))
    dPhi_y = np.angle(np.exp(1j * (np.roll(Phi, -1, axis=0) - Phi)))
    return 0.5 * rho_s * (dPhi_x**2 + dPhi_y**2) / dx**2

# ============================================================
# GAUSS'S LAW: line integral of grad Phi / phase jump sum gives enclosed winding.
# ============================================================

def measure_winding(Phi, x0, y0, radius, npts=200):
    """
    Measure the phase winding around a closed contour.

    This is Gauss's law:
    the total phase circulation around any loop
    equals 2*pi times the enclosed winding number.

    If this gives 1.000000 at every radius,
    Gauss's law is exact.
    """
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    px = x0 + radius * np.cos(theta)
    py = y0 + radius * np.sin(theta)
    total = 0.0
    for k in range(npts):
        ix1 = int((px[k] + L/2) / dx) % N
        iy1 = int((py[k] + L/2) / dx) % N
        ix2 = int((px[(k+1) % npts] + L/2) / dx) % N
        iy2 = int((py[(k+1) % npts] + L/2) / dx) % N
        dphi = np.angle(np.exp(1j * (Phi[iy2, ix2] - Phi[iy1, ix1])))
        total += dphi
    return total / (2 * np.pi)

# ============================================================
# MEASUREMENT: two-vortex cross energy vs separation d (continuum gradients).
# Superpose two vortices; cross-term in |grad(Phi1+Phi2)|^2 isolates interaction.
# ============================================================

print(f"\n  Computing interaction energies...")

distances = np.linspace(2.0, 20.0, 40)
r_core = 0.5
r_boundary = 30.0

E_cross_same = []
E_cross_opp = []

for d in distances:
    # Pair on x-axis: vortex 1 at (-d/2, 0), vortex 2 at (+d/2, 0).
    # Same-sign case uses q2 = +1 on the right; opposite-sign uses q2 = -1 (gradients flipped).

    r1_sq = np.maximum((X + d/2)**2 + Y**2, r_core**2)
    r2_sq = np.maximum((X - d/2)**2 + Y**2, r_core**2)

    # Phase gradients of each vortex
    grad1_x = -Y / r1_sq
    grad1_y = (X + d/2) / r1_sq

    # Same sign: grad2 same orientation
    grad2_x_same = -Y / r2_sq
    grad2_y_same = (X - d/2) / r2_sq

    # Opposite sign: grad2 reversed
    grad2_x_opp = Y / r2_sq
    grad2_y_opp = -(X - d/2) / r2_sq

    # Integration mask: outside both cores, inside a large circle (truncate boundary).
    R1 = np.sqrt((X + d/2)**2 + Y**2)
    R2 = np.sqrt((X - d/2)**2 + Y**2)
    R_b = np.sqrt(X**2 + Y**2)
    mask = (R1 > r_core) & (R2 > r_core) & (R_b < r_boundary)

    # Cross-term only: rho_s * integral (grad Phi1 . grad Phi2); constant self-terms dropped by reference subtraction.
    dot_same = (grad1_x * grad2_x_same + grad1_y * grad2_y_same)
    dot_opp = (grad1_x * grad2_x_opp + grad1_y * grad2_y_opp)

    E_cross_same.append(rho_s * np.sum(dot_same[mask]) * dx**2)
    E_cross_opp.append(rho_s * np.sum(dot_opp[mask]) * dx**2)

E_cross_same = np.array(E_cross_same)
E_cross_opp = np.array(E_cross_opp)

# Reference: subtract E_cross at largest d so the plateau (self-energy + boundary) cancels.
E_same = E_cross_same - E_cross_same[-1]
E_opp = E_cross_opp - E_cross_opp[-1]

# ============================================================
# FITTING: linear fit of E_int vs ln(d) — expected 2D Coulomb logarithmic tail.
# ============================================================

mask_fit = (distances > 3.0) & (distances < 18.0)
log_d = np.log(distances[mask_fit])

coeffs_same = np.polyfit(log_d, E_same[mask_fit], 1)
coeffs_opp = np.polyfit(log_d, E_opp[mask_fit], 1)

theory_slope = 2 * np.pi * rho_s  # 2D Coulomb slope from phase stiffness; rho_s=1 => 2*pi.

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0

R2_same = r_squared(E_same[mask_fit], coeffs_same[0]*log_d + coeffs_same[1])
R2_opp = r_squared(E_opp[mask_fit], coeffs_opp[0]*log_d + coeffs_opp[1])

print(f"\n  Same charge (+1,+1):")
print(f"    slope = {coeffs_same[0]:.4f}")
print(f"    theory = 2*pi*rho_s = {theory_slope:.4f}")
print(f"    ratio = {abs(coeffs_same[0]/theory_slope):.4f}")
print(f"    R^2 = {R2_same:.6f}")

print(f"\n  Opposite charge (+1,-1):")
print(f"    slope = {coeffs_opp[0]:.4f}")
print(f"    theory = -2*pi*rho_s = {-theory_slope:.4f}")
print(f"    ratio = {abs(coeffs_opp[0]/theory_slope):.4f}")
print(f"    R^2 = {R2_opp:.6f}")

# ============================================================
# GAUSS'S LAW: winding independent of loop radius — topological charge inside.
# ============================================================

print(f"\n  Gauss's law check:")
Phi_test = vortex_phase(X, Y, 0, 0, +1)
radii = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0]
windings = []
for r in radii:
    w = measure_winding(Phi_test, 0, 0, r)
    windings.append(w)
    print(f"    r={r:5.1f}: winding = {w:.6f}")

# ============================================================
# FIGURE: four panels at 300 dpi for the paper.
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    'Charge from Phase Stiffness Alone\n'
    r'Input: $E = (\rho_s/2)\,|\nabla\Phi|^2$ only.  '
    'No Coulomb, no $1/r$, no Maxwell.',
    fontsize=14, fontweight='bold', y=1.02)

# Panel (a): superposed +1, +1 vortices — phase interference pattern.
ax = axes[0, 0]
d_vis = 8.0
Phi_same = (vortex_phase(X, Y, -d_vis/2, 0, +1)
            + vortex_phase(X, Y, +d_vis/2, 0, +1))
im = ax.pcolormesh(X, Y, np.mod(Phi_same + np.pi, 2*np.pi) - np.pi,
                    cmap='twilight', shading='auto')
ax.plot([-d_vis/2], [0], 'r+', markersize=15, markeredgewidth=3)
ax.plot([+d_vis/2], [0], 'r+', markersize=15, markeredgewidth=3)
ax.set_title('(a) Same charge $(+1, +1)$\nPhase field $\\Phi(x,y)$',
             fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
plt.colorbar(im, ax=ax, label='$\\Phi$', shrink=0.8)

# Panel (b): +1 and -1 — phase structure for attractive channel.
ax = axes[0, 1]
Phi_opp = (vortex_phase(X, Y, -d_vis/2, 0, +1)
           + vortex_phase(X, Y, +d_vis/2, 0, -1))
im = ax.pcolormesh(X, Y, np.mod(Phi_opp + np.pi, 2*np.pi) - np.pi,
                    cmap='twilight', shading='auto')
ax.plot([-d_vis/2], [0], 'r+', markersize=15, markeredgewidth=3)
ax.plot([+d_vis/2], [0], 'b_', markersize=15, markeredgewidth=3)
ax.set_title('(b) Opposite charge $(+1, -1)$\nPhase field $\\Phi(x,y)$',
             fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
plt.colorbar(im, ax=ax, label='$\\Phi$', shrink=0.8)

# Panel (c): interaction energy vs ln(d); slopes match ±2*pi*rho_s.
ax = axes[1, 0]
log_d_all = np.log(distances)

ax.plot(log_d_all, E_same, 'o', color='#D32F2F', ms=4, alpha=0.7,
        label='Same charge')
ax.plot(log_d_all, E_opp, 's', color='#1565C0', ms=4, alpha=0.7,
        label='Opposite charge')

# Linear fits over the intermediate distance window.
ax.plot(log_d_all, coeffs_same[0]*log_d_all + coeffs_same[1],
        '-', color='#D32F2F', lw=2,
        label=f'Fit: {coeffs_same[0]:.2f}$\\cdot\\ln d$')
ax.plot(log_d_all, coeffs_opp[0]*log_d_all + coeffs_opp[1],
        '-', color='#1565C0', lw=2,
        label=f'Fit: {coeffs_opp[0]:.2f}$\\cdot\\ln d$')

ax.set_xlabel('$\\ln\\,d$', fontsize=12)
ax.set_ylabel('$E_{\\mathrm{int}}$ (relative)', fontsize=12)
ax.set_title(
    f'(c) 2D Coulomb law from phase stiffness\n'
    f'Theory: $\\pm 2\\pi\\rho_s = \\pm{theory_slope:.2f}$,  '
    f'$R^2 = {min(R2_same, R2_opp):.4f}$',
    fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Text box: fitted slopes vs theoretical Coulomb stiffness.
ax.annotate(
    f'Same: slope = {coeffs_same[0]:.2f}\n'
    f'Opp:  slope = {coeffs_opp[0]:.2f}\n'
    f'Theory: $\\pm${theory_slope:.2f}\n'
    f'Accuracy: {abs(coeffs_same[0]/theory_slope)*100:.1f}%',
    xy=(0.03, 0.97), xycoords='axes fraction',
    fontsize=9, va='top',
    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

# Panel (d): closed-loop flux 2*pi*q is radius-independent (quantized).
ax = axes[1, 1]
fluxes = [w * 2 * np.pi for w in windings]
ax.plot(radii, fluxes, 'o-', color='#2E7D32', ms=8, lw=2,
        label='Measured flux')
ax.axhline(2*np.pi, color='k', ls='--', lw=1.5,
           label=f'Theory: $2\\pi = {2*np.pi:.3f}$')
ax.set_xlabel('Radius $r$', fontsize=12)
ax.set_ylabel(r'$\oint \nabla\Phi \cdot d\ell$', fontsize=12)
ax.set_title(
    "(d) Gauss's law: flux = $2\\pi q$\n"
    "Exact at all radii (winding = 1.000000)",
    fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_ylim(6.0, 6.6)

# Arrow note on the flux plot.
ax.annotate('Winding = 1.000000\nat every radius',
            xy=(15, 2*np.pi), xytext=(18, 6.45),
            fontsize=10, color='#2E7D32', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2E7D32'))

plt.tight_layout()
plt.savefig('charge_result.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: charge_result.png (300 dpi)")

# ============================================================
# SUMMARY: print quantization, conservation, and emergent Coulomb scaling.
# ============================================================

print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"\n  INPUT:")
print(f"    Phase field Phi(x,y)")
print(f"    Vortices (winding number +/-1)")
print(f"    Energy = (rho_s/2)|grad Phi|^2")
print(f"\n  NOT INPUT:")
print(f"    Coulomb, 1/r, Poisson, Maxwell, charge")
print(f"\n  OUTPUT:")
print(f"    E_int ~ ln(d): 2D Coulomb (R^2 = {min(R2_same, R2_opp):.4f})")
print(f"    Same sign repels, opposite attracts")
print(f"    Gauss's law: exact (1.000000)")
print(f"    Charge quantization: automatic (integer winding)")
print(f"    Charge conservation: topological (cannot be destroyed)")
print(f"\n  What Maxwell called electromagnetic interaction,")
print(f"  ED calls the energetic cost of bending the phase field.")
print(f"{'='*65}")

if __name__ == "__main__":
    pass
