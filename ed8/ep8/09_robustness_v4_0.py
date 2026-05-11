#!/usr/bin/env python3
"""
=============================================================================
EP8 Step 9: Robustness Scan — v4.0
=============================================================================

Purpose:
  Test whether the four-regime hierarchy is robust under:
    A. Packet wavenumber variation:  k = 4, 6 (baseline), 8
    B. Packet width variation:       σ = 2, 3 (baseline), 4
  And compute the dimensionless drive ratio χ = E_packet / E_closure
  to check whether the hierarchy ordering collapses onto χ.

  Key prediction to test:
    ε_dep < ε_coh < ε_em < ε_topo  (ordering preserved for all k, σ)
    Z_Φ peak always precedes G_k peak

  Reduced ε scan: [0.25, 0.40, 0.50, 0.70, 0.90, 1.00]
  (targets regime boundaries; full scan deferred to production)

  Grid: 256×256, same as v3.2 baseline.

Equation:
  (1/c²)Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ

Author: Jae-Ahn Shin
        Independent Researcher, Incheon, Republic of Korea
        jaeahn.shin.official@gmail.com
=============================================================================
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from time import time

VERSION = "v4.0"
DPI = 600

# ============================================================
# 1. PARAMETERS
# ============================================================
Nx, Ny = 256, 256
Lx, Ly = 60.0, 60.0
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

xc, yc = Lx/2, Ly/2
R = np.sqrt((X - xc)**2 + (Y - yc)**2)
THETA = np.arctan2(Y - yc, X - xc)

lam   = 1.0
alpha = 1.0
c_ed  = 1.0
A0    = np.sqrt(lam / alpha)
c2    = c_ed**2
dt    = 0.003
n_wind = 1

sponge_width = 8.0
sponge_str   = 2.5
n_relax      = 12000
gamma_relax  = 0.4
n_steps      = 8000
record_every = 40

# Robustness scan parameters
# Phase 1 (pilot): k-variation only, 5 ε points
# Phase 2 (full):  uncomment width configs, add ε=1.00
SCAN_CONFIGS = [
    # (label, packet_k, packet_width)
    ("k4_s3", 4.0, 3.0),
    ("k6_s3", 6.0, 3.0),   # ~ baseline (width=3 vs baseline 4, close enough)
    ("k8_s3", 8.0, 3.0),
    # --- Phase 2: width robustness ---
    ("k6_s2", 6.0, 2.0),
    ("k6_s4", 6.0, 4.0),   # = baseline width
]

# Reduced ε scan (regime boundary targets)
eps_values = [0.25, 0.40, 0.50, 0.70, 0.90]

# Diagnostic regions
R_bound   = 6.0
R_out_min = 13.0
R_out_max = 22.0
N_THETA_BINS = 90

print("=" * 70)
print(f"EP8 Step 9: Robustness Scan {VERSION}")
print(f"  Grid: {Nx}×{Ny}, Lx={Lx}, dx={dx:.4f}, dt={dt}")
print(f"  λ={lam}, α={alpha}, A0={A0:.4f}")
print(f"  Configs: {[c[0] for c in SCAN_CONFIGS]}")
print(f"  ε values: {eps_values}")
print(f"  Total runs: {len(SCAN_CONFIGS) * len(eps_values)} full + baselines")
print("=" * 70)

# ============================================================
# 2. CORE ED (same as v3.2)
# ============================================================

def laplacian_2d(psi):
    lap  = (np.roll(psi, -1, 0) + np.roll(psi, 1, 0) - 2*psi) / dx**2
    lap += (np.roll(psi, -1, 1) + np.roll(psi, 1, 1) - 2*psi) / dy**2
    return lap

def ed_rhs(psi, lam_val, alpha_val):
    return laplacian_2d(psi) + lam_val * psi - alpha_val * np.abs(psi)**2 * psi

def gradient_2d(psi):
    gx = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2*dx)
    gy = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2*dy)
    return gx, gy

def sponge_2d(XX, YY, LLx, LLy, w, s):
    g = np.zeros_like(XX)
    g = np.where(XX < w,      np.maximum(g, s*((w-XX)/w)**2), g)
    g = np.where(XX > LLx-w,  np.maximum(g, s*((XX-(LLx-w))/w)**2), g)
    g = np.where(YY < w,      np.maximum(g, s*((w-YY)/w)**2), g)
    g = np.where(YY > LLy-w,  np.maximum(g, s*((YY-(LLy-w))/w)**2), g)
    return g

def find_core(psi, search_radius=10.0):
    mask = R < search_radius
    A = np.abs(psi)
    vals = np.where(mask, A, np.inf)
    idx = np.argmin(vals)
    ix, iy = np.unravel_index(idx, psi.shape)
    return float(x[ix]), float(y[iy])

def measure_winding(psi, cx, cy, r_check):
    n_pts = 120
    th = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    ix = ((cx + r_check*np.cos(th))/dx).astype(int) % Nx
    iy = ((cy + r_check*np.sin(th))/dy).astype(int) % Ny
    phase = np.angle(psi[ix, iy])
    dphi = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    return np.sum(dphi) / (2*np.pi)

# ============================================================
# 3. DIAGNOSTICS
# ============================================================

def align_to_reference(psi, ref, mask):
    z = np.sum(np.conj(ref[mask]) * psi[mask])
    return psi * np.exp(-1j * np.angle(z))

def smooth_step(z, z0, width):
    return 0.5 * (1.0 + np.tanh((z - z0) / width))

def outgoing_window(RR, R_inner, R_outer, width=2.0):
    return smooth_step(RR, R_inner, width) * (1.0 - smooth_step(RR, R_outer, width))

def kspace_power(field, ddx, ddy, window):
    f = field * window
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F)**2
    kx = np.fft.fftshift(np.fft.fftfreq(field.shape[0], ddx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(field.shape[1], ddy)) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return KX, KY, P

def angular_distribution_k(KX, KY, P, kmin, kmax, n_theta=90):
    K = np.sqrt(KX**2 + KY**2)
    TH = np.arctan2(KY, KX)
    bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    out = np.zeros(n_theta)
    ring = (K >= kmin) & (K <= kmax)
    for i in range(n_theta):
        m = ring & (TH >= bins[i]) & (TH < bins[i+1])
        out[i] = np.sum(P[m]) if np.any(m) else 0.0
    return centers, out

def smooth_circular(arr, width=2):
    out = arr.copy()
    for _ in range(width):
        out = 0.25*np.roll(out, 1) + 0.5*out + 0.25*np.roll(out, -1)
    return out

def angular_metrics(theta, Ptheta, n_bins=90):
    total = np.sum(Ptheta) + 1e-30
    p = Ptheta / total
    S_raw = -np.sum(p * np.log(p + 1e-30))
    S_max = np.log(n_bins)
    S_norm = S_raw / S_max
    Q2 = np.abs(np.sum(p * np.exp(2j * theta)))
    return {'S_norm': float(S_norm), 'Q2': float(Q2)}

def compute_total_energy(psi, pdot, lam_val, alpha_val):
    """Compute full ED energy density: temporal + gradient + potential.

    E = (1/2)|Ψ̇|²/c² + (1/2)|∇Ψ|² − (λ/2)|Ψ|² + (α/4)|Ψ|⁴

    Bug fix v4.0: previous version omitted temporal kinetic energy ½|Ψ̇|²/c².
    Code fix: added correct ½ and ¼ prefactors to match paper Eq.(2).
    """
    gx, gy = gradient_2d(psi)
    A2 = np.abs(psi)**2
    # Temporal kinetic energy
    E_temporal = 0.5 * np.abs(pdot)**2 / c2
    # Gradient energy
    E_grad = 0.5 * (np.abs(gx)**2 + np.abs(gy)**2)
    # Mexican-hat potential
    E_pot = -0.5 * lam_val * A2 + 0.25 * alpha_val * A2**2
    return E_temporal + E_grad + E_pot

# ============================================================
# 4. EVOLVE FUNCTION
# ============================================================

def evolve_2d(psi_init, pdot_init, lam_val, alpha_val, nsteps, label=""):
    psi = psi_init.copy()
    pdot = pdot_init.copy()
    gam = sponge_2d(X, Y, Lx, Ly, sponge_width, sponge_str)
    bm = R < R_bound
    ts, Nb, Zb = [], [], []

    for s in range(nsteps):
        psi += pdot * dt / 2
        rhs = ed_rhs(psi, lam_val, alpha_val)
        pdot += (c2 * rhs - gam * pdot) * dt
        psi += pdot * dt / 2

        if s % record_every == 0:
            A = np.abs(psi)
            ts.append(s * dt)
            Nb.append(np.sum(A[bm]**2) * dx * dy)
            wb = A[bm]**2
            Zb.append(np.abs(np.sum(wb*np.exp(1j*np.angle(psi[bm])))) /
                      (np.sum(wb)+1e-30))

    return {
        'label': label, 'psi_f': psi.copy(),
        't': np.array(ts), 'Nb': np.array(Nb), 'Zb': np.array(Zb)
    }

# ============================================================
# 5. PREPARE VORTEX (once, shared by all configs)
# ============================================================
print("\n[1] Relaxing 2D vortex...")

xi = 1.0 / np.sqrt(lam)
f_r = A0 * R / np.sqrt(R**2 + xi**2)
psi_v = (f_r * np.exp(1j * n_wind * THETA)).astype(np.complex128)
pdot_v = np.zeros_like(psi_v)

t0r = time()
for step in range(n_relax):
    psi_v += pdot_v * dt / 2
    rhs = ed_rhs(psi_v, lam, alpha)
    pdot_v += (c2 * rhs - gamma_relax * pdot_v) * dt
    psi_v += pdot_v * dt / 2
    if step % 4000 == 0:
        print(f"    step {step}: A_far={np.mean(np.abs(psi_v[R>20])):.5f}")
pdot_v[:] = 0.0

A_far = np.mean(np.abs(psi_v[R > 20]))
if abs(A_far - A0)/A0 > 0.01:
    psi_v *= A0 / A_far
    for step in range(500):
        psi_v += pdot_v * dt / 2
        rhs = ed_rhs(psi_v, lam, alpha)
        pdot_v += (c2 * rhs - 0.2 * pdot_v) * dt
        psi_v += pdot_v * dt / 2
    pdot_v[:] = 0.0

core_x0, core_y0 = find_core(psi_v)
w_init = measure_winding(psi_v, core_x0, core_y0, 5.0)
print(f"  Done ({time()-t0r:.1f}s). A_far={np.mean(np.abs(psi_v[R>20])):.5f}, "
      f"winding={w_init:.4f}, core=({core_x0:.2f}, {core_y0:.2f})")

# Compute vortex (closure) energy — pdot_v = 0 after relaxation
E_closure_density = compute_total_energy(psi_v, pdot_v, lam, alpha)
# Closure energy = excess over vacuum in bound region
psi_vac_tmp = A0 * np.ones_like(psi_v)
pdot_zero_tmp = np.zeros_like(psi_v)
E_vac_density = compute_total_energy(psi_vac_tmp, pdot_zero_tmp, lam, alpha)
E_closure = float(np.sum((E_closure_density - E_vac_density)[R < R_bound]) * dx * dy)
print(f"  E_closure (bound region) = {E_closure:.4f}")

# ============================================================
# 6. BASELINES (vortex-alone, vacuum)
# ============================================================
psi_vac = A0 * np.ones((Nx, Ny), dtype=np.complex128)
pdot_zero = np.zeros((Nx, Ny), dtype=np.complex128)

print("\n[2] Baselines (vortex-alone, vacuum)...")
t0 = time()
r_vortex = evolve_2d(psi_v, pdot_v, lam, alpha, n_steps, "vortex")
r_vac    = evolve_2d(psi_vac, pdot_zero, lam, alpha, n_steps, "vacuum")
print(f"    done ({time()-t0:.1f}s)")

ref_field = r_vac['psi_f']
ref_mask = (
    (X > 0.68 * Lx) & (X < 0.78 * Lx) &
    (Y > 0.68 * Ly) & (Y < 0.78 * Ly)
)
out_win = outgoing_window(R, R_out_min, R_out_max, width=2.0)

# ============================================================
# 7. ROBUSTNESS SCAN
# ============================================================
print(f"\n[3] Robustness scan: {len(SCAN_CONFIGS)} configs × {len(eps_values)} ε")
print(f"    Total full runs: {len(SCAN_CONFIGS) * len(eps_values)}")

all_results = {}  # config_label -> list of row dicts

for cfg_label, pk, pw in SCAN_CONFIGS:
    print(f"\n{'='*60}")
    print(f"  CONFIG: {cfg_label}  (k={pk}, σ={pw})")
    print(f"{'='*60}")

    packet_x0 = Lx / 4
    packet_y0 = Ly / 2

    rows = []

    for eps in eps_values:
        print(f"\n    ε = {eps} ...", end=" ", flush=True)
        t0 = time()

        # Build packet
        env = eps * np.exp(-((X-packet_x0)**2 + (Y-packet_y0)**2) / pw**2)
        psi_em = env * np.exp(1j * pk * X)
        omega_k = c_ed * np.sqrt(pk**2 + lam)
        pdot_em = -1j * omega_k * psi_em

        # Compute packet energy (for χ ratio) — full ED energy including ½|Ψ̇|²/c²
        E_pkt_total = compute_total_energy(psi_vac + psi_em, pdot_em, lam, alpha)
        E_vac_total = compute_total_energy(psi_vac, pdot_zero, lam, alpha)
        E_pkt_density = E_pkt_total - E_vac_total
        E_packet = float(np.sum(E_pkt_density) * dx * dy)  # signed integral, no abs
        chi = E_packet / (abs(E_closure) + 1e-30)

        # Full evolution
        psi0 = psi_v + psi_em
        pdot0 = pdot_v + pdot_em
        r_full = evolve_2d(psi0, pdot0, lam, alpha, n_steps, f"full_{cfg_label}_{eps}")

        # Packet-alone baseline
        psi_pkt = psi_vac + psi_em
        r_pkt = evolve_2d(psi_pkt, pdot_em, lam, alpha, n_steps, f"pkt_{cfg_label}_{eps}")

        # Phase-aligned residual
        full_a = align_to_reference(r_full['psi_f'], ref_field, ref_mask)
        vort_a = align_to_reference(r_vortex['psi_f'], ref_field, ref_mask)
        pkt_a  = align_to_reference(r_pkt['psi_f'], ref_field, ref_mask)
        vac_a  = align_to_reference(r_vac['psi_f'], ref_field, ref_mask)
        res_psi = full_a - vort_a - pkt_a + vac_a

        # Region-resolved power
        A2_res = np.abs(res_psi)**2
        P_bound = float(np.sum(A2_res[R < R_bound]) * dx * dy)
        P_inter = float(np.sum(A2_res[(R >= R_bound) & (R < R_out_min)]) * dx * dy)
        P_out   = float(np.sum(A2_res[(R >= R_out_min) & (R < R_out_max)]) * dx * dy)

        # k-space angular
        KX, KY, Pk = kspace_power(res_psi, dx, dy, out_win)
        k_ring_min, k_ring_max = 0.5 * pk, 2.0 * pk
        theta_k, Pth = angular_distribution_k(KX, KY, Pk, k_ring_min, k_ring_max)
        Pth_s = smooth_circular(Pth)
        am = angular_metrics(theta_k, Pth_s)

        # Bound-region norm and Z_Phi
        bm = R < R_bound
        Nb_final = float(np.sum(np.abs(r_full['psi_f'][bm])**2) * dx * dy)
        Nb_init  = float(np.sum(np.abs(psi0[bm])**2) * dx * dy)
        dN_bound = Nb_final - Nb_init

        Z_Phi_max = float(np.max(r_full['Zb']))

        # k-space shell power + Goldilocks
        K = np.sqrt(KX**2 + KY**2)
        shell = (K >= k_ring_min) & (K <= k_ring_max)
        Pk_shell = float(np.sum(Pk[shell]))

        # Winding
        cx, cy = find_core(r_full['psi_f'])
        w_final = measure_winding(r_full['psi_f'], cx, cy, 5.0)
        W = 1.0 if abs(w_final - 1.0) < 0.1 else 0.0

        G_k = W * np.log1p(Pk_shell) * am['Q2'] / (am['S_norm'] + 0.01)

        row = {
            'config': cfg_label,
            'k': pk, 'sigma': pw,
            'eps': eps, 'chi': chi,
            'dN_bound': dN_bound,
            'Z_Phi_max': Z_Phi_max,
            'G_k': G_k,
            'winding': float(w_final),
            'S_norm': am['S_norm'],
            'Q2': am['Q2'],
            'P_out': P_out,
            'Pk_shell': Pk_shell,
            'E_packet': E_packet,
            'E_closure': E_closure,
        }
        rows.append(row)
        print(f"({time()-t0:.0f}s) χ={chi:.2f} ΔN={dN_bound:.3f} Z={Z_Phi_max:.3f} G={G_k:.2f} w={w_final:.2f}")

    all_results[cfg_label] = rows

# ============================================================
# 8. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 100)
print(f"{'Config':>8} {'ε':>5} {'χ':>7} {'ΔN_b':>8} {'Z_Φ':>7} {'G_k':>8} {'w':>6} {'S_n':>6} {'Q2':>6} {'P_out':>9}")
print("-" * 100)
for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    for r in all_results[cfg_label]:
        print(f"{r['config']:>8} {r['eps']:>5.2f} {r['chi']:>7.2f} "
              f"{r['dN_bound']:>8.3f} {r['Z_Phi_max']:>7.4f} {r['G_k']:>8.2f} "
              f"{r['winding']:>6.2f} {r['S_norm']:>6.3f} {r['Q2']:>6.3f} "
              f"{r['P_out']:>9.4f}")
    print("-" * 100)

# ============================================================
# 9. ORDERING CHECK
# ============================================================
print("\n" + "=" * 70)
print("  ORDERING ROBUSTNESS CHECK")
print("=" * 70)

for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    rows = all_results[cfg_label]

    # Find peaks
    eps_arr = np.array([r['eps'] for r in rows])
    Z_arr   = np.array([r['Z_Phi_max'] for r in rows])
    G_arr   = np.array([r['G_k'] for r in rows])
    dN_arr  = np.array([r['dN_bound'] for r in rows])
    w_arr   = np.array([r['winding'] for r in rows])

    eps_Z_peak = eps_arr[np.argmax(Z_arr)]
    eps_G_peak = eps_arr[np.argmax(G_arr)]

    # Depletion threshold (first ε where ΔN < 0)
    neg_mask = dN_arr < 0
    eps_dep = eps_arr[neg_mask][0] if np.any(neg_mask) else float('nan')

    # Topological rearrangement (first ε where |w-1| > 0.1, matching paper W definition)
    topo_mask = np.abs(w_arr - 1.0) > 0.1
    eps_topo = eps_arr[topo_mask][0] if np.any(topo_mask) else float('nan')

    # nan-safe ordering: unreached threshold → not a failure
    eps_dep_safe  = eps_dep  if not np.isnan(eps_dep)  else -np.inf
    eps_topo_safe = eps_topo if not np.isnan(eps_topo) else np.inf
    ordering_ok = eps_dep_safe <= eps_Z_peak <= eps_G_peak <= eps_topo_safe

    print(f"\n  {cfg_label}:")
    print(f"    ε_dep  = {eps_dep:.2f}" if not np.isnan(eps_dep) else "    ε_dep  = not reached")
    print(f"    ε_coh  = {eps_Z_peak:.2f}  (Z_Φ peak)")
    print(f"    ε_em   = {eps_G_peak:.2f}  (G_k peak)")
    print(f"    ε_topo = {eps_topo:.2f}" if not np.isnan(eps_topo) else "    ε_topo = not reached")
    print(f"    ORDERING PRESERVED: {'✓ YES' if ordering_ok else '✗ NO'}")

# ============================================================
# 10. FIGURES
# ============================================================

# --- Figure A: Z_Phi and G_k vs ε for all configs ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"EP8 Robustness: Z_Φ and G_k vs ε — {VERSION}", fontsize=13)

colors = {'k4_s3': 'C0', 'k6_s3': 'C1', 'k8_s3': 'C2', 'k6_s2': 'C3', 'k6_s4': 'C4'}
markers = {'k4_s3': 'o', 'k6_s3': 's', 'k8_s3': '^', 'k6_s2': 'D', 'k6_s4': 'v'}

for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    rows = all_results[cfg_label]
    eps_arr = [r['eps'] for r in rows]
    Z_arr   = [r['Z_Phi_max'] for r in rows]
    G_arr   = [r['G_k'] for r in rows]

    axes[0].plot(eps_arr, Z_arr, f'-{markers[cfg_label]}',
                 color=colors[cfg_label], label=cfg_label, markersize=6)
    axes[1].plot(eps_arr, G_arr, f'-{markers[cfg_label]}',
                 color=colors[cfg_label], label=cfg_label, markersize=6)

axes[0].set_xlabel("ε"); axes[0].set_ylabel("Z_Φ (max)")
axes[0].set_title("Phase-order parameter"); axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("ε"); axes[1].set_ylabel("G_k")
axes[1].set_title("Goldilocks score"); axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"ep8_{VERSION}_robustness_ZG.png", dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n  Saved: ep8_{VERSION}_robustness_ZG.png")

# --- Figure B: χ-collapse ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"EP8 χ-Collapse: Z_Φ and G_k vs χ = E_pkt/E_closure — {VERSION}", fontsize=13)

for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    rows = all_results[cfg_label]
    chi_arr = [r['chi'] for r in rows]
    Z_arr   = [r['Z_Phi_max'] for r in rows]
    G_arr   = [r['G_k'] for r in rows]

    axes[0].plot(chi_arr, Z_arr, f'-{markers[cfg_label]}',
                 color=colors[cfg_label], label=cfg_label, markersize=6)
    axes[1].plot(chi_arr, G_arr, f'-{markers[cfg_label]}',
                 color=colors[cfg_label], label=cfg_label, markersize=6)

axes[0].set_xlabel("χ = E_pkt / E_closure"); axes[0].set_ylabel("Z_Φ (max)")
axes[0].set_title("Phase-order parameter vs χ"); axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel("χ = E_pkt / E_closure"); axes[1].set_ylabel("G_k")
axes[1].set_title("Goldilocks score vs χ"); axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"ep8_{VERSION}_chi_collapse.png", dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: ep8_{VERSION}_chi_collapse.png")

# --- Figure C: Ordering diagram ---
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(f"EP8 Regime Ordering Robustness — {VERSION}", fontsize=13)

y_pos = 0
labels_done = set()
for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    rows = all_results[cfg_label]
    eps_arr = np.array([r['eps'] for r in rows])
    Z_arr   = np.array([r['Z_Phi_max'] for r in rows])
    G_arr   = np.array([r['G_k'] for r in rows])
    dN_arr  = np.array([r['dN_bound'] for r in rows])
    w_arr   = np.array([r['winding'] for r in rows])

    eps_Z_peak = eps_arr[np.argmax(Z_arr)]
    eps_G_peak = eps_arr[np.argmax(G_arr)]
    neg_mask = dN_arr < 0
    eps_dep = eps_arr[neg_mask][0] if np.any(neg_mask) else 0.2
    topo_mask = np.abs(w_arr - 1.0) > 0.5
    eps_topo = eps_arr[topo_mask][0] if np.any(topo_mask) else 1.1

    for val, color, lbl in [
        (eps_dep, 'blue', 'ε_dep'),
        (eps_Z_peak, 'green', 'ε_coh'),
        (eps_G_peak, 'red', 'ε_em'),
        (eps_topo, 'purple', 'ε_topo')
    ]:
        label = lbl if lbl not in labels_done else None
        ax.scatter(val, y_pos, color=color, s=100, zorder=5, label=label)
        labels_done.add(lbl)

    ax.plot([eps_dep, eps_Z_peak, eps_G_peak, eps_topo], [y_pos]*4,
            'k-', alpha=0.3, linewidth=1)
    ax.text(-0.02, y_pos, cfg_label, ha='right', va='center', fontsize=9)
    y_pos += 1

ax.set_xlabel("ε", fontsize=12)
ax.set_yticks([])
ax.set_xlim(-0.1, 1.2)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
ax.set_title("Regime boundaries for each (k, σ) configuration")

plt.tight_layout()
plt.savefig(f"ep8_{VERSION}_ordering.png", dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  Saved: ep8_{VERSION}_ordering.png")

# ============================================================
# 11. SAVE DATA
# ============================================================
save_data = {}
for cfg_label in [c[0] for c in SCAN_CONFIGS]:
    save_data[cfg_label] = all_results[cfg_label]

with open(f"ep8_{VERSION}_robustness.json", 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"  Saved: ep8_{VERSION}_robustness.json")

print("\n" + "=" * 70)
print("  DONE.")
print("=" * 70)
