#!/usr/bin/env python3
"""
=============================================================================
EP8 Step 10: Matched-ℓ_reg Resolution Test — v1.2
=============================================================================

Purpose:
  Compare 256² (L=60) and 512² (L=120) at IDENTICAL Δx = 0.234375.

  Same registration scale ℓ_reg.  Same universe.
  The only difference is domain size — more room for outgoing radiation.

  Previous versions (v1.0, v1.1) compared different Δx on the same L,
  which per Corollary 1.1 (Shin 2026) amounts to comparing two
  different finite-registration worlds.  Energy-like quantities
  (Pk_shell, G_k) are ℓ_reg-dependent and cannot be matched
  across different Δx.

  This version fixes Δx and varies L, so any remaining difference
  is attributable to boundary effects, not registration scale.

  Packet-to-vortex distance is held constant at 15.0 in both runs.

Equation:
  (1/c²)Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ

Author: Jae-Ahn Shin
        Independent Researcher, Incheon, Republic of Korea
        jaeahn.shin.official@gmail.com
=============================================================================
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time

VERSION = "v1.2"
DPI = 600

# ============================================================
# PHYSICAL PARAMETERS (shared)
# ============================================================
lam   = 1.0
alpha = 1.0
c_ed  = 1.0
A0    = np.sqrt(lam / alpha)
c2    = c_ed**2
n_wind = 1

eps_target = 0.7
pk = 6.0
pw = 4.0

sponge_width = 8.0
sponge_str   = 2.5
gamma_relax  = 0.4

dt       = 0.003        # SAME dt — same Δx, same CFL
n_relax  = 24000        # same physical relaxation time T=72
n_steps  = 8000         # same physical evolution time T=24
record_every = 40

# Diagnostic radii (physical, same in both)
R_bound   = 6.0
R_out_min = 13.0
R_out_max = 22.0
N_THETA_BINS = 90

# Packet-to-vortex distance (fixed physical distance)
PACKET_DIST = 15.0

# ============================================================
# GRID CONFIGS: SAME Δx, DIFFERENT L
# ============================================================
# Δx = 60/256 = 120/512 = 0.234375
GRIDS = [
    # (label, N, L)
    ("256² L=60",  256,  60.0),
    ("512² L=120", 512, 120.0),
]

print("=" * 70)
print(f"EP8 Step 10: Matched-ℓ_reg Resolution Test — {VERSION}")
print(f"  ε={eps_target}, k={pk}, σ={pw}")
print(f"  Δx = {60.0/256:.6f} (identical for both grids)")
print(f"  dt = {dt}, n_steps = {n_steps}, T_final = {n_steps*dt:.1f}")
print(f"  Packet-to-vortex distance = {PACKET_DIST}")
for label, N, L in GRIDS:
    print(f"    {label}: N={N}, L={L}, Δx={L/N:.6f}")
print("=" * 70)

# ============================================================
# CORE FUNCTIONS
# ============================================================

def make_grid(N, L):
    dx = L / N
    dy = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    xc, yc = L/2, L/2
    R = np.sqrt((X - xc)**2 + (Y - yc)**2)
    THETA = np.arctan2(Y - yc, X - xc)
    return x, y, X, Y, R, THETA, dx, dy, xc, yc

def laplacian_2d(psi, dx, dy):
    lap  = (np.roll(psi, -1, 0) + np.roll(psi, 1, 0) - 2*psi) / dx**2
    lap += (np.roll(psi, -1, 1) + np.roll(psi, 1, 1) - 2*psi) / dy**2
    return lap

def ed_rhs(psi, lam_val, alpha_val, dx, dy):
    return laplacian_2d(psi, dx, dy) + lam_val * psi - alpha_val * np.abs(psi)**2 * psi

def gradient_2d(psi, dx, dy):
    gx = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2*dx)
    gy = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2*dy)
    return gx, gy

def sponge_2d(XX, YY, L, w, s):
    g = np.zeros_like(XX)
    g = np.where(XX < w,    np.maximum(g, s*((w-XX)/w)**2), g)
    g = np.where(XX > L-w,  np.maximum(g, s*((XX-(L-w))/w)**2), g)
    g = np.where(YY < w,    np.maximum(g, s*((w-YY)/w)**2), g)
    g = np.where(YY > L-w,  np.maximum(g, s*((YY-(L-w))/w)**2), g)
    return g

def find_core(psi, R, x, y, search_radius=10.0):
    mask = R < search_radius
    A = np.abs(psi)
    vals = np.where(mask, A, np.inf)
    idx = np.argmin(vals)
    ix, iy = np.unravel_index(idx, psi.shape)
    return float(x[ix]), float(y[iy])

def measure_winding(psi, cx, cy, r_check, dx, dy, N):
    n_pts = 120
    th = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    ix = ((cx + r_check*np.cos(th))/dx).astype(int) % N
    iy = ((cy + r_check*np.sin(th))/dy).astype(int) % N
    phase = np.angle(psi[ix, iy])
    dphi = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    return np.sum(dphi) / (2*np.pi)

def align_to_reference(psi, ref, mask):
    z = np.sum(np.conj(ref[mask]) * psi[mask])
    return psi * np.exp(-1j * np.angle(z))

def smooth_step(z, z0, width):
    return 0.5 * (1.0 + np.tanh((z - z0) / width))

def outgoing_window(RR, R_inner, R_outer, width=2.0):
    return smooth_step(RR, R_inner, width) * (1.0 - smooth_step(RR, R_outer, width))

def kspace_power(field, dx, dy, window):
    f = field * window
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F)**2
    kx = np.fft.fftshift(np.fft.fftfreq(field.shape[0], dx)) * 2*np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(field.shape[1], dy)) * 2*np.pi
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

# ============================================================
# RUN ONE GRID
# ============================================================

def run_one_grid(label, N, L):
    print(f"\n{'='*60}")
    print(f"  GRID: {label}  (N={N}, L={L}, Δx={L/N:.6f})")
    print(f"{'='*60}")

    x, y, X, Y, R, THETA, dx, dy, xc, yc = make_grid(N, L)
    gam = sponge_2d(X, Y, L, sponge_width, sponge_str)
    bm = R < R_bound
    out_win = outgoing_window(R, R_out_min, R_out_max, width=2.0)

    def evolve(psi_init, pdot_init, nsteps, elabel=""):
        psi = psi_init.copy()
        pdot = pdot_init.copy()
        ts, Nb, Zb = [], [], []
        for s in range(nsteps):
            psi += pdot * dt / 2
            rhs = ed_rhs(psi, lam, alpha, dx, dy)
            pdot += (c2 * rhs - gam * pdot) * dt
            psi += pdot * dt / 2
            if s % record_every == 0:
                A = np.abs(psi)
                ts.append(s * dt)
                Nb.append(np.sum(A[bm]**2) * dx * dy)
                wb = A[bm]**2
                Zb.append(np.abs(np.sum(wb*np.exp(1j*np.angle(psi[bm])))) /
                          (np.sum(wb)+1e-30))
        return {'label': elabel, 'psi_f': psi.copy(),
                't': np.array(ts), 'Nb': np.array(Nb), 'Zb': np.array(Zb)}

    # --- 1. Relax vortex ---
    print(f"  [1] Relaxing vortex...")
    xi = 1.0 / np.sqrt(lam)
    f_r = A0 * R / np.sqrt(R**2 + xi**2)
    psi_v = (f_r * np.exp(1j * n_wind * THETA)).astype(np.complex128)
    pdot_v = np.zeros_like(psi_v)

    t0 = time()
    for step in range(n_relax):
        psi_v += pdot_v * dt / 2
        rhs = ed_rhs(psi_v, lam, alpha, dx, dy)
        pdot_v += (c2 * rhs - gamma_relax * pdot_v) * dt
        psi_v += pdot_v * dt / 2
        if step % 8000 == 0:
            print(f"      step {step}: A_far={np.mean(np.abs(psi_v[R>20])):.5f}")
    pdot_v[:] = 0.0

    A_far = np.mean(np.abs(psi_v[R > 20]))
    if abs(A_far - A0)/A0 > 0.01:
        psi_v *= A0 / A_far
        for step in range(500):
            psi_v += pdot_v * dt / 2
            rhs = ed_rhs(psi_v, lam, alpha, dx, dy)
            pdot_v += (c2 * rhs - 0.2 * pdot_v) * dt
            psi_v += pdot_v * dt / 2
        pdot_v[:] = 0.0

    A_far = np.mean(np.abs(psi_v[R > 20]))
    w_init = measure_winding(psi_v, xc, yc, 5.0, dx, dy, N)
    print(f"      Done ({time()-t0:.1f}s). A_far={A_far:.5f}, winding={w_init:.4f}")

    # --- 2. Baselines ---
    print(f"  [2] Baselines...")
    psi_vac = A0 * np.ones((N, N), dtype=np.complex128)
    pdot_zero = np.zeros((N, N), dtype=np.complex128)

    t0 = time()
    r_vortex = evolve(psi_v, pdot_v, n_steps, "vortex")
    r_vac    = evolve(psi_vac, pdot_zero, n_steps, "vacuum")
    print(f"      done ({time()-t0:.1f}s)")

    ref_field = r_vac['psi_f']
    # Reference mask: fixed physical offset from center (same for both grids)
    ref_offset_lo = 15.0  # physical units from center
    ref_offset_hi = 18.0
    ref_mask = (
        (X > xc + ref_offset_lo) & (X < xc + ref_offset_hi) &
        (Y > yc + ref_offset_lo) & (Y < yc + ref_offset_hi)
    )

    # --- 3. Full + packet ---
    # Packet at fixed physical distance from vortex
    packet_x0 = xc - PACKET_DIST   # 15 units left of center
    packet_y0 = yc                  # same y as vortex
    print(f"  [3] Full run: ε={eps_target}, packet at ({packet_x0:.1f}, {packet_y0:.1f})...")

    t0 = time()
    env = eps_target * np.exp(-((X-packet_x0)**2 + (Y-packet_y0)**2) / pw**2)
    psi_em = env * np.exp(1j * pk * X)
    omega_k = c_ed * np.sqrt(pk**2 + lam)
    pdot_em = -1j * omega_k * psi_em

    psi0 = psi_v + psi_em
    pdot0 = pdot_v + pdot_em
    r_full = evolve(psi0, pdot0, n_steps, "full")

    psi_pkt = psi_vac + psi_em
    r_pkt = evolve(psi_pkt, pdot_em, n_steps, "packet")
    print(f"      done ({time()-t0:.1f}s)")

    # --- 4. Diagnostics ---
    print(f"  [4] Diagnostics...")
    full_a = align_to_reference(r_full['psi_f'], ref_field, ref_mask)
    vort_a = align_to_reference(r_vortex['psi_f'], ref_field, ref_mask)
    pkt_a  = align_to_reference(r_pkt['psi_f'], ref_field, ref_mask)
    vac_a  = align_to_reference(r_vac['psi_f'], ref_field, ref_mask)
    res_psi = full_a - vort_a - pkt_a + vac_a

    A2_res = np.abs(res_psi)**2
    P_out = float(np.sum(A2_res[(R >= R_out_min) & (R < R_out_max)]) * dx * dy)

    KX, KY, Pk = kspace_power(res_psi, dx, dy, out_win)
    k_ring_min, k_ring_max = 0.5 * pk, 2.0 * pk
    theta_k, Pth = angular_distribution_k(KX, KY, Pk, k_ring_min, k_ring_max)
    Pth_s = smooth_circular(Pth)
    am = angular_metrics(theta_k, Pth_s)

    Nb_final = float(np.sum(np.abs(r_full['psi_f'][bm])**2) * dx * dy)
    Nb_init  = float(np.sum(np.abs(psi0[bm])**2) * dx * dy)
    dN_bound = Nb_final - Nb_init
    Z_Phi_max = float(np.max(r_full['Zb']))

    K = np.sqrt(KX**2 + KY**2)
    shell = (K >= k_ring_min) & (K <= k_ring_max)
    Pk_shell = float(np.sum(Pk[shell]))

    cx, cy = find_core(r_full['psi_f'], R, x, y)
    w_final = measure_winding(r_full['psi_f'], cx, cy, 5.0, dx, dy, N)
    W = 1.0 if abs(w_final - 1.0) < 0.1 else 0.0
    G_k = W * np.log1p(Pk_shell) * am['Q2'] / (am['S_norm'] + 0.01)

    result = {
        'label': label, 'N': N, 'L': L, 'dx': dx,
        'dN_bound': dN_bound, 'Z_Phi_max': Z_Phi_max,
        'G_k': G_k, 'winding': float(w_final),
        'S_norm': am['S_norm'], 'Q2': am['Q2'],
        'P_out': P_out, 'Pk_shell': Pk_shell,
        'A_far': A_far,
        'theta_k': theta_k, 'Pth_s': Pth_s,
        'Zb_t': r_full['t'], 'Zb': r_full['Zb'],
    }

    print(f"      Z_Φ={Z_Phi_max:.4f}, G_k={G_k:.2f}, S={am['S_norm']:.3f}, "
          f"Q2={am['Q2']:.3f}, w={w_final:.2f}, Pk_shell={Pk_shell:.1f}")

    return result

# ============================================================
# RUN BOTH
# ============================================================
results = {}
for label, N, L in GRIDS:
    results[label] = run_one_grid(label, N, L)

r256 = results["256² L=60"]
r512 = results["512² L=120"]

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "=" * 78)
print(f"  MATCHED-ℓ_reg COMPARISON: Δx = {r256['dx']:.6f} (identical)")
print(f"  256² (L=60) vs 512² (L=120)")
print("=" * 78)
print(f"  {'Metric':>14}  {'256² L=60':>12}  {'512² L=120':>12}  {'Ratio':>8}  {'Pass':>6}")
print("-" * 78)

compare_keys = [
    ('A_far',      'A_far',      0.03),
    ('dN_bound',   'ΔN_bound',   None),
    ('Z_Phi_max',  'Z_Φ_max',    0.30),
    ('G_k',        'G_k',        0.30),
    ('winding',    'winding',    0.10),
    ('S_norm',     'S_norm',     0.10),
    ('Q2',         'Q2',         0.10),
    ('P_out',      'P_out',      0.30),
    ('Pk_shell',   'Pk_shell',   0.50),
]

for key, display, tol in compare_keys:
    v256 = r256[key]
    v512 = r512[key]
    if abs(v256) > 1e-10:
        ratio = v512 / v256
        if tol is not None:
            passed = "✓" if abs(ratio - 1.0) < tol else "✗"
        else:
            passed = "—"
        print(f"  {display:>14}  {v256:>12.4f}  {v512:>12.4f}  {ratio:>8.3f}  {passed:>6}")
    else:
        print(f"  {display:>14}  {v256:>12.4f}  {v512:>12.4f}  {'---':>8}  {'—':>6}")

print("-" * 78)

# ============================================================
# FIGURES
# ============================================================

# --- Figure 1: Angular + Z_Phi ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"EP8 Matched-ℓ_reg: Δx={r256['dx']:.4f}, 256² vs 512² — {VERSION}",
             fontsize=13)

for r, color, ls in [(r256, 'r', '--'), (r512, 'b', '-')]:
    Pn = r['Pth_s'] / (np.sum(r['Pth_s']) + 1e-30)
    axes[0].plot(r['theta_k'] * 180/np.pi, Pn,
                 color=color, ls=ls, linewidth=1.5,
                 label=f"{r['label']} (S={r['S_norm']:.3f})")
axes[0].set_xlabel("θ_k (deg)")
axes[0].set_ylabel("P(θ_k) normalized")
axes[0].set_title("(a) Angular distribution")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for r, color, ls in [(r256, 'r', '--'), (r512, 'b', '-')]:
    axes[1].plot(r['Zb_t'], r['Zb'],
                 color=color, ls=ls, linewidth=1,
                 label=f"{r['label']} (max={r['Z_Phi_max']:.3f})")
axes[1].set_xlabel("t")
axes[1].set_ylabel("Z_Φ(t)")
axes[1].set_title("(b) Phase-order parameter")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"ep8_matched_reg_{VERSION}.png", dpi=DPI, bbox_inches='tight')
plt.close()
print(f"\n  Saved: ep8_matched_reg_{VERSION}.png")

# Save reproducibility data
npz_file = f"matched_reg_{VERSION}.npz"
np.savez(npz_file,
         labels=np.array([r['label'] for r in results.values()]),
         grids=np.array([r['N'] for r in results.values()]),
         domains=np.array([r['L'] for r in results.values()]),
         A_far=np.array([r['A_far'] for r in results.values()]),
         Z_Phi_max=np.array([r['Z_Phi_max'] for r in results.values()]),
         S_norm=np.array([r['S_norm'] for r in results.values()]),
         Q2=np.array([r['Q2'] for r in results.values()]),
         G_k=np.array([r['G_k'] for r in results.values()]),
         P_out=np.array([r['P_out'] for r in results.values()]),
         winding=np.array([r['winding'] for r in results.values()]))
print(f"  Saved reproducibility data: {npz_file}")

print("\n" + "=" * 70)
print("  DONE.")
print("=" * 70)
