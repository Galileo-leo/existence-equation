#!/usr/bin/env python3
"""
=============================================================================
EP8 Step 8: 2D Vortex — v3.2 (ε-refined scan)
=============================================================================

Purpose:
  Refine the ε scan to map the four-regime hierarchy precisely:
    I.   Weak response          (low ε)
    II.  Phase-ordering peak    (Z_Φ max)
    III. Structured emission    (G_k max)
    IV.  Overdriven depletion   (entropy rise, norm loss)

  Key question: locate boundaries between regimes, especially
    - bound-sector depletion threshold (ΔN_bound = 0)
    - Z_Φ peak vs G_k peak separation

  No fitted strong-field machinery is inserted.
  No phenomenological ATI coupling, Volkov state, SFA,
  or transition matrix is imposed.

Base: v3.1 (Leo minor fixes) with overlap alignment,
      bulk patch ref, core tracking, normalized entropy.

Changes from v3.1:
  - eps = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
  - Version label in ALL figure titles
  - Leo terminology in labels
  - Interpolated regime boundary estimates

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
from scipy.interpolate import interp1d
from time import time

VERSION = "v3.2"
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

packet_x0    = Lx / 4
packet_y0    = Ly / 2
packet_width = 4.0
packet_k     = 6.0

sponge_width = 8.0
sponge_str   = 2.5
n_relax      = 12000
gamma_relax  = 0.4
n_steps      = 8000
record_every = 40

# REFINED ε SCAN (Leo recommendation)
eps_values = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

weak_scale = 0.1
alpha_weak = weak_scale * alpha
lam_weak   = weak_scale * lam

R_bound   = 6.0
R_inter   = 15.0
R_out_min = 13.0
R_out_max = 22.0

N_THETA_BINS = 90

print("=" * 70)
print(f"EP8 Step 8: 2D Vortex {VERSION} (ε-refined scan)")
print("  Mapping four-regime hierarchy:")
print("    I.  Weak response  II. Phase-ordering  III. Structured emission")
print("    IV. Overdriven depletion")
print(f"  Grid: {Nx}×{Ny}, Lx={Lx}, dx={dx:.4f}, dt={dt}")
print(f"  λ={lam}, α={alpha}, A0={A0:.4f}")
print(f"  Weak: λ_w={lam_weak}, α_w={alpha_weak}")
print(f"  Packet: k={packet_k}, width={packet_width}")
print(f"  ε values: {eps_values}")
print("=" * 70)

# ============================================================
# 2. CORE ED
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

def sponge_2d(X, Y, Lx, Ly, w, s):
    g = np.zeros_like(X)
    g = np.where(X < w,      np.maximum(g, s*((w-X)/w)**2), g)
    g = np.where(X > Lx-w,   np.maximum(g, s*((X-(Lx-w))/w)**2), g)
    g = np.where(Y < w,      np.maximum(g, s*((w-Y)/w)**2), g)
    g = np.where(Y > Ly-w,   np.maximum(g, s*((Y-(Ly-w))/w)**2), g)
    return g

def find_core(psi, search_radius=10.0):
    """Find vortex core position (minimum |Ψ| near center)."""
    mask = R < search_radius
    A = np.abs(psi)
    vals = np.where(mask, A, np.inf)
    idx = np.argmin(vals)
    ix, iy = np.unravel_index(idx, psi.shape)
    return float(x[ix]), float(y[iy])

def measure_winding(psi, cx, cy, r_check):
    """Closed-loop link-product winding around (cx, cy)."""
    n_pts = 120
    th = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
    ix = ((cx + r_check*np.cos(th))/dx).astype(int) % Nx
    iy = ((cy + r_check*np.sin(th))/dy).astype(int) % Ny
    phase = np.angle(psi[ix, iy])
    dphi = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    return np.sum(dphi) / (2*np.pi)

# ============================================================
# 3. ALIGNMENT + WINDOW + k-SPACE + METRICS
# ============================================================

def align_to_reference(psi, ref, mask):
    """Align psi's global phase to reference field via overlap."""
    z = np.sum(np.conj(ref[mask]) * psi[mask])
    return psi * np.exp(-1j * np.angle(z))

def smooth_step(z, z0, width):
    return 0.5 * (1.0 + np.tanh((z - z0) / width))

def outgoing_window(R, R_inner, R_outer, width=2.0):
    return smooth_step(R, R_inner, width) * (1.0 - smooth_step(R, R_outer, width))

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
    """Circular smoothing for angular data."""
    out = arr.copy()
    for _ in range(width):
        out = 0.25*np.roll(out, 1) + 0.5*out + 0.25*np.roll(out, -1)
    return out

def angular_metrics(theta, Ptheta, n_bins=90):
    """Angular structure metrics with normalized entropy."""
    total = np.sum(Ptheta) + 1e-30
    p = Ptheta / total

    S_raw = -np.sum(p * np.log(p + 1e-30))
    S_max = np.log(n_bins)
    S_norm = S_raw / S_max

    Q2 = np.abs(np.sum(p * np.exp(2j * theta)))

    d = np.pi / 6
    fwd = np.sum(Ptheta[np.abs(theta) < d])
    bwd_mask = (np.abs(theta) > np.pi - d)
    bwd = np.sum(Ptheta[bwd_mask])

    side_mask = (np.abs(np.abs(theta) - np.pi/2) < d)
    side = np.sum(Ptheta[side_mask])

    Pn = Ptheta / (np.max(Ptheta) + 1e-30)
    Pn_s = smooth_circular(Pn, width=2)
    # Cyclic peak detection: pad array to catch peaks at domain boundary
    pad = 5
    Pn_padded = np.concatenate([Pn_s[-pad:], Pn_s, Pn_s[:pad]])
    peaks_padded, _ = find_peaks(Pn_padded, height=0.1, distance=4)
    # Map back to original indices, deduplicate via modulo
    peaks_orig = (peaks_padded - pad) % len(Pn_s)
    n_lobes = len(np.unique(peaks_orig))

    return {
        'S_theta': float(S_raw),
        'S_norm': float(S_norm),
        'Q2': float(Q2),
        'FB': float(fwd / (bwd + 1e-30)),
        'Side_Fwd': float(side / (fwd + 1e-30)),
        'n_lobes': n_lobes
    }

def current_density(psi):
    gx, gy = gradient_2d(psi)
    jx = np.imag(np.conj(psi) * gx)
    jy = np.imag(np.conj(psi) * gy)
    return jx, jy

# ============================================================
# 4. PREPARE VORTEX
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
      f"A_core={np.abs(psi_v[Nx//2,Ny//2]):.5f}, winding={w_init:.4f}")
print(f"  Core position: ({core_x0:.2f}, {core_y0:.2f})")

# ============================================================
# 5. EVOLUTION
# ============================================================

def evolve_2d(psi_init, pdot_init, lam_val, alpha_val, n_steps, label=""):
    psi = psi_init.copy()
    pdot = pdot_init.copy()
    gam = sponge_2d(X, Y, Lx, Ly, sponge_width, sponge_str)

    bm = R < R_bound
    ts, Nb, Zb = [], [], []
    snaps, stimes = [], []
    snap_every = max(1, n_steps // 10)

    for s in range(n_steps):
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

        if s % snap_every == 0:
            snaps.append(psi.copy())
            stimes.append(s * dt)

    return {
        'label': label, 'psi_f': psi.copy(),
        't': np.array(ts), 'Nb': np.array(Nb), 'Zb': np.array(Zb),
        'snaps': snaps, 'stimes': np.array(stimes)
    }

# ============================================================
# 6. REFERENCE MASK + OUTGOING WINDOW
# ============================================================
ref_mask = (
    (X > 0.68 * Lx) & (X < 0.78 * Lx) &
    (Y > 0.68 * Ly) & (Y < 0.78 * Ly)
)
out_win = outgoing_window(R, R_out_min, R_out_max, width=2.0)

print(f"\n  Reference patch: {np.sum(ref_mask)} pixels")
print(f"  Outgoing window: R_in={R_out_min}, R_out={R_out_max}")

# ============================================================
# 7. BASELINES
# ============================================================
psi_vac = A0 * np.ones((Nx, Ny), dtype=np.complex128)
pdot_zero = np.zeros((Nx, Ny), dtype=np.complex128)

print("\n[2] Baselines...")
t0 = time()
r_vortex = evolve_2d(psi_v, pdot_v, lam, alpha, n_steps, "vortex")
r_vac    = evolve_2d(psi_vac, pdot_zero, lam, alpha, n_steps, "vacuum")
print(f"    done ({time()-t0:.1f}s)")

ref_field = r_vac['psi_f']

# ============================================================
# 8. ε SCAN
# ============================================================
print(f"\n[3] ε scan (refined): {eps_values}")

results = {}
rows = []

for eps in eps_values:
    print(f"\n  --- ε = {eps} ---")

    # Packet
    env = eps * np.exp(-((X-packet_x0)**2 + (Y-packet_y0)**2) / packet_width**2)
    psi_em = env * np.exp(1j * packet_k * X)
    omega_k = c_ed * np.sqrt(packet_k**2 + lam)
    pdot_em = -1j * omega_k * psi_em

    psi0 = psi_v + psi_em
    pdot0 = pdot_v + pdot_em

    # Full ED
    t0 = time()
    r_full = evolve_2d(psi0, pdot0, lam, alpha, n_steps, f"full_{eps}")
    tf = time() - t0

    # Weak control
    r_weak = evolve_2d(psi0, pdot0, lam_weak, alpha_weak, n_steps, f"weak_{eps}")

    # Packet alone
    psi_pkt = psi_vac + psi_em
    r_pkt = evolve_2d(psi_pkt, pdot_em, lam, alpha, n_steps, f"pkt_{eps}")

    # === PHASE-ALIGNED RESIDUAL (overlap method) ===
    full_a = align_to_reference(r_full['psi_f'], ref_field, ref_mask)
    vort_a = align_to_reference(r_vortex['psi_f'], ref_field, ref_mask)
    pkt_a  = align_to_reference(r_pkt['psi_f'], ref_field, ref_mask)
    vac_a  = align_to_reference(r_vac['psi_f'], ref_field, ref_mask)

    res_psi = full_a - vort_a - pkt_a + vac_a

    res_rho = (np.abs(r_full['psi_f'])**2 - np.abs(r_vortex['psi_f'])**2
               - np.abs(r_pkt['psi_f'])**2 + np.abs(r_vac['psi_f'])**2)

    jx_f, jy_f = current_density(r_full['psi_f'])
    jx_v, jy_v = current_density(r_vortex['psi_f'])
    jx_p, jy_p = current_density(r_pkt['psi_f'])
    jx_0, jy_0 = current_density(r_vac['psi_f'])
    res_jx = jx_f - jx_v - jx_p + jx_0
    res_jy = jy_f - jy_v - jy_p + jy_0
    res_j_mag = np.sqrt(res_jx**2 + res_jy**2)

    # === REGION-RESOLVED RESIDUAL POWER ===
    # Regions: bound [0, R_bound), inter [R_bound, R_out_min), out [R_out_min, R_out_max)
    # No overlap between inter and out (Leo bug fix)
    A2_res = np.abs(res_psi)**2
    P_bound = float(np.sum(A2_res[R < R_bound]) * dx * dy)
    P_inter = float(np.sum(A2_res[(R >= R_bound) & (R < R_out_min)]) * dx * dy)
    P_out   = float(np.sum(A2_res[(R >= R_out_min) & (R < R_out_max)]) * dx * dy)

    # === OUTGOING k-SPACE ===
    KX_k, KY_k, Pk = kspace_power(res_psi, dx, dy, out_win)
    K_mag = np.sqrt(KX_k**2 + KY_k**2)

    k_shell = (K_mag >= 0.5*packet_k) & (K_mag <= 2.0*packet_k)
    Pk_shell = float(np.sum(Pk[k_shell]))

    theta_k, Ptheta = angular_distribution_k(KX_k, KY_k, Pk,
                                              0.5*packet_k, 2.0*packet_k,
                                              n_theta=N_THETA_BINS)
    metrics = angular_metrics(theta_k, Ptheta, n_bins=N_THETA_BINS)

    # Winding with core tracking
    core_xf, core_yf = find_core(r_full['psi_f'])
    w_after = measure_winding(r_full['psi_f'], core_xf, core_yf, 5.0)
    core_shift = np.sqrt((core_xf - core_x0)**2 + (core_yf - core_y0)**2)

    # Z_Φ and ΔN
    Zb_final = float(r_full['Zb'][-1])
    dNb = float(r_full['Nb'][-1] - r_full['Nb'][0])

    # === GOLDILOCKS SCORES ===
    W = 1.0 if abs(w_after - 1.0) < 0.1 else 0.0
    G_real = W * np.log1p(P_out) * metrics['Q2'] / (metrics['S_norm'] + 0.01)
    G_k    = W * np.log1p(Pk_shell) * metrics['Q2'] / (metrics['S_norm'] + 0.01)

    print(f"    {tf:.1f}s | wind={w_after:.3f} (core shift={core_shift:.2f})")
    print(f"    Z_Φ={Zb_final:.4f} | ΔNb={dNb:.3f}")
    print(f"    P_res: bnd={P_bound:.2f} int={P_inter:.2f} out={P_out:.2f}")
    print(f"    k-shell={Pk_shell:.2e}")
    print(f"    S_norm={metrics['S_norm']:.4f} Q₂={metrics['Q2']:.4f} "
          f"FB={metrics['FB']:.3f} Side={metrics['Side_Fwd']:.3f} "
          f"lobes={metrics['n_lobes']}")
    print(f"    ★ G_real={G_real:.4f}  G_k={G_k:.4f}")

    row = {
        'eps': eps, 'w_after': float(w_after), 'core_shift': float(core_shift),
        'Zb_final': Zb_final, 'dNb': dNb,
        'P_bound': P_bound, 'P_inter': P_inter, 'P_out': P_out,
        'Pk_shell': Pk_shell,
        'S_theta': metrics['S_theta'], 'S_norm': metrics['S_norm'],
        'Q2': metrics['Q2'], 'FB': metrics['FB'],
        'Side_Fwd': metrics['Side_Fwd'], 'n_lobes': metrics['n_lobes'],
        'G_real': float(G_real), 'G_k': float(G_k)
    }
    rows.append(row)

    results[eps] = {
        'full': r_full, 'weak': r_weak, 'pkt': r_pkt,
        'res_psi': res_psi, 'res_rho': res_rho,
        'res_jx': res_jx, 'res_jy': res_jy, 'res_j_mag': res_j_mag,
        'P_bound': P_bound, 'P_inter': P_inter, 'P_out': P_out,
        'KX': KX_k, 'KY': KY_k, 'Pk': Pk, 'Pk_shell': Pk_shell,
        'theta_k': theta_k, 'Ptheta': Ptheta, 'metrics': metrics,
        'w_after': w_after, 'core_shift': core_shift,
        'Zb_final': Zb_final, 'dNb': dNb,
        'G_real': G_real, 'G_k': G_k
    }

# ============================================================
# 9. SAVE DATA
# ============================================================
with open(f"ep8_{VERSION}_summary.json", "w", encoding="utf-8") as f:
    json.dump({"version": VERSION, "rows": rows}, f, indent=2)
print(f"\nSaved: ep8_{VERSION}_summary.json")

eps_arr = np.array([r['eps'] for r in rows])
np.savez(f"ep8_{VERSION}_arrays.npz",
    eps=eps_arr,
    G_real=np.array([r['G_real'] for r in rows]),
    G_k=np.array([r['G_k'] for r in rows]),
    Zb=np.array([r['Zb_final'] for r in rows]),
    dNb=np.array([r['dNb'] for r in rows]),
    P_out=np.array([r['P_out'] for r in rows]),
    Pk_shell=np.array([r['Pk_shell'] for r in rows]),
    Q2=np.array([r['Q2'] for r in rows]),
    S_norm=np.array([r['S_norm'] for r in rows]),
    FB=np.array([r['FB'] for r in rows]),
    wind=np.array([r['w_after'] for r in rows]),
    core_shift=np.array([r['core_shift'] for r in rows]),
)
print(f"Saved: ep8_{VERSION}_arrays.npz")

# ============================================================
# 10. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 130)
print(f"EP8 {VERSION} — SUMMARY (ε-refined scan)")
print("=" * 130)
print(f"{'ε':>5}  {'wind':>5}  {'Δcore':>5}  {'Z_Φ':>6}  {'ΔNb':>7}  "
      f"{'P_out':>8}  {'Pk_sh':>9}  "
      f"{'S_n':>5}  {'Q₂':>6}  {'FB':>6}  {'lobe':>4}  "
      f"{'G_real':>8}  {'G_k':>8}")
print("-" * 130)

G_real_arr = np.array([results[e]['G_real'] for e in eps_values])
G_k_arr = np.array([results[e]['G_k'] for e in eps_values])

for eps in eps_values:
    r = results[eps]
    m = r['metrics']
    print(f"{eps:5.2f}  {r['w_after']:5.3f}  {r['core_shift']:5.2f}  "
          f"{r['Zb_final']:6.4f}  {r['dNb']:7.3f}  "
          f"{r['P_out']:8.2f}  {r['Pk_shell']:9.2e}  "
          f"{m['S_norm']:5.3f}  {m['Q2']:6.4f}  {m['FB']:6.3f}  "
          f"{m['n_lobes']:4d}  "
          f"{r['G_real']:8.3f}  {r['G_k']:8.3f}")

best_r = eps_values[np.argmax(G_real_arr)]
best_k = eps_values[np.argmax(G_k_arr)]
print("-" * 130)
print(f"★ Goldilocks (G_real): ε = {best_r}  ({G_real_arr.max():.4f})")
print(f"★ Goldilocks (G_k):    ε = {best_k}  ({G_k_arr.max():.4f})")

# ============================================================
# 11. REGIME BOUNDARY ESTIMATION
# ============================================================
print("\n" + "=" * 70)
print("REGIME BOUNDARY ANALYSIS")
print("=" * 70)

Zb_arr = np.array([results[e]['Zb_final'] for e in eps_values])
dNb_arr = np.array([results[e]['dNb'] for e in eps_values])

# Bound-sector depletion threshold: ΔN_bound = 0
try:
    f_dNb = interp1d(eps_arr, dNb_arr, kind='linear')
    # Find sign change
    eps_fine = np.linspace(eps_arr.min(), eps_arr.max(), 1000)
    dNb_fine = f_dNb(eps_fine)
    sign_changes = np.where(np.diff(np.sign(dNb_fine)))[0]
    if len(sign_changes) > 0:
        eps_depletion = float(eps_fine[sign_changes[0]])
        print(f"  Bound-sector depletion threshold: ε_dep ≈ {eps_depletion:.3f}")
    else:
        eps_depletion = None
        print("  Bound-sector depletion threshold: not found in scan range")
except Exception as e:
    eps_depletion = None
    print(f"  Depletion threshold estimation failed: {e}")

# Z_Φ peak
idx_Zmax = np.argmax(Zb_arr)
eps_Zpeak = eps_values[idx_Zmax]
print(f"  Phase-ordering peak: ε_coh = {eps_Zpeak} (Z_Φ = {Zb_arr[idx_Zmax]:.4f})")

# G_k peak
idx_Gmax = np.argmax(G_k_arr)
eps_Gpeak = eps_values[idx_Gmax]
print(f"  Structured emission peak: ε_em = {eps_Gpeak} (G_k = {G_k_arr[idx_Gmax]:.4f})")

# Winding breakdown
w_arr = np.array([results[e]['w_after'] for e in eps_values])
broken_mask = np.abs(w_arr - 1.0) > 0.1
if np.any(broken_mask):
    eps_broken = eps_values[np.argmax(broken_mask)]
    print(f"  Topological breakdown: ε ≥ {eps_broken}")
else:
    print("  Topological breakdown: not observed in scan range")

# Four-regime hierarchy summary
print(f"\n  Four-regime hierarchy:")
if eps_depletion is not None:
    print(f"    I.   Weak response:       ε < {eps_depletion:.2f}")
    print(f"    II.  Phase-ordering peak: ε ≈ {eps_Zpeak}")
    print(f"    III. Structured emission: ε ≈ {eps_Gpeak}")
    if np.any(broken_mask):
        print(f"    IV.  Overdriven:          ε ≥ {eps_broken}")
    else:
        print(f"    IV.  Overdriven:          ε > {eps_values[-1]} (outside scan)")

# ============================================================
# 12. FIGURES — ALL WITH VERSION LABEL
# ============================================================

FIGTAG = f"EP8 {VERSION}"

# --- Figure 1: Vortex prep ---
fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle(f'{FIGTAG}: Prepared 2D Vortex', fontsize=12, fontweight='bold')
im0 = ax1[0].pcolormesh(x, y, np.abs(psi_v).T, cmap='viridis', shading='auto')
ax1[0].set_title(f'|Ψ| (wind={w_init:.3f})'); ax1[0].set_aspect('equal')
plt.colorbar(im0, ax=ax1[0])
im1 = ax1[1].pcolormesh(x, y, np.angle(psi_v).T, cmap='hsv', shading='auto')
ax1[1].set_title('arg(Ψ)'); ax1[1].set_aspect('equal')
plt.colorbar(im1, ax=ax1[1])
r1d = np.linspace(0, 20, 200)
ix_r = ((xc+r1d)/dx).astype(int) % Nx
ax1[2].plot(r1d, np.abs(psi_v[ix_r, Ny//2]), 'b-', lw=2)
ax1[2].axhline(A0, color='r', ls='--'); ax1[2].set_xlabel('r')
ax1[2].set_title('Radial'); ax1[2].grid(alpha=0.3)
plt.tight_layout()
fig1.savefig(f'ep8_{VERSION}_vortex.png', dpi=DPI, bbox_inches='tight')
plt.close(fig1)
print(f"\nSaved: ep8_{VERSION}_vortex.png")

# --- Figure 2: 12-panel overview (regime hierarchy) ---
fig2 = plt.figure(figsize=(18, 26))
gs = GridSpec(6, 2, figure=fig2, hspace=0.38, wspace=0.3)
fig2.suptitle(f'{FIGTAG}: Four-Regime Hierarchy Scan\n'
              'ED: (1/c²)Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ\n'
              'No phenomenological ATI coupling, Volkov state, SFA, or transition matrix imposed',
              fontsize=11, fontweight='bold', y=0.995)

# (a) G scores — both G_real and G_k
ax = fig2.add_subplot(gs[0, 0])
ax.plot(eps_arr, G_real_arr, 'ko-', ms=6, lw=2, label='G_real')
ax.plot(eps_arr, G_k_arr, 'rs--', ms=6, lw=2, label='G_k')
ax.axvline(eps_Gpeak, color='gold', ls='--', lw=1, alpha=0.7, label=f'G_k peak (ε={eps_Gpeak})')
ax.set_xlabel('ε'); ax.set_ylabel('Score')
ax.set_title('(a) Goldilocks scores')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# (b) Z_Φ
ax = fig2.add_subplot(gs[0, 1])
ax.plot(eps_arr, Zb_arr, 'ro-', ms=6)
ax.axvline(eps_Zpeak, color='gold', ls='--', lw=1, alpha=0.7, label=f'Z_Φ peak (ε={eps_Zpeak})')
ax.set_xlabel('ε'); ax.set_ylabel('Z_Φ (bound)')
ax.set_title('(b) Phase order — bound region')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# (c) Region-resolved residual
ax = fig2.add_subplot(gs[1, 0])
ax.plot(eps_arr, [results[e]['P_bound'] for e in eps_values], 'ro-', ms=5, label='bound')
ax.plot(eps_arr, [results[e]['P_inter'] for e in eps_values], 'gs-', ms=5, label='inter')
ax.plot(eps_arr, [results[e]['P_out'] for e in eps_values], 'b^-', ms=5, label='outgoing')
ax.set_xlabel('ε'); ax.set_ylabel('Residual power')
ax.set_title('(c) Region-resolved residual (overlap-aligned)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (d) S_norm and Q₂
ax = fig2.add_subplot(gs[1, 1])
Sn_arr = [results[e]['metrics']['S_norm'] for e in eps_values]
Q_arr = [results[e]['metrics']['Q2'] for e in eps_values]
ax.plot(eps_arr, Sn_arr, 'bs-', ms=5, label='S_norm (entropy)')
ax.plot(eps_arr, Q_arr, 'r^-', ms=5, label='Q₂ (quadrupole)')
ax.set_xlabel('ε')
ax.set_title('(d) Angular structure (normalized)')
ax.legend(); ax.grid(alpha=0.3)

# (e) FB ratio
ax = fig2.add_subplot(gs[2, 0])
FB_arr = [results[e]['metrics']['FB'] for e in eps_values]
ax.plot(eps_arr, FB_arr, 'ko-', ms=6)
ax.axhline(1.0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('ε'); ax.set_ylabel('Forward / Backward')
ax.set_title('(e) k-space F/B ratio')
ax.grid(alpha=0.3)

# (f) P(θ_k) polar — best G_k
ax = fig2.add_subplot(gs[2, 1], projection='polar')
rb = results[eps_Gpeak]
Pn = rb['Ptheta'] / (np.max(rb['Ptheta']) + 1e-30)
ax.plot(rb['theta_k'], Pn, 'r-', lw=2)
ax.fill_between(rb['theta_k'], 0, Pn, alpha=0.2, color='red')
ax.set_title(f'(f) P(θ_k) @ ε={eps_Gpeak}\n(structured emission peak)', pad=20, fontsize=9)

# (g) Winding + core shift
ax = fig2.add_subplot(gs[3, 0])
ax.plot(eps_arr, w_arr, 'ko-', ms=6, label='winding')
ax.axhline(1.0, color='r', ls='--', lw=0.5, label='target')
ax.set_xlabel('ε'); ax.set_ylabel('Winding')
ax.set_title('(g) Topological survival')
ax2g = ax.twinx()
cs_arr = [results[e]['core_shift'] for e in eps_values]
ax2g.plot(eps_arr, cs_arr, 'r^--', ms=4, alpha=0.7, label='core shift')
ax2g.set_ylabel('Core shift (dx)', color='r')
ax.legend(loc='upper left', fontsize=7); ax2g.legend(loc='upper right', fontsize=7)
ax.grid(alpha=0.3)

# (h) ΔN_bound (bound-sector depletion)
ax = fig2.add_subplot(gs[3, 1])
ax.plot(eps_arr, dNb_arr, 'ko-', ms=6)
ax.axhline(0, color='gray', ls='--', lw=0.5)
if eps_depletion is not None:
    ax.axvline(eps_depletion, color='orange', ls='--', lw=1.5,
               label=f'depletion threshold ≈ {eps_depletion:.2f}')
    ax.legend(fontsize=7)
ax.set_xlabel('ε'); ax.set_ylabel('ΔN_bound')
ax.set_title('(h) Bound-sector depletion')
ax.grid(alpha=0.3)

# (i) Z_Φ(t)
ax = fig2.add_subplot(gs[4, 0])
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(eps_values)))
for i, eps in enumerate(eps_values):
    ax.plot(results[eps]['full']['t'], results[eps]['full']['Zb'],
            color=colors[i], lw=1, label=f'ε={eps}')
ax.set_xlabel('t'); ax.set_ylabel('Z_Φ(t)')
ax.set_title('(i) Phase order time evolution')
ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)

# (j) N_bound(t)
ax = fig2.add_subplot(gs[4, 1])
for i, eps in enumerate(eps_values):
    ax.plot(results[eps]['full']['t'], results[eps]['full']['Nb'],
            color=colors[i], lw=1, label=f'ε={eps}')
ax.set_xlabel('t'); ax.set_ylabel('N_bound(t)')
ax.set_title('(j) Bound norm evolution')
ax.legend(fontsize=5, ncol=2); ax.grid(alpha=0.3)

# (k) G_k vs Z_Φ overlay (normalized)
ax = fig2.add_subplot(gs[5, 0])
Zb_norm = Zb_arr / (np.max(Zb_arr) + 1e-30)
Gk_norm = G_k_arr / (np.max(G_k_arr) + 1e-30)
ax.plot(eps_arr, Zb_norm, 'ro-', ms=6, label=f'Z_Φ (peak @ ε={eps_Zpeak})')
ax.plot(eps_arr, Gk_norm, 'bs--', ms=6, label=f'G_k (peak @ ε={eps_Gpeak})')
ax.axvline(eps_Zpeak, color='r', ls=':', lw=0.8, alpha=0.5)
ax.axvline(eps_Gpeak, color='b', ls=':', lw=0.8, alpha=0.5)
ax.set_xlabel('ε'); ax.set_ylabel('Normalized')
ax.set_title('(k) Phase-ordering vs Structured emission')
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# (l) Regime map
ax = fig2.add_subplot(gs[5, 1])
ax.set_xlim(eps_arr.min()*0.9, eps_arr.max()*1.1)
ax.set_ylim(-0.5, 3.5)
# Color bands for regimes
regime_colors = ['#e8f5e9', '#fff3e0', '#e3f2fd', '#fce4ec']
regime_names = ['I. Weak\nresponse', 'II. Phase-\nordering', 'III. Structured\nemission', 'IV. Overdriven\ndepletion']
if eps_depletion is not None:
    boundaries = [eps_arr.min()*0.9, eps_depletion, eps_Zpeak, eps_Gpeak, eps_arr.max()*1.1]
else:
    boundaries = [eps_arr.min()*0.9, eps_Zpeak*0.7, eps_Zpeak, eps_Gpeak, eps_arr.max()*1.1]
for i in range(4):
    if i < len(boundaries) - 1:
        ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.4,
                   color=regime_colors[i])
        mid = 0.5*(boundaries[i] + boundaries[i+1])
        ax.text(mid, 1.5, regime_names[i], ha='center', va='center',
                fontsize=8, fontweight='bold')
ax.set_xlabel('ε')
ax.set_title('(l) Four-regime hierarchy map')
ax.set_yticks([])

plt.tight_layout()
fig2.savefig(f'ep8_{VERSION}_scan_overview.png', dpi=DPI, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: ep8_{VERSION}_scan_overview.png")

# --- Figure 3: k-space + residuals at Goldilocks ---
best_eps_fig = eps_Gpeak
rb = results[best_eps_fig]

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
fig3.suptitle(f'{FIGTAG}: k-space Residual @ ε={best_eps_fig} (structured emission peak)\n'
              'Overlap-aligned, outgoing-windowed',
              fontsize=11, fontweight='bold')

kx_arr_f = rb['KX'][:, 0]; ky_arr_f = rb['KY'][0, :]
pk = rb['Pk']; klim = 3*packet_k

im0 = axes3[0].pcolormesh(kx_arr_f, ky_arr_f, np.log10(pk.T+1e-20),
                           cmap='inferno', shading='auto')
axes3[0].set_xlim(-klim, klim); axes3[0].set_ylim(-klim, klim)
axes3[0].set_xlabel('kx'); axes3[0].set_ylabel('ky')
axes3[0].set_title('log₁₀|FFT(Ψ_res)|²'); axes3[0].set_aspect('equal')
axes3[0].axvline(packet_k, color='cyan', ls='--', lw=0.8, alpha=0.5)
plt.colorbar(im0, ax=axes3[0])

im1 = axes3[1].pcolormesh(x, y, np.abs(rb['res_psi']).T**2,
                           cmap='hot', shading='auto')
axes3[1].set_xlabel('x'); axes3[1].set_ylabel('y')
axes3[1].set_title('|Ψ_res|² (real space)'); axes3[1].set_aspect('equal')
c1 = plt.Circle((xc, yc), R_bound, fill=False, color='cyan', ls='--')
axes3[1].add_patch(c1); plt.colorbar(im1, ax=axes3[1])

im2 = axes3[2].pcolormesh(x, y, rb['res_rho'].T, cmap='RdBu_r', shading='auto')
axes3[2].set_xlabel('x'); axes3[2].set_ylabel('y')
axes3[2].set_title('Density residual: Δρ'); axes3[2].set_aspect('equal')
c2 = plt.Circle((xc, yc), R_bound, fill=False, color='k', ls='--')
axes3[2].add_patch(c2); plt.colorbar(im2, ax=axes3[2])

plt.tight_layout()
fig3.savefig(f'ep8_{VERSION}_kspace.png', dpi=DPI, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: ep8_{VERSION}_kspace.png")

# --- Figure 4: Current streamlines ---
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(f'{FIGTAG}: Current Residual @ ε={best_eps_fig}\n'
              'j = Im(Ψ*∇Ψ) — phase flow structure',
              fontsize=11, fontweight='bold')

im0 = axes4[0].pcolormesh(x, y, rb['res_j_mag'].T, cmap='magma', shading='auto')
axes4[0].set_title('|j_res|'); axes4[0].set_aspect('equal')
c0 = plt.Circle((xc, yc), R_bound, fill=False, color='cyan', ls='--')
axes4[0].add_patch(c0); plt.colorbar(im0, ax=axes4[0])

skip = 8
xs, ys = X[::skip, ::skip], Y[::skip, ::skip]
jxs, jys = rb['res_jx'][::skip, ::skip], rb['res_jy'][::skip, ::skip]
j_speed = np.sqrt(jxs**2 + jys**2)
axes4[1].streamplot(xs.T, ys.T, jxs.T, jys.T,
                     color=j_speed.T, cmap='plasma', linewidth=0.8,
                     density=1.5, arrowsize=0.8)
axes4[1].set_title('Current streamlines'); axes4[1].set_aspect('equal')
c1 = plt.Circle((xc, yc), R_bound, fill=False, color='cyan', ls='--', lw=1.5)
axes4[1].add_patch(c1)
axes4[1].set_xlim(0, Lx); axes4[1].set_ylim(0, Ly)

plt.tight_layout()
fig4.savefig(f'ep8_{VERSION}_current.png', dpi=DPI, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: ep8_{VERSION}_current.png")

# --- Figure 5: Angular distributions — select ε ---
# Show 6 representative: weak, near-threshold, ordering, transition, emission, overdriven
# Pick indices spread across the scan
n_show = min(6, len(eps_values))
show_idx = np.round(np.linspace(0, len(eps_values)-1, n_show)).astype(int)
show_eps = [eps_values[i] for i in show_idx]

fig5, axes5 = plt.subplots(2, 3, figsize=(18, 11), subplot_kw={'projection': 'polar'})
fig5.suptitle(f'{FIGTAG}: k-space Angular Distribution P(θ_k)\n'
              'Outgoing residual momentum (0.5k₀ < k < 2k₀)',
              fontsize=11, fontweight='bold', y=1.02)

for i, eps in enumerate(show_eps):
    ax = axes5[i//3, i%3]
    r = results[eps]
    Pn = r['Ptheta'] / (np.max(r['Ptheta']) + 1e-30)
    ax.plot(r['theta_k'], Pn, 'r-', lw=1.5)
    ax.fill_between(r['theta_k'], 0, Pn, alpha=0.2, color='red')
    m = r['metrics']
    ax.set_title(f'ε={eps}\nFB={m["FB"]:.1f} Q₂={m["Q2"]:.3f} '
                 f'lobes={m["n_lobes"]}', fontsize=9, pad=15)

plt.tight_layout()
fig5.savefig(f'ep8_{VERSION}_angular.png', dpi=DPI, bbox_inches='tight')
plt.close(fig5)
print(f"Saved: ep8_{VERSION}_angular.png")

# --- Figure 6: Three residuals — select ε ---
fig6, axes6 = plt.subplots(3, n_show, figsize=(4*n_show, 12))
fig6.suptitle(f'{FIGTAG}: Three Residual Types\n'
              'Top: |Ψ_res|²  Mid: Δρ  Bottom: |j_res|',
              fontsize=11, fontweight='bold')

for col, eps in enumerate(show_eps):
    r = results[eps]
    axes6[0, col].pcolormesh(x, y, np.abs(r['res_psi']).T**2,
                              cmap='hot', shading='auto')
    axes6[0, col].set_title(f'ε={eps}', fontsize=10); axes6[0, col].set_aspect('equal')
    axes6[1, col].pcolormesh(x, y, r['res_rho'].T,
                              cmap='RdBu_r', shading='auto')
    axes6[1, col].set_aspect('equal')
    axes6[2, col].pcolormesh(x, y, r['res_j_mag'].T,
                              cmap='magma', shading='auto')
    axes6[2, col].set_aspect('equal'); axes6[2, col].set_xlabel('x')
    for row in range(3):
        c = plt.Circle((xc, yc), R_bound, fill=False, color='cyan', ls='--', lw=0.5)
        axes6[row, col].add_patch(c)
        if col == 0: axes6[row, col].set_ylabel('y')

plt.tight_layout()
fig6.savefig(f'ep8_{VERSION}_three_residuals.png', dpi=DPI, bbox_inches='tight')
plt.close(fig6)
print(f"Saved: ep8_{VERSION}_three_residuals.png")

# ============================================================
# 17. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print(f"EP8 {VERSION} COMPLETE — ε-refined scan")
print("=" * 70)
print(f"""
Refined scan: {eps_values}

Regime boundaries:
  Bound-sector depletion threshold: ε_dep ≈ {eps_depletion if eps_depletion else 'N/A'}
  Phase-ordering peak (Z_Φ):        ε_coh = {eps_Zpeak} (Z_Φ = {Zb_arr[idx_Zmax]:.4f})
  Structured emission peak (G_k):   ε_em  = {eps_Gpeak} (G_k = {G_k_arr[idx_Gmax]:.4f})

Goldilocks (G_real): ε = {best_r}  ({G_real_arr.max():.4f})
Goldilocks (G_k):    ε = {best_k}  ({G_k_arr.max():.4f})

Leo's key question:
  Z_Φ peak @ ε = {eps_Zpeak}
  G_k peak @ ε = {eps_Gpeak}
  Separation: {'YES — phase-ordering precedes structured emission' if eps_Zpeak != eps_Gpeak else 'NO — same ε'}

Central claim (Leo approved):
  "We do not claim to reproduce ATI quantitatively. We test a prior question:
   whether a localized topological closure and a same-field high-drive packet,
   evolved only by the ED equation, can generate a structured residual
   momentum-angular response."

Files:
  ep8_{VERSION}_summary.json / ep8_{VERSION}_arrays.npz
  ep8_{VERSION}_vortex.png / ep8_{VERSION}_scan_overview.png
  ep8_{VERSION}_kspace.png / ep8_{VERSION}_current.png
  ep8_{VERSION}_angular.png / ep8_{VERSION}_three_residuals.png
""")
print("=" * 70)
