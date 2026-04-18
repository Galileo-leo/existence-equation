"""
3D ED simulation of a hydrogen-like atom from the ED field equation alone.

This script evolves a single complex scalar field Psi on a periodic N^3 grid using the
nonlinear Klein–Gordon-type ED equation (second order in time). The proton is modeled
as a stable vortex cluster (e.g. a tight triplet of like-sign phase singularities); the
electron is identified with an opposite-sign vortex in a bound configuration. After an
injection phase that drives vortex proliferation and an optional auto-quench, the grid
undergoes cosmological-style expansion (scale factor a). The frozen field is then analyzed
for: (A) linear confinement signatures along flux tubes between opposite vortices,
(B) charge via phase winding on circles around the proton (and comparison to the electron),
and (C) energy decomposition in a local ROI—especially phase-gradient (kinetic) energy
that reflects s-orbital–like, centrally peaked bound-state behavior when the envelope
is approximately spherical around the core.

This is the v21-proven parameter set at 256^3 (2× the 128^3 verification resolution);
same physics, higher grid resolution.

Jae-Ahn Shin & Kael · 2026-03-23
"""

import time as timer
import numpy as np
import json

try:
    import cupy as cp
    xp = cp
    USE_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"★ GPU: {props['name'].decode()}")
    print(f"  VRAM: {cp.cuda.Device(0).mem_info[1]/1e9:.1f} GB")
except ImportError:
    xp = np
    USE_GPU = False
    print("CPU mode")


def to_np(arr):
    return cp.asnumpy(arr) if USE_GPU else arr

def laplacian_3d(f, dx):
    # 7-point stencil Laplacian on a uniform periodic grid (spacing dx).
    return (xp.roll(f,1,0) + xp.roll(f,-1,0) +
            xp.roll(f,1,1) + xp.roll(f,-1,1) +
            xp.roll(f,1,2) + xp.roll(f,-1,2) - 6*f) / dx**2

def wrap_ph(dp):
    return (dp + np.pi) % (2*np.pi) - np.pi

def count_vortices_3d_slice(psi_slice):
    phase = xp.angle(psi_slice)
    dp_x = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=1) - phase)))
    dp_y = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=0) - phase)))
    curl = dp_x + xp.roll(dp_y,-1,axis=1) - xp.roll(dp_x,-1,axis=0) - dp_y
    winding = curl / (2*xp.pi)
    np_ = int(xp.sum(winding > 0.5))
    nm_ = int(xp.sum(winding < -0.5))
    return np_, nm_, winding

def count_vortices_3d_full(psi):
    phase = xp.angle(psi)
    total_plus = 0
    total_minus = 0
    dp_x = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=2) - phase)))
    dp_y = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=1) - phase)))
    curl_xy = dp_x + xp.roll(dp_y,-1,axis=2) - xp.roll(dp_x,-1,axis=1) - dp_y
    w_xy = curl_xy / (2*xp.pi)
    total_plus += int(xp.sum(w_xy > 0.5))
    total_minus += int(xp.sum(w_xy < -0.5))
    dp_z = xp.angle(xp.exp(1j * (xp.roll(phase,-1,axis=0) - phase)))
    curl_xz = dp_x + xp.roll(dp_z,-1,axis=2) - xp.roll(dp_x,-1,axis=0) - dp_z
    w_xz = curl_xz / (2*xp.pi)
    total_plus += int(xp.sum(w_xz > 0.5))
    total_minus += int(xp.sum(w_xz < -0.5))
    curl_yz = dp_y + xp.roll(dp_z,-1,axis=1) - xp.roll(dp_y,-1,axis=0) - dp_z
    w_yz = curl_yz / (2*xp.pi)
    total_plus += int(xp.sum(w_yz > 0.5))
    total_minus += int(xp.sum(w_yz < -0.5))
    return total_plus, total_minus, w_xy

def best_vortex_slice(psi, N):
    best_z = N//2
    best_count = 0
    best_w = None
    for z in range(N//8, 7*N//8, max(N//10, 1)):
        np_, nm_, w = count_vortices_3d_slice(psi[z])
        total = np_ + nm_
        if total > best_count:
            best_count = total
            best_z = z
            best_w = w
    if best_w is None:
        _, _, best_w = count_vortices_3d_slice(psi[best_z])
    return best_z, best_count, best_w

def generate_perturbation_3d(N, L, dx, n_blobs=25,
                              sigma_range=(1.0, 3.0), amp_range=(1.5, 3.0)):
    coords = np.arange(N)*dx - L/2
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')
    pert = np.zeros((N,N,N), dtype=complex)
    for _ in range(n_blobs):
        x0 = np.random.uniform(-L/2, L/2)
        y0 = np.random.uniform(-L/2, L/2)
        z0 = np.random.uniform(-L/2, L/2)
        sig = np.random.uniform(*sigma_range)
        amp = np.random.uniform(*amp_range)
        phi = np.random.uniform(0, 2*np.pi)
        pert += amp * np.exp(-((X-x0)**2+(Y-y0)**2+(Z-z0)**2)/(2*sig**2)) * np.exp(1j*phi)
    return xp.asarray(pert) if USE_GPU else pert

def find_best_triplet_3d(w_xy, dx, L):
    from itertools import combinations
    w_np = to_np(w_xy)
    yp, xp_ = np.where(w_np > 0.5)
    yn, xn_ = np.where(w_np < -0.5)
    best = None
    for sign, xs, ys in [('+', xp_, yp), ('-', xn_, yn)]:
        if len(xs) < 3: continue
        pos = np.column_stack([xs*dx - L/2, ys*dx - L/2])
        n_check = min(len(pos), 30)
        for i,j,k in combinations(range(n_check), 3):
            p1,p2,p3 = pos[i], pos[j], pos[k]
            d12 = np.sqrt(np.sum((p1-p2)**2))
            d23 = np.sqrt(np.sum((p2-p3)**2))
            d13 = np.sqrt(np.sum((p1-p3)**2))
            sides = sorted([d12,d23,d13])
            if sides[0]<0.5 or sides[2]>15.0: continue
            if sides[2]/max(sides[0],0.1) > 3.0: continue
            score = sides[0]/max(sides[2],0.1)
            center = (p1+p2+p3)/3
            if best is None or score > best['score']:
                best = {'sign':sign, 'positions':[p1,p2,p3],
                        'sides':sides, 'center':center, 'score':score}
    return best

def measure_winding_circle(phase_slice, cx, cy, radius, n_sample=120):
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


# ══════════════════════════════════════════════════════════════
# v21 PROVEN PARAMETERS — only N changed to 256
# PDE: (1/c^2) Psi_tt = Lap(Psi) + lam*Psi - alpha*|Psi|^2*Psi
# lam > 0: "mass" term; alpha > 0: defocusing nonlinearity; v_vac = sqrt(lam/alpha).
# ══════════════════════════════════════════════════════════════

c     = 1.0
lam   = 1.0
alpha = 1.0
v_vac = np.sqrt(lam / alpha)  # = 1.0

N     = 256
L0    = 20.0
dx0   = L0 / N
dt    = 0.25 * dx0 / (c * np.sqrt(3))

inject_interval    = 10
n_blobs_per_inject = 25
sigma_range        = (1.0, 3.0)
amp_range          = (1.5, 3.0)

H0        = 0.015
max_scale = 2.0

T_max_inject = 80.0
T_expand     = 100.0
T_hydrogen   = 0.0
STOP_AT_A_MAX = True

quench_patience       = 5
min_vortex_for_quench = 200
vortex_check_interval = 8
print_interval        = 12
full_count_interval   = 60

print(f"\n{'='*70}")
print(f"  ED v21-3D v5: 256^3 with v21 proven parameters")
print(f"  (1/c^2)Psi_tt = Lap Psi + lam Psi - alpha |Psi|^2 Psi")
print(f"{'='*70}")
print(f"  N={N}^3, L0={L0}, dx0={dx0:.4f}")
print(f"  lam={lam}, alpha={alpha}, v_vac={v_vac:.4f}")
print(f"  H0={H0}, a_max={max_scale}")
print(f"  amp={amp_range}")
print(f"  GPU: {'YES' if USE_GPU else 'NO'}")


# ══════════════════════════════════════════════════════════════
# Phase 1: Energy Injection
# Random Gaussian blobs add energy; nonlinear dynamics generate a dense vortex soup.
# When the 3D vortex count falls after a peak (patience), we auto-quench to freeze structure.
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  Phase 1: Energy Injection")
print(f"{'='*70}\n")

np.random.seed(42)
psi = v_vac * xp.ones((N,N,N), dtype=complex)
psi += 0.01 * v_vac * (xp.asarray(np.random.randn(N,N,N).astype(np.float32)) +
                         1j * xp.asarray(np.random.randn(N,N,N).astype(np.float32)))
psi_dot = xp.zeros_like(psi)

a_current = 1.0
dx_current = dx0
L_current = L0

peak_vortex_count_3d = 0
decline_counter = 0
prev_vortex_count_3d = 0
quench_triggered = False
t_quench = None

wall_start = timer.time()
check_count = 0
max_inject_steps = int(T_max_inject / dt)

for step in range(max_inject_steps):
    t = step * dt
    if step > 0 and step % inject_interval == 0:
        psi += generate_perturbation_3d(N, L0, dx0,
                n_blobs=n_blobs_per_inject,
                sigma_range=sigma_range, amp_range=amp_range)

    if step % vortex_check_interval == 0:
        check_count += 1
        best_z, best_count, w_best = best_vortex_slice(psi, N)

        do_full = (check_count % full_count_interval == 0) or (check_count == 1)
        if do_full:
            np_3d, nm_3d, _ = count_vortices_3d_full(psi)
            n_3d_total = np_3d + nm_3d
        else:
            n_3d_total = -1

        if check_count % (print_interval * 3) == 0 or check_count == 1:
            elapsed = timer.time() - wall_start
            n3d_str = f"3D={n_3d_total}" if n_3d_total >= 0 else "3D=?"
            print(f"  t={t:6.1f} | best_z[{best_z}]={best_count:3d} | {n3d_str} | {elapsed:.0f}s")

        if n_3d_total >= 0:
            if n_3d_total > peak_vortex_count_3d:
                peak_vortex_count_3d = n_3d_total
                decline_counter = 0
            elif n_3d_total < prev_vortex_count_3d:
                decline_counter += 1
            else:
                decline_counter = 0
            prev_vortex_count_3d = n_3d_total

            if decline_counter >= quench_patience and n_3d_total >= min_vortex_for_quench:
                t_quench = t
                quench_triggered = True
                print(f"\n  Auto quench! t={t:.1f}, peak={peak_vortex_count_3d}")
                break

    # Symplectic-style leapfrog: psi_dot <- psi_dot + accel*dt, psi <- psi + psi_dot*dt.
    lap = laplacian_3d(psi, dx0)
    accel = c**2 * (lap + lam*psi - alpha*xp.abs(psi)**2*psi)
    psi_dot += accel * dt
    psi += psi_dot * dt

if not quench_triggered:
    t_quench = T_max_inject
    np_3d, nm_3d, _ = count_vortices_3d_full(psi)
    n_3d_total = np_3d + nm_3d
    print(f"\n  Forced quench! t={t_quench:.1f}, 3D vortices={n_3d_total}")


# ══════════════════════════════════════════════════════════════
# Phase 2: Expansion (a -> 2.0)
# Comoving box grows: dx and L scale with a(t); timestep capped by CFL on current dx.
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  Phase 2: Expansion (a -> {max_scale})")
print(f"{'='*70}\n")

n_expand_steps = int(T_expand / dt)
check_count = 0

for dstep in range(n_expand_steps):
    t = t_quench + (dstep+1) * dt
    dt_since = t - t_quench

    a_current = min(1.0 + H0 * dt_since, max_scale)
    dx_current = dx0 * a_current
    L_current = L0 * a_current

    dt_eff = 0.25 * dx_current / (c * np.sqrt(3))
    dt_use = min(dt, dt_eff)

    # Same explicit time integration as Phase 1; dt_use shrinks when dx_current grows (CFL).
    lap = laplacian_3d(psi, dx_current)
    accel = c**2 * (lap + lam*psi - alpha*xp.abs(psi)**2*psi)
    psi_dot += accel * dt_use
    psi += psi_dot * dt_use
    if STOP_AT_A_MAX and a_current >= max_scale - 1e-12:
        break

    if dstep % vortex_check_interval == 0:
        check_count += 1
        if check_count % (print_interval * 3) == 0:
            best_z, best_count, _ = best_vortex_slice(psi, N)
            elapsed = timer.time() - wall_start
            print(f"  t={t:6.1f} | a={a_current:.3f} | best_z[{best_z}]={best_count:3d} | {elapsed:.0f}s")

t_expand_end = t
print(f"\n  Expansion done: a={a_current:.3f}")

best_z_final, best_count_final, w_best_final = best_vortex_slice(psi, N)
np_3d, nm_3d, _ = count_vortices_3d_full(psi)
print(f"  3D total: +{np_3d} / -{nm_3d} = {np_3d+nm_3d}")
print(f"  Best slice z={best_z_final}: {best_count_final}")


# ══════════════════════════════════════════════════════════════
# Freeze state
# ══════════════════════════════════════════════════════════════

print(f"\n  Saving frozen state (psi + psi_dot)...")
psi_np = to_np(psi)
psi_dot_np = to_np(psi_dot)
np.savez_compressed('v5_frozen_state.npz',
    psi=psi_np, psi_dot=psi_dot_np, dx=dx_current, L=L_current,
    a=a_current, best_z=best_z_final, N=N)
print(f"  Saved: v5_frozen_state.npz (incl. psi_dot)")


# ══════════════════════════════════════════════════════════════
# Phase 2.5: Measurements
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  Phase 2.5: Confinement / Charge / Energy")
print(f"{'='*70}")

phase_np = np.angle(psi_np)
amp_np = np.abs(psi_np)

# ── (A) Linear Confinement ──
# Along lines joining + and - vortex cores in the best z-slice: integrate 2D |grad Psi|^2
# (flux-tube energy proxy); fit E_tube vs separation to test linear confinement (sigma_eff).
print(f"\n  -- (A) Linear Confinement --")

psi_slice_A = psi_np[best_z_final]
grad_x = (np.roll(psi_slice_A,-1,axis=1) - np.roll(psi_slice_A,1,axis=1)) / (2*dx_current)
grad_y = (np.roll(psi_slice_A,-1,axis=0) - np.roll(psi_slice_A,1,axis=0)) / (2*dx_current)
E_grad_2d = 0.5 * (np.abs(grad_x)**2 + np.abs(grad_y)**2)
bg_E = float(np.median(E_grad_2d))

w_np = to_np(w_best_final)
yp_A, xp_A = np.where(w_np > 0.5)
yn_A, xn_A = np.where(w_np < -0.5)
print(f"    z={best_z_final}: +{len(xp_A)} / -{len(xn_A)} = {len(xp_A)+len(xn_A)}")

confinement_data = []
if len(xp_A) > 0 and len(xn_A) > 0:
    pp = np.column_stack([xp_A.astype(float), yp_A.astype(float)])
    pm = np.column_stack([xn_A.astype(float), yn_A.astype(float)])
    all_pairs = []
    for i in range(len(pp)):
        for j in range(len(pm)):
            diff = pp[i] - pm[j]
            diff = diff - N * np.round(diff / N)
            d = np.sqrt(np.sum(diff**2))
            if 5.0 < d < 40.0:
                all_pairs.append((i, j, d))
    all_pairs.sort(key=lambda x: x[2])
    n_cand = len(all_pairs)
    print(f"    Pair candidates: {n_cand}")

    if n_cand > 0:
        idx_pick = np.unique(np.round(np.linspace(0, n_cand-1, min(40, n_cand))).astype(int))
        for idx in idx_pick:
            i, j, dist = all_pairs[idx]
            p1, p2 = pp[i], pm[j]
            diff = p2 - p1
            diff = diff - N * np.round(diff / N)
            n_sample = max(int(dist * 3), 20)
            profile_E = []
            for frac in np.linspace(0, 1, n_sample):
                px = p1[0] + frac * diff[0]
                py = p1[1] + frac * diff[1]
                profile_E.append(E_grad_2d[int(py) % N, int(px) % N])
            mid_E = float(np.mean(profile_E[len(profile_E)//4:3*len(profile_E)//4]))
            tube_E = mid_E * dist * dx_current
            FT_ratio = mid_E / max(bg_E, 1e-10)
            confinement_data.append({
                'dist': float(dist * dx_current),
                'tube_E': float(tube_E),
                'FT_ratio': float(FT_ratio),
            })

sigma_eff = 0.0
r2_conf = 0.0
r2_ft = 0.0
if confinement_data:
    dists = np.array([c['dist'] for c in confinement_data])
    Es = np.array([c['tube_E'] for c in confinement_data])
    if len(dists) >= 3 and np.std(dists) > 0.01:
        coeffs = np.polyfit(dists, Es, 1)
        sigma_eff = float(coeffs[0])
        ss_res = np.sum((Es - np.polyval(coeffs, dists))**2)
        ss_tot = np.sum((Es - np.mean(Es))**2)
        r2_conf = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    ft_sub = [(c['dist'], c['tube_E']) for c in confinement_data if c['FT_ratio'] > 1.5]
    if len(ft_sub) >= 4:
        ds = np.array([a for a, _ in ft_sub])
        es = np.array([b for _, b in ft_sub])
        if np.std(ds) > 0.01:
            c2 = np.polyfit(ds, es, 1)
            ss2 = np.sum((es - np.polyval(c2, ds))**2)
            st2 = np.sum((es - np.mean(es))**2)
            r2_ft = float(1 - ss2 / st2) if st2 > 0 else 0.0
            print(f"    FT>1.5: sigma={c2[0]:.4f}, R^2={r2_ft:.4f}, n={len(ft_sub)}")

    n_ft = sum(1 for c in confinement_data if c['FT_ratio'] > 1.5)
    ft_max = max(c['FT_ratio'] for c in confinement_data)
    print(f"    All pairs: sigma={sigma_eff:.4f}, R^2={r2_conf:.4f}")
    print(f"    FT>1.5: {n_ft}/{len(confinement_data)}, max FT={ft_max:.2f}")


# ── (B) Charge Quantization ──
# Phase winding number on circles around the proton center approximates topological charge.
print(f"\n  -- (B) Charge Quantization: winding --")

proton = find_best_triplet_3d(w_best_final, dx_current, L_current)
if proton is not None:
    pc_xy = proton['center']
    pc_ix = int((pc_xy[0] + L_current/2) / dx_current) % N
    pc_iy = int((pc_xy[1] + L_current/2) / dx_current) % N
    pc_iz = best_z_final
    print(f"    Proton: ({pc_xy[0]:.1f},{pc_xy[1]:.1f}), z={pc_iz}")
    print(f"    Sign: {proton['sign']}, Score: {proton['score']:.3f}")
    print(f"    Sides: [{proton['sides'][0]:.1f}, {proton['sides'][1]:.1f}, {proton['sides'][2]:.1f}]")
else:
    pc_ix, pc_iy, pc_iz = N//2, N//2, best_z_final
    print(f"    No triplet -> center")

phase_slice = phase_np[pc_iz]
print(f"\n    {'R':>4s}  {'R_phys':>7s}  {'winding':>10s}")
print(f"    {'-'*30}")

charge_vs_r = []
R_min = int(round(0.47 / dx_current))
R_max = min(N//2 - 2, int(round(7.8 / dx_current)))
R_outer_cut = int(round(2.3 / dx_current))
amp_slice = amp_np[pc_iz]

for R in np.arange(max(R_min, 3), R_max + 1, 2):
    n_samp = max(int(8*R), 60)
    angles = np.linspace(0, 2*np.pi, n_samp, endpoint=False)
    amps_on_circle = []
    for ang in angles:
        ix = int(round(pc_ix + R * np.cos(ang))) % N
        iy = int(round(pc_iy + R * np.sin(ang))) % N
        amps_on_circle.append(amp_slice[iy, ix])
    amp_min_circle = float(np.min(amps_on_circle))
    if amp_min_circle < 0.3 * v_vac:
        continue

    w = measure_winding_circle(phase_slice, pc_ix, pc_iy, R, n_sample=n_samp)
    charge_vs_r.append({'R': int(R), 'R_phys': float(R*dx_current), 'winding': float(w)})
    m = " *" if abs(w - round(w)) < 0.15 and abs(round(w)) >= 1 else ""
    print(f"    {R:>4d}  {R*dx_current:>7.2f}  {w:>+10.4f}  amp_min={amp_min_circle:.3f}{m}")

outer = [c['winding'] for c in charge_vs_r if c['R'] >= R_outer_cut]
mean_charge = float(np.mean(outer)) if outer else 0.0
std_charge = float(np.std(outer)) if outer else 0.0
print(f"\n    Outer(R>=15) mean: {mean_charge:+.4f} +/- {std_charge:.4f}")


# ── P/E ratio ──
# Compare mean proton-side winding (large R) to windings around the electron vortex site.
print(f"\n  -- P/E Ratio --")

w_np_b = to_np(w_best_final)
if proton is not None and proton['sign'] == '-':
    yp_e, xp_e = np.where(w_np_b > 0.5)
    e_sign = "+"
else:
    yp_e, xp_e = np.where(w_np_b < -0.5)
    e_sign = "-"

if len(xp_e) > 0:
    e_pos = np.column_stack([xp_e, yp_e])
    p_pos = np.array([pc_ix, pc_iy])
    e_dists = np.sqrt(np.sum((e_pos - p_pos)**2, axis=1))
    e_target_lattice = int(round(20 / dx_current * (dx0 / (L0/128))))
    e_target_lattice = max(e_target_lattice, 20)
    best_idx = np.argmin(np.abs(e_dists - e_target_lattice))
    e_ix = int(e_pos[best_idx, 0])
    e_iy = int(e_pos[best_idx, 1])
    print(f"    Electron [{e_sign}]: ({e_ix},{e_iy}), dist={e_dists[best_idx]:.1f}")

    e_R_phys = [1.5, 1.9, 2.3, 2.8, 3.1]
    e_windings = []
    for rp in e_R_phys:
        R_e = int(round(rp / dx_current))
        if R_e < N//2 - 2:
            ew = measure_winding_circle(phase_slice, e_ix, e_iy, R_e,
                                         n_sample=max(int(8*R_e), 60))
            e_windings.append(ew)
            print(f"    Electron R={R_e} (phys={rp:.1f}): winding={ew:+.4f}")

    if e_windings and outer:
        e_mean = np.mean(e_windings)
        p_mean = mean_charge
        if abs(e_mean) > 0.1:
            pe_ratio = abs(p_mean / e_mean)
            print(f"\n    P/E ratio = {pe_ratio:.4f}")
            if abs(pe_ratio - 1.0) < 0.3:
                print(f"    Proton charge ~ Electron charge!")
else:
    print(f"    No opposite-sign vortex found.")


# ── (C) Energy Decomposition ──
# Local ROI around the proton: split |Psi| gradient, phase-gradient (superfluid velocity),
# quadratic "mass" and quartic interaction terms (vacuum subtracted)—relevant for s-like
# states where phase gradients carry much of the kinetic structure.
print(f"\n  -- (C) Energy Decomposition (ROI box=25) --")

ROI_BOX = 25
cx_roi, cy_roi, cz_roi = pc_iz, pc_iy, pc_ix
mid = N // 2
shifted = np.roll(psi_np, (mid-cx_roi, mid-cy_roi, mid-cz_roi), axis=(0,1,2))
roi = shifted[mid-ROI_BOX:mid+ROI_BOX, mid-ROI_BOX:mid+ROI_BOX, mid-ROI_BOX:mid+ROI_BOX]
roi_amp = np.abs(roi)
roi_phase = np.angle(roi)
dV = dx_current**3
n_cells = roi.size

E_grad_A_roi = 0.0
E_phase_roi = 0.0
for ax in range(3):
    dA = (np.roll(roi_amp,-1,axis=ax) - np.roll(roi_amp,1,axis=ax)) / (2*dx_current)
    E_grad_A_roi += float(np.sum(0.5 * dA**2)) * dV
    dphi = wrap_ph(np.roll(roi_phase,-1,axis=ax) - roi_phase) / dx_current
    E_phase_roi += float(np.sum(0.5 * roi_amp**2 * dphi**2)) * dV

E_rest = float(np.sum(0.5 * lam * roi_amp**2)) * dV
E_cond = float(np.sum(0.25 * alpha * roi_amp**4)) * dV
E_rest -= (0.5 * lam * v_vac**2) * n_cells * dV
E_cond -= (0.25 * alpha * v_vac**4) * n_cells * dV
E_total = E_grad_A_roi + E_phase_roi + E_rest + E_cond

print(f"    ROI: {2*ROI_BOX}^3 = {(2*ROI_BOX)**3:,} cells")
print(f"    <|Psi|> = {np.mean(roi_amp):.4f}  (v_vac={v_vac:.1f})")
print(f"    E_grad_A = {E_grad_A_roi:>12.2f}  ({E_grad_A_roi/max(abs(E_total),1e-10)*100:>5.1f}%)")
print(f"    E_phase  = {E_phase_roi:>12.2f}  ({E_phase_roi/max(abs(E_total),1e-10)*100:>5.1f}%)")
print(f"    E_rest   = {E_rest:>12.2f}  ({E_rest/max(abs(E_total),1e-10)*100:>5.1f}%)")
print(f"    E_cond   = {E_cond:>12.2f}  ({E_cond/max(abs(E_total),1e-10)*100:>5.1f}%)")
print(f"    E_total  = {E_total:>12.2f}")
print(f"    phase/total = {E_phase_roi/max(abs(E_total),1e-10)*100:.1f}%")


# ── Summary ──
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"    N={N}^3, lam={lam}, H={H0}, a={a_current:.3f}")
print(f"    Slice vortices: +{len(xp_A)} / -{len(xn_A)}")
print(f"    R^2(FT>1.5) = {r2_ft:.4f}")
if proton:
    print(f"    Triplet score = {proton['score']:.3f}")
else:
    print(f"    No triplet")
print(f"    Charge(outer) = {mean_charge:+.4f} +/- {std_charge:.4f}")
print(f"    Phase/total = {E_phase_roi/max(abs(E_total),1e-10)*100:.1f}%")

summary = {
    'params': {'N': N, 'lam': lam, 'H0': H0, 'a_max': max_scale,
               'a_final': float(a_current), 'amp_range': list(amp_range)},
    'confinement': confinement_data,
    'sigma_eff': sigma_eff, 'r2_all': r2_conf, 'r2_ft': r2_ft,
    'charge_vs_r': charge_vs_r, 'mean_charge': mean_charge, 'std_charge': std_charge,
    'triplet_score': proton['score'] if proton else 0.0,
    'energy_roi': {'E_phase': E_phase_roi, 'E_total': E_total,
                   'phase_frac': E_phase_roi/max(abs(E_total),1e-10)*100},
}
with open('v5_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"    Saved: v5_analysis.json")

elapsed = timer.time() - wall_start
print(f"\n    Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"{'='*70}")
