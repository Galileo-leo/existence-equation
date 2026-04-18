"""
analyze_v5_confinement_3d.py   (v2 — adaptive + core filtering)
================================================================
3D Flux Tube Confinement Analysis

Strategy:
  0. Compute |grad Psi|^2 first, find energy "core" center
  1. Detect vortices on ALL z-slices, keep only those near the core
  2. Trace vortex lines through z with relaxed matching distance
  3. Adaptive MIN_LINE_LENGTH based on actual distribution
  4. 3D cylindrical flux tube energy between +/- line pairs
  5. Linear fit  E vs d  →  sigma_eff

Input:  v5_frozen_state.npz
Output: v5_confinement_3d.json, ed_strong_confinement_3d.png

Jae-Ahn Shin · 2026-03
"""

import numpy as np
import json
import time as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_VERSION = "v6-honest-2026-03-20"
print("=" * 65)
print(f"  3D Flux Tube Confinement Analysis")
print(f"  VERSION: {_VERSION}")
print("=" * 65)

t0 = timer.time()

print("\n  Loading frozen state...")
data = np.load('v5_frozen_state.npz')
psi_np = data['psi']
dx = float(data['dx'])
L = float(data['L'])
N = int(data['N'])
best_z = int(data['best_z'])
print(f"  N={N}, dx={dx:.4f}, L={L:.2f}, best_z={best_z}")

# ══════════════════════════════════════════════════════════════
# STEP 0: 3D gradient energy + find energy core center
# ══════════════════════════════════════════════════════════════
print("\n  Step 0: Computing 3D |grad Psi|^2 and locating core...")

E_grad_3d = np.zeros((N, N, N), dtype=np.float32)
for ax in range(3):
    dpsi = (np.roll(psi_np, -1, axis=ax) - np.roll(psi_np, 1, axis=ax)) / (2 * dx)
    E_grad_3d += (0.5 * np.abs(dpsi)**2).astype(np.float32)
    del dpsi

bg_E_3d = float(np.median(E_grad_3d))
print(f"  Background E_grad median: {bg_E_3d:.2f}")

E_proj_xy = np.mean(E_grad_3d, axis=0)
threshold = np.percentile(E_proj_xy, 95)
high_mask = E_proj_xy > threshold
if np.sum(high_mask) > 0:
    ys, xs = np.where(high_mask)
    core_x = float(np.mean(xs))
    core_y = float(np.mean(ys))
else:
    core_x, core_y = N / 2.0, N / 2.0

core_x_phys = core_x * dx - L / 2
core_y_phys = core_y * dx - L / 2
print(f"  Energy core center: grid=({core_x:.1f}, {core_y:.1f}), "
      f"phys=({core_x_phys:.1f}, {core_y_phys:.1f})")

CORE_RADIUS_GRID = N * 0.15
print(f"  Core filter radius: {CORE_RADIUS_GRID:.0f} grid ({CORE_RADIUS_GRID*dx:.1f} phys)")

# ══════════════════════════════════════════════════════════════
# STEP 1: Detect vortices on EVERY z-slice, filter to core
# ══════════════════════════════════════════════════════════════
print("\n  Step 1: Detecting vortices (core-filtered)...")

def detect_vortices_2d(psi_slice):
    phase = np.angle(psi_slice)
    dp_x = np.angle(np.exp(1j * (np.roll(phase, -1, axis=1) - phase)))
    dp_y = np.angle(np.exp(1j * (np.roll(phase, -1, axis=0) - phase)))
    curl = dp_x + np.roll(dp_y, -1, axis=1) - np.roll(dp_x, -1, axis=0) - dp_y
    w = curl / (2 * np.pi)
    yp, xp = np.where(w > 0.5)
    yn, xn = np.where(w < -0.5)
    return xp, yp, xn, yn

def dist_to_core(x, y):
    ddx = x - core_x
    ddy = y - core_y
    ddx -= N * round(ddx / N)
    ddy -= N * round(ddy / N)
    return np.sqrt(ddx**2 + ddy**2)

vortices_by_z = {}
n_raw = 0
n_kept = 0
for z in range(N):
    xp, yp, xn, yn = detect_vortices_2d(psi_np[z])
    n_raw += len(xp) + len(xn)

    plus = [(int(x), int(y)) for x, y in zip(xp, yp)
            if dist_to_core(x, y) < CORE_RADIUS_GRID]
    minus = [(int(x), int(y)) for x, y in zip(xn, yn)
             if dist_to_core(x, y) < CORE_RADIUS_GRID]
    n_kept += len(plus) + len(minus)
    vortices_by_z[z] = {'plus': plus, 'minus': minus}

avg_kept = n_kept / N / 2
print(f"  Raw vortices: {n_raw}, kept (core): {n_kept}")
print(f"  Avg per slice after filter: +{avg_kept:.1f} / -{avg_kept:.1f}")

# ══════════════════════════════════════════════════════════════
# STEP 2: Trace vortex lines through z
# ══════════════════════════════════════════════════════════════
print("\n  Step 2: Tracing vortex lines through z...")

MAX_MATCH_DIST = 5.0

def trace_lines(vortices_by_z, sign_key, N):
    active_lines = []
    finished_lines = []

    initial = vortices_by_z[0][sign_key]
    for (x, y) in initial:
        active_lines.append({'points': [(x, y, 0)], 'last_xy': (x, y)})

    for z in range(1, N):
        current = vortices_by_z[z][sign_key]
        if not current:
            finished_lines.extend(active_lines)
            active_lines = []
            continue

        current_arr = np.array(current, dtype=float)
        matched_current = set()
        new_active = []

        for line in active_lines:
            lx, ly = line['last_xy']
            diffs = current_arr - np.array([lx, ly])
            diffs = diffs - N * np.round(diffs / N)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            best_idx = int(np.argmin(dists))

            if dists[best_idx] < MAX_MATCH_DIST and best_idx not in matched_current:
                cx, cy = current[best_idx]
                line['points'].append((cx, cy, z))
                line['last_xy'] = (cx, cy)
                matched_current.add(best_idx)
                new_active.append(line)
            else:
                finished_lines.append(line)

        for idx, (x, y) in enumerate(current):
            if idx not in matched_current:
                new_active.append({'points': [(x, y, z)], 'last_xy': (x, y)})

        active_lines = new_active

    finished_lines.extend(active_lines)
    return finished_lines

plus_lines = trace_lines(vortices_by_z, 'plus', N)
minus_lines = trace_lines(vortices_by_z, 'minus', N)

all_lengths_p = sorted([len(l['points']) for l in plus_lines], reverse=True)
all_lengths_m = sorted([len(l['points']) for l in minus_lines], reverse=True)

print(f"  + lines: {len(plus_lines)} total")
print(f"    top-10 lengths: {all_lengths_p[:10]}")
print(f"  - lines: {len(minus_lines)} total")
print(f"    top-10 lengths: {all_lengths_m[:10]}")

all_lengths = sorted(all_lengths_p + all_lengths_m, reverse=True)
if len(all_lengths) > 6:
    top_lines = all_lengths[:20]
    if top_lines[0] > 20:
        adaptive_min = max(top_lines[0] // 4, 8)
    else:
        adaptive_min = max(top_lines[0] // 2, 3)
else:
    adaptive_min = 3

MIN_LINE_LENGTH = adaptive_min
print(f"  Adaptive MIN_LINE_LENGTH = {MIN_LINE_LENGTH}")

long_plus = sorted([l for l in plus_lines if len(l['points']) >= MIN_LINE_LENGTH],
                   key=lambda l: len(l['points']), reverse=True)
long_minus = sorted([l for l in minus_lines if len(l['points']) >= MIN_LINE_LENGTH],
                    key=lambda l: len(l['points']), reverse=True)

MAX_LINES = 20
long_plus = long_plus[:MAX_LINES]
long_minus = long_minus[:MAX_LINES]

print(f"  Selected: +{len(long_plus)} / -{len(long_minus)} lines (max {MAX_LINES} each)")

def line_mean_xy(line):
    pts = np.array(line['points'])
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

for i, l in enumerate(long_plus):
    mx, my = line_mean_xy(l)
    mx_phys = mx * dx - L/2
    my_phys = my * dx - L/2
    print(f"    +line {i}: n={len(l['points'])}, mean=({mx_phys:.1f}, {my_phys:.1f})")

for i, l in enumerate(long_minus):
    mx, my = line_mean_xy(l)
    mx_phys = mx * dx - L/2
    my_phys = my * dx - L/2
    print(f"    -line {i}: n={len(l['points'])}, mean=({mx_phys:.1f}, {my_phys:.1f})")

# ══════════════════════════════════════════════════════════════
# Measurement function (reusable for sensitivity scan)
# ══════════════════════════════════════════════════════════════

y_indices, x_indices = np.mgrid[0:N, 0:N]

def measure_pairs(p_lines, m_lines, strip_hw, core_excl, min_ovlp, verbose=True):
    """Path-integral flux tube energy for all +/- line pairs."""
    results = []
    for ip, pline in enumerate(p_lines):
        for im, mline in enumerate(m_lines):
            p_pts = {p[2]: (p[0], p[1]) for p in pline['points']}
            m_pts = {p[2]: (p[0], p[1]) for p in mline['points']}
            common_z = sorted(set(p_pts.keys()) & set(m_pts.keys()))
            if len(common_z) < min_ovlp:
                continue
            seps, t_energies, ft_rs = [], [], []
            for z in common_z:
                px, py = p_pts[z]; mx2, my2 = m_pts[z]
                dfx = px - mx2; dfy = py - my2
                dfx -= N * round(dfx / N); dfy -= N * round(dfy / N)
                sep = np.sqrt(dfx**2 + dfy**2)
                if sep < 3.0:
                    seps.append(sep)
                    cx, cy = (px+mx2)/2.0, (py+my2)/2.0
                    ddx2 = x_indices.astype(float) - cx; ddy2 = y_indices.astype(float) - cy
                    ddx2 = ddx2 - N*np.round(ddx2/N); ddy2 = ddy2 - N*np.round(ddy2/N)
                    mask = np.sqrt(ddx2**2 + ddy2**2) < 2.0
                    if np.sum(mask) > 0:
                        t_energies.append(float(np.sum(E_grad_3d[z][mask]))*dx*dx)
                        ft_rs.append(float(np.mean(E_grad_3d[z][mask]))/max(bg_E_3d,1e-10))
                    continue
                seps.append(sep)
                ux, uy = dfx/sep, dfy/sep
                ddx2 = x_indices.astype(float) - px; ddy2 = y_indices.astype(float) - py
                ddx2 = ddx2 - N*np.round(ddx2/N); ddy2 = ddy2 - N*np.round(ddy2/N)
                pa = ddx2*ux + ddy2*uy
                pp = ddx2*(-uy) + ddy2*ux
                sm = (pa > core_excl) & (pa < sep-core_excl) & (np.abs(pp) < strip_hw)
                ns = int(np.sum(sm))
                if ns == 0:
                    t_energies.append(0.0); ft_rs.append(0.0); continue
                t_energies.append(float(np.sum(E_grad_3d[z][sm]))*dx*dx)
                ft_rs.append(float(np.mean(E_grad_3d[z][sm]))/max(bg_E_3d,1e-10))
            if len(t_energies) < min_ovlp:
                continue
            ms = float(np.mean(seps))*dx
            results.append({
                'plus_line': ip, 'minus_line': im,
                'dist_3d': round(ms, 4),
                'dist_std': round(float(np.std(seps))*dx, 4),
                'tube_E_path': round(float(np.mean(t_energies)), 2),
                'FT_ratio': round(float(np.mean(ft_rs)), 3),
                'n_overlap_z': len(common_z),
            })
            if verbose:
                print(f"    +{ip}--{im}: d={ms:.2f}, "
                      f"E_path={results[-1]['tube_E_path']:.1f}, "
                      f"FT={results[-1]['FT_ratio']:.2f}, z={len(common_z)}")
    return results

def fit_r2(x, y):
    if len(x) < 3:
        return np.array([0.0, 0.0]), 0.0
    c = np.polyfit(x, y, 1)
    ss_res = np.sum((y - np.polyval(c, x))**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return c, (1 - ss_res/ss_tot) if ss_tot > 0 else 0.0

# ══════════════════════════════════════════════════════════════
# STEP 3: Default measurement
# ══════════════════════════════════════════════════════════════
print("\n  Step 3: Flux tube energy (path integral, default params)...")

STRIP_HALF_WIDTH = 1.5
CORE_EXCLUDE = 1.5
MIN_OVERLAP_Z = max(MIN_LINE_LENGTH // 2, 3)

confinement_3d = measure_pairs(long_plus, long_minus, STRIP_HALF_WIDTH, CORE_EXCLUDE, MIN_OVERLAP_Z)
print(f"\n  Total pairs: {len(confinement_3d)}")

# ══════════════════════════════════════════════════════════════
# STEP 4: Honest confinement analysis
#
#   Problem: E_path = sum(energy over strip of length ~ d)
#            → strip area ∝ d → E_path ∝ d even for uniform bg
#   
#   Solution: use σ_eff = E_path / d (energy per unit length)
#   
#   If σ_eff = constant independent of d AND > background
#   → genuine string tension (linear confinement)
#   
#   Background σ_bg estimated from non-FT pairs or from median
# ══════════════════════════════════════════════════════════════
print("\n  Step 4: Honest confinement analysis (E/d metric)...")

if len(confinement_3d) >= 3:
    dists   = np.array([c['dist_3d'] for c in confinement_3d])
    e_path  = np.array([c['tube_E_path'] for c in confinement_3d])
    ft_arr  = np.array([c['FT_ratio'] for c in confinement_3d])
    ft_mask = ft_arr > 1.5
    n_ft    = int(np.sum(ft_mask))

    sigma_per_d = e_path / np.maximum(dists, 0.01)

    sigma_bg = float(np.median(sigma_per_d[~ft_mask])) if np.sum(~ft_mask) >= 3 else float(np.median(sigma_per_d))
    excess = sigma_per_d - sigma_bg

    print(f"\n  ── String tension density σ = E/d ──")
    print(f"  Background σ_bg (median of FT<1.5): {sigma_bg:.0f}")
    print(f"  {'':3s} {'d':>5s}  {'E_path':>8s}  {'σ=E/d':>7s}  {'excess':>7s}  {'FT':>5s}")
    for i, c in enumerate(confinement_3d):
        tag = "***" if c['FT_ratio'] > 1.5 else "   "
        sd = sigma_per_d[i]
        ex = excess[i]
        print(f"    {tag} {c['dist_3d']:5.2f}  {c['tube_E_path']:8.1f}  {sd:7.0f}  "
              f"{ex:+7.0f}  {c['FT_ratio']:.2f}")

    print(f"\n  ── Statistics ──")
    if n_ft > 0:
        ft_sigma = sigma_per_d[ft_mask]
        nft_sigma = sigma_per_d[~ft_mask] if np.sum(~ft_mask) > 0 else sigma_per_d
        print(f"  FT>1.5  mean σ = {np.mean(ft_sigma):.0f} ± {np.std(ft_sigma):.0f} (n={n_ft})")
        print(f"  FT<1.5  mean σ = {np.mean(nft_sigma):.0f} ± {np.std(nft_sigma):.0f} (n={len(nft_sigma)})")
        ratio = np.mean(ft_sigma) / sigma_bg if sigma_bg > 0 else 0
        print(f"  Ratio FT>1.5 / bg(median) = {ratio:.2f}")

    c_sigma, r2_sigma = fit_r2(dists, sigma_per_d)
    print(f"\n  σ=E/d vs d (ALL):    slope={c_sigma[0]:.1f}, R²={r2_sigma:.4f}")
    print(f"    → {'FLAT (good: constant string tension)' if r2_sigma < 0.3 else 'NOT flat (σ varies with d)'}")

    if n_ft >= 3:
        c_sigma_ft, r2_sigma_ft = fit_r2(dists[ft_mask], sigma_per_d[ft_mask])
        print(f"  σ=E/d vs d (FT>1.5): slope={c_sigma_ft[0]:.1f}, R²={r2_sigma_ft:.4f}")
        print(f"    → {'FLAT (good)' if r2_sigma_ft < 0.3 else 'NOT flat'}")

    # ══════════════════════════════════════════════════════════
    # FIGURE: 3-panel honest diagnostic
    # ══════════════════════════════════════════════════════════
    print("\n  Generating figure...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('3D Flux Tube — Honest Confinement Analysis (v6)',
                 fontsize=13, fontweight='bold', y=1.02)

    # (a) E_path vs d — shows geometric scaling (E ∝ d)
    ax = axes[0]
    if n_ft > 0:
        ax.scatter(dists[~ft_mask], e_path[~ft_mask], c='gray', s=25, alpha=0.4, zorder=2)
        ax.scatter(dists[ft_mask], e_path[ft_mask], c='#D32F2F', s=50, alpha=0.8, zorder=4,
                   edgecolors='darkred', linewidths=0.5)
    else:
        ax.scatter(dists, e_path, c='#1976D2', s=40, alpha=0.7)
    xf = np.linspace(0, max(dists)*1.1, 100)
    ax.plot(xf, sigma_bg * xf, 'k--', lw=1.5, alpha=0.5,
            label=f'Background: σ_bg={sigma_bg:.0f}')
    c_ep, r2_ep = fit_r2(dists, e_path)
    ax.plot(xf, np.polyval(c_ep, xf), '-', color='silver', lw=1.5, alpha=0.6,
            label=f'Fit: slope={c_ep[0]:.0f}, $R^2$={r2_ep:.3f}')
    ax.set_xlabel('Separation d (phys)', fontsize=11)
    ax.set_ylabel('$E_{path}$ (path-integrated)', fontsize=11)
    ax.set_title('(a) E vs d (mostly geometric)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (b) σ = E/d vs d — KEY: should be flat for confinement
    ax = axes[1]
    if n_ft > 0:
        ax.scatter(dists[~ft_mask], sigma_per_d[~ft_mask], c='gray', s=25, alpha=0.4, zorder=2,
                   label=f'FT<1.5 (n={np.sum(~ft_mask)})')
        ax.scatter(dists[ft_mask], sigma_per_d[ft_mask], c='#D32F2F', s=50, alpha=0.8, zorder=4,
                   edgecolors='darkred', linewidths=0.5,
                   label=f'FT>1.5 (n={n_ft})')
    else:
        ax.scatter(dists, sigma_per_d, c='#1976D2', s=40, alpha=0.7)
    ax.axhline(sigma_bg, color='gray', ls='--', lw=1.5, alpha=0.5,
               label=f'σ_bg={sigma_bg:.0f}')
    if n_ft > 0:
        ax.axhline(np.mean(sigma_per_d[ft_mask]), color='red', ls='-', lw=1.5, alpha=0.5,
                   label=f'FT>1.5 mean={np.mean(sigma_per_d[ft_mask]):.0f}')
    ax.set_xlabel('Separation d (phys)', fontsize=11)
    ax.set_ylabel('σ = E/d (string tension)', fontsize=11)
    ax.set_title(f'(b) String tension vs d (R²={r2_sigma:.3f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # (c) FT ratio vs d
    ax = axes[2]
    if n_ft > 0:
        ax.scatter(dists[ft_mask], ft_arr[ft_mask], c='#D32F2F', s=40, alpha=0.7,
                   label=f'FT>1.5 (n={n_ft})', zorder=3)
        ax.scatter(dists[~ft_mask], ft_arr[~ft_mask], c='gray', s=20, alpha=0.4,
                   label=f'FT<1.5')
    else:
        ax.scatter(dists, ft_arr, c='#1976D2', s=40, alpha=0.7)
    ax.axhline(1.5, color='orange', ls='--', lw=1.5, label='FT threshold')
    ax.set_xlabel('Separation d (phys)', fontsize=11)
    ax.set_ylabel('Flux tube / background ratio', fontsize=11)
    ax.set_title(f'(c) Signal quality ({n_ft}/{len(dists)} pass)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('ed_strong_confinement_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: ed_strong_confinement_3d.png")

    mean_sigma_ft = float(np.mean(sigma_per_d[ft_mask])) if n_ft > 0 else 0
    mean_sigma_all = float(np.mean(sigma_per_d))

    output = {
        'method': '3D vortex-line tracing, honest E/d metric (v6)',
        'strip_half_width': STRIP_HALF_WIDTH,
        'core_exclude': CORE_EXCLUDE,
        'core_radius_grid': CORE_RADIUS_GRID,
        'n_pairs': len(confinement_3d),
        'sigma_bg': round(sigma_bg, 2),
        'sigma_mean_all': round(mean_sigma_all, 2),
        'sigma_mean_ft': round(mean_sigma_ft, 2),
        'sigma_vs_d_r2': round(r2_sigma, 4),
        'sigma_vs_d_r2_ft': round(r2_sigma_ft, 4) if n_ft >= 3 else None,
        'n_ft_pairs': n_ft,
        'bg_E_3d': round(bg_E_3d, 4),
        'pairs': confinement_3d,
    }
    with open('v5_confinement_3d.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("  Saved: v5_confinement_3d.json")

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  HONEST RESULT (v6):                                    ║")
    print(f"  ║  σ_bg (background) = {sigma_bg:.0f}                            ║")
    print(f"  ║  σ_FT (FT>1.5 mean) = {mean_sigma_ft:.0f}                          ║")
    print(f"  ║  σ=E/d vs d: R² = {r2_sigma:.4f} (flat → confinement)       ║")
    print(f"  ║  FT>1.5 / bg ratio = {ratio:.2f}x                            ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

elif len(confinement_3d) > 0:
    print(f"  Only {len(confinement_3d)} pairs — printing raw data:")
    for c in confinement_3d:
        print(f"    d={c['dist_3d']:.2f}  E_path={c['tube_E_path']:.1f}  FT={c['FT_ratio']:.2f}")
    output = {'method': 'v6', 'n_pairs': len(confinement_3d), 'pairs': confinement_3d}
    with open('v5_confinement_3d.json', 'w') as f:
        json.dump(output, f, indent=2)

else:
    print("  WARNING: No pairs found.")

# ══════════════════════════════════════════════════════════════
# STEP 5: Parameter sensitivity scan
# ══════════════════════════════════════════════════════════════
print("\n  Step 5: Parameter sensitivity scan...")
print("  ┌──────────────────────────────────────────────────────────────────────┐")
print("  │  strip_hw  core_excl │  n_all  σ_all   R²_all │  n_ft  σ_ft   R²_ft │")
print("  ├──────────────────────┼────────────────────────┼──────────────────────┤")

scan_results = []
for shw in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for cex in [1.0, 1.5, 2.0, 2.5]:
        pairs = measure_pairs(long_plus, long_minus, shw, cex, MIN_OVERLAP_Z, verbose=False)
        if len(pairs) < 3:
            continue
        d = np.array([p['dist_3d'] for p in pairs])
        e = np.array([p['tube_E_path'] for p in pairs])
        ft = np.array([p['FT_ratio'] for p in pairs])
        ca, ra = fit_r2(d, e)
        ftm = ft > 1.5
        nf = int(np.sum(ftm))
        if nf >= 3:
            cf, rf = fit_r2(d[ftm], e[ftm])
        else:
            cf, rf = ca, ra
        marker = " ◀ default" if (shw == 1.5 and cex == 1.5) else ""
        print(f"  │  {shw:4.1f}      {cex:4.1f}      │  {len(pairs):3d}   {ca[0]:6.0f}  {ra:.4f} │"
              f"  {nf:3d}   {cf[0]:6.0f}  {rf:.4f} │{marker}")
        scan_results.append({
            'strip_hw': shw, 'core_excl': cex,
            'n_all': len(pairs), 'sigma_all': round(float(ca[0]),1), 'r2_all': round(ra,4),
            'n_ft': nf, 'sigma_ft': round(float(cf[0]),1), 'r2_ft': round(rf,4),
        })

print("  └──────────────────────────────────────────────────────────────────────┘")

r2_all_vals = [s['r2_all'] for s in scan_results]
r2_ft_vals  = [s['r2_ft'] for s in scan_results if s['n_ft'] >= 3]
print(f"\n  R²_all  range: {min(r2_all_vals):.4f} — {max(r2_all_vals):.4f}")
if r2_ft_vals:
    print(f"  R²_ft   range: {min(r2_ft_vals):.4f} — {max(r2_ft_vals):.4f}")

if min(r2_all_vals) > 0.90:
    print("  ✓ ROBUST: R²_all > 0.90 across ALL parameter choices")
elif min(r2_all_vals) > 0.80:
    print("  ✓ STABLE: R²_all > 0.80 across all parameter choices")
else:
    print("  ⚠️  SENSITIVE: R²_all varies significantly with parameters")

elapsed = timer.time() - t0
print(f"\n  Total time: {elapsed:.1f}s")
print("=" * 65)
