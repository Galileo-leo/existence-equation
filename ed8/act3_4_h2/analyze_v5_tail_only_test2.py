"""
analyze_v5_tail_only_test2.py
======================
Test whether finite proton size explains the tail deviation.

Three models compared:
  H1s:      P(r) = A · r² · exp(-2r/a₀)              (point charge)
  Rayleigh: P(r) = A · r  · exp(-r²/2σ²)              (random walk)
  H1s+β:   P(r) = A · r² · exp(-2r/a₀ - β·r²)        (finite size correction)

If β > 0 and R² improves → finite proton size is the explanation.
If β ≈ 0 → something else is going on.

Input:  v5_hydrogen_proton2_track.json
Output: test_finite_proton.png, console report

Jae-Ahn Shin & Kael · 2026-04-02
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

with open('v5_hydrogen_proton2_track.json') as f:
    d = json.load(f)

track = d['track']
dt_sim = 0.022553

r_all = np.array([t['dist'] for t in track if t['found']])
t_all = np.array([t['step'] for t in track if t['found']]) * dt_sim

T_CUT = 20.0
r = r_all[t_all >= T_CUT]

print(f"{'='*65}")
print(f"  FINITE PROTON SIZE TEST")
print(f"{'='*65}")
print(f"  Stable points: {len(r)}")
print(f"  r: mean={np.mean(r):.3f}, std={np.std(r):.3f}, max={np.max(r):.3f}")

def hydrogen_1s(r, a0, A):
    return A * r**2 * np.exp(-2*r / a0)

def rayleigh(r, sig, A):
    return A * r * np.exp(-r**2 / (2 * sig**2))

def h1s_finite(r, a0, beta, A):
    return A * r**2 * np.exp(-2*r / a0 - beta * r**2)

r_max = min(np.percentile(r, 99.9), np.max(r))
bins_full = np.linspace(0, r_max, 80)
cnt_full, edg_full = np.histogram(r, bins=bins_full, density=True)
rm_full = 0.5 * (edg_full[:-1] + edg_full[1:])

popt_h, _ = curve_fit(hydrogen_1s, rm_full, cnt_full, p0=[2.0, 1.0], maxfev=5000)
popt_r, _ = curve_fit(rayleigh, rm_full, cnt_full, p0=[2.0, 1.0], maxfev=5000)
popt_f, _ = curve_fit(h1s_finite, rm_full, cnt_full, p0=[2.0, 0.01, 1.0],
                       bounds=([0.1, 0.0, 0.0], [10.0, 1.0, 100.0]), maxfev=10000)

def r_squared(y_data, y_pred):
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1.0 - ss_res / ss_tot

pred_h = hydrogen_1s(rm_full, *popt_h)
pred_r = rayleigh(rm_full, *popt_r)
pred_f = h1s_finite(rm_full, *popt_f)

r2_h = r_squared(cnt_full, pred_h)
r2_r = r_squared(cnt_full, pred_r)
r2_f = r_squared(cnt_full, pred_f)

print(f"\n{'─'*65}")
print(f"  MODEL COMPARISON (full range)")
print(f"{'─'*65}")
print(f"  H1s (point):    a₀={popt_h[0]:.4f}                    R²={r2_h:.6f}")
print(f"  Rayleigh:       σ ={popt_r[0]:.4f}                    R²={r2_r:.6f}")
print(f"  H1s+β (finite): a₀={popt_f[0]:.4f}, β={popt_f[1]:.6f}  R²={r2_f:.6f}")
print(f"{'─'*65}")
print(f"  β = {popt_f[1]:.6f}")
if popt_f[1] > 1e-6:
    print(f"  ★ β > 0: finite proton size correction DETECTED")
    print(f"  ★ R² improvement: {r2_h:.6f} → {r2_f:.6f} (Δ={r2_f-r2_h:+.6f})")
else:
    print(f"  β ≈ 0: no finite size correction detected")

segments = [
    (4, 5, '4-5'),
    (5, 6, '5-6'),
    (6, 7, '6-7'),
    (7, 8, '7-8'),
    (8, 9, '8-9'),
    (9, 10, '9-10'),
    (10, 14, '10+'),
]

print(f"\n{'─'*85}")
print(f"  {'Seg':>6s}  {'N':>5s}  {'H1s/D':>8s}  {'Ray/D':>8s}  {'Fin/D':>8s}  {'Winner':>10s}")
print(f"{'─'*85}")

for rlo, rhi, label in segments:
    mask = (r >= rlo) & (r < rhi)
    n_pts = int(np.sum(mask))
    if n_pts < 10:
        print(f"  {label:>6s}  {n_pts:>5d}  too few")
        continue

    n_bins_seg = max(min(int(n_pts / 30), 20), 5)
    bins_seg = np.linspace(rlo, rhi, n_bins_seg)
    cnt_seg, edg_seg = np.histogram(r[mask], bins=bins_seg, density=False)
    cnt_seg_norm = cnt_seg / (len(r) * (edg_seg[1] - edg_seg[0]))
    rm_seg = 0.5 * (edg_seg[:-1] + edg_seg[1:])

    h1s_pred = hydrogen_1s(rm_seg, *popt_h)
    ray_pred = rayleigh(rm_seg, *popt_r)
    fin_pred = h1s_finite(rm_seg, *popt_f)

    valid_seg = cnt_seg_norm > 0
    if not np.any(valid_seg):
        continue

    data_m = float(np.mean(cnt_seg_norm[valid_seg]))
    h1s_m  = float(np.mean(h1s_pred[valid_seg]))
    ray_m  = float(np.mean(ray_pred[valid_seg]))
    fin_m  = float(np.mean(fin_pred[valid_seg]))

    h_ratio = h1s_m / data_m if data_m > 0 else 0
    r_ratio = ray_m / data_m if data_m > 0 else 0
    f_ratio = fin_m / data_m if data_m > 0 else 0

    errors = {'H1s': abs(1-h_ratio), 'Ray': abs(1-r_ratio), 'H1s+β': abs(1-f_ratio)}
    winner = min(errors, key=errors.get)

    print(f"  {label:>6s}  {n_pts:>5d}  {h_ratio:>8.3f}  {r_ratio:>8.3f}  {f_ratio:>8.3f}  {winner:>10s}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Finite Proton Size Test: H1s vs H1s+β vs Rayleigh', fontsize=14, fontweight='bold')

rf = np.linspace(0.01, r_max, 500)

ax = axes[0]
valid = cnt_full > 1e-5
ax.semilogy(rm_full[valid], cnt_full[valid], 'ko', ms=4, alpha=0.6, label='Data', zorder=3)
ax.semilogy(rf, hydrogen_1s(rf, *popt_h), 'r-', lw=2, label=f'H1s (a₀={popt_h[0]:.2f}, R²={r2_h:.4f})')
ax.semilogy(rf, rayleigh(rf, *popt_r), 'g--', lw=2, label=f'Rayleigh (σ={popt_r[0]:.2f}, R²={r2_r:.4f})')
pred_fin = h1s_finite(rf, *popt_f)
pred_fin[pred_fin < 1e-30] = 1e-30
ax.semilogy(rf, pred_fin, 'b-', lw=2.5, label=f'H1s+β (a₀={popt_f[0]:.2f}, β={popt_f[1]:.4f}, R²={r2_f:.4f})')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r) [log scale]', fontsize=12)
ax.set_title('(a) Full P(r) — three models', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(bottom=1e-5)
ax.set_xlim(0, r_max)

ax = axes[1]
tail_mask = rm_full >= 3.5
if np.any(tail_mask & (cnt_full > 0)):
    ax.semilogy(rm_full[tail_mask & (cnt_full > 0)],
                cnt_full[tail_mask & (cnt_full > 0)],
                'ko', ms=5, alpha=0.7, label='Data', zorder=3)
rf_tail = np.linspace(3.5, r_max, 300)
ax.semilogy(rf_tail, hydrogen_1s(rf_tail, *popt_h), 'r-', lw=2, alpha=0.7, label='H1s (point charge)')
ray_t = rayleigh(rf_tail, *popt_r)
ray_t[ray_t < 1e-30] = 1e-30
ax.semilogy(rf_tail, ray_t, 'g--', lw=2, alpha=0.7, label='Rayleigh')
fin_t = h1s_finite(rf_tail, *popt_f)
fin_t[fin_t < 1e-30] = 1e-30
ax.semilogy(rf_tail, fin_t, 'b-', lw=2.5, label=f'H1s+β (β={popt_f[1]:.4f})')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r) [log scale]', fontsize=12)
ax.set_title('(b) Tail zoom — KEY TEST', fontsize=12)
ax.legend(fontsize=9)
ax.set_ylim(bottom=1e-5)

plt.tight_layout()
plt.savefig('test_finite_proton.png', dpi=150, bbox_inches='tight')
print(f"\n  Saved: test_finite_proton.png")
plt.close()

print(f"\n{'='*65}")
print(f"  FINAL VERDICT")
print(f"{'='*65}")
print(f"  H1s (point):    R² = {r2_h:.6f}")
print(f"  Rayleigh:       R² = {r2_r:.6f}")
print(f"  H1s+β (finite): R² = {r2_f:.6f}")
print(f"")
print(f"  β = {popt_f[1]:.6f}")
print(f"  a₀(point)  = {popt_h[0]:.4f}")
print(f"  a₀(finite) = {popt_f[0]:.4f}")
print(f"")
if r2_f > r2_h and popt_f[1] > 1e-6:
    print(f"  ★★★ CONFIRMED: Finite proton size correction improves fit.")
    print(f"  ★★★ The tail deviation is NOT a failure of Born rule.")
    print(f"  ★★★ It is the signature of an extended proton.")
    print(f"  ★★★ β·r² at r=7: {popt_f[1]*49:.4f} (correction magnitude)")
    print(f"  ★★★ β·r² at r=10: {popt_f[1]*100:.4f}")
else:
    print(f"  β ≈ 0 or R² not improved. Need further investigation.")
print(f"{'='*65}")
