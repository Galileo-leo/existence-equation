"""
analyze_v5_tail_only.py
========================
Tail-region-only analysis comparing hydrogen 1s vs Rayleigh radial PDFs.

Uses the stable segment (t >= T_CUT) of the ED v5 electron radial distance r. Fits both
hydrogen_1s(r) and rayleigh(r) to the full-range normalized histogram, then evaluates those
fits in tail bins where large-r behavior differs: H 1s decays ~ exp(-2r/a0) (Coulomb-type
exponential tail in r) while Rayleigh falls ~ exp(-r^2/(2 sigma^2)) (Gaussian tail).

Segments (phys units): 4-5, 5-6, 6-7, 7-8, 8-9, 9-10, 10+ (fine-grained tail).
Compares segment-mean model predictions to binned data via ratio tables and a summary figure.

Input:  v5_hydrogen_proton2_track.json
Output: v5_tail_only.png, v5_tail_only.json

Jae-Ahn Shin & Kael · 2026-03-24
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
# Late-time samples only so the tail is not dominated by transient orbits
r = r_all[t_all >= T_CUT]

print(f"  Stable points: {len(r)}")
print(f"  r: mean={np.mean(r):.3f}, max={np.max(r):.3f}")

# ══════════════════════════════════════════════════════════════
# Fit global shape on 0..r_max, then score each tail segment (same parameters everywhere)
# ══════════════════════════════════════════════════════════════

def hydrogen_1s(r, a0, A):
    return A * r**2 * np.exp(-2*r / a0)

def rayleigh(r, sig, A):
    return A * r * np.exp(-r**2 / (2 * sig**2))

r_max = min(np.percentile(r, 99.9), np.max(r))
bins_full = np.linspace(0, r_max, 80)
cnt_full, edg_full = np.histogram(r, bins=bins_full, density=True)
rm_full = 0.5 * (edg_full[:-1] + edg_full[1:])

popt_h, _ = curve_fit(hydrogen_1s, rm_full, cnt_full, p0=[2.0, 1.0], maxfev=5000)
popt_r, _ = curve_fit(rayleigh, rm_full, cnt_full, p0=[2.0, 1.0], maxfev=5000)

print(f"  Full fit H 1s: a0={popt_h[0]:.4f}")
print(f"  Full fit Rayleigh: sigma={popt_r[0]:.4f}")

# ══════════════════════════════════════════════════════════════
# Analysis by segment
# ══════════════════════════════════════════════════════════════

segments = [
    (4, 5, '4-5'),
    (5, 6, '5-6'),
    (6, 7, '6-7'),
    (7, 8, '7-8'),
    (8, 9, '8-9'),
    (9, 10, '9-10'),
    (10, 14, '10+'),
]

print(f"\n  {'Segment':>10s}  {'N':>6s}  {'Data_mean':>10s}  {'H1s_pred':>10s}  {'Ray_pred':>10s}  {'H1s/Data':>10s}  {'Ray/Data':>10s}")
print(f"  {'─'*75}")

seg_results = []
for rlo, rhi, label in segments:
    mask = (r >= rlo) & (r < rhi)
    n_pts = int(np.sum(mask))
    if n_pts < 10:
        print(f"  {label:>10s}  {n_pts:>6d}  too few")
        continue

    n_bins_seg = max(min(int(n_pts / 30), 20), 5)
    bins_seg = np.linspace(rlo, rhi, n_bins_seg)
    cnt_seg, edg_seg = np.histogram(r[mask], bins=bins_seg, density=False)
    cnt_seg_norm = cnt_seg / (len(r) * (edg_seg[1] - edg_seg[0]))
    rm_seg = 0.5 * (edg_seg[:-1] + edg_seg[1:])

    h1s_pred = hydrogen_1s(rm_seg, *popt_h)
    ray_pred = rayleigh(rm_seg, *popt_r)

    valid_seg = cnt_seg_norm > 0
    if not np.any(valid_seg):
        print(f"  {label:>10s}  {n_pts:>6d}  no valid bins")
        continue
    data_mean = float(np.mean(cnt_seg_norm[valid_seg]))
    h1s_mean = float(np.mean(h1s_pred[valid_seg]))
    ray_mean = float(np.mean(ray_pred[valid_seg]))

    h1s_ratio = h1s_mean / data_mean if data_mean > 0 else 0
    ray_ratio = ray_mean / data_mean if data_mean > 0 else 0

    seg_results.append({
        'label': label, 'rlo': rlo, 'rhi': rhi,
        'n_points': n_pts,
        'data_mean': round(data_mean, 6),
        'h1s_mean': round(h1s_mean, 6),
        'ray_mean': round(ray_mean, 6),
        'h1s_ratio': round(h1s_ratio, 4),
        'ray_ratio': round(ray_ratio, 4),
    })

    print(f"  {label:>10s}  {n_pts:>6d}  {data_mean:>10.6f}  {h1s_mean:>10.6f}  {ray_mean:>10.6f}  {h1s_ratio:>10.4f}  {ray_ratio:>10.4f}")


# ══════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('P(r) Tail: H 1s vs Rayleigh (segment comparison)', fontsize=14, fontweight='bold')

rf = np.linspace(0.01, r_max, 500)

# (a) Full P(r) log scale -- full overview
ax = axes[0, 0]
valid = cnt_full > 1e-5
ax.semilogy(rm_full[valid], cnt_full[valid], 'ko', ms=4, alpha=0.6, label='Data', zorder=3)
ax.semilogy(rf, hydrogen_1s(rf, *popt_h), 'r-', lw=2.5, label=f'H 1s (a0={popt_h[0]:.2f})')
ax.semilogy(rf, rayleigh(rf, *popt_r), 'g--', lw=2.5, label=f'Rayleigh (s={popt_r[0]:.2f})')
for rlo, rhi, label in segments:
    ax.axvspan(rlo, min(rhi, r_max), alpha=0.08, color='blue')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r)  [log scale]', fontsize=12)
ax.set_title('(a) Full P(r) log-scale', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(bottom=1e-5)
ax.set_xlim(0, r_max)

# (b) Tail zoom: r = 4 ~ max, linear scale
ax = axes[0, 1]
tail_mask_full = rm_full >= 4.0
if np.any(tail_mask_full):
    ax.bar(rm_full[tail_mask_full], cnt_full[tail_mask_full],
           width=edg_full[1]-edg_full[0], alpha=0.4, color='steelblue', label='Data')
rf_tail = np.linspace(4.0, r_max, 300)
ax.plot(rf_tail, hydrogen_1s(rf_tail, *popt_h), 'r-', lw=2.5, label='H 1s')
ax.plot(rf_tail, rayleigh(rf_tail, *popt_r), 'g--', lw=2.5, label='Rayleigh')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r)', fontsize=12)
ax.set_title('(b) TAIL ZOOM r > 4 (linear)', fontsize=12)
ax.legend(fontsize=10)

# (c) Tail zoom: r = 4 ~ max, LOG scale
ax = axes[1, 0]
if np.any(tail_mask_full & (cnt_full > 0)):
    ax.semilogy(rm_full[tail_mask_full & (cnt_full > 0)],
                cnt_full[tail_mask_full & (cnt_full > 0)],
                'ko', ms=5, alpha=0.7, label='Data', zorder=3)
ax.semilogy(rf_tail, hydrogen_1s(rf_tail, *popt_h), 'r-', lw=2.5, label='H 1s')
ray_tail = rayleigh(rf_tail, *popt_r)
ray_tail[ray_tail < 1e-30] = 1e-30
ax.semilogy(rf_tail, ray_tail, 'g--', lw=2.5, label='Rayleigh')
ax.set_xlabel('r (phys)', fontsize=12)
ax.set_ylabel('P(r)  [log scale]', fontsize=12)
ax.set_title('(c) TAIL ZOOM r > 4 (log) -- KEY PANEL', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(bottom=1e-5)

# (d) Segment-by-segment ratio bar chart
ax = axes[1, 1]
if seg_results:
    labels = [s['label'] for s in seg_results]
    h_ratios = [s['h1s_ratio'] for s in seg_results]
    r_ratios = [s['ray_ratio'] for s in seg_results]

    x = np.arange(len(labels))
    w = 0.35
    bars_h = ax.bar(x - w/2, h_ratios, w, color='red', alpha=0.7, label='H 1s / Data')
    bars_r = ax.bar(x + w/2, r_ratios, w, color='green', alpha=0.7, label='Rayleigh / Data')

    ax.axhline(1.0, color='black', ls='-', lw=1.5, label='Perfect = 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Model / Data ratio', fontsize=12)
    ax.set_title('(d) Model accuracy by segment', fontsize=12)
    ax.legend(fontsize=10)

    for bar, val in zip(bars_h, h_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='red')
    for bar, val in zip(bars_r, r_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('v5_tail_only.png', dpi=150, bbox_inches='tight')
print(f"\n  Saved: v5_tail_only.png")
plt.close()

# ══════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════

with open('v5_tail_only.json', 'w') as f:
    json.dump({
        'full_fit': {
            'H1s': {'a0': round(popt_h[0], 4), 'A': round(popt_h[1], 4)},
            'Rayleigh': {'sigma': round(popt_r[0], 4), 'A': round(popt_r[1], 4)},
        },
        'segments': seg_results,
        'n_stable': len(r),
    }, f, indent=2)
print(f"  Saved: v5_tail_only.json")

print(f"\n{'='*60}")
print(f"  TAIL VERDICT")
print(f"{'='*60}")
for s in seg_results:
    h_err = abs(1.0 - s['h1s_ratio'])
    r_err = abs(1.0 - s['ray_ratio'])
    winner = 'H 1s' if h_err < r_err else 'Rayleigh'
    print(f"  {s['label']:>6s}: H1s/Data={s['h1s_ratio']:.3f}, Ray/Data={s['ray_ratio']:.3f} -> {winner}")
print(f"{'='*60}")
