"""
analyze_h2_multi.py — Post-analysis of Multi-Hydrogen H2 experiment
=====================================================================
Reads v5_h2_multi_track.json and produces:
  1. Per-atom binding statistics
  2. H2 sharing analysis (P_B-P_C focus)
  3. Electron hopping / loyalty timeseries
  4. H atom center-of-mass trajectories (Kepler check)
  5. Inter-H-atom distance evolution
  6. 6-panel publication figure

Author: Jae-Ahn Shin
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 65)
print("  H2 Multi-Hydrogen Analysis")
print("=" * 65)

with open('v5_h2_multi_track.json', 'r') as f:
    data = json.load(f)

pairs = data['pairs']
results = data['results']
n_steps = data['n_steps']
labels = [p['label'] for p in pairs]
proton_phys = {p['label']: p['proton_phys'] for p in pairs}

print(f"  Loaded: {len(pairs)} pairs, {n_steps} steps")
print(f"  Labels: {labels}")

# ══════════════════════════════════════════════════════════════
# 1. Binding summary
# ══════════════════════════════════════════════════════════════

print(f"\n  ── Binding Summary ──")
print(f"  {'Label':6s} {'Bound%':>7s} {'Mean_d':>7s} {'Loyal%':>7s} {'Hops':>6s}")
print(f"  {'-'*40}")
for label in labels:
    r = results[label]
    print(f"  {label:6s} {r['bound_pct']:6.1f}% {r['mean_dist']:7.2f} "
          f"{r.get('loyalty_pct', 0):6.1f}% {r.get('hops', 0):5d}")

# ══════════════════════════════════════════════════════════════
# 2. P_B-P_C sharing deep dive
# ══════════════════════════════════════════════════════════════

print(f"\n  ── P_B-P_C Sharing Analysis ──")

pb = next(p for p in pairs if p['label'] == 'P_B')
pc = next(p for p in pairs if p['label'] == 'P_C')

pb_track = pb['track']
pc_track = pc['track']

pb_nearest = [t.get('nearest_p', 'P_B') for t in pb_track]
pc_nearest = [t.get('nearest_p', 'P_C') for t in pc_track]

pb_at_pc = sum(1 for n in pb_nearest if n == 'P_C')
pc_at_pb = sum(1 for n in pc_nearest if n == 'P_B')
total = len(pb_nearest)

print(f"  P_B electron visits P_C: {pb_at_pc}/{total} = {pb_at_pc/total*100:.1f}%")
print(f"  P_C electron visits P_B: {pc_at_pb}/{total} = {pc_at_pb/total*100:.1f}%")
print(f"  Combined migration: {(pb_at_pc + pc_at_pb)/(2*total)*100:.1f}%")

# Hopping events
pb_hops_to_pc = 0
pc_hops_to_pb = 0
prev_b, prev_c = None, None
for i in range(len(pb_nearest)):
    if prev_b is not None:
        if prev_b == 'P_B' and pb_nearest[i] == 'P_C':
            pb_hops_to_pc += 1
        if prev_b == 'P_C' and pb_nearest[i] == 'P_B':
            pb_hops_to_pc += 1
    if prev_c is not None:
        if prev_c == 'P_C' and pc_nearest[i] == 'P_B':
            pc_hops_to_pb += 1
        if prev_c == 'P_B' and pc_nearest[i] == 'P_C':
            pc_hops_to_pb += 1
    prev_b = pb_nearest[i]
    prev_c = pc_nearest[i]

print(f"  P_B electron hop events: {pb_hops_to_pc}")
print(f"  P_C electron hop events: {pc_hops_to_pb}")

# ══════════════════════════════════════════════════════════════
# 3. Extract trajectories
# ══════════════════════════════════════════════════════════════

trajectories = {}
for p in pairs:
    label = p['label']
    steps = [t['step'] for t in p['track']]
    dists = [t['dist'] for t in p['track']]
    e_x = [t.get('e_x', 0) for t in p['track']]
    e_y = [t.get('e_y', 0) for t in p['track']]
    cm_x = [t.get('cm_x', 0) for t in p['track']]
    cm_y = [t.get('cm_y', 0) for t in p['track']]
    nearest = [t.get('nearest_p', label) for t in p['track']]

    trajectories[label] = {
        'steps': np.array(steps),
        'dists': np.array(dists),
        'e_x': np.array(e_x), 'e_y': np.array(e_y),
        'cm_x': np.array(cm_x), 'cm_y': np.array(cm_y),
        'nearest': nearest,
        'px': p['proton_phys'][0], 'py': p['proton_phys'][1],
    }

# ══════════════════════════════════════════════════════════════
# 4. Inter-H-atom CM analysis (Kepler)
# ══════════════════════════════════════════════════════════════

print(f"\n  ── Inter-H-atom CM Trajectories ──")

pair_combos = []
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        la, lb = labels[i], labels[j]
        ta, tb = trajectories[la], trajectories[lb]
        d_proton = np.sqrt((ta['px'] - tb['px'])**2 + (ta['py'] - tb['py'])**2)

        min_len = min(len(ta['cm_x']), len(tb['cm_x']))
        cm_a = np.column_stack([ta['cm_x'][:min_len], ta['cm_y'][:min_len]])
        cm_b = np.column_stack([tb['cm_x'][:min_len], tb['cm_y'][:min_len]])

        d_cm = np.sqrt(np.sum((cm_a - cm_b)**2, axis=1))
        rel = cm_a - cm_b
        angles = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        total_angle = np.degrees(angles[-1] - angles[0])

        combo = {
            'pair': f"{la}-{lb}", 'd_proton': d_proton,
            'cm_d_mean': np.mean(d_cm), 'cm_d_std': np.std(d_cm),
            'angle_sweep': total_angle,
            'rel_x': rel[:, 0], 'rel_y': rel[:, 1],
            'd_cm': d_cm, 'angles': angles,
        }
        pair_combos.append(combo)

        flag = ""
        if abs(total_angle) > 90:
            flag = " ★ ORBITAL!"
        if d_proton < 5:
            flag += " ★★ H2!"

        print(f"  {la}-{lb}: p_d={d_proton:.1f}, cm_d={np.mean(d_cm):.1f}±{np.std(d_cm):.1f}, "
              f"sweep={total_angle:+.0f}°{flag}")

# ══════════════════════════════════════════════════════════════
# 5. Figure — 6 panels
# ══════════════════════════════════════════════════════════════

colors = {'P_A': '#1565C0', 'P_B': '#D32F2F', 'P_C': '#2E7D32',
          'P_D': '#FF6F00', 'P_E': '#7B1FA2'}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ED Multi-Hydrogen H₂ Experiment\n'
             '5 proton candidates (sign=+) × 1 electron, 30k steps',
             fontsize=14, fontweight='bold')

# (a) Electron-proton distance vs time
ax = axes[0, 0]
for label in labels:
    t = trajectories[label]
    subsample = slice(None, None, 50)
    ax.plot(t['steps'][subsample], t['dists'][subsample],
            color=colors[label], lw=0.8, alpha=0.7, label=label)
ax.set_xlabel('Step'); ax.set_ylabel('Electron-proton distance')
ax.set_title('(a) Binding distance vs time')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
ax.set_ylim(0, 15)

# (b) P_B-P_C electron sharing (nearest_p timeseries)
ax = axes[0, 1]
pb_at = np.array([1 if n == 'P_C' else 0 for n in pb_nearest])
pc_at = np.array([1 if n == 'P_B' else 0 for n in pc_nearest])
window = min(500, len(pb_at) // 10)
if window > 0:
    pb_frac = np.convolve(pb_at, np.ones(window)/window, mode='valid')
    pc_frac = np.convolve(pc_at, np.ones(window)/window, mode='valid')
    x_conv = np.arange(len(pb_frac))
    ax.plot(x_conv, pb_frac * 100, color=colors['P_B'], lw=1.5,
            label='P_B e⁻ at P_C')
    ax.plot(x_conv, pc_frac * 100, color=colors['P_C'], lw=1.5,
            label='P_C e⁻ at P_B')
ax.set_xlabel('Step'); ax.set_ylabel('Migration %')
ax.set_title(f'(b) P_B↔P_C electron sharing (d={3.4:.1f})')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# (c) H atom CM trajectories (all 5)
ax = axes[0, 2]
for label in labels:
    t = trajectories[label]
    subsample = slice(None, None, 50)
    ax.plot(t['cm_x'][subsample], t['cm_y'][subsample],
            color=colors[label], lw=0.5, alpha=0.6)
    ax.plot(t['cm_x'][0], t['cm_y'][0], 'o', color=colors[label], ms=8)
    ax.plot(t['cm_x'][-1], t['cm_y'][-1], '*', color=colors[label], ms=12)
    ax.plot(t['px'], t['py'], '+', color=colors[label], ms=15, mew=2)
for label in labels:
    t = trajectories[label]
    ax.annotate(label, (t['px'], t['py']), fontsize=8, fontweight='bold',
                color=colors[label])
ax.set_xlabel('x (phys)'); ax.set_ylabel('y (phys)')
ax.set_title('(c) H atom center-of-mass trajectories')
ax.set_aspect('equal'); ax.grid(True, alpha=0.2)

# (d) P_B-P_C relative CM orbit
bc_combo = next(c for c in pair_combos if c['pair'] == 'P_B-P_C')
ax = axes[1, 0]
subsample = slice(None, None, 50)
ax.plot(bc_combo['rel_x'][subsample], bc_combo['rel_y'][subsample],
        'k-', lw=0.5, alpha=0.5)
ax.plot(bc_combo['rel_x'][0], bc_combo['rel_y'][0], 'go', ms=10, label='Start')
ax.plot(bc_combo['rel_x'][-1], bc_combo['rel_y'][-1], 'r*', ms=15, label='End')
ax.plot(0, 0, 'k+', ms=20, mew=3)
ax.set_xlabel('Δx (phys)'); ax.set_ylabel('Δy (phys)')
ax.set_title(f'(d) P_B-P_C relative CM (sweep={bc_combo["angle_sweep"]:+.0f}°)')
ax.set_aspect('equal'); ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

# (e) Inter-pair CM distance evolution (top pairs)
ax = axes[1, 1]
for combo in sorted(pair_combos, key=lambda c: c['d_proton'])[:5]:
    subsample = slice(None, None, 50)
    ax.plot(np.arange(len(combo['d_cm']))[subsample],
            combo['d_cm'][subsample],
            lw=1, alpha=0.8, label=f"{combo['pair']} (p_d={combo['d_proton']:.1f})")
ax.set_xlabel('Step'); ax.set_ylabel('CM distance')
ax.set_title('(e) Inter-H-atom CM distance')
ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# (f) Summary box
ax = axes[1, 2]
ax.axis('off')

sharing = data.get('sharing_counts', {})
h2_cands = data.get('h2_candidates', [])

txt = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
txt += "  H₂ EXPERIMENT RESULTS\n"
txt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
txt += f"  Protons: {len(pairs)} (sign=+)\n"
txt += f"  Electrons: {len(pairs)} (wind=-1)\n"
txt += f"  Steps: {n_steps}\n\n"

txt += "  Sharing frequency:\n"
for key in sorted(sharing, key=lambda k: -sharing[k]):
    cnt = sharing[key]
    pct = cnt / n_steps * 100
    tag = " ★★ H₂" if pct > 10 else ""
    txt += f"    {key}: {pct:.1f}%{tag}\n"

txt += f"\n  H₂ candidates: "
if h2_cands:
    txt += ', '.join(f"{k}({p:.0f}%)" for k, p in h2_cands)
else:
    txt += "None"

txt += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
txt += "  One equation.\n"
txt += "  No covalent bond postulate.\n"
txt += "  The field decides.\n"
txt += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                  edgecolor='green'))

plt.tight_layout()
plt.savefig('h2_multi_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: h2_multi_analysis.png (300 dpi)")

# ══════════════════════════════════════════════════════════════
# Final summary
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*65}")
print(f"  FINAL SUMMARY")
print(f"{'='*65}")

if h2_cands:
    print(f"\n  ★★ H₂ FORMATION DETECTED ★★")
    for key, pct in h2_cands:
        print(f"    {key}: electron sharing {pct:.1f}% of time")
else:
    print(f"\n  No sustained H₂ formation detected")

for combo in pair_combos:
    if abs(combo['angle_sweep']) > 90:
        print(f"\n  ★ ORBITAL MOTION: {combo['pair']}")
        print(f"    Angle sweep: {combo['angle_sweep']:+.0f}°")
        print(f"    CM distance: {combo['cm_d_mean']:.1f} ± {combo['cm_d_std']:.1f}")

print(f"\n{'='*65}")
