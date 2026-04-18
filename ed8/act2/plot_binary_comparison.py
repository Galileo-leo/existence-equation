"""
plot_binary_comparison.py — ED Binary Vortex: Expansion vs Static
=================================================================
Reads binary_noexpand.json and binary_expand.json,
produces ed_binary_vortex_result.png for the paper.

Usage:
  %cd /content/drive/MyDrive/existence_equation/ed8/act2
  %run plot_binary_comparison.py

Author: Jae-Ahn Shin & Kael · 2026-04-06
=================================================================
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ════════════════════════════════════════════════════════════════
#   FILE PATHS
# ════════════════════════════════════════════════════════════════

NOEXPAND_JSON = "binary_noexpand.json"
EXPAND_JSON   = "binary_expand.json"
OUTPUT_PNG    = "ed_binary_vortex_result.png"

# ════════════════════════════════════════════════════════════════
#   LOAD DATA
# ════════════════════════════════════════════════════════════════

with open(NOEXPAND_JSON, 'r') as f:
    d_ne = json.load(f)
with open(EXPAND_JSON, 'r') as f:
    d_ex = json.load(f)

def extract(data):
    h = data['history']
    steps = np.array([r['step'] for r in h])
    amp1  = np.array([r['amp1'] for r in h])
    amp2  = np.array([r['amp2'] for r in h])
    dist  = np.array([r['dist'] for r in h])
    a_t   = np.array([r.get('a_t', 1.0) for r in h])
    p = data['params']
    return steps, amp1, amp2, dist, a_t, p

s_ne, a1_ne, a2_ne, d_ne_arr, at_ne, p_ne = extract(d_ne)
s_ex, a1_ex, a2_ex, d_ex_arr, at_ex, p_ex = extract(d_ex)

V_VAC = p_ne['v_vac']
RELAX = p_ne['relax']

print(f"  Loaded: {NOEXPAND_JSON} ({len(s_ne)} points)")
print(f"  Loaded: {EXPAND_JSON} ({len(s_ex)} points)")

# ════════════════════════════════════════════════════════════════
#   FIGURE
# ════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 2.5, 2.5],
              hspace=0.35, wspace=0.30)

C_NE = '#2166ac'
C_EX = '#b2182b'
C_VAC = '#333333'

# (a) Separation
ax_sep = fig.add_subplot(gs[0, 0])
ax_sep.plot(s_ne, d_ne_arr, C_NE, lw=1, alpha=0.8)
ax_sep.axvline(RELAX, color='gray', ls='--', alpha=0.4, lw=0.8)
ax_sep.set_ylabel('Distance', fontsize=9)
ax_sep.set_title('(a) Inspiral → Merger', fontsize=11, fontweight='bold')
ax_sep.set_xlim(0, 4000)
ax_sep.grid(True, alpha=0.2)
ax_sep.text(1800, d_ne_arr[0]*0.5, 'merger\n(d→0)',
            ha='center', fontsize=8, color='gray')

# (b) Scale factor
ax_sf = fig.add_subplot(gs[0, 1])
ax_sf.plot(s_ex, at_ex, 'k-', lw=2)
ax_sf.set_ylabel('a(t)', fontsize=9)
ax_sf.set_title('(b) Cosmic Expansion', fontsize=11, fontweight='bold')
ax_sf.grid(True, alpha=0.2)
ax_sf.text(50000, 1.5, 'a(t) = 1 + R·(t − t₀)',
           fontsize=8, color='gray', style='italic')

# (c) No expansion
ax_ne = fig.add_subplot(gs[1, 0])
ax_ne.plot(s_ne, a1_ne, C_NE, lw=0.8, alpha=0.9)
ax_ne.axhline(V_VAC, color=C_VAC, ls=':', alpha=0.6, lw=1,
              label=f'A₀ = {V_VAC}')
ax_ne.axhline(0, color='gray', ls='-', alpha=0.15, lw=0.5)
ax_ne.set_ylabel('Amplitude A(center)', fontsize=9)
ax_ne.set_title('(c) Static Universe  (a = 1)',
                fontsize=11, fontweight='bold', color=C_NE)
ax_ne.set_ylim(-2, V_VAC * 1.15)
ax_ne.legend(fontsize=8, loc='upper right')
ax_ne.grid(True, alpha=0.2)
ax_ne.set_xlabel('Step', fontsize=9)
ax_ne.annotate('permanent oscillation\n(breather)',
               xy=(60000, 28), fontsize=9, color=C_NE,
               ha='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', fc='white',
                         ec=C_NE, alpha=0.7))

# (d) With expansion
ax_ex = fig.add_subplot(gs[1, 1])
ax_ex.plot(s_ex, a1_ex, C_EX, lw=0.8, alpha=0.9)
ax_ex.axhline(V_VAC, color=C_VAC, ls=':', alpha=0.6, lw=1,
              label=f'A₀ = {V_VAC}')
ax_ex.axhline(0, color='gray', ls='-', alpha=0.15, lw=0.5)
expand_start = p_ex.get('expand_start', 4000)
ax_ex.axvline(expand_start, color='green', ls='--', alpha=0.4, lw=0.8)
ax_ex.set_ylabel('Amplitude A(center)', fontsize=9)
ax_ex.set_title('(d) Expanding Universe  (a → 3.0)',
                fontsize=11, fontweight='bold', color=C_EX)
ax_ex.set_ylim(-2, V_VAC * 1.15)
ax_ex.legend(fontsize=8, loc='upper right')
ax_ex.grid(True, alpha=0.2)
ax_ex.set_xlabel('Step', fontsize=9)
ax_ex.annotate('damped → evaporation\n(A → 0)',
               xy=(70000, 5), fontsize=9, color=C_EX,
               ha='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', fc='white',
                         ec=C_EX, alpha=0.7))
ax_ex.text(expand_start + 1000, V_VAC * 1.05, 'expansion\nstarts',
           fontsize=7, color='green', alpha=0.7)

# (e) Overlay
ax_ov = fig.add_subplot(gs[2, :])
ax_ov.plot(s_ne, a1_ne, C_NE, lw=0.7, alpha=0.7, label='Static (a=1)')
ax_ov.plot(s_ex, a1_ex, C_EX, lw=0.7, alpha=0.7, label='Expanding (a→3)')
ax_ov.axhline(V_VAC, color=C_VAC, ls=':', alpha=0.4, lw=1)
ax_ov.axhline(0, color='gray', ls='-', alpha=0.15, lw=0.5)
ax_ov.set_ylabel('Amplitude A(center)', fontsize=9)
ax_ov.set_xlabel('Step', fontsize=9)
ax_ov.set_title('(e) Comparison: Static vs Expanding',
                fontsize=11, fontweight='bold')
ax_ov.set_ylim(-2, V_VAC * 1.15)
ax_ov.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax_ov.grid(True, alpha=0.2)

fig.suptitle(
    'Binary Vortex Merger: Post-Merger Dynamics\n'
    f'n = (+1, −1),  sep = {p_ne["sep"]},  ξ = {p_ne["xi"]},  '
    f'{p_ne["N"]}³ grid,  v_vac = {V_VAC}',
    fontsize=13, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"\n  ★ Saved: {OUTPUT_PNG}")
