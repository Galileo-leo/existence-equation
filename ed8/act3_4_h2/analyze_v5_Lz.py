"""
analyze_v5_Lz.py
=================
Angular momentum Lz from the ED v5 electron trajectory (post-run analysis only).

Loads the saved relative-motion track from the 3D ED hydrogen run and estimates
Lz(t) and angular velocity omega(t). This probes whether the motion is consistent
with an m=0 (s-like) state versus a state with net orbital angular momentum.

Physics (2D relative coordinates, natural units m=1):
  Lz = r x p  ->  Lz = rx * vy - ry * vx
  omega = d(theta)/dt with theta unwrapped for cumulative rotation.

Expected signatures:
  m = 0:  mean Lz ~ 0, symmetric Lz distribution, no systematic winding of theta.
  m != 0: mean Lz != 0, monotonic drift in unwrapped theta.

The script compares statistics over all samples vs a late-time "stable" window
(t >= T_CUT) to reduce transient effects from initialization.

Input:  v5_hydrogen_proton2_track.json
Output: v5_Lz_analysis.json, ed_strong_Lz.png (300 dpi)

Jae-Ahn Shin & Kael · 2026-03-24
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 60)
print("v5 Angular Momentum (Lz) Analysis")
print("=" * 60)

with open('v5_hydrogen_proton2_track.json') as f:
    data = json.load(f)

track = data['track']
dt_sim = 0.022553

found_mask = [t['found'] for t in track]
rx = np.array([t['rx'] for t in track if t['found']])
ry = np.array([t['ry'] for t in track if t['found']])
theta = np.array([t['theta'] for t in track if t['found']])
dist = np.array([t['dist'] for t in track if t['found']])
steps = np.array([t['step'] for t in track if t['found']])
t_phys = steps * dt_sim

# Kinematics: central-difference velocity in simulation time units
vx = np.gradient(rx, dt_sim)
vy = np.gradient(ry, dt_sim)

# Canonical z-component of angular momentum (relative 2D motion)
Lz = rx * vy - ry * vx

# omega from unwrapped polar angle (avoids 2pi jumps)
theta_unwrap = np.unwrap(theta)
omega = np.gradient(theta_unwrap, dt_sim)

# Net rotation since t=0 (in radians; used for revolution count)
delta_theta = theta_unwrap - theta_unwrap[0]

# Late-time window: discard early transient before computing means / m=0 test
T_CUT = 20.0
stable = t_phys >= T_CUT

print(f"Total points: {len(rx)}")
print(f"Stable points (t >= {T_CUT}): {np.sum(stable)}")

# Statistics
def stats(arr, label):
    print(f"  {label}: mean={np.mean(arr):+.4f}, std={np.std(arr):.4f}, "
          f"median={np.median(arr):+.4f}")
    return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)), 'max': float(np.max(arr))}

print("\nALL data:")
Lz_all_stats = stats(Lz, "Lz")
omega_all_stats = stats(omega, "omega")

print("\nSTABLE data:")
Lz_stable_stats = stats(Lz[stable], "Lz")
omega_stable_stats = stats(omega[stable], "omega")

# Net rotation
total_revolutions = delta_theta[-1] / (2 * np.pi)
stable_start = np.argmax(stable)
stable_revolutions = (delta_theta[-1] - delta_theta[stable_start]) / (2 * np.pi)
stable_time = t_phys[-1] - t_phys[stable_start]

print(f"\nNet rotation (total): {total_revolutions:+.2f} revolutions")
print(f"Net rotation (stable): {stable_revolutions:+.2f} revolutions "
      f"in {stable_time:.1f} time")
print(f"Mean angular velocity (stable): {np.mean(omega[stable]):+.4f} rad/t")

# m=0 test: |<Lz>| / SEM < ~3  ->  consistent with zero mean (rough 3-sigma rule)
Lz_mean = np.mean(Lz[stable])
Lz_std = np.std(Lz[stable])
Lz_sem = Lz_std / np.sqrt(np.sum(stable))
snr = abs(Lz_mean) / Lz_sem if Lz_sem > 0 else 0

print(f"\n{'='*60}")
print(f"m=0 TEST:")
print(f"  <Lz> = {Lz_mean:+.4f} +/- {Lz_sem:.4f} (SEM)")
print(f"  |<Lz>| / SEM = {snr:.1f}")
if snr < 3:
    print(f"  RESULT: Consistent with m=0 (|<Lz>|/SEM < 3)")
    verdict = "CONSISTENT with m=0"
else:
    print(f"  RESULT: Lz significantly nonzero (|<Lz>|/SEM >= 3)")
    verdict = f"Lz significantly nonzero (SNR={snr:.1f})"
print(f"{'='*60}")

# For m=0, Lz PDF is often near-symmetric about zero (skewness ~ 0)
Lz_skew = float(np.mean(((Lz[stable] - Lz_mean) / Lz_std)**3))
print(f"  Lz skewness: {Lz_skew:+.3f} (0 = symmetric)")

# Save results
results = {
    'T_cut': T_CUT,
    'n_total': int(len(rx)),
    'n_stable': int(np.sum(stable)),
    'Lz_all': Lz_all_stats,
    'Lz_stable': Lz_stable_stats,
    'omega_stable': omega_stable_stats,
    'net_revolutions_total': float(total_revolutions),
    'net_revolutions_stable': float(stable_revolutions),
    'Lz_mean': float(Lz_mean),
    'Lz_sem': float(Lz_sem),
    'snr': float(snr),
    'Lz_skewness': Lz_skew,
    'verdict': verdict,
}

with open('v5_Lz_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: v5_Lz_analysis.json")

# ============================================================
# Figure
# ============================================================
plt.rcParams.update({
    'font.size': 11, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ED v5 Angular Momentum Analysis (3D simulation, NO toy model)',
             fontsize=14, fontweight='bold')

# (a) Lz(t)
ax = axes[0, 0]
ax.plot(t_phys, Lz, lw=0.3, alpha=0.5, color='steelblue')
ax.axhline(0, color='black', lw=1.5)
ax.axvline(T_CUT, color='orange', ls='--', lw=2, label=f'$T_{{cut}}$={T_CUT}')
ax.axhline(Lz_mean, color='red', ls='--', lw=1,
           label=f'$\\langle L_z \\rangle$={Lz_mean:+.3f}')
ax.set_xlabel('Time (phys)')
ax.set_ylabel('$L_z$')
ax.set_title('(a) $L_z(t)$')
ax.legend(fontsize=8)

# (b) Lz histogram
ax = axes[0, 1]
Lz_s = Lz[stable]
ax.hist(Lz_s, bins=100, density=True, alpha=0.7, color='steelblue',
        edgecolor='white', lw=0.3)
ax.axvline(0, color='black', lw=1.5)
ax.axvline(Lz_mean, color='red', ls='--', lw=1.5,
           label=f'mean={Lz_mean:+.3f}')
ax.set_xlabel('$L_z$')
ax.set_ylabel('Probability density')
ax.set_title(f'(b) $L_z$ distribution (skew={Lz_skew:+.2f})')
ax.legend(fontsize=9)

# (c) Cumulative angle
ax = axes[0, 2]
ax.plot(t_phys, delta_theta / (2*np.pi), lw=0.5, color='steelblue')
ax.axvline(T_CUT, color='orange', ls='--', lw=2)
ax.set_xlabel('Time (phys)')
ax.set_ylabel('Cumulative revolutions')
ax.set_title(f'(c) Net rotation: {stable_revolutions:+.1f} rev (stable)')

# (d) omega(t)
ax = axes[1, 0]
ax.plot(t_phys, omega, lw=0.2, alpha=0.4, color='purple')
ax.axhline(0, color='black', lw=1)
ax.axhline(np.mean(omega[stable]), color='red', ls='--', lw=1,
           label=f'$\\langle\\omega\\rangle$={np.mean(omega[stable]):+.3f}')
ax.set_xlabel('Time (phys)')
ax.set_ylabel('$\\omega$ (rad/t)')
ax.set_title('(d) Angular velocity')
ax.legend(fontsize=8)

# (e) omega histogram
ax = axes[1, 1]
om_s = omega[stable]
om_clip = om_s[np.abs(om_s) < np.percentile(np.abs(om_s), 99)]
ax.hist(om_clip, bins=100, density=True, alpha=0.7, color='purple',
        edgecolor='white', lw=0.3)
ax.axvline(0, color='black', lw=1.5)
ax.set_xlabel('$\\omega$ (rad/t)')
ax.set_ylabel('Probability density')
ax.set_title('(e) $\\omega$ distribution')

# (f) Summary text
ax = axes[1, 2]
ax.axis('off')
txt = (
    f"Angular Momentum Test\n"
    f"{'='*35}\n\n"
    f"Data: 3D ED simulation (256³)\n"
    f"Points: {np.sum(stable)} (stable)\n\n"
    f"<Lz> = {Lz_mean:+.4f}\n"
    f"std(Lz) = {Lz_std:.4f}\n"
    f"SEM = {Lz_sem:.4f}\n"
    f"|<Lz>|/SEM = {snr:.1f}\n"
    f"Skewness = {Lz_skew:+.3f}\n\n"
    f"Net rotation = {stable_revolutions:+.1f} rev\n"
    f"<omega> = {np.mean(omega[stable]):+.4f}\n\n"
    f"VERDICT:\n"
    f"  {verdict}\n\n"
    f"s-orbital (m=0):\n"
    f"  Lz ~ 0, no rotation bias\n"
    f"  isotropic probability cloud"
)
ax.text(0.05, 0.95, txt, transform=ax.transAxes,
        fontsize=10, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('ed_strong_Lz.png')
plt.close()
print("Saved: ed_strong_Lz.png")
