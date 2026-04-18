"""
analyze_v5_proton2_profile.py  v1.1
====================================
Proton #2 radial profile analysis on the frozen ED v5 field (measurement only).

Centers on the known Proton #2 location in physical coordinates, rolls the grid so that
site becomes the origin, and builds spherically averaged profiles vs 3D radius r:
  - Amplitude A(r)
  - Energy densities: gradient of A, phase-gradient (kinetic) part, double-well restore,
    and quartic self-interaction

v1.1 subtracts a spatially uniform background in each density using medians over the
shifted volume (A_bg = median(|psi|), etc.), so plots emphasize the defect over bulk.

Also reports: phase winding vs circle radius R in the best_z slice (charge quantization),
a 1D cross-section along x through the core for |psi| and E_phase (flux-tube view),
ROI-integrated excess energies inside r < ROI_R, and JSON export.

Input:  v5_frozen_state.npz
Output: v5_proton2_profile.png, v5_proton2_profile.json

Proton #2 reference: center=(-3.5,-4.2), z=best_z, sign=+, score=0.992

Jae-Ahn Shin & Kael · 2026-03-23
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def wrap_ph(dp):
    return (dp + np.pi) % (2*np.pi) - np.pi

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


print("=" * 65)
print("  Proton #2 Full Profile Analysis")
print("=" * 65)

data = np.load('v5_frozen_state.npz')
psi = data['psi']
dx = float(data['dx'])
L = float(data['L'])
N = int(data['N'])
best_z = int(data['best_z'])

lam = 1.0
alpha = 1.0
v_vac = np.sqrt(lam / alpha)

PROTON_X_PHYS = -9.6
PROTON_Y_PHYS = -17.6
pc_ix = int((PROTON_X_PHYS + L/2) / dx) % N
pc_iy = int((PROTON_Y_PHYS + L/2) / dx) % N
pc_iz = best_z

print(f"  N={N}, dx={dx:.4f}, L={L:.2f}, v_vac={v_vac:.4f}")
print(f"  Proton #2: grid=({pc_ix},{pc_iy},{pc_iz})")

amp = np.abs(psi)
phase = np.angle(psi)


# ══════════════════════════════════════════════════════════════
# (1) Spherical shells in 3D: mean fields vs r after shifting proton to box center
# ══════════════════════════════════════════════════════════════

print("\n  Computing 3D radial profiles...")

mid = N // 2
shifted_psi = np.roll(psi, (mid - pc_iz, mid - pc_iy, mid - pc_ix), axis=(0, 1, 2))
shifted_amp = np.abs(shifted_psi)
shifted_phase = np.angle(shifted_psi)

coords = np.arange(N) * dx
Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')
cx_phys = mid * dx
cy_phys = mid * dx
cz_phys = mid * dx
dX = X - cx_phys
dY = Y - cy_phys
dZ = Z - cz_phys
dX = dX - L * np.round(dX / L)
dY = dY - L * np.round(dY / L)
dZ = dZ - L * np.round(dZ / L)
r_3d = np.sqrt(dX**2 + dY**2 + dZ**2)

# Local energy densities (same splitting as analyze_v5_proton ROI)
E_grad_A = np.zeros_like(shifted_amp)
E_phase = np.zeros_like(shifted_amp)
for ax in range(3):
    dA = (np.roll(shifted_amp, -1, axis=ax) - np.roll(shifted_amp, 1, axis=ax)) / (2*dx)
    E_grad_A += 0.5 * dA**2
    dphi = wrap_ph(np.roll(shifted_phase, -1, axis=ax) - shifted_phase) / dx
    E_phase += 0.5 * shifted_amp**2 * dphi**2

E_restore = 0.5 * lam * shifted_amp**2
E_cond = 0.25 * alpha * shifted_amp**4

A_bg = float(np.median(shifted_amp))
print(f"  A_bg (median) = {A_bg:.4f}  (v_vac={v_vac:.4f})")

E_phase_bg = float(np.median(E_phase))
E_grad_A_bg = float(np.median(E_grad_A))
E_restore_bg = 0.5 * lam * A_bg**2
E_cond_bg = 0.25 * alpha * A_bg**4

# Excess over bulk: removes uniform floor so radial means highlight the proton core
E_phase_sub = E_phase - E_phase_bg
E_grad_A_sub = E_grad_A - E_grad_A_bg
E_restore_sub = E_restore - E_restore_bg
E_cond_sub = E_cond - E_cond_bg

R_MAX_PHYS = 12.0
N_BINS = 60
r_edges = np.linspace(0, R_MAX_PHYS, N_BINS + 1)
r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])

prof_amp = np.zeros(N_BINS)
prof_E_phase = np.zeros(N_BINS)
prof_E_grad_A = np.zeros(N_BINS)
prof_E_restore = np.zeros(N_BINS)
prof_E_cond = np.zeros(N_BINS)
prof_count = np.zeros(N_BINS)

r_flat = r_3d.ravel()
amp_flat = shifted_amp.ravel()
ep_flat = E_phase_sub.ravel()
eg_flat = E_grad_A_sub.ravel()
er_flat = E_restore_sub.ravel()
ec_flat = E_cond_sub.ravel()

bin_idx = np.digitize(r_flat, r_edges) - 1
valid = (bin_idx >= 0) & (bin_idx < N_BINS)

for i in range(N_BINS):
    mask = (bin_idx == i) & valid
    cnt = np.sum(mask)
    if cnt > 0:
        prof_amp[i] = np.mean(amp_flat[mask])
        prof_E_phase[i] = np.mean(ep_flat[mask])
        prof_E_grad_A[i] = np.mean(eg_flat[mask])
        prof_E_restore[i] = np.mean(er_flat[mask])
        prof_E_cond[i] = np.mean(ec_flat[mask])
        prof_count[i] = cnt


# ══════════════════════════════════════════════════════════════
# (2) Winding vs R in the z=best_z plane: plateau near integer -> effective vortex charge
# ══════════════════════════════════════════════════════════════

print("  Computing winding vs R...")

phase_slice = phase[pc_iz]
amp_slice = amp[pc_iz]

R_list = np.arange(2, min(N//2 - 2, 80), 1)
winding_vs_R = []
for R in R_list:
    R_phys = R * dx
    w = measure_winding_circle(phase_slice, pc_ix, pc_iy, R,
                                n_sample=max(int(8*R), 60))
    winding_vs_R.append({'R': int(R), 'R_phys': round(R_phys, 4), 'w': round(w, 4)})

R_phys_arr = np.array([d['R_phys'] for d in winding_vs_R])
w_arr = np.array([d['w'] for d in winding_vs_R])


# ══════════════════════════════════════════════════════════════
# (3) Flux tube: line cut through core along x at fixed y (2D slice at best_z)
# ══════════════════════════════════════════════════════════════

print("  Computing flux tube profile...")

phase_2d = phase[pc_iz]
amp_2d = amp[pc_iz]

w_2d_x = wrap_ph(np.roll(phase_2d, -1, axis=1) - phase_2d) / dx
w_2d_y = wrap_ph(np.roll(phase_2d, -1, axis=0) - phase_2d) / dx
E_phase_2d = 0.5 * amp_2d**2 * (w_2d_x**2 + w_2d_y**2)

grad_ax = (np.roll(amp_2d, -1, axis=1) - np.roll(amp_2d, 1, axis=1)) / (2*dx)
grad_ay = (np.roll(amp_2d, -1, axis=0) - np.roll(amp_2d, 1, axis=0)) / (2*dx)
E_grad_2d = 0.5 * (grad_ax**2 + grad_ay**2)

FT_LINE_LEN = 40
ft_x_profile = np.zeros(2 * FT_LINE_LEN + 1)
ft_y_profile = np.zeros(2 * FT_LINE_LEN + 1)
ft_amp_profile = np.zeros(2 * FT_LINE_LEN + 1)
ft_Ephase_profile = np.zeros(2 * FT_LINE_LEN + 1)

for i, offset in enumerate(range(-FT_LINE_LEN, FT_LINE_LEN + 1)):
    ix = (pc_ix + offset) % N
    ft_amp_profile[i] = amp_2d[pc_iy, ix]
    ft_Ephase_profile[i] = E_phase_2d[pc_iy, ix]

ft_offsets_phys = np.arange(-FT_LINE_LEN, FT_LINE_LEN + 1) * dx


# ══════════════════════════════════════════════════════════════
# (4) ROI energy totals
# ══════════════════════════════════════════════════════════════

print("  Computing ROI energy totals...")

ROI_R = 8.0
roi_mask = r_3d < ROI_R
dV = dx**3
n_roi = int(np.sum(roi_mask))

roi_Eph = float(np.sum(E_phase_sub[roi_mask])) * dV
roi_EgA = float(np.sum(E_grad_A_sub[roi_mask])) * dV
roi_Er = float(np.sum(E_restore_sub[roi_mask])) * dV
roi_Ec = float(np.sum(E_cond_sub[roi_mask])) * dV
roi_Et = roi_Eph + roi_EgA + roi_Er + roi_Ec

print(f"    ROI (r<{ROI_R}): {n_roi} cells")
print(f"    E_phase    = {roi_Eph:12.2f}  ({roi_Eph/max(abs(roi_Et),1e-10)*100:5.1f}%)")
print(f"    E_grad_A   = {roi_EgA:12.2f}  ({roi_EgA/max(abs(roi_Et),1e-10)*100:5.1f}%)")
print(f"    E_restore  = {roi_Er:12.2f}  ({roi_Er/max(abs(roi_Et),1e-10)*100:5.1f}%)")
print(f"    E_cond     = {roi_Ec:12.2f}  ({roi_Ec/max(abs(roi_Et),1e-10)*100:5.1f}%)")
print(f"    E_total    = {roi_Et:12.2f}")


# ══════════════════════════════════════════════════════════════
# Figure: 2x3 panels
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) A(r)
ax = axes[0, 0]
ax.plot(r_mids, prof_amp, 'b-', lw=2)
ax.axhline(v_vac, color='red', ls='--', lw=1, label=f'v_vac={v_vac:.2f}')
ax.axhline(A_bg, color='orange', ls='--', lw=1, label=f'A_bg={A_bg:.2f}')
ax.set_xlabel('r (phys)')
ax.set_ylabel('A(r)')
ax.set_title('(a) Amplitude profile A(r)')
ax.legend(fontsize=8)

# (b) Energy density profiles
ax = axes[0, 1]
ax.plot(r_mids, prof_E_phase, 'r-', lw=2, label='E_phase')
ax.plot(r_mids, prof_E_grad_A, 'b-', lw=2, label='E_grad_A')
ax.plot(r_mids, prof_E_cond, 'g-', lw=2, label='E_cond')
ax.plot(r_mids, prof_E_restore, 'm-', lw=2, label='E_restore')
ax.set_xlabel('r (phys)')
ax.set_ylabel('Energy density')
ax.set_title('(b) Excess energy density vs r (bg subtracted)')
ax.legend(fontsize=8)
ax.set_yscale('symlog', linthresh=1.0)
ax.axhline(0, color='gray', ls='-', lw=0.5)

# (c) Winding vs R
ax = axes[0, 2]
ax.plot(R_phys_arr, w_arr, 'ko-', ms=2, lw=1)
ax.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5, label='Q=+1')
ax.axhline(0.0, color='gray', ls='-', lw=0.5)
ax.set_xlabel('R (phys)')
ax.set_ylabel('Winding number')
ax.set_title('(c) Charge quantization: winding vs R')
ax.legend()

# (d) Flux tube: amplitude cross-section
ax = axes[1, 0]
ax.plot(ft_offsets_phys, ft_amp_profile, 'b-', lw=2)
ax.axhline(v_vac, color='red', ls='--', lw=1, label=f'v_vac={v_vac:.2f}')
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_xlabel('offset from proton (phys)')
ax.set_ylabel('A')
ax.set_title('(d) Flux tube: A cross-section (y=const)')
ax.legend()

# (e) Flux tube: phase energy cross-section
ax = axes[1, 1]
ax.plot(ft_offsets_phys, ft_Ephase_profile, 'r-', lw=2)
ax.axvline(0, color='gray', ls=':', lw=1)
ax.set_xlabel('offset from proton (phys)')
ax.set_ylabel('E_phase density')
ax.set_title('(e) Flux tube: E_phase cross-section')

# (f) ROI energy pie
ax = axes[1, 2]
labels = ['E_phase', 'E_grad_A', 'E_restore', 'E_cond']
vals = [roi_Eph, roi_EgA, roi_Er, roi_Ec]
colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
bars = ax.bar(labels, vals, color=colors)
ax.axhline(0, color='gray', ls='-', lw=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v, f'{v:.0f}',
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
ax.set_ylabel('Excess energy')
ax.set_title(f'(f) ROI excess energy (r<{ROI_R}, bg subtracted)')

plt.suptitle('Proton #2 Full Profile (v5_frozen_state)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('v5_proton2_profile.png', dpi=200, bbox_inches='tight')
plt.show()
print(f"\n  Saved: v5_proton2_profile.png")


# ══════════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════════

result = {
    'proton': {
        'center_phys': [PROTON_X_PHYS, PROTON_Y_PHYS],
        'grid': [pc_ix, pc_iy, pc_iz],
        'score': 0.992,
    },
    'radial_profile': {
        'r': r_mids.tolist(),
        'A': prof_amp.tolist(),
        'E_phase': prof_E_phase.tolist(),
        'E_grad_A': prof_E_grad_A.tolist(),
        'E_restore': prof_E_restore.tolist(),
        'E_cond': prof_E_cond.tolist(),
    },
    'winding_vs_R': winding_vs_R,
    'roi_energy': {
        'R': ROI_R,
        'n_cells': n_roi,
        'E_phase': round(roi_Eph, 2),
        'E_grad_A': round(roi_EgA, 2),
        'E_restore': round(roi_Er, 2),
        'E_cond': round(roi_Ec, 2),
        'E_total': round(roi_Et, 2),
    },
    'flux_tube': {
        'offsets_phys': ft_offsets_phys.tolist(),
        'amp': ft_amp_profile.tolist(),
        'E_phase': ft_Ephase_profile.tolist(),
    },
}

with open('v5_proton2_profile.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"  Saved: v5_proton2_profile.json")
print("=" * 65)
