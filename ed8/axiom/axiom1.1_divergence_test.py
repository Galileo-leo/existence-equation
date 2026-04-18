"""
ED Discreteness Test — 2D
=========================
On the same physical domain (L=20), vary only the lattice resolution (N)
and evolve the ED equation, confirming that discreteness (finite dx)
is responsible for preventing divergence.

PDE: (1/c^2) Psi_tt = Lap(Psi) + lambda*Psi - alpha*|Psi|^2*Psi

Experiment design:
  - N = [64, 128, 256, 512, 1024] (same L=20 -> dx decreases)
  - Identical initial perturbation (same physical blob position/size/amplitude)
  - Free evolution after injection at each resolution
  - Measure total energy, max|Psi|, high-frequency mode energy ratio per step

Expectations:
  - With lattice (dx > 0): energy finite, max|Psi| finite -> stable
  - dx -> 0 (N -> inf): energy cascade to high-freq modes -> energy/amplitude divergence

Run on Colab: uses cupy (GPU) if available, otherwise numpy (CPU)

Jae-Ahn Shin & Kael, 2026-04
"""

import time as timer
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
    xp = cp
    USE_GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}")
except ImportError:
    xp = np
    USE_GPU = False
    print("CPU mode")


def to_np(arr):
    return cp.asnumpy(arr) if USE_GPU else arr


# ══════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════

c     = 1.0
lam   = 1.0
alpha = 1.0
v_vac = np.sqrt(lam / alpha)  # = 1.0

L     = 20.0      # fixed physical domain size

N_LIST = [64, 128, 256, 512, 1024, 2048, 4096]

# Injection settings
N_BLOBS       = 15
SIGMA_RANGE   = (1.0, 3.0)   # physical blob size
AMP_RANGE     = (1.5, 3.0)   # blob amplitude
SEED          = 42

# Time settings
T_INJECT      = 30.0   # injection duration
T_FREE        = 50.0   # free evolution after injection
T_TOTAL       = T_INJECT + T_FREE
INJECT_INTERVAL_TIME = 2.0  # physical time interval between injections

# Measurement interval (physical time)
MEASURE_DT    = 1.0

# Print interval (physical time)
PRINT_EVERY   = 5.0


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

def laplacian_2d(f, dx):
    """5-point stencil Laplacian, periodic BC."""
    return (xp.roll(f, 1, 0) + xp.roll(f, -1, 0) +
            xp.roll(f, 1, 1) + xp.roll(f, -1, 1) - 4*f) / dx**2


def generate_perturbation_2d(N, dx, blobs, blob_seed_offset=0):
    """
    Physical-coordinate-based perturbation generation.
    blobs: list of (x0, y0, sigma, amp, phi) in physical units
    """
    coords = xp.arange(N, dtype=np.float64) * dx - L/2
    Y, X = xp.meshgrid(coords, coords, indexing='ij')
    pert = xp.zeros((N, N), dtype=complex)
    for (x0, y0, sig, amp, phi) in blobs:
        pert += amp * xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2*sig**2)) * np.exp(1j*phi)
    return pert


def make_blob_list(n_blobs, seed):
    """Generate blob list in physical coordinates (resolution-independent)."""
    rng = np.random.RandomState(seed)
    blobs = []
    for _ in range(n_blobs):
        x0  = rng.uniform(-L/2, L/2)
        y0  = rng.uniform(-L/2, L/2)
        sig = rng.uniform(*SIGMA_RANGE)
        amp = rng.uniform(*AMP_RANGE)
        phi = rng.uniform(0, 2*np.pi)
        blobs.append((x0, y0, sig, amp, phi))
    return blobs


def compute_energy_2d(psi, psi_dot, dx):
    """Total energy density integral (2D)."""
    dV = dx**2

    # kinetic: (1/2c^2)|Psi_t|^2
    E_kin = float(xp.sum(0.5 / c**2 * xp.abs(psi_dot)**2).item()) * dV

    # gradient: (1/2)|grad Psi|^2
    gx = (xp.roll(psi, -1, 1) - xp.roll(psi, 1, 1)) / (2*dx)
    gy = (xp.roll(psi, -1, 0) - xp.roll(psi, 1, 0)) / (2*dx)
    E_grad = float(xp.sum(0.5 * (xp.abs(gx)**2 + xp.abs(gy)**2)).item()) * dV

    # potential: -(lambda/2)|Psi|^2 + (alpha/4)|Psi|^4  (Mexican hat)
    A2 = xp.abs(psi)**2
    E_pot = float(xp.sum(-0.5*lam*A2 + 0.25*alpha*A2**2).item()) * dV

    E_total = E_kin + E_grad + E_pot
    return E_total, E_kin, E_grad, E_pot


def compute_highfreq_ratio(psi, dx):
    """
    High-frequency mode energy ratio: power fraction of FFT modes
    with |k| > k_max/2. Discrete lattice Nyquist = pi/dx.
    Half-Nyquist = pi/(2*dx).
    """
    N = psi.shape[0]
    psi_np = to_np(psi)
    fft2 = np.fft.fft2(psi_np)
    power = np.abs(fft2)**2

    kx = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    k_nyquist = np.pi / dx
    k_half = k_nyquist / 2

    total_power = np.sum(power)
    high_power  = np.sum(power[K > k_half])

    ratio = high_power / max(total_power, 1e-30)
    return float(ratio)


# ══════════════════════════════════════════════════════════════
# Pre-generate blob lists (resolution-independent, physical coords)
# ══════════════════════════════════════════════════════════════

n_injections = int(T_INJECT / INJECT_INTERVAL_TIME)
all_blob_lists = []
for i in range(n_injections):
    all_blob_lists.append(make_blob_list(N_BLOBS, seed=SEED + i*100))

print(f"\n{'='*70}")
print(f"  ED Discreteness Test — 2D")
print(f"  PDE: (1/c^2) Psi_tt = Lap(Psi) + lambda*Psi - alpha*|Psi|^2*Psi")
print(f"{'='*70}")
print(f"  L={L}, c={c}, lambda={lam}, alpha={alpha}, v_vac={v_vac:.4f}")
print(f"  N_LIST = {N_LIST}")
print(f"  T_inject={T_INJECT}, T_free={T_FREE}, T_total={T_TOTAL}")
print(f"  Blobs per injection: {N_BLOBS}, injections: {n_injections}")
print(f"  GPU: {'YES' if USE_GPU else 'NO'}")
print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════
# Main loop: simulation at each resolution
# ══════════════════════════════════════════════════════════════

results = {}  # N -> {times, energies, max_amps, hf_ratios}

for N in N_LIST:
    dx = L / N
    dt = 0.25 * dx / (c * np.sqrt(2))  # CFL for 2D
    n_steps = int(T_TOTAL / dt)
    inject_interval_steps = max(1, int(INJECT_INTERVAL_TIME / dt))
    measure_interval_steps = max(1, int(MEASURE_DT / dt))
    print_interval_steps = max(1, int(PRINT_EVERY / dt))
    n_inject_steps = int(T_INJECT / dt)

    print(f"\n{'─'*60}")
    print(f"  N = {N},  dx = {dx:.6f},  dt = {dt:.6f}")
    print(f"  Steps: {n_steps},  inject_steps: {n_inject_steps}")
    print(f"  Nyquist k = pi/dx = {np.pi/dx:.2f}")
    print(f"{'─'*60}")

    # Initial condition: vacuum + small noise
    np.random.seed(SEED)
    psi = v_vac * xp.ones((N, N), dtype=complex)
    psi += 0.01 * v_vac * (xp.asarray(np.random.randn(N, N).astype(np.float64)) +
                            1j * xp.asarray(np.random.randn(N, N).astype(np.float64)))
    psi_dot = xp.zeros_like(psi)

    times_list    = []
    energy_list   = []
    max_amp_list  = []
    hf_ratio_list = []

    injection_idx = 0
    wall_start = timer.time()

    for step in range(n_steps):
        t = step * dt

        # Injection
        if step < n_inject_steps and step % inject_interval_steps == 0:
            if injection_idx < len(all_blob_lists):
                pert = generate_perturbation_2d(N, dx, all_blob_lists[injection_idx])
                psi += pert
                injection_idx += 1

        # Leapfrog integration
        lap = laplacian_2d(psi, dx)
        accel = c**2 * (lap + lam*psi - alpha*xp.abs(psi)**2 * psi)
        psi_dot += accel * dt
        psi += psi_dot * dt

        # Measurement
        if step % measure_interval_steps == 0:
            E_total, E_kin, E_grad, E_pot = compute_energy_2d(psi, psi_dot, dx)
            max_amp = float(xp.max(xp.abs(psi)).item())
            hf_ratio = compute_highfreq_ratio(psi, dx)

            times_list.append(t)
            energy_list.append(E_total)
            max_amp_list.append(max_amp)
            hf_ratio_list.append(hf_ratio)

        # Print
        if step % print_interval_steps == 0:
            E_total, E_kin, E_grad, E_pot = compute_energy_2d(psi, psi_dot, dx)
            max_amp = float(xp.max(xp.abs(psi)).item())
            elapsed = timer.time() - wall_start
            phase_str = "INJECT" if step < n_inject_steps else "FREE  "
            print(f"  [{phase_str}] t={t:6.1f} | E={E_total:>12.1f} | "
                  f"max|Psi|={max_amp:>8.3f} | {elapsed:.0f}s")

            # Divergence detection -> stop immediately
            if np.isnan(E_total) or np.isinf(E_total) or max_amp > 1e6:
                print(f"\n  *** DIVERGENCE DETECTED at t={t:.2f} ***")
                print(f"      E_total = {E_total}")
                print(f"      max|Psi| = {max_amp}")
                times_list.append(t)
                energy_list.append(E_total)
                max_amp_list.append(max_amp)
                hf_ratio_list.append(1.0)
                break

    elapsed_total = timer.time() - wall_start
    print(f"  Done: N={N}, {elapsed_total:.0f}s")

    results[N] = {
        'times':     np.array(times_list),
        'energies':  np.array(energy_list),
        'max_amps':  np.array(max_amp_list),
        'hf_ratios': np.array(hf_ratio_list),
        'dx': dx,
        'dt': dt,
    }


# ══════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ED Discreteness Test: Resolution Dependence (2D)\n'
             r'PDE: $(1/c^2)\Psi_{tt} = \nabla^2\Psi + \lambda\Psi - \alpha|\Psi|^2\Psi$',
             fontsize=13)

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_LIST)))

# (1) Total Energy vs Time
ax = axes[0, 0]
for i, N in enumerate(N_LIST):
    r = results[N]
    ax.plot(r['times'], r['energies'], color=colors[i],
            label=f'N={N} (dx={r["dx"]:.4f})', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Total Energy')
ax.set_title('(a) Total Energy vs Time')
ax.legend(fontsize=8)
ax.axvline(T_INJECT, color='gray', linestyle='--', alpha=0.5, label='inject end')
ax.grid(True, alpha=0.3)

# (2) max|Psi| vs Time
ax = axes[0, 1]
for i, N in enumerate(N_LIST):
    r = results[N]
    ax.plot(r['times'], r['max_amps'], color=colors[i],
            label=f'N={N}', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('max |Psi|')
ax.set_title('(b) Maximum Amplitude vs Time')
ax.legend(fontsize=8)
ax.axvline(T_INJECT, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# (3) High-frequency mode ratio vs Time
ax = axes[1, 0]
for i, N in enumerate(N_LIST):
    r = results[N]
    ax.plot(r['times'], r['hf_ratios'], color=colors[i],
            label=f'N={N}', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('High-freq Power Ratio (|k| > k_Nyq/2)')
ax.set_title('(c) UV Mode Energy Fraction vs Time')
ax.legend(fontsize=8)
ax.axvline(T_INJECT, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# (4) Final Energy vs 1/dx (= N/L)
ax = axes[1, 1]
final_energies = []
final_max_amps = []
final_hf = []
dxs = []
for N in N_LIST:
    r = results[N]
    final_energies.append(r['energies'][-1])
    final_max_amps.append(r['max_amps'][-1])
    final_hf.append(r['hf_ratios'][-1])
    dxs.append(r['dx'])

inv_dx = [1.0/d for d in dxs]
ax.plot(inv_dx, final_energies, 'o-', color='red', linewidth=2, markersize=8,
        label='Final E_total')
ax.set_xlabel('1/dx  (= N/L, resolution)')
ax.set_ylabel('Final Total Energy')
ax.set_title('(d) Final Energy vs Resolution')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Power law trend line (log-log)
if len(inv_dx) >= 3:
    log_x = np.log(inv_dx)
    log_y = np.log(np.abs(final_energies))
    valid = np.isfinite(log_y)
    if np.sum(valid) >= 2:
        coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
        ax.text(0.05, 0.95, f'Power law: E ~ (1/dx)^{coeffs[0]:.2f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('ed_discreteness_test_2d.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Figure saved: ed_discreteness_test_2d.png")


# ══════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  SUMMARY TABLE")
print(f"{'='*70}")
print(f"  {'N':>6s}  {'dx':>10s}  {'dt':>10s}  {'E_final':>14s}  "
      f"{'max|Psi|':>10s}  {'HF ratio':>10s}  {'Diverged?':>10s}")
print(f"  {'-'*76}")

for N in N_LIST:
    r = results[N]
    E_f = r['energies'][-1]
    ma_f = r['max_amps'][-1]
    hf_f = r['hf_ratios'][-1]
    diverged = "YES" if (np.isnan(E_f) or np.isinf(E_f) or ma_f > 1e6) else "no"
    print(f"  {N:>6d}  {r['dx']:>10.6f}  {r['dt']:>10.6f}  {E_f:>14.2f}  "
          f"{ma_f:>10.3f}  {hf_f:>10.4f}  {diverged:>10s}")

print(f"\n{'='*70}")
print(f"  Interpretation:")
print(f"  - E_final increasing sharply with N -> discreteness (finite dx) keeps energy finite")
print(f"  - HF ratio increasing with N -> energy cascade to UV modes (UV problem)")
print(f"  - Diverged = YES -> divergence confirmed in continuum limit")
print(f"{'='*70}")
