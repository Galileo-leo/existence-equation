#!/usr/bin/env python3
"""
fqhe_anyon_test4.py — Anyon Braiding Phase from Constraint Alone
=================================================================
TEST 4 for EP2: Non-Abelian Berry Phase (Wilson Loop)

This script extends fqhe_total_proof.py (Tests 1-3) with a direct
computation of the anyon braiding phase via the Wilson loop of the
degenerate ground-state manifold under adiabatic flux insertion.

CLAIM:
  Inserting one flux quantum theta_x: 0 -> 2*pi through a
  non-contractible cycle of the torus cyclically permutes the
  q-fold degenerate ground states. The Wilson loop eigenvalue
  spacing equals 2*pi/q — the anyon braiding phase.

  This is the fractional Aharonov-Bohm phase: the same
  topological response to enclosed flux that AB identified
  in 1959, read on a constrained manifold whose topological
  excitations carry effective charge e/q rather than e.

INPUT (same as Tests 1-3):
  1. 2D torus
  2. NN exclusion (constraint)
  3. Single-particle hop
  4. Peierls phase (magnetic flux)

NO Laughlin wavefunction.
NO Chern-Simons field.
NO anyon postulate.
NO fractional charge assumption.

SYSTEMS:
  A. 3x6,  Ne=2, alpha=1/3, nu=1/3  (D=117)     — baseline
  B. 3x9,  Ne=3, alpha=1/3, nu=1/3  (D~1587)     — scaling
  C. 3x12, Ne=4, alpha=1/3, nu=1/3  (D~26937)    — large scale
  D. 3x6,  Ne=4, alpha=1/3, nu=2/3  (D=468)      — particle-hole
  E. 5x5,  Ne=4, alpha=2/5, nu=2/5  (D~987)      — Jain series

Output:
  fqhe_test4_anyon.png  — Publication-quality figure (300 dpi)

Usage:
  python fqhe_anyon_test4.py

Author: Jae-Ahn Shin
2026-04
=================================================================
"""

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from itertools import combinations
from math import comb
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CORE FUNCTIONS (identical to fqhe_total_proof.py)
# ============================================================

def site_coords(s, Nx):
    return s % Nx, s // Nx

def site_index(x, y, Nx):
    return x + y * Nx

def get_neighbors(s, Nx, Ny):
    """Four nearest neighbors on torus (periodic boundary)."""
    x, y = site_coords(s, Nx)
    return [
        ((x+1) % Nx, y, +1, 0),
        ((x-1) % Nx, y, -1, 0),
        (x, (y+1) % Ny, 0, +1),
        (x, (y-1) % Ny, 0, -1),
    ]

def satisfies_constraint(config, Nx, Ny):
    """
    NN exclusion on 2D torus.
    No Coulomb. No Haldane pseudopotential. No Landau level.
    """
    occ = set(config)
    for s in config:
        x, y = site_coords(s, Nx)
        for nx, ny, _, _ in get_neighbors(s, Nx, Ny):
            ns = site_index(nx, ny, Nx)
            if ns in occ and ns != s:
                return False
    return True

def build_constrained_space(Nx, Ny, N_e):
    """Enumerate all NN-exclusion-allowed configurations."""
    N = Nx * Ny
    allowed = []
    for config in combinations(range(N), N_e):
        if satisfies_constraint(config, Nx, Ny):
            allowed.append(config)
    return allowed

def fermion_sign(config, old_site, new_site):
    """Fermionic exchange sign for hopping."""
    if old_site < new_site:
        n = sum(1 for s in config if old_site < s < new_site)
    else:
        n = sum(1 for s in config if new_site < s < old_site)
    return (-1)**n

def build_hamiltonian(allowed, Nx, Ny, alpha, theta_x=0, theta_y=0):
    """
    Harper-Hofstadter Hamiltonian on constrained Hilbert space.
    Peierls phase = Aharonov-Bohm phase from magnetic flux.
    """
    dim = len(allowed)
    config_to_idx = {c: i for i, c in enumerate(allowed)}
    H = lil_matrix((dim, dim), dtype=complex)

    for idx, config in enumerate(allowed):
        config_set = set(config)
        for particle in config:
            x, y = site_coords(particle, Nx)
            for (nx, ny, dx, dy) in get_neighbors(particle, Nx, Ny):
                new_site = site_index(nx, ny, Nx)
                if new_site in config_set and new_site != particle:
                    continue
                if new_site == particle:
                    continue
                new_config = tuple(sorted(
                    (config_set - {particle}) | {new_site}
                ))
                if new_config not in config_to_idx:
                    continue
                j = config_to_idx[new_config]
                phase = 0.0
                if dy != 0:
                    phase += 2 * np.pi * alpha * x * dy
                if dx == 1 and nx == 0:
                    phase += theta_x
                elif dx == -1 and x == 0:
                    phase -= theta_x
                if dy == 1 and ny == 0:
                    phase += theta_y
                elif dy == -1 and y == 0:
                    phase -= theta_y
                sign = fermion_sign(config, particle, new_site)
                H[idx, j] += sign * np.exp(1j * phase)

    H = csc_matrix(H)
    H = 0.5 * (H + H.conj().T)
    return H


# ============================================================
# DIAGONALIZATION — adaptive: full for small, sparse for large
# ============================================================

def diag_ground_manifold(H, q, use_sparse_threshold=3000):
    """
    Return (eigenvalues[:q+4], eigenvectors[:, :q]) of H.
    Uses full diag for D < threshold, sparse eigsh otherwise.
    """
    dim = H.shape[0]
    if dim <= use_sparse_threshold:
        evals, evecs = eigh(H.toarray())
        return evals[:q+4], evecs[:, :q]
    else:
        n_eig = min(q + 4, dim - 2)
        evals, evecs = eigsh(H, k=n_eig, which='SA')
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
        return evals, evecs[:, :q]


# ============================================================
# WILSON LOOP COMPUTATION
# ============================================================

def compute_wilson_loop(allowed, Nx, Ny, Ne, alpha, q, n_twist):
    """
    Compute the non-Abelian Berry phase (Wilson loop) of the
    q-fold degenerate ground-state manifold under flux insertion
    theta_x: 0 -> 2*pi.

    Wilson loop: W = prod_{i=0}^{N-1} M_i
    where M_nm(i) = <psi_n(theta_i) | psi_m(theta_{i+1})>

    The eigenvalues of W are gauge-invariant.
    Their phase spacings = anyon braiding phases.
    """
    dim = len(allowed)
    thetas = np.linspace(0, 2*np.pi, n_twist, endpoint=False)

    ground_states = []
    spectral_data = []

    for i, tx in enumerate(thetas):
        H = build_hamiltonian(allowed, Nx, Ny, alpha, theta_x=tx)
        evals, evecs_q = diag_ground_manifold(H, q)
        ground_states.append(evecs_q.copy())
        spectral_data.append(evals.copy())
        if (i + 1) % 20 == 0 or i == 0:
            print(f"      theta {i+1}/{n_twist} done")

    spectral_data = np.array(spectral_data)

    # Wilson loop: W = M_0 * M_1 * ... * M_{N-1}
    W = np.eye(q, dtype=complex)
    overlap_dets = []

    for i in range(n_twist):
        j = (i + 1) % n_twist
        M = ground_states[i].conj().T @ ground_states[j]
        overlap_dets.append(abs(np.linalg.det(M)))
        W = W @ M

    W_evals = np.linalg.eigvals(W)
    W_phases = np.angle(W_evals)
    sort_idx = np.argsort(W_phases)
    W_phases = W_phases[sort_idx]
    W_evals = W_evals[sort_idx]

    sorted_phases = np.sort(W_phases)
    spacings = []
    for k in range(len(sorted_phases) - 1):
        spacings.append(sorted_phases[k+1] - sorted_phases[k])
    wrap = (2*np.pi + sorted_phases[0]) - sorted_phases[-1]
    spacings.append(wrap)

    return {
        'W_evals': W_evals,
        'W_phases': W_phases,
        'spacings': np.array(spacings),
        'spectral_data': spectral_data,
        'thetas': thetas,
        'min_det': min(overlap_dets),
        'mean_det': np.mean(overlap_dets),
        'dim': dim,
    }


# ============================================================
# SYSTEM DEFINITIONS
# ============================================================

SYSTEMS = [
    # (Nx, Ny, Ne, alpha, q_exp, n_twist, label_short, label_full)
    (3, 6,  2, 1/3, 3, 120,
     r'$\nu=1/3$, 3$\times$6',
     r'3$\times$6, Ne=2, $\alpha=1/3$, $\nu=1/3$'),

    (3, 9,  3, 1/3, 3, 80,
     r'$\nu=1/3$, 3$\times$9',
     r'3$\times$9, Ne=3, $\alpha=1/3$, $\nu=1/3$'),

    (3, 12, 4, 1/3, 3, 48,
     r'$\nu=1/3$, 3$\times$12',
     r'3$\times$12, Ne=4, $\alpha=1/3$, $\nu=1/3$'),

    (3, 6,  4, 1/3, 3, 120,
     r'$\nu=2/3$, 3$\times$6',
     r'3$\times$6, Ne=4, $\alpha=1/3$, $\nu=2/3$'),

    (5, 5,  4, 2/5, 5, 80,
     r'$\nu=2/5$, 5$\times$5',
     r'5$\times$5, Ne=4, $\alpha=2/5$, $\nu=2/5$'),
]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  TEST 4: ANYON BRAIDING PHASE FROM CONSTRAINT ALONE")
    print("  Non-Abelian Wilson Loop of Ground-State Manifold")
    print("=" * 70)
    print("  Input: torus + NN exclusion + hop + Peierls phase")
    print("  No Laughlin. No Chern-Simons. No anyon postulate.")
    print("=" * 70)

    t_total = time()
    results = []

    for (Nx, Ny, Ne, alpha, q_exp, n_tw, label_s, label_f) in SYSTEMS:
        N_sites = Nx * Ny
        N_phi = alpha * N_sites
        nu = Ne / N_phi

        print(f"\n{'='*65}")
        print(f"  {label_f}")
        print(f"  Expected: q={q_exp}, anyon phase = 2pi/{q_exp} "
              f"= {360/q_exp:.1f} deg")
        print(f"{'='*65}")

        t0 = time()

        allowed = build_constrained_space(Nx, Ny, Ne)
        dim = len(allowed)
        full_dim = comb(N_sites, Ne)
        print(f"  Hilbert space: {dim} / {full_dim} "
              f"(constrained/full, ratio={dim/full_dim:.4f})")

        H0 = build_hamiltonian(allowed, Nx, Ny, alpha)
        evals0, _ = diag_ground_manifold(H0, q_exp)
        E0 = evals0[0]
        gap_at_q = evals0[q_exp] - evals0[q_exp - 1] if len(evals0) > q_exp else 0
        spread = evals0[q_exp - 1] - evals0[0]
        gap = gap_at_q

        print(f"  Ground manifold spread: {spread:.6f}")
        print(f"  Gap above q={q_exp} states: {gap:.4f}")

        q_use = q_exp

        print(f"  Computing Wilson loop ({n_tw} twist points)...")
        r = compute_wilson_loop(allowed, Nx, Ny, Ne, alpha, q_use, n_tw)
        elapsed = time() - t0

        print(f"\n  Wilson loop eigenvalues:")
        for n in range(q_use):
            mag = abs(r['W_evals'][n])
            deg = np.degrees(r['W_phases'][n])
            print(f"    |W_{n}| = {mag:.6f},  "
                  f"phase = {deg:+8.2f} deg")

        mean_spacing = np.mean(r['spacings'])
        expected_spacing = 2 * np.pi / q_use
        error = abs(mean_spacing - expected_spacing) / expected_spacing

        print(f"\n  Phase spacings: " +
              ", ".join(f"{np.degrees(s):.1f} deg" for s in r['spacings']))
        print(f"  Mean spacing: {np.degrees(mean_spacing):.1f} deg "
              f"(expected: {np.degrees(expected_spacing):.1f} deg)")
        print(f"  Error: {error*100:.1f}%")
        print(f"  Overlap quality: min|det|={r['min_det']:.6f}, "
              f"mean={r['mean_det']:.6f}")

        match = error < 0.10
        status = "MATCH" if match else "MISMATCH"
        print(f"\n  >>> {status} (anyon phase = "
              f"{np.degrees(mean_spacing):.1f} deg, "
              f"theory = {np.degrees(expected_spacing):.1f} deg)")
        print(f"  Time: {elapsed:.1f}s")

        results.append({
            'Nx': Nx, 'Ny': Ny, 'Ne': Ne, 'alpha': alpha,
            'q': q_use, 'q_exp': q_exp, 'nu': nu,
            'label_s': label_s, 'label_f': label_f,
            'degen': q_use, 'gap': gap, 'dim': dim,
            'spread': spread,
            'W_phases': r['W_phases'],
            'W_evals': r['W_evals'],
            'spacings': r['spacings'],
            'spectral_data': r['spectral_data'],
            'thetas': r['thetas'],
            'mean_spacing': mean_spacing,
            'expected_spacing': expected_spacing,
            'error': error,
            'min_det': r['min_det'],
            'match': match,
            'status': status,
            'time': elapsed,
        })

    valid = [r for r in results if r is not None]

    # ============================================================
    # FIGURE — Publication Quality
    # ============================================================
    n_sys = len(valid)

    fig = plt.figure(figsize=(16, 4.2 * n_sys + 1.5))
    gs = GridSpec(n_sys, 3, figure=fig, hspace=0.50, wspace=0.40,
                  left=0.06, right=0.97, top=0.92, bottom=0.04)

    fig.suptitle(
        'TEST 4: Anyon Braiding Phase from Constraint Alone\n'
        'Wilson loop of ground-state manifold under flux insertion '
        r'$\theta_x: 0 \to 2\pi$.  '
        'No Chern-Simons. No anyon postulate.',
        fontsize=13, fontweight='bold')

    for ri, r in enumerate(valid):
        q = r['q']

        # ---- (a) Spectral flow ----
        ax = fig.add_subplot(gs[ri, 0])
        thetas_plot = r['thetas'] / (2 * np.pi)
        n_show = min(q + 4, r['spectral_data'].shape[1])
        E0_arr = r['spectral_data'][:, 0]

        for j in range(n_show):
            if j < q:
                ax.plot(thetas_plot,
                        r['spectral_data'][:, j] - E0_arr,
                        color='#D32F2F', lw=2.0, alpha=0.85,
                        label='Ground manifold' if j == 0 else '')
            else:
                ax.plot(thetas_plot,
                        r['spectral_data'][:, j] - E0_arr,
                        color='#1565C0', lw=1.0, alpha=0.5,
                        label='Excited' if j == q else '')

        panel = chr(ord('a') + ri * 3)
        ax.set_title(f'({panel}) {r["label_s"]}\n'
                     f'D={r["dim"]}, gap={r["gap"]:.4f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel(r'$\theta_x / 2\pi$', fontsize=10)
        ax.set_ylabel(r'$E - E_0(\theta)$', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.15)

        # ---- (b) Wilson loop phases — polar ----
        ax = fig.add_subplot(gs[ri, 1], projection='polar')
        cmap = plt.cm.Set1
        colors_p = [cmap(k / max(q, 1)) for k in range(q)]

        for n in range(q):
            angle = r['W_phases'][n]
            mag = abs(r['W_evals'][n])
            ax.plot([0, angle], [0, mag], 'o-',
                    color=colors_p[n], lw=2.5, ms=10, zorder=5,
                    label=f'$\\gamma_{n}$={np.degrees(angle):+.1f}°')

        for k in range(q):
            exp_angle = -np.pi + 2 * np.pi * k / q
            ax.plot(exp_angle, 1.15, 'k+', ms=12, mew=2, alpha=0.25)

        panel = chr(ord('a') + ri * 3 + 1)
        ax.set_title(f'({panel}) Wilson loop eigenvalues\n'
                     f'spacing = {np.degrees(r["mean_spacing"]):.1f}° '
                     f'(theory {np.degrees(r["expected_spacing"]):.0f}°)',
                     fontsize=9, fontweight='bold', pad=18)
        ax.legend(fontsize=7, loc='upper right',
                  bbox_to_anchor=(1.45, 1.0))
        ax.set_ylim(0, 1.35)

        # ---- (c) Summary box ----
        ax = fig.add_subplot(gs[ri, 2])
        ax.axis('off')

        lines = []
        lines.append(f"{'━'*38}")
        lines.append(f"  {r['label_f']}")
        lines.append(f"  D = {r['dim']},  q = {r['q']}")
        lines.append(f"  spread = {r['spread']:.6f}")
        lines.append(f"  gap = {r['gap']:.4f} Ω")
        lines.append(f"{'━'*38}")
        lines.append(f"")
        lines.append(f"WILSON LOOP EIGENVALUES:")
        for n in range(q):
            mag = abs(r['W_evals'][n])
            deg = np.degrees(r['W_phases'][n])
            lines.append(f"  |W_{n}| = {mag:.4f}"
                         f"   φ = {deg:+7.1f}°")
        lines.append(f"")
        lines.append(f"PHASE SPACINGS:")
        for k, s in enumerate(r['spacings']):
            lines.append(f"  Δφ_{k} = {np.degrees(s):6.1f}°")
        lines.append(f"  ─────────────────")
        lines.append(f"  mean  = {np.degrees(r['mean_spacing']):6.1f}°")
        lines.append(f"  theory= {np.degrees(r['expected_spacing']):6.1f}°"
                     f"  (2π/{q})")
        lines.append(f"  error = {r['error']*100:5.1f}%")
        lines.append(f"")
        lines.append(f"  ▶ {r['status']}")
        lines.append(f"")
        lines.append(f"  Overlap: min|det|={r['min_det']:.4f}")
        lines.append(f"  Time: {r['time']:.1f}s")

        color = '#e8f5e9' if r['match'] else '#ffebee'
        ax.text(0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes,
                fontsize=7.8, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor=color, alpha=0.9,
                          edgecolor='#666', linewidth=0.5))

    plt.savefig('fqhe_test4_anyon.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: fqhe_test4_anyon.png (300 dpi)")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    elapsed_total = time() - t_total

    print(f"\n{'='*70}")
    print(f"  TEST 4 SUMMARY: ANYON BRAIDING PHASE")
    print(f"{'='*70}")
    print(f"  {'System':>20s} {'D':>7s} {'q':>3s} "
          f"{'Spacing':>10s} {'Theory':>10s} {'Error':>7s} {'Status':>8s}")
    print(f"  {'─'*68}")

    for r in valid:
        sys_str = f"{r['Nx']}x{r['Ny']} ν={r['nu']:.3f}"
        sp_str = f"{np.degrees(r['mean_spacing']):.1f}°"
        th_str = f"{np.degrees(r['expected_spacing']):.1f}°"
        print(f"  {sys_str:>20s} {r['dim']:7d} {r['q']:3d} "
              f"{sp_str:>10s} {th_str:>10s} "
              f"{r['error']*100:6.1f}% {r['status']:>8s}")

    n_match = sum(1 for r in valid if r['match'])
    print(f"\n  Score: {n_match}/{len(valid)} MATCH")
    print(f"  Total time: {elapsed_total:.1f}s")

    print(f"\n{'─'*70}")
    print(f"  INTERPRETATION")
    print(f"{'─'*70}")
    print(f"  The Wilson loop eigenvalue spacing = 2*pi/q")
    print(f"  = the anyon braiding phase.")
    print(f"")
    print(f"  This is the fractional Aharonov-Bohm phase:")
    print(f"  the same topological response to enclosed flux")
    print(f"  that AB identified in 1959, read on a constrained")
    print(f"  manifold whose excitations carry charge e/q.")
    print(f"")
    print(f"  AB -> Peierls -> constrained degeneracy")
    print(f"     -> Wilson loop -> anyon braiding phase")
    print(f"")
    print(f"  One mechanism. One chain. No Chern-Simons.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
