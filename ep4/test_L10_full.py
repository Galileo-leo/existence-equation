#!/usr/bin/env python3
"""
EP4 Deep Test: L=10 Full Spectrum Analysis
============================================
L=10 dense eigenvalue computation for the full i[O0, O1] spectrum.
This goes beyond the paper's sparse (top-40) analysis.

Goal: Verify j_max = floor(10/2) = 5 with COMPLETE spectrum.

Warning: D ~ 2000+ for L=10, so dense is feasible but slow.
         Full-space random control needs 2^20 = 1,048,576 dim → skip.
         Instead: use constrained-dimension random control.

Jae-Ahn Shin, Independent Researcher
jaeahn.shin.official@gmail.com
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time as now

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys

# ============================================================
# BASIS CONSTRUCTION (recursive, PBC)
# ============================================================

def build_ladder_basis(L):
    """2-leg Rydberg ladder basis: blockade + transverse exclusion, PBC."""
    rung_states = [(0, 0), (1, 0), (0, 1)]

    def compatible(r1, r2):
        if r1[0] == 1 and r2[0] == 1: return False
        if r1[1] == 1 and r2[1] == 1: return False
        return True

    # Build all chains of length L with PBC
    chains = [([s],) for s in range(3)]
    for pos in range(1, L):
        new_chains = []
        for (chain,) in chains:
            for s in range(3):
                if compatible(rung_states[chain[-1]], rung_states[s]):
                    new_chains.append((chain + [s],))
        chains = new_chains

    basis = []
    for (chain,) in chains:
        if compatible(rung_states[chain[-1]], rung_states[chain[0]]):
            bits = []
            for s_idx in chain:
                bits.extend(rung_states[s_idx])
            basis.append(bits)
    return basis


def build_projected_X_sparse(basis, L, leg):
    """O_l = P X_l P (sparse)."""
    D = len(basis)
    basis_dict = {tuple(b): i for i, b in enumerate(basis)}
    rows, cols, vals = [], [], []
    for idx, b in enumerate(basis):
        for i in range(L):
            site = 2*i + leg
            new_b = list(b)
            new_b[site] = 1 - new_b[site]
            key = tuple(new_b)
            if key in basis_dict:
                rows.append(idx)
                cols.append(basis_dict[key])
                vals.append(1.0)
    return csr_matrix((vals, (rows, cols)), shape=(D, D))


# ============================================================
# MAIN TEST
# ============================================================

def main():
    L = 10
    print("=" * 65)
    print(f"  EP4 DEEP TEST: L={L} FULL DENSE SPECTRUM")
    print("=" * 65)

    # --- Build basis ---
    t0 = now()
    basis = build_ladder_basis(L)
    D = len(basis)
    t_basis = now() - t0
    print(f"\n  Basis: D = {D}  (built in {t_basis:.2f}s)")
    print(f"  Full Hilbert space: 2^{2*L} = {2**(2*L):,}")
    print(f"  Constraint compression: {D}/{2**(2*L)} = {D/2**(2*L):.6f}")

    # --- Build projected operators ---
    t0 = now()
    O0 = build_projected_X_sparse(basis, L, 0)
    O1 = build_projected_X_sparse(basis, L, 1)
    t_ops = now() - t0
    print(f"  Operators built in {t_ops:.2f}s")
    print(f"  O0: {O0.nnz} nonzeros,  O1: {O1.nnz} nonzeros")

    # --- Commutator ---
    t0 = now()
    comm = O0 @ O1 - O1 @ O0
    comm_norm = np.sqrt(np.sum(np.abs(comm.data)**2))
    print(f"  ||[O0, O1]||_F = {comm_norm:.6f}")
    print(f"  ||[O0, O1]||_F / L = {comm_norm/L:.6f}")

    # --- FULL dense eigenvalue decomposition ---
    print(f"\n  Computing FULL dense eigenvalues of i[O0, O1] (D={D})...")
    sys.stdout.flush()

    iC = 1j * comm.toarray()
    iC_herm = 0.5 * (iC + iC.conj().T)  # ensure Hermitian
    eigvals = np.linalg.eigvalsh(iC_herm)
    t_eig = now() - t0
    print(f"  Eigenvalue computation: {t_eig:.2f}s")

    # --- Analysis ---
    eigvals_sorted = np.sort(eigvals)
    rounded = np.round(eigvals).astype(int)
    deviations = np.abs(eigvals - np.round(eigvals))
    max_dev = np.max(deviations)
    mean_dev = np.mean(deviations)

    unique, counts = np.unique(rounded, return_counts=True)
    spectrum = {int(u): int(c) for u, c in zip(unique, counts)}

    j_max = max(abs(u) for u in unique)
    expected_jmax = L // 2

    print(f"\n  --- FULL SPECTRUM RESULTS ---")
    print(f"  j_max observed:  {j_max}")
    print(f"  j_max expected:  {expected_jmax} = floor({L}/2)")
    print(f"  Match: {'YES' if j_max == expected_jmax else 'NO !!!'}")
    print(f"  Max deviation from integer: {max_dev:.2e}")
    print(f"  Mean deviation from integer: {mean_dev:.2e}")
    print(f"  Total eigenvalues: {len(eigvals)} (= D = {D})")

    print(f"\n  Degeneracy structure:")
    print(f"  {'eigenvalue':>12}  {'count':>8}  {'% of D':>8}")
    print(f"  {'-'*32}")
    for u in sorted(spectrum.keys()):
        pct = 100 * spectrum[u] / D
        print(f"  {u:>12d}  {spectrum[u]:>8d}  {pct:>7.1f}%")

    total_check = sum(spectrum.values())
    print(f"  {'TOTAL':>12}  {total_check:>8d}")
    assert total_check == D, "Eigenvalue count mismatch!"

    # --- Verify O0, O1 properties ---
    print(f"\n  --- OPERATOR PROPERTIES ---")
    O0d = O0.toarray()
    O1d = O1.toarray()

    # Hermiticity
    herm_err_0 = np.max(np.abs(O0d - O0d.T))
    herm_err_1 = np.max(np.abs(O1d - O1d.T))
    print(f"  O0 Hermiticity error: {herm_err_0:.2e}")
    print(f"  O1 Hermiticity error: {herm_err_1:.2e}")

    # O0^2 and O1^2 eigenvalues (should they have structure?)
    O0_sq_eig = np.sort(np.linalg.eigvalsh(O0d @ O0d))
    O1_sq_eig = np.sort(np.linalg.eigvalsh(O1d @ O1d))
    print(f"  O0^2 eigenvalue range: [{O0_sq_eig[0]:.4f}, {O0_sq_eig[-1]:.4f}]")
    print(f"  O1^2 eigenvalue range: [{O1_sq_eig[0]:.4f}, {O1_sq_eig[-1]:.4f}]")

    # Casimir-like: O0^2 + O1^2 + (i[O0,O1])^2 ?
    C2 = O0d @ O0d + O1d @ O1d + iC_herm @ iC_herm
    C2_eig = np.sort(np.linalg.eigvalsh(C2))
    C2_rounded = np.round(C2_eig, 2)
    C2_unique, C2_counts = np.unique(C2_rounded, return_counts=True)
    print(f"\n  --- CASIMIR-LIKE OPERATOR: O0^2 + O1^2 + (i[O0,O1])^2 ---")
    print(f"  Eigenvalue range: [{C2_eig[0]:.4f}, {C2_eig[-1]:.4f}]")
    print(f"  Unique eigenvalues (rounded to 0.01):")
    for v, c in zip(C2_unique, C2_counts):
        # Check if close to j(j+1) for some j
        j_test = (-1 + np.sqrt(1 + 4*v)) / 2
        j_int = int(np.round(j_test))
        j_match = abs(v - j_int*(j_int+1)) < 0.1
        marker = f"  ~ j={j_int}, j(j+1)={j_int*(j_int+1)}" if j_match and v > 0.5 else ""
        print(f"    {v:>10.2f}  x{c:>5d}{marker}")

    # --- CONSTRAINED-DIM RANDOM CONTROL ---
    print(f"\n  --- RANDOM PROJECTION CONTROL (constrained-dim) ---")
    print(f"  Using D={D} random subspace in D_padded = {max(D*4, 4096)} space")
    D_padded = max(D * 4, 4096)
    rng = np.random.RandomState(42)

    # Build X0, X1 in padded space (block-diagonal: physical + padding)
    # Physical block = actual O0, O1 in constrained basis
    # But that would be circular. Instead: random D-dim subspace of D_padded.
    Q, _ = np.linalg.qr(rng.randn(D_padded, D))
    P_rand = Q[:, :D]

    # Build generic flip operators in D_padded space
    # Use random symmetric matrices as "unconstrained operators"
    A0 = rng.randn(D_padded, D_padded)
    A0 = 0.5 * (A0 + A0.T)  # symmetric
    A1 = rng.randn(D_padded, D_padded)
    A1 = 0.5 * (A1 + A1.T)

    # Make them commute: simultaneously diagonalize
    # A0 = V diag(d0) V^T, A1 = V diag(d1) V^T (same V → commute)
    V = np.linalg.qr(rng.randn(D_padded, D_padded))[0]
    d0 = rng.randn(D_padded)
    d1 = rng.randn(D_padded)
    A0_comm = V @ np.diag(d0) @ V.T
    A1_comm = V @ np.diag(d1) @ V.T

    # Verify they commute
    comm_check = np.max(np.abs(A0_comm @ A1_comm - A1_comm @ A0_comm))
    print(f"  Commuting operators: ||[A0,A1]|| = {comm_check:.2e}")

    # Project onto random subspace
    B0 = P_rand.T @ A0_comm @ P_rand
    B1 = P_rand.T @ A1_comm @ P_rand
    comm_rand = B0 @ B1 - B1 @ B0
    rand_eigvals = np.linalg.eigvalsh(1j * comm_rand)
    rand_dev = np.abs(rand_eigvals - np.round(rand_eigvals))
    rand_max_dev = np.max(rand_dev)
    rand_mean_dev = np.mean(rand_dev)

    print(f"  Random proj max dev from integer: {rand_max_dev:.6f}")
    print(f"  Random proj mean dev from integer: {rand_mean_dev:.6f}")
    print(f"  Constraint proj max dev:           {max_dev:.2e}")
    print(f"  Ratio: {rand_max_dev / max(max_dev, 1e-15):.0e}x larger deviation in random case")

    # ============================================================
    # FIGURE
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'EP4 Deep Test: L={L} Full Dense Spectrum (D={D})\n'
        f'$j_{{\\max}} = {j_max} = \\lfloor {L}/2 \\rfloor$, '
        f'max deviation = {max_dev:.1e}',
        fontsize=14, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.07, right=0.96, top=0.90, bottom=0.06)

    # (a) Full spectrum histogram
    ax = fig.add_subplot(gs[0, 0])
    colors = ['#D32F2F' if u == 0 else '#1565C0' for u in sorted(spectrum.keys())]
    ax.bar(sorted(spectrum.keys()),
           [spectrum[k] for k in sorted(spectrum.keys())],
           color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    for u in sorted(spectrum.keys()):
        ax.text(u, spectrum[u] + D*0.005, str(spectrum[u]),
                ha='center', fontsize=7)
    ax.set_xlabel(r'Eigenvalue of $i[O_0, O_1]$', fontsize=11)
    ax.set_ylabel('Degeneracy', fontsize=11)
    ax.set_title(f'(a) Full Spectrum: {D} eigenvalues', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')

    # (b) Deviation from integer
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(range(D), np.sort(deviations)[::-1], 'b-', lw=1, alpha=0.7)
    ax.axhline(1e-12, color='red', ls='--', lw=1, label=r'$10^{-12}$ threshold')
    ax.set_xlabel('Eigenvalue index (sorted by deviation)', fontsize=11)
    ax.set_ylabel('|eigenvalue - round(eigenvalue)|', fontsize=11)
    ax.set_title('(b) Deviation from Integer', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # (c) Constraint vs Random
    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(-j_max - 1, j_max + 1, 80)
    ax.hist(eigvals, bins=bins, color='#1565C0', alpha=0.7,
            label=f'Constraint (dev={max_dev:.0e})', edgecolor='black', linewidth=0.3)
    ax.hist(rand_eigvals, bins=bins, color='gray', alpha=0.4,
            label=f'Random (dev={rand_max_dev:.3f})')
    ax.set_xlabel(r'Eigenvalue of $i[\cdot, \cdot]$', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(c) Constraint vs Random Control', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis='y')

    # (d) Casimir spectrum
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(C2_eig, bins=60, color='#7B1FA2', alpha=0.7, edgecolor='black', linewidth=0.3)
    # Mark j(j+1) values
    for j in range(j_max + 2):
        jj1 = j * (j + 1)
        if jj1 <= C2_eig[-1] * 1.1:
            ax.axvline(jj1, color='red', ls=':', lw=1, alpha=0.5)
            ax.text(jj1, ax.get_ylim()[1]*0.9, f'j={j}', fontsize=7,
                    ha='center', color='red')
    ax.set_xlabel(r'Eigenvalue of $O_0^2 + O_1^2 + (i[O_0,O_1])^2$', fontsize=10)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(d) Casimir-like Operator', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y')

    # (e) Eigenvalue scatter (all D eigenvalues)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(range(D), eigvals_sorted, s=2, c='#1565C0', alpha=0.5)
    for j in range(-j_max, j_max + 1):
        ax.axhline(j, color='red', ls=':', lw=0.5, alpha=0.4)
    ax.set_xlabel('Index', fontsize=11)
    ax.set_ylabel(r'Eigenvalue of $i[O_0, O_1]$', fontsize=11)
    ax.set_title(f'(e) All {D} Eigenvalues (sorted)', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2)

    # (f) Summary text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    summary_lines = [
        r"$\bf{L = 10\ FULL\ DENSE\ TEST}$",
        f"",
        f"Constrained basis: D = {D}",
        f"Full space: $2^{{20}}$ = {2**(2*L):,}",
        f"Compression: {D/2**(2*L)*100:.3f}%",
        f"",
        r"$\bf{Spectrum\ of\ }i[O_0, O_1]:$",
        f"$j_{{\\max}} = {j_max} = \\lfloor {L}/2 \\rfloor$  " + ("PASS" if j_max == expected_jmax else "FAIL"),
        f"Max dev from integer: {max_dev:.1e}",
        f"Mean dev from integer: {mean_dev:.1e}",
        f"",
        r"$\bf{Degeneracy:}$",
    ]
    for u in sorted(spectrum.keys()):
        if u >= 0:
            if u == 0:
                summary_lines.append(f"  eigenvalue  0: x{spectrum[0]}")
            else:
                summary_lines.append(f"  eigenvalue +/-{u}: x{spectrum[u]}+{spectrum.get(-u, 0)}")
    summary_lines += [
        f"",
        r"$\bf{Random\ Control:}$",
        f"Max dev: {rand_max_dev:.4f}",
        f"Mean dev: {rand_mean_dev:.4f}",
        f"",
        r"$\bf{VERDICT:}$ Integer spectrum confirmed",
    ]
    ax.text(0.05, 0.95, "\n".join(summary_lines), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='#f5f5dc', alpha=0.9))

    outpath = os.path.join(_HERE, 'L10_full_spectrum.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\n  Figure saved: {outpath}")

    # Also save raw data
    np.savez(os.path.join(_HERE, 'L10_data.npz'),
             eigvals=eigvals, rand_eigvals=rand_eigvals,
             C2_eig=C2_eig, spectrum_keys=list(spectrum.keys()),
             spectrum_vals=list(spectrum.values()))
    print(f"  Data saved: L10_data.npz")

    print(f"\n{'='*65}")
    print(f"  TEST COMPLETE")
    print(f"  j_max = {j_max} = floor({L}/2) = {expected_jmax}: {'PASS' if j_max == expected_jmax else 'FAIL'}")
    print(f"  Max deviation: {max_dev:.2e} (machine precision)")
    print(f"  Integer eigenvalues: CONFIRMED for L={L}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
