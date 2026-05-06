#!/usr/bin/env python3
"""
EP4 Casimir Convergence Test
==============================
The big question: under what limit does the constraint-induced
Casimir operator converge to j(j+1)?

Tests:
  A. Joint (m, C²) spectrum — do states cluster on j(j+1) parabolas?
  B. Algebra closure — does [J₃, O₀] ∝ O₁ improve with L?
  C. Low-lying sector — are low-Casimir states closer to j(j+1)?
  D. L-scaling — does convergence improve as L grows?
  E. Restricted subspace — project onto the "best SU(2)" subspace

Jae-Ahn Shin, Independent Researcher
jaeahn.shin.official@gmail.com
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time as now
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# BASIS + OPERATORS (same as EP4)
# ============================================================

def build_ladder_basis(L):
    rung_states = [(0, 0), (1, 0), (0, 1)]
    def compatible(r1, r2):
        return not (r1[0]==1 and r2[0]==1) and not (r1[1]==1 and r2[1]==1)
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
                rows.append(idx); cols.append(basis_dict[key]); vals.append(1.0)
    return csr_matrix((vals, (rows, cols)), shape=(D, D))


def analyze_single_L(L, verbose=True):
    """Full analysis for a given L. Returns dict with all results."""
    t0 = now()
    basis = build_ladder_basis(L)
    D = len(basis)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  L = {L},  D = {D}")
        print(f"{'='*60}")

    O0 = build_projected_X_sparse(basis, L, 0).toarray()
    O1 = build_projected_X_sparse(basis, L, 1).toarray()

    # J3 = i[O0, O1]  (Hermitian)
    comm = O0 @ O1 - O1 @ O0
    J3 = 1j * comm
    J3 = 0.5 * (J3 + J3.conj().T)  # enforce Hermiticity

    # Casimir-like: C2 = O0^2 + O1^2 + J3^2
    C2 = O0 @ O0 + O1 @ O1 + J3 @ J3

    # ============================================================
    # A. Joint diagonalization: eigenstates of J3, then C2 expectation
    # ============================================================
    j3_eigvals, j3_eigvecs = np.linalg.eigh(J3)
    m_values = np.round(j3_eigvals).astype(int)

    # For each eigenstate of J3, compute <C2>
    c2_expect = np.array([
        np.real(v.conj() @ C2 @ v) for v in j3_eigvecs.T
    ])

    # ============================================================
    # B. Algebra closure test
    #    SU(2) requires: [J3, O0] = α O1,  [J3, O1] = β O0
    #    More precisely, defining J± = O0 ± iO1:
    #    [J3, J+] = J+  and  [J3, J-] = -J-
    #    So [J3, O0] = i O1  and  [J3, O1] = -i O0
    # ============================================================
    comm_J3_O0 = J3 @ O0 - O0 @ J3  # should be ~ i*O1
    comm_J3_O1 = J3 @ O1 - O1 @ J3  # should be ~ -i*O0

    # Closure metric: ||[J3, O0] - iO1|| / ||iO1||
    target_0 = 1j * O1
    target_1 = -1j * O0
    closure_err_0 = np.linalg.norm(comm_J3_O0 - target_0) / np.linalg.norm(target_0)
    closure_err_1 = np.linalg.norm(comm_J3_O1 - target_1) / np.linalg.norm(target_1)

    # Also check: does the algebra close at all?
    # [J3, O0] should be expressible as linear combination of O0, O1, J3
    A_basis = np.column_stack([O0.flatten(), O1.flatten(), J3.flatten()])
    coeffs_0, res_0, _, _ = np.linalg.lstsq(A_basis, comm_J3_O0.flatten(), rcond=None)
    coeffs_1, res_1, _, _ = np.linalg.lstsq(A_basis, comm_J3_O1.flatten(), rcond=None)
    closure_residual_0 = np.linalg.norm(comm_J3_O0.flatten() - A_basis @ coeffs_0) / np.linalg.norm(comm_J3_O0.flatten())
    closure_residual_1 = np.linalg.norm(comm_J3_O1.flatten() - A_basis @ coeffs_1) / np.linalg.norm(comm_J3_O1.flatten())

    if verbose:
        print(f"\n  --- ALGEBRA CLOSURE ---")
        print(f"  ||[J3,O0] - iO1|| / ||iO1|| = {closure_err_0:.6f}")
        print(f"  ||[J3,O1] + iO0|| / ||iO0|| = {closure_err_1:.6f}")
        print(f"  [J3,O0] span residual (in {{O0,O1,J3}}): {closure_residual_0:.6f}")
        print(f"  [J3,O1] span residual (in {{O0,O1,J3}}): {closure_residual_1:.6f}")
        print(f"  Coefficients [J3,O0] = {coeffs_0[0]:.4f}*O0 + {coeffs_0[1]:.4f}*O1 + {coeffs_0[2]:.4f}*J3")
        print(f"  Coefficients [J3,O1] = {coeffs_1[0]:.4f}*O0 + {coeffs_1[1]:.4f}*O1 + {coeffs_1[2]:.4f}*J3")

    # ============================================================
    # C. m-sector decomposition: within each m sector, diagonalize C2
    # ============================================================
    m_unique = np.unique(m_values)
    sector_data = {}

    for m in m_unique:
        mask = (m_values == m)
        indices = np.where(mask)[0]
        dim_m = len(indices)

        # Extract C2 in this m-sector (using J3 eigenbasis)
        V_m = j3_eigvecs[:, indices]  # D x dim_m
        C2_m = V_m.conj().T @ C2 @ V_m
        C2_m = 0.5 * (C2_m + C2_m.conj().T)  # Hermitize
        c2_eigs_m = np.sort(np.real(np.linalg.eigvalsh(C2_m)))

        # For pure SU(2), states with quantum number m can only come from
        # representations j >= |m|. The Casimir would be j(j+1) for j = |m|, |m|+1, ...
        j_min = abs(int(m))
        expected_c2 = [j*(j+1) for j in range(j_min, j_min + 10)]

        # Distance of each C2 eigenvalue to nearest j(j+1)
        distances = []
        for c2v in c2_eigs_m:
            min_dist = min(abs(c2v - jj1) for jj1 in expected_c2 if jj1 < c2v + 50)
            distances.append(min_dist)

        sector_data[int(m)] = {
            'dim': dim_m,
            'c2_eigs': c2_eigs_m,
            'distances': np.array(distances),
            'mean_dist': np.mean(distances),
            'median_dist': np.median(distances),
            'frac_close': np.mean(np.array(distances) < 0.1),  # fraction within 0.1 of j(j+1)
        }

    if verbose:
        print(f"\n  --- M-SECTOR C² ANALYSIS ---")
        print(f"  {'m':>4} {'dim':>6} {'mean_dist':>10} {'med_dist':>10} {'frac<0.1':>10}")
        print(f"  {'-'*44}")
        for m in sorted(sector_data.keys()):
            sd = sector_data[m]
            print(f"  {m:>4d} {sd['dim']:>6d} {sd['mean_dist']:>10.4f} "
                  f"{sd['median_dist']:>10.4f} {sd['frac_close']:>10.3f}")

    # ============================================================
    # D. Find "best SU(2)" subspace
    #    Strategy: find states where C2 is closest to j(j+1)
    # ============================================================
    # For each state, find nearest j(j+1)
    j_max_obs = max(abs(m) for m in m_unique)
    all_jj1 = [j*(j+1) for j in range(j_max_obs + 5)]

    nearest_jj1 = np.array([
        min(all_jj1, key=lambda x: abs(c2_expect[i] - x))
        for i in range(D)
    ])
    dist_to_jj1 = np.abs(c2_expect - nearest_jj1)

    # What fraction of states are within various thresholds of j(j+1)?
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    if verbose:
        print(f"\n  --- PROXIMITY TO j(j+1) ---")
        for thr in thresholds:
            frac = np.mean(dist_to_jj1 < thr)
            count = np.sum(dist_to_jj1 < thr)
            print(f"  |C² - j(j+1)| < {thr:.2f}: {count:>5d} / {D}  ({frac*100:.1f}%)")

    # ============================================================
    # E. J+/J- ladder test
    #    In SU(2): J+ |j,m> = sqrt(j(j+1)-m(m+1)) |j,m+1>
    #    Define J+ = O0 + iO1, J- = O0 - iO1
    #    Check if J+ maps m-sector to (m+1)-sector cleanly
    # ============================================================
    Jp = O0 + 1j * O1
    Jm = O0 - 1j * O1

    # Check: J+ J- = O0^2 + O1^2 + J3  (should be C2 - J3^2 + J3 in SU(2))
    JpJm = Jp @ Jm
    expected_JpJm = C2 - J3 @ J3 + J3  # = j(j+1) - m(m-1) in SU(2)
    JpJm_err = np.linalg.norm(JpJm - expected_JpJm) / np.linalg.norm(JpJm)

    # Also: J- J+ = C2 - J3^2 - J3
    JmJp = Jm @ Jp
    expected_JmJp = C2 - J3 @ J3 - J3
    JmJp_err = np.linalg.norm(JmJp - expected_JmJp) / np.linalg.norm(JmJp)

    if verbose:
        print(f"\n  --- LADDER OPERATOR TEST ---")
        print(f"  ||J+J- - (C² - J3² + J3)|| / ||J+J-|| = {JpJm_err:.2e}")
        print(f"  ||J-J+ - (C² - J3² - J3)|| / ||J-J+|| = {JmJp_err:.2e}")
        print(f"  (These should be ~0 if C² = O0²+O1²+J3² is the true Casimir)")

    # ============================================================
    # F. Nested commutator: does the algebra eventually close?
    #    Compute [J3, [J3, O0]], [J3, [J3, [J3, O0]]], ...
    #    In SU(2): [J3, [J3, O0]] = -O0 (since [J3, O0] = iO1, [J3, iO1] = -O0)
    # ============================================================
    nested_1 = J3 @ O0 - O0 @ J3  # [J3, O0]
    nested_2 = J3 @ nested_1 - nested_1 @ J3  # [J3, [J3, O0]]

    # In SU(2): nested_2 = -O0
    nested2_vs_O0 = np.linalg.norm(nested_2 + O0) / np.linalg.norm(O0)

    # Also: [J3, [J3, O1]] = -O1
    nested_1b = J3 @ O1 - O1 @ J3
    nested_2b = J3 @ nested_1b - nested_1b @ J3
    nested2_vs_O1 = np.linalg.norm(nested_2b + O1) / np.linalg.norm(O1)

    if verbose:
        print(f"\n  --- NESTED COMMUTATOR TEST ---")
        print(f"  ||[J3,[J3,O0]] + O0|| / ||O0|| = {nested2_vs_O0:.6f}")
        print(f"  ||[J3,[J3,O1]] + O1|| / ||O1|| = {nested2_vs_O1:.6f}")
        print(f"  (Should be 0 for exact SU(2))")

    elapsed = now() - t0
    if verbose:
        print(f"\n  Time: {elapsed:.1f}s")

    return {
        'L': L, 'D': D,
        'm_values': m_values, 'c2_expect': c2_expect,
        'j3_eigvals': j3_eigvals,
        'closure_err': (closure_err_0, closure_err_1),
        'closure_residual': (closure_residual_0, closure_residual_1),
        'closure_coeffs': (coeffs_0, coeffs_1),
        'sector_data': sector_data,
        'dist_to_jj1': dist_to_jj1,
        'nearest_jj1': nearest_jj1,
        'nested2_err': (nested2_vs_O0, nested2_vs_O1),
        'JpJm_err': JpJm_err,
        'j_max': max(abs(m) for m in np.unique(m_values)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  EP4 CASIMIR CONVERGENCE TEST")
    print("  Does constraint-induced algebra converge to SU(2)?")
    print("=" * 65)

    L_list = [3, 4, 5, 6, 8, 10]
    all_results = {}

    for L in L_list:
        all_results[L] = analyze_single_L(L)

    # ============================================================
    # SCALING SUMMARY
    # ============================================================
    print("\n" + "=" * 65)
    print("  SCALING SUMMARY ACROSS L")
    print("=" * 65)

    print(f"\n  {'L':>3} {'D':>7} {'j_max':>6} "
          f"{'closure_0':>10} {'closure_1':>10} "
          f"{'nested_0':>10} {'nested_1':>10} "
          f"{'frac<0.1':>9}")
    print(f"  {'-'*70}")

    for L in L_list:
        r = all_results[L]
        frac_close = np.mean(r['dist_to_jj1'] < 0.1)
        print(f"  {L:3d} {r['D']:7d} {r['j_max']:6d} "
              f"{r['closure_err'][0]:10.6f} {r['closure_err'][1]:10.6f} "
              f"{r['nested2_err'][0]:10.6f} {r['nested2_err'][1]:10.6f} "
              f"{frac_close:9.3f}")

    # Key question: do closure errors DECREASE with L?
    closure_0_vals = [all_results[L]['closure_err'][0] for L in L_list]
    closure_1_vals = [all_results[L]['closure_err'][1] for L in L_list]
    nested_0_vals = [all_results[L]['nested2_err'][0] for L in L_list]
    nested_1_vals = [all_results[L]['nested2_err'][1] for L in L_list]

    print(f"\n  Closure error trend (||[J3,O0]-iO1||/||iO1||):")
    print(f"    L=3: {closure_0_vals[0]:.6f} → L=10: {closure_0_vals[-1]:.6f}")
    if closure_0_vals[-1] < closure_0_vals[0]:
        print(f"    DECREASING: possible convergence to SU(2)!")
    else:
        print(f"    NOT decreasing: algebra does not close in large-L limit")

    print(f"\n  Nested commutator trend (||[J3,[J3,O0]]+O0||/||O0||):")
    print(f"    L=3: {nested_0_vals[0]:.6f} → L=10: {nested_0_vals[-1]:.6f}")
    if nested_0_vals[-1] < nested_0_vals[0]:
        print(f"    DECREASING: SU(2) structure improves with L!")
    else:
        print(f"    NOT decreasing: nested structure stable")

    # ============================================================
    # FIGURES
    # ============================================================

    # --- Figure 1: (m, C²) scatter for each L ---
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle('Joint $(m, C^2)$ Spectrum: States vs $j(j+1)$ Parabolas\n'
                  'Do constraint-induced states fall on SU(2) representation curves?',
                  fontsize=13, fontweight='bold')

    for idx, L in enumerate(L_list):
        ax = axes1.flat[idx]
        r = all_results[L]
        j_max = r['j_max']

        # Plot j(j+1) curves
        m_range = np.linspace(-j_max - 0.5, j_max + 0.5, 200)
        for j in range(j_max + 2):
            jj1 = j * (j + 1)
            valid = np.abs(m_range) <= j
            ax.plot(m_range[valid], np.full(np.sum(valid), jj1),
                    'r-', lw=1.5, alpha=0.3)
            ax.text(j + 0.2, jj1, f'j={j}', fontsize=7, color='red', alpha=0.5)

        # Scatter: (m, C²) for each eigenstate
        sc = ax.scatter(r['m_values'] + np.random.randn(r['D'])*0.05,  # jitter
                        r['c2_expect'],
                        c=r['dist_to_jj1'], cmap='viridis_r',
                        s=3, alpha=0.5, vmin=0, vmax=2)
        ax.set_xlabel('$m$ (eigenvalue of $J_3$)', fontsize=10)
        ax.set_ylabel('$\\langle C^2 \\rangle = O_0^2+O_1^2+J_3^2$', fontsize=10)
        ax.set_title(f'L={L}, D={r["D"]}, $j_{{max}}$={j_max}', fontsize=10, fontweight='bold')
        ax.set_xlim(-j_max - 0.8, j_max + 0.8)
        plt.colorbar(sc, ax=ax, label='dist to $j(j+1)$', shrink=0.8)

    plt.tight_layout()
    fig1.savefig(os.path.join(_HERE, 'casimir_mC2_scatter.png'), dpi=200, bbox_inches='tight')
    print(f"\n  Saved: {os.path.join(_HERE, 'casimir_mC2_scatter.png')}")

    # --- Figure 2: Scaling trends ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Convergence to SU(2): Scaling with System Size $L$',
                  fontsize=13, fontweight='bold')

    # (a) Closure error vs L
    ax = axes2[0, 0]
    ax.plot(L_list, closure_0_vals, 'bo-', lw=2, ms=8, label=r'$\|[J_3,O_0]-iO_1\|/\|iO_1\|$')
    ax.plot(L_list, closure_1_vals, 'rs-', lw=2, ms=8, label=r'$\|[J_3,O_1]+iO_0\|/\|iO_0\|$')
    ax.set_xlabel('$L$ (rungs)', fontsize=11)
    ax.set_ylabel('Closure Error', fontsize=11)
    ax.set_title('(a) Algebra Closure Error vs $L$', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(L_list)

    # (b) Nested commutator error vs L
    ax = axes2[0, 1]
    ax.plot(L_list, nested_0_vals, 'bo-', lw=2, ms=8, label=r'$\|[J_3,[J_3,O_0]]+O_0\|/\|O_0\|$')
    ax.plot(L_list, nested_1_vals, 'rs-', lw=2, ms=8, label=r'$\|[J_3,[J_3,O_1]]+O_1\|/\|O_1\|$')
    ax.set_xlabel('$L$ (rungs)', fontsize=11)
    ax.set_ylabel('Nested Commutator Error', fontsize=11)
    ax.set_title('(b) $[J_3,[J_3,O_\\ell]] = -O_\\ell$ Test', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(L_list)

    # (c) Fraction of states near j(j+1) vs L
    ax = axes2[1, 0]
    for thr in [0.05, 0.1, 0.2, 0.5]:
        fracs = [np.mean(all_results[L]['dist_to_jj1'] < thr) for L in L_list]
        ax.plot(L_list, fracs, 'o-', lw=2, ms=7, label=f'$< {thr}$')
    ax.set_xlabel('$L$ (rungs)', fontsize=11)
    ax.set_ylabel('Fraction of states near $j(j+1)$', fontsize=11)
    ax.set_title('(c) States Approaching $j(j+1)$ vs $L$', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, title='$|C^2 - j(j+1)|$')
    ax.grid(alpha=0.3)
    ax.set_xticks(L_list)
    ax.set_ylim(0, 1)

    # (d) Closure coefficients: [J3,O0] = a*O0 + b*O1 + c*J3
    # In SU(2): a=0, b=i (≈1 in magnitude), c=0
    ax = axes2[1, 1]
    a_vals = [np.abs(all_results[L]['closure_coeffs'][0][0]) for L in L_list]
    b_vals = [np.abs(all_results[L]['closure_coeffs'][0][1]) for L in L_list]
    c_vals = [np.abs(all_results[L]['closure_coeffs'][0][2]) for L in L_list]
    ax.plot(L_list, a_vals, 'go-', lw=2, ms=7, label='$|a|$ (O₀ coeff, should→0)')
    ax.plot(L_list, b_vals, 'bo-', lw=2, ms=7, label='$|b|$ (O₁ coeff, should→1)')
    ax.plot(L_list, c_vals, 'ro-', lw=2, ms=7, label='$|c|$ (J₃ coeff, should→0)')
    ax.axhline(1.0, color='blue', ls=':', alpha=0.3)
    ax.axhline(0.0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('$L$ (rungs)', fontsize=11)
    ax.set_ylabel('Coefficient magnitude', fontsize=11)
    ax.set_title('(d) $[J_3, O_0] = aO_0 + bO_1 + cJ_3$\n(SU(2) predicts $a=0, b=i, c=0$)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(L_list)

    plt.tight_layout()
    fig2.savefig(os.path.join(_HERE, 'casimir_scaling.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {os.path.join(_HERE, 'casimir_scaling.png')}")

    # --- Figure 3: Per-L histogram of distance to j(j+1) ---
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
    fig3.suptitle('Distribution of $|C^2 - j(j+1)|$: How Close to Pure SU(2)?',
                  fontsize=13, fontweight='bold')

    for idx, L in enumerate(L_list):
        ax = axes3.flat[idx]
        r = all_results[L]
        ax.hist(r['dist_to_jj1'], bins=50, color='steelblue', alpha=0.7,
                edgecolor='black', linewidth=0.3)
        ax.axvline(0.1, color='red', ls='--', lw=1.5, label='threshold=0.1')
        frac = np.mean(r['dist_to_jj1'] < 0.1)
        ax.set_title(f'L={L}, D={r["D"]}\n{frac*100:.1f}% within 0.1 of $j(j+1)$',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('$|C^2 - j(j+1)|$', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis='y')
        ax.set_xlim(0, min(5, np.percentile(r['dist_to_jj1'], 99)*1.2))

    plt.tight_layout()
    fig3.savefig(os.path.join(_HERE, 'casimir_distance_hist.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {os.path.join(_HERE, 'casimir_distance_hist.png')}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)

    c0_trend = "CONVERGING" if closure_0_vals[-1] < closure_0_vals[0] * 0.9 else "STABLE/DIVERGING"
    n0_trend = "CONVERGING" if nested_0_vals[-1] < nested_0_vals[0] * 0.9 else "STABLE/DIVERGING"
    f_trend_vals = [np.mean(all_results[L]['dist_to_jj1'] < 0.1) for L in L_list]
    f_trend = "INCREASING" if f_trend_vals[-1] > f_trend_vals[0] * 1.1 else "STABLE/DECREASING"

    print(f"\n  Algebra closure error (L=3→10): {c0_trend}")
    print(f"  Nested commutator error (L=3→10): {n0_trend}")
    print(f"  Fraction near j(j+1) (L=3→10): {f_trend}")

    if c0_trend == "CONVERGING" and f_trend == "INCREASING":
        print(f"\n  >>> EVIDENCE FOR CONVERGENCE TO SU(2) IN LARGE-L LIMIT <<<")
        print(f"  >>> The constraint-induced algebra approaches exact SU(2)")
        print(f"  >>> as the system size grows.")
    elif c0_trend == "CONVERGING" or f_trend == "INCREASING":
        print(f"\n  >>> PARTIAL EVIDENCE for SU(2) emergence.")
        print(f"  >>> Some indicators improve, others don't.")
        print(f"  >>> The relationship is more subtle than simple convergence.")
    else:
        print(f"\n  >>> NO CONVERGENCE to standard SU(2) in simple large-L limit.")
        print(f"  >>> The constraint-induced algebra is a DISTINCT structure")
        print(f"  >>> that shares the integer commutator property with SU(2)")
        print(f"  >>> but has its own representation theory.")
        print(f"  >>> This is itself a significant finding.")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
