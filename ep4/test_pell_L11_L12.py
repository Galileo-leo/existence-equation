#!/usr/bin/env python3
"""
EP4 Pell-Lucas Verification: L=11 and L=12
=============================================
Predictions from confirmed Pell-Lucas recurrence (companion Pell / Pell-Lucas):
  Q_n = 2*Q_{n-1} + Q_{n-2},  Q_0=2, Q_1=2
  Closed form: Q_n = (1+√2)^n + (1-√2)^n

  Conjecture:  Tr(C²) = L * Q_{L-2}

  L=11: Q_9  = 2786  → Tr(C²) = 30646,  d(j_max=5) = 11
  L=12: Q_10 = 6726  → Tr(C²) = 80712,  d(j_max=6) = 2

Strategy for large D:
  - Tr(C²) via Frobenius norm: Tr(C²) = ||C||_F² = sum|C_ij|² (sparse, O(nnz))
  - d(j_max) via scipy.sparse.linalg.eigsh on COMPLEX Hermitian iC directly
  - NO full dense eigen, NO higher moments (fill-in risk)

CRITICAL NOTE (Leo's correction):
  O0, O1 are real symmetric → K=[O0,O1] is real antisymmetric
  → C = iK is pure-imaginary Hermitian (NOT real symmetric!)
  → eigsh must operate on the complex Hermitian matrix directly
  → iC.real ≈ 0, so never use iC.real for eigsh!
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from time import time as now
import gc

def build_ladder_basis(L):
    """Build constrained 2-leg Rydberg ladder basis with PBC."""
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
    """Build projected flip operator O_l = P X_l P as sparse matrix."""
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


def compute_sparse_commutator(O0, O1):
    """Compute C = i[O0, O1], Hermitianized.

    O0, O1 real symmetric → [O0,O1] real antisymmetric → i[O0,O1] pure-imaginary Hermitian.
    Hermitianization is a no-op in exact arithmetic, but cleans numerical noise.
    """
    comm = O0 @ O1 - O1 @ O0  # real antisymmetric
    iC = 1j * comm              # pure-imaginary Hermitian
    iC = 0.5 * (iC + iC.conj().T)  # enforce exact Hermiticity
    return iC


def tr2_from_frobenius(iC_sparse):
    """Compute Tr(C²) = ||C||_F² = sum |C_ij|² for Hermitian C.
    Works entirely in sparse format — O(nnz) time, no dense conversion.
    Correct for both real and complex sparse matrices."""
    return np.real(iC_sparse.multiply(iC_sparse.conj()).sum())


def test_single_L(L):
    """Test a single L value. Lean version: only Tr(C²) + d(j_max)."""
    t0 = now()

    # ---- Build basis ----
    print(f"\n  Building basis for L={L}...")
    basis = build_ladder_basis(L)
    D = len(basis)
    j_max = L // 2
    print(f"  D = {D}, j_max = {j_max}")
    t_basis = now() - t0
    print(f"  Basis built: {t_basis:.1f}s")

    # ---- Build operators (sparse) ----
    print(f"  Building projected operators...")
    t1 = now()
    O0 = build_projected_X_sparse(basis, L, 0)
    O1 = build_projected_X_sparse(basis, L, 1)
    print(f"  O0 nnz: {O0.nnz}, O1 nnz: {O1.nnz}")

    # ---- Compute commutator (sparse) ----
    print(f"  Computing commutator...")
    iC = compute_sparse_commutator(O0, O1)
    print(f"  iC nnz: {iC.nnz}, sparsity: {iC.nnz / D**2:.6f}")
    t_ops = now() - t1
    print(f"  Operators + commutator: {t_ops:.1f}s")

    # Free O0, O1 to save memory
    del O0, O1, basis
    gc.collect()

    # ---- Verify iC is pure-imaginary Hermitian ----
    if iC.nnz > 0:
        max_real = np.max(np.abs(iC.data.real))
        max_imag = np.max(np.abs(iC.data.imag))
        print(f"  Max |Re(iC_ij)| = {max_real:.2e}  (should be ~0)")
        print(f"  Max |Im(iC_ij)| = {max_imag:.2e}  (should be large)")
        print(f"  iC is complex Hermitian; imaginary off-diagonal entries expected.")

    # ===========================================================
    # METHOD 1: Tr(C²) via Frobenius norm (ALWAYS works, O(nnz))
    # ===========================================================
    print(f"\n  === Tr(C²) via Frobenius norm ===")
    t2 = now()
    tr2_frob = tr2_from_frobenius(iC)
    tr2_int = int(np.round(tr2_frob))
    t_frob = now() - t2
    print(f"  Tr(C²) = {tr2_int}  (raw: {tr2_frob:.6f})")
    print(f"  Tr(C²)/L = {tr2_int/L:.4f}")
    print(f"  Deviation from integer: {abs(tr2_frob - tr2_int):.2e}")
    print(f"  Time: {t_frob:.2f}s")

    # ===========================================================
    # METHOD 2: d(j_max) via sparse eigsh on COMPLEX Hermitian iC
    # ===========================================================
    print(f"\n  === d(j_max) via sparse eigsh (complex Hermitian) ===")
    t4 = now()
    n_top = min(50, D - 2)
    d_jmax = -1

    try:
        # eigsh works on complex Hermitian sparse matrices directly
        top_eigs = eigsh(iC, k=n_top, which='LA',
                         return_eigenvectors=False, tol=1e-10)
        top_rounded = np.round(top_eigs).astype(int)
        d_jmax = int(np.sum(top_rounded == j_max))
        max_eig = np.max(top_eigs)
        max_dev = np.max(np.abs(top_eigs - top_rounded))

        print(f"  Top {n_top} eigenvalues computed")
        print(f"  Max eigenvalue: {max_eig:.10f} (expected {j_max})")
        print(f"  Max deviation from integer: {max_dev:.2e}")
        print(f"  d(j_max={j_max}) = {d_jmax}")

        # Show distribution of top eigenvalues
        top_unique, top_counts = np.unique(top_rounded, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(top_unique, top_counts)}
        print(f"  Top eigenvalue distribution: {dist}")
    except Exception as e:
        print(f"  eigsh failed: {e}")
        print(f"  Attempting with increased maxiter...")
        try:
            top_eigs = eigsh(iC, k=n_top, which='LA',
                             return_eigenvectors=False, tol=1e-8, maxiter=5000)
            top_rounded = np.round(top_eigs).astype(int)
            d_jmax = int(np.sum(top_rounded == j_max))
            max_eig = np.max(top_eigs)
            print(f"  Retry succeeded: max eig = {max_eig:.10f}, d(j_max) = {d_jmax}")
        except Exception as e2:
            print(f"  Retry also failed: {e2}")
            d_jmax = -1

    t_eig = now() - t4
    print(f"  Time: {t_eig:.1f}s")

    elapsed = now() - t0
    print(f"\n  Total time for L={L}: {elapsed:.1f}s")

    return {
        'L': L, 'D': D, 'j_max': j_max,
        'tr2': tr2_int, 'd_jmax': d_jmax,
    }


def main():
    print("=" * 65)
    print("  PELL-LUCAS VERIFICATION: L=11, L=12")
    print("  Conjecture: Tr(C²) = L * Q_{L-2}")
    print("  Q_n = companion Pell (Pell-Lucas): Q_0=2, Q_1=2,")
    print("  Q_n = 2*Q_{n-1} + Q_{n-2}")
    print("  Q_n = (1+√2)^n + (1-√2)^n")
    print("=" * 65)

    # Pell-Lucas / companion Pell sequence Q_n
    Q = {0: 2, 1: 2}
    for n in range(2, 15):
        Q[n] = 2 * Q[n-1] + Q[n-2]

    print("\n  Companion Pell sequence Q_n:")
    for n in range(0, 13):
        print(f"    Q_{n:2d} = {Q[n]:>10}")

    print(f"\n  Conjecture table: Tr(C²) = L * Q_{{L-2}}")
    for L in range(3, 14):
        print(f"    L={L:2d}: Q_{L-2:2d} = {Q[L-2]:>10} → Tr(C²) = {L * Q[L-2]:>10}")

    # Predictions
    pred = {
        11: {'tr2': 11 * Q[9],  'd_jmax': 11},  # 11 * 2786 = 30646
        12: {'tr2': 12 * Q[10], 'd_jmax': 2},    # 12 * 6726 = 80712
    }

    print(f"\n  PREDICTIONS TO VERIFY:")
    for L in [11, 12]:
        p = pred[L]
        print(f"    L={L}: Tr(C²) = {p['tr2']}, "
              f"Tr(C²)/L = Q_{{{L-2}}} = {Q[L-2]}, "
              f"d(j_max={L//2}) = {p['d_jmax']}")

    # ============================================================
    # TEST L=11
    # ============================================================
    print("\n" + "=" * 65)
    print("  TESTING L = 11")
    print("=" * 65)
    r11 = test_single_L(11)
    gc.collect()

    # ============================================================
    # TEST L=12
    # ============================================================
    print("\n" + "=" * 65)
    print("  TESTING L = 12")
    print("=" * 65)
    r12 = test_single_L(12)
    gc.collect()

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "=" * 65)
    print("  FINAL VERDICT")
    print("=" * 65)

    all_pass = True
    for L, result in [(11, r11), (12, r12)]:
        p = pred[L]
        tr2_match = result['tr2'] == p['tr2']
        d_match = result['d_jmax'] == p['d_jmax']
        all_pass = all_pass and tr2_match and d_match

        print(f"\n  L = {L} (D = {result['D']}):")
        print(f"    Tr(C²):      predicted {p['tr2']:>8}, actual {result['tr2']:>8}  "
              f"{'✓ MATCH' if tr2_match else '✗ MISMATCH'}")
        print(f"    d(j_max={result['j_max']}): predicted {p['d_jmax']:>8}, actual {result['d_jmax']:>8}  "
              f"{'✓ MATCH' if d_match else '✗ MISMATCH'}")

    # Complete Tr(C²)/L table
    known = {
        3:  (6,    2),    4:  (24,   6),    5:  (70,   14),
        6:  (204,  34),   7:  (574,  82),   8:  (1584, 198),
        9:  (4302, 478),  10: (11540, 1154),
    }

    print(f"\n  === COMPLETE Tr(C²) = L * Q_{{L-2}} TABLE ===")
    print(f"  {'L':>3} {'D':>8} {'Tr(C²)':>10} {'Tr(C²)/L':>10} {'Q_{L-2}':>10} {'match':>6}")
    print(f"  {'-'*55}")

    for L in range(3, 13):
        if L in known:
            tr2, ratio = known[L]
        elif L == 11:
            tr2 = r11['tr2']; ratio = tr2 // L
        elif L == 12:
            tr2 = r12['tr2']; ratio = tr2 // L
        q_val = Q[L - 2]
        match = abs(ratio - q_val) < 0.5
        sym = '✓' if match else '✗'
        D_str = str(r11['D']) if L == 11 else (str(r12['D']) if L == 12 else '')
        print(f"  {L:3d} {D_str:>8} {tr2:10d} {ratio:10d} {q_val:10d} {sym:>6}")

    if all_pass:
        print(f"\n  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║                                                          ║")
        print(f"  ║   Tr(C²) = L · Q_{{L-2}}                                  ║")
        print(f"  ║   Q_n = (1+√2)^n + (1-√2)^n   (Pell-Lucas sequence)    ║")
        print(f"  ║                                                          ║")
        print(f"  ║   d(j_max) = L  (L odd),  d(j_max) = 2  (L even)       ║")
        print(f"  ║                                                          ║")
        print(f"  ║   Verified: L = 3, 4, 5, 6, 7, 8, 9, 10, 11, 12       ║")
        print(f"  ║   10 consecutive values — EXACT integer match           ║")
        print(f"  ║                                                          ║")
        print(f"  ║   STATUS: UNIVERSAL LAW CONFIRMED                       ║")
        print(f"  ║                                                          ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")

    # Next predictions
    print(f"\n  === NEXT PREDICTIONS (untested) ===")
    for L_pred in [13, 14, 15]:
        q = Q[L_pred - 2]
        tr2_pred = L_pred * q
        d_pred = L_pred if L_pred % 2 == 1 else 2
        print(f"  L={L_pred}: Tr(C²) = {L_pred} * Q_{{{L_pred-2}}} = "
              f"{L_pred} * {q} = {tr2_pred}, d(j_max={L_pred//2}) = {d_pred}")

    print(f"\n{'='*65}")
    print(f"  TEST COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
