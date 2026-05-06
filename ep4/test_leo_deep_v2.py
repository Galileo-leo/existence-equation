#!/usr/bin/env python3
"""
EP4 Deep Structure Test v2 — Leo's Three Questions (corrected)
===============================================================
Fixes per Leo's review:
  1. Do NOT diagonalize non-Hermitian term_A, term_B individually.
     Report norms, ranks, singular values, forbidden-state usage only.
     Diagonalize only i(A-B) = i[O0,O1].
  2. Dense leakage for L<=6 only.
  3. Algebra closure uses Hermitian Lie bracket:
     hbracket(A,B) = HermitianPart(i(AB - BA))
     Real-linear rank of Hermitian matrices.
  4. Check ||P_m iC P_n|| off-block norms before interpreting m-sector spectra.
  5. Trace moment diagnostics: Tr(C^2), Tr(C^4), Tr(C^6)
     Compare with sum_j N_j j^{2k}.

Jae-Ahn Shin, Independent Researcher
jaeahn.shin.official@gmail.com
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time as now
import sys

# ============================================================
# BASIS + OPERATORS
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

def build_full_X(L, leg):
    """Full-space flip operator X_leg (2^{2L} x 2^{2L})."""
    N = 2**(2*L)
    rows, cols, vals = [], [], []
    for s in range(N):
        bits = [(s >> b) & 1 for b in range(2*L)]
        for i in range(L):
            site = 2*i + leg
            new_bits = bits.copy()
            new_bits[site] = 1 - new_bits[site]
            s2 = sum(b << j for j, b in enumerate(new_bits))
            rows.append(s); cols.append(s2); vals.append(1.0)
    return csr_matrix((vals, (rows, cols)), shape=(N, N))

def build_projector(basis, L):
    N = 2**(2*L)
    D = len(basis)
    P = np.zeros((N, D))
    for j, b in enumerate(basis):
        s = sum(bit << k for k, bit in enumerate(b))
        P[s, j] = 1.0
    return P


# ============================================================
# TEST 1: LEAKAGE DECOMPOSITION (corrected)
# ============================================================

def test1_leakage(L):
    """
    [O0,O1] = P(X1 Q X0 - X0 Q X1)P  where Q = I - PP^T.

    Corrected: do NOT diagonalize individual terms.
    term_B = term_A^T (not Hermitian), so eigenvalue analysis is invalid.
    Report: norms, ranks, singular values, forbidden-state structure.
    """
    print("=" * 65)
    print(f"  TEST 1: LEAKAGE DECOMPOSITION (L={L}) — CORRECTED")
    print("  [O0,O1] = P(X1 Q X0 - X0 Q X1)P")
    print("=" * 65)

    basis = build_ladder_basis(L)
    D = len(basis)
    N = 2**(2*L)
    print(f"  D = {D}, N = {N}, forbidden dim = {N - D}")

    t0 = now()
    P_mat = build_projector(basis, L)
    X0_full = build_full_X(L, 0).toarray()
    X1_full = build_full_X(L, 1).toarray()
    PP = P_mat @ P_mat.T
    Q_mat = np.eye(N) - PP

    # Term A: P^T X1 Q X0 P
    QX0P = Q_mat @ X0_full @ P_mat
    term_A = P_mat.T @ X1_full @ QX0P

    # Term B: P^T X0 Q X1 P
    QX1P = Q_mat @ X1_full @ P_mat
    term_B = P_mat.T @ X0_full @ QX1P

    # Verify: term_B = term_A^T ?
    BmAT = np.linalg.norm(term_B - term_A.T) / max(np.linalg.norm(term_A), 1e-15)
    print(f"\n  ||term_B - term_A^T|| / ||term_A|| = {BmAT:.2e}")
    print(f"  → term_B = term_A^T: {'YES' if BmAT < 1e-12 else 'NO'}")

    # Commutator from leakage
    comm_leakage = term_A - term_B
    O0 = build_projected_X_sparse(basis, L, 0).toarray()
    O1 = build_projected_X_sparse(basis, L, 1).toarray()
    comm_direct = O0 @ O1 - O1 @ O0
    match_err = np.linalg.norm(comm_leakage - comm_direct) / np.linalg.norm(comm_direct)
    print(f"  Verification: ||leakage - direct|| / ||direct|| = {match_err:.2e}")

    # --- Norms ---
    norm_A = np.linalg.norm(term_A, 'fro')
    norm_B = np.linalg.norm(term_B, 'fro')
    norm_C = np.linalg.norm(comm_leakage, 'fro')
    print(f"\n  --- NORMS ---")
    print(f"  ||A||_F = {norm_A:.6f}")
    print(f"  ||B||_F = {norm_B:.6f}  (= ||A||_F since B = A^T)")
    print(f"  ||A-B||_F = {norm_C:.6f}")
    print(f"  ||A||² + ||B||² = {norm_A**2 + norm_B**2:.6f}")
    print(f"  ||A-B||² = {norm_C**2:.6f}")
    print(f"  tr(A^T B) = {np.trace(term_A.T @ term_B):.6f}")
    print(f"  → ||A-B||² = ||A||² + ||B||² - 2 tr(A^T B) = "
          f"{norm_A**2 + norm_B**2 - 2*np.trace(term_A.T @ term_B):.6f}")

    # --- Ranks ---
    rank_A = np.linalg.matrix_rank(term_A, tol=1e-10)
    rank_C = np.linalg.matrix_rank(comm_leakage, tol=1e-10)
    print(f"\n  --- RANKS ---")
    print(f"  rank(A) = {rank_A}")
    print(f"  rank(A-B) = rank([O0,O1]) = {rank_C}")
    print(f"  D = {D}")

    # --- Singular values of term_A ---
    svA = np.linalg.svd(term_A, compute_uv=False)
    svA = svA[svA > 1e-14]
    print(f"\n  --- SINGULAR VALUES of term_A ---")
    print(f"  Number of nonzero SVs: {len(svA)}")
    print(f"  Top 10: {np.round(svA[:10], 6)}")
    if len(svA) > 10:
        print(f"  Bottom 5: {np.round(svA[-5:], 6)}")

    # Are singular values integers or half-integers?
    sv_dev_int = np.max(np.abs(svA - np.round(svA)))
    sv_dev_half = np.max(np.abs(svA - np.round(2*svA)/2))
    print(f"  Max dev from integer: {sv_dev_int:.6f}")
    print(f"  Max dev from half-integer: {sv_dev_half:.6f}")

    # --- Forbidden state analysis ---
    # Which forbidden states are accessed?
    forbidden_usage_0 = np.sum(np.abs(QX0P)**2, axis=1)
    forbidden_usage_1 = np.sum(np.abs(QX1P)**2, axis=1)

    n_forb_0 = np.sum(forbidden_usage_0 > 1e-10)
    n_forb_1 = np.sum(forbidden_usage_1 > 1e-10)
    print(f"\n  --- FORBIDDEN STATES ---")
    print(f"  Accessed by X0: {n_forb_0} / {N-D}")
    print(f"  Accessed by X1: {n_forb_1} / {N-D}")

    # Classify by violation type
    n_block, n_rung, n_both = 0, 0, 0
    for s in range(N):
        bits = [(s >> b) & 1 for b in range(2*L)]
        viol_block, viol_rung = False, False
        for i in range(L):
            if bits[2*i] == 1 and bits[2*i+1] == 1:
                viol_rung = True
            i_next = (i+1) % L
            if bits[2*i] == 1 and bits[2*i_next] == 1:
                viol_block = True
            if bits[2*i+1] == 1 and bits[2*i_next+1] == 1:
                viol_block = True
        is_constrained = (not viol_block) and (not viol_rung)
        if not is_constrained:
            if forbidden_usage_0[s] > 1e-10 or forbidden_usage_1[s] > 1e-10:
                if viol_block and viol_rung: n_both += 1
                elif viol_block: n_block += 1
                elif viol_rung: n_rung += 1

    print(f"  Blockade only: {n_block}")
    print(f"  Rung only:     {n_rung}")
    print(f"  Both:          {n_both}")

    # --- Antisymmetric part: the commutator ---
    # A - B = A - A^T  (antisymmetric part of A, times 2)
    antisym = 0.5 * (term_A - term_A.T)  # A_antisym = (A - A^T)/2
    sym = 0.5 * (term_A + term_A.T)       # A_sym = (A + A^T)/2
    print(f"\n  --- SYMMETRIC / ANTISYMMETRIC DECOMPOSITION ---")
    print(f"  A = A_sym + A_antisym")
    print(f"  ||A_sym||_F = {np.linalg.norm(sym):.6f}")
    print(f"  ||A_antisym||_F = {np.linalg.norm(antisym):.6f}")
    print(f"  [O0,O1] = 2 * A_antisym")
    print(f"  → Non-commutativity lives entirely in the antisymmetric part")

    elapsed = now() - t0
    print(f"\n  Time: {elapsed:.1f}s")


# ============================================================
# TEST 2: ALGEBRA DIMENSION GROWTH (Hermitian Lie bracket)
# ============================================================

def hermitian_part(M):
    return 0.5 * (M + M.conj().T)

def hbracket(A, B):
    """Hermitian Lie bracket: HermitianPart(i(AB - BA))."""
    return hermitian_part(1j * (A @ B - B @ A))

def herm_to_real_vec(M):
    """Vectorize Hermitian matrix to real vector.
    Real part of upper triangle + imaginary part of strict upper triangle."""
    D = M.shape[0]
    # For real-linear independence of Hermitian matrices,
    # flatten real and imaginary parts separately
    return np.concatenate([M.real.flatten(), M.imag.flatten()])

def test2_algebra_growth(L):
    """
    Corrected: use Hermitian Lie bracket throughout.
    Start from Hermitian O0, O1, close under hbracket.
    Track real-linear rank of the generated Hermitian algebra.
    """
    print("\n" + "=" * 65)
    print(f"  TEST 2: ALGEBRA DIMENSION GROWTH (L={L}) — Hermitian bracket")
    print("  hbracket(A,B) = HermitianPart(i(AB - BA))")
    print("=" * 65)

    basis = build_ladder_basis(L)
    D = len(basis)
    O0 = build_projected_X_sparse(basis, L, 0).toarray()
    O1 = build_projected_X_sparse(basis, L, 1).toarray()

    # Verify Hermiticity
    assert np.allclose(O0, O0.T), "O0 not Hermitian!"
    assert np.allclose(O1, O1.T), "O1 not Hermitian!"

    # Initialize algebra with generators
    generators = [('O0', O0), ('O1', O1)]
    algebra_vecs = [herm_to_real_vec(O0), herm_to_real_vec(O1)]

    def current_rank():
        mat = np.array(algebra_vecs)
        return np.linalg.matrix_rank(mat, tol=1e-10)

    rank = current_rank()
    max_rank = D * D  # real dimension of Hermitian D×D matrices

    print(f"  D = {D}, max Hermitian algebra dim = {max_rank}")
    print(f"\n  {'level':>6} {'new':>5} {'total_vecs':>11} {'rank':>6}")
    print(f"  {'-'*35}")
    print(f"  {'init':>6} {'':>5} {len(algebra_vecs):>11} {rank:>6}")

    # Track all Hermitian operators generated so far
    all_ops = [O0, O1]
    current_level_ops = [O0, O1]

    for level in range(1, 12):
        new_ops = []

        # Bracket each current-level op with all generators
        for op in current_level_ops:
            for gen in [O0, O1]:
                hb = hbracket(op, gen)

                # Normalize
                norm = np.linalg.norm(hb)
                if norm < 1e-14:
                    continue

                vec = herm_to_real_vec(hb)
                test_vecs = algebra_vecs + [vec]
                test_mat = np.array(test_vecs)
                new_rank = np.linalg.matrix_rank(test_mat, tol=1e-10)

                if new_rank > rank:
                    algebra_vecs.append(vec)
                    all_ops.append(hb)
                    new_ops.append(hb)
                    rank = new_rank

        print(f"  {level:>6} {len(new_ops):>5} {len(algebra_vecs):>11} {rank:>6}")

        if len(new_ops) == 0:
            print(f"\n  >>> ALGEBRA CLOSED at level {level}! <<<")
            print(f"  >>> Hermitian algebra dimension = {rank} <<<")
            break

        current_level_ops = new_ops

        if rank >= max_rank * 0.95:
            print(f"\n  >>> Approaching full Hermitian algebra ({max_rank}) <<<")
            break

    print(f"\n  Final algebra dimension: {rank}")
    print(f"  D² (full Hermitian) = {max_rank}")
    print(f"  Ratio: {rank / max_rank:.6f}")
    print(f"  su(2) Hermitian dim: 3")
    print(f"  su(D) Hermitian dim: {D*D - 1}")

    # Check: is it su(k) for some k?
    for k in range(2, D+1):
        if rank == k*k - 1:
            print(f"  → Matches su({k}) dimension!")
            break
    else:
        if rank == max_rank:
            print(f"  → Full Hermitian matrix algebra")
        else:
            print(f"  → Novel algebra of dimension {rank}")

    return rank


# ============================================================
# TEST 3: DEGENERACY + OFF-BLOCK + TRACE MOMENTS
# ============================================================

def test3_degeneracy(L_list=[3, 4, 5, 6, 8, 10]):
    """
    Corrected:
    - Check off-block norms ||P_m iC P_n|| before interpreting m-sector spectra.
    - Add trace moment diagnostics: Tr(C^2), Tr(C^4), Tr(C^6).
    """
    print("\n" + "=" * 65)
    print("  TEST 3: DEGENERACY + OFF-BLOCK + TRACE MOMENTS")
    print("=" * 65)

    all_spectra = {}

    for L in L_list:
        print(f"\n{'─'*60}")
        print(f"  L = {L}")
        print(f"{'─'*60}")

        basis = build_ladder_basis(L)
        D = len(basis)
        O0 = build_projected_X_sparse(basis, L, 0).toarray()
        O1 = build_projected_X_sparse(basis, L, 1).toarray()
        comm = O0 @ O1 - O1 @ O0
        iC = 1j * comm
        iC = 0.5 * (iC + iC.conj().T)  # Hermitian
        C = iC  # C = i[O0,O1], Hermitian

        eigvals = np.linalg.eigvalsh(C)
        rounded = np.round(eigvals).astype(int)
        unique, counts = np.unique(rounded, return_counts=True)
        spectrum = {int(u): int(c) for u, c in zip(unique, counts)}
        all_spectra[L] = spectrum

        j_max = L // 2
        deg_seq = [spectrum.get(m, 0) for m in range(-j_max, j_max+1)]
        print(f"  D={D}, j_max={j_max}")
        print(f"  Degeneracy: {deg_seq}")

        # ============================================================
        # OFF-BLOCK NORM ANALYSIS
        # ============================================================
        # Basis m-sectors: group basis states by m = sum(n0_i - n1_i)
        m_indices = {}
        for idx, b in enumerate(basis):
            m_val = sum(b[2*i] - b[2*i+1] for i in range(L))
            if m_val not in m_indices:
                m_indices[m_val] = []
            m_indices[m_val].append(idx)

        m_vals = sorted(m_indices.keys())

        # Build projectors P_m and compute ||P_m C P_n||
        print(f"\n  --- OFF-BLOCK NORMS ||P_m C P_n||_F ---")
        print(f"  (Should be nonzero if C mixes basis m-sectors)")

        block_norms = {}
        for m in m_vals:
            for n in m_vals:
                # Extract block C[m_indices, n_indices]
                C_block = C[np.ix_(m_indices[m], m_indices[n])]
                block_norms[(m, n)] = np.linalg.norm(C_block)

        # Print as matrix
        if len(m_vals) <= 11:
            header = "  m\\n " + " ".join(f"{n:>7d}" for n in m_vals)
            print(header)
            for m in m_vals:
                row = f"  {m:>3d}  " + " ".join(
                    f"{block_norms[(m,n)]:>7.3f}" for n in m_vals)
                print(row)

        # Which off-diagonal blocks are nonzero?
        n_nonzero_blocks = sum(1 for (m,n), v in block_norms.items()
                              if m != n and v > 1e-10)
        n_total_off = len(m_vals) * (len(m_vals) - 1)
        print(f"\n  Nonzero off-diagonal blocks: {n_nonzero_blocks} / {n_total_off}")

        # What is the block structure? Which Δm are connected?
        delta_m_norms = {}
        for m in m_vals:
            for n in m_vals:
                if m != n:
                    dm = n - m
                    if dm not in delta_m_norms:
                        delta_m_norms[dm] = []
                    delta_m_norms[dm].append(block_norms[(m, n)])

        print(f"\n  Average ||block|| by Δm:")
        for dm in sorted(delta_m_norms.keys()):
            avg = np.mean(delta_m_norms[dm])
            mx = np.max(delta_m_norms[dm])
            if avg > 1e-10:
                print(f"    Δm = {dm:>3d}: avg = {avg:.4f}, max = {mx:.4f}")

        # ============================================================
        # TRACE MOMENT DIAGNOSTICS
        # ============================================================
        print(f"\n  --- TRACE MOMENTS ---")

        # C = i[O0,O1] (Hermitian), eigenvalues are integers
        # Tr(C^{2k}) = sum_j N_j * j^{2k}
        C2 = C @ C
        C4 = C2 @ C2
        C6 = C4 @ C2

        tr2 = np.real(np.trace(C2))
        tr4 = np.real(np.trace(C4))
        tr6 = np.real(np.trace(C6))

        # Predicted from degeneracy
        pred_tr2 = sum(spectrum.get(m, 0) * m**2 for m in range(-j_max, j_max+1))
        pred_tr4 = sum(spectrum.get(m, 0) * m**4 for m in range(-j_max, j_max+1))
        pred_tr6 = sum(spectrum.get(m, 0) * m**6 for m in range(-j_max, j_max+1))

        print(f"  Tr(C²) = {tr2:.2f}  (from deg: {pred_tr2})")
        print(f"  Tr(C⁴) = {tr4:.2f}  (from deg: {pred_tr4})")
        print(f"  Tr(C⁶) = {tr6:.2f}  (from deg: {pred_tr6})")
        print(f"  Tr(C²)/D = {tr2/D:.4f}")
        print(f"  Tr(C⁴)/Tr(C²)² = {tr4/tr2**2:.6f}")

        # Odd moments (should be zero by symmetry)
        C3 = C2 @ C
        tr1 = np.real(np.trace(C))
        tr3 = np.real(np.trace(C3))
        print(f"  Tr(C) = {tr1:.2e}  (should be 0)")
        print(f"  Tr(C³) = {tr3:.2e}  (should be 0)")

        # ============================================================
        # LEAKAGE LOOP INTERPRETATION
        # ============================================================
        # Tr(C²) = Tr((i[O0,O1])²) = -Tr([O0,O1]²)
        # = -Tr((O0 O1 - O1 O0)²)
        # = -Tr(O0 O1 O0 O1 - O0 O1² O0 - O1 O0² O1 + O1 O0 O1 O0)
        # Each O_l = P X_l P: so Tr(C²) counts length-4 closed paths
        # through the constraint graph.
        print(f"\n  --- LEAKAGE LOOP INTERPRETATION ---")
        print(f"  Tr(C^{{2k}}) counts closed oriented leakage loops of length 4k")
        print(f"  Tr(C²) / L = {tr2/L:.4f}  (per-rung loop density)")
        print(f"  Tr(C²) / L² = {tr2/L**2:.4f}")

        # ============================================================
        # DEGENERACY CROSS-CHECK
        # ============================================================
        # Basis m-sector count
        basis_m_count = {m: len(indices) for m, indices in m_indices.items()}
        basis_seq = [basis_m_count.get(m, 0) for m in range(-j_max, j_max+1)]
        print(f"\n  --- DEGENERACY ---")
        print(f"  Commutator:  {deg_seq}")
        print(f"  Basis m-sec: {basis_seq}")
        print(f"  Difference:  {[d-b for d,b in zip(deg_seq, basis_seq)]}")

        # d(j_max) analysis
        d_jmax = spectrum.get(j_max, 0)
        d_neg_jmax = spectrum.get(-j_max, 0)
        print(f"  d(j_max={j_max}) = {d_jmax}, d(-j_max) = {d_neg_jmax}")

    # ============================================================
    # CROSS-L TRACE MOMENT SCALING
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  CROSS-L TRACE MOMENT SCALING")
    print(f"{'='*65}")
    print(f"\n  {'L':>3} {'D':>7} {'j_max':>6} {'Tr(C²)':>12} {'Tr(C²)/L²':>12} "
          f"{'Tr(C⁴)/Tr(C²)²':>16} {'d(j_max)':>9}")
    print(f"  {'-'*68}")

    for L in L_list:
        basis = build_ladder_basis(L)
        D = len(basis)
        j_max = L // 2
        spec = all_spectra[L]
        tr2 = sum(spec.get(m, 0) * m**2 for m in range(-j_max, j_max+1))
        tr4 = sum(spec.get(m, 0) * m**4 for m in range(-j_max, j_max+1))
        d_jmax = spec.get(j_max, 0)
        print(f"  {L:3d} {D:7d} {j_max:6d} {tr2:12.0f} {tr2/L**2:12.4f} "
              f"{tr4/tr2**2 if tr2 > 0 else 0:16.6f} {d_jmax:>9d}")

    # ============================================================
    # L=10 DEEP DEGENERACY
    # ============================================================
    if 10 in all_spectra:
        print(f"\n{'='*65}")
        print(f"  L=10 DEEP DEGENERACY ANALYSIS")
        print(f"{'='*65}")
        spec = all_spectra[10]
        seq = [spec.get(m, 0) for m in range(-5, 6)]
        print(f"  Sequence: {seq}")

        # Ratios d(m)/C(2j,j+m) — is there a transfer-matrix formula?
        binom = [int(comb(10, 5+m, exact=True)) for m in range(-5, 6)]
        print(f"  C(10,5+m):  {binom}")
        ratios = [seq[i]/binom[i] if binom[i] > 0 else 0 for i in range(11)]
        print(f"  d(m)/C(10,5+m): {[f'{r:.4f}' for r in ratios]}")

        # Check: are ratios related to Catalan, Motzkin, or Riordan numbers?
        # d(±5) = 2: there are exactly 2 states at max polarization
        print(f"\n  d(5) = d(-5) = 2")
        print(f"  → 2 maximally-polarized states (likely two Néel-like configs)")

        # Check generating function structure
        # Define F(x) = sum_{m} d(m) x^m
        # F(1) = D = 6727
        poly = np.array([spec.get(m, 0) for m in range(-5, 6)])
        print(f"\n  F(x) = sum d(m) x^m as polynomial:")
        print(f"  F(1) = {sum(poly)} (= D)")
        print(f"  F(-1) = {sum((-1)**i * poly[i] for i in range(11))}")

        # Check: Tr(C^2) from moments
        tr2_from_spec = sum(spec.get(m,0) * m**2 for m in range(-5,6))
        tr4_from_spec = sum(spec.get(m,0) * m**4 for m in range(-5,6))
        tr6_from_spec = sum(spec.get(m,0) * m**6 for m in range(-5,6))
        print(f"\n  Tr(C²) = {tr2_from_spec}")
        print(f"  Tr(C⁴) = {tr4_from_spec}")
        print(f"  Tr(C⁶) = {tr6_from_spec}")
        print(f"  Tr(C²)/D = {tr2_from_spec/6727:.6f}")

    return all_spectra


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  EP4 DEEP STRUCTURE v2 — LEO'S CORRECTIONS")
    print("=" * 65)

    # Test 1: Leakage (L=4 and L=6 only — dense full-space)
    test1_leakage(4)
    print()
    test1_leakage(6)

    # Test 2: Algebra growth (Hermitian bracket)
    for L_alg in [3, 4, 5, 6]:
        test2_algebra_growth(L_alg)

    # Test 3: Degeneracy + off-block + moments
    test3_degeneracy([3, 4, 5, 6, 8, 10])

    print("\n" + "=" * 65)
    print("  ALL TESTS COMPLETE (v2)")
    print("=" * 65)


if __name__ == "__main__":
    main()
