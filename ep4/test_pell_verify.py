#!/usr/bin/env python3
"""
EP4 Pell-Lucas Verification
==============================
Leo's prediction:
  Tr(C²)/L follows recurrence a_L = 2*a_{L-1} + a_{L-2}
  Known: L=3:2, L=4:6, L=5:14, L=6:34, L=8:198, L=10:1154

  Prediction:
    L=7:  Tr(C²)/L = 82  → Tr(C²) = 574
    L=9:  Tr(C²)/L = 478 → Tr(C²) = 4302

  Also: d(j_max) = L if L odd, 2 if L even.

Quick targeted test: L=7 and L=9 only.
"""

import numpy as np
from scipy.sparse import csr_matrix
from time import time as now

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


def test_single_L(L):
    """Compute Tr(C²), Tr(C⁴), degeneracy for a single L."""
    t0 = now()
    basis = build_ladder_basis(L)
    D = len(basis)
    j_max = L // 2
    print(f"\n  L = {L}, D = {D}, j_max = {j_max}")

    O0 = build_projected_X_sparse(basis, L, 0).toarray()
    O1 = build_projected_X_sparse(basis, L, 1).toarray()
    comm = O0 @ O1 - O1 @ O0
    iC = 1j * comm
    iC = 0.5 * (iC + iC.conj().T)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(iC)
    rounded = np.round(eigvals).astype(int)
    max_dev = np.max(np.abs(eigvals - rounded))

    unique, counts = np.unique(rounded, return_counts=True)
    spectrum = {int(u): int(c) for u, c in zip(unique, counts)}

    # Degeneracy sequence
    deg_seq = [spectrum.get(m, 0) for m in range(-j_max, j_max+1)]

    # Trace moments (from eigenvalues, exact for integers)
    tr2 = int(np.round(np.sum(eigvals**2)))
    tr4 = int(np.round(np.sum(eigvals**4)))
    tr6 = int(np.round(np.sum(eigvals**6)))

    # Also direct matrix trace for verification
    C2 = iC @ iC
    tr2_direct = int(np.round(np.real(np.trace(C2))))

    elapsed = now() - t0

    print(f"  Max eigenvalue dev from integer: {max_dev:.2e}")
    print(f"  Degeneracy: {deg_seq}")
    print(f"  d(j_max={j_max}) = {spectrum.get(j_max, 0)}")
    print(f"  d(-j_max={-j_max}) = {spectrum.get(-j_max, 0)}")
    print(f"  Tr(C²) = {tr2}  (direct: {tr2_direct})")
    print(f"  Tr(C²)/L = {tr2/L:.4f}")
    print(f"  Tr(C⁴) = {tr4}")
    print(f"  Tr(C⁶) = {tr6}")
    print(f"  Time: {elapsed:.1f}s")

    return L, D, j_max, tr2, spectrum, deg_seq


def main():
    print("=" * 65)
    print("  PELL-LUCAS RECURRENCE VERIFICATION")
    print("=" * 65)

    # Leo's prediction table
    print("\n  Leo's Pell-Lucas prediction: a_L = 2*a_{L-1} + a_{L-2}")
    print("  Known Tr(C²)/L values:")
    known = {3: 2, 4: 6, 5: 14, 6: 34, 8: 198, 10: 1154}
    for L, val in sorted(known.items()):
        print(f"    L={L}: Tr(C²)/L = {val}, Tr(C²) = {val*L}")

    # Predicted
    print("\n  Predictions:")
    print(f"    L=7:  Tr(C²)/L = 82   → Tr(C²) = 574")
    print(f"    L=9:  Tr(C²)/L = 478  → Tr(C²) = 4302")
    print(f"\n  d(j_max) prediction:")
    print(f"    L odd:  d(j_max) = L")
    print(f"    L even: d(j_max) = 2")

    # Run L=7
    print("\n" + "=" * 65)
    print("  TESTING L = 7")
    print("=" * 65)
    L7, D7, j7, tr2_7, spec7, deg7 = test_single_L(7)

    # Run L=9
    print("\n" + "=" * 65)
    print("  TESTING L = 9")
    print("=" * 65)
    L9, D9, j9, tr2_9, spec9, deg9 = test_single_L(9)

    # ============================================================
    # VERIFICATION
    # ============================================================
    print("\n" + "=" * 65)
    print("  VERIFICATION RESULTS")
    print("=" * 65)

    # Pell-Lucas check
    pred_7 = 574
    pred_9 = 4302
    print(f"\n  --- Tr(C²) PELL-LUCAS CHECK ---")
    print(f"  L=7: predicted Tr(C²) = {pred_7}, actual = {tr2_7}")
    print(f"        MATCH: {'YES !!!' if tr2_7 == pred_7 else 'NO — actual/predicted = %.6f' % (tr2_7/pred_7)}")
    print(f"  L=9: predicted Tr(C²) = {pred_9}, actual = {tr2_9}")
    print(f"        MATCH: {'YES !!!' if tr2_9 == pred_9 else 'NO — actual/predicted = %.6f' % (tr2_9/pred_9)}")

    # d(j_max) check
    d_jmax_7 = spec7.get(j7, 0)
    d_jmax_9 = spec9.get(j9, 0)
    print(f"\n  --- d(j_max) PARITY CHECK ---")
    print(f"  L=7 (odd):  predicted d(j_max=3) = 7, actual = {d_jmax_7}")
    print(f"              MATCH: {'YES !!!' if d_jmax_7 == 7 else 'NO'}")
    print(f"  L=9 (odd):  predicted d(j_max=4) = 9, actual = {d_jmax_9}")
    print(f"              MATCH: {'YES !!!' if d_jmax_9 == 9 else 'NO'}")

    # Full Tr(C²)/L table
    print(f"\n  --- COMPLETE Tr(C²)/L TABLE ---")
    all_data = {**known}
    all_data[7] = tr2_7 // 7
    all_data[9] = tr2_9 // 9
    print(f"  {'L':>3} {'Tr(C²)':>10} {'Tr(C²)/L':>10} {'Pell pred':>10} {'match':>6}")
    print(f"  {'-'*44}")

    # Build full Pell sequence: a_1=?, a_2=?, a_L = 2*a_{L-1} + a_{L-2}
    # From known: a_3=2, a_4=6
    # Back-calculate: a_3 = 2*a_2 + a_1 → need a_1, a_2
    # a_4 = 2*a_3 + a_2 = 4 + a_2 = 6 → a_2 = 2
    # a_3 = 2*a_2 + a_1 = 4 + a_1 = 2 → a_1 = -2
    # Or maybe the recurrence starts at L=3?
    # a_3=2, a_4=6: a_4 = 2*a_3 + a_2 → a_2 = 6 - 4 = 2
    # a_5 = 2*a_4 + a_3 = 12 + 2 = 14 ✓
    # a_6 = 2*a_5 + a_4 = 28 + 6 = 34 ✓
    # a_7 = 2*a_6 + a_5 = 68 + 14 = 82
    # a_8 = 2*a_7 + a_6 = 164 + 34 = 198 ✓
    # a_9 = 2*a_8 + a_7 = 396 + 82 = 478
    # a_10 = 2*a_9 + a_8 = 956 + 198 = 1154 ✓

    pell = {2: 2, 3: 2}  # seed: a_2=2, a_3=2
    for n in range(4, 13):
        pell[n] = 2 * pell[n-1] + pell[n-2]

    for L in sorted(all_data.keys()):
        tr2_val = all_data[L] * L if L in known else (tr2_7 if L==7 else tr2_9)
        ratio = tr2_val / L
        pell_pred = pell.get(L, '?')
        match = '✓' if isinstance(pell_pred, int) and abs(ratio - pell_pred) < 0.5 else '✗'
        print(f"  {L:3d} {tr2_val:10d} {ratio:10.1f} {str(pell_pred):>10} {match:>6}")

    # Predict L=11, 12
    print(f"\n  --- PREDICTIONS ---")
    for L_pred in [11, 12]:
        pred_ratio = pell[L_pred]
        pred_tr2 = pred_ratio * L_pred
        d_pred = L_pred if L_pred % 2 == 1 else 2
        print(f"  L={L_pred}: Tr(C²)/L = {pred_ratio}, Tr(C²) = {pred_tr2}, d(j_max) = {d_pred}")

    print(f"\n  Recurrence: a_L = 2*a_{{L-1}} + a_{{L-2}}")
    print(f"  Seed: a_2 = 2, a_3 = 2")
    print(f"  This is a Companion Pell sequence.")

    # Check: standard Pell numbers P_n: 0,1,2,5,12,29,70,169,408,985,...
    # Half-companion Pell H_n: 1,1,3,7,17,41,99,239,577,1393,...
    # Our sequence: 2,2,6,14,34,82,198,478,1154,...
    # = 2 * (1,1,3,7,17,41,99,239,577,...) = 2 * H_n
    # H_n = half-companion Pell numbers!
    print(f"\n  Our sequence / 2: {[pell[n]//2 for n in range(2, 12)]}")
    print(f"  Half-companion Pell H_n: 1, 1, 3, 7, 17, 41, 99, 239, 577, 1393")
    half_pell_match = all(pell[n] == 2 * h for n, h in
                         zip(range(2, 11), [1, 1, 3, 7, 17, 41, 99, 239, 577]))
    print(f"  a_L = 2 * H_{{L-1}}: {half_pell_match}")

    if half_pell_match:
        print(f"\n  >>> Tr(C²) / L = 2 * H_{{L-1}} <<<")
        print(f"  >>> Tr(C²) = 2L * H_{{L-1}} <<<")
        print(f"  >>> where H_n is the half-companion Pell sequence <<<")
        print(f"  >>> H_n = (1+√2)^n + (1-√2)^n) / 2 <<<")

    print(f"\n{'='*65}")
    print(f"  TEST COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
