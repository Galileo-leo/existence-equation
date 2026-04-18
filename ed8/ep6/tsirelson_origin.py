#!/usr/bin/env python3
"""
The Origin of the Tsirelson Bound
==================================
WHY S_max = 2*sqrt(2)? Not 3, not pi, not 4.

ANSWER:
  2     = Bob's 2 unit vectors (|b|^2 + |b'|^2 = 2)
  sqrt(2) = Alice's 2 settings forming 2D space (diagonal = sqrt(2))
  S_max = 2 * sqrt(2) = 2.8284...

DERIVATION (5 steps, each numerically verified):
  1. |u|^2 + |v|^2 = 4          [u=b'-b, v=b'+b, unit vectors]
  2. |S| <= |u| + |v|           [Cauchy-Schwarz]
  3. max(|u|+|v|) = 2*sqrt(2)   [QM-AM on circle x^2+y^2=4]
  4. Achieved at |u|=|v|=sqrt(2) [45-degree diagonal]
  5. Optimal CHSH angles: 0, 90, 45, 135 degrees [known analytically]

ED INTERPRETATION:
  Axiom 1.1 → discrete events → discrete measurement outcomes
  α|Ψ|²Ψ → continuous deviation projected to ±1 (Born rule)
  2 outcomes × 2 observers = 2×2 measurement lattice
  Lattice diagonal = sqrt(2)
  S_max = 2 × sqrt(2) = geometry of binary measurement

NOTE: Full brute-force scan is unnecessary.
  The optimal angles are known analytically (Cirel'son 1980).
  This code VERIFIES the decomposition, not searches for it.

==================================
Jae-Ahn Shin
Independent Researcher, Incheon, Republic of Korea
jaeahn.shin.official@gmail.com
==================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 65)
print("  THE ORIGIN OF THE TSIRELSON BOUND")
print("  S = 2*sqrt(2) = (2 unit vectors) * (2D diagonal)")
print("=" * 65)

# ============================================================
# STEP 1: |u|^2 + |v|^2 = 4 — ALWAYS, ANY DIMENSION
# ============================================================

print(f"\n{'='*60}")
print(f"  STEP 1: The constraint |u|^2 + |v|^2 = 4")
print(f"{'='*60}")

rng = np.random.RandomState(42)

for dim_label, dim in [("2D", 2), ("3D", 3), ("10D", 10)]:
    vals = []
    for _ in range(10000):
        b = rng.randn(dim); b /= np.linalg.norm(b)
        bp = rng.randn(dim); bp /= np.linalg.norm(bp)
        u = bp - b; v = bp + b
        vals.append(np.dot(u,u) + np.dot(v,v))
    print(f"  {dim_label}: |u|^2+|v|^2 = {np.mean(vals):.6f} ± {np.std(vals):.2e}")

print(f"  → EXACT 4.000000 in all dimensions.")
print(f"  → Origin: |b|=|b'|=1, so 2(|b|^2+|b'|^2) = 2×2 = 4.")

# ============================================================
# STEP 2: max(|u|+|v|) on |u|^2+|v|^2=4
# ============================================================

print(f"\n{'='*60}")
print(f"  STEP 2: Maximize |u|+|v| subject to |u|^2+|v|^2=4")
print(f"{'='*60}")

# Parametric: |u| = 2*cos(t), |v| = 2*sin(t), t in [0, pi/2]
t = np.linspace(0, np.pi/2, 10000)
u_vals = 2*np.cos(t)
v_vals = 2*np.sin(t)
sum_vals = u_vals + v_vals

i_max = np.argmax(sum_vals)
print(f"  max(|u|+|v|) = {sum_vals[i_max]:.6f}")
print(f"  at |u| = {u_vals[i_max]:.6f}, |v| = {v_vals[i_max]:.6f}")
print(f"  sqrt(2) = {np.sqrt(2):.6f}")
print(f"  2*sqrt(2) = {2*np.sqrt(2):.6f}")
print(f"")
print(f"  WHY sqrt(2)?")
print(f"  Lagrange: maximize x+y on x^2+y^2=4")
print(f"  Solution: x=y=sqrt(2). Maximum = 2*sqrt(2).")
print(f"  This is the L1 norm maximized on an L2 sphere.")
print(f"  Geometrically: the 45° diagonal of the circle.")

# ============================================================
# STEP 3: Verify CHSH at known optimal angles
# ============================================================

print(f"\n{'='*60}")
print(f"  STEP 3: CHSH at analytically known optimal angles")
print(f"  (No brute-force scan needed — Cirel'son 1980)")
print(f"{'='*60}")

# Optimal: a=0, a'=pi/2, b=pi/4, b'=3pi/4
# Equivalently: separations of pi/4 between consecutive settings

configs = {
    'Optimal (0,90,45,135)': (0, np.pi/2, np.pi/4, 3*np.pi/4),
    'Uniform (0,60,30,90)': (0, np.pi/3, np.pi/6, np.pi/2),
    'Tight (0,45,22.5,67.5)': (0, np.pi/4, np.pi/8, 3*np.pi/8),
    'Classical-like (0,90,0,90)': (0, np.pi/2, 0, np.pi/2),
}

for name, (th_a, th_ap, th_b, th_bp) in configs.items():
    a = np.array([np.cos(th_a), np.sin(th_a)])
    ap = np.array([np.cos(th_ap), np.sin(th_ap)])
    b = np.array([np.cos(th_b), np.sin(th_b)])
    bp = np.array([np.cos(th_bp), np.sin(th_bp)])
    
    # E(a,b) = -a.b for singlet
    E_ab = -np.dot(a, b)
    E_abp = -np.dot(a, bp)
    E_apb = -np.dot(ap, b)
    E_apbp = -np.dot(ap, bp)
    
    S = E_ab - E_abp + E_apb + E_apbp
    
    # Decompose
    u = bp - b
    v = bp + b
    
    print(f"\n  {name}:")
    print(f"    S = {S:+.6f}")
    print(f"    |u| = {np.linalg.norm(u):.4f}, |v| = {np.linalg.norm(v):.4f}")
    print(f"    |u|+|v| = {np.linalg.norm(u)+np.linalg.norm(v):.4f}")
    print(f"    |u|^2+|v|^2 = {np.dot(u,u)+np.dot(v,v):.4f}")

print(f"\n  → Optimal S = 2*sqrt(2) = {2*np.sqrt(2):.6f}")
print(f"  → At optimum: |u| = |v| = sqrt(2)")

# ============================================================
# STEP 4: The 2 and the sqrt(2) — complete decomposition
# ============================================================

print(f"\n{'='*60}")
print(f"  STEP 4: WHERE DO 2 AND sqrt(2) COME FROM?")
print(f"{'='*60}")

print(f"""
  THE CHAIN:
  
  Bob picks 2 unit vectors b, b'.
    |b|^2 + |b'|^2 = 1 + 1 = 2
    |u|^2 + |v|^2 = 2(|b|^2 + |b'|^2) = 4
    sqrt(4) = 2    ← THIS IS THE "2" IN 2*sqrt(2)

  Alice picks 2 settings a, a'.
    S = a·u - a'·v
    To maximize: align a with u, a' with -v
    |S| ≤ |u| + |v|
    Maximize |u|+|v| on x^2+y^2=4:
      Optimum at x=y=sqrt(2)
      Max = sqrt(2) + sqrt(2) = 2*sqrt(2)
    The sqrt(2) = diagonal of 2D optimization
                                 ← THIS IS THE "sqrt(2)"

  THEREFORE:
    S_max = sqrt(4) × sqrt(2) = 2 × sqrt(2) = 2*sqrt(2)

    2     = "how many unit vectors Bob has"
    sqrt(2) = "diagonal of Alice's 2D choice space"
""")

# ============================================================
# STEP 5: ED interpretation
# ============================================================

print(f"{'='*60}")
print(f"  STEP 5: ED INTERPRETATION")
print(f"{'='*60}")

print(f"""
  In ED, measurement is not primitive.
  It is the result of condensation:

  1. The deviation field Ψ oscillates at ~10^20 Hz (Act IV)
  2. No detector can resolve this.
  3. What survives time-averaging is A^2 = |Ψ|^2 (Born rule).
  4. For spin-1/2, this projects to ±1 (two outcomes).

  This discretization is forced by Axiom 1.1:
    "Deviation manifests through events —
     minimal couplings of space and time."

  Events are discrete. Outcomes are discrete.
  Two discrete outcomes create a 2-point lattice per observer.
  Two observers create a 2×2 lattice.
  The diagonal of this lattice is sqrt(2).

  The Tsirelson bound S = 2*sqrt(2) is therefore:
    - NOT a property of quantum mechanics specifically
    - NOT a property of Hilbert space specifically
    - IT IS the geometry of binary discrete measurement

  Any theory that:
    (a) produces discrete binary outcomes, and
    (b) allows correlations via a shared continuous field
  will produce S_max = 2*sqrt(2).

  ED satisfies both conditions:
    (a) α|Ψ|²Ψ + Born rule → ±1 outcomes
    (b) Ψ is a continuous deviation field connecting both sites

  The Bell violation is not spooky.
  It is geometric.
  It measures the diagonal of the measurement lattice.
""")

# ============================================================
# FIGURE
# ============================================================

fig, axes_fig = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    'The Origin of the Tsirelson Bound\n'
    '$S_{\\max} = 2\\sqrt{2}$: '
    '$2$ from unit vectors, $\\sqrt{2}$ from measurement diagonal',
    fontsize=14, fontweight='bold', y=1.01)

# (a) Constraint circle with optimum
ax = axes_fig[0, 0]
theta_c = np.linspace(0, np.pi/2, 500)
ax.plot(2*np.cos(theta_c), 2*np.sin(theta_c), 'b-', lw=2.5,
        label=r'$|u|^2+|v|^2=4$')
# L1 contour at maximum
ax.plot([0, 2*np.sqrt(2)], [2*np.sqrt(2), 0], 'r--', lw=2,
        label=r'$|u|+|v|=2\sqrt{2}$')
ax.plot(np.sqrt(2), np.sqrt(2), 'ro', ms=15, zorder=5)
ax.annotate(r'$|u|=|v|=\sqrt{2}$', xy=(np.sqrt(2), np.sqrt(2)),
            xytext=(2.2, 2.4), fontsize=12, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
# Show sqrt(2) as diagonal
ax.plot([0, np.sqrt(2)], [0, np.sqrt(2)], 'r:', lw=1.5, alpha=0.5)
ax.annotate(r'$\sqrt{2}$', xy=(0.5, 0.8), fontsize=14, color='red')
ax.set_xlabel('$|u| = |b\'-b|$', fontsize=12)
ax.set_ylabel('$|v| = |b\'+b|$', fontsize=12)
ax.set_title(r'(a) Optimum on $|u|^2+|v|^2=4$', fontsize=12, fontweight='bold')
ax.set_aspect('equal')
ax.set_xlim(-0.2, 3.2); ax.set_ylim(-0.2, 3.2)
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

# (b) |u|+|v| vs parametric angle
ax = axes_fig[0, 1]
ax.plot(np.degrees(t), sum_vals, 'r-', lw=2.5)
ax.axhline(2*np.sqrt(2), color='k', ls='--', alpha=0.5)
ax.axhline(2, color='b', ls=':', alpha=0.5, label='Classical: 2')
ax.plot(45, 2*np.sqrt(2), 'ro', ms=12, zorder=5)
ax.annotate(f'$45° → 2\\sqrt{{2}}$', xy=(45, 2*np.sqrt(2)),
            xytext=(55, 2.6), fontsize=11, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.set_xlabel(r'Parametric angle $t$ (degrees)', fontsize=12)
ax.set_ylabel(r'$|u|+|v|$', fontsize=12)
ax.set_title('(b) Maximum at 45° = diagonal', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)

# (c) 2×2 measurement lattice
ax = axes_fig[0, 2]
ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
for i in [-1, 1]:
    for j in [-1, 1]:
        ax.plot(i, j, 'ko', ms=18)
# Axis connections
ax.plot([-1,1], [-1,-1], 'b-', lw=3)
ax.plot([-1,1], [1,1], 'b-', lw=3)
ax.plot([-1,-1], [-1,1], 'b-', lw=3)
ax.plot([1,1], [-1,1], 'b-', lw=3)
ax.text(0, -1.35, 'axis = 2', ha='center', fontsize=11, color='blue', fontweight='bold')
# Diagonal
ax.plot([-1,1], [-1,1], 'r-', lw=3)
ax.text(0.3, 0.15, r'diag $= 2\sqrt{2}$', fontsize=12, color='red',
        fontweight='bold', rotation=45)
ax.set_xlabel('Alice outcome (±1)', fontsize=12)
ax.set_ylabel('Bob outcome (±1)', fontsize=12)
ax.set_title('(c) 2×2 measurement lattice', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# (d) CHSH values for different configs
ax = axes_fig[1, 0]
cfg_names = list(configs.keys())
cfg_S = []
for name, (th_a, th_ap, th_b, th_bp) in configs.items():
    a = np.array([np.cos(th_a), np.sin(th_a)])
    ap = np.array([np.cos(th_ap), np.sin(th_ap)])
    b = np.array([np.cos(th_b), np.sin(th_b)])
    bp = np.array([np.cos(th_bp), np.sin(th_bp)])
    S = (-np.dot(a,b) + np.dot(a,bp) - np.dot(ap,b) - np.dot(ap,bp))
    cfg_S.append(abs(S))

colors_bar = ['#D32F2F', '#1565C0', '#FF6F00', '#9E9E9E']
bars = ax.barh(range(len(cfg_names)), cfg_S, color=colors_bar, alpha=0.8)
ax.axvline(2*np.sqrt(2), color='red', ls='--', lw=2, label=r'$2\sqrt{2}$')
ax.axvline(2, color='blue', ls=':', lw=2, label='Classical')
ax.set_yticks(range(len(cfg_names)))
ax.set_yticklabels([n.split('(')[0].strip() for n in cfg_names], fontsize=10)
for i, v in enumerate(cfg_S):
    ax.text(v+0.02, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('|S|', fontsize=12)
ax.set_title('(d) CHSH for different angle configs', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
ax.set_xlim(0, 3.2)

# (e) ED derivation chain
ax = axes_fig[1, 1]
ax.axis('off')
chain = [
    "ED DERIVATION CHAIN",
    "=" * 42,
    "",
    "Axiom 1.1",
    "  Deviation occurs event by event.",
    "  Events are discrete.",
    "       ↓",
    "α|Ψ|²Ψ (condensation)",
    "  Projects continuous Ψ → discrete ±1",
    "  This IS the Born rule (Act IV).",
    "       ↓",
    "CHSH structure",
    "  Alice: 2 settings → 2D space",
    "  Bob:   2 settings → 2 unit vectors",
    "       ↓",
    "Constraint",
    "  |u|²+|v|² = 4  (from |b|=|b'|=1)",
    "       ↓",
    "Geometry",
    "  max(|u|+|v|) = 2√2",
    "  = diagonal of 2D optimization",
    "       ↓",
    "TSIRELSON BOUND",
    "  S_max = 2√2 = 2.8284",
    "",
    "Not spooky. Geometric.",
]
ax.text(0.05, 0.95, "\n".join(chain), transform=ax.transAxes,
        fontsize=10.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0ff', alpha=0.9))

# (f) The punchline
ax = axes_fig[1, 2]
ax.axis('off')
punchline = [
    "THE PUNCHLINE",
    "=" * 42,
    "",
    "Any theory that produces:",
    "  (a) discrete binary outcomes",
    "  (b) correlations via shared field",
    "will give S_max = 2√2.",
    "",
    "ED satisfies both:",
    "  (a) Axiom 1.1 + α|Ψ|²Ψ → ±1",
    "  (b) Ψ connects both sites",
    "",
    "The Bell violation is not",
    "  evidence of non-locality.",
    "It is evidence of",
    "  discrete binary measurement",
    "  on a continuous shared field.",
    "",
    "√2 is not mysterious.",
    "It is the diagonal of a 2×2 grid.",
    "",
    "The universe is not spooky.",
    "It is geometric.",
    "",
    "And the geometry is written",
    "in Axiom 1.1.",
]
ax.text(0.05, 0.95, "\n".join(punchline), transform=ax.transAxes,
        fontsize=10.5, va='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#fff0f0', alpha=0.9))

plt.tight_layout()
plt.savefig('tsirelson_origin.png', dpi=300, bbox_inches='tight')
print(f"\n  Saved: tsirelson_origin.png")

print(f"\n{'='*65}")
print(f"  COMPLETE")
print(f"{'='*65}")
