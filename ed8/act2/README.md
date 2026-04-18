# ED Gravity Simulations — ed8/act2/

## Existence Dynamics (ED) Gravity Sector

**Equation:**  `(1/c²)Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ`

**Author:** Jae-Ahn Shin · 2026

No Einstein field equations, no Christoffel symbols, no metric tensor are
postulated.  Gravity emerges as elastic distortion of the ED field.

---

## Python Sources (5 files)

### 1. `gravity_static_paper.py` — Static Gravity from Elasticity

3D periodic elastic lattice.  Displacement field `u` on each site produces
finite-difference curvature (Riemann, Ricci, Einstein tensors `G_ab`)
using the same algebraic definitions as differential geometry, but computed
from lattice strain — not imported from GR.

- Deformations tested: point-mass compression, vortex, dipole, uniform (control)
- Checks: tensor symmetries, contracted Bianchi identity `∇·G ≈ 0`
- Alignment of Einstein tensor `G` with Cauchy stress `T` (cosine, κ, residual)
- **Output:** `gravity_static_result.png` (300 dpi, 4-panel)
- **Key result:** G_ab = κ T_ab emerges from Hooke's Law — no GR postulate

### 2. `gravity_kerr_paper.py` — Kerr-like Geometry from ED Vortex

Axisymmetric vortex solution of the static ED equation in spherical coordinates.
The full complex field is `Ψ = A(r,θ) exp(i n φ)` with integer winding `n`.

- Winding `n=0` → Schwarzschild-like; `n=1` → Kerr-like
- Frame-dragging proxy ~ 1/r³ from phase gradient
- Winding-linked angular-momentum ratios
- **Output:** `gravity_kerr_result.png` (300 dpi)
- **Key result:** Kerr geometry emerges from phase winding, not metric postulate

### 3. `ed_binary_vortex.py` — Binary Vortex Merger + Ringdown

Two opposite-winding topological defects (n=+1, n=−1) on a 3D grid.
Inspiral → merger (winding cancellation) → post-merger ringdown.

This is the core dynamic gravity simulation.  A single master switch
`EXPAND` controls two modes:

| Mode | `EXPAND` | Physics | Post-merger behavior |
|------|----------|---------|---------------------|
| Static | `False` | a(t) = 1 (fixed dx) | Permanent breather oscillation |
| Expanding | `True` | a(t) = 1 + R·(t−t₀) | Damped ringdown → evaporation |

Cosmic expansion uses the same technique as Act III (baryon formation)
and EP V (galaxy rotation curves): **only dx changes, the equation is untouched.**

#### How to run

**Both modes must be run to produce the comparison figure:**

```python
%cd /content/drive/MyDrive/existence_equation/ed8/act2

# Run 1: Static universe (EXPAND = False)
# Edit ed_binary_vortex.py → set EXPAND = False
%run ed_binary_vortex.py
# → Produces: binary_noexpand.json, ringdown_noexpand.txt

# Run 2: Expanding universe (EXPAND = True)
# Edit ed_binary_vortex.py → set EXPAND = True
%run ed_binary_vortex.py
# → Produces: binary_expand.json, ringdown_expand.txt

# Run 3: Comparison figure (requires both JSONs)
%run plot_binary_comparison.py
# → Produces: ed_binary_vortex_result.png
```

- **Output (run 1):** `binary_noexpand.json`, `ringdown_noexpand.txt`
- **Output (run 2):** `binary_expand.json`, `ringdown_expand.txt`
- **Output (run 3):** `ed_binary_vortex_result.png` (5-panel, 200 dpi)
- **Key result:** Static universe → permanent oscillation; expanding universe → damped ringdown.  The difference is purely from dx(t) = dx₀·a(t).

### 4. `plot_binary_comparison.py` — Publication Figure Generator

Reads the two JSON files produced by `ed_binary_vortex.py` and generates
a 5-panel comparison figure for the paper:

| Panel | Content |
|-------|---------|
| (a) | Inspiral → merger (separation vs time) |
| (b) | Scale factor a(t) |
| (c) | Static universe: permanent breather oscillation |
| (d) | Expanding universe: damped ringdown → evaporation |
| (e) | Overlay comparison of both modes |

**Requires:** `binary_noexpand.json` + `binary_expand.json` in the same directory.

### 5. `ed_lensing_v5_redshift_polarization.py` — Gravitational Lensing

2D complex scalar field with a localized amplitude defect (mass).
A continuous plane wave is driven from the left.  From a single simulation
run, three classic GR tests are diagnosed:

- **(A) Gravitational redshift:** local wavenumber k_local = |∇φ| ratio
- **(B) Wavefront deflection:** propagation angle δθ near mass
- **(C) Shapiro delay:** group velocity variation through the defect region
- **Output:** `gravity_lensing_v5_deflection.png`, `gravity_lensing_v5_redshift.png`, `gravity_lensing_v5_shapiro.png` (300 dpi)
- **Key result:** Three independent GR predictions from one ED simulation

---

## Execution Order (Google Colab, A100 recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/existence_equation/ed8/act2

# Step 1: Static gravity
%run gravity_static_paper.py

# Step 2: Kerr geometry
%run gravity_kerr_paper.py

# Step 3a: Binary merger — static universe
# (set EXPAND = False in ed_binary_vortex.py)
%run ed_binary_vortex.py

# Step 3b: Binary merger — expanding universe
# (set EXPAND = True in ed_binary_vortex.py)
%run ed_binary_vortex.py

# Step 3c: Comparison figure
%run plot_binary_comparison.py

# Step 4: Gravitational lensing
%run ed_lensing_v5_redshift_polarization.py
```

---

## Directory Structure

```
ed8/act2/
├── README.md                                  ← This file
├── RESULT.md                                  ← Numerical results log
│
├── gravity_static_paper.py                    ← G_ab = κ T_ab from elasticity
├── gravity_kerr_paper.py                      ← Kerr-like from ED vortex
├── ed_binary_vortex.py                        ← Binary merger (EXPAND switch)
├── plot_binary_comparison.py                  ← Paper figure from 2 JSONs
├── ed_lensing_v5_redshift_polarization.py     ← Redshift + lensing + Shapiro
│
├── binary_noexpand.json                       ← Output: static run data
├── binary_expand.json                         ← Output: expanding run data
├── ed_binary_vortex_result.png                ← Output: comparison figure
│
└── archive/                                   ← Previous versions (reference only)
    ├── ed_binary_vortex.py
    ├── ed_binary_vortex_expand.py
    ├── ed_binary_vortex_noexpand.py
    ├── ed_gravity_dynamic_ringdown_old.py
    └── ed_gravity_dynamic_ringdown_novortex.py
```

---

## What This Proves

| Phenomenon | Source | Method |
|------------|--------|--------|
| Einstein field equations | `gravity_static_paper.py` | G_ab = κ T_ab from Hooke's Law |
| Kerr geometry / frame dragging | `gravity_kerr_paper.py` | Phase winding → angular momentum |
| Binary inspiral + merger | `ed_binary_vortex.py` | Topological defect dynamics |
| Ringdown / Hawking evaporation | `ed_binary_vortex.py` | dx(t) = dx₀·a(t) expansion |
| Gravitational redshift | `ed_lensing_v5...py` | k_local variation near defect |
| Light deflection | `ed_lensing_v5...py` | Wavefront bending |
| Shapiro delay | `ed_lensing_v5...py` | Group velocity reduction |

**All from one equation.  No GR postulates.**
