# Existence Deviation (ED) — Numerical Evidence

**One equation. Four forces. No postulates.**

$$\frac{1}{c^2}\ddot{\Psi} = \nabla^2\Psi + \lambda\Psi - \alpha|\Psi|^2\Psi$$

This repository contains all numerical simulations, analysis code,
and documentation for the Existence Deviation framework.
Every result emerges from the single ED equation above — no additional
assumptions, no fitting parameters, no standard-model machinery.

**Author:** Jae-Ahn Shin
(jaeahn.shin.official@gmail.com)
Independent Researcher, Incheon, Republic of Korea

**Main Paper Concept DOI:**
[10.5281/zenodo.18639316](https://doi.org/10.5281/zenodo.18639316)

---

## Repository Structure

```
existence-equation/
├── act1/          Electromagnetism simulations
├── act2/          Gravity simulations
├── act3_4/        Strong force + Hydrogen atom + H₂ molecule simulations
│
├── axiom/         Axiom tests (discreteness, UV divergence)
│
├── ep1/           Evidence Paper I   — PXP quantum scars
├── ep2/           Evidence Paper II  — FQHE topological degeneracy + anyon braiding
├── ep3/           Evidence Paper III — Helical quantum scars
├── ep4/           Evidence Paper IV  — Non-commutativity & algebra
├── ep5/           Evidence Paper V   — Dark matter as topological phase persistence
├── ep6/           Evidence Paper VI  — Tsirelson bound (Bell inequality)
├── ep7/           Evidence Paper VII — Josephson effect (GP as low-energy ED)
│
└── README.md      This file
```

---

## Main Paper — Acts (ED Equation Simulations)

### `act1/` — Electromagnetism

ED vortex dynamics reproduce Maxwell electromagnetism:
charge quantization, Coulomb scaling, spin-1/2, SU(2) structure,
topological vector potential, and the Aharonov-Bohm effect.

| File | What it proves |
|---|---|
| `charge_constraint_paper.py` | Charge quantization Q = ±1 from phase winding; 2D Coulomb 1/r; charge conservation |
| `maxwell_vortex_test.py` | Vector potential A ~ 1/r from vortex; curl(A) = B_z; Maxwell-type EM from ED |
| `spin_constraint_paper.py` | Spin-1/2 from phase topology; SU(2) equivalence; Pauli exclusion; Zeeman splitting |
| `vortex_line_3d.py` | 3D vortex line: 1/r phase gradient, z-invariance, λ sweep (topological protection) |
| `ed_aharonov_bohm_test.py` | Aharonov-Bohm effect: phase circulation from enclosed vortex winding; charge additivity |

### `act2/` — Gravity

ED elastic distortion reproduces general relativity:
Einstein tensor, Kerr metric, gravitational lensing, binary merger ringdown.
See `act2/README.md` for details. Galaxy rotation curves → `ep5/`.

| File | What it proves |
|---|---|
| `gravity_static_paper.py` | G_ab ∝ T_ab from elastic distortion; Riemann/Einstein tensors emerge |
| `gravity_kerr_paper.py` | Kerr-like geometry from rotating vortex; frame-dragging 1/r³; quantized angular momentum |
| `ed_binary_vortex.py` | Binary vortex merger + ringdown; EXPAND switch for static vs expanding universe |
| `ed_lensing_v5_redshift_polarization.py` | Three GR tests from one simulation: gravitational redshift, deflection, Shapiro delay |
| `plot_binary_comparison.py` | Comparison plotter: static vs expanding ringdown (7.3× damping ratio) |

### `act3_4/` — Strong Force + Hydrogen Atom + H₂ Molecule

ED vortex clusters reproduce confinement, proton structure,
hydrogen s-orbital quantum probability, and spontaneous H₂ covalent bonding.
See `act3_4/README.md` for the full branching pipeline and
`act3_4/RESULT.md` for detailed numerical results (Part I: single H, Part II: multi-body H₂).

**Branch A — Single Hydrogen Atom** (1 proton + 1 electron):

| File | What it proves |
|---|---|
| `ed_hydrogen_v21_3d_v5_proven.py` | 3D ED simulation: proton as Y-junction vortex cluster (GPU, ~38 min) |
| `ed_v5_hydrogen_proton2.py` | Electron-proton binding: 30k-step trajectory (GPU, ~25 min) |
| `analyze_v5_proton.py` | Y-junction search: triplet score 0.995, charge Q=+1, 120° dev=0.3° |
| `analyze_v5_proton2_profile.py` | Radial profiles centered on Proton #2; energy decomposition |
| `analyze_v5_hydrogen_orbit.py` | Radial probability P(r) matches H(1s) with R²=0.968 |
| `analyze_v5_tail_only.py` | Tail analysis: H(1s) vs Rayleigh; finite proton effect at r>7 |
| `analyze_v5_Lz.py` | Angular momentum Lz → m=0 (s-orbital confirmed) |
| `analyze_v5_confinement_3d.py` | 3D flux tube: σ_FT=1412 vs σ_bg=672 (2.10× excess) |
| `generate_paper_figures.py` | All publication figures at 300 dpi |

**Branch B — Multi-body H₂ Molecule** (5 protons + 5 electrons):

| File | What it proves |
|---|---|
| `ed_v5_hydrogen_multi.py` | 5 proton candidates × 5 electrons: 30k-step multi-body evolution (GPU, ~85 min) |
| `analyze_h2_multi.py` | P_B–P_C covalent bond: 96.7% sharing, 8 orbits, equilibrium bond length 3.4 phys |

---

## Axiom Tests

### `axiom/` — Fundamental Properties of the ED Equation

| File | What it proves |
|---|---|
| `axiom1.1_divergence_test.py` | Lattice discreteness prevents UV divergence: energy increases with decreasing dx, but max\|Ψ\| remains stable. Discreteness is a physical feature, not a bug. |

---

## Evidence Papers (EP Series)

Each EP directory contains simulation code (`.py`) and related documentation.
All EP papers are published on Zenodo with concept DOIs, which always resolve
to the latest version.

### `ep1/` — PXP Quantum Scars from Constraint

**Concept DOI:** [10.5281/zenodo.19327776](https://doi.org/10.5281/zenodo.19327776)

**Claim:** 1D chain + NN exclusion + single-site flip = PXP Hamiltonian.
No formula written by hand. Quantum many-body scars emerge automatically.

| File | Description |
|---|---|
| `pxp_constraint.py` | L=8–24 scan; H_constraint = H_PXP (exact); T_rev → 4.72 (Turner 2018) |

### `ep2/` — FQHE Topological Degeneracy + Anyon Braiding from Constraint

**Concept DOI:** [10.5281/zenodo.19329294](https://doi.org/10.5281/zenodo.19329294)

**Claim:** 2D torus + NN exclusion + Peierls phase = fractional quantum Hall effect.
No Laughlin wavefunction. No Coulomb interaction. No composite fermion.
Anyon braiding phases (2π/q) emerge from Wilson loop on constrained Hilbert space.

| File | Description |
|---|---|
| `fqhe_total_proof.py` | Tests 1–3: core degeneracy (ν=1/3), alpha scan (1/3,1/5,2/5,2/3), twist boundary |
| `fqhe_anyon_test4.py` | Test 4: Anyon braiding phase via Wilson loop; 2π/q spacing (120° for q=3, 72° for q=5) with 0.0% error; 5 systems from ν=1/3 to ν=2/5 |

### `ep3/` — Helical Quantum Scars from Constraint

**Concept DOI:** [10.5281/zenodo.19329296](https://doi.org/10.5281/zenodo.19329296)

**Claim:** 2-leg ladder + blockade + rung exclusion + flip = helical scars
with emergent chirality (±q degeneracy). No symmetry imposed.

| File | Description |
|---|---|
| `helix_constraint_paper.py` | L=6,8,10 scan; fidelity revivals; chirality degeneracy; phase-space orbits |

### `ep4/` — Non-Commutativity from Constraint

**Concept DOI:** [10.5281/zenodo.19329300](https://doi.org/10.5281/zenodo.19329300)

**Claim:** Projected operators [O₀, O₁] ≠ 0 despite [X₀, X₁] = 0.
Eigenvalues of i[O₀, O₁] are exactly integers: angular momentum
quantization from geometry, not SU(2).

| File | Description |
|---|---|
| `algebra_constraint_paper.py` | L=3–10; integer spectrum; random projection control (proves constraint specificity) |

### `ep5/` — Dark Matter as Topological Phase Persistence

**Concept DOI:** [10.5281/zenodo.19329314](https://doi.org/10.5281/zenodo.19329314)

**Claim:** Flat rotation curves are a structural attractor of the ED equation.
No dark matter particle. No new field. No free parameter.
Phase-gradient energy ½A²|∇Φ|² persists where amplitude energy |∇A|² has decayed
— topological phase persistence is dark matter.

| File | Description |
|---|---|
| `ep5_step2_single_vortex.py` | Single vortex validation on 2048² grid: v_θ(r) = 1/r to 4-digit precision, winding conservation |
| `ep5_step3_two_vortex.py` | Two-vortex amplitude-phase decoupling: (3A) annihilation (+1,−1), (3B) same-sign (+1,+1) → cores dissolve but winding +2 persists = dark matter |
| `ep5_step5_distributions.py` | N=10 vortex distributions: uniform disk vs inverse-radial (σ∝1/r) → flat rotation curve (flatness 0.82) |
| `ep5_make_figure1_final.py` | 4-panel publication figure from JSON data (300 dpi) |

### `ep6/` — Tsirelson Bound from Measurement Geometry

**Concept DOI:** [10.5281/zenodo.19329316](https://doi.org/10.5281/zenodo.19329316)

**Claim:** S_max = 2√2 is the geometry of binary discrete measurement,
not a property of Hilbert space. 2 = unit vectors, √2 = diagonal of 2D optimization.

| File | Description |
|---|---|
| `tsirelson_origin.py` | 5-step derivation with numerical verification; ED interpretation; 6-panel figure |

### `ep7/` — Josephson Effect: Gross-Pitaevskii as Low-Energy Limit of ED

**Concept DOI:** [10.5281/zenodo.19638425](https://doi.org/10.5281/zenodo.19638425)

**Claim:** The Gross-Pitaevskii equation is the low-energy (non-relativistic) limit
of the ED equation. The Josephson effect — DC current J = J_c sin(Δφ) and
AC frequency dΔφ/dt ∝ Δμ — emerges directly from the ED equation with a
barrier potential, with no GP assumption required.

| File | Description |
|---|---|
| `josephson_ed_final.py` | DC Josephson (sin fit), AC Josephson (linear frequency response), temporal phase gradient channel; 512-point 1D lattice |
| `ep7_figures.py` | Dispersion relation and regime map figures for publication |

---

## The Logic

```
Constraint  →  Structure  →  Algebra  →  Symmetry
   (P)           (scars)      ([O,O']≠0)   (SU(2), topology, ...)

Never the reverse.
The algebra is not assumed. It is FORCED by the constraint.
```

| EP | Dimension | Constraint | What emerges |
|---|---|---|---|
| I | 1D chain | NN exclusion + flip | PXP Hamiltonian, quantum scars |
| II | 2D torus | NN exclusion + hop + Peierls | FQHE topological degeneracy, anyon braiding |
| III | 2-leg ladder | Blockade + rung exclusion + flip | Helical scars, emergent chirality |
| IV | 2-leg ladder | Same + projected operators | Non-commutativity, integer spectrum |
| V | 2048² continuum | ED vortex + damped relaxation | Amplitude-phase decoupling → dark matter; flat rotation curve (flatness 0.82) |
| VI | Measurement | Binary outcomes + shared field | Tsirelson bound 2√2 |
| VII | 1D continuum | ED equation + barrier potential | Josephson effect; GP as low-energy limit |

---

## Zenodo Concept DOIs

These DOIs are Zenodo concept DOIs, so each link always resolves to the latest
published version.

| # | Key | Title | Concept DOI |
|---|---|---|---|
| Main | Shin2026main | The Existence Equation: The Grammar of Persistence | [10.5281/zenodo.18639316](https://doi.org/10.5281/zenodo.18639316) |
| EP1 | ShinEP1 | Quantum Many-Body Scars as Temporal Phase Closure of the Existence Equation | [10.5281/zenodo.19327776](https://doi.org/10.5281/zenodo.19327776) |
| EP2 | ShinEP2 | Fractional Quantum Hall States as Spatial Phase Closure of the Existence Equation | [10.5281/zenodo.19329294](https://doi.org/10.5281/zenodo.19329294) |
| EP3 | ShinEP3 | Helical Rydberg Scars as Mixed-Dimensional Phase Closure of the Existence Equation | [10.5281/zenodo.19329296](https://doi.org/10.5281/zenodo.19329296) |
| EP4 | ShinEP4 | Non-Commutativity from Geometric Constraint: Projected Algebra of the Existence Equation | [10.5281/zenodo.19329300](https://doi.org/10.5281/zenodo.19329300) |
| EP5 | ShinEP5 | Dark Matter as Topological Phase Persistence of the Existence Equation | [10.5281/zenodo.19329314](https://doi.org/10.5281/zenodo.19329314) |
| EP6 | ShinEP6 | The Tsirelson Bound as Measurement Geometry of the Existence Equation | [10.5281/zenodo.19329316](https://doi.org/10.5281/zenodo.19329316) |
| EP7 | ShinEP7 | Non-Relativistic Reduction of the Existence Equation: Gross–Pitaevskii as Low-Energy Limit and Falsifiable Predictions | [10.5281/zenodo.19638425](https://doi.org/10.5281/zenodo.19638425) |

---

## Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- CuPy (optional, for GPU acceleration in act1, act2, act3_4, ep5)

## How to Run

Each script is self-contained. Run from the corresponding directory:

```bash
cd ep1 && python pxp_constraint.py
cd ep2 && python fqhe_total_proof.py
cd ep2 && python fqhe_anyon_test4.py
cd ep3 && python helix_constraint_paper.py
cd ep4 && python algebra_constraint_paper.py
cd ep6 && python tsirelson_origin.py
cd ep7 && python josephson_ed_final.py
cd axiom && python axiom1.1_divergence_test.py
```

For GPU-intensive simulations (act1, act2, act3_4, ep5), use Google Colab.
See `act3_4/README.md` and `act2/README.md` for detailed Colab instructions.

Output figures are saved as 300 dpi PNG in the working directory.

---

## License

All code and results by Jae-Ahn Shin.
For academic use and review. Please cite appropriately.

---

*Constraint forces structure. Obstruction creates closure.*
