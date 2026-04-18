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

**Main Paper DOI:**
[10.5281/zenodo.18639317](https://doi.org/10.5281/zenodo.18639317)

---

## Repository Structure

```
ed8/
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
├── ep5/           Evidence Paper V   — Galaxy rotation curves
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
All EP papers are published on Zenodo with DOIs.

### `ep1/` — PXP Quantum Scars from Constraint

**DOI:** [10.5281/zenodo.19327777](https://doi.org/10.5281/zenodo.19327777)

**Claim:** 1D chain + NN exclusion + single-site flip = PXP Hamiltonian.
No formula written by hand. Quantum many-body scars emerge automatically.

| File | Description |
|---|---|
| `pxp_constraint.py` | L=8–24 scan; H_constraint = H_PXP (exact); T_rev → 4.72 (Turner 2018) |

### `ep2/` — FQHE Topological Degeneracy + Anyon Braiding from Constraint

**DOI:** [10.5281/zenodo.19329295](https://doi.org/10.5281/zenodo.19329295)

**Claim:** 2D torus + NN exclusion + Peierls phase = fractional quantum Hall effect.
No Laughlin wavefunction. No Coulomb interaction. No composite fermion.
Anyon braiding phases (2π/q) emerge from Wilson loop on constrained Hilbert space.

| File | Description |
|---|---|
| `fqhe_total_proof.py` | Tests 1–3: core degeneracy (ν=1/3), alpha scan (1/3,1/5,2/5,2/3), twist boundary |
| `fqhe_anyon_test4.py` | Test 4: Anyon braiding phase via Wilson loop; 2π/q spacing (120° for q=3, 72° for q=5) with 0.0% error; 5 systems from ν=1/3 to ν=2/5 |

### `ep3/` — Helical Quantum Scars from Constraint

**DOI:** [10.5281/zenodo.19329297](https://doi.org/10.5281/zenodo.19329297)

**Claim:** 2-leg ladder + blockade + rung exclusion + flip = helical scars
with emergent chirality (±q degeneracy). No symmetry imposed.

| File | Description |
|---|---|
| `helix_constraint_paper.py` | L=6,8,10 scan; fidelity revivals; chirality degeneracy; phase-space orbits |

### `ep4/` — Non-Commutativity from Constraint

**DOI:** [10.5281/zenodo.19329301](https://doi.org/10.5281/zenodo.19329301)

**Claim:** Projected operators [O₀, O₁] ≠ 0 despite [X₀, X₁] = 0.
Eigenvalues of i[O₀, O₁] are exactly integers: angular momentum
quantization from geometry, not SU(2).

| File | Description |
|---|---|
| `algebra_constraint_paper.py` | L=3–10; integer spectrum; random projection control (proves constraint specificity) |

### `ep5/` — Galaxy Rotation Curves from ED

**DOI:** [10.5281/zenodo.19329315](https://doi.org/10.5281/zenodo.19329315)

**Claim:** Flat rotation curves are a structural attractor of the ED equation.
No dark matter. Phase gradient energy = "missing mass."

| File | Description |
|---|---|
| `ep5_step2_single_vortex.py` | Single vortex evolution and profile measurement |
| `ep5_step3_two_vortex.py` | Two-vortex interaction and energy analysis |
| `ep5_step5_distributions.py` | Distribution analysis across parameter space |
| `ep5_make_figure1_final.py` | Publication figure generation |

### `ep6/` — Tsirelson Bound from Measurement Geometry

**DOI:** [10.5281/zenodo.19329317](https://doi.org/10.5281/zenodo.19329317)

**Claim:** S_max = 2√2 is the geometry of binary discrete measurement,
not a property of Hilbert space. 2 = unit vectors, √2 = diagonal of 2D optimization.

| File | Description |
|---|---|
| `tsirelson_origin.py` | 5-step derivation with numerical verification; ED interpretation; 6-panel figure |

### `ep7/` — Josephson Effect: Gross-Pitaevskii as Low-Energy Limit of ED

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
| V | 2D continuum | ED equation + expansion | Flat rotation curves, f_phase ~ 52% |
| VI | Measurement | Binary outcomes + shared field | Tsirelson bound 2√2 |
| VII | 1D continuum | ED equation + barrier potential | Josephson effect; GP as low-energy limit |

---

## Zenodo DOIs

| Paper | DOI |
|---|---|
| Main: The Existence Equation | [10.5281/zenodo.18639317](https://doi.org/10.5281/zenodo.18639317) |
| EP I: PXP Quantum Scars | [10.5281/zenodo.19327777](https://doi.org/10.5281/zenodo.19327777) |
| EP II: FQHE Topological Degeneracy | [10.5281/zenodo.19329295](https://doi.org/10.5281/zenodo.19329295) |
| EP III: Helical Rydberg Scars | [10.5281/zenodo.19329297](https://doi.org/10.5281/zenodo.19329297) |
| EP IV: Non-Commutativity | [10.5281/zenodo.19329301](https://doi.org/10.5281/zenodo.19329301) |
| EP V: Galaxy Rotation Curves | [10.5281/zenodo.19329315](https://doi.org/10.5281/zenodo.19329315) |
| EP VI: Tsirelson Bound | [10.5281/zenodo.19329317](https://doi.org/10.5281/zenodo.19329317) |

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
