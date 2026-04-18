# ED Strong Force & Hydrogen Atom Simulation — ed8/act3_4/

## Existence Deviation (ED) Strong Force Simulation

**Equation:**  `(1/c²)Ψ̈ = ∇²Ψ + λΨ - α|Ψ|²Ψ`

**Lattice:** 256³, λ=1.0, α=1.0, GPU: NVIDIA A100-SXM4-80GB

**Authors:** Jae-Ahn Shin · 2026-03

---

## Overview

This directory contains two simulation branches that share a single
frozen field (`v5_frozen_state.npz`):

| Branch | Purpose | Key Question |
|--------|---------|-------------|
| **A. Single Hydrogen** | Inject 1 electron on Proton #2, evolve 30k steps | Does the ED equation reproduce atomic orbital structure (P(r), Lz, binding)? |
| **B. Multi-body H₂** | Inject 5 electrons on all 5 proton candidates, evolve 30k steps | Does covalent bonding and H₂ molecule formation emerge spontaneously? |

---

## Execution Order (Google Colab)

**Step 1 is shared.  Then the pipeline branches.**

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  STEP 1  ed_hydrogen_v21_3d_v5_proven.py   [GPU, ~38 min]     │
 │          3D ED field evolution → vortex formation               │
 │          Output: v5_frozen_state.npz, v5_analysis.json          │
 └──────────────────────────┬──────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
  ╔═══════════════════════╗     ╔═══════════════════════════╗
  ║  Branch A:            ║     ║  Branch B:                ║
  ║  Single Hydrogen      ║     ║  Multi-body H₂            ║
  ║  (1 proton + 1 e⁻)   ║     ║  (5 protons + 5 e⁻)      ║
  ╚══════════╤════════════╝     ╚════════════╤══════════════╝
             ▼                               ▼
  ┌─────────────────────────┐   ┌──────────────────────────────┐
  │ STEP A2                 │   │ STEP B2                      │
  │ ed_v5_hydrogen_proton2  │   │ ed_v5_hydrogen_multi.py      │
  │ [GPU, ~25 min]          │   │ [GPU, ~85 min]               │
  │ → track.json            │   │ → v5_h2_multi_track.json     │
  └────────────┬────────────┘   └──────────────┬───────────────┘
               ▼                               ▼
    ┌──────────┴──────────┐     ┌──────────────────────────────┐
    ▼                     ▼     │ STEP B3                      │
 npz-based         track-based  │ analyze_h2_multi.py          │
 (proton)          (electron)   │ → h2_multi_analysis.png      │
    ▼                     ▼     └──────────────────────────────┘
 A3: proton.py     A5: orbit.py
 A4: profile.py    A6: tail.py
                   A7: Lz.py
    └──────┬──────────┘
           ▼
 ┌─────────────────────────────┐
 │ STEP A8  confinement_3d.py  │
 │ → v5_confinement_3d.json    │
 └─────────────┬───────────────┘
               ▼
 ┌─────────────────────────────┐
 │ STEP A9  paper_figures.py   │
 │ → 5 publication PNGs        │
 └─────────────────────────────┘
```

---

## Colab Commands

### 0. Setup (common)

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/existence_equation/ed8/act3_4
```

### Step 1: 3D ED Simulation (shared)

```python
# GPU required | ~38 min | 256³ grid
# Output: v5_frozen_state.npz, v5_analysis.json
# *** SKIP if npz already exists ***
%run ed_hydrogen_v21_3d_v5_proven.py
```

---

### Branch A: Single Hydrogen Atom

```python
# STEP A2: Electron injection — single proton binding trajectory
#   GPU required | ~25 min | 30,000 steps
#   Input:  v5_frozen_state.npz
#   Output: v5_hydrogen_proton2_track.json
%run ed_v5_hydrogen_proton2.py

# STEP A3: Proton structure analysis
#   CPU OK | Y-junction search, confinement, charge winding
%run analyze_v5_proton.py

# STEP A4: Proton #2 radial profile
#   CPU OK | A(r), energy decomposition, flux tube cross-section
%run analyze_v5_proton2_profile.py

# STEP A5: Hydrogen orbital analysis
#   CPU OK | P(r) fitting: H 1s vs Rayleigh vs Maxwell-3D
%run analyze_v5_hydrogen_orbit.py

# STEP A6: Tail region analysis
#   CPU OK | Decisive H 1s vs Rayleigh in large-r bins
%run analyze_v5_tail_only.py

# STEP A7: Angular momentum Lz
#   CPU OK | Confirms m=0 (s-orbital)
%run analyze_v5_Lz.py

# STEP A8: 3D Confinement analysis (honest metric)
#   CPU OK | ~7 sec | 3D vortex-line tracing + path integral
%run analyze_v5_confinement_3d.py

# STEP A9: Publication figures (300 dpi)
#   CPU OK | 5 figures for the paper
%run generate_paper_figures.py
```

---

### Branch B: Multi-body H₂ Molecule

```python
# STEP B2: 5 protons × 5 electrons — multi-body evolution
#   GPU required | ~85 min | 30,000 steps
#   Input:  v5_frozen_state.npz
#   Output: v5_h2_multi_track.json
%run ed_v5_hydrogen_multi.py

# STEP B3: Binding + covalent sharing analysis
#   CPU OK | Binding stats, loyalty, hopping, H₂ sharing frequency
#   Output: h2_multi_analysis.png
%run analyze_h2_multi.py
```

---

## Source Files

### Simulation (GPU required)

| File | Branch | Description | Runtime | Output |
|------|--------|-------------|---------|--------|
| `ed_hydrogen_v21_3d_v5_proven.py` | Shared | Phase 1: Energy injection → Phase 2: Cosmic expansion (a: 1→2) → Spontaneous vortex formation → Linear confinement / charge / energy measurement | ~38 min | `v5_frozen_state.npz`, `v5_analysis.json` |
| `ed_v5_hydrogen_proton2.py` | A | Place electron (-1 vortex) near Proton #2 → 30k step evolution → trajectory recording | ~25 min | `v5_hydrogen_proton2_track.json` |
| `ed_v5_hydrogen_multi.py` | B | Insert -1 vortex at each of 5 proton candidates → 30k step evolution → all-pair tracking with sharing detection | ~85 min | `v5_h2_multi_track.json` |

### Analysis (CPU only)

| Step | File | Branch | Input | Description | Output |
|------|------|--------|-------|-------------|--------|
| A3 | `analyze_v5_proton.py` | A | `npz` | Proton Y-junction search. Triplet score, 120° symmetry, FT gate, charge measurement | `v5_proton_analysis.json` |
| A4 | `analyze_v5_proton2_profile.py` | A | `npz` | Radial profile centered on Proton #2. A(r), phase, energy decomposition, winding vs R, flux tube cross-section | `v5_proton2_profile.json` |
| A5 | `analyze_v5_hydrogen_orbit.py` | A | `track.json` | Electron trajectory statistics. P(r) distribution, H 1s / Rayleigh / Maxwell-3D fitting, 2D trajectory, angular velocity | `v5_hydrogen_orbit_stats.json` |
| A6 | `analyze_v5_tail_only.py` | A | `track.json` | Tail analysis. Model/Data ratio for r=4–10+ bins. Decisive: H 1s vs Rayleigh | `v5_tail_only.json` |
| A7 | `analyze_v5_Lz.py` | A | `track.json` | Angular momentum. Lz(t), omega(t), cumulative rotation. Confirms m=0 (s-orbital) | `v5_Lz_analysis.json` |
| A8 | `analyze_v5_confinement_3d.py` | A | `npz` | 3D confinement. Vortex-line tracing through all z-slices, path-integral energy, honest σ=E/d metric, parameter sensitivity scan | `v5_confinement_3d.json` |
| A9 | `generate_paper_figures.py` | A | `npz` + JSONs | Batch generation of 5 paper-quality figures at 300 dpi | 5 PNGs |
| B3 | `analyze_h2_multi.py` | B | `multi_track.json` | Binding %, loyalty, hop count, P_B↔P_C sharing frequency, CM trajectory analysis | `h2_multi_analysis.png` |

### Auxiliary (not used in final pipeline)

| File | Note |
|------|------|
| `analyze_v5_tail_only_test2.py` | Exploratory: finite-proton correction test (β·r² term). Not used in final results. |

---

## Key Results

### Branch A: Single Hydrogen Atom

| Phenomenon | Measurement | Significance |
|------------|-------------|--------------|
| **Y-junction (baryon)** | Triplet score 0.995, 120° deviation 0.3° | Spontaneous baryon-like structure |
| **Flux tube confinement** | 3D: σ_FT=1412 vs σ_bg=672 (2.10×), σ FLAT vs d | Genuine string tension |
| **Charge quantization** | Q = +1 (multiple radii) | Topological winding = electric charge |
| **Electron binding** | 99.8% bound, 30k steps | Stable hydrogen-like atom |
| **P(r) = H 1s** | R² = 0.968, tail r>7: finite proton effect | Quantum probability from ED |
| **Angular momentum m=0** | |⟨Lz⟩|/SEM = 0.9, skew = -0.014 | s-orbital confirmed |

### Branch B: Multi-body H₂ Molecule

| Phenomenon | Measurement | Significance |
|------------|-------------|--------------|
| **Covalent bond** | P_B–P_C sharing 96.7%, ~5300 hops each | Spontaneous electron exchange |
| **Selective bonding** | Only d=3.4 pair bonds; d>11 does not | Distance-dependent bonding threshold |
| **Bond-length equilibrium** | Proton drift 1.3 → 3.4 phys | Self-found equilibrium distance |
| **Orbital motion** | +2874° (8 orbits), T ~ 23 steps, ω ~ 0.275 | Stable molecular rotation |
| **Independent H atoms** | P_A, P_D, P_E loyalty > 94% | Non-bonded atoms remain isolated |

---

## Directory Structure

```
ed8/act3_4/
├── README.md
├── RESULT.md
│
├── [Shared Simulation — GPU]
│   └── ed_hydrogen_v21_3d_v5_proven.py    ← STEP 1: Main (frozen field)
│
├── [Branch A: Single Hydrogen]
│   ├── ed_v5_hydrogen_proton2.py          ← STEP A2: 1 electron binding
│   ├── analyze_v5_proton.py               ← STEP A3: Y-junction search
│   ├── analyze_v5_proton2_profile.py      ← STEP A4: Energy decomposition
│   ├── analyze_v5_hydrogen_orbit.py       ← STEP A5: P(r) + trajectory
│   ├── analyze_v5_tail_only.py            ← STEP A6: Tail H 1s vs Rayleigh
│   ├── analyze_v5_Lz.py                   ← STEP A7: Angular momentum m=0
│   ├── analyze_v5_confinement_3d.py       ← STEP A8: 3D confinement
│   └── generate_paper_figures.py          ← STEP A9: Publication figures
│
├── [Branch B: Multi-body H₂]
│   ├── ed_v5_hydrogen_multi.py            ← STEP B2: 5 protons × 5 electrons
│   └── analyze_h2_multi.py               ← STEP B3: Binding + sharing
│
└── [Auxiliary]
    └── analyze_v5_tail_only_test2.py      ← Exploratory test (not in pipeline)
```
