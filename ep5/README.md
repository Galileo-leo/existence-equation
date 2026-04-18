# EP V — Dark Matter as Topological Phase Persistence

**DOI:** [10.5281/zenodo.19329315](https://doi.org/10.5281/zenodo.19329315)

**Equation:** `(1/c²)Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ`

**Grid:** 2048², λ=1.0, α=1.0, GPU: NVIDIA A100-SXM4-80GB

**Author:** Jae-Ahn Shin · 2026

---

## Claim

Flat rotation curves are a structural attractor of the ED equation.
No dark matter particle. No new field. No free parameter.

The phase-gradient energy ½A²|∇Φ|² persists where the amplitude
energy |∇A|² has decayed — **topological phase persistence is dark matter.**

---

## Execution Order

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  STEP 2  ep5_step2_single_vortex.py       [GPU, ~15 sec]       │
 │          Single vortex pipeline validation                      │
 │          Output: ep5_step2_single_vortex.json                   │
 └──────────────────────────┬──────────────────────────────────────┘
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STEP 3  ep5_step3_two_vortex.py          [GPU, ~30 sec]       │
 │          Two-vortex amplitude-phase decoupling                  │
 │          Output: ep5_step3_two_vortex.json                      │
 └──────────────────────────┬──────────────────────────────────────┘
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  STEP 5  ep5_step5_distributions.py       [GPU, ~2 sec]        │
 │          Distribution-dependent rotation curves                 │
 │          Output: ep5_step5_distributions.json                   │
 └──────────────────────────┬──────────────────────────────────────┘
                            ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  FIGURE  ep5_make_figure1_final.py        [CPU, ~5 sec]        │
 │          4-panel publication figure (300 dpi)                    │
 │          Input:  all 3 JSON files above                         │
 │          Output: ed_topological_phase_result.png                │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Colab Commands

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/existence_equation/ed8/ep5

# Step 2: Single vortex validation (~15 sec)
%run ep5_step2_single_vortex.py

# Step 3: Two-vortex decoupling (~30 sec)
%run ep5_step3_two_vortex.py

# Step 5: Distribution rotation curves (~2 sec)
%run ep5_step5_distributions.py

# Figure: 4-panel composite (CPU)
%run ep5_make_figure1_final.py
```

---

## Source Files

| Step | File | Description | Runtime | Output |
|------|------|-------------|---------|--------|
| 2 | `ep5_step2_single_vortex.py` | Single vortex on 2048² grid. Validates pipeline: winding conservation, v_θ(r) = 1/r to 4-digit precision, \|Ψ\| → v_vac far from core | ~15 sec | `ep5_step2_single_vortex.json` |
| 3 | `ep5_step3_two_vortex.py` | Two-vortex pair tests. **(3A)** Vortex-antivortex (+1,−1): annihilation, net winding 0 preserved. **(3B)** Same-sign (+1,+1): cores dissolve but large-loop winding +2 persists — the dynamical definition of dark matter | ~30 sec | `ep5_step3_two_vortex.json` |
| 5 | `ep5_step5_distributions.py` | N=10 vortex distributions. Uniform disk vs inverse-radial (σ∝1/r). Tests v_θ(r) = N_enc(r)/r. Inverse-r → flat rotation curve (flatness 0.82); uniform → declining (0.63) | ~2 sec | `ep5_step5_distributions.json` |
| Fig | `ep5_make_figure1_final.py` | 4-panel publication figure from JSON data. (a) single vortex, (b) amplitude-phase decoupling, (c) rotation curves, (d) vortex position maps | ~5 sec | `ed_topological_phase_result.png` |

---

## Method

- 2D finite-difference ED equation on 2048×2048 grid
- Full-field multiplicative vortex ansatz: Ψ = v_vac · tanh(r/ξ) · e^(iθ)
- Damped leapfrog integrator, γ = 0.02 (Steps 2–3), γ = 0.001 (Step 5)
- CuPy (GPU) with NumPy fallback

---

## Key Results

| Experiment | Setup | Result | Metric |
|------------|-------|--------|--------|
| **Single vortex** | Isolated defect | v_θ(r) = 1/r (4-digit match, 3 < r < 15) | ⟨v_θ·r⟩ = 1.0000 ± 0.0001 |
| **Same-sign pair** | (+1,+1) under damped relaxation | Amplitude smooths to vacuum; circulation persists | ⟨v_θ·r⟩ = 2.0000 ± 0.0003 |
| **Vortex-antivortex** | (+1,−1) annihilation | Both cores dissolve; net winding 0 preserved | Control test |
| **Inverse-r distribution** | N=10, σ(r) ∝ 1/r | Flat rotation curve | Flatness 0.82 |
| **Uniform distribution** | N=10, σ = const | Declining rotation curve | Flatness 0.63 |

---

## Physical Interpretation

1. **Amplitude-phase decoupling** (Step 3B): After relaxation, the field
   amplitude smooths to an apparently featureless vacuum — nothing an
   electromagnetic probe would detect. Yet the phase gradient persists,
   carrying full macroscopic circulation. This is dark matter: gravitationally
   active, electromagnetically silent.

2. **Distribution → rotation curve** (Step 5): The shape of the rotation
   curve depends only on the spatial distribution of topological charge.
   A 1/r density profile produces an approximately flat curve — the hallmark
   of galaxy dynamics that has driven dark matter searches for four decades.

3. **The mechanism**: The phase-gradient energy ½A²|∇Φ|² persists where
   the amplitude energy |∇A|² has decayed. Topological phase persistence
   is the structural origin of dark matter.

---

## Directory Structure

```
ep5/
├── README.md
├── ep5_step2_single_vortex.py     ← STEP 2: Pipeline validation
├── ep5_step3_two_vortex.py        ← STEP 3: Amplitude-phase decoupling
├── ep5_step5_distributions.py     ← STEP 5: Rotation curves
└── ep5_make_figure1_final.py      ← FIGURE: 4-panel composite
```
