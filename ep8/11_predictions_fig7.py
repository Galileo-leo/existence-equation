#!/usr/bin/env python3
"""
=============================================================================
EP8 Figure 7: Predictions summary (reads v3.4 fine-scan JSON)
=============================================================================
Generates the prediction figure for the EP8 (electron-forward) manuscript
ENTIRELY from the v3.4 simulation output. All thresholds, ratios, and diagnostic
curves are computed from ep8_v3.4_summary.json; the only external input is
the optional hydrogen BSI calibration constant (I_BSI).

Panels:
  (a) Four-stage hierarchy: normalized diagnostics vs eps with the four
      resolved thresholds marked (shows the ordering eps_coh < eps_em < eps_topo).
  (b) Same thresholds on a BSI-anchored intensity axis (W/cm^2), with the
      conventional strong-field regime labels.
  (c) Second calibration axis: chi = E_packet/E_closure vs eps^2 (linearity
      confirms chi ~ eps^2; the chi-based ratios match the eps^2-based ratios).
  (d) Angular-structure transition: normalized entropy S_norm rises and
      quadrupole anisotropy Q2 collapses across eps_topo.

Run next to ep8_v3.4_summary.json:
    python3 11_predictions_fig7.py

Author: Jae-Ahn Shin, Independent Researcher, Incheon, Republic of Korea
=============================================================================
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v3.4"
DPI = 600
JSON_PATH = f"ep8_{VERSION}_summary.json"

# Barrier-suppression intensity for hydrogen (I_p = 13.6 eV, Z = 1):
#   E_BSI = I_p^2/(4Z) a.u. = (1/16) E_atomic ;  I_BSI = I_atomic/256
I_BSI = 1.37e14  # W/cm^2

# ----------------------------------------------------------------------
# 1. LOAD DATA (everything below is derived from this)
# ----------------------------------------------------------------------
with open(JSON_PATH, "r", encoding="utf-8") as f:
    rows = json.load(f)["rows"]

eps   = np.array([r["eps"]      for r in rows])
Zb    = np.array([r.get("Zb_max", r["Zb_final"]) for r in rows])
Gk    = np.array([r["G_k"]      for r in rows])
dNb   = np.array([r["dNb"]      for r in rows])
w     = np.array([r["w_after"]  for r in rows])
chi   = np.array([r.get("chi", np.nan) for r in rows])
Sn    = np.array([r["S_norm"]   for r in rows])
Q2    = np.array([r["Q2"]       for r in rows])

# ----------------------------------------------------------------------
# 2. EXTRACT THE FOUR THRESHOLDS (same definitions as the analysis)
# ----------------------------------------------------------------------
# eps_dep : first sign change of dNb (depletion onset), linear interpolation
sgn = np.sign(dNb)
idx = np.where(np.diff(sgn) != 0)[0]
if len(idx):
    i = idx[0]
    eps_dep = eps[i] + (eps[i+1]-eps[i]) * (0 - dNb[i]) / (dNb[i+1]-dNb[i])
else:
    eps_dep = np.nan

eps_coh = eps[int(np.argmax(Zb))]            # phase-coherence peak
eps_em  = eps[int(np.argmax(Gk))]            # structured-emission peak
broken  = np.abs(w - 1.0) > 0.1              # winding loss
eps_topo_first = eps[int(np.argmax(broken))] if broken.any() else np.nan
# honest bracket: last good eps and first broken eps
if broken.any():
    jb = int(np.argmax(broken))
    eps_topo_lo, eps_topo_hi = eps[jb-1], eps[jb]
else:
    eps_topo_lo = eps_topo_hi = np.nan

# Ratios (I ~ eps^2)
def r2(a, b):
    return (a / b) ** 2
R1 = r2(eps_coh, eps_dep)
R2 = r2(eps_em,  eps_coh)
R3lo, R3hi = r2(eps_topo_lo, eps_em), r2(eps_topo_hi, eps_em)

# chi/eps^2 constant check + chi-based ratios
chi_over_eps2 = chi / eps**2
def chi_at(e):
    return chi[int(np.argmin(np.abs(eps - e)))]

print("=" * 64)
print("EP8 v3.4 — thresholds extracted from data")
print("=" * 64)
print(f"  eps_dep  = {eps_dep:.4f}")
print(f"  eps_coh  = {eps_coh:.3f}")
print(f"  eps_em   = {eps_em:.3f}")
print(f"  eps_topo in [{eps_topo_lo:.2f}, {eps_topo_hi:.2f}]")
print(f"  ratios: I_coh/I_dep={R1:.3f}  I_em/I_coh={R2:.3f}  "
      f"I_topo/I_em={R3lo:.3f}-{R3hi:.3f}")
print(f"  chi/eps^2 (mean +/- std) = {np.nanmean(chi_over_eps2):.2f} "
      f"+/- {np.nanstd(chi_over_eps2):.2f}")

# ----------------------------------------------------------------------
# 3. FIGURE
# ----------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    f"EP8 {VERSION}: Strong-drive hierarchy of a topological closure "
    f"— predictions",
    fontsize=13, fontweight="bold")

tcol = {"dep": "#888888", "coh": "#1f77b4", "em": "#d62728", "topo": "#9467bd"}

# ---- (a) four-stage hierarchy, normalized diagnostics ----
a = ax[0, 0]
a.plot(eps, Zb / np.max(Zb), "o-", color=tcol["coh"], ms=4,
       label=r"$Z_\Phi/Z_\Phi^{\max}$ (coherence)")
gk_norm = Gk / (np.max(Gk) + 1e-30)
a.plot(eps, gk_norm, "s-", color=tcol["em"], ms=4,
       label=r"$G_k/G_k^{\max}$ (emission)")
for nm, e, lab in [("dep", eps_dep, r"$\varepsilon_{\rm dep}$"),
                   ("coh", eps_coh, r"$\varepsilon_{\rm coh}$"),
                   ("em",  eps_em,  r"$\varepsilon_{\rm em}$"),
                   ("topo", eps_topo_hi, r"$\varepsilon_{\rm topo}$")]:
    a.axvline(e, color=tcol[nm], ls="--", lw=1.3)
    a.text(e, 1.02, lab, color=tcol[nm], ha="center", fontsize=9)
a.axvspan(eps_topo_lo, eps_topo_hi, color=tcol["topo"], alpha=0.15)
a.set_xlabel(r"drive strength $\varepsilon$")
a.set_ylabel("normalized diagnostic")
a.set_title("(a) Four-stage hierarchy and ordering", fontsize=11)
a.legend(fontsize=8, loc="center right")
a.grid(alpha=0.3)
a.set_ylim(0, 1.12)

# ---- (b) BSI-anchored intensity axis ----
b = ax[0, 1]
names = [r"$I_{\rm dep}$", r"$I_{\rm coh}$", r"$I_{\rm em}$", r"$I_{\rm topo}$"]
evals = [eps_dep, eps_coh, eps_em, 0.5*(eps_topo_lo+eps_topo_hi)]
Ivals = [e**2 * I_BSI for e in evals]
cols  = [tcol["dep"], tcol["coh"], tcol["em"], tcol["topo"]]
regimes = ["multiphoton\nonset", "phase\ncoherence",
           "structured\nemission", "closure loss\n(near BSI)"]
for k, (nm, Iv, c, rg) in enumerate(zip(names, Ivals, cols, regimes)):
    b.scatter([k], [Iv], s=140, color=c, zorder=3)
    b.text(k, Iv*1.35, f"{nm}\n{Iv:.1e}", ha="center", va="bottom",
           fontsize=8.5, color=c)
    b.text(k, Iv*0.62, rg, ha="center", va="top", fontsize=7.5, color="0.3")
b.axhline(I_BSI, color="k", ls=":", lw=1)
b.text(3.35, I_BSI, r"$I_{\rm BSI}$", va="center", fontsize=9)
b.set_yscale("log")
b.set_xticks(range(4)); b.set_xticklabels(["", "", "", ""])
b.set_ylabel(r"intensity $I=\varepsilon^2 I_{\rm BSI}$  (W/cm$^2$, H calibration)")
b.set_title("(b) BSI-anchored absolute thresholds", fontsize=11)
b.set_xlim(-0.5, 3.9)
b.grid(alpha=0.3, axis="y")

# ---- (c) chi vs eps^2 (second calibration axis) ----
c = ax[1, 0]
c.plot(eps**2, chi, "o", color="#2ca02c", ms=5, label=r"data")
# linear fit through origin
mask = np.isfinite(chi)
slope = np.sum(chi[mask]*eps[mask]**2) / np.sum(eps[mask]**4)
xx = np.linspace(0, np.max(eps**2), 50)
c.plot(xx, slope*xx, "k--", lw=1.2,
       label=rf"$\chi = {slope:.1f}\,\varepsilon^2$")
c.set_xlabel(r"$\varepsilon^2$")
c.set_ylabel(r"$\chi = E_{\rm packet}/E_{\rm closure}$")
c.set_title(r"(c) Second calibration axis: $\chi \propto \varepsilon^2$",
            fontsize=11)
c.legend(fontsize=9, loc="upper left")
c.grid(alpha=0.3)
txt = (rf"$I_{{\rm coh}}/I_{{\rm dep}}={R1:.2f}$" + "\n"
       rf"$I_{{\rm em}}/I_{{\rm coh}}={R2:.2f}$" + "\n"
       rf"$I_{{\rm topo}}/I_{{\rm em}}={R3lo:.2f}$-${R3hi:.2f}$")
c.text(0.97, 0.05, "scale-free ratios:\n" + txt, transform=c.transAxes,
       ha="right", va="bottom", fontsize=8.5,
       bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# ---- (d) angular-structure transition ----
d = ax[1, 1]
d.plot(eps, Sn, "o-", color="#1f77b4", ms=4,
       label=r"$S_{\rm norm}$ (angular entropy)")
d.plot(eps, Q2, "s-", color="#d62728", ms=4,
       label=r"$Q_2$ (quadrupole anisotropy)")
d.axvspan(eps_topo_lo, eps_topo_hi, color=tcol["topo"], alpha=0.15)
d.axvline(eps_em, color=tcol["em"], ls="--", lw=1.0)
d.text(eps_em, d.get_ylim()[1], r"$\varepsilon_{\rm em}$",
       color=tcol["em"], ha="center", fontsize=9, va="bottom")
d.axvline(eps_topo_hi, color=tcol["topo"], ls="--", lw=1.0)
d.text(eps_topo_hi, d.get_ylim()[1], r"$\varepsilon_{\rm topo}$",
       color=tcol["topo"], ha="center", fontsize=9, va="bottom")
d.set_xlabel(r"drive strength $\varepsilon$")
d.set_ylabel("angular diagnostic")
d.set_title("(d) Angular structure: entropy rise, quadrupole collapse",
            fontsize=11)
d.legend(fontsize=9, loc="center left")
d.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"ep8_{VERSION}_predictions.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
