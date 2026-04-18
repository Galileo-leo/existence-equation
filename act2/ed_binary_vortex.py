"""
ed_binary_vortex.py — ED Binary Vortex Merger
==============================================================================
Two opposite-winding topological defects on a 3D lattice.
Inspiral → merger (winding cancellation) → post-merger dynamics.

Equation: (1/c²) Ψ̈ = ∇²Ψ + λΨ − α|Ψ|²Ψ

With cosmic expansion (EXPAND=True):
  dx(t) = dx₀ · a(t),   a(t) = 1 + R·(t − t_expand)
  The equation is untouched. Only dx changes.
  "dx만 변한다" — the core ED principle.

This is the SAME technique used in:
  - Act III  (flux tube / baryon formation)
  - EP V     (galaxy rotation curves)

Two modes:
  EXPAND = False  →  static universe    →  permanent breather oscillation
  EXPAND = True   →  expanding universe →  damped ringdown → evaporation

Output:
  binary_noexpand.json / binary_expand.json   (data)
  ed_binary_vortex_result.png                 (comparison figure)

Author: Jae-Ahn Shin & Kael · 2026-04-06
==============================================================================
"""

import numpy as np
import time
import json
import os

# ════════════════════════════════════════════════════════════════
# ★★★  MASTER SWITCH  ★★★
# ════════════════════════════════════════════════════════════════

EXPAND = False       # True = expanding universe, False = static universe

# ════════════════════════════════════════════════════════════════
#   VORTEX PARAMETERS
# ════════════════════════════════════════════════════════════════

N1     = 1          # Winding number, vortex 1  (+1 = counterclockwise)
N2     = -1         # Winding number, vortex 2  (-1 = clockwise)
DEPTH1 = 0.3        # V1 amplitude depth  (0 = no defect, 1 = full well)
DEPTH2 = 0.2        # V2 amplitude depth
XI     = 3.0        # Core radius (physical units)
SEP    = 22.0       # Initial separation between vortex centers
KICK   = 0.1        # Tangential phase kick strength

# ════════════════════════════════════════════════════════════════
#   EXPANSION CONTROL
# ════════════════════════════════════════════════════════════════

EXPAND_START  = 4000          # Start expansion after merger
EXPAND_RATE   = 2.174e-05     # a(t) = 1 + rate*(step - start)
A_MAX         = 3.0           # Cap expansion

# ════════════════════════════════════════════════════════════════
#   EVOLUTION CONTROL
# ════════════════════════════════════════════════════════════════

RELAX_STEPS   = 2000          # Damped relaxation before kick
FREE_STEPS    = 94000         # Free evolution after kick
GAMMA         = 0.008         # Damping coefficient during relaxation
KICK_WIDTH    = 1.5           # Kick envelope = XI * this

# ════════════════════════════════════════════════════════════════
#   GRID
# ════════════════════════════════════════════════════════════════

N_GRID = 320                  # 320 for test, 640 for production
L      = 80.0                 # Physical box size

# ════════════════════════════════════════════════════════════════
#   EQUATION PARAMETERS
# ════════════════════════════════════════════════════════════════

V_VAC = 30.0                          # Vacuum amplitude A₀ = √(λ/α)
LAM   = 1.0 / (8.0 * L * L)          # λ = 1/(8L²) = 1.953125e-05
ALPHA = LAM / (V_VAC**2)              # α = λ/v²_vac
C     = 1.0                           # Wave speed
DT    = 0.005                         # Timestep

# ════════════════════════════════════════════════════════════════
#   MEASUREMENT
# ════════════════════════════════════════════════════════════════

MEASURE_EVERY = 50
PRINT_EVERY   = 500
WINDING_EVERY = 500

# ════════════════════════════════════════════════════════════════
#   OUTPUT
# ════════════════════════════════════════════════════════════════

OUTDIR = "."
# OUTDIR = "/content/drive/MyDrive/existence_equation/gravity_binary_vortex"

# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
#    ↓↓↓  CODE — normally don't edit  ↓↓↓
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════

try:
    import cupy as cp
    xp = cp; GPU = True
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(f"GPU: {props['name'].decode()}, "
          f"VRAM: {cp.cuda.Device(0).mem_info[1]/1e9:.1f} GB")
except ImportError:
    xp = np; GPU = False; print("CPU mode")

def to_np(a):
    return cp.asnumpy(a) if GPU else a

# Derived
dx0 = L / N_GRID
total_steps = RELAX_STEPS + FREE_STEPS
center = L / 2.0
kick_env = XI * KICK_WIDTH
os.makedirs(OUTDIR, exist_ok=True)

# Auto-tag
TAG = "expand" if EXPAND else "noexpand"

print(f"\n{'═'*65}")
print(f"  ED BINARY VORTEX MERGER")
print(f"  Mode: {'EXPANDING UNIVERSE' if EXPAND else 'STATIC UNIVERSE'}")
print(f"{'═'*65}")
print(f"  V1: n={N1:+d}  depth={DEPTH1}  ξ={XI}")
print(f"  V2: n={N2:+d}  depth={DEPTH2}  ξ={XI}")
print(f"  sep={SEP}  kick={KICK}")
print(f"  Grid: {N_GRID}³, L={L}, dx₀={dx0:.4f}")
print(f"  v_vac={V_VAC}, λ={LAM:.6e}, α={ALPHA:.6e}")
print(f"  Relax={RELAX_STEPS}, Free={FREE_STEPS}, total={total_steps}")
if EXPAND:
    print(f"  Expansion: start={EXPAND_START}, rate={EXPAND_RATE:.4e}")
    print(f"    a_max={A_MAX}")
    step_a2 = EXPAND_START + int(1.0 / EXPAND_RATE)
    print(f"    a=2.0 at step {step_a2}")
else:
    print(f"  Expansion: OFF (static universe)")
print(f"  Tag: {TAG}")
print(f"{'═'*65}")

# ── Scale factor ──

def get_scale_factor(step):
    if not EXPAND or step < EXPAND_START:
        return 1.0
    return min(1.0 + EXPAND_RATE * (step - EXPAND_START), A_MAX)

# ── Grid + vortices ──

print("\n  Creating vortex pair...")
_x = xp.linspace(0, L, N_GRID, dtype=xp.float32)
_y = xp.linspace(0, L, N_GRID, dtype=xp.float32)
_z = xp.linspace(0, L, N_GRID, dtype=xp.float32)
X, Y, Z = xp.meshgrid(_x, _y, _z, indexing='ij')
del _x, _y, _z

x1, x2 = center - SEP / 2, center + SEP / 2
y1, y2 = center, center

Psi = xp.ones((N_GRID, N_GRID, N_GRID), dtype=xp.complex64) * V_VAC
Psi_dot = xp.zeros_like(Psi)

for vx, vy, nw, dep, lab in [(x1, y1, N1, DEPTH1, "V1"),
                               (x2, y2, N2, DEPTH2, "V2")]:
    r = xp.sqrt((X - vx)**2 + (Y - vy)**2) + 1e-10
    th = xp.arctan2(Y - vy, X - vx)
    core = (1.0 - dep) + dep * xp.tanh(r / XI)
    phase = xp.exp(1j * nw * th)
    Psi *= core * phase
    print(f"  {lab}: ({vx:.1f}, {vy:.1f}) n={nw:+d} depth={dep}")
    del r, th, core, phase

print(f"  |Ψ|: [{float(xp.min(xp.abs(Psi))):.4f}, "
      f"{float(xp.max(xp.abs(Psi))):.4f}]")
del X, Y, Z
if GPU:
    cp.get_default_memory_pool().free_all_blocks()
print("  Ready.\n")

# ── Physics ──

def lap3d(f, dx_t):
    r = xp.roll(f, 1, 0); r += xp.roll(f, -1, 0)
    r += xp.roll(f, 1, 1); r += xp.roll(f, -1, 1)
    r += xp.roll(f, 1, 2); r += xp.roll(f, -1, 2)
    r -= 6 * f
    r /= (dx_t * dx_t)
    return r

def ed_rhs(psi, dx_t):
    l = lap3d(psi, dx_t)
    l += LAM * psi - ALPHA * xp.abs(psi)**2 * psi
    l *= C**2
    return l

# ── Tracking ──

def find_core(Psi, nom, other=None, sr=80, mj=8.0):
    amp = to_np(xp.abs(Psi))
    sl = amp[:, :, N_GRID // 2]
    xi_i, yi_i = int(nom[0] / dx0), int(nom[1] / dx0)
    x0r = max(xi_i - sr, 0); x1r = min(xi_i + sr, N_GRID)
    y0r = max(yi_i - sr, 0); y1r = min(yi_i + sr, N_GRID)
    reg = sl[x0r:x1r, y0r:y1r]
    if reg.size == 0:
        return nom[0], nom[1], V_VAC
    loc = np.unravel_index(np.argmin(reg), reg.shape)
    px = (x0r + loc[0]) * dx0
    py = (y0r + loc[1]) * dx0
    lm = float(reg[loc])
    if np.sqrt((px - nom[0])**2 + (py - nom[1])**2) > mj:
        return nom[0], nom[1], lm
    if other:
        do = np.sqrt((px - other[0])**2 + (py - other[1])**2)
        ds = np.sqrt((px - nom[0])**2 + (py - nom[1])**2)
        if do < ds and do < 5:
            return nom[0], nom[1], lm
    return px, py, lm

def meas_wind(Psi, cx, cy, rad):
    ph = to_np(xp.angle(Psi[:, :, N_GRID // 2]))
    ns = max(int(8 * rad / dx0), 60)
    angs = np.linspace(0, 2 * np.pi, ns, endpoint=False)
    ci, cj = cx / dx0, cy / dx0
    ps = []
    for a in angs:
        ix = int(round(ci + (rad / dx0) * np.cos(a))) % N_GRID
        iy = int(round(cj + (rad / dx0) * np.sin(a))) % N_GRID
        ps.append(ph[ix, iy])
    ps = np.array(ps)
    dp = np.diff(ps)
    dp = np.where(dp > np.pi, dp - 2 * np.pi, dp)
    dp = np.where(dp < -np.pi, dp + 2 * np.pi, dp)
    ld = ps[0] - ps[-1]
    ld -= 2 * np.pi * round(ld / (2 * np.pi))
    return float((np.sum(dp) + ld) / (2 * np.pi))

def safe_winding_radius(dist):
    sr = min(3 * XI, dist * 0.4)
    sr = max(sr, XI)
    if dist < 2 * XI:
        sr = -1
    return sr

# ── Evolution loop ──

print(f"  Evolving {total_steps} steps...\n")
history = []
nom1, nom2 = [x1, y1], [x2, y2]
t0 = time.time()

for step in range(total_steps):
    a_t = get_scale_factor(step)
    dx_t = dx0 * a_t

    acc = ed_rhs(Psi, dx_t)

    if step < RELAX_STEPS:
        acc -= GAMMA * Psi_dot

    # Kick at phase transition
    if step == RELAX_STEPS and KICK > 0:
        print(f"  ★ KICK step {step} ({time.time()-t0:.1f}s) k={KICK}")
        _x = xp.linspace(0, L, N_GRID, dtype=xp.float32)
        _y = xp.linspace(0, L, N_GRID, dtype=xp.float32)
        _z = xp.linspace(0, L, N_GRID, dtype=xp.float32)
        _X, _Y, _Z = xp.meshgrid(_x, _y, _z, indexing='ij')
        for pos, sgn in [(nom1, +1), (nom2, -1)]:
            dsq = (_X - pos[0])**2 + (_Y - pos[1])**2 + (_Z - center)**2
            m = xp.exp(-dsq / (2 * kick_env**2))
            Psi *= xp.exp(1j * sgn * KICK * (_Y - pos[1]) * m)
            del dsq, m
        del _X, _Y, _Z, _x, _y, _z
        if GPU:
            cp.get_default_memory_pool().free_all_blocks()
        print(f"    |Ψ|: [{float(xp.min(xp.abs(Psi))):.4f}, "
              f"{float(xp.max(xp.abs(Psi))):.4f}]\n")

    # Leapfrog
    Psi_dot += DT * acc
    Psi += DT * Psi_dot
    del acc

    # Measurement
    if step % MEASURE_EVERY == 0:
        p1x, p1y, a1 = find_core(Psi, nom1, nom2)
        p2x, p2y, a2 = find_core(Psi, nom2, nom1)
        d = np.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)
        cx_n = (p1x + p2x) / 2
        cy_n = (p1y + p2y) / 2
        ang = np.degrees(np.arctan2(p1y - cy_n, p1x - cx_n))
        nom1, nom2 = [p1x, p1y], [p2x, p2y]

        rec = {
            'step': int(step),
            'dist': round(float(d), 4),
            'p1x': round(float(p1x), 3), 'p1y': round(float(p1y), 3),
            'p2x': round(float(p2x), 3), 'p2y': round(float(p2y), 3),
            'amp1': round(float(a1), 4), 'amp2': round(float(a2), 4),
            'angle': round(float(ang), 2),
            'a_t': round(float(a_t), 5),
        }

        if step % WINDING_EVERY == 0:
            try:
                sr = safe_winding_radius(d)
                if sr > 0:
                    rec['w1'] = round(meas_wind(Psi, p1x, p1y, sr), 3)
                    rec['w2'] = round(meas_wind(Psi, p2x, p2y, sr), 3)
                    rec['w_rad'] = round(float(sr), 3)
                else:
                    rec['w1'] = None; rec['w2'] = None
                    rec['w_rad'] = 0; rec['w_note'] = 'OVERLAP'
            except:
                pass
        history.append(rec)

        if step % PRINT_EVERY == 0:
            el = time.time() - t0
            ph = "RELAX" if step < RELAX_STEPS else "FREE"
            ws = ""
            if 'w1' in rec and rec['w1'] is not None:
                ws = f"  w=({rec['w1']:+.2f},{rec['w2']:+.2f})"
            a_str = f"  a={a_t:.3f}" if a_t > 1.001 else ""
            print(f"  [{ph}] {step:5d} ({el:6.1f}s): "
                  f"d={d:7.2f}  amp=({a1:.2f},{a2:.2f}){ws}{a_str}")

# ── Post-analysis ──

elapsed = time.time() - t0
h = history
steps_h = np.array([r['step'] for r in h])
dist_h = np.array([r['dist'] for r in h])
p1x_h = np.array([r['p1x'] for r in h])
p1y_h = np.array([r['p1y'] for r in h])
p2x_h = np.array([r['p2x'] for r in h])
p2y_h = np.array([r['p2y'] for r in h])
a_t_h = np.array([r.get('a_t', 1.0) for r in h])
fm = steps_h >= RELAX_STEPS

print(f"\n{'═'*65}")
print(f"  RESULT: {TAG}")
print(f"  n₁={N1:+d} n₂={N2:+d} | depth=({DEPTH1},{DEPTH2}) | ξ={XI}")
print(f"  sep={SEP}  kick={KICK}")
if EXPAND:
    print(f"  Expansion: start={EXPAND_START}, rate={EXPAND_RATE:.4e}")
    print(f"  Final a(t) = {get_scale_factor(total_steps):.3f}")
else:
    print(f"  Expansion: OFF")
print(f"{'═'*65}")

md = float(np.min(dist_h[fm])) if np.any(fm) else float(np.min(dist_h))
if md < 2 * XI:
    print(f"  ★ MERGER (min_d={md:.2f} < 2ξ={2*XI:.1f})")
print(f"  Dist: init={dist_h[0]:.2f}  final={dist_h[-1]:.2f}")

a1h = np.array([r['amp1'] for r in h])
a_free = a1h[fm] if np.any(fm) else a1h
print(f"  Amp: min={np.min(a_free):.2f}  max={np.max(a_free):.2f}  "
      f"final={a_free[-1]:.2f}")

wr = [(r['step'], r.get('w1'), r.get('w2'))
      for r in h if r.get('w1') is not None]
if wr:
    print(f"  Winding: first=({wr[0][1]:+.2f},{wr[0][2]:+.2f})  "
          f"last=({wr[-1][1]:+.2f},{wr[-1][2]:+.2f})")

print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

# ── Save JSON ──

out = {
    'params': {
        'n1': N1, 'n2': N2, 'depth1': DEPTH1, 'depth2': DEPTH2,
        'sep': SEP, 'kick': KICK, 'xi': XI,
        'N': N_GRID, 'L': L, 'lam': LAM, 'alpha': ALPHA, 'v_vac': V_VAC,
        'dt': DT, 'relax': RELAX_STEPS, 'free': FREE_STEPS,
        'expand': EXPAND,
        'expand_start': EXPAND_START if EXPAND else None,
        'expand_rate': EXPAND_RATE if EXPAND else 0,
        'a_max': A_MAX if EXPAND else 1.0,
    },
    'summary': {
        'min_dist': round(float(md), 3),
        'final_dist': round(float(dist_h[-1]), 3),
        'amp_min': round(float(np.min(a_free)), 3),
        'amp_max': round(float(np.max(a_free)), 3),
        'amp_final': round(float(a_free[-1]), 3),
        'final_a': round(get_scale_factor(total_steps), 4),
        'time_s': round(elapsed, 1),
    },
    'history': history,
}
jp = os.path.join(OUTDIR, f'binary_{TAG}.json')
with open(jp, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: {jp}")

# ── Save drive log ──

try:
    dp = os.path.join(OUTDIR, f"ringdown_{TAG}.txt")
    with open(dp, 'w') as f:
        f.write(f"ED Binary Vortex Merger — {TAG}\n")
        f.write(f"{'='*60}\n")
        f.write(f"n1={N1:+d} n2={N2:+d} depth=({DEPTH1},{DEPTH2}) xi={XI}\n")
        f.write(f"sep={SEP} kick={KICK}\n")
        f.write(f"Grid={N_GRID}^3, L={L}, v_vac={V_VAC}\n")
        if EXPAND:
            f.write(f"Expansion: start={EXPAND_START} rate={EXPAND_RATE} "
                    f"a_max={A_MAX}\n")
            f.write(f"Final a = {get_scale_factor(total_steps):.4f}\n")
        else:
            f.write(f"Expansion: OFF (static universe)\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"{'Step':>6}  {'amp1':>7}  {'amp2':>7}  {'dist':>7}  "
                f"{'a(t)':>6}\n")
        f.write(f"{'-'*40}\n")
        for r in history:
            if r['step'] % PRINT_EVERY == 0:
                f.write(f"{r['step']:>6}  {r['amp1']:7.4f}  {r['amp2']:7.4f}  "
                        f"{r['dist']:7.2f}  {r.get('a_t',1.0):6.3f}\n")
    print(f"  Saved: {dp}")
except Exception as e:
    print(f"  Drive log: {e}")

print(f"\n{'═'*65}")
print(f"  DONE: {TAG}")
print(f"{'═'*65}")
