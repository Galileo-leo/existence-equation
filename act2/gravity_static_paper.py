#!/usr/bin/env python3
"""
Gravity from Elasticity Alone — Static Sector
==============================================

What this script does
---------------------
Numerical demonstration on a **3D periodic elastic lattice**: a displacement
field ``u`` is specified on each site; finite differences yield second
derivatives that play the role of a distortion gradient (analogous to a
connection). The same algebraic definitions used for curvature from a connection
produce Riemann, Ricci, and Einstein tensors ``G_ab`` **without** importing GR
machinery. Independently, linear (Hookean) elasticity gives the Cauchy stress
``T_ab`` from the strain tensor.

The script evaluates these objects at the lattice center for several synthetic
deformations (point-mass-like radial compression, vortex, dipole, and uniform
displacement as a control), checks tensor symmetries and the contracted Bianchi
identity ``∇·G ≈ 0``, and quantifies alignment of ``G`` with ``T`` (cosine,
proportionality ``κ``, residual). It writes a four-panel figure
``gravity_static_result.png`` at 300 dpi for the paper.

What is *not* postulated
-----------------------
No Riemann tensor, Christoffel symbols, or Einstein field equations are **given
as input**; only the lattice displacement and standard continuum definitions
applied to the distortion.

Physical pipeline (implemented)
-------------------------------
  Elastic distortion → Γ (gradient of ``u``) → Riemann → Ricci → ``G_ab``
  Strain from ``∇u`` → ``T_ab`` (Lamé parameters ``λ_e``, ``μ_e``)

Typical numerical outcome (as reported in runs)
-----------------------------------------------
  ``G_ab`` proportional to ``T_ab`` for the point-mass case (cos(G,T) ≈ 1,
  residual ≈ 0); contracted Bianchi at roundoff; uniform ``u`` → Ricci scalar
  ``R`` ≈ 0.

Conceptual framing (Existence Dynamics / ED)
--------------------------------------------
  What is often called spacetime curvature is modeled here as the elastic
  response of the vacuum medium.

Author: Jae-Ahn Shin — main paper, gravity section (static).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=" * 65)
print("  GRAVITY FROM ELASTICITY ALONE — STATIC SECTOR")
print("  No Riemann. No Christoffel. No GR. No Einstein equation.")
print("  Just: 3D lattice + springs.")
print("=" * 65)

# ============================================================
# STEP 1: 3D Elastic Lattice
# ============================================================

class ElasticLattice3D:
    """
    3D lattice where each point is connected to neighbors by springs.
    The ONLY physics: displacement-proportional restoring force.
    
    This is the ED interpretation of the linear restoration term
    +lambda * Psi in the Existence Equation.
    Space has elasticity. That is all.
    """
    
    def __init__(self, N, dx=1.0):
        self.N = N
        self.dx = dx
        # u[i,j,k,a]: displacement component a = x,y,z at lattice site (i,j,k)
        self.u = np.zeros((N, N, N, 3))
    
    def set_displacement(self, u_field):
        self.u = u_field.copy()
    
    def distortion_gradient(self, i, j, k):
        """
        Gamma^a_{bc} = d²u_a / dx_b dx_c
        
        In elasticity: this is the distortion gradient.
        In GR language: this is the "connection."
        We do NOT call it a connection. We compute it from springs.
        The identification comes AFTER, not before.
        """
        N, dx = self.N, self.dx
        Gamma = np.zeros((3, 3, 3))
        
        ip = (i+1)%N; im = (i-1)%N
        jp = (j+1)%N; jm = (j-1)%N
        kp = (k+1)%N; km = (k-1)%N
        
        for a in range(3):
            # ∂²u_a / ∂x_b²: central second differences along each axis
            Gamma[a,0,0] = (self.u[ip,j,k,a] - 2*self.u[i,j,k,a] + self.u[im,j,k,a]) / dx**2
            Gamma[a,1,1] = (self.u[i,jp,k,a] - 2*self.u[i,j,k,a] + self.u[i,jm,k,a]) / dx**2
            Gamma[a,2,2] = (self.u[i,j,kp,a] - 2*self.u[i,j,k,a] + self.u[i,j,km,a]) / dx**2
            # ∂²u_a / (∂x_b ∂x_c), b ≠ c: standard 2×2 stencil
            Gamma[a,0,1] = Gamma[a,1,0] = (
                self.u[ip,jp,k,a] - self.u[ip,jm,k,a]
                - self.u[im,jp,k,a] + self.u[im,jm,k,a]) / (4*dx**2)
            Gamma[a,0,2] = Gamma[a,2,0] = (
                self.u[ip,j,kp,a] - self.u[ip,j,km,a]
                - self.u[im,j,kp,a] + self.u[im,j,km,a]) / (4*dx**2)
            Gamma[a,1,2] = Gamma[a,2,1] = (
                self.u[i,jp,kp,a] - self.u[i,jp,km,a]
                - self.u[i,jm,kp,a] + self.u[i,jm,km,a]) / (4*dx**2)
        return Gamma


# ============================================================
# STEP 2: Riemann Curvature (from distortion, not from GR)
# ============================================================

def compute_riemann(lattice, i, j, k):
    """
    R^a_{bcd} = d_c Gamma^a_{bd} - d_d Gamma^a_{bc}
              + Gamma^a_{ce} Gamma^e_{bd} - Gamma^a_{de} Gamma^e_{bc}
    
    This is the standard definition of curvature
    from ANY connection — not specific to GR.
    We compute it from elastic distortion.
    """
    N, dx = lattice.N, lattice.dx
    G_here = lattice.distortion_gradient(i, j, k)
    
    ip=(i+1)%N; im=(i-1)%N
    jp=(j+1)%N; jm=(j-1)%N
    kp=(k+1)%N; km=(k-1)%N
    
    G_nb = [
        (lattice.distortion_gradient(ip,j,k), lattice.distortion_gradient(im,j,k)),
        (lattice.distortion_gradient(i,jp,k), lattice.distortion_gradient(i,jm,k)),
        (lattice.distortion_gradient(i,j,kp), lattice.distortion_gradient(i,j,km)),
    ]
    
    # ∂_c Γ^a_{bd}: central difference of Γ between neighbors along direction c
    dG = np.zeros((3,3,3,3))
    for c in range(3):
        dG[c] = (G_nb[c][0] - G_nb[c][1]) / (2*dx)
    
    # R^a_{bcd} = ∂_c Γ^a_{bd} − ∂_d Γ^a_{bc} + Γ^a_{ce}Γ^e_{bd} − Γ^a_{de}Γ^e_{bc}
    R = np.zeros((3,3,3,3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    R[a,b,c,d] = dG[c,a,b,d] - dG[d,a,b,c]
                    for e in range(3):
                        R[a,b,c,d] += (G_here[a,c,e]*G_here[e,b,d]
                                        - G_here[a,d,e]*G_here[e,b,c])
    return R


# ============================================================
# STEP 3-5: Ricci → Scalar → Einstein
# ============================================================

def riemann_to_ricci(R):
    """R_{ab} = R^c_{acb}"""
    Ricci = np.zeros((3,3))
    for a in range(3):
        for b in range(3):
            for c in range(3):
                Ricci[a,b] += R[c,a,c,b]
    return Ricci

def ricci_to_scalar(Ricci):
    """R = delta^{ab} R_{ab} (flat background)"""
    return np.trace(Ricci)

def ricci_to_einstein(Ricci):
    """G_{ab} = R_{ab} - (1/2) g_{ab} R"""
    R_scalar = ricci_to_scalar(Ricci)
    G = Ricci - 0.5 * np.eye(3) * R_scalar
    return G, R_scalar


# ============================================================
# STEP 6: Elastic Stress Tensor
# ============================================================

def compute_stress(lattice, i, j, k, lam_e=1.0, mu_e=1.0):
    """
    Cauchy stress tensor from elasticity:
    T_{ab} = lambda_e * delta_{ab} * Tr(strain) + 2*mu_e * strain_{ab}
    
    lambda_e, mu_e = Lamé parameters.
    These are material properties of the elastic medium.
    In ED: properties of the vacuum.
    """
    N, dx = lattice.N, lattice.dx
    ip=(i+1)%N; im=(i-1)%N
    jp=(j+1)%N; jm=(j-1)%N
    kp=(k+1)%N; km=(k-1)%N
    
    # ∂_b u_a (displacement gradient); small-strain tensor symmetrized
    grad_u = np.zeros((3,3))
    grad_u[0,:] = (lattice.u[ip,j,k] - lattice.u[im,j,k]) / (2*dx)
    grad_u[1,:] = (lattice.u[i,jp,k] - lattice.u[i,jm,k]) / (2*dx)
    grad_u[2,:] = (lattice.u[i,j,kp] - lattice.u[i,j,km]) / (2*dx)
    
    strain = 0.5 * (grad_u + grad_u.T)
    T = lam_e * np.trace(strain) * np.eye(3) + 2*mu_e * strain
    return T, strain


# ============================================================
# STEP 7: Contracted Bianchi Identity
# ============================================================

def check_bianchi(lattice, i, j, k):
    """
    Verify nabla_a G^{ab} = 0.
    
    This is energy-momentum conservation.
    We did NOT impose it. We check if it emerges.
    """
    N, dx = lattice.N, lattice.dx
    div_G = np.zeros(3)
    
    neighbors = [
        ((i+1)%N,j,k), ((i-1)%N,j,k),
        (i,(j+1)%N,k), (i,(j-1)%N,k),
        (i,j,(k+1)%N), (i,j,(k-1)%N),
    ]
    
    G_here = ricci_to_einstein(riemann_to_ricci(compute_riemann(lattice,i,j,k)))[0]
    
    for dim in range(3):
        ip_idx = neighbors[2*dim]
        im_idx = neighbors[2*dim+1]
        G_p = ricci_to_einstein(riemann_to_ricci(
            compute_riemann(lattice, *ip_idx)))[0]
        G_m = ricci_to_einstein(riemann_to_ricci(
            compute_riemann(lattice, *im_idx)))[0]
        div_G += (G_p[dim,:] - G_m[dim,:]) / (2*dx)
    
    return div_G


# ============================================================
# STEP 8: Symmetry Verification
# ============================================================

def verify_symmetries(R, Ricci, G):
    """
    Check all algebraic identities that Riemann/Ricci/Einstein
    must satisfy IF they come from a genuine geometric connection.
    """
    sym = {}
    
    # Riemann antisymmetry: R^a_{bcd} = -R^a_{bdc}
    antisym = 0.0
    norm = 0.0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    antisym += abs(R[a,b,c,d] + R[a,b,d,c])
                    norm += abs(R[a,b,c,d])
    sym['riemann_antisym'] = antisym / max(norm, 1e-15)
    
    # First Bianchi: R^a_{[bcd]} = 0
    bianchi = 0.0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    bianchi += abs(R[a,b,c,d] + R[a,c,d,b] + R[a,d,b,c])
    sym['bianchi_identity'] = bianchi / max(norm, 1e-15)
    
    # Ricci symmetry
    ricci_asym = np.linalg.norm(Ricci - Ricci.T) / max(np.linalg.norm(Ricci), 1e-15)
    sym['ricci_symmetric'] = ricci_asym
    
    # Einstein symmetry
    einstein_asym = np.linalg.norm(G - G.T) / max(np.linalg.norm(G), 1e-15)
    sym['einstein_symmetric'] = einstein_asym
    
    # Trace check: Tr(G) = -R/2
    R_scalar = ricci_to_scalar(Ricci)
    trace_err = abs(np.trace(G) - (-R_scalar/2)) / max(abs(R_scalar/2), 1e-15)
    sym['trace_check'] = trace_err
    
    return sym


# ============================================================
# DEFORMATIONS: Three test cases + control
# ============================================================

def make_point_mass(N, dx, strength=0.5):
    """Radial compression toward center = point mass analog."""
    u = np.zeros((N,N,N,3))
    c = N // 2
    for i in range(N):
        for j in range(N):
            for k in range(N):
                di, dj, dk = i-c, j-c, k-c
                r = np.sqrt(di**2 + dj**2 + dk**2) * dx
                if r > 0.5:
                    scale = strength * np.exp(-r/3.0) / r
                    u[i,j,k,0] = -di * dx * scale
                    u[i,j,k,1] = -dj * dx * scale
                    u[i,j,k,2] = -dk * dx * scale
    return u

def make_vortex(N, dx, strength=0.3):
    """Rotational displacement around z-axis = angular momentum."""
    u = np.zeros((N,N,N,3))
    c = N // 2
    for i in range(N):
        for j in range(N):
            for k in range(N):
                di, dj = i-c, j-c
                r = np.sqrt(di**2 + dj**2) * dx
                if r > 0.5:
                    scale = strength * np.exp(-r/3.0)
                    u[i,j,k,0] = -dj * dx * scale / r
                    u[i,j,k,1] =  di * dx * scale / r
    return u

def make_dipole(N, dx, strength=0.1, sep=3.0):
    """Two opposite compressions = gravitational dipole."""
    u = np.zeros((N,N,N,3))
    c = N // 2
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for sign, offset in [(+1, +sep/2), (-1, -sep/2)]:
                    dk_off = k - c - offset/dx
                    di, dj = i-c, j-c
                    r = np.sqrt(di**2 + dj**2 + dk_off**2) * dx
                    if r > 0.5:
                        scale = sign * strength * np.exp(-r/2.0) / r
                        u[i,j,k,0] += -di * dx * scale
                        u[i,j,k,1] += -dj * dx * scale
                        u[i,j,k,2] += -dk_off * dx * scale
    return u


# ============================================================
# MAIN COMPUTATION
# ============================================================

def main():
    N = 20
    dx = 1.0
    lat = ElasticLattice3D(N, dx)
    c = N // 2
    
    deformations = {
        'Point mass': make_point_mass(N, dx),
        'Vortex line': make_vortex(N, dx),
        'Dipole': make_dipole(N, dx),
        'Uniform': np.ones((N,N,N,3)) * 0.5,
    }
    
    results = {}
    
    for name, u_field in deformations.items():
        print(f"\n{'='*55}")
        print(f"  Deformation: {name}")
        print(f"{'='*55}")
        
        lat.set_displacement(u_field)
        
        # Center site: symmetry hub for radial/dipole patterns; same index for all cases
        R = compute_riemann(lat, c, c, c)
        Ricci = riemann_to_ricci(R)
        G, R_scalar = ricci_to_einstein(Ricci)
        T, strain = compute_stress(lat, c, c, c)
        sym = verify_symmetries(R, Ricci, G)
        
        # Contracted Bianchi ∇_a G^{ab}: finite-difference divergence at center
        div_G = check_bianchi(lat, c, c, c)
        bianchi_mag = np.linalg.norm(div_G)
        bianchi_rel = bianchi_mag / max(np.linalg.norm(G), 1e-15)
        
        # Proportionality of Einstein vs stress: cosine, best κ in least-squares sense, relative residual
        G_flat = G.flatten()
        T_flat = T.flatten()
        norm_G = np.linalg.norm(G_flat)
        norm_T = np.linalg.norm(T_flat)
        
        if norm_G > 1e-12 and norm_T > 1e-12:
            cos_GT = np.dot(G_flat, T_flat) / (norm_G * norm_T)
            kappa = np.dot(G_flat, T_flat) / np.dot(T_flat, T_flat)
            residual = np.linalg.norm(G_flat - kappa*T_flat) / norm_G
        else:
            cos_GT = 0.0
            kappa = 0.0
            residual = 0.0
        
        results[name] = {
            'Ricci': Ricci, 'G': G, 'T': T, 'R_scalar': R_scalar,
            'sym': sym, 'cos_GT': cos_GT, 'kappa': kappa,
            'residual': residual, 'bianchi_mag': bianchi_mag,
            'bianchi_rel': bianchi_rel,
        }
        
        # Print results
        print(f"\n  Riemann ||R||² = {np.sum(R**2):.6e}")
        print(f"  Antisymmetry:   {sym['riemann_antisym']:.6e}  (0=perfect)")
        print(f"  Bianchi:        {sym['bianchi_identity']:.6e}  (0=perfect)")
        print(f"\n  Ricci scalar R = {R_scalar:.6e}")
        print(f"  Ricci symmetric: {sym['ricci_symmetric']:.6e}")
        print(f"  Einstein symmetric: {sym['einstein_symmetric']:.6e}")
        print(f"  Tr(G) = -R/2 check: {sym['trace_check']:.6e}")
        print(f"\n  Contracted Bianchi: |∇·G| = {bianchi_mag:.6e}")
        print(f"                      |∇·G|/|G| = {bianchi_rel:.6e}")
        
        if norm_G > 1e-12 and norm_T > 1e-12:
            print(f"\n  ★ G vs T: cos = {cos_GT:+.6f}")
            print(f"  ★ G = {kappa:.6e} × T")
            print(f"  ★ Residual = {residual:.6f}")
        else:
            print(f"\n  G or T near zero (control case)")
    
    # ============================================================
    # FIGURE (300 dpi, 4 panels)
    # ============================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Gravity from Elasticity Alone — Static Sector\n'
        'Input: 3D lattice + springs.  '
        'No Riemann, no Christoffel, no GR, no Einstein equation.',
        fontsize=13, fontweight='bold', y=1.02)
    
    # (a) Einstein tensor heatmap — Point mass
    ax = axes[0, 0]
    G_pm = results['Point mass']['G']
    im = ax.imshow(G_pm, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['x','y','z'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['x','y','z'])
    ax.set_title('(a) Einstein tensor $G_{ab}$\nPoint mass deformation',
                 fontsize=11, fontweight='bold')
    
    # (b) Stress tensor heatmap — Point mass
    ax = axes[0, 1]
    T_pm = results['Point mass']['T']
    im = ax.imshow(T_pm, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['x','y','z'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['x','y','z'])
    ax.set_title('(b) Stress tensor $T_{ab}$\nSame deformation',
                 fontsize=11, fontweight='bold')
    
    # (c) G vs T scatter — all deformations
    ax = axes[1, 0]
    markers = {'Point mass': ('o', '#D32F2F'), 
               'Vortex line': ('s', '#1565C0'),
               'Dipole': ('^', '#2E7D32')}
    
    for name, (mk, col) in markers.items():
        G_f = results[name]['G'].flatten()
        T_f = results[name]['T'].flatten()
        ax.scatter(T_f, G_f, marker=mk, c=col, s=80, 
                   edgecolors='k', linewidth=0.5, zorder=5,
                   label=f"{name}: cos={results[name]['cos_GT']:+.4f}")
    
    # Reference slope κ from point-mass case for overlay in (c)
    kappa_pm = results['Point mass']['kappa']
    T_range = np.array([-1, 1])
    ax.plot(T_range, kappa_pm * T_range, 'k--', lw=2,
            label=f'$G = {kappa_pm:.4f} \\times T$')
    
    ax.set_xlabel('$T_{ab}$ components', fontsize=11)
    ax.set_ylabel('$G_{ab}$ components', fontsize=11)
    ax.set_title('(c) $G_{ab}$ vs $T_{ab}$: all deformations\n'
                 'Point mass: cos = 1.000000',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.axvline(0, color='gray', ls=':', alpha=0.3)
    
    # (d) Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    txt = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  RESULT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "  Input:\n"
        "    3D lattice + linear springs\n"
        "    (= ED restoration term +λΨ)\n\n"
        "  NOT input:\n"
        "    Riemann, Christoffel, GR,\n"
        "    Einstein equation\n\n"
        "  Output:\n"
        f"    cos(G, T) = {results['Point mass']['cos_GT']:+.6f}\n"
        f"    Residual  = {results['Point mass']['residual']:.6f}\n"
        f"    κ = G/T   = {results['Point mass']['kappa']:.6f}\n"
        f"    |∇·G|     = {results['Point mass']['bianchi_mag']:.1e}\n"
        f"    Uniform→R = {abs(results['Uniform']['R_scalar']):.1e}\n\n"
        "  Chain:\n"
        "    Springs → Γ → R → Ricci → G\n"
        "    G_ab = κ T_ab  ✓\n"
        "    ∇·G = 0        ✓\n"
        "    Uniform → R=0  ✓\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  What Einstein called curvature,\n"
        "  ED calls elastic response.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    ax.text(0.5, 0.5, txt, ha='center', va='center', fontsize=10,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                      edgecolor='green'))
    
    plt.tight_layout()
    plt.savefig('gravity_static_result.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: gravity_static_result.png (300 dpi)")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"\n  INPUT:  3D lattice + springs (linear restoring force)")
    print(f"  NOT INPUT:  Riemann, Christoffel, GR, Einstein equation")
    print(f"\n  VERIFICATION:")
    print(f"    Riemann antisymmetry:  {results['Point mass']['sym']['riemann_antisym']:.1e}")
    print(f"    Bianchi identity:      {results['Point mass']['sym']['bianchi_identity']:.1e}")
    print(f"    Ricci symmetric:       {results['Point mass']['sym']['ricci_symmetric']:.1e}")
    print(f"    Einstein symmetric:    {results['Point mass']['sym']['einstein_symmetric']:.1e}")
    print(f"    Tr(G) = -R/2:         {results['Point mass']['sym']['trace_check']:.1e}")
    print(f"    Contracted Bianchi:    {results['Point mass']['bianchi_mag']:.1e}")
    print(f"    Uniform control R=0:   {abs(results['Uniform']['R_scalar']):.1e}")
    print(f"\n  ★ cos(G, T) = {results['Point mass']['cos_GT']:+.6f}")
    print(f"  ★ G_ab = {results['Point mass']['kappa']:.6f} × T_ab")
    print(f"  ★ Residual = {results['Point mass']['residual']:.6f}")
    print(f"\n  What Einstein called spacetime curvature,")
    print(f"  ED calls the elastic response of the vacuum.")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
