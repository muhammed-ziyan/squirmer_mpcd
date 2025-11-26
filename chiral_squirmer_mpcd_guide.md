
# MPCD Squirmer With Azimuthal Slip — Complete Simulation Guide

A practical, implementation‑ready guide to turn a standard squirmer into a **chiral-path swimmer** (circle/helix) in an MPCD/SRD fluid **without any external forces or torques**. The curvature comes from an **internal azimuthal slip** component, misaligned with the propulsion axis.

---

## 0) What You’re Building (In One Line)

A spherical squirmer with standard meridional slip (B1, B2) plus a weak, **misaligned azimuthal slip** \(C_1\) that makes the swimmer **spin about an internal axis** not collinear with propulsion \(\Rightarrow\) constant‑velocity **helix/circle** in bulk. All momentum/torque exchanges come from surface slip; externally it’s force/torque‑free.

---

## 1) Physics Model (Continuous)

- **Geometry**: sphere of radius \(a\). Body-frame unit axes:
  - \(\hat{\mathbf p}\): propulsion axis (the usual squirmer axis)
  - \(\hat{\mathbf n}\): swirl axis
  - Misalignment angle \(\alpha=\angle(\hat{\mathbf p},\hat{\mathbf n})\neq0\)

- **Surface slip (vector form; no singular bases):**
  \[
  \mathbf u_{\text{slip}}(\hat{\mathbf r})=
  \underbrace{\big[B_1 + B_2(\hat{\mathbf p}\!\cdot\!\hat{\mathbf r})\big]\big(\hat{\mathbf p}-(\hat{\mathbf p}\!\cdot\!\hat{\mathbf r})\hat{\mathbf r}\big)}_{\text{standard squirmer}}
  \;+\;
  \underbrace{C_1\,(\hat{\mathbf n}\times \hat{\mathbf r})}_{\text{azimuthal (chiral) term}}
  \]
  where \(\hat{\mathbf r}\) is the outward surface normal.

- **Useful kinematics (bulk, far from walls):**
  - Swim speed (no swirl): \(U_0 \approx \tfrac{2}{3}B_1\).
  - Body spin scales with swirl: \(\Omega_0 \propto C_1/a\) (measure once).
  - Helix geometry (if \(U_0,\Omega_0\) roughly constant):
    \[
    R=\frac{U_0\sin\alpha}{\Omega_0},\qquad
    P=\frac{2\pi U_0\cos\alpha}{\Omega_0}.
    \]
  - \( \beta=B_2/B_1 \) toggles pusher/puller; it modulates hydrodynamics but the bulk helix is set mainly by \(C_1,\alpha\).

---

## 2) MPCD/SRD Fluid — Safe, Standard Parameters

Use reduced units \(a_0=1\) (cell length), \(m=1\), \(k_BT=1\).

- **Collision rule**: SRD+a (angular momentum conserving variant if you want better no‑slip).
- **Rotation angle**: \(\alpha_{\text{SRD}} \in [110^\circ,130^\circ]\) (130° common).
- **Collision period**: \(h\approx 0.1\).
- **Mean particles per cell**: \(n_0\in[5,10]\) (10 is robust).
- **Mean free path**: \(\lambda = h\sqrt{k_BT/m}\); keep \(\lambda/a_0\lesssim 0.1{-}0.3\).
- **Domain**: periodic box; choose \(L\ge 8a\) and also \(L \gtrsim 4R\) and \(L_z \gtrsim 2P\) to avoid self‑interaction via PBC.
- **Viscosity**: measure once (Couette/Poiseuille) to get \(\eta\). Check
  \[
  \text{Re}=\frac{\rho U_0 a}{\eta}\ll 1,\qquad
  \text{Ma}=\frac{U_0}{c_s}\ll 0.1.
  \]

---

## 3) Squirmer–Fluid Coupling (How Slip Enters)

Rigid-body surface velocity:
\[
\mathbf u_{\text{wall}}(\hat{\mathbf r}) = \mathbf V + \boldsymbol\Omega\times(a\hat{\mathbf r}) + \mathbf u_{\text{slip}}(\hat{\mathbf r}).
\]

Two robust enforcement routes:

### A) Ghost (Virtual) Particles — *Best No‑Slip Quality*
- During **collision**, fill the squirmer volume (cells overlapped by the sphere) with ghost particles whose Maxwellian **mean** is \(\mathbf u_{\text{wall}}\) at the nearest surface (or a smooth interior extension matching the boundary).
- Include ghosts in SRD collisions; apply equal‑and‑opposite total impulse/torque accumulated from ghost‑fluid exchanges to the squirmer.
- During **streaming**, prevent penetration via specular/bounce‑back (normal component only).

**Pros**: excellent no‑slip incl. tangential; easy to thermostat.  
**Cons**: a bit more code.

### B) Deterministic Surface Scattering on Impacts — *Simpler*
When a solvent particle crosses the spherical boundary during **streaming**:

1. Compute contact point \(a\hat{\mathbf r}\) and \(\mathbf u_{\text{wall}}(\hat{\mathbf r})\).
2. Relative pre‑impact velocity: \(\mathbf v_{\text{rel}}=\mathbf v-\mathbf u_{\text{wall}}\).
3. Decompose: \(\mathbf v_n=(\mathbf v_{\text{rel}}\!\cdot\!\hat{\mathbf r})\hat{\mathbf r}\); \(\mathbf v_t=\mathbf v_{\text{rel}}-\mathbf v_n\).
4. **No‑slip scattering**: set
   \[
   \mathbf v'=\mathbf u_{\text{wall}}-\mathbf v_n.
   \]
   (Optionally add small Gaussian noise \(\boldsymbol\xi_t\) tangent to the surface for thermal roughness.)
5. Impulse to fluid: \(\Delta\mathbf p_f=m(\mathbf v'-\mathbf v)\). Apply equal and opposite to the squirmer:
   \[
   \Delta \mathbf P_{\text{sq}}=-\Delta\mathbf p_f,\qquad
   \Delta \mathbf L_{\text{sq}}=-(a\hat{\mathbf r})\times \Delta\mathbf p_f.
   \]

This keeps the swimmer **externally** force‑/torque‑free; motion arises from **slip‑driven hydrodynamic stresses**.

**Implementation tip**: use the **vector** slip formulas; avoid normalizations/divisions that blow up at poles:
- \( \mathbf u_{\theta}=(\mathbf I-\hat{\mathbf r}\hat{\mathbf r}^\top)\hat{\mathbf p}\) times \((B_1+B_2\,\hat{\mathbf p}\!\cdot\!\hat{\mathbf r})\)
- \( \mathbf u_{\phi}^{(\text{swirl})}= C_1\,(\hat{\mathbf n}\times \hat{\mathbf r}) \)

---

## 4) Integrators & Orientation Handling

- Update \(\mathbf V,\boldsymbol\Omega\) from accumulated impulses/torques each step; integrate position and quaternion \(Q\); **re‑normalize \(Q\)**.
- Recompute \(\hat{\mathbf p},\hat{\mathbf n}\) each step from \(Q\). Keep \(\alpha\) **fixed in the body frame** (i.e., \(\hat{\mathbf n}\) is a constant vector in that frame at angle \(\alpha\) to \(\hat{\mathbf p}\)).

---

## 5) “Golden” Starting Values (Work Out of the Box)

- **Geometry**: \(a=4a_0\) (≈ 8 cells across the diameter).
- **Fluid**: \(n_0=10\), \(\alpha_{\text{SRD}}=130^\circ\), \(h=0.1\), \(k_BT=1\), \(m=1\).
- **Squirmer**: \(B_1=0.05\Rightarrow U_0\approx 0.033\); \( \beta=B_2/B_1=0 \) (neutral).
- **Chirality**: \(\alpha=20^\circ\) (axis misalignment), \(C_1=0.02\) (moderate spin).

This gives a gentle helix in bulk. Increase \(C_1\) or \(\alpha\) to tighten curvature; for near‑planar circles, let \(\alpha\to 90^\circ\).

---

## 6) Step‑by‑Step Runbook (Coding/Run Order)

1. **Plain squirmer sanity** \((C_1=0)\):
   - Verify constant \(U_0\approx 2B_1/3\).
   - Check total momentum conservation and thermal stability.

2. **Spin‑only check** \((C_1\neq 0,\ \alpha=0^\circ)\):
   - Straight translation + body spin about \(\hat{\mathbf p}\).
   - Measure \(\Omega_0(C_1)\) (linear at small \(C_1\)).

3. **Helix** \((C_1\neq 0,\ \alpha>0)\):
   - Track COM \(\mathbf X(t)\) and body axes.
   - Fit \(R,P\) using the geometry formulas and compare to the trajectory.

4. **Parameter map**:
   - Sweep \(C_1\in[0,0.06]\), \(\alpha\in[5^\circ,60^\circ]\) to tabulate \(R,P\).
   - (Optional) Vary \(\beta\) to probe robustness.

5. **Box effects**:
   - Increase \(L\) until \(R,P\) change \< 1% (finite‑size convergence).

6. **Viscosity & low‑Re checks**:
   - Measure \(\eta\); ensure \(\text{Re}\ll 1\), \(\text{Ma}\ll 0.1\).
   - If not, reduce \(B_1,C_1\) or increase \(\eta\) (e.g., higher \(n_0\) or SRD angle closer to \(180^\circ\)).

---

## 7) Diagnostics to Log

- Time series: \(U(t)=\|\mathbf V\|\), \(\Omega(t)=\|\boldsymbol\Omega\|\).
- Body‑frame angles: \(\angle(\boldsymbol\Omega,\hat{\mathbf n})\), \(\angle(\mathbf V,\hat{\mathbf p})\).
- Helix fit: instantaneous radius \(R(t)\), pitch \(P(t)\), and centerline axis.
- Conservation: total linear and angular momentum (fluid + body) constant in periodic bulk.
- Surface work rate (sanity): approximate \( \int_S \mathbf t\cdot \mathbf u_{\text{slip}}\,\mathrm dS \) via accumulated impulses; should be positive and steady.

---

## 8) Common Failure Modes (and Fixes)

- **No curvature**: \(\alpha=0\) or \(C_1\) too small; verify \(\Omega_0>0\).
- **Unsteady/wobbling path**: mean free path too large; reduce \(h\) or increase \(n_0\). Increase sphere resolution (bigger \(a\)).
- **Heating/cooling at surface**: if adding tangential noise, apply the **same** opposite stochastic impulse to the body; otherwise thermostat after streaming.
- **Momentum leaks**: every surface bounce must apply the exact opposite impulse (and torque) to the squirmer—never “teleport” velocities.
- **Box artifacts**: if helix aligns with lattice directions, enlarge box or randomize initial orientation.

---

## 9) Minimal Slip Implementation (Copy‑Paste Formulas)

At a contact normal \(\hat{\mathbf r}\) (unit), with body axes \(\hat{\mathbf p},\hat{\mathbf n}\):

- \(s=\hat{\mathbf p}\!\cdot\!\hat{\mathbf r}\)
- \(\mathbf u_{\theta} = \hat{\mathbf p}-s\hat{\mathbf r}\)
- \(\mathbf u_{\text{polar}} = (B_1 + B_2 s)\,\mathbf u_{\theta}\)
- \(\mathbf u_{\text{chiral}} = C_1\,(\hat{\mathbf n}\times \hat{\mathbf r})\)
- \(\mathbf u_{\text{slip}} = \mathbf u_{\text{polar}}+\mathbf u_{\text{chiral}}\)
- \(\mathbf u_{\text{wall}} = \mathbf V + \boldsymbol\Omega\times(a\hat{\mathbf r}) + \mathbf u_{\text{slip}}\)

**Scattering rule (Route B):**
\[
\mathbf v' = \mathbf u_{\text{wall}} - \big[(\mathbf v-\mathbf u_{\text{wall}})\!\cdot\!\hat{\mathbf r}\big]\hat{\mathbf r},\qquad
\Delta\mathbf p_f = m(\mathbf v'-\mathbf v),\qquad
\Delta \mathbf L_{\text{sq}} = - (a\hat{\mathbf r})\times \Delta\mathbf p_f.
\]

---

## 10) Circles Without Bulk Chirality (Optional Control)

Run the same swimmer with \(C_1=0\) at a controlled gap \(h\sim 1.1{-}1.5\,a\) from a no‑slip wall or around a large passive sphere. Pure hydrodynamic coupling produces circling/orbiting; use this as a control to validate your surface coupling and force/torque bookkeeping.

---

## TL;DR Checklist

- Use the **vector** slip definition (no poles, no divisions).
- Misalign swirl and propulsion axes by \(\alpha>0\).
- Start with \(C_1 \lesssim B_1\); calibrate \(\Omega_0(C_1)\).
- Enforce wall velocity with ghost particles *or* deterministic scattering.
- Apply equal‑and‑opposite impulses/torques to the squirmer every time you modify a fluid particle at the surface.
- Verify **straight + spin** for \(\alpha=0\) before turning on \(\alpha>0\).
- Fit and confirm helix radius \(R\) and pitch \(P\) against trajectory.

