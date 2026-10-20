module

public import Transformer.Causal.Section4_SubspaceGeometry

/-!
# A linear lift of the original single-token dynamics

ArXiv:2411.04990v2, §3, `lemma1`, and §4, `thm2`.
An explicit positive integrating factor removes the radial projection
from the original equation. Its lifted trajectory solves the shifted
linear value equation. Under the manuscript's invariant leading splitting,
the leading projection of this lift is exactly its initial projection.
The construction does not assume existence, convergence, or a Jordan form.
-/

@[expose] public section

open Real

namespace Transformer.Causal

variable {d : ℕ}

/-- The positive radial integrating factor for the original projected
value equation, shifted by the leading eigenvalue.
Source: arXiv:2411.04990v2, §3, the linear-equation representation in `lemma1`. -/
noncomputable def singleTokenScale (V : ParamMatrix d) (lam : ℝ)
    (x : ℝ → EucSpace d) (t : ℝ) : ℝ :=
  Real.exp (∫ s in (0 : ℝ)..t, inner ℝ (x s) (V (x s)) - lam)

/-- The actual positive radial lift used in the spectral argument for
`lemma1` and `thm2` of arXiv:2411.04990v2, §§3–4. -/
noncomputable def singleTokenLift (V : ParamMatrix d) (lam : ℝ)
    (x : ℝ → EucSpace d) (t : ℝ) : EucSpace d :=
  singleTokenScale V lam x t • x t

/-- The integrating factor is positive at every real time, independently
of the trajectory. Source: arXiv:2411.04990v2, §3, `lemma1`. -/
theorem singleTokenScale_pos (V : ParamMatrix d) (lam : ℝ)
    (x : ℝ → EucSpace d) (t : ℝ) : 0 < singleTokenScale V lam x t :=
  Real.exp_pos _

/-- The original radial coefficient is continuous, so the integrating
factor has its ordinary derivative at every time.
Source: arXiv:2411.04990v2, §3, `lemma1`. -/
theorem hasDerivAt_singleTokenScale (V : ParamMatrix d) (lam : ℝ)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x) (t : ℝ) :
    HasDerivAt (singleTokenScale V lam x)
      (singleTokenScale V lam x t * (inner ℝ (x t) (V (x t)) - lam)) t := by
  have hc : Continuous x := continuous_iff_continuousAt.mpr fun s => (hx s).continuousAt
  have hq : Continuous (fun s => inner ℝ (x s) (V (x s)) - lam) :=
    (hc.inner (V.continuous.comp hc)).sub continuous_const
  exact (intervalIntegral.integral_hasDerivAt_right (hq.intervalIntegrable _ _)
    (hq.stronglyMeasurableAtFilter _ _) hq.continuousAt).exp

/-- Multiplying the actual original solution by its explicit integrating
factor gives `y' = V y - λ y`. Source: arXiv:2411.04990v2, §3, `lemma1`.
This is an identity for the original ODE, not an auxiliary assumed flow. -/
theorem hasDerivAt_singleTokenLift (V : ParamMatrix d) (lam : ℝ)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x) (t : ℝ) :
    HasDerivAt (singleTokenLift V lam x)
      (V (singleTokenLift V lam x t) - lam • singleTokenLift V lam x t) t := by
  have h := (hasDerivAt_singleTokenScale V lam x hx t).smul (hx t)
  convert! h using 1
  simp only [singleTokenLift, map_smul, proj, smul_sub, smul_smul]
  module

/-- The lifted leading component is exactly the original initial leading
component at every real time. Source: arXiv:2411.04990v2, §§3–4,
`lemma1` and `thm2`. The actual invariant complement is required. -/
theorem starProjection_singleTokenLift (V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (x : ℝ → EucSpace d) (hx : singleTokenODE d V x) (t : ℝ) :
    L.starProjection (singleTokenLift V lam x t) = L.starProjection (x 0) := by
  have hd (s : ℝ) : HasDerivAt (fun u => L.starProjection (singleTokenLift V lam x u)) 0 s := by
    have h := (hasDerivAt_const s L.starProjection).clm_apply
      (hasDerivAt_singleTokenLift V lam x hx s)
    simpa! only [zero_apply, zero_add, map_sub, map_smul,
      starProjection_value_eq V L lam heig hinv, sub_self] using h
  have hconst := is_const_of_deriv_eq_zero (fun s => (hd s).differentiableAt)
    (fun s => (hd s).deriv) t 0
  simpa only [singleTokenLift, singleTokenScale, intervalIntegral.integral_same,
    Real.exp_zero, one_smul] using hconst

/-- A nonzero stationary eigenvector realizes both derivative statements
and the invariant leading-splitting assumptions, at an arbitrary shift.
Source: arXiv:2411.04990v2, §§3–4, `lemma1` and `thm2`. -/
example (lam : ℝ) : ∃ (V : ParamMatrix 2) (L : Submodule ℝ (EucSpace 2))
    (x : ℝ → EucSpace 2),
    (∀ v ∈ L, V v = lam • v) ∧ (∀ z ∈ Lᗮ, V z ∈ Lᗮ) ∧
    singleTokenODE 2 V x ∧ L.starProjection (x 0) ≠ 0 := by
  let u : EucSpace 2 := basePoint 1
  have hu : ‖u‖ = 1 := mem_sphere_zero_iff_norm.mp (basePoint 1).2
  refine ⟨lam • ContinuousLinearMap.id ℝ _, ⊤, fun _ => u, ?_, ?_, ?_, ?_⟩
  · intro v _
    rfl
  · intro z hz
    simpa only [smul_apply, ContinuousLinearMap.id_apply]
      using (⊤ : Submodule ℝ (EucSpace 2))ᗮ.smul_mem lam hz
  · intro t
    simpa only [smul_apply, ContinuousLinearMap.id_apply,
      proj_smul_self hu] using hasDerivAt_const t u
  · simpa only [Submodule.starProjection_top, ContinuousLinearMap.id_apply]
      using norm_ne_zero_iff.mp (hu ▸ one_ne_zero)

end Transformer.Causal
