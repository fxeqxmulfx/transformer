module

public import Transformer.Normalization.Section3_LogarithmicClock
public import Transformer.Normalization.Section3_RadialComparison
public import Transformer.Normalization.Velocities
public import Transformer.Perspective.SphereFlowLimit

/-!
# §3 — Critical angular limits for Pre-LN

Karagodin, Polyanskiy, Rigollet, arXiv:2510.22026v2, supplement,
Appendix D, proof of Theorem 3.1, Steps 1–2.

For an actual Pre-LN solution, positive upper and lower bounds on `r_j(t)/t`
give a critical limiting configuration. The logarithmic clock makes the
angular equation a bounded positive block scaling of the gradient of the
analytic spherical energy. Sphere invariance and the equation are derived
from `SchemeDynamics`; compactness or convergence is not assumed.
The positive energy of `Perspective` ascends; the source descends its
negative, with the same equations and coefficients.

This is the deterministic implication on the event isolated in Step 2.
It does not assert that a Gaussian sample satisfies that event, nor that
the limiting critical point is synchronized. The source's Steps 3–5 require
a separate argument about unstable critical points and null basins.
-/

@[expose] public section

open Real Filter MeasureTheory
open scoped Topology BigOperators

namespace Transformer.Normalization

open Perspective

variable {d n : ℕ} {β τ T a b : ℝ} {α : ℝ → ℝ}
  {θ : ℝ → Idx n → EucSpace d} {r : ℝ → Idx n → ℝ}

/-- The actual Pre-LN directions converge to a critical spherical tuple
whenever their rescaled radii stay in `[a,b]`, `0 < a ≤ b`, after time `T`.

The angular gradient coefficients lie in
`[(b n exp |β|)⁻¹, (a n exp (-|β|))⁻¹]`. Applying the forward spherical
Łojasiewicz theorem and undoing the logarithmic clock gives the limit at
the original times. The proof requires only `β ≠ 0`; the source uses positive
temperature. Source: arXiv:2510.22026v2, Appendix D, Steps 1–2. -/
theorem SchemeDynamics.pre_tendsto_isCritical_of_scaled_bounds
    (hD : SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ .pre θ r)
    (hd : 1 ≤ d) (hn : 0 < n) (hβ : β ≠ 0) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (hT : 0 < T) (ha : 0 < a) (hab : a ≤ b)
    (hq : ∀ u, 0 ≤ u → ∀ j, a ≤ rescaledRadius T r u j ∧ rescaledRadius T r u j ≤ b) :
    ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ j, Tendsto (fun t => θ t j) atTop (𝓝 (Xstar j : EucSpace d)) := by
  have hb : 0 < b := ha.trans_le hab
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hunit (u : ℝ) (j : Idx n) : ‖θ (logTime T u) j‖ = 1 :=
    hD.norm_eq_one (Or.inl rfl) h0 _ (logTime_pos hT u).le j
  let X : ℝ → SphereTuple d n := fun u j =>
    ⟨θ (logTime T u) j, mem_sphere_zero_iff_norm.mpr (hunit u j)⟩
  let Z := partitionSA d n β X
  let c : ℝ → Idx n → ℝ := fun u j => (rescaledRadius T r u j * Z u j)⁻¹
  let lo := (b * (n * Real.exp |β|))⁻¹
  let hi := (a * (n * Real.exp (-|β|)))⁻¹
  have hlo : 0 < lo := by dsimp [lo]; positivity
  have hZ (u : ℝ) (j : Idx n) :
      n * Real.exp (-|β|) ≤ Z u j ∧ Z u j ≤ n * Real.exp |β| :=
    partitionSA_mem_Icc β X u j
  have hbound (u : ℝ) (hu : 0 ≤ u) (j : Idx n) : lo ≤ c u j ∧ c u j ≤ hi := by
    have hqpos := ha.trans_le (hq u hu j).1
    have hZpos : 0 < Z u j := (by positivity : (0 : ℝ) < n * Real.exp (-|β|)).trans_le
      (hZ u j).1
    have hlow : a * (n * Real.exp (-|β|)) ≤ rescaledRadius T r u j * Z u j :=
      mul_le_mul (hq u hu j).1 (hZ u j).1 (by positivity) hqpos.le
    have hupp : rescaledRadius T r u j * Z u j ≤ b * (n * Real.exp |β|) :=
      mul_le_mul (hq u hu j).2 (hZ u j).2 hZpos.le hb.le
    exact ⟨inv_anti₀ (mul_pos hqpos hZpos) hupp,
      inv_anti₀ (by positivity) hlow⟩
  have hlohi : lo ≤ hi := (hbound 0 le_rfl ⟨0, hn⟩).1.trans (hbound 0 le_rfl ⟨0, hn⟩).2
  have hangle (u : ℝ) :
      HasDerivAt (fun v => (WithLp.toLp 2 fun j => (X v j : EucSpace d) : TupleSpace d n))
        (blockScale (c u) (gradient (tupleEnergy hd β)
          (WithLp.toLp 2 fun j => (X u j : EucSpace d)))) u := by
    apply hasDerivAt_toLp_tuple_iff.mpr
    intro j
    rw [blockScale_apply, gradient_tupleEnergy_toLp_apply hd hβ]
    change HasDerivAt (fun v => θ (logTime T v) j) _ u
    apply ((hD.pre_logTime_equations hT u j).1).congr_deriv
    simp only [c, Z, X, partitionSA, attentionVec, idParams, ContinuousLinearMap.id_apply,
      mul_inv, proj_smul, smul_smul]
  obtain ⟨ystar, hstar, hgrad, hlim⟩ := exists_tendsto_of_hasDerivAt_blockScale_forward
    hd hlo hlohi hbound (y := fun u => WithLp.toLp 2 fun j => (X u j : EucSpace d))
    (fun u _ j => hunit u j) (fun u _ => hangle u)
  let Xstar : SphereTuple d n := fun j => ⟨ystar.ofLp j, mem_sphere_zero_iff_norm.mpr (hstar j)⟩
  have hys : ystar = WithLp.toLp 2 fun j => (Xstar j : EucSpace d) := rfl
  refine ⟨Xstar, fun j => ?_, fun j => ?_⟩
  · have hj := congrArg (fun v : TupleSpace d n => v.ofLp j) hgrad
    simpa only [hys, gradient_tupleEnergy_toLp_apply hd hβ, PiLp.zero_apply] using hj
  · apply (tendsto_comp_logTime_iff hT).mp
    exact ((PiLp.continuous_apply 2 (fun _ : Idx n => EucSpace d) j).tendsto ystar).comp hlim

/-- A unit identity singleton with `r(t)=1+t` realizes every hypothesis:
for `u ≥ 0`, its rescaled radius is between one and two. -/
example : ∃ Xstar : SphereTuple 2 1, IsCriticalEBeta 2 1 1 Xstar ∧
    ∀ j, Tendsto (fun _ : ℝ => (basePoint 1 : EucSpace 2)) atTop (𝓝 (Xstar j : EucSpace 2)) := by
  have hD := schemeDynamics_singleton (d := 2) 1 1 0 0 (basePoint 1)
    (by simp [basePoint, PiLp.norm_single]) .pre (Or.inl rfl)
  apply hD.pre_tendsto_isCritical_of_scaled_bounds (by decide) one_pos one_ne_zero
    (fun _ => by simp [basePoint, PiLp.norm_single]) (T := 1) one_pos (a := 1) (b := 2)
    one_pos (by norm_num)
  intro u hu j
  have he : 1 ≤ Real.exp u := Real.one_le_exp_iff.mpr hu
  simp only [rescaledRadius, logTime, one_mul]
  constructor
  · rw [le_div_iff₀ (Real.exp_pos u)]
    linarith
  · rw [div_le_iff₀ (Real.exp_pos u)]
    linarith

/-- Positive radial derivative lower limits alone supply the scaled-radius
bounds needed by Step 2. The bound `|r_j'| ≤ 1` follows from unit directions
and the convex-combination attention vector along the actual Pre-LN flow.
Source: arXiv:2510.22026v2, Appendix D, Steps 1–2. This proves existence of a
critical limit, while synchronization still requires Steps 3–5. -/
theorem SchemeDynamics.pre_tendsto_isCritical_of_pos_radial_liminf
    (hD : SchemeDynamics d n β (idParams d) (idParams d) (idParams d) α τ .pre θ r)
    (hd : 1 ≤ d) (hn : 0 < n) (hβ : β ≠ 0) (h0 : ∀ j, ‖θ 0 j‖ = 1)
    (hlim : ∀ j, 0 < liminf (fun t => radialDerivative d n β
      (idParams d) (idParams d) (idParams d) θ τ .pre t j) atTop) :
    ∃ Xstar : SphereTuple d n, IsCriticalEBeta d n β Xstar ∧
      ∀ j, Tendsto (fun t => θ t j) atTop (𝓝 (Xstar j : EucSpace d)) := by
  have hvel (t : ℝ) (ht : 0 ≤ t) (j : Idx n) :
      |radialDerivative d n β (idParams d) (idParams d) (idParams d) θ τ .pre t j| ≤ 1 := by
    have hunit := hD.norm_eq_one (Or.inl rfl) h0 t ht
    have hA := norm_attentionVec_le_one d n β (idParams d t) (idParams d t) (idParams d t)
      ContinuousLinearMap.norm_id_le (θ t) hunit j
    have hinner := abs_real_inner_le_norm (θ t j)
      (attentionVec d n β (idParams d t) (idParams d t) (idParams d t) (θ t) j)
    rw [hunit j, one_mul] at hinner
    exact hinner.trans hA
  obtain ⟨a, b, T, ha, hab, hT, hq⟩ := exists_rescaled_bounds_of_pos_liminf hn r
    (fun t j => radialDerivative d n β (idParams d) (idParams d) (idParams d) θ τ .pre t j)
    (fun t ht j => (hD t ht j).2) hvel hlim
  exact hD.pre_tendsto_isCritical_of_scaled_bounds hd hn hβ h0 hT ha hab hq

/-- The genuine identity singleton realizes the complete positive-lower-limit
hypothesis: its radial velocity is identically one. -/
example : ∃ Xstar : SphereTuple 2 1, IsCriticalEBeta 2 1 1 Xstar ∧
    ∀ j, Tendsto (fun _ : ℝ => (basePoint 1 : EucSpace 2)) atTop (𝓝 (Xstar j : EucSpace 2)) := by
  have hD := schemeDynamics_singleton (d := 2) 1 1 0 0 (basePoint 1)
    (by simp [basePoint, PiLp.norm_single]) .pre (Or.inl rfl)
  apply hD.pre_tendsto_isCritical_of_pos_radial_liminf (by decide) one_pos one_ne_zero
    (fun _ => by simp [basePoint, PiLp.norm_single])
  intro j
  have heq : (fun t => radialDerivative 2 1 1 (idParams 2) (idParams 2) (idParams 2)
      (fun _ _ => (basePoint 1 : EucSpace 2)) 0 .pre t j) = fun _ => (1 : ℝ) := by
    funext t
    simp [radialDerivative, attentionVec_singleton, idParams, basePoint, PiLp.norm_single]
  rw [heq, Filter.liminf_const]
  exact one_pos

end Transformer.Normalization
