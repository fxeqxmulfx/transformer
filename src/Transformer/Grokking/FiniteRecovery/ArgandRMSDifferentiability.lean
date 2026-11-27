module

public import Transformer.Grokking.FiniteRecovery.ArgandAttention
public import Mathlib.Analysis.InnerProductSpace.Calculus

/-! The norm-plus-offset RMS in experiments/phasor_value_transport/ARGAND.md
is continuously differentiable in real coordinates, including at zero. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Asymptotics Transformer.PhasorV2
open scoped Topology

private theorem hasFDerivAt_normSmul_zero {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] :
    HasFDerivAt (fun x : E => ‖x‖ • x) (0 : E →L[ℝ] E) 0 := by
  rw [hasFDerivAt_iff_isLittleO]
  simp only [norm_zero, zero_smul, sub_zero, zero_apply]
  have hn : Tendsto (norm : E → ℝ) (𝓝 0) (𝓝 0) := by
    simpa only [norm_zero] using (continuous_norm.continuousAt (x := (0 : E))).tendsto
  simpa only [one_smul] using
    ((isLittleO_one_iff ℝ).2 hn).smul_isBigO (isBigO_refl (fun x : E => x) (𝓝 0))

private theorem contDiff_normSmul {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] :
    ContDiff ℝ 1 (fun x : E => ‖x‖ • x) := by
  have haway {x : E} (hx : x ≠ 0) : ContDiffAt ℝ 1 (fun y : E => ‖y‖ • y) x :=
    (contDiffAt_norm ℂ hx).smul contDiffAt_id
  have hdiff : Differentiable ℝ (fun x : E => ‖x‖ • x) := fun x => by
    by_cases hx : x = 0
    · subst x
      exact hasFDerivAt_normSmul_zero.differentiableAt
    · exact (haway hx).differentiableAt one_ne_zero
  refine contDiff_one_iff_fderiv.mpr ⟨hdiff, continuous_iff_continuousAt.mpr fun x => ?_⟩
  by_cases hx : x = 0
  · subst x
    have hb (y : E) : ‖fderiv ℝ (fun z : E => ‖z‖ • z) y‖ ≤ 2 * ‖y‖ := by
      by_cases hy : y = 0
      · subst y
        rw [hasFDerivAt_normSmul_zero.fderiv]
        simp
      · have hn : HasFDerivAt norm (fderiv ℝ norm y) y :=
          ((contDiffAt_norm ℂ hy : ContDiffAt ℝ 1 (norm : E → ℝ) y).differentiableAt one_ne_zero).hasFDerivAt
        have hdF : HasFDerivAt (fun z : E => ‖z‖ • z)
            (‖y‖ • ContinuousLinearMap.id ℝ E + (fderiv ℝ norm y).smulRight y) y :=
          hn.smul (hasFDerivAt_id y)
        rw [hdF.fderiv]
        refine (norm_add_le _ _).trans ?_
        rw [norm_smul, Real.norm_of_nonneg (norm_nonneg _), ContinuousLinearMap.norm_smulRight_apply]
        have hd : ‖fderiv ℝ norm y‖ ≤ 1 := by
          simpa using norm_fderiv_le_of_lipschitz (x₀ := y) ℝ (lipschitzWith_one_norm (E := E))
        have hi : ‖ContinuousLinearMap.id ℝ E‖ ≤ 1 := ContinuousLinearMap.norm_id_le
        linarith [mul_le_mul_of_nonneg_left hi (norm_nonneg y), mul_le_mul_of_nonneg_right hd (norm_nonneg y)]
    unfold ContinuousAt
    rw [hasFDerivAt_normSmul_zero.fderiv]
    apply tendsto_zero_iff_norm_tendsto_zero.mpr
    refine squeeze_zero (fun _ => norm_nonneg _) hb ?_
    simpa only [norm_zero, mul_zero] using ((continuous_norm.continuousAt (x := (0 : E))).tendsto).const_mul 2
  · exact (haway hx).continuousAt_fderiv one_ne_zero

private theorem contDiff_radial {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    {ε : ℝ} (hε : 0 < ε) : ContDiff ℝ 1 (fun x : E => (‖x‖ + ε)⁻¹ • x) := by
  apply contDiff_iff_contDiffAt.mpr
  intro x
  by_cases hx : x = 0
  · subst x
    have hden : ContDiffAt ℝ 1 (fun x : E => ε ^ 2 - ‖x‖ ^ 2) 0 :=
      contDiffAt_const.sub (contDiff_norm_sq ℂ).contDiffAt
    have hn : (ε ^ 2 - ‖(0 : E)‖ ^ 2) ≠ 0 := by simpa using pow_ne_zero 2 hε.ne'
    have hp : ContDiffAt ℝ 1 (fun x : E => ε • x - ‖x‖ • x) 0 :=
      (contDiffAt_id.const_smul ε).sub contDiff_normSmul.contDiffAt
    apply ((hden.inv hn).smul hp).congr_of_eventuallyEq
    have hlim : Tendsto (norm : E → ℝ) (𝓝 0) (𝓝 0) := by
      simpa only [norm_zero] using (continuous_norm.continuousAt (x := (0 : E))).tendsto
    have hnear : ∀ᶠ y : E in 𝓝 0, ‖y‖ < ε := hlim.eventually (gt_mem_nhds hε)
    filter_upwards [hnear] with y hy
    change (‖y‖ + ε)⁻¹ • y = (ε ^ 2 - ‖y‖ ^ 2)⁻¹ • (ε • y - ‖y‖ • y)
    rw [← sub_smul, smul_smul]
    congr 1
    have ha : ε - ‖y‖ ≠ 0 := (sub_pos.mpr hy).ne'
    have hb : ‖y‖ + ε ≠ 0 := (add_pos_of_nonneg_of_pos (norm_nonneg _) hε).ne'
    rw [show ε ^ 2 - ‖y‖ ^ 2 = (ε - ‖y‖) * (‖y‖ + ε) by ring]
    field_simp
  · exact (((contDiffAt_norm ℂ hx).add contDiffAt_const).inv
      (add_pos_of_nonneg_of_pos (norm_nonneg _) hε).ne').smul contDiffAt_id

/-- The exact complex RMS map is C¹ with respect to real parameters. -/
theorem contDiff_argandRMS {c : ℕ} {ε : ℝ} (hε : 0 < ε) :
    ContDiff ℝ 1 (argandRMS ε : CSpace c → CSpace c) := by
  have he (x : CSpace c) : argandRMS ε x = Real.sqrt c • ((‖x‖ + ε)⁻¹ • x) := by
    rw [argandRMS, smul_smul, div_eq_mul_inv]
    rfl
  change ContDiff ℝ 1 (fun x : CSpace c => argandRMS ε x)
  simp_rw [he]
  exact (contDiff_radial hε).const_smul _

example := contDiff_argandRMS (c := 64) (ε := 1 / 1000000) (by norm_num)

end Transformer.Grokking.FiniteRecovery.Argand
