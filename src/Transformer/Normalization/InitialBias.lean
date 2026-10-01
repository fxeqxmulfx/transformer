/-
# Bias of the exponential spherical tilt

The expectation estimate in Appendix B, proof of Theorem 4.2 of
arXiv:2510.22026v2, obtained from the first two spherical moments.
-/

import Transformer.Normalization.InitialSphereMoments
import Mathlib.Analysis.Complex.ExponentialBounds

open MeasureTheory

namespace Transformer.Normalization

variable {d : ℕ}

/-- The tilted unit vector appearing in the attention numerator.
Source: arXiv:2510.22026v2, Appendix B, proof of Theorem 4.2. -/
noncomputable def tiltedSphere (z : EucSpace d) (x : SSphere d) : EucSpace d :=
  Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) • (x : EucSpace d)

/-- The spherical exponential tilt is continuous. Source:
arXiv:2510.22026v2, Appendix B, the numerator in Theorem 4.2. -/
theorem continuous_tiltedSphere (z : EucSpace d) : Continuous (tiltedSphere z) := by
  unfold tiltedSphere
  fun_prop

/-- A contracted query gives uniformly bounded summands. Source:
arXiv:2510.22026v2, Appendix B, the vector concentration step. -/
theorem norm_tiltedSphere_le (z : EucSpace d) (hz : ‖z‖ ≤ 1) (x : SSphere d) :
    ‖tiltedSphere z x‖ ≤ 3 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.property
  have hi : inner (𝕜 := ℝ) z (x : EucSpace d) ≤ 1 :=
    (le_abs_self _).trans ((abs_real_inner_le_norm _ _).trans (by simpa [hx] using hz))
  rw [tiltedSphere, norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), hx, mul_one]
  exact (Real.exp_le_exp.mpr hi).trans (by linarith [Real.exp_one_lt_d9])

/-- The norm bound on the query is realized by the zero vector. -/
example : ‖(0 : EucSpace 1)‖ ≤ 1 := by simp

/-- The mean tilted vector has norm at most `2/d`. This uses centering,
isotropy, and `|exp(u)-1| ≤ 2|u|` for `|u| ≤ 1`, so no higher moments are
needed. Source: arXiv:2510.22026v2, Appendix B, the expectation estimate
in the proof of Theorem 4.2. -/
theorem norm_integral_tiltedSphere_le (hd : 0 < d) (μ : Measure (SSphere d))
    [IsProbabilityMeasure μ]
    (hμ : ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
      μ.map (Perspective.sphereMap d U) = μ) (z : EucSpace d) (hz : ‖z‖ ≤ 1) :
    ‖∫ x, tiltedSphere z x ∂μ‖ ≤ 2 / (d : ℝ) := by
  let m := ∫ x, tiltedSphere z x ∂μ
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  by_cases hm : m = 0
  · change ‖m‖ ≤ _
    rw [hm, norm_zero]
    positivity
  have hnm : 0 < ‖m‖ := norm_pos_iff.mpr hm
  let v : EucSpace d := ‖m‖⁻¹ • m
  have hv : ‖v‖ = 1 := by
    dsimp [v]
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hnm), inv_mul_cancel₀ hnm.ne']
  have hvm : inner (𝕜 := ℝ) v m = ‖m‖ := by
    dsimp [v]
    rw [real_inner_smul_left, real_inner_self_eq_norm_sq]
    field_simp
  have hc : Continuous (fun x : SSphere d =>
      (Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) - 1) • (x : EucSpace d)) := by fun_prop
  have hmean : (∫ x : SSphere d,
      (Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) - 1) • (x : EucSpace d) ∂μ) = m := by
    simp_rw [sub_smul, one_smul]
    change (∫ x : SSphere d, tiltedSphere z x - (x : EucSpace d) ∂μ) = m
    rw [integral_sub (Perspective.integrable_of_continuous_compact (continuous_tiltedSphere z) μ)
      (Perspective.integrable_of_continuous_compact continuous_subtype_val μ),
      integral_sphere_eq_zero μ hμ, sub_zero]
  have hinner : inner (𝕜 := ℝ) v m = ∫ x : SSphere d,
      (Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) - 1) *
        inner (𝕜 := ℝ) v (x : EucSpace d) ∂μ := by
    rw [← hmean]
    simpa only [innerSL_apply_apply, real_inner_smul_right] using
      ((innerSL ℝ v).integral_comp_comm
        (Perspective.integrable_of_continuous_compact hc μ)).symm
  have hpoint : ∀ x : SSphere d,
      (Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) - 1) *
        inner (𝕜 := ℝ) v (x : EucSpace d) ≤
      inner (𝕜 := ℝ) z (x : EucSpace d) ^ 2 + inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2 := by
    intro x
    have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.property
    have hu : |inner (𝕜 := ℝ) z (x : EucSpace d)| ≤ 1 :=
      (abs_real_inner_le_norm _ _).trans (by simpa [hx] using hz)
    calc _ ≤ |(Real.exp (inner (𝕜 := ℝ) z (x : EucSpace d)) - 1) *
        inner (𝕜 := ℝ) v (x : EucSpace d)| := le_abs_self _
      _ ≤ 2 * |inner (𝕜 := ℝ) z (x : EucSpace d)| *
          |inner (𝕜 := ℝ) v (x : EucSpace d)| := by
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_right (Real.abs_exp_sub_one_le hu) (abs_nonneg _)
      _ ≤ _ := by
        nlinarith [sq_nonneg (|inner (𝕜 := ℝ) z (x : EucSpace d)| -
          |inner (𝕜 := ℝ) v (x : EucSpace d)|),
          sq_abs (inner (𝕜 := ℝ) z (x : EucSpace d)),
          sq_abs (inner (𝕜 := ℝ) v (x : EucSpace d))]
  change ‖m‖ ≤ _
  rw [← hvm, hinner]
  calc _ ≤ ∫ x : SSphere d,
        inner (𝕜 := ℝ) z (x : EucSpace d) ^ 2 + inner (𝕜 := ℝ) v (x : EucSpace d) ^ 2 ∂μ :=
      integral_mono (Perspective.integrable_of_continuous_compact (by fun_prop) μ)
        (Perspective.integrable_of_continuous_compact (by fun_prop) μ) hpoint
    _ = ‖z‖ ^ 2 / (d : ℝ) + 1 / (d : ℝ) := by
      rw [integral_add (Perspective.integrable_of_continuous_compact (by fun_prop) μ)
        (Perspective.integrable_of_continuous_compact (by fun_prop) μ),
        integral_sphere_inner_sq hd μ hμ z, integral_sphere_inner_sq hd μ hμ v, hv, one_pow]
    _ ≤ 2 / (d : ℝ) := by
      have hz2 : ‖z‖ ^ 2 ≤ 1 := by nlinarith [norm_nonneg z]
      have h := (div_le_div_of_nonneg_right hz2 hd'.le)
      calc _ ≤ 1 / (d : ℝ) + 1 / (d : ℝ) := add_le_add h le_rfl
        _ = _ := by ring

/-- The two-point uniform law and the zero query realize all hypotheses. -/
example : 0 < (1 : ℕ) ∧ IsProbabilityMeasure oneDimUniform ∧
    (∀ U : EucSpace 1 ≃ₗᵢ[ℝ] EucSpace 1,
      oneDimUniform.map (Perspective.sphereMap 1 U) = oneDimUniform) ∧
    ‖(0 : EucSpace 1)‖ ≤ 1 :=
  ⟨by decide, inferInstance, oneDimUniform_invariant, by simp⟩

end Transformer.Normalization
