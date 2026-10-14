/-
# Certified inward signs of the original anisotropic follower

Source: arXiv:2411.04990v2, §4, Conjecture `thm1.5`, and
`eqn:csa`, at `V=diag(2,1)`, `Q=I`, `K=diag(0,4)`, `β=1`.
When the leader has first coordinate at least `99/100` and second
coordinate of absolute value at most `1/100`, the follower's actual
angular numerator has opposite signs at second coordinates `1/2`
and `1/4` in the negative-first-coordinate semicircle.

The exponential estimates below are proved with rational bounds.
No numerical approximation, assumed sign, or stability conclusion
enters the original field. These scalar inequalities are used with
the actual unit-circle identities in the subsequent barrier proof.
-/

import Transformer.Causal.Section4_AnisotropicFlow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

open Real Set

namespace Transformer.Causal

/-- A rational lower bound for the actual self weight at second coordinate
one half. Source: arXiv:2411.04990v2, §4, the self term in `eqn:csa`. -/
theorem anisotropic_exp_one_lower : (13 / 5 : ℝ) ≤ Real.exp 1 := by
  have h := Real.quadratic_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 1 / 2)
  rw [show (1 : ℝ) = 1 / 2 + 1 / 2 by ring, Real.exp_add]
  nlinarith [sq_nonneg (Real.exp (1 / 2) - 13 / 8)]

/-- A rational upper bound for the actual self weight at second coordinate
one quarter. Source: arXiv:2411.04990v2, §4, the self term in `eqn:csa`. -/
theorem anisotropic_exp_quarter_upper : Real.exp (1 / 4 : ℝ) ≤ 4 / 3 := by
  calc
    _ ≤ (2 + (1 / 4 : ℝ)) / (2 - 1 / 4) :=
      Real.exp_le_two_add_div_two_sub (by norm_num) (by norm_num)
    _ ≤ _ := by norm_num

/-- The actual leader weight at the half-coordinate boundary is close
to one. Source: arXiv:2411.04990v2, §4, the predecessor term in `eqn:csa`. -/
theorem anisotropic_exp_twice_small_upper (s : ℝ) (hs : |s| ≤ 1 / 100) :
    Real.exp (2 * s) ≤ 103 / 100 := by
  have hu := (abs_le.mp hs).2
  calc
    _ ≤ Real.exp (1 / 50 : ℝ) := Real.exp_le_exp.mpr (by linarith)
    _ ≤ (2 + (1 / 50 : ℝ)) / (2 - 1 / 50) :=
      Real.exp_le_two_add_div_two_sub (by norm_num) (by norm_num)
    _ ≤ _ := by norm_num

/-- The actual leader weight at the quarter-coordinate boundary has a
positive rational lower bound. Source: arXiv:2411.04990v2, §4,
the predecessor term in `eqn:csa`. -/
theorem anisotropic_exp_small_lower (s : ℝ) (hs : |s| ≤ 1 / 100) :
    (99 / 100 : ℝ) ≤ Real.exp s := by
  have hl := (abs_le.mp hs).1
  linarith [Real.add_one_le_exp s]

/-- The actual cross-coordinate contribution is small near the leader axis.
Source: arXiv:2411.04990v2, §4, the mixed value term in `eqn:csa`. -/
theorem anisotropic_small_product (s c : ℝ) (hs : |s| ≤ 1 / 100) (hc : |c| ≤ 1) :
    |s * c| ≤ 1 / 100 := by
  rw [abs_mul]
  calc
    _ ≤ (1 / 100 : ℝ) * 1 := by gcongr
    _ = _ := by ring

/-- At the actual half-coordinate boundary the follower's angular
numerator is strictly positive. Source: arXiv:2411.04990v2,
§4, `eqn:csa` and Conjecture `thm1.5`.
Here `c₀,s₀,c₁` are the original first-token cosine and sine and
second-token cosine; the second sine is exactly one half. -/
theorem anisotropic_follower_half_numerator_pos (c₀ s₀ c₁ : ℝ)
    (hc₀ : c₀ ≤ 1) (hs₀ : |s₀| ≤ 1 / 100) (hc₁ : c₁ ∈ Icc (-1) (-17 / 20)) :
    0 < -Real.exp 1 * c₁ / 2 + Real.exp (2 * s₀) * (-c₀ + s₀ * c₁) := by
  have hcabs : |c₁| ≤ 1 := abs_le.mpr ⟨hc₁.1, by linarith [hc₁.2]⟩
  have hprod := (abs_le.mp (anisotropic_small_product s₀ c₁ hs₀ hcabs)).1
  have hfirst : (221 / 200 : ℝ) ≤ -Real.exp 1 * c₁ / 2 := by
    have h := mul_le_mul_of_nonneg_left (neg_le_neg hc₁.2)
      (show 0 ≤ Real.exp 1 / 2 by positivity)
    nlinarith [anisotropic_exp_one_lower]
  have hcross : -(101 / 100 : ℝ) ≤ -c₀ + s₀ * c₁ := by linarith
  have hmul := mul_le_mul_of_nonneg_left hcross (Real.exp_nonneg (2 * s₀))
  have hweight := mul_le_mul_of_nonpos_right (anisotropic_exp_twice_small_upper s₀ hs₀)
    (by norm_num : -(101 / 100 : ℝ) ≤ 0)
  nlinarith

/-- At the actual quarter-coordinate boundary the follower's angular
numerator is strictly negative. Source: arXiv:2411.04990v2,
§4, `eqn:csa` and Conjecture `thm1.5`.
The original predecessor and self weights are both retained. -/
theorem anisotropic_follower_quarter_numerator_neg (c₀ s₀ c₁ : ℝ)
    (hc₀ : 99 / 100 ≤ c₀) (hs₀ : |s₀| ≤ 1 / 100) (hc₁ : c₁ ∈ Icc (-1) 0) :
    -Real.exp (1 / 4 : ℝ) * c₁ / 4 + Real.exp s₀ * (-c₀ / 2 + s₀ * c₁) < 0 := by
  have hcabs : |c₁| ≤ 1 := abs_le.mpr ⟨hc₁.1, by linarith [hc₁.2]⟩
  have hprod := (abs_le.mp (anisotropic_small_product s₀ c₁ hs₀ hcabs)).2
  have hfirst : -Real.exp (1 / 4 : ℝ) * c₁ / 4 ≤ 1 / 3 := by
    have h := mul_le_mul_of_nonneg_left (neg_le_neg hc₁.1)
      (show 0 ≤ Real.exp (1 / 4 : ℝ) / 4 by positivity)
    nlinarith [anisotropic_exp_quarter_upper]
  have hcross : -c₀ / 2 + s₀ * c₁ ≤ -(97 / 200 : ℝ) := by linarith
  have hmul := mul_le_mul_of_nonneg_left hcross (Real.exp_nonneg s₀)
  have hweight := mul_le_mul_of_nonpos_right (anisotropic_exp_small_lower s₀ hs₀)
    (by norm_num : -(97 / 200 : ℝ) ≤ 0)
  nlinarith

/-- The original angular follower velocity is strictly positive at
the half-coordinate edge in the negative-cosine semicircle.
Source: arXiv:2411.04990v2, §4, `eqn:csa`, at the stated actual matrices. -/
theorem anisotropicAngleField_pos_at_half (φ : Fin 2 → ℝ)
    (hs₀ : |Real.sin (φ 0)| ≤ 1 / 100) (hs₁ : Real.sin (φ 1) = 1 / 2)
    (hc₁ : Real.cos (φ 1) ≤ 0) : 0 < anisotropicAngleField φ 1 := by
  have hcos : Real.cos (φ 1) ∈ Icc (-1) (-17 / 20) := by
    refine ⟨Real.neg_one_le_cos _, ?_⟩
    have h := (sq_le_sq₀ (by norm_num : (0 : ℝ) ≤ 17 / 20) (neg_nonneg.mpr hc₁)).mp
      (show (17 / 20 : ℝ) ^ 2 ≤ (-Real.cos (φ 1)) ^ 2 by nlinarith [Real.sin_sq_add_cos_sq (φ 1)])
    linarith
  have hn := anisotropic_follower_half_numerator_pos (Real.cos (φ 0)) (Real.sin (φ 0))
    (Real.cos (φ 1)) (Real.cos_le_one _) hs₀ hcos
  change 0 < _ / _
  apply div_pos ?_ (anisotropicAngleDen_pos φ)
  rw [hs₁, show 4 * (1 / 2 : ℝ) ^ 2 = 1 by norm_num,
    show 4 * Real.sin (φ 0) * (1 / 2) = 2 * Real.sin (φ 0) by ring]
  convert hn using 1
  ring

/-- The original angular follower velocity is strictly negative at
the quarter-coordinate edge while the original leader remains near its axis.
Source: arXiv:2411.04990v2, §4, `eqn:csa`, at the stated actual matrices. -/
theorem anisotropicAngleField_neg_at_quarter (φ : Fin 2 → ℝ)
    (hc₀ : 99 / 100 ≤ Real.cos (φ 0)) (hs₀ : |Real.sin (φ 0)| ≤ 1 / 100)
    (hs₁ : Real.sin (φ 1) = 1 / 4) (hc₁ : Real.cos (φ 1) ≤ 0) :
    anisotropicAngleField φ 1 < 0 := by
  have hn := anisotropic_follower_quarter_numerator_neg (Real.cos (φ 0)) (Real.sin (φ 0))
    (Real.cos (φ 1)) hc₀ hs₀ ⟨Real.neg_one_le_cos _, hc₁⟩
  change _ / _ < 0
  apply div_neg_of_neg_of_pos ?_ (anisotropicAngleDen_pos φ)
  rw [hs₁, show 4 * (1 / 4 : ℝ) ^ 2 = 1 / 4 by norm_num,
    show 4 * Real.sin (φ 0) * (1 / 4) = Real.sin (φ 0) by ring]
  convert hn using 1
  ring

/-- The numerical sign hypotheses hold jointly with nonzero follower
cosine, first-coordinate leader one, and second-coordinate leader zero.
Source: arXiv:2411.04990v2, §4, the scalar boundary tests for `eqn:csa`. -/
example : (1 : ℝ) ≤ 1 ∧ (99 / 100 : ℝ) ≤ 1 ∧ |(0 : ℝ)| ≤ 1 / 100 ∧
    (-9 / 10 : ℝ) ∈ Icc (-1) (-17 / 20) ∧ (-9 / 10 : ℝ) ∈ Icc (-1) 0 ∧
    |(-9 / 10 : ℝ)| ≤ 1 := by norm_num

/-- The certified inequalities apply to the actual two retained weights
at these scalar boundary inputs, with a strictly inward sign at each edge.
Source: arXiv:2411.04990v2, §4, `eqn:csa`. -/
example : 0 < -Real.exp 1 * (-9 / 10) / 2 + Real.exp (2 * (0 : ℝ)) * (-1 + 0 * (-9 / 10)) ∧
    -Real.exp (1 / 4 : ℝ) * (-9 / 10) / 4 + Real.exp (0 : ℝ) * (-1 / 2 + 0 * (-9 / 10)) < 0 := by
  exact ⟨anisotropic_follower_half_numerator_pos 1 0 (-9 / 10) (by norm_num) (by norm_num)
    (by norm_num), anisotropic_follower_quarter_numerator_neg 1 0 (-9 / 10)
    (by norm_num) (by norm_num) (by norm_num)⟩

/-- Both actual boundary hypotheses are supplied by genuine unit circle
points with follower sine ranging from one quarter to one half and leader
at angle zero. Source: arXiv:2411.04990v2, §4, the original circle in `eqn:csa`. -/
example : ∀ r ∈ Icc (1 / 4 : ℝ) (1 / 2), ∃ φ : Fin 2 → ℝ,
    |Real.sin (φ 0)| ≤ 1 / 100 ∧ 99 / 100 ≤ Real.cos (φ 0) ∧
    Real.sin (φ 1) = r ∧ Real.cos (φ 1) ≤ 0 := by
  intro r hr
  refine ⟨![0, π - Real.arcsin r], by norm_num, by norm_num, ?_, ?_⟩
  · simpa only [Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.sin_pi_sub]
      using Real.sin_arcsin (by linarith [hr.1]) (by linarith [hr.2])
  · simp only [Matrix.cons_val_one, Matrix.cons_val_fin_one, Real.cos_pi_sub, Real.cos_arcsin]
    exact neg_nonpos.mpr (Real.sqrt_nonneg _)

end Transformer.Causal
