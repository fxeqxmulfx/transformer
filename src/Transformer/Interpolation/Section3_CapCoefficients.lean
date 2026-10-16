/-
# Coefficients for a constant two-neuron cap attractor

Auxiliaries for arXiv:2411.04551v3, Appendix, `lem: two.balls`,
with the original two-neuron perceptron from §1, `eq: vf`.

The first output points toward the outer cap center. The second output
is calibrated so that the combined output at the desired target is radial.
Its radial coefficient can be chosen large enough both to point strictly
into a smaller cap and to dominate the tangential calibration error.

All bounds are proved for the explicit output vector. The final existence
statement chooses a positive boundary strip and a finite positive output
coefficient; no dynamical conclusion is inserted into their definitions.
-/

import Transformer.Interpolation.Section3_NestedCaps

open Real Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The explicit second output balancing the center output at the target.
Auxiliary construction for arXiv:2411.04551v3, Appendix, `lem: two.balls`;
its target equilibrium is `twoRidgeField_target`. -/
noncomputable def capTargetOutput (z w ω : SSphere d) (c h ℓ : ℝ) : EucSpace d :=
  (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)⁻¹ •
    (ℓ • (ω : EucSpace d) - (inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) • (z : EucSpace d))

/-- A finite norm bound for the actual second output.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem norm_capTargetOutput_le (z w ω : SSphere d) (c h ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) (hℓ : 0 ≤ ℓ) :
    ‖capTargetOutput z w ω c h ℓ‖ ≤
      (ℓ + (inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)) /
        (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) := by
  let a := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let b := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  have hn := norm_sub_le (ℓ • (ω : EucSpace d)) (a • (z : EucSpace d))
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg hℓ, abs_of_nonneg hα, mem_sphere_zero_iff_norm.mp ω.2,
    mem_sphere_zero_iff_norm.mp z.2, mul_one, mul_one] at hn
  change ‖b⁻¹ • (ℓ • (ω : EucSpace d) - a • (z : EucSpace d))‖ ≤ (ℓ + a) / b
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hβ)]
  calc
    _ ≤ b⁻¹ * (ℓ + a) := mul_le_mul_of_nonneg_left hn (inv_nonneg.mpr hβ.le)
    _ = _ := by ring

/-- A sufficiently large radial coefficient puts the actual second output
inside the cone pointing into the inner cap's boundary strip.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem capTargetOutput_cone (z w ω : SSphere d) (c h σ ℓ : ℝ)
    (hα : 0 ≤ inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hh : 0 ≤ h) (hσ : 0 ≤ σ)
    (hσβ : σ ≤ (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) / 2)
    (hℓ : 4 * (inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
      (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) ≤ ℓ) :
    (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
      inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d) := by
  let a := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let b := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  have hℓ0 : 0 ≤ ℓ := (div_nonneg (mul_nonneg (by norm_num) hα) hβ.le).trans hℓ
  have hw := abs_real_inner_le_norm (ω : EucSpace d) (w : EucSpace d)
  have hz := abs_real_inner_le_norm (z : EucSpace d) (w : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp ω.2, mem_sphere_zero_iff_norm.mp w.2, mul_one] at hw
  rw [mem_sphere_zero_iff_norm.mp z.2, mem_sphere_zero_iff_norm.mp w.2, mul_one] at hz
  have hw1 := (abs_le.mp hw).2
  have hz1 := (abs_le.mp hz).2
  have hhσ : h + σ ≤ 1 := by linarith
  have hn := mul_le_mul_of_nonneg_left (norm_capTargetOutput_le z w ω c h ℓ hα hβ hℓ0)
    (show 0 ≤ h + σ by linarith)
  have hlarge : 4 * a ≤ ℓ * b := (div_le_iff₀ hβ).mp hℓ
  have hmargin : ℓ * (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h - σ) ≥ 2 * a := by
    have hm := mul_le_mul_of_nonneg_left hσβ hℓ0
    change ℓ * σ ≤ ℓ * (b / 2) at hm
    change _ ≥ 2 * a
    dsimp only [b] at hm hlarge
    nlinarith
  have hright : inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d) =
      (ℓ * inner ℝ (ω : EucSpace d) (w : EucSpace d) - a * inner ℝ (z : EucSpace d) (w : EucSpace d)) / b := by
    simp only [capTargetOutput, real_inner_smul_left, inner_sub_left]
    dsimp only [a, b]
    ring
  rw [hright]
  refine hn.trans ?_
  rw [← mul_div_assoc, div_le_div_iff_of_pos_right hβ]
  have ha1 := mul_le_mul_of_nonneg_left hz1 hα
  have ha2 := mul_le_mul_of_nonneg_left hhσ hα
  dsimp only [a] at hmargin ⊢
  nlinarith

/-- One finite positive coefficient simultaneously supplies the inward
cone bound and dominates twice the tangential error plus a unit margin.
The inner cap contains both the center and target strictly.
Auxiliary for arXiv:2411.04551v3, Appendix, `lem: two.balls`. -/
theorem exists_capTarget_coefficients (z w ω : SSphere d) (c h : ℝ)
    (hα : 0 < inner ℝ (ω : EucSpace d) (z : EucSpace d) - c)
    (hβ : 0 < inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)
    (hz : h < inner ℝ (z : EucSpace d) (w : EucSpace d)) (hh : 0 ≤ h) :
    ∃ σ ℓ : ℝ, 0 < σ ∧ 0 < ℓ ∧
      σ < inner ℝ (z : EucSpace d) (w : EucSpace d) - h ∧
      σ ≤ inner ℝ (ω : EucSpace d) (w : EucSpace d) - h ∧
      (h + σ) * ‖capTargetOutput z w ω c h ℓ‖ ≤
        inner ℝ (capTargetOutput z w ω c h ℓ) (w : EucSpace d) ∧
      2 * ‖(z : EucSpace d) -
        ((inner ℝ (ω : EucSpace d) (z : EucSpace d) - c) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h)) • (w : EucSpace d)‖ + 1 ≤
        ℓ * σ * ((h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2) /
          (inner ℝ (ω : EucSpace d) (w : EucSpace d) - h) := by
  let a := inner ℝ (ω : EucSpace d) (z : EucSpace d) - c
  let b := inner ℝ (ω : EucSpace d) (w : EucSpace d) - h
  let κ := (h + inner ℝ (ω : EucSpace d) (w : EucSpace d)) ^ 2 / 2
  let A := ‖(z : EucSpace d) - (a / b) • (w : EucSpace d)‖
  let σ := min ((inner ℝ (z : EucSpace d) (w : EucSpace d) - h) / 2) (b / 2)
  have hσ : 0 < σ := lt_min (by linarith) (by positivity)
  have hσz : σ ≤ (inner ℝ (z : EucSpace d) (w : EucSpace d) - h) / 2 := min_le_left _ _
  have hσb : σ ≤ b / 2 := min_le_right _ _
  have hκ : 0 < κ := by
    have hp : 0 < h + inner ℝ (ω : EucSpace d) (w : EucSpace d) := by linarith
    exact div_pos (sq_pos_of_pos hp) (by norm_num)
  let ℓ := 4 * a / b + b * (2 * A + 1) / (σ * κ) + 1
  have hfirst : 0 ≤ 4 * a / b := by positivity
  have hsecond : 0 < b * (2 * A + 1) / (σ * κ) := by
    have hA : 0 ≤ A := norm_nonneg _
    positivity
  have hℓ : 0 < ℓ := by dsimp only [ℓ]; linarith
  have hℓa : 4 * a / b ≤ ℓ := by dsimp only [ℓ]; linarith
  have hℓb : b * (2 * A + 1) / (σ * κ) ≤ ℓ := by dsimp only [ℓ]; linarith
  refine ⟨σ, ℓ, hσ, hℓ, by linarith, by linarith, ?_, ?_⟩
  · exact capTargetOutput_cone z w ω c h σ ℓ hα.le hβ hh hσ.le hσb hℓa
  · change 2 * A + 1 ≤ ℓ * σ * κ / b
    have he := (div_le_iff₀ (mul_pos hσ hκ)).mp hℓb
    apply (le_div_iff₀ hβ).mpr
    nlinarith

/-- Actual unit axes, nonzero activation margins, a positive strip and a
finite positive output jointly satisfy every coefficient hypothesis. -/
example : ∃ (z w ω : SSphere 2) (c h σ ℓ : ℝ),
    0 < inner ℝ (ω : EucSpace 2) (z : EucSpace 2) - c ∧
    0 < inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h ∧
    h < inner ℝ (z : EucSpace 2) (w : EucSpace 2) ∧
    0 ≤ h ∧ 0 ≤ σ ∧
    σ ≤ (inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h) / 2 ∧
    4 * (inner ℝ (ω : EucSpace 2) (z : EucSpace 2) - c) /
      (inner ℝ (ω : EucSpace 2) (w : EucSpace 2) - h) ≤ ℓ ∧ 0 ≤ ℓ := by
  refine ⟨basePoint 1, basePoint 1, basePoint 1, 0, 1 / 2, 1 / 8, 100, ?_⟩
  norm_num [basePoint]

end Transformer.Interpolation
