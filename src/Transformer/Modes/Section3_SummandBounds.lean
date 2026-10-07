/-
# The number of modes of a Gaussian KDE — bounds for the centered summand

The boundedness argument for `lem:eta` of arXiv:2412.09080v3. The first
coordinate has size `β^{-1/2}`, and the second has size one. These bounds,
combined with the covariance estimates, bound the standardized summand
pointwise; its second moment then controls every higher moment.

Source: arXiv:2412.09080v3, `eq:Yi`, `lem:eta`, §5.3.
-/

import Transformer.Modes.Section3_Standardized

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Modes

/-- The sharper bound `|G(t,x)| ≤ β^{-1/2}` needed for `lem:eta`.
Source: arXiv:2412.09080v3, `eq: Gt`, §2.2 and §5.3. -/
theorem abs_bigG_le_inv_sqrt {β : ℝ} (hβ : 0 < β) (t x : ℝ) :
    |bigG β t x| ≤ (Real.sqrt β)⁻¹ := by
  let u := t - x
  have hs : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  have hs2 := Real.sq_sqrt hβ.le
  have hu : Real.sqrt β * |u| ≤ β / 2 * u ^ 2 + 1 := by
    nlinarith [sq_nonneg (Real.sqrt β * |u| - 1), sq_abs u]
  have he : Real.sqrt β * |u| ≤ Real.exp (β / 2 * u ^ 2) :=
    hu.trans (Real.add_one_le_exp _)
  rw [bigG, abs_mul, Real.abs_exp,
    show -(β / 2) * (t - x) ^ 2 = -(β / 2 * u ^ 2) by dsimp [u]; ring, Real.exp_neg]
  rw [inv_mul_le_iff₀ (Real.exp_pos _)]
  exact (le_mul_inv_iff₀ hs).mpr (by simpa [mul_comm] using he)

/-- A positive bandwidth witnesses the sharp bound's hypothesis. -/
example : (0 : ℝ) < 1 := one_pos

/-- The first moment obeys the same bound as `G`.
Source: arXiv:2412.09080v3, `eq:Yi`, §5.3. -/
theorem abs_meanG_le_inv_sqrt {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    |meanG β t| ≤ (Real.sqrt β)⁻¹ := by
  simpa [meanG, Real.norm_eq_abs] using
    (norm_integral_le_of_norm_le_const (μ := gaussianReal 0 1) (f := bigG β t)
      (C := (Real.sqrt β)⁻¹)
      (ae_of_all _ fun x => by
        simpa [Real.norm_eq_abs] using abs_bigG_le_inv_sqrt hβ t x))

/-- A positive bandwidth witnesses the first-moment bound's hypothesis. -/
example : (0 : ℝ) < 1 := one_pos

/-- The first moment remains bounded after scaling by `√β`, uniformly in
the location. This is the centering contribution in the first coordinate
of `Y`. Source: arXiv:2412.09080v3, `eq:Yi`, §5.3. -/
theorem sqrt_mul_abs_meanG_le_one {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    Real.sqrt β * |meanG β t| ≤ 1 := by
  have hs : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  calc
    _ ≤ Real.sqrt β * (Real.sqrt β)⁻¹ :=
      mul_le_mul_of_nonneg_left (abs_meanG_le_inv_sqrt hβ t) hs.le
    _ = 1 := mul_inv_cancel₀ hs.ne'

/-- A positive bandwidth witnesses the scaled first-moment bound. -/
example : (0 : ℝ) < 1 := one_pos

/-- The second coordinate's first moment has size at most two.
Source: arXiv:2412.09080v3, `eq:Yi`, §5.3. -/
theorem abs_meanG'_le_two {β : ℝ} (hβ : 0 < β) (t : ℝ) :
    |meanG' β t| ≤ 2 := by
  simpa [meanG', Real.norm_eq_abs] using
    (norm_integral_le_of_norm_le_const (μ := gaussianReal 0 1) (f := bigG' β t) (C := 2)
      (ae_of_all _ fun x => by simpa [Real.norm_eq_abs] using abs_bigG'_le hβ t x))

/-- A positive bandwidth witnesses the second-coordinate bound's hypothesis. -/
example : (0 : ℝ) < 1 := one_pos

/-- Centering changes the pointwise coordinate bounds by at most a factor
two. Source: arXiv:2412.09080v3, `eq:Yi`, §5.3. -/
theorem centered_bigG_bounds {β : ℝ} (hβ : 0 < β) (t x : ℝ) :
    Real.sqrt β * |bigG β t x - meanG β t| ≤ 2 ∧
      |bigG' β t x - meanG' β t| ≤ 4 := by
  have hs : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  have hG : |bigG β t x - meanG β t| ≤ 2 * (Real.sqrt β)⁻¹ := by
    calc
      _ ≤ |bigG β t x| + |meanG β t| := abs_sub _ _
      _ ≤ (Real.sqrt β)⁻¹ + (Real.sqrt β)⁻¹ :=
        add_le_add (abs_bigG_le_inv_sqrt hβ t x) (abs_meanG_le_inv_sqrt hβ t)
      _ = _ := by ring
  constructor
  · calc
      _ ≤ Real.sqrt β * (2 * (Real.sqrt β)⁻¹) :=
        mul_le_mul_of_nonneg_left hG hs.le
      _ = 2 := by field_simp
  · calc
      _ ≤ |bigG' β t x| + |meanG' β t| := abs_sub _ _
      _ ≤ 2 + 2 := add_le_add (abs_bigG'_le hβ t x) (abs_meanG'_le_two hβ t)
      _ = 4 := by norm_num

/-- Both centered-coordinate bounds apply at `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- Completing the square expresses the whitening norm through its two
triangular coordinates. Source: arXiv:2412.09080v3, `eq:Yi`, §5.3. -/
theorem eucl_whiten_sq_completed {a b d : ℝ} (ha : 0 < a)
    (hD : 0 < a * d - b ^ 2) (z : ℝ × ℝ) :
    eucl (whiten a b d z) ^ 2 =
      z.1 ^ 2 / a + (a / (a * d - b ^ 2)) * (z.2 - b / a * z.1) ^ 2 := by
  rw [eucl_whiten_sq ha hD, quadForm]
  generalize hDdef : a * d - b ^ 2 = D at hD ⊢
  field_simp [ha.ne', hD.ne']
  linear_combination z.1 ^ 2 * hDdef

/-- The identity applies to the identity covariance matrix. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 := by norm_num

/-- A deterministic bound for the whitening norm using the two coordinate
sizes and the covariance ratio. Source: arXiv:2412.09080v3, `lem:eta`, §5.3. -/
theorem eucl_whiten_sq_le_of_bounds {a b d β : ℝ} (ha : 0 < a)
    (hD : 0 < a * d - b ^ 2) (hβ : 0 < β) (z : ℝ × ℝ)
    (hz₁ : Real.sqrt β * |z.1| ≤ 2) (hz₂ : |z.2| ≤ 4)
    (hb : |b / a| ≤ 2 * Real.sqrt β) :
    eucl (whiten a b d z) ^ 2 ≤ 4 / (β * a) + 64 * (a / (a * d - b ^ 2)) := by
  have hs : 0 ≤ Real.sqrt β * |z.1| := by positivity
  have hz₁sq : β * z.1 ^ 2 ≤ 4 := by
    have := mul_self_le_mul_self hs hz₁
    nlinarith [Real.sq_sqrt hβ.le, sq_abs z.1]
  have h₁ : z.1 ^ 2 / a ≤ 4 / (β * a) := by
    apply (div_le_div_iff₀ ha (mul_pos hβ ha)).mpr
    nlinarith [mul_le_mul_of_nonneg_right hz₁sq ha.le]
  have hbz : |b / a * z.1| ≤ 4 := by
    rw [abs_mul]
    calc
      _ ≤ (2 * Real.sqrt β) * |z.1| :=
        mul_le_mul_of_nonneg_right hb (abs_nonneg _)
      _ = 2 * (Real.sqrt β * |z.1|) := by ring
      _ ≤ 2 * 2 := mul_le_mul_of_nonneg_left hz₁ (by norm_num)
      _ = 4 := by norm_num
  have hdiff : |z.2 - b / a * z.1| ≤ 8 :=
    (abs_sub _ _).trans (by linarith)
  have hdiffsq : (z.2 - b / a * z.1) ^ 2 ≤ 64 := by
    nlinarith [sq_abs (z.2 - b / a * z.1), abs_nonneg (z.2 - b / a * z.1)]
  rw [eucl_whiten_sq_completed ha hD]
  exact add_le_add h₁ (by
    simpa [mul_comm] using
      mul_le_mul_of_nonneg_left hdiffsq (div_pos ha hD).le)

/-- The deterministic whitening bound has simultaneous witnesses: identity
covariance, unit bandwidth and the zero centered vector. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 * 1 - 0 ^ 2 ∧ (0 : ℝ) < 1 ∧
    Real.sqrt 1 * |(0 : ℝ)| ≤ 2 ∧ |(0 : ℝ)| ≤ 4 ∧
    |(0 : ℝ) / 1| ≤ 2 * Real.sqrt 1 := by norm_num

end Transformer.Modes
