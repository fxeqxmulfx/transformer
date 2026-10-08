import Transformer.Modes.Section5_PhaseCurve

/-!
# Coefficient bounds for the Gaussian phase

The determinant argument in arXiv:2412.09080v3, §5.5, gives pointwise
nondegeneracy of the Gaussian phase. A quantitative bound additionally
needs control of the entries of the derivative matrix on a bounded
interval. This module supplies that control with explicit constants.

For a direction normalized in the max norm, Cramer's identities bound
`|A C - B²|` by `(|A| + |B| + |C|) * max (|φ'|, |φ''|)`.
The Gaussian exponential factor is kept separate from the polynomial
entries `1 - βu²`, `βu(βu² - 3)`, and `β(-β²u⁴ + 6βu² - 3)`.
Their absolute values sum to at most `gaussianPhaseCoefficientBound β R`
whenever `β ≥ 0` and `|u| ≤ R`.

This coefficient bound is positive, increases with the radius, and grows
at most as `(1 + 7β + 7β² + β³) * (1 + R)⁴`. The exponent four comes from
the degree of `g'''` after its Gaussian factor is removed. Tracking this
polynomial growth is needed when the local estimates are later summed
against the Gaussian amplitude rather than truncating a fixed tail.

The direction uses the max norm of `ℝ × ℝ` in the formal Fourier statement.
The source uses Euclidean unit directions; rescaling the direction and
frequency together preserves the phase in the oscillatory exponential.

Each curve derivative contains one Gaussian factor, while the determinant
contains its square. Keeping that factor separate lets the cofactor bound
cancel one copy, which gives the required dependence on the argument radius.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- Cramer's identities turn a nonzero determinant into a quantitative
bound for a direction of max norm one.
Source: arXiv:2412.09080v3, §5.5, the determinant argument. -/
theorem determinant_le_phase_bound {A B C u v : ℝ} (huv : max |u| |v| = 1) :
    |A * C - B ^ 2| ≤ (|A| + |B| + |C|) * max |u * A + v * B| |u * B + v * C| := by
  let M := max |u * A + v * B| |u * B + v * C|
  have hM : 0 ≤ M := (abs_nonneg _).trans (le_max_left _ _)
  have h1 : |u| * |A * C - B ^ 2| ≤ (|C| + |B|) * M := by
    rw [← abs_mul, show u * (A * C - B ^ 2) =
      C * (u * A + v * B) - B * (u * B + v * C) by ring]
    calc _ ≤ |C * (u * A + v * B)| + |B * (u * B + v * C)| := abs_sub _ _
      _ = |C| * |u * A + v * B| + |B| * |u * B + v * C| := by rw [abs_mul, abs_mul]
      _ ≤ |C| * M + |B| * M := by
        exact add_le_add (mul_le_mul_of_nonneg_left (le_max_left _ _) (abs_nonneg _))
          (mul_le_mul_of_nonneg_left (le_max_right _ _) (abs_nonneg _))
      _ = _ := by ring
  have h2 : |v| * |A * C - B ^ 2| ≤ (|A| + |B|) * M := by
    rw [← abs_mul, show v * (A * C - B ^ 2) =
      A * (u * B + v * C) - B * (u * A + v * B) by ring]
    calc _ ≤ |A * (u * B + v * C)| + |B * (u * A + v * B)| := abs_sub _ _
      _ = |A| * |u * B + v * C| + |B| * |u * A + v * B| := by rw [abs_mul, abs_mul]
      _ ≤ |A| * M + |B| * M := by
        exact add_le_add (mul_le_mul_of_nonneg_left (le_max_right _ _) (abs_nonneg _))
          (mul_le_mul_of_nonneg_left (le_max_left _ _) (abs_nonneg _))
      _ = _ := by ring
  rcases le_total |u| |v| with h | h
  · rw [max_eq_right h] at huv
    rw [huv, one_mul] at h2
    exact h2.trans (by nlinarith [abs_nonneg C])
  · rw [max_eq_left h] at huv
    rw [huv, one_mul] at h1
    exact h1.trans (by nlinarith [abs_nonneg A])


example : max |(1 : ℝ)| |0| = 1 := by norm_num

/-- A bound for the sum of the three Gaussian polynomial coefficients
on `|u| ≤ R`. Source: arXiv:2412.09080v3, §5.5, the derivative display. -/
def gaussianPhaseCoefficientBound (β R : ℝ) : ℝ :=
  1 + β * R ^ 2 + β * R * (β * R ^ 2 + 3) + β * (β ^ 2 * R ^ 4 + 6 * β * R ^ 2 + 3)

/-- The coefficient bound is strictly positive even at radius zero.
Source: arXiv:2412.09080v3, §5.5, bounded derivative coefficients. -/
theorem gaussianPhaseCoefficientBound_pos {β R : ℝ} (hβ : 0 ≤ β) (hR : 0 ≤ R) :
    0 < gaussianPhaseCoefficientBound β R := by
  unfold gaussianPhaseCoefficientBound
  positivity

example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := ⟨zero_le_one, zero_le_one⟩

/-- The actual polynomial entries of `(g', g'', g''')` obey the stated
coefficient bound. Source: arXiv:2412.09080v3, §5.5, the derivative display. -/
theorem gaussian_coefficients_bound {β R u : ℝ} (hβ : 0 ≤ β) (hR : 0 ≤ R)
    (hu : |u| ≤ R) :
    |1 - β * u ^ 2| + |β * u * (β * u ^ 2 - 3)| +
      |β * (-(β ^ 2) * u ^ 4 + 6 * β * u ^ 2 - 3)| ≤ gaussianPhaseCoefficientBound β R := by
  have hu2 : u ^ 2 ≤ R ^ 2 := by
    simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg u) hu 2
  have hu4 : u ^ 4 ≤ R ^ 4 := by
    calc _ = (u ^ 2) ^ 2 := by ring
      _ ≤ (R ^ 2) ^ 2 := pow_le_pow_left₀ (sq_nonneg u) hu2 2
      _ = _ := by ring
  have hβu2 := mul_le_mul_of_nonneg_left hu2 hβ
  have hA : |1 - β * u ^ 2| ≤ 1 + β * R ^ 2 := by
    apply abs_le.mpr
    constructor <;> nlinarith [mul_nonneg hβ (sq_nonneg u), mul_nonneg hβ (sq_nonneg R)]
  have hBI : |β * u ^ 2 - 3| ≤ β * R ^ 2 + 3 := by
    apply abs_le.mpr
    constructor <;> nlinarith [mul_nonneg hβ (sq_nonneg u), mul_nonneg hβ (sq_nonneg R)]
  have hB : |β * u * (β * u ^ 2 - 3)| ≤ β * R * (β * R ^ 2 + 3) := by
    rw [abs_mul, abs_mul, abs_of_nonneg hβ]
    exact mul_le_mul (mul_le_mul_of_nonneg_left hu hβ) hBI (abs_nonneg _) (mul_nonneg hβ hR)
  have hβu4 := mul_le_mul_of_nonneg_left hu4 (sq_nonneg β)
  have h6u2 := mul_le_mul_of_nonneg_left hu2 (by positivity : 0 ≤ 6 * β)
  have hCI : |-(β ^ 2) * u ^ 4 + 6 * β * u ^ 2 - 3| ≤ β ^ 2 * R ^ 4 + 6 * β * R ^ 2 + 3 := by
    apply abs_le.mpr
    constructor <;> nlinarith [mul_nonneg (sq_nonneg β) (by positivity : 0 ≤ u ^ 4),
      mul_nonneg (by positivity : 0 ≤ 6 * β) (sq_nonneg u),
      mul_nonneg (sq_nonneg β) (by positivity : 0 ≤ R ^ 4),
      mul_nonneg (by positivity : 0 ≤ 6 * β) (sq_nonneg R)]
  have hC : |β * (-(β ^ 2) * u ^ 4 + 6 * β * u ^ 2 - 3)| ≤
      β * (β ^ 2 * R ^ 4 + 6 * β * R ^ 2 + 3) := by
    rw [abs_mul, abs_of_nonneg hβ]
    exact mul_le_mul_of_nonneg_left hCI hβ
  exact add_le_add (add_le_add hA hB) hC

example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ |(0 : ℝ)| ≤ 1 := by norm_num

/-- The explicit coefficient bound has at most quartic growth in `R`.
Source: arXiv:2412.09080v3, §5.5, bounded derivative coefficients. -/
theorem gaussianPhaseCoefficientBound_le {β R : ℝ} (hβ : 0 ≤ β) (hR : 0 ≤ R) :
    gaussianPhaseCoefficientBound β R ≤ (1 + 7 * β + 7 * β ^ 2 + β ^ 3) * (1 + R) ^ 4 := by
  rw [← sub_nonneg]
  unfold gaussianPhaseCoefficientBound
  ring_nf
  positivity


example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 := ⟨zero_le_one, zero_le_one⟩

/-- Enlarging the argument radius increases the coefficient bound.
Source: arXiv:2412.09080v3, §5.5, the compact interval `[-2R, 2R]`. -/
theorem gaussianPhaseCoefficientBound_mono_radius {β r R : ℝ}
    (hβ : 0 ≤ β) (hr : 0 ≤ r) (hrR : r ≤ R) :
    gaussianPhaseCoefficientBound β r ≤ gaussianPhaseCoefficientBound β R := by
  have hR : 0 ≤ R := hr.trans hrR
  unfold gaussianPhaseCoefficientBound
  gcongr

example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 2 := by norm_num

end Transformer.Modes
