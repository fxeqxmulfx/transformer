import Transformer.Modes.Section5_CoefficientBounds

/-!
# Quantitative nondegeneracy of the Gaussian phase

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, uses the
nonvanishing curve determinant to control stationary and nonstationary
intervals. Here the determinant and coefficient estimates give an
explicit positive floor for the larger of the first two phase derivatives.

If `β > 0`, `|t - x| ≤ R`, and the direction has max norm one, then
`max (|φ'|, |φ''|) ≥ 3β exp (-βR²/2) / gaussianPhaseCoefficientBound β R`.
The determinant carries the square of the Gaussian factor; Cramer's
identities cancel one copy against the coefficient matrix. The resulting
floor retains one Gaussian factor and a denominator of quartic growth.

The last theorem applies the local phase estimate to the actual Gaussian
phase on an interval of length at most one where its second derivative
has one sign. All derivative identities and continuity conditions are
proved from `Section5_PhaseCurve`. A later partition at the zeros of the
second derivative is needed for an interval where that sign changes.
The floor applies to every positive `β`, including `β ≥ 2`.
The global `β < 2` restriction concerns the Gaussian sum over intervals.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- An explicit derivative floor on a bounded argument interval.
Source: arXiv:2412.09080v3, §5.5, the constants `c_*` and `c_W`. -/
noncomputable def gaussianPhaseFloor (β R : ℝ) : ℝ :=
  3 * β * Real.exp (-(β / 2) * R ^ 2) / gaussianPhaseCoefficientBound β R

/-- The explicit floor is strictly positive for `β > 0`.
Source: arXiv:2412.09080v3, §5.5, the nonvanishing determinant. -/
theorem gaussianPhaseFloor_pos {β R : ℝ} (hβ : 0 < β) (hR : 0 ≤ R) :
    0 < gaussianPhaseFloor β R := by
  unfold gaussianPhaseFloor
  exact div_pos (mul_pos (by positivity) (Real.exp_pos _))
    (gaussianPhaseCoefficientBound_pos hβ.le hR)

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨one_pos, zero_le_one⟩

/-- A normalized phase direction has its first or second derivative
at least the explicit floor on `|t - x| ≤ R`.
Source: arXiv:2412.09080v3, §5.5, the determinant argument. -/
theorem gaussian_phase_floor {β R : ℝ} (hβ : 0 < β) (hR : 0 ≤ R) {θ : ℝ × ℝ}
    (hθ : max |θ.1| |θ.2| = 1) (t x : ℝ) (hu : |t - x| ≤ R) :
    gaussianPhaseFloor β R ≤
      max |θ.1 * bigG' β t x + θ.2 * bigG'' β t x|
        |θ.1 * bigG'' β t x + θ.2 * bigG''' β t x| := by
  let u := t - x
  let A := 1 - β * u ^ 2
  let B := β * u * (β * u ^ 2 - 3)
  let C := β * (-(β ^ 2) * u ^ 4 + 6 * β * u ^ 2 - 3)
  let E := Real.exp (-(β / 2) * u ^ 2)
  let M := max |θ.1 * A + θ.2 * B| |θ.1 * B + θ.2 * C|
  have hM : 0 ≤ M := (abs_nonneg _).trans (le_max_left _ _)
  have hepos : 0 < E := Real.exp_pos _
  have hcoeff : |A| + |B| + |C| ≤ gaussianPhaseCoefficientBound β R :=
    gaussian_coefficients_bound hβ.le hR hu
  have hdet := determinant_le_phase_bound (A := A) (B := B) (C := C) hθ
  have hd : A * C - B ^ 2 = -β * (β ^ 2 * u ^ 4 + 3) := by dsimp [A, B, C]; ring
  have hdetmin : 3 * β ≤ |A * C - B ^ 2| := by
    rw [hd, abs_mul, abs_neg, abs_of_nonneg hβ.le,
      abs_of_nonneg (by positivity : 0 ≤ β ^ 2 * u ^ 4 + 3)]
    nlinarith [mul_nonneg hβ.le (mul_nonneg (sq_nonneg β) (by positivity : 0 ≤ u ^ 4))]
  have hdiv : 3 * β / gaussianPhaseCoefficientBound β R ≤ M := by
    apply (div_le_iff₀ (gaussianPhaseCoefficientBound_pos hβ.le hR)).mpr
    calc _ ≤ |A * C - B ^ 2| := hdetmin
      _ ≤ (|A| + |B| + |C|) * M := hdet
      _ ≤ gaussianPhaseCoefficientBound β R * M :=
        mul_le_mul_of_nonneg_right hcoeff hM
      _ = _ := mul_comm _ _
  have hu2 : u ^ 2 ≤ R ^ 2 := by
    simpa only [sq_abs] using pow_le_pow_left₀ (abs_nonneg u) hu 2
  have he : Real.exp (-(β / 2) * R ^ 2) ≤ E := by
    apply Real.exp_le_exp.mpr
    nlinarith [mul_le_mul_of_nonneg_left hu2 hβ.le]
  have hp : θ.1 * bigG' β t x + θ.2 * bigG'' β t x = E * (θ.1 * A + θ.2 * B) := by
    dsimp [bigG', bigG'', E, A, B, u]
    ring
  have hq : θ.1 * bigG'' β t x + θ.2 * bigG''' β t x = E * (θ.1 * B + θ.2 * C) := by
    dsimp [bigG'', bigG''', E, B, C, u]
    ring
  rw [hp, hq]
  simp only [abs_mul, abs_of_nonneg hepos.le]
  rw [← mul_max_of_nonneg _ _ hepos.le]
  calc gaussianPhaseFloor β R =
        (3 * β / gaussianPhaseCoefficientBound β R) * Real.exp (-(β / 2) * R ^ 2) := by
          unfold gaussianPhaseFloor
          ring
    _ ≤ M * E := mul_le_mul hdiv he (Real.exp_pos _).le hM
    _ = _ := mul_comm _ _


example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧ max |(0 : ℝ)| |1| = 1 ∧ |(0 : ℝ) - 0| ≤ 1 := by
  norm_num

/-- The derivative floor loses only a quartic polynomial in addition
to its Gaussian factor. Source: arXiv:2412.09080v3, §5.5, derivative bounds. -/
theorem gaussianPhaseFloor_lower_bound {β R : ℝ} (hβ : 0 < β) (hR : 0 ≤ R) :
    3 * β * Real.exp (-(β / 2) * R ^ 2) /
      ((1 + 7 * β + 7 * β ^ 2 + β ^ 3) * (1 + R) ^ 4) ≤ gaussianPhaseFloor β R := by
  unfold gaussianPhaseFloor
  exact div_le_div_of_nonneg_left (by positivity)
    (gaussianPhaseCoefficientBound_pos hβ.le hR) (gaussianPhaseCoefficientBound_le hβ.le hR)

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨one_pos, zero_le_one⟩

/-- The actual Gaussian phase satisfies the local oscillatory estimate
on any short interval where its second derivative has one sign.
Source: arXiv:2412.09080v3, §5.5, the stationary-phase estimate. -/
theorem gaussian_phase_interval_bound {a b β ρ R : ℝ} {θ : ℝ × ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hβ : 0 < β) (hR : 0 ≤ R)
    (hθ : max |θ.1| |θ.2| = 1)
    (harg : ∀ u ∈ Set.uIcc a b, |u| ≤ R)
    (hq : (∀ u ∈ Set.uIcc a b, 0 ≤ θ.1 * bigG'' β u 0 + θ.2 * bigG''' β u 0) ∨
      (∀ u ∈ Set.uIcc a b, θ.1 * bigG'' β u 0 + θ.2 * bigG''' β u 0 ≤ 0)) :
    ‖∫ u in a..b, oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u‖ ≤
      10 / Real.sqrt (|ρ| * gaussianPhaseFloor β R) := by
  apply unit_interval_phase_test hab hlen hρ (gaussianPhaseFloor_pos hβ hR)
    (fun u _ => hasDerivAt_gaussianPhase β 0 u θ)
    (fun u _ => hasDerivAt_gaussianPhase' β 0 u θ) (by unfold bigG'' bigG'''; fun_prop) hq
  intro u hu
  exact gaussian_phase_floor hβ hR hθ u 0 (by simpa only [sub_zero] using harg u hu)

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) - 0 ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ max |(1 : ℝ)| |0| = 1 ∧
    (∀ u ∈ Set.uIcc (0 : ℝ) 1, |u| ≤ 1) ∧
    ((∀ u ∈ Set.uIcc (0 : ℝ) 1, 0 ≤ 1 * bigG'' 1 u 0 + 0 * bigG''' 1 u 0) ∨
      (∀ u ∈ Set.uIcc (0 : ℝ) 1, 1 * bigG'' 1 u 0 + 0 * bigG''' 1 u 0 ≤ 0)) := by
  refine ⟨zero_le_one, by norm_num, one_ne_zero, one_pos, zero_le_one, by norm_num,
    ?_, Or.inr ?_⟩
  · intro u hu
    rw [Set.uIcc_of_le zero_le_one] at hu
    rw [abs_of_nonneg hu.1]
    exact hu.2
  · intro u hu
    rw [Set.uIcc_of_le zero_le_one] at hu
    have hu2 : u ^ 2 ≤ 1 := by simpa using pow_le_pow_left₀ hu.1 hu.2 2
    have hpoly : u * (u ^ 2 - 3) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hu.1 (by linarith)
    have h := mul_nonpos_of_nonneg_of_nonpos (Real.exp_pos (-(1 / 2 : ℝ) * u ^ 2)).le hpoly
    simpa only [one_mul, zero_mul, add_zero, bigG'', sub_zero] using h

end Transformer.Modes
