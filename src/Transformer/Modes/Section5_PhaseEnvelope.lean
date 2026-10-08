import Transformer.Modes.Section5_GaussianAmplitude

/-!
# A Gaussian envelope for the local Fourier estimates

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, treats a
fixed-radius tail as negligible at large frequency. That tail bound does
not tend to zero with frequency. Here every unit interval is retained,
and the shifted Gaussian amplitude is combined with the explicit phase
nondegeneracy floor.

The inverse square root of that floor is bounded by a quadratic factor
in the argument radius times `exp (βR²/4)`, with a constant depending on
`β`. On `[m, m + 1]`, use `R = |m| + 1`. The amplitude and its variation
contribute `exp (-m²/2 + (1 + |t|) |m|)` and one linear factor.

Completing the square bounds their combined exponential by a constant
depending on `β,t` times `exp (-(2 - β)m²/8)`. The remaining polynomial
has degree three. The final theorem gives this explicit envelope for
every real interval index, every nonzero frequency, and every direction
of max norm one, with a single constant independent of those variables.
The normalization uses the product norm on `ℝ × ℝ`, matching the norm
in the formal Fourier statement. The determinant bound applies to every
direction normalized in this way.

The corrected hypothesis `β < 2` is needed here, when the two Gaussian
rates are combined. The source prints the estimate for every positive
`β` and a constant independent of `t`; those statements have separate
counterexamples in `Section5_PtBddFourier` and `Section5_PtBddDecayFalse`.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The floor loses a quadratic radius factor and half its Gaussian
exponent when inverted under a square root.
Source: arXiv:2412.09080v3, §5.5, quantitative nondegeneracy. -/
theorem gaussianPhaseFloor_inv_sqrt_bound {β R : ℝ} (hβ : 0 < β) (hR : 0 ≤ R) :
    1 / Real.sqrt (gaussianPhaseFloor β R) ≤
      (Real.sqrt (1 + 7 * β + 7 * β ^ 2 + β ^ 3) / Real.sqrt (3 * β)) *
        (1 + R) ^ 2 * Real.exp (β * R ^ 2 / 4) := by
  let C := 1 + 7 * β + 7 * β ^ 2 + β ^ 3
  have hC : 0 < C := by dsimp [C]; positivity
  have hβ3 : 0 < 3 * β := by positivity
  have hL : 0 < 3 * β * Real.exp (-(β / 2) * R ^ 2) / (C * (1 + R) ^ 4) := by positivity
  have h := one_div_le_one_div_of_le (Real.sqrt_pos.mpr hL)
    (Real.sqrt_le_sqrt (gaussianPhaseFloor_lower_bound hβ hR))
  apply h.trans_eq
  rw [Real.sqrt_div (by positivity), Real.sqrt_mul hβ3.le,
    Real.sqrt_mul hC.le, ← Real.exp_half]
  rw [show (1 + R) ^ 4 = ((1 + R) ^ 2) ^ 2 by ring, Real.sqrt_sq (sq_nonneg _)]
  rw [show (-(β / 2) * R ^ 2) / 2 = -(β * R ^ 2 / 4) by ring, Real.exp_neg]
  field_simp
  dsimp [C]
  congr 1
  ring

example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := ⟨one_pos, zero_le_one⟩

/-- Completing the square retains a Gaussian with positive decay rate
when `β < 2`. Source: arXiv:2412.09080v3, §5.5, the corrected tail estimate. -/
theorem gaussian_phase_exponent_bound {β : ℝ} (hβ2 : β < 2) (m t : ℝ) :
    -(m ^ 2) / 2 + (1 + |t|) * |m| + β * (|m| + 1) ^ 2 / 4 ≤
      β / 4 + 2 * (1 + |t| + β / 2) ^ 2 / (2 - β) - ((2 - β) / 8) * m ^ 2 := by
  let d := 2 - β
  let K := 1 + |t| + β / 2
  have hd : 0 < d := by dsimp [d]; linarith
  have hs : 0 ≤ d ^ 2 * m ^ 2 - 8 * d * K * |m| + 16 * K ^ 2 := by
    convert sq_nonneg (d * |m| - 4 * K) using 1
    ring_nf
    rw [sq_abs]
  have hy : K * |m| - d * m ^ 2 / 8 ≤ 2 * K ^ 2 / d := by
    apply (le_div_iff₀ hd).mpr
    nlinarith [hs]
  have hm : |m| ^ 2 = m ^ 2 := sq_abs _
  dsimp [d, K] at hy
  nlinarith [hy, hm]

example : (1 : ℝ) < 2 := one_lt_two

/-- The quadratic radius loss and linear amplitude variation combine to
one cubic envelope. Source: arXiv:2412.09080v3, §5.5, the weighted phase estimate. -/
theorem gaussian_phase_polynomial_factor_bound (m t : ℝ) :
    (2 + |m|) ^ 2 * (2 + |m| + |t|) ≤ 4 * (2 + |t|) * (1 + |m|) ^ 3 := by
  have h : 0 ≤ 4 * (2 + |t|) * (1 + |m|) ^ 3 -
      (2 + |m|) ^ 2 * (2 + |m| + |t|) := by
    ring_nf
    positivity
  linarith

/-- Each local integral has a summable Gaussian envelope, uniformly in
frequency and direction, for the corrected range `0 < β < 2`.
The constant retains the dependence on the fixed translation `t`.
Source: arXiv:2412.09080v3, §5.5, `eq:uniform-decay`. -/
theorem gaussian_phase_local_envelope {β : ℝ} (hβ : 0 < β) (hβ2 : β < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ρ : ℝ, ρ ≠ 0 → ∀ θ : ℝ × ℝ, max |θ.1| |θ.2| = 1 →
      ∀ m : ℝ, ‖∫ u in m..m + 1,
        oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u *
          (gaussianPhaseAmplitude t u : ℂ)‖ ≤
        (C / Real.sqrt |ρ|) * ((1 + |m|) ^ 3 * Real.exp (-((2 - β) / 8) * m ^ 2)) := by
  let D := Real.sqrt (1 + 7 * β + 7 * β ^ 2 + β ^ 3) / Real.sqrt (3 * β)
  let L := β / 4 + 2 * (1 + |t| + β / 2) ^ 2 / (2 - β)
  let C := 640 * D * (2 + |t|) * Real.exp L
  have hD : 0 < D := by
    dsimp [D]
    exact div_pos (Real.sqrt_pos.mpr (by positivity)) (Real.sqrt_pos.mpr (by positivity))
  refine ⟨C, by dsimp [C]; positivity, ?_⟩
  intro ρ hρ θ hθ m
  let R := |m| + 1
  let E := Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|)
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hfloor := gaussianPhaseFloor_inv_sqrt_bound hβ hR
  have hpoly := gaussian_phase_polynomial_factor_bound m t
  have hexp : Real.exp (β * R ^ 2 / 4) * E ≤
      Real.exp L * Real.exp (-((2 - β) / 8) * m ^ 2) := by
    rw [← Real.exp_add, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    dsimp [E, R, L]
    linarith [gaussian_phase_exponent_bound hβ2 m t]
  have hinner : (1 / Real.sqrt (gaussianPhaseFloor β R)) * (2 + |m| + |t|) * E ≤
      D * (4 * (2 + |t|) * (1 + |m|) ^ 3) *
        (Real.exp L * Real.exp (-((2 - β) / 8) * m ^ 2)) := by
    calc _ ≤ (D * (1 + R) ^ 2 * Real.exp (β * R ^ 2 / 4)) * (2 + |m| + |t|) * E :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hfloor (by positivity))
            (Real.exp_pos _).le
      _ = D * ((2 + |m|) ^ 2 * (2 + |m| + |t|)) * (Real.exp (β * R ^ 2 / 4) * E) := by
        dsimp [R]
        ring
      _ ≤ _ := mul_le_mul (mul_le_mul_of_nonneg_left hpoly hD.le) hexp
        (by positivity) (by positivity)
  have h := gaussian_phase_weighted_unit_bound hρ hβ hθ m t
  have heq : (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) * ((2 + |m| + |t|) * E) =
      (160 / Real.sqrt |ρ|) * ((1 / Real.sqrt (gaussianPhaseFloor β R)) * (2 + |m| + |t|) * E) := by
    rw [Real.sqrt_mul (abs_nonneg ρ)]
    ring
  change _ ≤ (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) * ((2 + |m| + |t|) * E) at h
  rw [heq] at h
  calc _ ≤ (160 / Real.sqrt |ρ|) *
        ((1 / Real.sqrt (gaussianPhaseFloor β R)) * (2 + |m| + |t|) * E) := h
    _ ≤ (160 / Real.sqrt |ρ|) * (D * (4 * (2 + |t|) * (1 + |m|) ^ 3) *
        (Real.exp L * Real.exp (-((2 - β) / 8) * m ^ 2))) :=
      mul_le_mul_of_nonneg_left hinner (by positivity)
    _ = _ := by dsimp [C]; ring

example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧ (1 : ℝ) ≠ 0 ∧ max |(1 : ℝ)| |0| = 1 := by
  norm_num

end Transformer.Modes
