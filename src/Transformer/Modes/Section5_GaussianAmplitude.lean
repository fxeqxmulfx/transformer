import Transformer.Modes.Section5_PhaseRoots

/-!
# The shifted Gaussian amplitude in the Fourier integral

In arXiv:2412.09080v3, §5.5, the Fourier integral has phase `g` and `g'`
and a Gaussian amplitude. Translating the random variable to make the
phase independent of `t` changes that amplitude to
`exp (-(u - t)²/2) / sqrt (2π)`. The source's reduction to `t = 0` by
"translation invariance of the standard Gaussian" omits this change.

This module retains the shifted amplitude, before its constant Gaussian
normalization. Its derivative is the actual derivative of that function.
The local Gaussian phase bound transfers to the weighted integral by
integration by parts against the phase primitive; both the endpoint
amplitude and its variation remain in the bound.

For `u ∈ [m, m + 1]`, expanding `u - t = m + (u - m - t)` gives
`(u - t)² ≥ m² - 2 (1 + |t|) |m|`. Therefore the amplitude is bounded by
`exp (-m²/2 + (1 + |t|) |m|)` throughout the interval. Its endpoint plus
variation is at most `2 + |m| + |t|` times this exponential. These
estimates hold for every real `m` and `t`; no tail is discarded.

The next step must combine this amplitude bound with the growing inverse
square root of `gaussianPhaseFloor β (|m| + 1)`. Their exponential rates
leave a summable Gaussian precisely when `β < 2`.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The shifted Gaussian amplitude after `u = t - x`, before dividing
by `sqrt (2π)`. Source: arXiv:2412.09080v3, §5.5, the Fourier integral. -/
noncomputable def gaussianPhaseAmplitude (t u : ℝ) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * (u - t) ^ 2)

/-- The actual Gaussian amplitude is positive everywhere.
Source: arXiv:2412.09080v3, §5.5, the Gaussian weight. -/
theorem gaussianPhaseAmplitude_pos (t u : ℝ) : 0 < gaussianPhaseAmplitude t u :=
  Real.exp_pos _

/-- Differentiating the shifted amplitude retains its dependence on `t`.
Source: arXiv:2412.09080v3, §5.5, the amplitude derivative. -/
theorem hasDerivAt_gaussianPhaseAmplitude (t u : ℝ) :
    HasDerivAt (gaussianPhaseAmplitude t) (-(u - t) * gaussianPhaseAmplitude t u) u := by
  unfold gaussianPhaseAmplitude
  simpa only [one_mul] using hasDerivAt_bump 1 t u

/-- The local phase estimate with the actual shifted Gaussian weight.
Source: arXiv:2412.09080v3, §5.5, the weighted stationary-phase bound. -/
theorem weighted_gaussian_phase_unit_interval_bound {a b β ρ R : ℝ} {θ : ℝ × ℝ}
    (hab : a ≤ b) (hlen : b - a ≤ 1) (hρ : ρ ≠ 0) (hβ : 0 < β) (hR : 0 ≤ R)
    (hθ : max |θ.1| |θ.2| = 1) (harg : ∀ u ∈ Set.uIcc a b, |u| ≤ R) (t : ℝ) :
    ‖∫ u in a..b, oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u *
      (gaussianPhaseAmplitude t u : ℂ)‖ ≤
      (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R)) *
        (gaussianPhaseAmplitude t b + ∫ u in a..b, |u - t| * gaussianPhaseAmplitude t u) := by
  have h := weighted_integral_le_of_primitive_bound hab
    (f := oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0))
    (A := gaussianPhaseAmplitude t) (B := fun u => -(u - t) * gaussianPhaseAmplitude t u)
    (by unfold oscillatoryKernel bigG bigG'; fun_prop)
    (fun u _ => hasDerivAt_gaussianPhaseAmplitude t u)
    (by unfold gaussianPhaseAmplitude; fun_prop) (C := 160 / Real.sqrt (|ρ| * gaussianPhaseFloor β R))
    (fun y hy => by
      rw [Set.uIcc_of_le hab] at hy
      apply gaussian_phase_unit_interval_bound hy.1 (by linarith [hy.2]) hρ hβ hR hθ
      intro u hu
      apply harg
      rw [Set.uIcc_of_le hy.1] at hu
      rw [Set.uIcc_of_le hab]
      exact ⟨hu.1, hu.2.trans hy.2⟩)
  simpa only [gaussianPhaseAmplitude, abs_mul, abs_neg, abs_of_pos (Real.exp_pos _)] using h

example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) - 0 ≤ 1 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧
    (0 : ℝ) ≤ 1 ∧ max |(1 : ℝ)| |0| = 1 ∧
    (∀ u ∈ Set.uIcc (0 : ℝ) 1, |u| ≤ 1) := by
  refine ⟨zero_le_one, by norm_num, one_ne_zero, one_pos, zero_le_one, by norm_num, ?_⟩
  intro u hu
  rw [Set.uIcc_of_le zero_le_one] at hu
  rw [abs_of_nonneg hu.1]
  exact hu.2

/-- A unit interval stays inside the argument radius `|m| + 1`.
Source: arXiv:2412.09080v3, §5.5, the argument bound for the phase. -/
theorem unit_interval_abs_le {m u : ℝ} (hu : u ∈ Set.Icc m (m + 1)) : |u| ≤ |m| + 1 := by
  have huv : 0 ≤ u - m ∧ u - m ≤ 1 := ⟨by linarith [hu.1], by linarith [hu.2]⟩
  calc |u| = |m + (u - m)| := by congr 1; ring
    _ ≤ |m| + |u - m| := abs_add_le _ _
    _ ≤ |m| + 1 := by rw [abs_of_nonneg huv.1]; exact add_le_add le_rfl huv.2

example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) (0 + 1) := by norm_num

/-- A Gaussian envelope valid throughout the whole unit interval.
Source: arXiv:2412.09080v3, §5.5, replacing the fixed-radius tail estimate. -/
theorem gaussianPhaseAmplitude_unit_bound {m u : ℝ} (hu : u ∈ Set.Icc m (m + 1)) (t : ℝ) :
    gaussianPhaseAmplitude t u ≤ Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|) := by
  have huv : 0 ≤ u - m ∧ u - m ≤ 1 := ⟨by linarith [hu.1], by linarith [hu.2]⟩
  have hv : |u - m - t| ≤ 1 + |t| := by
    calc _ ≤ |u - m| + |t| := abs_sub _ _
      _ ≤ 1 + |t| := by rw [abs_of_nonneg huv.1]; exact add_le_add huv.2 le_rfl
  have hmul : |m * (u - m - t)| ≤ |m| * (1 + |t|) := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left hv (abs_nonneg m)
  have hlo : -(|m| * (1 + |t|)) ≤ m * (u - m - t) :=
    (neg_le_neg hmul).trans (neg_abs_le _)
  unfold gaussianPhaseAmplitude
  apply Real.exp_le_exp.mpr
  nlinarith [sq_nonneg (u - m - t)]

example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) (0 + 1) := by norm_num

/-- The endpoint amplitude plus its total variation has the same Gaussian
envelope, with one additional linear factor.
Source: arXiv:2412.09080v3, §5.5, the amplitude and derivative estimates. -/
theorem gaussianPhaseAmplitude_variation_bound (m t : ℝ) :
    gaussianPhaseAmplitude t (m + 1) + ∫ u in m..m + 1, |u - t| * gaussianPhaseAmplitude t u ≤
      (2 + |m| + |t|) * Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|) := by
  let E := Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|)
  have he := gaussianPhaseAmplitude_unit_bound (m := m) ⟨by linarith, le_rfl⟩ t
  have hJ : (∫ u in m..m + 1, |u - t| * gaussianPhaseAmplitude t u) ≤ (|m| + 1 + |t|) * E := by
    have hpoint (u : ℝ) (hu : u ∈ Set.Icc m (m + 1)) :
        |u - t| * gaussianPhaseAmplitude t u ≤ (|m| + 1 + |t|) * E := by
      have huabs : |u - t| ≤ |m| + 1 + |t| :=
        (abs_sub _ _).trans (add_le_add (unit_interval_abs_le hu) le_rfl)
      exact mul_le_mul huabs (gaussianPhaseAmplitude_unit_bound hu t)
        (gaussianPhaseAmplitude_pos t u).le (by positivity)
    have h := intervalIntegral.integral_mono_on (μ := volume) (a := m) (b := m + 1) (by linarith)
      (by unfold gaussianPhaseAmplitude; exact (by fun_prop : Continuous (fun u : ℝ =>
        |u - t| * Real.exp (-(1 / 2 : ℝ) * (u - t) ^ 2))).intervalIntegrable m (m + 1))
      (continuous_const.intervalIntegrable m (m + 1)) hpoint
    simpa only [intervalIntegral.integral_const, smul_eq_mul, add_sub_cancel_left, one_mul] using h
  calc _ ≤ E + (|m| + 1 + |t|) * E := add_le_add he hJ
    _ = _ := by dsimp [E]; ring

/-- A complete weighted bound on each unit interval, with the radius loss
and the shifted Gaussian envelope both visible.
Source: arXiv:2412.09080v3, §5.5, combining the local phase and tail estimates. -/
theorem gaussian_phase_weighted_unit_bound {β ρ : ℝ} {θ : ℝ × ℝ}
    (hρ : ρ ≠ 0) (hβ : 0 < β) (hθ : max |θ.1| |θ.2| = 1) (m t : ℝ) :
    ‖∫ u in m..m + 1, oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u *
      (gaussianPhaseAmplitude t u : ℂ)‖ ≤
      (160 / Real.sqrt (|ρ| * gaussianPhaseFloor β (|m| + 1))) *
        ((2 + |m| + |t|) * Real.exp (-(m ^ 2) / 2 + (1 + |t|) * |m|)) := by
  have h := weighted_gaussian_phase_unit_interval_bound (a := m) (b := m + 1)
    (by linarith) (by linarith) hρ hβ (R := |m| + 1) (by positivity) hθ
    (fun u hu => unit_interval_abs_le (by simpa only [Set.uIcc_of_le (by linarith : m ≤ m + 1)]
      using hu)) t
  exact h.trans (mul_le_mul_of_nonneg_left (gaussianPhaseAmplitude_variation_bound m t)
    (by positivity))

example : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ max |(1 : ℝ)| |0| = 1 := by norm_num

end Transformer.Modes
