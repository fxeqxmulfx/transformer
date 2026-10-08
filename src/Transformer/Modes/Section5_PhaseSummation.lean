import Transformer.Modes.Section5_PhaseEnvelope
import Mathlib.Algebra.Order.ToIntervalMod
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Topology.Algebra.InfiniteSum.NatInt

/-!
# Summing the Gaussian phase estimates on the whole real line

The proof of `eq:uniform-decay` in arXiv:2412.09080v3, §5.5, needs a
frequency-dependent bound on the entire Fourier integral. Its fixed
cutoff leaves a tail independent of frequency. Here the real line is
partitioned into all unit intervals, and every local integral is kept.

`Section5_PhaseEnvelope` bounds each local integral by a uniform
frequency factor times a cubic polynomial and
`exp (-(2 - β)m²/8)`. This module proves that envelope is summable over
integer indices for `β < 2`. On the nonnegative integers, expand the
cubic and compare each Gaussian with a decaying exponential; then join
the positive and negative indices.

The half-open intervals `(m, m + 1]` are pairwise disjoint and cover the
real line. For an integrable function, their integral sum is exactly the
whole integral. Passing to interval integrals preserves this identity
and lets the estimates from `Section5_PhaseEnvelope` apply directly.

The integrand is the actual oscillatory exponential times the shifted
Gaussian amplitude. Its norm equals that amplitude, so integrability is
proved independently of the oscillatory estimates. The final theorem
sums the local estimates to give a single bound `C / sqrt |ρ|`, uniform
in the normalized direction. The constant depends on `β` and fixed `t`.
There is no unproved regularity, partition, or summability input.
Source: arXiv:2412.09080v3, §5.5, proof of `eq:uniform-decay`.
-/

open Real MeasureTheory
open scoped Interval

namespace Transformer.Modes

/-- The cubic envelope left by the local phase estimate is summable
when its Gaussian rate is positive.
Source: arXiv:2412.09080v3, §5.5, the corrected whole-line tail estimate. -/
theorem summable_cubic_gaussian {c : ℝ} (hc : 0 < c) :
    Summable fun m : ℤ => (1 + |(m : ℝ)|) ^ 3 * Real.exp (-c * (m : ℝ) ^ 2) := by
  have hlin : Summable fun n : ℕ => (1 + (n : ℝ)) ^ 3 * Real.exp (-c * n) := by
    have h0 := Real.summable_pow_mul_exp_neg_nat_mul 0 hc
    have h1 := (Real.summable_pow_mul_exp_neg_nat_mul 1 hc).mul_left 3
    have h2 := (Real.summable_pow_mul_exp_neg_nat_mul 2 hc).mul_left 3
    have h3 := Real.summable_pow_mul_exp_neg_nat_mul 3 hc
    convert ((h0.add h1).add h2).add h3 using 1
    funext n
    ring
  have hnat : Summable fun n : ℕ => (1 + (n : ℝ)) ^ 3 * Real.exp (-c * (n : ℝ) ^ 2) := by
    apply Summable.of_nonneg_of_le (fun n => by positivity) (fun n => ?_) hlin
    apply mul_le_mul_of_nonneg_left _ (by positivity)
    apply Real.exp_le_exp.mpr
    have hsq : (n : ℝ) ≤ (n : ℝ) ^ 2 := by exact_mod_cast Nat.le_self_pow two_ne_zero n
    nlinarith
  exact Summable.of_nat_of_neg (by simpa using hnat) (by simpa using hnat)

example : (0 : ℝ) < 1 := one_pos

/-- The integral over all unit intervals sums to the whole integral.
The disjoint half-open sets cover every point exactly once.
Source: arXiv:2412.09080v3, §5.5, partitioning the Fourier integral. -/
theorem integral_eq_tsum_unit_intervals {f : ℝ → ℂ} (hf : Integrable f) :
    (∫ u : ℝ, f u) = ∑' m : ℤ, ∫ u in (m : ℝ)..(m : ℝ) + 1, f u := by
  have h := integral_iUnion (s := fun m : ℤ => Set.Ioc (m : ℝ) (m + 1))
    (fun _ => measurableSet_Ioc) (Set.pairwise_disjoint_Ioc_intCast ℝ) hf.integrableOn
  rw [iUnion_Ioc_intCast, Measure.restrict_univ] at h
  calc _ = ∑' m : ℤ, ∫ u in Set.Ioc (m : ℝ) (m + 1), f u := h
    _ = _ := tsum_congr (fun m => (intervalIntegral.integral_of_le (by linarith :
      (m : ℝ) ≤ m + 1)).symm)

/-- The actual shifted Fourier integrand, before the `sqrt (2π)`
normalization. Source: arXiv:2412.09080v3, §5.5, the Fourier display. -/
noncomputable def gaussianPhaseIntegrand (β ρ : ℝ) (θ : ℝ × ℝ) (t u : ℝ) : ℂ :=
  oscillatoryKernel ρ (fun u => θ.1 * bigG β u 0 + θ.2 * bigG' β u 0) u *
    (gaussianPhaseAmplitude t u : ℂ)

/-- The oscillatory factor has unit norm, leaving exactly the Gaussian
amplitude. Source: arXiv:2412.09080v3, §5.5, the absolute tail bound. -/
theorem norm_gaussianPhaseIntegrand (β ρ : ℝ) (θ : ℝ × ℝ) (t u : ℝ) :
    ‖gaussianPhaseIntegrand β ρ θ t u‖ = gaussianPhaseAmplitude t u := by
  rw [gaussianPhaseIntegrand, norm_mul, norm_oscillatoryKernel, one_mul,
    Complex.norm_real, Real.norm_eq_abs, abs_of_pos (gaussianPhaseAmplitude_pos t u)]

/-- The Fourier integrand is continuous in its real argument for every
choice of its parameters. Source: arXiv:2412.09080v3, §5.5, the Fourier display. -/
theorem continuous_gaussianPhaseIntegrand (β ρ : ℝ) (θ : ℝ × ℝ) (t : ℝ) :
    Continuous (gaussianPhaseIntegrand β ρ θ t) := by
  unfold gaussianPhaseIntegrand oscillatoryKernel gaussianPhaseAmplitude bigG bigG'
  fun_prop

/-- The actual Fourier integrand is dominated by an integrable Gaussian.
Source: arXiv:2412.09080v3, §5.5, the Fourier integral. -/
theorem integrable_gaussianPhaseIntegrand (β ρ : ℝ) (θ : ℝ × ℝ) (t : ℝ) :
    Integrable (gaussianPhaseIntegrand β ρ θ t) := by
  have hA : Integrable (gaussianPhaseAmplitude t) :=
    (integrable_exp_neg_mul_sq (by norm_num : (0 : ℝ) < 1 / 2)).comp_sub_right t
  exact hA.mono' (continuous_gaussianPhaseIntegrand β ρ θ t).aestronglyMeasurable
    (ae_of_all _ fun u => (norm_gaussianPhaseIntegrand β ρ θ t u).le)

example : Integrable (gaussianPhaseIntegrand 1 1 (1, 0) 0) :=
  integrable_gaussianPhaseIntegrand 1 1 (1, 0) 0

/-- Translation preserves the total Gaussian amplitude, even though the
oscillatory integral still depends on `t`. This is the normalization in
the source's Fourier display, with the shifted weight retained.
Source: arXiv:2412.09080v3, §5.5, the factor `1 / sqrt (2π)`. -/
theorem gaussianPhaseAmplitude_integral (t : ℝ) :
    (∫ u : ℝ, gaussianPhaseAmplitude t u) = Real.sqrt (2 * Real.pi) := by
  calc _ = ∫ u : ℝ, Real.exp (-(1 / 2 : ℝ) * u ^ 2) :=
        integral_sub_right_eq_self (fun u : ℝ => Real.exp (-(1 / 2 : ℝ) * u ^ 2)) t
    _ = Real.sqrt (Real.pi / (1 / 2 : ℝ)) := integral_gaussian _
    _ = _ := by
      congr 1
      ring

/-- Summing every local estimate gives the entire oscillatory integral a
uniform inverse square root bound in frequency, for `0 < β < 2`.
Source: arXiv:2412.09080v3, §5.5, the corrected `eq:uniform-decay`. -/
theorem gaussian_phase_integral_bound {β : ℝ} (hβ : 0 < β) (hβ2 : β < 2) (t : ℝ) :
    ∃ C : ℝ, 0 < C ∧ ∀ ρ : ℝ, ρ ≠ 0 → ∀ θ : ℝ × ℝ, max |θ.1| |θ.2| = 1 →
      ‖∫ u : ℝ, gaussianPhaseIntegrand β ρ θ t u‖ ≤ C / Real.sqrt |ρ| := by
  obtain ⟨C, hC, hlocal⟩ := gaussian_phase_local_envelope hβ hβ2 t
  let F := fun m : ℤ => (1 + |(m : ℝ)|) ^ 3 * Real.exp (-((2 - β) / 8) * (m : ℝ) ^ 2)
  have hF : Summable F := summable_cubic_gaussian (by linarith)
  have hS : 0 ≤ ∑' m, F m := tsum_nonneg (fun _ => by dsimp [F]; positivity)
  refine ⟨C * (1 + ∑' m, F m), by positivity, ?_⟩
  intro ρ hρ θ hθ
  let J := fun m : ℤ => ∫ u in (m : ℝ)..(m : ℝ) + 1, gaussianPhaseIntegrand β ρ θ t u
  have hb (m : ℤ) : ‖J m‖ ≤ (C / Real.sqrt |ρ|) * F m := hlocal ρ hρ θ hθ m
  have hmajor : Summable fun m : ℤ => (C / Real.sqrt |ρ|) * F m := hF.mul_left _
  have hJ : Summable fun m : ℤ => ‖J m‖ :=
    Summable.of_nonneg_of_le (fun m => norm_nonneg _) hb hmajor
  rw [integral_eq_tsum_unit_intervals (integrable_gaussianPhaseIntegrand β ρ θ t)]
  change ‖∑' m, J m‖ ≤ _
  calc _ ≤ ∑' m, ‖J m‖ := norm_tsum_le_tsum_norm hJ
    _ ≤ ∑' m, (C / Real.sqrt |ρ|) * F m := hJ.tsum_le_tsum hb hmajor
    _ = (C / Real.sqrt |ρ|) * ∑' m, F m := tsum_mul_left
    _ ≤ C * (1 + ∑' m, F m) / Real.sqrt |ρ| := by
      have hnon : 0 ≤ C / Real.sqrt |ρ| := by positivity
      have hh := mul_le_mul_of_nonneg_left (show (∑' m, F m) ≤ 1 + ∑' m, F m by linarith) hnon
      convert hh using 1
      ring

example : (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 ∧ (1 : ℝ) ≠ 0 ∧ max |(1 : ℝ)| |0| = 1 := by
  norm_num

end Transformer.Modes
