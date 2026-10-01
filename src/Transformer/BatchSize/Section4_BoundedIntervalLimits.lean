/-
# Bounded generator integrals with moving grid endpoints

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
Uniform actual generator bounds permit dominated convergence on a fixed
interval and control the vanishing piece at a neighboring grid endpoint.
-/

import Transformer.BatchSize.Section4_BoundedObservableLimits

open MeasureTheory Filter
open scoped Topology

noncomputable section

namespace Transformer.BatchSize

/-- A genuine measurable bounded real function is integrable on
every finite physical interval, Section 4.3 (2)--(3). -/
theorem bounded_measurable_intervalIntegrable (g : ℝ → ℝ) (hg : Measurable g)
    (C : ℝ) (hC : ∀ u, |g u| ≤ C) (s t : ℝ) : IntervalIntegrable g volume s t := by
  apply intervalIntegrable_iff.mpr
  apply Measure.integrableOn_of_bounded
    (by rw [Real.volume_uIoc]; exact ENNReal.ofReal_ne_top) hg.aestronglyMeasurable
  exact Eventually.of_forall fun u => by simpa only [Real.norm_eq_abs] using hC u

/-- Bounded actual generator integrals converge even when their
starting dyadic node moves toward the observation time,
Section 4.3 (2)--(3). Both integrability and the endpoint error
are proved; no default-zero integral is used as a substitute. -/
theorem bounded_intervalIntegral_moving_start_tendsto
    (G : ℕ → ℝ → ℝ) (g : ℝ → ℝ) (hGm : ∀ m, Measurable (G m))
    (C : ℝ) (hC : ∀ m u, |G m u| ≤ C) (s t : ℝ)
    (hlim : ∀ u ∈ Set.uIoc s t, Tendsto (fun m => G m u) atTop (𝓝 (g u)))
    (start : ℕ → ℝ) (hstart : Tendsto start atTop (𝓝 s)) :
    Tendsto (fun m => ∫ u in start m..t, G m u) atTop (𝓝 (∫ u in s..t, g u)) := by
  have hfixed : Tendsto (fun m => ∫ u in s..t, G m u) atTop (𝓝 (∫ u in s..t, g u)) :=
    intervalIntegral.tendsto_integral_filter_of_dominated_convergence (fun _ => C)
      (Eventually.of_forall fun m => (hGm m).aestronglyMeasurable.restrict)
      (Eventually.of_forall fun m => Eventually.of_forall fun u _ => by
        simpa only [Real.norm_eq_abs] using hC m u)
      intervalIntegrable_const (Eventually.of_forall hlim)
  have hsmall : Tendsto (fun m => ∫ u in s..start m, G m u) atTop (𝓝 0) := by
    apply squeeze_zero_norm (a := fun m => C * |start m - s|) (fun m =>
      intervalIntegral.norm_integral_le_of_norm_le_const (fun u _ => by
        simpa only [Real.norm_eq_abs] using hC m u))
    have h := (hstart.sub (tendsto_const_nhds (x := s))).abs.const_mul C
    simpa only [sub_self, abs_zero, mul_zero] using h
  have heq (m : ℕ) : (∫ u in start m..t, G m u) =
      (∫ u in s..t, G m u) - ∫ u in s..start m, G m u := by
    have h := intervalIntegral.integral_add_adjacent_intervals
      (bounded_measurable_intervalIntegrable (G m) (hGm m) C (hC m) s (start m))
      (bounded_measurable_intervalIntegrable (G m) (hGm m) C (hC m) (start m) t)
    linarith
  simpa only [heq, sub_zero] using hfixed.sub hsmall

/-- Joint nonvacuity of the bounded interval convergence
hypotheses, Section 4.3: the genuine nonconstant sine function,
a positive horizon and its constant initial observation node. -/
example : (∀ _ : ℕ, Measurable (Real.sin : ℝ → ℝ)) ∧
    (∀ (m : ℕ) (u : ℝ), |(fun _ : ℕ => Real.sin) m u| ≤ 1) ∧
    (∀ u ∈ Set.uIoc (0 : ℝ) 1,
      Tendsto (fun _ : ℕ => Real.sin u) atTop (𝓝 (Real.sin u))) ∧
    Tendsto (fun _ : ℕ => (0 : ℝ)) atTop (𝓝 0) :=
  ⟨fun _ => Real.continuous_sin.measurable, fun _ u => Real.abs_sin_le_one u,
    fun _ _ => tendsto_const_nhds, tendsto_const_nhds⟩

end Transformer.BatchSize
