/-
# Bounded continuous observables along actual measurable paths

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
These bounds ensure that the generator and drift expectations used
in the weak comparison are genuine probability and time integrals.
-/

import Transformer.BatchSize.Section4_PathCompensation
import Transformer.BatchSize.Section4_BoundedIntervalLimits

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Expected bounded continuous path observables are measurable
and bounded at every real time, Section 4.3, Theorem 1. -/
theorem continuousPath_expectation_measurable_bounded {Ω : Type*} [MeasurableSpace Ω]
    {d : ℕ} (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → DiffusionPath d)
    (hZ : Measurable Z) (g : EucSpace d → ℝ) (hg : Continuous g)
    (K : ℝ) (hK : ∀ x, |g x| ≤ K) :
    Measurable (fun u : ℝ => ∫ ω, g (Z ω u) ∂P) ∧
      ∀ u : ℝ, |∫ ω, g (Z ω u) ∂P| ≤ K := by
  have hm : Measurable (fun p : ℝ × Ω => g (Z p.2 p.1)) :=
    hg.measurable.comp (continuousPath_measurable_time_sample Z hZ)
  refine ⟨hm.stronglyMeasurable.integral_prod_right'.measurable, ?_⟩
  intro u
  have h := norm_integral_le_of_norm_le_const (f := fun ω => g (Z ω u))
    (ae_of_all P (fun ω => by simpa only [Real.norm_eq_abs] using hK (Z ω u)))
  simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using h

/-- Every fixed-time bounded continuous path observable is genuinely
integrable under the probability law, Section 4.3, Theorem 1. -/
theorem continuousPath_observable_integrable {Ω : Type*} [MeasurableSpace Ω]
    {d : ℕ} (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → DiffusionPath d)
    (hZ : Measurable Z) (g : EucSpace d → ℝ) (hg : Continuous g)
    (K : ℝ) (hK : ∀ x, |g x| ≤ K) (t : ℝ) :
    Integrable (fun ω => g (Z ω t)) P := by
  have hm : Measurable (fun ω => g (Z ω t)) :=
    hg.measurable.comp ((ContinuousMap.measurable_eval t).comp hZ)
  apply (integrable_const K).mono' hm.aestronglyMeasurable
  exact ae_of_all _ (fun ω => by simpa only [Real.norm_eq_abs] using hK (Z ω t))

/-- Every finite-time expected bounded continuous path observable
is actually interval integrable, Section 4.3, Theorem 1. -/
theorem continuousPath_expectation_intervalIntegrable {Ω : Type*} [MeasurableSpace Ω]
    {d : ℕ} (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → DiffusionPath d)
    (hZ : Measurable Z) (g : EucSpace d → ℝ) (hg : Continuous g)
    (K : ℝ) (hK : ∀ x, |g x| ≤ K) (s t : ℝ) :
    IntervalIntegrable (fun u => ∫ ω, g (Z ω u) ∂P) volume s t := by
  obtain ⟨hm, hb⟩ := continuousPath_expectation_measurable_bounded P Z hZ g hg K hK
  exact bounded_measurable_intervalIntegrable _ hm K hb s t

/-- Joint nonvacuity of the path expectation hypotheses,
Section 4.3: a genuine probability measure, constant continuous paths
and a nonconstant bounded continuous spatial observable. -/
example : IsProbabilityMeasure (Measure.dirac ()) ∧
    Measurable (fun _ : Unit => (ContinuousMap.const ℝ (0 : EucSpace 1))) ∧
    Continuous (fun x : EucSpace 1 => Real.sin (x 0)) ∧
    (∀ x : EucSpace 1, |Real.sin (x 0)| ≤ 1) :=
  ⟨inferInstance, measurable_const,
    Real.continuous_sin.comp (PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) 0),
    fun x => Real.abs_sin_le_one (x 0)⟩

end Transformer.BatchSize
