/-
# The probability law of the continuous optimizer Euler limit

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
This is the pushforward of the actual Brownian sample law by the proved
continuous modification. Its initial condition and all fixed-time
marginals follow from the constructed adapted mean-square limits.
Identification of its generator martingale problem is a separate step.
-/

import Transformer.BatchSize.Section4_EulerContinuousLimit

open MeasureTheory ProbabilityTheory Filter
open scoped NNReal

noncomputable section

namespace Transformer.BatchSize

variable {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
  (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
  (hη : 0 ≤ η) (hB : 0 < B) (hmodel : RegularGaussianModel f σ) (x₀ : EucSpace d)

/-- A continuous modification of the actual optimizer Euler limit,
Section 4.3 (2)--(3), selected from the proved existence theorem. -/
def optimizerContinuousEuler : ℝ≥0 → BrownianSample d → EucSpace d :=
  Classical.choose (optimizerEulerLimit_exists_continuous_modification method η B f σ hη hB hmodel x₀)

/-- Each evaluation of the continuous Euler limit is measurable,
Section 4.3 (2)--(3). -/
theorem optimizerContinuousEuler_measurable (t : ℝ≥0) :
    Measurable (optimizerContinuousEuler method η B f σ hη hB hmodel x₀ t) :=
  (Classical.choose_spec (optimizerEulerLimit_exists_continuous_modification method η B f σ hη hB hmodel x₀)).1 t

/-- The continuous modification preserves the actual Euler limit
at each fixed time, Section 4.3 (2)--(3). -/
theorem optimizerContinuousEuler_ae_eq (t : ℝ≥0) :
    optimizerContinuousEuler method η B f σ hη hB hmodel x₀ t =ᵐ[brownianNoiseLaw d]
      optimizerEulerLimit method η B f σ hη hB hmodel x₀ t :=
  (Classical.choose_spec (optimizerEulerLimit_exists_continuous_modification method η B f σ hη hB hmodel x₀)).2.1 t

/-- Every selected optimizer Euler sample path is continuous,
Section 4.3 (2)--(3). -/
theorem optimizerContinuousEuler_continuous (ω : BrownianSample d) :
    Continuous (fun t => optimizerContinuousEuler method η B f σ hη hB hmodel x₀ t ω) :=
  (Classical.choose_spec (optimizerEulerLimit_exists_continuous_modification method η B f σ hη hB hmodel x₀)).2.2 ω

/-- The actual continuous optimizer limit on the paper's path space,
Section 4.3 (2)--(3). Negative times are held at the time-zero value. -/
def optimizerEulerPath (ω : BrownianSample d) : DiffusionPath d :=
  ⟨fun u => optimizerContinuousEuler method η B f σ hη hB hmodel x₀ u.toNNReal ω,
    (optimizerContinuousEuler_continuous method η B f σ hη hB hmodel x₀ ω).comp continuous_real_toNNReal⟩

/-- Measurability of the actual continuous optimizer path sampler,
Section 4.3 (2)--(3). -/
theorem optimizerEulerPath_measurable :
    Measurable (optimizerEulerPath method η B f σ hη hB hmodel x₀) := by
  apply ContinuousMap.measurable_iff_eval.mpr
  intro u
  exact optimizerContinuousEuler_measurable method η B f σ hη hB hmodel x₀ u.toNNReal

/-- Each nonnegative path evaluation agrees almost surely with the
adapted mean-square Euler limit, Section 4.3 (2)--(3). -/
theorem optimizerEulerPath_eval_ae_eq (t : ℝ≥0) :
    (fun ω => optimizerEulerPath method η B f σ hη hB hmodel x₀ ω (t : ℝ)) =ᵐ[brownianNoiseLaw d]
      optimizerEulerLimit method η B f σ hη hB hmodel x₀ t := by
  simpa only [optimizerEulerPath, ContinuousMap.coe_mk, Real.toNNReal_coe] using
    optimizerContinuousEuler_ae_eq method η B f σ hη hB hmodel x₀ t

/-- The constructed optimizer path starts at the given state almost
surely under its actual Brownian driver, Section 4.3 (2)--(3). -/
theorem optimizerEulerPath_initial :
    ∀ᵐ ω ∂brownianNoiseLaw d, optimizerEulerPath method η B f σ hη hB hmodel x₀ ω 0 = x₀ := by
  filter_upwards [optimizerEulerPath_eval_ae_eq method η B f σ hη hB hmodel x₀ 0,
    optimizerEulerLimit_zero method η B f σ hη hB hmodel x₀] with ω hω hzero
  exact hω.trans hzero

/-- The genuine continuous-path probability law of the optimizer
Euler limit, Section 4.3 (2)--(3). -/
def optimizerEulerPathLaw : Measure (DiffusionPath d) :=
  (brownianNoiseLaw d).map (optimizerEulerPath method η B f σ hη hB hmodel x₀)

/-- The constructed optimizer limit's path law is normalized,
Section 4.3 (2)--(3). -/
theorem optimizerEulerPathLaw_probability :
    IsProbabilityMeasure (optimizerEulerPathLaw method η B f σ hη hB hmodel x₀) :=
  (Measure.isProbabilityMeasure_map_iff
    (optimizerEulerPath_measurable method η B f σ hη hB hmodel x₀).aemeasurable).mpr (by infer_instance)

/-- The continuous optimizer path law satisfies the paper's initial
condition almost surely, Section 4.3 (2)--(3). -/
theorem optimizerEulerPathLaw_initial :
    ∀ᵐ ω ∂optimizerEulerPathLaw method η B f σ hη hB hmodel x₀, ω 0 = x₀ := by
  have hevent : MeasurableSet {ω : DiffusionPath d | ω 0 = x₀} :=
    measurableSet_eq_fun (ContinuousMap.measurable_eval (0 : ℝ)) measurable_const
  rw [optimizerEulerPathLaw, ae_map_iff
    (optimizerEulerPath_measurable method η B f σ hη hB hmodel x₀).aemeasurable hevent]
  exact optimizerEulerPath_initial method η B f σ hη hB hmodel x₀

/-- All nonnegative marginals of the continuous optimizer path law
are precisely the constructed Euler-limit distributions,
Section 4.3 (2)--(3). -/
theorem optimizerEulerPathLaw_marginal (t : ℝ≥0) :
    HasLaw (fun ω : DiffusionPath d => ω (t : ℝ))
      ((brownianNoiseLaw d).map (optimizerEulerLimit method η B f σ hη hB hmodel x₀ t))
        (optimizerEulerPathLaw method η B f σ hη hB hmodel x₀) := by
  refine ⟨(ContinuousMap.measurable_eval (t : ℝ)).aemeasurable, ?_⟩
  rw [optimizerEulerPathLaw, Measure.map_map (ContinuousMap.measurable_eval (t : ℝ))
    (optimizerEulerPath_measurable method η B f σ hη hB hmodel x₀)]
  exact Measure.map_congr (optimizerEulerPath_eval_ae_eq method η B f σ hη hB hmodel x₀ t)

/-- Joint nonvacuity of every continuous-law hypothesis,
Section 4.3: positive rate and batch, flat loss, and unit noise. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧ 0 < (1 : ℕ) ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) :=
  ⟨by norm_num, by norm_num, regularGaussianModel_flat 2⟩

end Transformer.BatchSize
