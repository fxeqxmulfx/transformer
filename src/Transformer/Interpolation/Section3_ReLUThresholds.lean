/-
# Real threshold tests for the barycenter perturbation argument

Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.
The remaining perturbation statement needs actual perceptron controls
which separate two measures, rather than a prescribed separating flow.

The ReLU payoffs `x ↦ max(b-x,0)` determine a finite real measure
whenever they are integrable. Positive finite differences of these
payoffs converge, with a uniform bound of one, to the indicator of
`(-∞,a]`. This recovers every closed half-line mass by dominated
convergence, including any atom at the threshold itself.

This is a measure-determination step toward constructing a neural
perturbation. It asserts neither existence nor uniqueness of the
continuity-equation solution in the source's still-open lemma.
-/

import Transformer.Interpolation.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- A positive threshold difference of the actual ReLU activation.
The ramp retains the value one at the closed half-line endpoint.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
noncomputable def reluThresholdRamp (a h x : ℝ) : ℝ :=
  (max (a + h - x) 0 - max (a - x) 0) / h

/-- Every positive-width ReLU threshold ramp lies between zero and one.
This bound supplies dominated convergence for arbitrary finite measures.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem reluThresholdRamp_bounds (a x h : ℝ) (hh : 0 < h) :
    0 ≤ reluThresholdRamp a h x ∧ reluThresholdRamp a h x ≤ 1 := by
  unfold reluThresholdRamp
  constructor
  · apply div_nonneg _ hh.le
    apply sub_nonneg.mpr
    exact max_le_max (by linarith) le_rfl
  · apply (div_le_iff₀ hh).mpr
    by_cases hx : x ≤ a
    · rw [max_eq_left (by linarith : 0 ≤ a + h - x), max_eq_left (by linarith : 0 ≤ a - x)]
      linarith
    · rw [max_eq_right (by linarith : a - x ≤ 0)]
      exact (by simpa using max_le (by linarith : a + h - x ≤ h) hh.le)

/-- Shrinking the positive width recovers the indicator of `(-∞,a]`
pointwise, with no exception for atoms or for the point `x=a`.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem tendsto_reluThresholdRamp (a x : ℝ) :
    Tendsto (fun n : ℕ => reluThresholdRamp a (1 / ((n : ℝ) + 1)) x)
      atTop (nhds ((Iic a).indicator (fun _ => (1 : ℝ)) x)) := by
  have hh (n : ℕ) : (0 : ℝ) < 1 / ((n : ℝ) + 1) := by positivity
  by_cases hx : x ≤ a
  · have heq (n : ℕ) : reluThresholdRamp a (1 / ((n : ℝ) + 1)) x = 1 := by
      unfold reluThresholdRamp
      rw [max_eq_left (by linarith [hh n] : 0 ≤ a + 1 / ((n : ℝ) + 1) - x),
        max_eq_left (by linarith : 0 ≤ a - x)]
      convert div_self (hh n).ne' using 1
      ring
    rw [Set.indicator_of_mem (show x ∈ Iic a from hx)]
    simpa only [heq] using (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1))
  · have hxpos : 0 < x - a := by linarith
    have hevent : ∀ᶠ n : ℕ in atTop, 1 / ((n : ℝ) + 1) < x - a :=
      (tendsto_one_div_add_atTop_nhds_zero_nat (𝕜 := ℝ)).eventually (Iio_mem_nhds hxpos)
    rw [Set.indicator_of_notMem (show x ∉ Iic a from hx)]
    apply Tendsto.congr' _ tendsto_const_nhds
    filter_upwards [hevent] with n hn
    unfold reluThresholdRamp
    rw [max_eq_right (by linarith : a + 1 / ((n : ℝ) + 1) - x ≤ 0),
      max_eq_right (by linarith : a - x ≤ 0)]
    simp

/-- Actual finite differences of ReLU integrals recover each half-line
mass. Finiteness of the measure supplies the integrable constant bound.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem tendsto_integral_reluThresholdRamp (μ : Measure ℝ) [IsFiniteMeasure μ] (a : ℝ) :
    Tendsto (fun n : ℕ => ∫ x, reluThresholdRamp a (1 / ((n : ℝ) + 1)) x ∂μ)
      atTop (nhds (μ.real (Iic a))) := by
  have ht := tendsto_integral_of_dominated_convergence (μ := μ) (fun _ : ℝ => (1 : ℝ))
    (F := fun n x => reluThresholdRamp a (1 / ((n : ℝ) + 1)) x)
    (f := (Iic a).indicator (fun _ => (1 : ℝ)))
    (fun n => (show Continuous (fun x => reluThresholdRamp a (1 / ((n : ℝ) + 1)) x) by
      unfold reluThresholdRamp
      fun_prop).aestronglyMeasurable)
    (integrable_const _) (fun n => ae_of_all _ fun x => ?_) (ae_of_all _ fun x => tendsto_reluThresholdRamp a x)
  · simpa using ht
  · have hb := reluThresholdRamp_bounds a x (1 / ((n : ℝ) + 1)) (by positivity)
    rw [Real.norm_eq_abs, abs_of_nonneg hb.1]
    exact hb.2

/-- Equality of all actual ReLU threshold integrals determines a finite
real measure. Integrability is explicit, so undefined integral values
cannot hide a failure of separation.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem measure_eq_of_relu_threshold_integrals (μ ν : Measure ℝ)
    [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (hμ : ∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) μ)
    (hν : ∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) ν)
    (heq : ∀ b : ℝ, (∫ x : ℝ, max (b - x) 0 ∂μ) = ∫ x : ℝ, max (b - x) 0 ∂ν) :
    μ = ν := by
  apply Measure.ext_of_Iic
  intro a
  have hi (n : ℕ) : (∫ x, reluThresholdRamp a (1 / ((n : ℝ) + 1)) x ∂μ) =
      ∫ x, reluThresholdRamp a (1 / ((n : ℝ) + 1)) x ∂ν := by
    unfold reluThresholdRamp
    rw [integral_div, integral_div, integral_sub (hμ _) (hμ _), integral_sub (hν _) (hν _), heq, heq]
  have hr : μ.real (Iic a) = ν.real (Iic a) := tendsto_nhds_unique
    (by simpa only [hi] using tendsto_integral_reluThresholdRamp μ a)
    (tendsto_integral_reluThresholdRamp ν a)
  have ho := congrArg ENNReal.ofReal hr
  simpa only [Measure.real, ENNReal.ofReal_toReal (measure_ne_top _ _)] using ho

/-- Distinct finite measures admit an actual distinguishing ReLU threshold.
This is the constructive witness form needed for a perturbation control.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_relu_threshold_distinguishes (μ ν : Measure ℝ)
    [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (hμ : ∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) μ)
    (hν : ∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) ν) (hne : μ ≠ ν) :
    ∃ b : ℝ, (∫ x : ℝ, max (b - x) 0 ∂μ) ≠ ∫ x : ℝ, max (b - x) 0 ∂ν := by
  by_contra h
  push Not at h
  exact hne (measure_eq_of_relu_threshold_integrals μ ν hμ hν h)

/-- Positive ramp widths and an atom at the threshold coexist. The value
at that atom remains one, as the limiting closed half-line requires. -/
example : (0 : ℝ) < 1 ∧ reluThresholdRamp 0 1 0 = 1 := by
  norm_num [reluThresholdRamp]

/-- Two genuine finite measures satisfy every integrability premise and
are distinguished by a threshold. This also realizes the inequality
hypothesis in `exists_relu_threshold_distinguishes`. -/
example :
    IsFiniteMeasure (Measure.dirac (0 : ℝ)) ∧ IsFiniteMeasure (Measure.dirac (2 : ℝ)) ∧
      (∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) (Measure.dirac 0)) ∧
      (∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) (Measure.dirac 2)) ∧
      Measure.dirac (0 : ℝ) ≠ Measure.dirac (2 : ℝ) ∧
      (∫ x : ℝ, max (1 - x) 0 ∂Measure.dirac 0) ≠
        ∫ x : ℝ, max (1 - x) 0 ∂Measure.dirac 2 := by
  have hd : (∫ x : ℝ, max (1 - x) 0 ∂Measure.dirac 0) ≠
      ∫ x : ℝ, max (1 - x) 0 ∂Measure.dirac 2 := by norm_num
  refine ⟨inferInstance, inferInstance, fun b => integrable_dirac (by simp),
    fun b => integrable_dirac (by simp), ?_, hd⟩
  intro h
  exact hd (congrArg (fun μ : Measure ℝ => ∫ x : ℝ, max (1 - x) 0 ∂μ) h)

/-- Identical nonzero Dirac laws jointly realize the integrability and
payoff-equality hypotheses in the measure-determination theorem. -/
example :
    (∀ b : ℝ, Integrable (fun x : ℝ => max (b - x) 0) (Measure.dirac 0)) ∧
      ∀ b : ℝ, (∫ x : ℝ, max (b - x) 0 ∂Measure.dirac 0) =
        ∫ x : ℝ, max (b - x) 0 ∂Measure.dirac 0 :=
  ⟨fun _ => integrable_dirac (by simp), fun _ => rfl⟩

end Transformer.Interpolation
