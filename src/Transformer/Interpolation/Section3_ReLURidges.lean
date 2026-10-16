/-
# ReLU ridge tests determine measures on the sphere

Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.
A ReLU ridge is an actual single-neuron activation of an affine
functional of the token. Equality of all such integrals determines
a finite measure, so two distinct measures have a distinguishing ridge.

The proof first recovers every one-dimensional linear projection using
the threshold tests in `Section3_ReLUThresholds`. Characteristic-function
uniqueness then recovers the ambient measure. A measurable embedding
transfers the result to the original sphere. Compactness supplies all
integrability conditions there, including for atomic measures.

This is the measure-separation input toward an actual perceptron
perturbation, not an assumption of the desired separation conclusion.
-/

import Transformer.Interpolation.Section3_ReLUThresholds
import Mathlib.MeasureTheory.Measure.CharacteristicFunction.Basic
import Mathlib.MeasureTheory.Function.LocallyIntegrable

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- All actual affine ReLU integrals determine a finite Euclidean measure.
Integrability is explicit; equality of undefined integrals is insufficient.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem measure_eq_of_relu_ridge_integrals {d : ℕ} (μ ν : Measure (EucSpace d))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (hμ : ∀ (u : EucSpace d) (b : ℝ), Integrable (fun x => max (b - inner ℝ x u) 0) μ)
    (hν : ∀ (u : EucSpace d) (b : ℝ), Integrable (fun x => max (b - inner ℝ x u) 0) ν)
    (heq : ∀ (u : EucSpace d) (b : ℝ),
      (∫ x, max (b - inner ℝ x u) 0 ∂μ) = ∫ x, max (b - inner ℝ x u) 0 ∂ν) :
    μ = ν := by
  have hproj (u : EucSpace d) : μ.map (fun x => inner ℝ x u) = ν.map (fun x => inner ℝ x u) := by
    apply measure_eq_of_relu_threshold_integrals
    · intro b
      exact (integrable_map_measure (by fun_prop) (by fun_prop)).mpr (hμ u b)
    · intro b
      exact (integrable_map_measure (by fun_prop) (by fun_prop)).mpr (hν u b)
    · intro b
      rw [integral_map (by fun_prop) (by fun_prop), integral_map (by fun_prop) (by fun_prop)]
      exact heq u b
  apply Measure.ext_of_charFun
  funext u
  have hm := charFun_map_eq_charFun_map_inner_one (μ := μ) (Y := id) aemeasurable_id u
  have hn := charFun_map_eq_charFun_map_inner_one (μ := ν) (Y := id) aemeasurable_id u
  simp only [Measure.map_id, id_eq] at hm hn
  rw [hproj u] at hm
  exact hm.trans hn.symm


/-- All affine ReLU integrals determine the original finite measure on the
sphere. Compactness supplies integrability, with no density or atomlessness
hypothesis. Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem measure_eq_of_sphere_relu_ridges {d : ℕ} (μ ν : Measure (SSphere d))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (heq : ∀ (u : EucSpace d) (b : ℝ),
      (∫ x, max (b - inner ℝ (x : EucSpace d) u) 0 ∂μ) =
        ∫ x, max (b - inner ℝ (x : EucSpace d) u) 0 ∂ν) :
    μ = ν := by
  let : CompactSpace (SSphere d) := Metric.sphere.compactSpace (0 : EucSpace d) 1
  have hm : MeasurableEmbedding (fun x : SSphere d => (x : EucSpace d)) :=
    Metric.isClosed_sphere.isClosedEmbedding_subtypeVal.measurableEmbedding
  apply hm.map_injective
  apply measure_eq_of_relu_ridge_integrals
  · intro u b
    apply (integrable_map_measure (by fun_prop) hm.measurable.aemeasurable).mpr
    exact (show Continuous (fun x : SSphere d => max (b - inner ℝ (x : EucSpace d) u) 0) by
      fun_prop).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  · intro u b
    apply (integrable_map_measure (by fun_prop) hm.measurable.aemeasurable).mpr
    exact (show Continuous (fun x : SSphere d => max (b - inner ℝ (x : EucSpace d) u) 0) by
      fun_prop).integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  · intro u b
    rw [hm.integral_map, hm.integral_map]
    exact heq u b

/-- Two distinct finite sphere measures have an actual distinguishing
single-neuron ridge, with its vector and threshold produced in the conclusion.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_sphere_relu_ridge_distinguishes {d : ℕ} (μ ν : Measure (SSphere d))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν] (hne : μ ≠ ν) :
    ∃ (u : EucSpace d) (b : ℝ),
      (∫ x, max (b - inner ℝ (x : EucSpace d) u) 0 ∂μ) ≠
        ∫ x, max (b - inner ℝ (x : EucSpace d) u) 0 ∂ν := by
  by_contra h
  push Not at h
  exact hne (measure_eq_of_sphere_relu_ridges μ ν h)



open scoped ENNReal

/-- A continuous weight positive almost everywhere under both measures
preserves ridge separation. The weighted measures are genuine finite
measures, and multiplying their equality by the reciprocal weight would
recover the original measures. This is the form needed when the spherical
projection contributes a positive factor to a perceptron variation.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem exists_positive_weighted_sphere_relu_ridge {d : ℕ} (μ ν : Measure (SSphere d))
    [IsFiniteMeasure μ] [IsFiniteMeasure ν] (w : SSphere d → ℝ) (hw : Continuous w)
    (hμ : ∀ᵐ x ∂μ, 0 < w x) (hν : ∀ᵐ x ∂ν, 0 < w x) (hne : μ ≠ ν) :
    ∃ (u : EucSpace d) (b : ℝ),
      (∫ x, w x * max (b - inner ℝ (x : EucSpace d) u) 0 ∂μ) ≠
        ∫ x, w x * max (b - inner ℝ (x : EucSpace d) u) 0 ∂ν := by
  let : CompactSpace (SSphere d) := Metric.sphere.compactSpace (0 : EucSpace d) 1
  let f : SSphere d → ℝ≥0∞ := fun x => ENNReal.ofReal (w x)
  have hf : Measurable f := hw.measurable.ennreal_ofReal
  have htop (ξ : Measure (SSphere d)) : ∀ᵐ x ∂ξ, f x ≠ ⊤ :=
    ae_of_all _ fun _ => ENNReal.ofReal_ne_top
  have hint (ξ : Measure (SSphere d)) [IsFiniteMeasure ξ] : Integrable w ξ :=
    hw.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have : IsFiniteMeasure (μ.withDensity f) :=
    isFiniteMeasure_withDensity_ofReal (hint μ).hasFiniteIntegral
  have : IsFiniteMeasure (ν.withDensity f) :=
    isFiniteMeasure_withDensity_ofReal (hint ν).hasFiniteIntegral
  have hneq : μ.withDensity f ≠ ν.withDensity f := by
    intro h
    have hm := withDensity_inv_same hf (hμ.mono fun x hx => (ENNReal.ofReal_pos.mpr hx).ne') (htop μ)
    have hn := withDensity_inv_same hf (hν.mono fun x hx => (ENNReal.ofReal_pos.mpr hx).ne') (htop ν)
    apply hne
    rw [← hm, h, hn]
  obtain ⟨u, b, hub⟩ := exists_sphere_relu_ridge_distinguishes _ _ hneq
  have heq (ξ : Measure (SSphere d)) (hξ : ∀ᵐ x ∂ξ, 0 < w x) :
      (∫ x, max (b - inner ℝ (x : EucSpace d) u) 0 ∂ξ.withDensity f) =
        ∫ x, w x * max (b - inner ℝ (x : EucSpace d) u) 0 ∂ξ := by
    rw [integral_withDensity_eq_integral_toReal_smul hf
      (ae_of_all _ fun _ => ENNReal.ofReal_lt_top)]
    apply integral_congr_ae
    filter_upwards [hξ] with x hx
    rw [show (f x).toReal = w x from ENNReal.toReal_ofReal hx.le, smul_eq_mul]
  exact ⟨u, b, by simpa only [heq μ hμ, heq ν hν] using hub⟩

/-- A finite Euclidean Dirac measure satisfies every ridge-integrability
and equality premise of the ambient determination theorem. -/
example :
    (∀ (u : EucSpace 1) (b : ℝ), Integrable (fun x => max (b - inner ℝ x u) 0)
      (Measure.dirac (0 : EucSpace 1))) ∧
      (∀ (u : EucSpace 1) (b : ℝ),
        (∫ x, max (b - inner ℝ x u) 0 ∂Measure.dirac (0 : EucSpace 1)) =
          ∫ x, max (b - inner ℝ x u) 0 ∂Measure.dirac (0 : EucSpace 1)) :=
  ⟨fun _ _ => integrable_dirac (by simp), fun _ _ => rfl⟩

/-- Two different sphere atoms and the positive continuous weight one
jointly realize the weighted-separation hypotheses. In particular the
sphere theorem permits atomic measures and needs no density assumption. -/
example : ∃ μ ν : Measure (SSphere 1),
    IsFiniteMeasure μ ∧ IsFiniteMeasure ν ∧ μ ≠ ν ∧
      Continuous (fun _ : SSphere 1 => (1 : ℝ)) ∧
      (∀ᵐ x ∂μ, (0 : ℝ) < (fun _ : SSphere 1 => (1 : ℝ)) x) ∧
      ∀ᵐ x ∂ν, (0 : ℝ) < (fun _ : SSphere 1 => (1 : ℝ)) x := by
  let p : SSphere 1 := basePoint 0
  let q : SSphere 1 := antipode 1 p
  refine ⟨Measure.dirac p, Measure.dirac q, inferInstance, inferInstance, ?_,
    continuous_const, ae_of_all _ fun _ => zero_lt_one, ae_of_all _ fun _ => zero_lt_one⟩
  intro h
  have hi := congrArg (fun ξ : Measure (SSphere 1) => ∫ x, (x : EucSpace 1) 0 ∂ξ) h
  norm_num [p, q, antipode, basePoint] at hi

/-- The sphere determination theorem's finite-measure and equal-ridge
hypotheses hold jointly for a genuine nonzero Dirac law. -/
example : IsFiniteMeasure (Measure.dirac (basePoint 0)) ∧
    ∀ (u : EucSpace 1) (b : ℝ),
      (∫ x : SSphere 1, max (b - inner ℝ (x : EucSpace 1) u) 0 ∂Measure.dirac (basePoint 0)) =
        ∫ x : SSphere 1, max (b - inner ℝ (x : EucSpace 1) u) 0 ∂Measure.dirac (basePoint 0) :=
  ⟨inferInstance, fun _ _ => rfl⟩

end Transformer.Interpolation
