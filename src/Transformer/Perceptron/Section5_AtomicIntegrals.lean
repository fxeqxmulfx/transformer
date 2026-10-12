/-
# Actual singleton weights and finite-support integration

ArXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`, and §5.5,
`rem:strictSOPD-perceptron`. Under the source's strict angular
curvature condition, the preceding support reduction gives a finite
topological support. Its actual singleton masses give the weights
in the finite-sum version of the source's second variation.

This file evaluates actual integrals and actual pushforwards.
No continuity of the integrand is necessary on finite support.
The weights are nonnegative and sum to one because the input is
the original probability measure, rather than a new atomic model.
The energy formula keeps both interaction integrals and the
one-half prefactor of the perceptron potential.
-/

import Transformer.Perceptron.Section5_AngularSupport
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped BigOperators
open MeasureTheory

namespace Transformer.Perceptron

variable {d : ℕ}

/-- **The finite-support presentation uses the actual singleton masses.**
Source: arXiv:2601.21366v2, §5.2 and §5.5, the measures in
`eq:Hess-bilinear.std` and `rem:strictSOPD-perceptron`.
Second countability makes the topological support carry the measure. -/
theorem measure_eq_sum_singleton_dirac (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite) :
    (μ : Measure (SSphere d)) =
      ∑ z ∈ hfinite.toFinset, (μ : Measure (SSphere d)) {z} • Measure.dirac z := by
  classical
  apply Measure.ae_mem_finset_iff.mp
  filter_upwards [Measure.support_mem_ae (μ := (μ : Measure (SSphere d)))] with z hz
  exact hfinite.mem_toFinset.mpr hz

/-- **Every actual finite-support weight is finite.**
Source: arXiv:2601.21366v2, §5.2, the probability masses in
`eq:Hess-bilinear.std`. This also applies outside support, where
the singleton mass may be zero. -/
theorem singleton_mass_ne_top (μ : Perspective.ProbSphere d) (z : SSphere d) :
    (μ : Measure (SSphere d)) {z} ≠ ⊤ := measure_ne_top _ _

/-- **Every actual real singleton weight is nonnegative.**
Source: arXiv:2601.21366v2, §5.2 and §5.5, the weights in
the finite-sum version of `eq:Hess-bilinear.std`. -/
theorem singleton_mass_toReal_nonneg (μ : Perspective.ProbSphere d) (z : SSphere d) :
    0 ≤ ((μ : Measure (SSphere d)) {z}).toReal := ENNReal.toReal_nonneg

/-- **Every finite-valued function is integrable on finite support.**
Source: arXiv:2601.21366v2, §5.2, the integrals in
`eq:Hess-bilinear.std`. No measurability or continuity assumption
on the function is needed for the finite Dirac presentation. -/
theorem integrable_finite_support {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite) (f : SSphere d → E) :
    Integrable f (μ : Measure (SSphere d)) := by
  rw [measure_eq_sum_singleton_dirac μ hfinite, integrable_finsetSum_measure]
  intro z hz
  exact (integrable_dirac (f := f) (a := z) (by finiteness)).smul_measure
    (singleton_mass_ne_top μ z)

/-- **Integration uses the original measure's actual singleton weights.**
Source: arXiv:2601.21366v2, §5.2 and §5.5,
`eq:Hess-bilinear.std` and `rem:strictSOPD-perceptron`.
The formula applies to scalar functions and actual vector fields. -/
theorem integral_finite_support {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite) (f : SSphere d → E) :
    ∫ z, f z ∂(μ : Measure (SSphere d)) =
      ∑ z ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {z}).toReal • f z := by
  conv_lhs => rw [measure_eq_sum_singleton_dirac μ hfinite]
  rw [integral_finsetSum_measure (fun z hz =>
      (integrable_dirac (f := f) (a := z) (by finiteness)).smul_measure
        (singleton_mass_ne_top μ z))]
  simp only [integral_smul_measure, integral_dirac]

/-- **The actual real singleton weights sum to one.**
Source: arXiv:2601.21366v2, §5.5, the probability normalization
used in the displayed identity for the squared velocity difference. -/
theorem sum_singleton_mass_toReal (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite) :
    ∑ z ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {z}).toReal = 1 := by
  have h := integral_finite_support μ hfinite (fun _ => (1 : ℝ))
  simpa using h.symm

/-- **A Monge map preserves the finite presentation and its weights.**
Source: arXiv:2601.21366v2, §2.2 and §5.2, the pushforwards
used to define `eq:Hess-bilinear.std`. The image points may coincide. -/
theorem map_finite_support (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite)
    (T : SSphere d → SSphere d) (hT : Measurable T) :
    (μ : Measure (SSphere d)).map T =
      ∑ z ∈ hfinite.toFinset, (μ : Measure (SSphere d)) {z} • Measure.dirac (T z) := by
  conv_lhs => rw [measure_eq_sum_singleton_dirac μ hfinite]
  rw [Measure.map_finset_sum hT.aemeasurable]
  simp only [Measure.map_smul _ hT.aemeasurable, Measure.map_dirac]

/-- **An integral along the actual pushforward is the same weighted sum.**
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear.std`.
This formula requires no regularity of the function or of the image atoms. -/
theorem integral_finite_support_map {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [CompleteSpace E] (μ : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite)
    (T : SSphere d → SSphere d) (hT : Measurable T) (f : SSphere d → E) :
    ∫ z, f z ∂(μ : Measure (SSphere d)).map T =
      ∑ z ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {z}).toReal • f (T z) := by
  rw [map_finite_support μ hfinite T hT,
    integral_finsetSum_measure (fun z hz =>
      (integrable_dirac (f := f) (a := T z) (by finiteness)).smul_measure
        (singleton_mass_ne_top μ z))]
  simp only [integral_smul_measure, integral_dirac]

/-- All finite-support and map hypotheses above are jointly realized
on the actual nonempty circle by a Dirac probability and the identity. -/
example : (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support.Finite ∧
    Measurable (id : SSphere 2 → SSphere 2) := by
  refine ⟨?_, measurable_id⟩
  exact finite_support_of_finitely_atomic _ (isFinitelyAtomic_diracProb (basePoint 1))

/-- **The actual coupled energy of a finite-support pushforward.**
Source: arXiv:2601.21366v2, §2.3, the coupled energy, and
§5.2, the energy differentiated to obtain `eq:Hess-bilinear.std`.
Both integrals retain their original probability weights. -/
theorem energy_finite_support_map (β : ℝ) (φ : ℝ → ℝ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ ν : Perspective.ProbSphere d)
    (hfinite : (μ : Measure (SSphere d)).support.Finite)
    (T : SSphere d → SSphere d) (hT : Measurable T)
    (hν : (ν : Measure (SSphere d)) = (μ : Measure (SSphere d)).map T) :
    energy β φ ω a ν = (2 * β)⁻¹ *
      (∑ x ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {x}).toReal *
        ∑ y ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {y}).toReal *
          Real.exp (β * inner (𝕜 := ℝ) (T x : EucSpace d) (T y : EucSpace d))) +
      (1 / 2 : ℝ) * (∑ x ∈ hfinite.toFinset, ((μ : Measure (SSphere d)) {x}).toReal *
        potential φ ω a (T x : EucSpace d)) := by
  unfold energy Perspective.interactionEnergy
  rw [hν]
  simp only [integral_finite_support_map μ hfinite T hT, smul_eq_mul, one_div]

/-- The energy's additional pushforward hypothesis is realized by
the identity acting on the same actual finite-support Dirac probability. -/
example : let μ := Perspective.diracProb 2 (basePoint 1)
    (μ : Measure (SSphere 2)) = (μ : Measure (SSphere 2)).map id := by
  intro μ
  exact Measure.map_id.symm

end Transformer.Perceptron
