/-
# The optimizer's Markov expectation operator

arXiv:2506.12543v1, Section 4.3, Theorem 1.
The contraction used in the weak-error telescope is proved for the actual
Gaussian SGD/SignSGD update, not postulated as a stability assumption.
-/

import Transformer.BatchSize.Section4_DiscreteLaw
import Transformer.BatchSize.Section4_Telescoping
import Mathlib.MeasureTheory.Integral.Prod

open MeasureTheory

noncomputable section

namespace Transformer.BatchSize

/-- Pull an observable back along a measurable sampler,
Section 4.3's optimizer expectations. -/
def BoundedObservable.comp {E Z : Type*} [MeasurableSpace E] [MeasurableSpace Z]
    (φ : BoundedObservable E) (g : Z → E) (hg : Measurable g) : BoundedObservable Z :=
  ⟨fun z => φ.val (g z), φ.property.1.comp hg,
    by obtain ⟨C, hC⟩ := φ.property.2; exact ⟨C, fun z => hC (g z)⟩⟩

/-- The actual one-step expectation under independent standard Gaussian
innovations, Section 4.3, equations (2)--(3). It preserves boundedness and
measurability for both optimizers, including the discontinuous sign update. -/
def discreteExpectationOperator {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ) :
    BoundedObservable (EucSpace d) → BoundedObservable (EucSpace d) := fun φ =>
  ⟨fun x => ∫ z, φ.val (stochasticStep method η B f σ x z)
      ∂diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1),
    ((φ.property.1.comp (stochasticStep_measurable method η B f σ hf hσ)).stronglyMeasurable.integral_prod_right').measurable,
    by
      obtain ⟨C, hC⟩ := φ.property.2
      refine ⟨C, fun x => ?_⟩
      have h := norm_integral_le_of_norm_le_const
        (f := fun z => φ.val (stochasticStep method η B f σ x z))
        (μ := diagonalNoiseLaw 1 (fun _ => 0) (fun _ => 1))
        (ae_of_all _ (fun z => by
          simpa [Real.norm_eq_abs] using hC (stochasticStep method η B f σ x z)))
      simpa [Real.norm_eq_abs] using h⟩

/-- The Gaussian update's expectation operator is a uniform contraction,
Section 4.3, Theorem 1. This supplies the actual discrete operator needed
by the weak-error telescope. -/
theorem discreteExpectationOperator_contraction {d : ℕ} (method : UpdateKind)
    (η : ℝ) (B : ℕ) (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hf : ContDiff ℝ 1 f) (hσ : Continuous σ)
    (φ ψ : BoundedObservable (EucSpace d)) (r : ℝ)
    (hclose : ∀ x, |φ.val x - ψ.val x| ≤ r) (x : EucSpace d) :
    |(discreteExpectationOperator method η B f σ hf hσ φ).val x -
      (discreteExpectationOperator method η B f σ hf hσ ψ).val x| ≤ r := by
  have hs : Measurable (stochasticStep method η B f σ x) := by
    exact (stochasticStep_measurable method η B f σ hf hσ).comp
      (measurable_const.prodMk measurable_id)
  exact probability_expectation_contraction
    (diagonalNoiseLaw 1 (fun _ : Fin d => 0) (fun _ => 1))
    (φ.comp (stochasticStep method η B f σ x) hs)
    (ψ.comp (stochasticStep method η B f σ x) hs) r
    (fun z => hclose (stochasticStep method η B f σ x z))

/-- Joint nonvacuity of the contraction hypotheses, Section 4.3:
a flat smooth loss, unit noise and equal bounded observables. -/
example : ∃ (f : EucSpace 1 → ℝ) (σ : EucSpace 1 → EucSpace 1)
    (φ ψ : BoundedObservable (EucSpace 1)), ContDiff ℝ 1 f ∧ Continuous σ ∧
      ∀ x, |φ.val x - ψ.val x| ≤ (0 : ℝ) := by
  let φ : BoundedObservable (EucSpace 1) := ⟨fun _ => 0, measurable_const, 0, by simp⟩
  exact ⟨fun _ => 0, fun _ => EuclideanSpace.single (0 : Fin 1) 1,
    φ, φ, contDiff_const, continuous_const, by simp⟩

end Transformer.BatchSize
