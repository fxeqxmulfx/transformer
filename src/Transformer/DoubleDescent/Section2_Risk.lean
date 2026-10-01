import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Lebesgue.Countable

/-!
# Training procedures and risks

arXiv:1912.02292v1, Section 2, Definition 1. Samples retain multiplicity:
`Fin n → Z` models an iid draw from `D^n`, rather than a mathematical set.
Nonnegative extended losses cover classification error and unbounded losses.
For measurable losses the lower integral is the usual nonnegative expectation.
-/

namespace Transformer.DoubleDescent

open MeasureTheory
open scoped ENNReal

variable {Z M : Type*} [MeasurableSpace Z]

/-- Section 2: a training procedure takes a sample and returns a model.
Random training seeds can be included in the model of the experiment separately. -/
abbrev TrainingProcedure (Z M : Type*) := (n : ℕ) → (Fin n → Z) → M

/-- Section 2, Definition 1: mean training loss, with empty-sample loss zero. -/
noncomputable def empiricalRisk (loss : M → Z → ℝ≥0∞)
    {n : ℕ} (model : M) (sample : Fin n → Z) : ℝ≥0∞ :=
  (∑ i, loss model (sample i)) / (n : ℝ≥0∞)

/-- Section 2, Definition 1: expectation over the iid training sample `D^n`. -/
noncomputable def expectedTrainingRisk (D : Measure Z) (loss : M → Z → ℝ≥0∞)
    (train : TrainingProcedure Z M) (n : ℕ) : ℝ≥0∞ :=
  ∫⁻ sample, empiricalRisk loss (train n sample) sample ∂Measure.pi (fun _ : Fin n => D)

/-- Sections 2 and 7: population test risk, averaging both the iid training
sample and a fresh independent test example. -/
noncomputable def expectedTestRisk (D : Measure Z) (loss : M → Z → ℝ≥0∞)
    (train : TrainingProcedure Z M) (n : ℕ) : ℝ≥0∞ :=
  ∫⁻ sample, (∫⁻ z, loss (train n sample) z ∂D) ∂Measure.pi (fun _ : Fin n => D)

/-- Section 2, Definition 1: a pointwise reduction in sample training loss
reduces its expectation. No test-risk conclusion is implicit here. -/
theorem expectedTrainingRisk_le (D : Measure Z) (loss : M → Z → ℝ≥0∞)
    (train₁ train₂ : TrainingProcedure Z M)
    (h : ∀ n sample, empiricalRisk loss (train₁ n sample) sample ≤
      empiricalRisk loss (train₂ n sample) sample) (n : ℕ) :
    expectedTrainingRisk D loss train₁ n ≤ expectedTrainingRisk D loss train₂ n := by
  exact lintegral_mono (h n)

/-- Section 2: the training-loss comparison is satisfiable by equal procedures. -/
example (loss : M → Z → ℝ≥0∞) (train : TrainingProcedure Z M) :
    ∀ n sample, empiricalRisk loss (train n sample) sample ≤
      empiricalRisk loss (train n sample) sample := fun _ _ => le_rfl

omit [MeasurableSpace Z] in
/-- Section 2, Definition 1: exact training interpolation gives zero risk. -/
theorem empiricalRisk_zero (loss : M → Z → ℝ≥0∞) {n : ℕ}
    (model : M) (sample : Fin n → Z) (h : ∀ i, loss model (sample i) = 0) :
    empiricalRisk loss model sample = 0 := by
  simp [empiricalRisk, h]

/-- Section 2: exact interpolation is satisfiable, for example for the zero
model and squared loss on a zero-valued sample. -/
example {n : ℕ} : ∀ i : Fin n,
    ENNReal.ofReal (((0 : ℝ) - (fun _ : Fin n => (0 : ℝ)) i) ^ 2) = 0 := by
  intro i
  simp

/-- Section 2, Definition 1: an everywhere perfectly fitted procedure has
zero expected training risk for every sample size. -/
theorem expectedTrainingRisk_zero (D : Measure Z) (loss : M → Z → ℝ≥0∞)
    (train : TrainingProcedure Z M)
    (h : ∀ n sample i, loss (train n sample) (sample i) = 0) (n : ℕ) :
    expectedTrainingRisk D loss train n = 0 := by
  have hs : ∀ sample, empiricalRisk loss (train n sample) sample = 0 :=
    fun sample => empiricalRisk_zero loss (train n sample) sample (h n sample)
  simp [expectedTrainingRisk, hs]

/-- Section 2: the perfect-fitting premise is realized by singleton-label
classification, regardless of the inputs or the training procedure. -/
example (train : TrainingProcedure Z (Z → Unit)) :
    ∀ n sample i, (if train n sample (sample i) = () then (0 : ℝ≥0∞) else 1) = 0 := by
  intro n sample i
  simp

end Transformer.DoubleDescent
