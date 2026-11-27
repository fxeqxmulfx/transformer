module

public import Transformer.Grokking.FiniteRecovery.ArgandHiddenWords
public import Mathlib.Probability.Distributions.Gaussian.Real
public import Mathlib.MeasureTheory.Constructions.Pi

/-! Independent complex Gaussian tensor draws for the real-arithmetic
initialization. Variance is specified per real component of each entry.
Fixed scores, gates and other deterministic initial values are applied
by `sgdInitialWeights` after the draw. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open MeasureTheory ProbabilityTheory
open scoped NNReal

/-- One complex entry with independent centered real and imaginary parts. -/
noncomputable def complexGaussianDraw (v : ℝ≥0) : Measure ℂ :=
  ((gaussianReal 0 v).prod (gaussianReal 0 v)).map Complex.equivRealProdCLM.symm

instance (v : ℝ≥0) : IsProbabilityMeasure (complexGaussianDraw v) :=
  (Measure.isProbabilityMeasure_map_iff Complex.equivRealProdCLM.symm.continuous.aemeasurable).mpr
    inferInstance

theorem complexGaussianDraw_openPos {v : ℝ≥0} (hv : 0 < v) :
    (complexGaussianDraw v).IsOpenPosMeasure := by
  let : (gaussianReal 0 v).IsOpenPosMeasure :=
    (gaussianReal_absolutelyContinuous' 0 hv.ne').isOpenPosMeasure
  exact Complex.equivRealProdCLM.symm.continuous.isOpenPosMeasure_map
    Complex.equivRealProdCLM.symm.surjective

variable {ι : Type*} [Fintype ι]

/-- The product law on all raw entries of the parameter tensors. -/
noncomputable def independentGaussianDraws (variance : ι → ℝ≥0) : Measure (ι → ℂ) :=
  Measure.pi (fun i => complexGaussianDraw (variance i))

instance (variance : ι → ℝ≥0) : IsProbabilityMeasure (independentGaussianDraws variance) :=
  inferInstanceAs (IsProbabilityMeasure (Measure.pi _))

theorem independentGaussianDraws_openPos (variance : ι → ℝ≥0) (hv : ∀ i, 0 < variance i) :
    (independentGaussianDraws variance).IsOpenPosMeasure := by
  let (i : ι) : (complexGaussianDraw (variance i)).IsOpenPosMeasure := complexGaussianDraw_openPos (hv i)
  exact inferInstanceAs (Measure.IsOpenPosMeasure (Measure.pi _))

/-- The distributional hypothesis in the hidden-word theorem holds
for any positive tensor variances, including fan-in dependent scales. -/
theorem gaussian_mixedWords_momentum_recovers {layers : ℕ}
    (variance : ComplexParameter Bool 1 1 layers → ℝ≥0) (hv : ∀ i, 0 < variance i)
    (n repeats : ℕ) (mode : Option Bool) :
    ∃ lam₀ > 0, ∀ lam, 0 < lam → lam ≤ lam₀ →
      ∃ U : Set (ComplexParameter Bool 1 1 layers → ℂ), IsOpen U ∧
        0 < independentGaussianDraws variance U ∧
        ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ U,
          ∃ star : Weights Bool 1 1 layers,
            Filter.Tendsto (fun t => (fullGradientMomentumRun true repeats (mixedPositionBatch n mode)
              lam (fun _ => η) draw t).weights) Filter.atTop (nhds star) ∧
            (Model.fromWeights star).realizes repeats (lastTokenTask (T := n + 2) mode) := by
  let := independentGaussianDraws_openPos variance hv
  exact mixedWords_momentum_recovers _ n repeats mode

example := gaussian_mixedWords_momentum_recovers (layers := 2) (fun _ => 1) (by intro; norm_num)
  29 2 none

end Transformer.Grokking.FiniteRecovery.Argand
