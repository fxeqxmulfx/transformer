/-
# AdaFisher: complete state transition for a layer block

arXiv:2405.16397v3, Algorithm 1 and equations `eq:expkronfactors`, `eq:FIMDiag`.
Fresh factor diagonals come from the layer formulas in Appendix A.3.
The transition explicitly carries raw momentum and unnormalized EMA factors.
-/

import Transformer.AdaFisher.Section3_Algorithm
import Mathlib.Logic.Equiv.Fin.Basic

noncomputable section

namespace Transformer.AdaFisher

variable {a b : ℕ}

/-- State of one layer block in Algorithm 1. A model with multiple layers
uses one block per layer and a shared positive batch counter. -/
structure OptimizerState (a b : ℕ) where
  time : ℕ
  factorH : Fin a → ℝ
  factorS : Fin b → ℝ
  rawMoment : Fin (a * b) → ℝ
  parameters : Fin (a * b) → ℝ

/-- Algorithm 1 initialization. The source specifies zero raw momentum
and an initial identity Fisher but omits the initial EMA factor buffers.
We explicitly initialize those buffers to zero. Its identity Fisher is
overwritten before the first parameter update and need not be stored. -/
def initializeOptimizer (parameters : Fin (a * b) → ℝ) : OptimizerState a b :=
  ⟨0, 0, 0, 0, parameters⟩

/-- Normalized/damped KF diagonal in column-first flat parameter order,
§2 and §3.2, `eq:FIMDiag`. Positive factor dimensions define their extrema. -/
def normalizedFisher [NeZero a] [NeZero b] (δ : ℝ)
    (h : Fin a → ℝ) (s : Fin b → ℝ) (i : Fin (a * b)) : ℝ :=
  fisherDiagonal δ (minMaxDiagonal h) (minMaxDiagonal s) (finProdFinEquiv.symm i)

/-- One complete AdaFisher/AdaFisherW block transition, Algorithm 1.
`γ` weights fresh KF diagonals, then min-max scaling and damping create
the Fisher. Raw momentum is updated from `g`, and bias corrected using
the new positive batch count before the parameter update. The undefined
`h`, zero denominator at t=0, and inconsistent moment indices in the
printed pseudocode are corrected explicitly. κ=0 selects AdaFisher. -/
def optimizerStep [NeZero a] [NeZero b] (η κ β γ δ : ℝ)
    (state : OptimizerState a b) (freshH : Fin a → ℝ) (freshS : Fin b → ℝ)
    (g : Fin (a * b) → ℝ) : OptimizerState a b :=
  let h := factorEMA γ state.factorH freshH
  let s := factorEMA γ state.factorS freshS
  let m := fun i => β * state.rawMoment i + (1 - β) * g i
  let corrected := fun i => m i / (1 - β ^ (state.time + 1))
  ⟨state.time + 1, h, s, m,
    parameterStep η κ (normalizedFisher δ h s) corrected state.parameters⟩

/-- The complete transition's Fisher satisfies the eigenvalue/entry bounds
of Proposition 3.2, independently of the fresh data and EMA history. -/
theorem normalizedFisher_bounds [NeZero a] [NeZero b] (δ : ℝ)
    (h : Fin a → ℝ) (s : Fin b → ℝ) (i : Fin (a * b)) :
    δ ≤ normalizedFisher δ h s i ∧ normalizedFisher δ h s i ≤ 1 + δ :=
  fisherDiagonal_bounds δ _ _ (minMaxDiagonal_bounds h) (minMaxDiagonal_bounds s) _

/-- With positive damping every actual state transition has a well-defined
diagonal solve, Proposition 3.2 and Algorithm 1. -/
theorem normalizedFisher_pos [NeZero a] [NeZero b] (δ : ℝ)
    (hδ : 0 < δ) (h : Fin a → ℝ) (s : Fin b → ℝ) (i : Fin (a * b)) :
    0 < normalizedFisher δ h s i :=
  lt_of_lt_of_le hδ (normalizedFisher_bounds δ h s i).1

example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- The complete first update uses the current gradient after bias
correction, Algorithm 1. This checks the correction inside the full state
transition, including EMA, min-max normalization, damping and weight decay. -/
theorem optimizerStep_first [NeZero a] [NeZero b] (η κ β γ δ : ℝ)
    (θ g : Fin (a * b) → ℝ) (freshH : Fin a → ℝ) (freshS : Fin b → ℝ)
    (hβ : β ≠ 1) :
    (optimizerStep η κ β γ δ (initializeOptimizer θ) freshH freshS g).parameters =
      parameterStep η κ
        (normalizedFisher δ (factorEMA γ 0 freshH) (factorEMA γ 0 freshS)) g θ := by
  have hden : 1 - β ≠ 0 := sub_ne_zero.mpr (Ne.symm hβ)
  funext i
  simp only [optimizerStep, initializeOptimizer, Pi.zero_apply, mul_zero, zero_add,
    pow_one, parameterStep]
  field_simp

example : (9 / 10 : ℝ) ≠ 1 := by norm_num

end Transformer.AdaFisher
