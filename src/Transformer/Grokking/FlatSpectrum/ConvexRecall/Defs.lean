module

public import Transformer.Grokking.FlatSpectrum.Recall.Defs
public import Transformer.Landscape.Neural.Loss

/-!
# A convex matching model with ordinary answer cross-entropy

Source: `review/flat_spectrum.md`, *Convex answer-loss recall*, CR1. This new
construction reads an input dictionary, as in AR1. The learned parameters are
one stochastic matrix on keys. Values are copied through a fixed categorical
readout. There are no learned embeddings, factorization, route labels or
nuclear-norm penalty. The simplex constraint and supplied key/value binding
are explicit architectural assumptions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexRecall

open scoped BigOperators

variable {κ ν S : Type*} [Fintype κ] [Fintype ν] [Fintype S]

/-- Nonnegative matching weights with one unit of mass in each query row
(`review/flat_spectrum.md`, CR1). -/
def IsStochastic (A : Matrix κ κ ℝ) : Prop :=
  (∀ q k, 0 ≤ A q k) ∧ ∀ q, ∑ k, A q k = 1

variable [DecidableEq ν]

/-- The logit of a value is the mass of keys carrying that value. The
dictionary is input data; the query's answer is not an argument (CR1). -/
def logits (A : Matrix κ κ ℝ) (q : κ) (D : κ → ν) (c : ν) : ℝ :=
  ∑ k, if D k = c then A q k else 0

/-- Ordinary softmax cross-entropy on the supplied dictionaries and their
queried values, summed over samples. No auxiliary matching loss (CR1). -/
noncomputable def answerLoss (query : S → κ) (data : S → κ → ν)
    (A : Matrix κ κ ℝ) : ℝ :=
  ∑ s, Landscape.Neural.crossEntropy (logits A (query s) (data s)) (data s (query s))

/-- Each wrong key is distinguished from a query by at least one training
dictionary asked at that query. This is a condition on input data, not on
learned parameters or future predictions (`review/flat_spectrum.md`, CR3). -/
def Separates (query : S → κ) (data : S → κ → ν) : Prop :=
  ∀ q k, k ≠ q → ∃ s, query s = q ∧ data s k ≠ data s q

end Transformer.Grokking.FlatSpectrum.ConvexRecall
