/-
# A shared simplex matrix is not a reparameterization of self-attention

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §3.1,
`sec:convex_attention`, equations (3)–(4).

The displayed fact `∀U, ∃W ∈ Δ, softmax(U)X = WX` is true for a single
score matrix.  The subsequent training objective uses one trainable `W`
shared by every sample, whereas the original score matrix depends on the
sample.  The two-sample example below proves that the shared-`W` model is
not a reparameterization of the self-attention operation.
-/

import Transformer.Convexifying.Section3_ConvexAttention
import Transformer.Convexifying.Section3_ScalarCounterexample

open scoped BigOperators

namespace Transformer.Convexifying

/-- Query-key scores with scalar query and key weights both equal to one. -/
def scalarScores (X : Fin 2 → Vec 1) : Mat 2 2 :=
  fun r k => X r 0 * X k 0

/-- The first output coordinate of self-attention with scalar value weight
one; the data determine the score matrix for each sample. -/
noncomputable def sampleSoftmaxOutput (X : Fin 2 → Vec 1) (r : Fin 2) : ℝ :=
  ∑ k, rowSoftmax (scalarScores X) r k * X k 0

/-- On the first sample, row zero attends to token zero with weight
`exp(1)/(exp(1)+1)`. -/
theorem first_sample_output :
    sampleSoftmaxOutput (separatingData 0) 0 =
      Real.exp 1 / (Real.exp 1 + 1) := by
  norm_num [sampleSoftmaxOutput, rowSoftmax, scalarScores,
    separatingData, Fin.sum_univ_two, Real.exp_zero]

/-- On the second sample, row zero attends uniformly, giving output `1/2`. -/
theorem second_sample_output :
    sampleSoftmaxOutput (separatingData 1) 0 = 1 / 2 := by
  norm_num [sampleSoftmaxOutput, rowSoftmax, scalarScores,
    separatingData, Fin.sum_univ_two, Real.exp_zero]

/-- A single row-stochastic matrix cannot reproduce self-attention on both
data samples, even with unit query, key, and value weights.  Thus the move
from equation (3) to a shared simplex weight in equation (4) changes the
model class; it is not an exact reformulation of the original network.
Source: arXiv:2211.11052v1, §3.1, paragraph before `eq:simplex_regression`. -/
theorem no_shared_attention_reparameterization :
    ¬ ∃ W : Mat 2 2, IsRowStochastic W ∧
      ∀ i : Fin 2, ∀ r : Fin 2,
        (∑ k, W r k * separatingData i k 0) =
          sampleSoftmaxOutput (separatingData i) r := by
  rintro ⟨W, hW, houtput⟩
  have h0 := houtput 0 0
  have h1 := houtput 1 0
  rw [first_sample_output] at h0
  rw [second_sample_output] at h1
  have hsum := (hW 0).2
  simp [separatingData] at h0 h1
  simp [Fin.sum_univ_two] at hsum
  have he : 1 < Real.exp 1 := by
    have h := Real.exp_lt_exp.mpr (show (0 : ℝ) < 1 by norm_num)
    rw [Real.exp_zero] at h
    exact h
  have hden : Real.exp 1 + 1 ≠ 0 := by positivity
  have hratio : Real.exp 1 / (Real.exp 1 + 1) = 1 / 2 := by linarith
  apply (div_eq_iff hden).mp at hratio
  nlinarith

end Transformer.Convexifying
