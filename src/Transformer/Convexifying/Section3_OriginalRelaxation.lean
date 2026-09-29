/-
# The original and simplex-relaxed training objectives differ

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equations (3)–(4)
(`eq:attention_only_obj` and `eq:simplex_regression`).  The paper says the
original regularized self-attention problem can be reformulated using one
simplex attention matrix.  This file gives a one-sample counterexample to
equality of the two optimal values.  It retains the query and key penalties
in the original objective and omits them in the relaxed objective, exactly
as the two printed equations do.
-/

import Transformer.Convexifying.Section3_ConvexAttention
import Transformer.Convexifying.Section3_ScalarCounterexample

open scoped BigOperators

namespace Transformer.Convexifying

/-- The one-sample token matrix `X = [1, -1]ᵀ`, with two tokens and one
feature.  Source: arXiv:2211.11052v1, equations (3)–(4), specialized. -/
def oppositeTokens : Fin 2 → Vec 1 :=
  fun r _ => if r = 0 then 1 else -1

/-- Scalar query-key scores `X Q Kᵀ Xᵀ` for the specialized input.
Source: arXiv:2211.11052v1, equation (3). -/
def originalScores (q k : ℝ) : Mat 2 2 :=
  fun r t => (oppositeTokens r 0 * q) * (oppositeTokens t 0 * k)

/-- One output row of the original attention-only network, where `q`, `k`,
`v`, and `o` are the one-dimensional query, key, value, and output weights.
Source: arXiv:2211.11052v1, equations (2)–(3). -/
noncomputable def originalOutput (q k v o : ℝ) (r : Fin 2) : ℝ :=
  (∑ t, rowSoftmax (originalScores q k) r t * (oppositeTokens t 0 * v)) * o

/-- Squared Frobenius loss against the two-row target `[1, 1]ᵀ`.
Source: arXiv:2211.11052v1, equation (3), using an allowed convex loss. -/
def twoRowSquareLoss (u : Vec 2) : ℝ :=
  ∑ r, (u r - 1) ^ 2

/-- The chosen two-row loss is convex, as required by the paper's loss
hypothesis.  Source: arXiv:2211.11052v1, equation (3). -/
theorem twoRowSquareLoss_convex (u v : Vec 2) (t : ℝ)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    twoRowSquareLoss (fun r => (1 - t) * u r + t * v r) ≤
      (1 - t) * twoRowSquareLoss u + t * twoRowSquareLoss v := by
  unfold twoRowSquareLoss
  calc
    (∑ r, ((1 - t) * u r + t * v r - 1) ^ 2) ≤
        ∑ r, ((1 - t) * (u r - 1) ^ 2 + t * (v r - 1) ^ 2) := by
          apply Finset.sum_le_sum
          intro r _
          simpa [squareLoss] using squareLoss_convex (u r) (v r) 1 t ht0 ht1
    _ = (1 - t) * (∑ r, (u r - 1) ^ 2) +
        t * (∑ r, (v r - 1) ^ 2) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- The convexity interval is nonempty. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- Equation (3), specialized to `N = 1`, `n = 2`, `d = c = 1`, squared
loss, and `β = 1`.  All four weight-decay terms are present. -/
noncomputable def originalObjective (q k v o : ℝ) : ℝ :=
  twoRowSquareLoss (originalOutput q k v o) +
    (q ^ 2 + k ^ 2 + v ^ 2 + o ^ 2) / 2

/-- Equation (4)'s shared-simplex prediction for the same data.
Source: arXiv:2211.11052v1, §3.1, `eq:simplex_regression`. -/
def relaxedOutput (W : Mat 2 2) (v o : ℝ) (r : Fin 2) : ℝ :=
  (∑ t, W r t * (oppositeTokens t 0 * v)) * o

/-- Equation (4), specialized to the same data, loss, and `β = 1`.
Its regularizer contains only value and output weights. -/
noncomputable def relaxedObjective (W : Mat 2 2) (v o : ℝ) : ℝ :=
  twoRowSquareLoss (relaxedOutput W v o) + (v ^ 2 + o ^ 2) / 2

/-- The valid one-sample relation is a relaxation: taking `W` to be the
softmax matrix preserves predictions, and dropping the nonnegative query/key
penalty cannot increase the objective.  This does not give one shared `W`
across multiple samples.  Source: arXiv:2211.11052v1, §3.1, (3)–(4). -/
theorem one_sample_relaxation_bound (q k v o : ℝ) :
    IsRowStochastic (rowSoftmax (originalScores q k)) ∧
      relaxedObjective (rowSoftmax (originalScores q k)) v o ≤
        originalObjective q k v o := by
  constructor
  · exact rowSoftmax_stochastic (by norm_num) _
  · have hpred : relaxedOutput (rowSoftmax (originalScores q k)) v o =
        originalOutput q k v o := rfl
    simp only [relaxedObjective, originalObjective, hpred]
    nlinarith [sq_nonneg q, sq_nonneg k]

/-- A shared attention matrix whose two rows both select the positive token.
Source: arXiv:2211.11052v1, equation (4)'s simplex domain. -/
def witnessW : Mat 2 2 := fun _ t => if t = 0 then 1 else 0

/-- The witness satisfies the simplex constraint of equation (4). -/
theorem witnessW_stochastic : IsRowStochastic witnessW := by
  intro r
  change IsSimplex (fun t : Fin 2 => if t = 0 then 1 else 0)
  exact simplex_basis 0

/-- The relaxed objective has value `1` at this feasible point; both target
rows are fitted exactly.  Source: arXiv:2211.11052v1, equation (4). -/
theorem witness_value : relaxedObjective witnessW 1 1 = 1 := by
  norm_num [relaxedObjective, twoRowSquareLoss, relaxedOutput, witnessW,
    oppositeTokens, Fin.sum_univ_two]

/-- A strictly positive shared attention matrix, so the counterexample is
not caused solely by the simplex boundary.  Both rows use weights `3/4`
and `1/4`.  Source: arXiv:2211.11052v1, equation (4). -/
noncomputable def positiveW : Mat 2 2 :=
  fun _ t => if t = 0 then 3 / 4 else 1 / 4

/-- The positive witness also satisfies equation (4)'s simplex constraint. -/
theorem positiveW_stochastic : IsRowStochastic positiveW := by
  intro r
  constructor
  · intro t
    fin_cases t <;> norm_num [positiveW]
  · norm_num [positiveW, Fin.sum_univ_two]

/-- Every entry of the second witness is strictly positive. -/
theorem positiveW_strict : ∀ r t, 0 < positiveW r t := by
  intro r t
  fin_cases t <;> norm_num [positiveW]

/-- The positive witness reaches value `7/4`, still below the original
optimum `2`.  Source: arXiv:2211.11052v1, equation (4). -/
theorem positiveW_value : relaxedObjective positiveW (3 / 2) 1 = 7 / 4 := by
  norm_num [relaxedObjective, twoRowSquareLoss, relaxedOutput, positiveW,
    oppositeTokens, Fin.sum_univ_two]

/-- The original self-attention outputs are opposite on the two rows for
every choice of query, key, value, and output weights.
Source: arXiv:2211.11052v1, equations (2)–(3). -/
theorem original_antisym (q k v o : ℝ) :
    originalOutput q k v o 1 = -originalOutput q k v o 0 := by
  simp [originalOutput, rowSoftmax, originalScores, oppositeTokens,
    Fin.sum_univ_two, mul_neg, neg_mul]
  ring

/-- The original objective is at least `2`: opposite output rows cannot
fit the constant target, and the four weight-decay terms are nonnegative.
Source: arXiv:2211.11052v1, equation (3). -/
theorem original_lower_bound (q k v o : ℝ) :
    2 ≤ originalObjective q k v o := by
  have hsum : originalOutput q k v o 1 = -originalOutput q k v o 0 :=
    original_antisym q k v o
  have hsq : 0 ≤ (originalOutput q k v o 0) ^ 2 := sq_nonneg _
  have hq : 0 ≤ q ^ 2 := sq_nonneg _
  have hk : 0 ≤ k ^ 2 := sq_nonneg _
  have hv : 0 ≤ v ^ 2 := sq_nonneg _
  have ho : 0 ≤ o ^ 2 := sq_nonneg _
  simp only [originalObjective, twoRowSquareLoss, Fin.sum_univ_two]
  rw [hsum]
  nlinarith

/-- The lower bound is attained by setting all original weights to zero.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
theorem original_zero_value : originalObjective 0 0 0 0 = 2 := by
  norm_num [originalObjective, twoRowSquareLoss, originalOutput,
    Fin.sum_univ_two]

/-- **Counterexample to the claimed reformulation from equation (3) to
equation (4).**  With one sample, two tokens, one feature, one output,
convex squared loss, and positive regularization `β = 1`, the original
optimum is `2`, while the simplex model has a feasible point of value `1`.
Thus their objective sublevel sets, and hence their optimal values, differ.
The paper's preceding single-score softmax/simplex inclusion remains true;
it does not establish equivalence of these regularized training problems.

Source: arXiv:2211.11052v1, §3.1, paragraph between
`eq:attention_only_obj` and `eq:simplex_regression`. -/
theorem attention_only_reformulation_false :
    ¬ ∀ r : ℝ,
      (∃ q k v o : ℝ, originalObjective q k v o ≤ r) ↔
      (∃ W : Mat 2 2, IsRowStochastic W ∧
        ∃ v o : ℝ, relaxedObjective W v o ≤ r) := by
  intro h
  obtain ⟨q, k, v, o, hq⟩ := (h 1).2
    ⟨witnessW, witnessW_stochastic, 1, 1, witness_value.le⟩
  have hbound := original_lower_bound q k v o
  linarith

/-- The same inequivalence remains when the relaxed matrix is required to
have strictly positive entries.  Thus excluding boundary points of the
simplex does not repair the reformulation from equation (3) to (4).
Source: arXiv:2211.11052v1, §3.1. -/
theorem positive_attention_reformulation_false :
    ¬ ∀ r : ℝ,
      (∃ q k v o : ℝ, originalObjective q k v o ≤ r) ↔
      (∃ W : Mat 2 2, IsRowStochastic W ∧
        (∀ s t, 0 < W s t) ∧
        ∃ v o : ℝ, relaxedObjective W v o ≤ r) := by
  intro h
  obtain ⟨q, k, v, o, hq⟩ := (h (7 / 4)).2
    ⟨positiveW, positiveW_stochastic, positiveW_strict,
      3 / 2, 1, positiveW_value.le⟩
  have hbound := original_lower_bound q k v o
  linarith

end Transformer.Convexifying
