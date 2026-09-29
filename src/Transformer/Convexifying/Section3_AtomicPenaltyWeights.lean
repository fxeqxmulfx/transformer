/-
# Four-factor weight decay obstructs exact convexification

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equations
`eq:attention_only` and `eq:attention_only_obj`.  On a two-token sample,
the attainable prediction-budget epigraph is nonconvex even when any finite
number of ordinary softmax heads with trainable Q, K, V, and O is allowed.
-/

import Transformer.Convexifying.Section3_AtomicMixture

namespace Transformer.Convexifying

/-- Four trainable scalar matrices in one head of equation
`eq:attention_only_obj`. -/
structure ScalarFourWeights where
  q : ℝ
  k : ℝ
  v : ℝ
  o : ℝ

/-- The paper's four-matrix weight decay for a scalar head, at `β = 1`.
Source: arXiv:2211.11052v1, equation `eq:attention_only_obj`. -/
noncomputable def scalarFourWeightCost (h : ScalarFourWeights) : ℝ :=
  (h.q ^ 2 + h.k ^ 2 + h.v ^ 2 + h.o ^ 2) / 2

/-- AM–GM shows that four-matrix weight decay is at least the cost of the
two product parameters.  Source: equation `eq:attention_only_obj`. -/
theorem scalarFourWeightCost_ge_reduced (h : ScalarFourWeights) :
    reducedHeadCost (h.q * h.k) (h.v * h.o) ≤ scalarFourWeightCost h := by
  have hq : 2 * (|h.q| * |h.k|) ≤ h.q ^ 2 + h.k ^ 2 := by
    nlinarith [sq_nonneg (|h.q| - |h.k|), sq_abs h.q, sq_abs h.k]
  have hv : 2 * (|h.v| * |h.o|) ≤ h.v ^ 2 + h.o ^ 2 := by
    nlinarith [sq_nonneg (|h.v| - |h.o|), sq_abs h.v, sq_abs h.o]
  simp only [reducedHeadCost, scalarFourWeightCost, abs_mul]
  linarith

/-- The difference between the two output rows of the paper's genuine
softmax head equals the reduced nonuniform component.  The input is the
two-token matrix `[1, 0]` from `toyData 0`.
Source: equation `eq:attention_only`, specialized. -/
theorem toyOutput_difference (h : ScalarFourWeights) :
    toyOutput 0 0 h.q h.k h.v h.o - toyOutput 0 1 h.q h.k h.v h.o =
      reducedHeadDifference (h.q * h.k) (h.v * h.o) := by
  simp [toyOutput, toyScores, toyData, rowSoftmax,
    reducedHeadDifference, centeredSoftmax, Fin.sum_univ_two, Real.exp_zero]
  ring

/-- Sum of nonuniform outputs for finitely many original heads.
Source: equation `eq:attention_only`, extended to finite width. -/
noncomputable def originalMixtureDifference (heads : List ScalarFourWeights) : ℝ :=
  (heads.map fun h => toyOutput 0 0 h.q h.k h.v h.o -
    toyOutput 0 1 h.q h.k h.v h.o).sum

/-- The full two-row output of a finite collection of genuine softmax
heads on the input `[1, 0]`.  Source: equation `eq:attention_only`, finite
width extension. -/
noncomputable def originalMixtureRows (heads : List ScalarFourWeights) :
    Fin 2 → ℝ :=
  fun r => (heads.map fun h => toyOutput 0 r h.q h.k h.v h.o).sum

/-- The projected output used below is exactly the difference of the two
rows of the full genuine-softmax prediction.  Source: equation
`eq:attention_only`, specialized. -/
theorem originalMixtureDifference_eq_rows (heads : List ScalarFourWeights) :
    originalMixtureDifference heads =
      originalMixtureRows heads 0 - originalMixtureRows heads 1 := by
  induction heads with
  | nil => simp [originalMixtureDifference, originalMixtureRows]
  | cons h hs ih =>
      simp only [originalMixtureDifference, originalMixtureRows,
        List.map_cons, List.sum_cons] at ih ⊢
      linarith

/-- Sum of the paper's four-matrix weight decay for finitely many heads.
Source: equation `eq:attention_only_obj`, extended to finite width. -/
noncomputable def originalMixturePenalty (heads : List ScalarFourWeights) : ℝ :=
  (heads.map scalarFourWeightCost).sum

/-- Product reduction of genuine, trainable-score heads preserves their
nonuniform output.  Source: equation `eq:attention_only`, specialized. -/
theorem originalMixtureDifference_eq_reduced (heads : List ScalarFourWeights) :
    originalMixtureDifference heads =
      reducedMixtureDifference (heads.map fun h => (h.q * h.k, h.v * h.o)) := by
  induction heads with
  | nil => simp [originalMixtureDifference, reducedMixtureDifference]
  | cons h hs ih =>
      simp only [originalMixtureDifference, reducedMixtureDifference,
        List.map_cons, List.sum_cons] at ih ⊢
      rw [toyOutput_difference, ih]

/-- The nonuniform-output hypothesis used below is realized by an actual
head with all four matrices trainable.  Source: equation
`eq:attention_only`, specialized. -/
example : ∃ heads : List ScalarFourWeights,
    originalMixtureDifference heads = 1 / 600 := by
  refine ⟨[⟨Real.log 2, 1, 1 / 100, 1⟩], ?_⟩
  simp [originalMixtureDifference, toyOutput_difference,
    reducedHeadDifference, centeredSoftmax,
    Real.exp_log (by norm_num : (0 : ℝ) < 2)]
  norm_num

/-- The actual four-matrix weight decay dominates its product reduction for
every finite collection of heads.  Source: `eq:attention_only_obj`. -/
theorem originalMixturePenalty_ge_reduced (heads : List ScalarFourWeights) :
    reducedMixtureCost (heads.map fun h => (h.q * h.k, h.v * h.o)) ≤
      originalMixturePenalty heads := by
  induction heads with
  | nil => simp [originalMixturePenalty, reducedMixtureCost]
  | cons h hs ih =>
      simp only [originalMixturePenalty, reducedMixtureCost,
        List.map_cons, List.sum_cons] at ih ⊢
      exact add_le_add (scalarFourWeightCost_ge_reduced h) ih

/-- The row-output difference is bounded by the square of the paper's
actual four-matrix weight decay, for any finite number of trainable-score
heads.  Source: arXiv:2211.11052v1, equations `eq:attention_only`–
`eq:attention_only_obj`, specialized. -/
theorem originalMixtureDifference_quadratic (heads : List ScalarFourWeights) :
    |originalMixtureDifference heads| ≤ originalMixturePenalty heads ^ 2 := by
  rw [originalMixtureDifference_eq_reduced]
  have hbound := reducedMixtureDifference_quadratic
    (heads.map fun h => (h.q * h.k, h.v * h.o))
  have hcost := originalMixturePenalty_ge_reduced heads
  have hred0 := reducedMixtureCost_nonneg
    (heads.map fun h => (h.q * h.k, h.v * h.o))
  have hR0 : 0 ≤ originalMixturePenalty heads := le_trans hred0 hcost
  have hprod : 0 ≤
      (originalMixturePenalty heads -
        reducedMixtureCost (heads.map fun h => (h.q * h.k, h.v * h.o))) *
      (originalMixturePenalty heads +
        reducedMixtureCost (heads.map fun h => (h.q * h.k, h.v * h.o))) :=
    mul_nonneg (sub_nonneg.mpr hcost) (add_nonneg hR0 hred0)
  nlinarith

/-- **Original weight-decay gap at arbitrary finite width.**  To produce
the nonuniform output `1/600` on `[1, 0]`, genuine softmax attention with
any finite number of heads and all four matrices trainable must pay more
than `1/50` in the paper's weight decay.  `smallAtomicDifference` and
`smallAtomicCost` give a genuine softmax atom with the same output projection
and atomic cost at most `1/50`.  Any exact match of the full output would
also match this projection, so the positive-homogeneous atomic mixture
cannot preserve equation `eq:attention_only_obj`'s regularizer. -/
theorem originalMixturePenalty_gap (heads : List ScalarFourWeights)
    (houtput : originalMixtureDifference heads = 1 / 600) :
    1 / 50 < originalMixturePenalty heads := by
  have hred : reducedMixtureDifference
      (heads.map fun h => (h.q * h.k, h.v * h.o)) = 1 / 600 := by
    rw [← originalMixtureDifference_eq_reduced]
    exact houtput
  have hgap := no_small_original_mixture _ hred
  have hcost := originalMixturePenalty_ge_reduced heads
  linarith

end Transformer.Convexifying
