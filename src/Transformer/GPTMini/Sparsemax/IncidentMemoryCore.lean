import Transformer.GPTMini.Sparsemax.BipartiteMemoryGram
import Transformer.GPTMini.Sparsemax.BoundedGramWidth
import Transformer.GPTMini.Sparsemax.LocalMemoryScoreIdentities

/-!
# Genuine sparsemax memory with separate local budgets

Derived extension before arXiv:1602.02068v2, Eq. (1). The existing affine
path Gram is exactly the outer-product lift of its scores. Its separate
incident budgets give nonnegative doubly stochastic scores, hence PSD,
bounded entries, normalized sparsemax rows and the requested self-weight
floor. A floor above one half proves an actual attention inverse.

The old global identity coefficient need not be positive. Positivity is
proved from genuine outer products instead of assuming it or changing the
projection. All learned path coordinates and their affine variation are
preserved. Same-family orthogonality and possible path connections remain
fixed restrictions; this is not unrestricted transformer attention.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The affine path Gram is exactly the stochastic outer-product lift of its cross scores.
Source: the derived embedding chart before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_eq_bipartite {N : ℕ} (t : Fin N → ℝ) :
    localMemoryCore t = bipartiteMemoryGram (memoryGramScores (localMemoryCore t)) := by
  ext x y
  rcases x with i | i <;> rcases y with j | j
  · rw [(localMemoryCore_sameSide t i j).1, bipartiteMemoryGram_query,
      localMemoryCore_scores_rowSum]
  · rw [bipartiteMemoryGram_cross]
    rfl
  · rw [localMemoryCore_symmetric t (Sum.inr i) (Sum.inl j),
      bipartiteMemoryGram_cross_reverse]
    rfl
  · rw [(localMemoryCore_sameSide t i j).2, bipartiteMemoryGram_key,
      localMemoryCore_scores_columnSum]

/-- Separate budgets imply PSD without the former global mixture bound.
Source: explicit score-weighted outer products before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryCore_posSemidef {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    (localMemoryCore t).PosSemidef := by
  rw [localMemoryCore_eq_bipartite]
  exact bipartiteMemoryGram_posSemidef _ (incidentMemory_scores_nonneg floor t hf ht)

/-- A point outside the former domain genuinely inhabits the PSD premises. -/
example : (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))).PosSemidef :=
  incidentMemoryCore_posSemidef _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Separate incident budgets imply every full bounded-memory constraint.
Source: the genuine Gram and probability inputs before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryCore_mem {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    localMemoryCore t ∈ memoryGramDomain N 1 floor := by
  refine ⟨⟨incidentMemoryCore_posSemidef floor t hf ht, ?_⟩, ?_, ?_⟩
  · intro x y
    rcases x with i | i <;> rcases y with j | j
    · rw [(localMemoryCore_sameSide t i j).1]
      split_ifs <;> norm_num
    · change -1 ≤ memoryGramScores (localMemoryCore t) i j ∧
        memoryGramScores (localMemoryCore t) i j ≤ 1
      exact ⟨by linarith [incidentMemory_scores_nonneg floor t hf ht i j],
        incidentMemory_scores_le_one floor t hf ht i j⟩
    · rw [localMemoryCore_symmetric t (Sum.inr i) (Sum.inl j)]
      change -1 ≤ memoryGramScores (localMemoryCore t) j i ∧
        memoryGramScores (localMemoryCore t) j i ≤ 1
      exact ⟨by linarith [incidentMemory_scores_nonneg floor t hf ht j i],
        incidentMemory_scores_le_one floor t hf ht j i⟩
    · rw [(localMemoryCore_sameSide t i j).2]
      split_ifs <;> norm_num
  · intro i
    exact ⟨incidentMemory_scores_nonneg floor t hf ht i, localMemoryCore_scores_rowSum t i,
      fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · exact incidentMemory_scores_floor floor t ht

/-- Three simultaneous edges inhabit the bounded PSD probability domain. -/
example : localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ)) ∈ memoryGramDomain 3 1 (3 / 4) :=
  incidentMemoryCore_mem _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Actual variational sparsemax fixes the locally budgeted path scores.
Source: arXiv:1602.02068v2, Eq. (1), on the proved probability-score domain. -/
theorem incidentMemoryCore_normalized {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    memoryGramAttention (localMemoryCore t) = memoryGramScores (localMemoryCore t) :=
  memoryGramAttention_normalized 1 floor _ (incidentMemoryCore_mem floor t hf ht)

/-- The enlarged-domain witness uses the actual sparsemax projection. -/
example : memoryGramAttention (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) =
    memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) :=
  incidentMemoryCore_normalized _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Local budgets and a strict floor guarantee an actual attention inverse.
Source: diagonal dominance of the genuine arXiv:1602.02068v2, Eq. (1) memory. -/
theorem incidentMemoryCore_det_unit {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 1 / 2 < floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    IsUnit (memoryGramAttention (localMemoryCore t)).det :=
  memoryGramAttention_det_unit 1 floor _ hf (incidentMemoryCore_mem floor t (by linarith) ht)

/-- A real three-edge enlarged-domain point inhabits nonsingularity. -/
example : IsUnit (memoryGramAttention (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ)))).det :=
  incidentMemoryCore_det_unit _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Locally constrained scores admit exact genuine embeddings of width at most twice slot count.
Source: the proved PSD lift and spectral recovery preceding arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryCore_fixedWidth {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      localMemoryCore t = featureGram features :=
  memoryGram_fixedWidth 1 floor _ (incidentMemoryCore_mem floor t hf ht)

/-- A three-edge point has a genuine width-eight realization. -/
example : ∃ features : Fin 8 → Sum (Fin 4) (Fin 4) → ℝ,
    localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ)) = featureGram features :=
  incidentMemoryCore_fixedWidth _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- One shared original value table exactly decodes arbitrary learned output coordinates.
Source: the inverse chart after arXiv:1602.02068v2, Eq. (1), applied to separate budgets. -/
theorem incidentMemoryCoreValues_exact {N D : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : t ∈ incidentMemoryWeightDomain N floor) :
    memoryValueOutput (localMemoryCore t) (recoverMemoryValues (localMemoryCore t) Z) = Z :=
  recoverMemoryValues_exact 1 floor _ Z hf (incidentMemoryCore_mem floor t (by linarith) ht)

/-- Nonconstant outputs and formerly excluded edges inhabit the common-value decoder premises. -/
example : memoryValueOutput (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ)))
    (recoverMemoryValues (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ)))
      (Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ)))) =
    Matrix.of (fun j : Fin 4 => fun _ : Fin 1 => (j.val : ℝ)) :=
  incidentMemoryCoreValues_exact _ _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Actual attention stays affine as all locally constrained edge parameters change.
Source: the linear structural domain before arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemoryCoreAttention_affine {N : ℕ} (floor : ℝ) (t s : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor)
    (hs : s ∈ incidentMemoryWeightDomain N floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    memoryGramAttention (localMemoryCore (a • t + b • s)) =
      a • memoryGramAttention (localMemoryCore t) + b • memoryGramAttention (localMemoryCore s) := by
  rw [localMemoryCore_affine t s a b hab]
  exact memoryGramAttention_affine 1 floor _ _ (incidentMemoryCore_mem floor t hf ht)
    (incidentMemoryCore_mem floor s hf hs) a b ha hb hab

/-- A midpoint joins identity to a point excluded by the former global budget. -/
example : memoryGramAttention (localMemoryCore
    ((1 / 2 : ℝ) • (0 : Fin 3 → ℝ) + (1 / 2 : ℝ) • (fun _ : Fin 3 => (1 / 8 : ℝ)))) =
    (1 / 2 : ℝ) • memoryGramAttention (localMemoryCore (0 : Fin 3 → ℝ)) +
      (1 / 2 : ℝ) • memoryGramAttention (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) :=
  incidentMemoryCoreAttention_affine (3 / 4) _ _ (by norm_num)
    (zero_mem_incidentMemoryWeightDomain _ _ (by norm_num)) incidentMemoryExampleWeights_mem
    _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
