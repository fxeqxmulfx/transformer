import Transformer.GPTMini.Sparsemax.JointGramValues

/-!
# Two different learned embedding matrices with invertible causal attention

Derived finite witness for sparsemax arXiv:1602.02068v2, Eq. (1), and
the joint value coordinates for `attn @ v` at `73f8a0b`. Two token IDs
are distinct and both causal rows share one Gram and one value table.
The initial attention is identity. Changed query and key embeddings give
the matrix `[[1, 0], [1/2, 1/2]]`, adding a visible active position.

Both embedding families change their squared norms. Both actual sparsemax
matrices satisfy the same convex normalized-Gram domain, entry cap four
and self-weight floor one half. The first causal row necessarily remains
a self route; the second need not. This is a mathematical prototype for
the value-coordinate change, not an experiment or arbitrary-context claim.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Initial actual query/key embeddings are the two coordinate vectors.
Source: the derived finite Gram witness for the joint value architecture. -/
def valueExampleStartFeatures (d : Fin 2) : Sum (Fin 2) (Fin 2) → ℝ
  | Sum.inl i => if d = i then 1 else 0
  | Sum.inr j => if d = j then 1 else 0

/-- Changed actual query/key embeddings produce triangular normalized scores.
Source: the derived finite witness; query coordinates double and keys change. -/
def valueExampleStopFeatures (d : Fin 2) : Sum (Fin 2) (Fin 2) → ℝ
  | Sum.inl i => if d = i then 2 else 0
  | Sum.inr j => if d = 0 then (if j = 0 then 1 / 2 else 0) else 1 / 4

/-- One genuine initial shared embedding Gram.
Source: the derived two-token architecture preceding sparsemax Eq. (1). -/
def valueExampleStartGram : EmbeddingGram 2 := featureGram valueExampleStartFeatures

/-- One genuine changed shared embedding Gram, with both Q/K families learned.
Source: the derived two-token architecture preceding sparsemax Eq. (1). -/
def valueExampleStopGram : EmbeddingGram 2 := featureGram valueExampleStopFeatures

/-- Both original feature tables satisfy the same PSD and entry constraints.
Source: the derived cap-four embedding restriction for the exact value model. -/
theorem valueExampleStartGram_bounded : valueExampleStartGram ∈ embeddingGramDomain 2 4 := by
  refine ⟨featureGram_posSemidef _, ?_⟩
  intro i j
  rcases i with i | i <;> rcases j with j | j <;> fin_cases i <;> fin_cases j <;>
    norm_num [valueExampleStartGram, featureGram_apply, valueExampleStartFeatures, Fin.sum_univ_two]

/-- The changed feature table also has genuine PSD and bounded entries.
Source: the actual changed embeddings in the derived cap-four restriction. -/
theorem valueExampleStopGram_bounded : valueExampleStopGram ∈ embeddingGramDomain 2 4 := by
  refine ⟨featureGram_posSemidef _, ?_⟩
  intro i j
  rcases i with i | i <;> rcases j with j | j <;> fin_cases i <;> fin_cases j <;>
    norm_num [valueExampleStopGram, featureGram_apply, valueExampleStopFeatures, Fin.sum_univ_two]

/-- Initial scores exactly select the current token in both causal rows.
Source: the actual embedding scalar products preceding sparsemax Eq. (1). -/
theorem valueExampleStartGram_scores (i : Fin 2) :
    embeddingScores valueExampleStartGram (fun j => j) i = basis i := by
  funext j
  fin_cases i <;> fin_cases j <;> norm_num [embeddingScores, valueExampleStartGram,
    featureGram_apply, valueExampleStartFeatures, Fin.sum_univ_two, basis]

/-- Changed scores mix both values in the second row, with the first unchanged.
Source: the changed actual embedding products preceding sparsemax Eq. (1). -/
theorem valueExampleStopGram_scores (i : Fin 2) :
    embeddingScores valueExampleStopGram (fun j => j) i =
      fun j => if i = 0 then (if j = 0 then 1 else 0) else 1 / 2 := by
  funext j
  fin_cases i <;> fin_cases j <;> norm_num [embeddingScores, valueExampleStopGram,
    featureGram_apply, valueExampleStopFeatures, Fin.sum_univ_two]

/-- Initial embeddings satisfy normalization and the structural diagonal floor.
Source: the derived invertible causal domain for the exact learned-value change. -/
theorem valueExampleStartGram_mem_domain :
    valueExampleStartGram ∈ invertibleGramDomain 2 4 (1 / 2) := by
  refine ⟨⟨valueExampleStartGram_bounded, ?_⟩, ?_⟩
  · intro i
    change embeddingScores valueExampleStartGram (fun j => j) i ∈ simplexOn {j : Fin 2 | j ≤ i}
    rw [valueExampleStartGram_scores]
    exact basis_mem_simplex {j : Fin 2 | j ≤ i} i (by change i ≤ i; exact le_rfl)
  · intro i
    rw [valueExampleStartGram_scores]
    norm_num [basis]

/-- Changed embeddings satisfy the same convex normalization and floor constraints.
Source: the derived invertible causal domain for `attn @ v`. -/
theorem valueExampleStopGram_mem_domain :
    valueExampleStopGram ∈ invertibleGramDomain 2 4 (1 / 2) := by
  refine ⟨⟨valueExampleStopGram_bounded, ?_⟩, ?_⟩
  · intro i
    change embeddingScores valueExampleStopGram (fun j => j) i ∈ simplexOn {j : Fin 2 | j ≤ i}
    rw [valueExampleStopGram_scores]
    refine ⟨?_, ?_, ?_⟩
    · intro j
      fin_cases i <;> fin_cases j <;> norm_num
    · fin_cases i <;> norm_num [Fin.sum_univ_two]
    · intro j hj
      fin_cases i <;> fin_cases j <;> norm_num at *
  · intro i
    rw [valueExampleStopGram_scores]
    fin_cases i <;> norm_num

/-- Actual initial variational sparsemax attention is the identity matrix.
Source: Eq. (1) on the proved normalized initial Gram. -/
theorem valueExampleStartGram_attention : causalGramAttention valueExampleStartGram = 1 := by
  rw [causalGramAttention_normalized 4 _ valueExampleStartGram_mem_domain.1]
  ext i j
  change embeddingScores valueExampleStartGram (fun j => j) i j = (1 : Matrix (Fin 2) (Fin 2) ℝ) i j
  rw [valueExampleStartGram_scores]
  fin_cases i <;> fin_cases j <;> norm_num [basis]

/-- Actual changed sparsemax attention adds a positive earlier position.
Source: Eq. (1) on the proved normalized changed Gram. -/
theorem valueExampleStopGram_attention : causalGramAttention valueExampleStopGram =
    Matrix.of (fun i j => if i = 0 then (if j = 0 then 1 else 0) else 1 / 2) := by
  rw [causalGramAttention_normalized 4 _ valueExampleStopGram_mem_domain.1]
  funext i j
  exact congrFun (valueExampleStopGram_scores i) j

/-- Both embedding families change, rather than only the attention row or values.
Source: genuine squared norms in the two explicit finite embedding tables. -/
theorem valueExample_both_embedding_families_change :
    valueExampleStartGram (Sum.inl 0) (Sum.inl 0) ≠
      valueExampleStopGram (Sum.inl 0) (Sum.inl 0) ∧
    valueExampleStartGram (Sum.inr 0) (Sum.inr 0) ≠
      valueExampleStopGram (Sum.inr 0) (Sum.inr 0) := by
  norm_num [valueExampleStartGram, valueExampleStopGram, featureGram_apply,
    valueExampleStartFeatures, valueExampleStopFeatures, Fin.sum_univ_two]

/-- The visible support changes while the inverse guarantee remains valid.
Source: actual sparsemax Eq. (1) in the two feasible embedding examples. -/
theorem valueExample_support_changes :
    causalGramAttention valueExampleStartGram 1 0 = 0 ∧
    0 < causalGramAttention valueExampleStopGram 1 0 := by
  rw [valueExampleStartGram_attention, valueExampleStopGram_attention]
  norm_num

/-- The changed actual attention is genuinely nonsingular, with determinant one half.
Source: the finite witness for the structural `attn @ v` recovery guarantee. -/
theorem valueExampleStopGram_det : (causalGramAttention valueExampleStopGram).det = 1 / 2 := by
  rw [Matrix.det_fin_two, valueExampleStopGram_attention]
  norm_num

end Transformer.GPTMini.Sparsemax
