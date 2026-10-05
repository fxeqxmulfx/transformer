import Transformer.GPTMini.Sparsemax.GramRoutingEnergy
import Transformer.GPTMini.Sparsemax.ClosedForm

/-!
# Guaranteed nonsaturation with trainable embeddings and changing supports

Derived upstream constraints for arXiv:1602.02068v2, §2.2, Proposition 1.
A Gram entry cap below one half makes every pairwise score gap less than
one. Any causal row with two visible slots therefore has two positive
weights, for every feasible assignment of both embedding families.

This guarantee needs no attention targets, frozen Q/K coordinates or
prescribed support. A shared one-feature embedding example has an exact
visible zero, whereas the feasible zero Gram gives full support. The
score bound therefore permits sparse supports to change. It excludes only
singleton saturation of the score-to-weight map, not cancellation in a
value readout or every downstream task gradient.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Convex Gram constraints guarantee two active slots under every learned
embedding assignment. Source: the derived sub-unit range condition for
sparsemax Proposition 1, with two actually visible positions. -/
theorem embeddingRoutes_has_two_positive {V R T : ℕ} (cap : ℝ)
    (G : EmbeddingGram V) (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (r : Fin R) (a b : Fin T) (hG : G ∈ embeddingGramDomain V cap)
    (hc : cap < 1 / 2) (ha : a ≤ rows r) (hb : b ≤ rows r) (hne : a ≠ b) :
    ∃ j k, j ≠ k ∧ 0 < embeddingRoutes G tokens rows r j ∧
      0 < embeddingRoutes G tokens rows r k := by
  apply sparseWeights_has_two_positive_of_pairwise_gap _ _ a b ha hb hne
  intro j k
  have hu := (hG.2 (Sum.inl (tokens r (rows r))) (Sum.inr (tokens r j))).2
  have hl := (hG.2 (Sum.inl (tokens r (rows r))) (Sum.inr (tokens r k))).1
  change G (Sum.inl (tokens r (rows r))) (Sum.inr (tokens r j)) <
    G (Sum.inl (tokens r (rows r))) (Sum.inr (tokens r k)) + 1
  linarith

/-- One shared three-token context inhabits every nonsaturation premise. -/
example : ∃ j k : Fin 3, j ≠ k ∧
    0 < embeddingRoutes (0 : EmbeddingGram 3) (fun _ : Fin 1 => fun n => n)
      (fun _ => 2) 0 j ∧
    0 < embeddingRoutes (0 : EmbeddingGram 3) (fun _ : Fin 1 => fun n => n)
      (fun _ => 2) 0 k :=
  embeddingRoutes_has_two_positive (3 / 8) _ _ _ _ 0 1
    (zero_mem_embeddingGramDomain _ _ (by norm_num))
    (by norm_num) (by decide) (by decide) (by decide)

/-- A scalar embedding column: all queries and the first two keys agree,
while the third key has opposite sign. Source: a derived finite witness
for the bounded-Gram architecture preceding sparsemax §2.2. -/
def gramSparseVector : Sum (Fin 3) (Fin 3) → ℝ :=
  fun i => if i = Sum.inr 2 then -(3 / 5) else 3 / 5

/-- Its genuine one-feature Gram matrix, with both Q/K families nonzero.
Source: the derived bounded embedding witness for sparsemax §2.2. -/
def gramSparseExample : EmbeddingGram 3 :=
  Matrix.vecMulVec gramSparseVector gramSparseVector

/-- Ordinary token identities, shared by every row of this finite example.
Source: the derived Gram witness; there are no private occurrence embeddings. -/
def gramExampleTokens : Fin 1 → Fin 3 → Fin 3 := fun _ j => j

/-- The final causal query sees all three positions.
Source: the causal specialization of sparsemax Eq. (1). -/
def gramExampleRows : Fin 1 → Fin 3 := fun _ => 2

/-- The nonzero example satisfies the full PSD and entry-bound domain.
Source: the actual scalar embedding witness for the derived §2.2 constraint. -/
theorem gramSparseExample_mem_domain :
    gramSparseExample ∈ embeddingGramDomain 3 (3 / 8) := by
  refine ⟨?_, ?_⟩
  · simpa only [gramSparseExample, star_trivial] using
      Matrix.posSemidef_vecMulVec_self_star gramSparseVector
  · intro i j
    simp only [gramSparseExample, Matrix.vecMulVec_apply, gramSparseVector]
    split_ifs <;> norm_num

/-- The embedding Gram gives two equal high scores and one lower score.
Source: actual scalar products before sparsemax Eq. (1). -/
theorem gramSparseExample_scores : embeddingScores gramSparseExample (fun j => j) 2 =
    fun j : Fin 3 => if j = 2 then -(9 / 25) else 9 / 25 := by
  funext j
  fin_cases j <;> norm_num [embeddingScores, gramSparseExample,
    Matrix.vecMulVec_apply, gramSparseVector]

/-- Exact sparsemax weights: the lower visible position is truly zero.
Source: sparsemax Proposition 1, with threshold `-7/50` for this Gram. -/
theorem gramSparseExample_weights :
    embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows 0 =
      fun j : Fin 3 => if j = 2 then 0 else 1 / 2 := by
  change sparseWeights (embeddingScores gramSparseExample (fun j => j) 2) 2 = _
  rw [gramSparseExample_scores]
  let scores : Fin 3 → ℝ := fun j => if j = 2 then -(9 / 25) else 9 / 25
  have hsum : ∑ j, thresholdWeights scores 2 (-(7 / 50)) j = 1 := by
    norm_num [thresholdWeights, scores, Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights scores _ _ hsum]
  funext j
  fin_cases j <;> norm_num [thresholdWeights, scores]

/-- Another feasible embedding assignment has full support on the same row.
Source: sparsemax Proposition 1, with threshold `-1/3` at the zero Gram. -/
theorem gramZeroExample_weights :
    embeddingRoutes (0 : EmbeddingGram 3) gramExampleTokens gramExampleRows 0 =
      fun _ : Fin 3 => 1 / 3 := by
  change sparseWeights (embeddingScores (0 : EmbeddingGram 3) (fun j => j) 2) 2 = _
  have hs : embeddingScores (0 : EmbeddingGram 3) (fun j => j) 2 = fun _ => 0 := by
    funext j
    exact Matrix.zero_apply _ _
  rw [hs]
  have hsum : ∑ j : Fin 3, thresholdWeights (fun _ => 0) 2 (-(1 / 3)) j = 1 := by
    norm_num [thresholdWeights, Fin.sum_univ_three]
  rw [← thresholdWeights_eq_sparseWeights _ _ _ hsum]
  funext j
  fin_cases j <;> norm_num [thresholdWeights]

/-- Both endpoints lie in one convex domain but have different supports.
Source: the derived bounded embedding architecture for sparsemax §2.2.
The domain guarantees nonsaturation rather than imposing support equality. -/
theorem gramExample_support_changes :
    (0 : EmbeddingGram 3) ∈ embeddingGramDomain 3 (3 / 8) ∧
    gramSparseExample ∈ embeddingGramDomain 3 (3 / 8) ∧
    0 < embeddingRoutes (0 : EmbeddingGram 3) gramExampleTokens gramExampleRows 0 2 ∧
    embeddingRoutes gramSparseExample gramExampleTokens gramExampleRows 0 2 = 0 := by
  refine ⟨zero_mem_embeddingGramDomain _ _ (by norm_num),
    gramSparseExample_mem_domain, ?_, ?_⟩
  · rw [gramZeroExample_weights]
    norm_num
  · rw [gramSparseExample_weights]
    norm_num

/-- An entire moving-embedding segment remains in the convex domain.
Source: the bounded PSD restriction preceding sparsemax §2.2. Its endpoint
supports differ, so this is not the earlier fixed-support key segment. -/
theorem gramExampleSegment_feasible (t : ℝ) (ht : 0 ≤ t) (hu : t ≤ 1) :
    (1 - t) • (0 : EmbeddingGram 3) + t • gramSparseExample ∈
      embeddingGramDomain 3 (3 / 8) := by
  exact embeddingGramDomain_convex _ _
    (zero_mem_embeddingGramDomain _ _ (by norm_num)) gramSparseExample_mem_domain
    (by linarith) ht (by ring)

/-- The nonzero midpoint inhabits both segment hypotheses. -/
example : (1 - (1 / 2 : ℝ)) • (0 : EmbeddingGram 3) +
    (1 / 2 : ℝ) • gramSparseExample ∈ embeddingGramDomain 3 (3 / 8) :=
  gramExampleSegment_feasible _ (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
