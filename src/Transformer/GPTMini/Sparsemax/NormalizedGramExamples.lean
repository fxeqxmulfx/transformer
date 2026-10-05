import Transformer.GPTMini.Sparsemax.NormalizedGram

/-!
# Exact convex attention while embeddings and sparse supports change

Derived finite instance of the normalized-Gram architecture preceding
arXiv:1602.02068v2, Eq. (1). Four token positions share one trainable Gram.
Both endpoints are genuine one-feature Q/K embedding matrices, but their
query and key squared norms differ. The first attends uniformly; the
second assigns one third to the first three slots and exactly zero to the
last. Their actual sparsemax midpoint is the mean of those attention rows.

All matrices satisfy the same convex PSD and entry bound `3/8` and the
linear causal normalization constraints. The sub-half cap excludes
singleton saturation. These are exact real-valued witnesses, not a trained
Python model, a fixed-width recovery theorem for every feasible matrix,
or a guarantee for arbitrary new contexts. Values and task loss are deferred.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- One shared four-token context for the normalized embedding witness.
Source: the derived finite sparsemax Eq. (1) architecture. -/
def normalizedExampleTokens : Fin 1 → Fin 4 → Fin 4 := fun _ j => j

/-- The queried row sees all four positions causally.
Source: the causal specialization of sparsemax Eq. (1). -/
def normalizedExampleRows : Fin 1 → Fin 4 := fun _ => 3

/-- Initial genuine query/key embedding Gram, with content scores `1/4`.
Source: the derived normalized-Gram witness; both families start at `1/2`. -/
def normalizedUniformGram : EmbeddingGram 4 :=
  Matrix.vecMulVec (fun _ => 1 / 2) (fun _ => 1 / 2)

/-- Changed query and key embeddings, including an exactly zero last key.
Source: a finite learned-embedding witness for the new Eq. (1) restriction. -/
def normalizedSparseVector : Sum (Fin 4) (Fin 4) → ℝ
  | Sum.inl _ => 3 / 5
  | Sum.inr j => if j = 3 then 0 else 5 / 9

/-- Genuine one-feature Gram of the changed embedding families.
Source: the derived normalized-Gram witness; queries and keys both change. -/
def normalizedSparseGram : EmbeddingGram 4 :=
  Matrix.vecMulVec normalizedSparseVector normalizedSparseVector

/-- The uniform embedding table satisfies the full bounded PSD constraints.
Source: the explicit `3/8` upstream bound for sparsemax Proposition 1. -/
theorem normalizedUniformGram_bounded :
    normalizedUniformGram ∈ embeddingGramDomain 4 (3 / 8) := by
  refine ⟨?_, ?_⟩
  · simpa only [normalizedUniformGram, star_trivial] using
      Matrix.posSemidef_vecMulVec_self_star (fun _ : Sum (Fin 4) (Fin 4) => (1 / 2 : ℝ))
  · intro i j
    norm_num [normalizedUniformGram, Matrix.vecMulVec_apply]

/-- The changed table satisfies exactly the same bounded PSD constraints.
Source: the genuine scalar embedding witness for the derived §2.2 bound. -/
theorem normalizedSparseGram_bounded :
    normalizedSparseGram ∈ embeddingGramDomain 4 (3 / 8) := by
  refine ⟨?_, ?_⟩
  · simpa only [normalizedSparseGram, star_trivial] using
      Matrix.posSemidef_vecMulVec_self_star normalizedSparseVector
  · intro i j
    cases i <;> cases j <;>
      simp only [normalizedSparseGram, Matrix.vecMulVec_apply, normalizedSparseVector] <;>
      (try split_ifs) <;> norm_num

/-- Uniform Gram scores are already causal probabilities.
Source: the actual embedding scalar products before sparsemax Eq. (1). -/
theorem normalizedUniformGram_scores :
    embeddingScores normalizedUniformGram (fun j => j) 3 = fun _ : Fin 4 => 1 / 4 := by
  funext j
  norm_num [embeddingScores, normalizedUniformGram, Matrix.vecMulVec_apply]

/-- Changed Gram scores are sparse causal probabilities.
Source: the actual changed embedding products before sparsemax Eq. (1). -/
theorem normalizedSparseGram_scores :
    embeddingScores normalizedSparseGram (fun j => j) 3 =
      fun j : Fin 4 => if j = 3 then 0 else 1 / 3 := by
  funext j
  fin_cases j <;> norm_num [embeddingScores, normalizedSparseGram,
    Matrix.vecMulVec_apply, normalizedSparseVector]

/-- The uniform matrix inhabits the exact normalized embedding domain.
Source: the derived simplex constraint before sparsemax Eq. (1). -/
theorem normalizedUniformGram_mem_domain : normalizedUniformGram ∈
    normalizedGramDomain 4 (3 / 8) normalizedExampleTokens normalizedExampleRows := by
  refine ⟨normalizedUniformGram_bounded, ?_⟩
  intro r
  change embeddingScores normalizedUniformGram (fun j => j) 3 ∈ simplexOn {j : Fin 4 | j ≤ 3}
  rw [normalizedUniformGram_scores]
  refine ⟨by intro j; norm_num, by norm_num [Fin.sum_univ_four], ?_⟩
  intro j hj
  fin_cases j <;> norm_num at hj

/-- The changed matrix inhabits the same exact normalized embedding domain.
Source: the derived simplex constraint before sparsemax Eq. (1). -/
theorem normalizedSparseGram_mem_domain : normalizedSparseGram ∈
    normalizedGramDomain 4 (3 / 8) normalizedExampleTokens normalizedExampleRows := by
  refine ⟨normalizedSparseGram_bounded, ?_⟩
  intro r
  change embeddingScores normalizedSparseGram (fun j => j) 3 ∈ simplexOn {j : Fin 4 | j ≤ 3}
  rw [normalizedSparseGram_scores]
  refine ⟨?_, by norm_num [Fin.sum_univ_four], ?_⟩
  · intro j
    change 0 ≤ if j = 3 then (0 : ℝ) else 1 / 3
    split_ifs <;> norm_num
  · intro j hj
    fin_cases j <;> norm_num at hj

/-- Sparsemax returns the uniform endpoint scores exactly.
Source: Eq. (1) under the proved normalized-Gram constraint. -/
theorem normalizedUniformGram_weights :
    embeddingRoutes normalizedUniformGram normalizedExampleTokens normalizedExampleRows 0 =
      fun _ : Fin 4 => 1 / 4 := by
  rw [normalizedGram_routes_eq_scores (3 / 8) _ _ _ normalizedUniformGram_mem_domain]
  exact normalizedUniformGram_scores

/-- Sparsemax returns the changed sparse endpoint scores exactly.
Source: Eq. (1) under the proved normalized-Gram constraint. -/
theorem normalizedSparseGram_weights :
    embeddingRoutes normalizedSparseGram normalizedExampleTokens normalizedExampleRows 0 =
      fun j : Fin 4 => if j = 3 then 0 else 1 / 3 := by
  rw [normalizedGram_routes_eq_scores (3 / 8) _ _ _ normalizedSparseGram_mem_domain]
  exact normalizedSparseGram_scores

/-- Genuine query and key embedding norms both change between endpoints.
Source: the diagonal Gram entries of the derived finite embedding witness. -/
theorem normalizedExample_both_embedding_families_change :
    normalizedUniformGram (Sum.inl 0) (Sum.inl 0) ≠
      normalizedSparseGram (Sum.inl 0) (Sum.inl 0) ∧
    normalizedUniformGram (Sum.inr 0) (Sum.inr 0) ≠
      normalizedSparseGram (Sum.inr 0) (Sum.inr 0) := by
  norm_num [normalizedUniformGram, normalizedSparseGram,
    Matrix.vecMulVec_apply, normalizedSparseVector]

/-- Support changes inside the exact convex attention domain.
Source: the derived normalized-Gram restriction of sparsemax Eq. (1). -/
theorem normalizedExample_support_changes :
    0 < embeddingRoutes normalizedUniformGram normalizedExampleTokens normalizedExampleRows 0 3 ∧
    embeddingRoutes normalizedSparseGram normalizedExampleTokens normalizedExampleRows 0 3 = 0 := by
  rw [normalizedUniformGram_weights, normalizedSparseGram_weights]
  norm_num

/-- The actual midpoint attention is exactly the mean endpoint attention.
Source: the new exact convex Gram architecture; the endpoint supports differ. -/
theorem normalizedExample_midpoint_weights : embeddingRoutes
    ((1 / 2 : ℝ) • normalizedUniformGram + (1 / 2 : ℝ) • normalizedSparseGram)
    normalizedExampleTokens normalizedExampleRows 0 =
      fun j : Fin 4 => if j = 3 then 1 / 8 else 7 / 24 := by
  have hm := normalizedGramDomain_convex 4 (3 / 8) normalizedExampleTokens normalizedExampleRows
    normalizedUniformGram_mem_domain normalizedSparseGram_mem_domain
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  rw [normalizedGram_routes_eq_scores (3 / 8) _ _ _ hm]
  change embeddingScores _ (fun j => j) 3 = _
  rw [embeddingScores_affine, normalizedUniformGram_scores, normalizedSparseGram_scores]
  funext j
  fin_cases j <;> norm_num

/-- Exact convex Gram interpolation can require a larger embedding width.
Source: the derived normalized-Gram example; the endpoints use one feature,
but the midpoint's positive two-by-two minor forbids one-feature recovery. -/
theorem normalizedExample_midpoint_not_one_feature :
    ¬ ∃ features : Fin 1 → Sum (Fin 4) (Fin 4) → ℝ,
      (1 / 2 : ℝ) • normalizedUniformGram + (1 / 2 : ℝ) • normalizedSparseGram =
        featureGram features := by
  rintro ⟨features, hf⟩
  have hq := congrArg (fun G : EmbeddingGram 4 => G (Sum.inl 0) (Sum.inl 0)) hf
  have hk := congrArg (fun G : EmbeddingGram 4 => G (Sum.inr 3) (Sum.inr 3)) hf
  have hc := congrArg (fun G : EmbeddingGram 4 => G (Sum.inl 0) (Sum.inr 3)) hf
  rw [featureGram_apply] at hq hk hc
  norm_num [normalizedUniformGram, normalizedSparseGram, Matrix.vecMulVec_apply,
    normalizedSparseVector, Fin.sum_univ_one] at hq hk hc
  have hprod : (features 0 (Sum.inl 0) * features 0 (Sum.inl 0)) *
      (features 0 (Sum.inr 3) * features 0 (Sum.inr 3)) =
      (features 0 (Sum.inl 0) * features 0 (Sum.inr 3)) ^ 2 := by ring
  rw [← hq, ← hk, ← hc] at hprod
  norm_num at hprod

end Transformer.GPTMini.Sparsemax
