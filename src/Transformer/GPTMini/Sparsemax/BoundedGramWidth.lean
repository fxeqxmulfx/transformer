import Transformer.GPTMini.Sparsemax.ContextMemory

/-!
# Exact embedding recovery with a dictionary-size width bound

Derived finite-memory architecture for arXiv:1602.02068v2, Eq. (1).
A PSD Gram on the query and key copies of P slots has genuine embedding
coordinates of width 2P. This is a consequence of positive spectral
decomposition, rather than a rank restriction on the convex domain.

The spectral proof below retains the explicit dimension in Mathlib's
`ContinuousLinearMap.isPositive_iff_eq_sum_rankOne` and its matrix corollary
(Anatole Dedecker, Apache 2.0). The original results existentially quantify
the width; their proofs already use an orthonormal eigenvector basis.
The bound grows with the dictionary and does not imply constant head width.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open InnerProductSpace RCLike LinearMap ContinuousLinearMap
open scoped BigOperators InnerProduct ComplexConjugate

/-- Positive spectral decomposition uses exactly the ambient dimension.
Source: dimension-explicit adaptation of Mathlib's
`ContinuousLinearMap.isPositive_iff_eq_sum_rankOne`, for the Gram lifting
preceding sparsemax arXiv:1602.02068v2, Eq. (1). Zero eigenvalues are allowed. -/
theorem positive_fixedWidth_rankOne {𝕜 E : Type*} [RCLike 𝕜]
    [NormedAddCommGroup E] [InnerProductSpace 𝕜 E] [FiniteDimensional 𝕜 E]
    (T : E →L[𝕜] E) (hT : T.IsPositive) :
    ∃ u : Fin (Module.finrank 𝕜 E) → E,
      T = ∑ i, rankOne 𝕜 (u i) (u i) := by
  let a (i : Fin (Module.finrank 𝕜 E)) : E :=
    ((hT.isSymmetric.eigenvalues rfl i).sqrt : 𝕜) •
      hT.isSymmetric.eigenvectorBasis rfl i
  refine ⟨a, ContinuousLinearMap.ext fun _ => ?_⟩
  simp_rw [_root_.sum_apply, rankOne_apply, a, inner_smul_left, smul_smul,
    mul_assoc, conj_ofReal, mul_comm (inner 𝕜 _ _), ← mul_assoc, ← ofReal_mul,
    ← Real.sqrt_mul (hT.toLinearMap.nonneg_eigenvalues rfl _),
    Real.sqrt_mul_self (hT.toLinearMap.nonneg_eigenvalues rfl _),
    mul_comm _ (inner 𝕜 _ _), ← smul_eq_mul, smul_assoc,
    ← hT.isSymmetric.apply_eigenvectorBasis, ← map_smul, ← map_sum,
    ← OrthonormalBasis.repr_apply_apply, OrthonormalBasis.sum_repr,
    ContinuousLinearMap.coe_coe]

/-- A genuine positive operator inhabits the spectral-decomposition premise. -/
example : ∃ u : Fin (Module.finrank ℝ (EuclideanSpace ℝ (Fin 2))) →
    EuclideanSpace ℝ (Fin 2),
    (0 : EuclideanSpace ℝ (Fin 2) →L[ℝ] EuclideanSpace ℝ (Fin 2)) =
      ∑ i, rankOne ℝ (u i) (u i) :=
  positive_fixedWidth_rankOne _ ContinuousLinearMap.isPositive_zero

/-- Every finite PSD matrix is an ordinary feature Gram of width its order.
Source: dimension-explicit adaptation of Mathlib's
`Matrix.posSemidef_iff_eq_sum_vecMulVec`, for the derived arXiv:1602.02068v2, Eq. (1) architecture. -/
theorem posSemidef_featureGram_card {ι : Type*} [Fintype ι]
    (G : Matrix ι ι ℝ) (hG : G.PosSemidef) :
    ∃ features : Fin (Fintype.card ι) → ι → ℝ, G = featureGram features := by
  classical
  have hp := (LinearMap.isPositive_toContinuousLinearMap_iff G.toEuclideanLin).2
    (Matrix.isPositive_toEuclideanLin_iff.2 hG)
  have hd := positive_fixedWidth_rankOne _ hp
  rw [finrank_euclideanSpace (𝕜 := ℝ) (ι := ι)] at hd
  obtain ⟨u, hu⟩ := hd
  refine ⟨fun i => (u i).ofLp, ?_⟩
  simp_rw [eq_comm, ← LinearEquiv.symm_apply_eq, coe_toContinuousLinearMap_symm,
    ContinuousLinearMap.toLinearMap_sum, map_sum, symm_toEuclideanLin_rankOne,
    eq_comm] at hu
  simpa only [featureGram, Pi.star_def, star_trivial] using hu

/-- The width theorem applies to a concrete PSD matrix, without assumed features. -/
example : ∃ features : Fin 2 → Fin 2 → ℝ,
    (1 : Matrix (Fin 2) (Fin 2) ℝ) = featureGram features := by
  have h := posSemidef_featureGram_card (1 : Matrix (Fin 2) (Fin 2) ℝ) Matrix.PosSemidef.one
  rw [Fintype.card_fin] at h
  exact h

/-- Every bounded Q/K Gram has width 2V coordinates with the original norm cap.
Source: the dimension bound for the learned Gram preceding sparsemax
arXiv:1602.02068v2, Eq. (1); the convex domain receives no rank restriction. -/
theorem embeddingGramDomain_fixedWidth {V : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (hG : G ∈ embeddingGramDomain V cap) :
    ∃ features : Fin (2 * V) → Sum (Fin V) (Fin V) → ℝ,
      G = featureGram features ∧ ∀ i, (∑ d, (features d i) ^ 2) ≤ cap := by
  have h := posSemidef_featureGram_card G hG.1
  have hc : Fintype.card (Sum (Fin V) (Fin V)) = 2 * V := by
    simp only [Fintype.card_sum, Fintype.card_fin, two_mul]
  rw [hc] at h
  obtain ⟨features, hf⟩ := h
  exact ⟨features, hf, recovered_embedding_sq_bound cap G features hG hf⟩

/-- A nonzero learned Gram satisfies recovery and the cap at the stated width. -/
example : ∃ features : Fin 2 → Sum (Fin 1) (Fin 1) → ℝ,
    normalizedGramUnit = featureGram features ∧
      ∀ i, (∑ d, (features d i) ^ 2) ≤ (1 : ℝ) :=
  embeddingGramDomain_fixedWidth 1 _ normalizedGramUnit_mem_memory.1

/-- A learned memory on P slots always has a genuine Q/K realization of width 2P.
Source: the derived compact memory for arXiv:1602.02068v2, Eq. (1).
Normalization and the floor constrain scores without increasing feature width. -/
theorem memoryGram_fixedWidth {N : ℕ} (cap floor : ℝ) (G : EmbeddingGram (N + 1))
    (hG : G ∈ memoryGramDomain N cap floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      G = featureGram features := by
  obtain ⟨features, hf, _⟩ := embeddingGramDomain_fixedWidth cap G hG.1
  exact ⟨features, hf⟩

/-- The width-bound premise is inhabited by actual nonzero dictionary embeddings. -/
example : ∃ features : Fin 2 → Sum (Fin 1) (Fin 1) → ℝ,
    normalizedGramUnit = featureGram features :=
  memoryGram_fixedWidth 1 (3 / 4) _ normalizedGramUnit_mem_memory

/-- Arbitrarily many context queries share the same bounded-width learned Q/K table.
Source: genuine mixture-query inner products before sparsemax
arXiv:1602.02068v2, Eq. (1); context count never enters the feature width. -/
theorem contextMemory_fixedWidth_scores {R N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      G = featureGram features ∧ ∀ r k, contextMemoryScores G M r k =
        ∑ d, (∑ q, M r q * features d (Sum.inl q)) * features d (Sum.inr k) := by
  obtain ⟨features, hf⟩ := memoryGram_fixedWidth cap floor G hG
  exact ⟨features, hf, contextMemoryScores_recovered G M features hf⟩

/-- The actual nonzero memory and data mixture inhabit joint bounded-width scores. -/
example : ∃ features : Fin 2 → Sum (Fin 1) (Fin 1) → ℝ,
    normalizedGramUnit = featureGram features ∧ ∀ r k,
      contextMemoryScores normalizedGramUnit
        (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) r k =
      ∑ d, (∑ q : Fin 1, (1 : ℝ) * features d (Sum.inl q)) * features d (Sum.inr k) :=
  contextMemory_fixedWidth_scores 1 (3 / 4) _ _ normalizedGramUnit_mem_memory

/-- The actual sparsemax output has the same width-2P realization on the convex domain.
Source: arXiv:1602.02068v2, Eq. (1), fixes normalized mixture scores; spectral recovery
therefore realizes attention itself, including every allowed support change. -/
theorem contextMemory_fixedWidth_attention {R N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) (hM : M ∈ contextCodeDomain R N) :
    ∃ features : Fin (2 * (N + 1)) → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ,
      G = featureGram features ∧ ∀ r k, contextMemoryAttention G M r k =
        ∑ d, (∑ q, M r q * features d (Sum.inl q)) * features d (Sum.inr k) := by
  obtain ⟨features, hf⟩ := memoryGram_fixedWidth cap floor G hG
  refine ⟨features, hf, fun r k => ?_⟩
  rw [contextMemoryAttention_eq_product cap floor G M hG hM,
    memoryGramAttention_normalized cap floor G hG]
  exact contextMemoryScores_recovered G M features hf r k

/-- Actual embeddings and a real probability code satisfy the exact attention premises. -/
example : ∃ features : Fin 2 → Sum (Fin 1) (Fin 1) → ℝ,
    normalizedGramUnit = featureGram features ∧ ∀ r k,
      contextMemoryAttention normalizedGramUnit
        (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) r k =
      ∑ d, (∑ q : Fin 1, (1 : ℝ) * features d (Sum.inl q)) * features d (Sum.inr k) :=
  contextMemory_fixedWidth_attention 1 (3 / 4) _ _
    normalizedGramUnit_mem_memory unitContextCode_mem

end Transformer.GPTMini.Sparsemax
