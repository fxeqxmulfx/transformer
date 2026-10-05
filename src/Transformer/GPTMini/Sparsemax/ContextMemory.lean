import Transformer.GPTMini.Sparsemax.MemoryValues

/-!
# Actual attention for data-dependent mixtures of learned queries

Derived dictionary architecture for arXiv:1602.02068v2, Eq. (1), and
`attn @ v` at `73f8a0b`. A context supplies a probability code M, rather
than a target attention route. Its query is the M-weighted mixture of
learned dictionary query embeddings; keys are the shared learned memory.
The resulting genuine Q/K scores are M times the learned cross Gram.

Normalization makes actual variational sparsemax equal M times dictionary
attention. Arbitrarily many contexts may use different codes and repeated
tokens. All memory slots precede the query; they are learned parameters,
not future data. This is memory attention with fixed data codes, not
ordinary self-attention to the input occurrences or trainable code mixtures.
QKNorm, RoPE and fixed small feature width are not retained.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Data codes are probability mixtures of learned dictionary queries.
Source: the derived memory architecture before sparsemax Eq. (1). -/
def contextCodeDomain (R N : ℕ) : Set (Matrix (Fin R) (Fin (N + 1)) ℝ) :=
  {M | ∀ r, M r ∈ simplexOn Set.univ}

/-- Genuine mixture-query scores against the common learned memory keys.
Source: the derived Gram realization of Q/K scalar products in Eq. (1). -/
def contextMemoryScores {R N : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  M * memoryGramScores G

/-- Actual sparsemax projection over every learned memory slot.
Source: Eq. (1), with a last-slot causal mask after the dictionary. -/
def contextMemoryAttention {R N : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  Matrix.of (fun r => sparseWeights (contextMemoryScores G M r) (Fin.last N))

/-- Every actual context attention remains a probability row, even outside
the structural domain. Source: the feasible simplex in sparsemax Eq. (1). -/
theorem contextMemoryAttention_simplex {R N : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ) (r : Fin R) :
    contextMemoryAttention G M r ∈ simplexOn Set.univ := by
  have hs := (sparseWeights_spec (contextMemoryScores G M r) (Fin.last N)).1
  exact ⟨hs.1, hs.2.1, fun j hj => False.elim (hj (Set.mem_univ j))⟩

/-- Mixture scores are actual products of mixed learned Q and common K.
Source: the derived finite Gram architecture; M is data, not a routing target.
The feature family is recoverable from every feasible PSD Gram. -/
theorem contextMemoryScores_recovered {R N F : ℕ} (G : EmbeddingGram (N + 1))
    (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (features : Fin F → Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)
    (hf : G = featureGram features) (r : Fin R) (k : Fin (N + 1)) :
    contextMemoryScores G M r k =
      ∑ d, (∑ q, M r q * features d (Sum.inl q)) * features d (Sum.inr k) := by
  unfold contextMemoryScores
  rw [Matrix.mul_apply]
  simp only [memoryGramScores, Matrix.of_apply]
  rw [hf]
  simp only [featureGram_apply]
  simp_rw [Finset.mul_sum, Finset.sum_mul, mul_assoc]
  exact Finset.sum_comm

/-- An actual nonzero feature table inhabits exact mixture-score recovery. -/
example : contextMemoryScores (featureGram (fun _ : Fin 1 => fun _ : Sum (Fin 1) (Fin 1) => (1 : ℝ)))
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) 0 0 =
    ∑ d : Fin 1, (∑ q : Fin 1, (1 : ℝ) *
      (fun _ : Fin 1 => fun _ : Sum (Fin 1) (Fin 1) => (1 : ℝ)) d (Sum.inl q)) * 1 :=
  contextMemoryScores_recovered _ _ _ rfl _ _

/-- Probability codes and normalized learned rows give normalized mixed scores.
Source: the new dictionary/query architecture before sparsemax Eq. (1). -/
theorem contextMemoryScores_simplex {R N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) (hM : M ∈ contextCodeDomain R N) (r : Fin R) :
    contextMemoryScores G M r ∈ simplexOn Set.univ := by
  refine ⟨?_, ?_, fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · intro j
    rw [contextMemoryScores, Matrix.mul_apply]
    exact Finset.sum_nonneg fun q _ => mul_nonneg ((hM r).1 q) ((hG.2.1 q).1 j)
  · change (∑ j, (M * memoryGramScores G) r j) = 1
    simp only [Matrix.mul_apply]
    rw [Finset.sum_comm]
    simp_rw [← Finset.mul_sum, (hG.2.1 _).2.1, mul_one]
    exact (hM r).2.1

/-- Every actual context attention factors through the one learned dictionary.
Source: Eq. (1) on the proved normalized genuine Q/K mixture scores. -/
theorem contextMemoryAttention_eq_product {R N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) (hM : M ∈ contextCodeDomain R N) :
    contextMemoryAttention G M = M * memoryGramAttention G := by
  rw [memoryGramAttention_normalized cap floor G hG]
  funext r
  have hs := contextMemoryScores_simplex cap floor G M hG hM r
  apply sparseWeights_eq_self_of_simplex
  exact ⟨hs.1, hs.2.1, fun j hj => False.elim (hj (Fin.le_last j))⟩

/-- A concrete probability code inhabits the data-code domain.
Source: the one-slot query mixture in the derived memory architecture. -/
theorem unitContextCode_mem :
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ)) ∈ contextCodeDomain 1 0 := by
  intro r
  refine ⟨?_, ?_, fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · intro j
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- Genuine learned embeddings and data codes satisfy normalization together. -/
example : contextMemoryScores normalizedGramUnit
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) 0 ∈ simplexOn Set.univ :=
  contextMemoryScores_simplex 1 _ _ _ normalizedGramUnit_mem_memory unitContextCode_mem _

/-- The actual sparsemax factorization hypotheses have a nonempty instance. -/
example : contextMemoryAttention normalizedGramUnit
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) =
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) *
        memoryGramAttention normalizedGramUnit :=
  contextMemoryAttention_eq_product 1 _ _ _ normalizedGramUnit_mem_memory unitContextCode_mem

/-- Actual context attention remains affine while both dictionary Q/K families change.
Source: the derived normalized-memory restriction of Eq. (1); codes are fixed data. -/
theorem contextMemoryAttention_affine {R N : ℕ} (cap floor : ℝ)
    (G H : EmbeddingGram (N + 1)) (M : Matrix (Fin R) (Fin (N + 1)) ℝ)
    (hG : G ∈ memoryGramDomain N cap floor) (hH : H ∈ memoryGramDomain N cap floor)
    (hM : M ∈ contextCodeDomain R N) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    contextMemoryAttention (a • G + b • H) M =
      a • contextMemoryAttention G M + b • contextMemoryAttention H M := by
  have hm := memoryGramDomain_convex N cap floor hG hH ha hb hab
  rw [contextMemoryAttention_eq_product cap floor _ _ hm hM,
    memoryGramAttention_affine cap floor G H hG hH a b ha hb hab,
    Matrix.mul_add, Matrix.mul_smul, Matrix.mul_smul,
    ← contextMemoryAttention_eq_product cap floor G M hG hM,
    ← contextMemoryAttention_eq_product cap floor H M hH hM]

/-- All affinity premises are jointly satisfied by actual embeddings and codes. -/
example : contextMemoryAttention
    ((1 / 2 : ℝ) • normalizedGramUnit + (1 / 2 : ℝ) • normalizedGramUnit)
    (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) =
    (1 / 2 : ℝ) • contextMemoryAttention normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) +
    (1 / 2 : ℝ) • contextMemoryAttention normalizedGramUnit
      (Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 => (1 : ℝ))) :=
  contextMemoryAttention_affine 1 _ _ _ _ normalizedGramUnit_mem_memory normalizedGramUnit_mem_memory
    unitContextCode_mem _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
