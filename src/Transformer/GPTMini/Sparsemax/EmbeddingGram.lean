import Transformer.GPTMini.Sparsemax.NonSaturation
import Mathlib.Analysis.InnerProductSpace.Positive

/-!
# A convex representation of learned query and key embeddings

Derived architecture for arXiv:1602.02068v2, §2.1–§2.2, before its
simplex projection. Learn one positive semidefinite Gram matrix on the
query and key copies of a finite vocabulary. Its cross entries are content
scores, shared across every occurrence of a token. Both embedding families
are free; their coordinates are recovered after optimization.

This replaces normalized Q/K products at `73f8a0b` by unnormalized Gram
scores and an explicit convex entry bound. It does not preserve QKNorm,
RoPE, a fixed embedding width, or the original parameter regularizer.
Finite feature recovery is exact without a width bound. No assertion about
learned value mixtures, an FFN, or a downstream task loss is made here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Joint Gram coordinates for two learned embedding families.
Source: the derived content-score representation preceding §2.2 of
arXiv:1602.02068v2; query and key copies need not coincide. -/
abbrev EmbeddingGram (V : ℕ) :=
  Matrix (Sum (Fin V) (Fin V)) (Sum (Fin V) (Fin V)) ℝ

/-- An ordinary finite embedding table, represented by its Gram matrix.
Source: the derived lifting of the query/key inner products; the feature
dimension is explicit and is not assumed to equal a model's head width. -/
def featureGram {ι : Type*} {D : ℕ} (features : Fin D → ι → ℝ) : Matrix ι ι ℝ :=
  ∑ d, Matrix.vecMulVec (features d) (features d)

/-- Each matrix entry is the scalar product of the two feature columns.
Source: the derived Gram representation preceding sparsemax Eq. (1). -/
theorem featureGram_apply {ι : Type*} {D : ℕ} (features : Fin D → ι → ℝ) (i j : ι) :
    featureGram features i j = ∑ d, features d i * features d j := by
  rw [featureGram, Matrix.sum_apply]
  exact Finset.sum_congr rfl fun d _ => Matrix.vecMulVec_apply _ _ _ _

/-- Positive semidefinite matrices are exactly finite embedding Grams.
Source: the derived lifting of the inner products preceding Eq. (1) of
arXiv:1602.02068v2. The converse allows the feature dimension to grow. -/
theorem posSemidef_iff_featureGram {ι : Type*} [Finite ι] (G : Matrix ι ι ℝ) :
    G.PosSemidef ↔ ∃ D : ℕ, ∃ features : Fin D → ι → ℝ, G = featureGram features := by
  rw [Matrix.posSemidef_iff_eq_sum_vecMulVec]
  constructor
  · rintro ⟨D, features, h⟩
    refine ⟨D, features, ?_⟩
    simpa only [featureGram, Pi.star_def, star_trivial] using h
  · rintro ⟨D, features, h⟩
    refine ⟨D, features, ?_⟩
    simpa only [featureGram, Pi.star_def, star_trivial] using h

/-- An actual embedding table always supplies a positive semidefinite Gram.
Source: the derived content-score lifting, not a claim of convexity in
the original feature coordinates. -/
theorem featureGram_posSemidef {ι : Type*} [Finite ι] {D : ℕ}
    (features : Fin D → ι → ℝ) : (featureGram features).PosSemidef :=
  (posSemidef_iff_featureGram _).2 ⟨D, features, rfl⟩

/-- PSD and linear entry bounds define the trainable embedding domain.
Source: a new upstream score constraint for §2.2 of arXiv:1602.02068v2.
All diagonal and cross entries remain trainable. -/
def embeddingGramDomain (V : ℕ) (cap : ℝ) : Set (EmbeddingGram V) :=
  {G | G.PosSemidef ∧ ∀ i j, -cap ≤ G i j ∧ G i j ≤ cap}

/-- The entire bounded embedding domain is convex, including both Q/K
blocks. Source: the derived Gram architecture; no rank constraint is
included, since such a constraint would remove this guarantee. -/
theorem embeddingGramDomain_convex (V : ℕ) (cap : ℝ) :
    Convex ℝ (embeddingGramDomain V cap) := by
  intro G hG H hH a b ha hb hab
  refine ⟨(hG.1.smul ha).add (hH.1.smul hb), ?_⟩
  intro i j
  change -cap ≤ a * G i j + b * H i j ∧ a * G i j + b * H i j ≤ cap
  have hgl := mul_le_mul_of_nonneg_left (hG.2 i j).1 ha
  have hhl := mul_le_mul_of_nonneg_left (hH.2 i j).1 hb
  have hgu := mul_le_mul_of_nonneg_left (hG.2 i j).2 ha
  have hhu := mul_le_mul_of_nonneg_left (hH.2 i j).2 hb
  constructor <;> nlinarith

/-- A nonnegative cap gives an inhabited convex domain.
Source: the derived upstream constraint for sparsemax §2.2. -/
theorem zero_mem_embeddingGramDomain (V : ℕ) (cap : ℝ) (hc : 0 ≤ cap) :
    (0 : EmbeddingGram V) ∈ embeddingGramDomain V cap := by
  refine ⟨Matrix.PosSemidef.zero, ?_⟩
  intro i j
  change -cap ≤ 0 ∧ (0 : ℝ) ≤ cap
  constructor <;> linarith

/-- A positive cap inhabits the bounded embedding hypotheses. -/
example : (0 : EmbeddingGram 3) ∈ embeddingGramDomain 3 (3 / 8) :=
  zero_mem_embeddingGramDomain _ _ (by norm_num)

/-- The diagonal entries are genuine nonnegative embedding squared norms.
Source: the PSD condition in the derived content-score Gram architecture. -/
theorem embeddingGramDomain_diagonal {V : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (hG : G ∈ embeddingGramDomain V cap) (i : Sum (Fin V) (Fin V)) :
    0 ≤ G i i ∧ G i i ≤ cap := by
  exact ⟨hG.1.diag_nonneg, (hG.2 i i).2⟩

/-- A feasible learned matrix supplies an inhabited diagonal bound. -/
example : 0 ≤ (0 : EmbeddingGram 2) (Sum.inl 0) (Sum.inl 0) ∧
    (0 : EmbeddingGram 2) (Sum.inl 0) (Sum.inl 0) ≤ (3 / 8 : ℝ) :=
  embeddingGramDomain_diagonal _ _
    (zero_mem_embeddingGramDomain _ _ (by norm_num)) _

/-- Every feasible learned matrix has genuine finite embedding coordinates.
Source: the derived Gram lift; this is exact recovery without a prescribed
small dimension, rather than an assumption about the learned matrix. -/
theorem embeddingGramDomain_recovery {V : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (hG : G ∈ embeddingGramDomain V cap) :
    ∃ D : ℕ, ∃ features : Fin D → Sum (Fin V) (Fin V) → ℝ,
      ∀ i j, G i j = ∑ d, features d i * features d j := by
  obtain ⟨D, features, hf⟩ := (posSemidef_iff_featureGram G).1 hG.1
  refine ⟨D, features, ?_⟩
  intro i j
  rw [hf, featureGram_apply]

/-- Recovery has a concrete feasible instance without supplying features
as an unproved input. -/
example : ∃ D : ℕ, ∃ features : Fin D → Sum (Fin 2) (Fin 2) → ℝ,
    ∀ i j, (0 : EmbeddingGram 2) i j = ∑ d, features d i * features d j :=
  embeddingGramDomain_recovery _ _
    (zero_mem_embeddingGramDomain 2 (3 / 8) (by norm_num))

/-- Recovered embedding columns satisfy the squared norm cap.
Source: the diagonal part of the derived convex Gram constraint. -/
theorem recovered_embedding_sq_bound {V D : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (features : Fin D → Sum (Fin V) (Fin V) → ℝ)
    (hG : G ∈ embeddingGramDomain V cap) (hf : G = featureGram features)
    (i : Sum (Fin V) (Fin V)) : (∑ d, (features d i) ^ 2) ≤ cap := by
  have hb := (hG.2 i i).2
  rw [hf, featureGram_apply] at hb
  simpa only [pow_two] using hb

/-- Explicit zero features satisfy both recovery and norm-bound premises. -/
example : (∑ d : Fin 1, ((fun _ : Fin 1 => fun _ : Sum (Fin 2) (Fin 2) => (0 : ℝ))
    d (Sum.inl (0 : Fin 2))) ^ 2) ≤
    (3 / 8 : ℝ) := by
  have hf : (0 : EmbeddingGram 2) = featureGram (fun _ : Fin 1 => fun _ => 0) := by
    ext i j
    norm_num [featureGram_apply]
  exact recovered_embedding_sq_bound (3 / 8) (0 : EmbeddingGram 2)
    (fun _ : Fin 1 => fun _ => 0)
    (zero_mem_embeddingGramDomain _ _ (by norm_num)) hf (Sum.inl (0 : Fin 2))

end Transformer.GPTMini.Sparsemax
