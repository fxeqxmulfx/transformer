import Transformer.GPTMini.Sparsemax.NormalizedGram

/-!
# Convex constraints guaranteeing invertible causal attention

Derived architecture for sparsemax arXiv:1602.02068v2, Eq. (1), and the
value mixture `attn @ v` at `73f8a0b`. One context consists of distinct
token identities `Fin T`; every causal row uses the same learned Gram.
Add a positive lower bound on each self-attention diagonal. These are
linear constraints on the normalized Gram scores, so the domain stays
convex. The actual sparsemax matrix is lower triangular with positive
determinant, even though its off-diagonal supports may change.

This provides a structural guarantee rather than a fixed attention matrix.
It does not cover every collection of contexts sharing a smaller token
vocabulary, and it does not prove a uniform inverse norm bound. Earlier
sub-half entry caps cannot normalize the first causal row; the present
domain therefore leaves the cap explicit and permits that forced self row.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- The same distinct-token context is used by all of its causal queries.
Source: the derived square attention architecture for `attn @ v`. -/
def causalGramTokens (T : ℕ) : Fin T → Fin T → Fin T := fun _ j => j

/-- Every causal row is included once, with its ordinary position index.
Source: the repository's lower triangular attention mask at `73f8a0b`. -/
def causalGramRows (T : ℕ) : Fin T → Fin T := fun i => i

/-- The actual variational sparsemax attention matrix, not its score surrogate.
Source: sparsemax Eq. (1), applied to all rows of the shared Gram context. -/
def causalGramAttention {T : ℕ} (G : EmbeddingGram T) : Matrix (Fin T) (Fin T) ℝ :=
  Matrix.of (embeddingRoutes G (causalGramTokens T) (causalGramRows T))

/-- The actual learned attention always has the required triangular zeros.
Source: the causal sparsemax specialization of Eq. (1). -/
theorem causalGramAttention_lowerTriangular {T : ℕ} (G : EmbeddingGram T) :
    (causalGramAttention G).IsLowerTriangular := by
  intro i j hij
  exact embeddingRoutes_zero_above G _ _ i j hij

/-- Every actual matrix row remains a normalized causal probability vector.
Source: sparsemax Eq. (1), without assuming normalized Gram scores. -/
theorem causalGramAttention_simplex {T : ℕ} (G : EmbeddingGram T) (i : Fin T) :
    causalGramAttention G i ∈ simplexOn {j : Fin T | j ≤ i} :=
  embeddingRoutes_simplex G _ _ i

/-- In the normalized domain the true attention equals the shared cross Gram.
Source: the new linear restriction before sparsemax Eq. (1). -/
theorem causalGramAttention_normalized {T : ℕ} (cap : ℝ) (G : EmbeddingGram T)
    (hG : G ∈ normalizedGramDomain T cap (causalGramTokens T) (causalGramRows T)) :
    causalGramAttention G = Matrix.of (fun i j => G (Sum.inl i) (Sum.inr j)) := by
  exact normalizedGram_routes_eq_scores cap G _ _ hG

/-- A linear self-weight floor adds an invertibility guarantee.
Source: the new normalized-Gram restriction for the value mixture at
`73f8a0b`. Positivity of the floor is stated on the determinant theorem. -/
def invertibleGramDomain (T : ℕ) (cap floor : ℝ) : Set (EmbeddingGram T) :=
  {G | G ∈ normalizedGramDomain T cap (causalGramTokens T) (causalGramRows T) ∧
    ∀ i, floor ≤ embeddingScores G (fun j => j) i i}

/-- The structural invertibility domain is convex in both embedding families.
Source: the derived linear self-weight restriction; no determinant inequality
or rank assumption is hidden in the definition. -/
theorem invertibleGramDomain_convex (T : ℕ) (cap floor : ℝ) :
    Convex ℝ (invertibleGramDomain T cap floor) := by
  intro G hG H hH a b ha hb hab
  refine ⟨normalizedGramDomain_convex T cap _ _ hG.1 hH.1 ha hb hab, ?_⟩
  intro i
  change floor ≤ a * embeddingScores G (fun j => j) i i +
    b * embeddingScores H (fun j => j) i i
  have hg := mul_le_mul_of_nonneg_left (hG.2 i) ha
  have hh := mul_le_mul_of_nonneg_left (hH.2 i) hb
  calc
    floor = a * floor + b * floor := by rw [← add_mul, hab, one_mul]
    _ ≤ _ := add_le_add hg hh

/-- All feasible actual attention diagonals inherit the stated positive floor.
Source: the normalized-Gram constraint, used before recovering learned values. -/
theorem causalGramAttention_diag_floor {T : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hG : G ∈ invertibleGramDomain T cap floor) (i : Fin T) :
    floor ≤ causalGramAttention G i i := by
  rw [causalGramAttention_normalized cap G hG.1]
  exact hG.2 i

/-- Causality and the floor imply a positive determinant of the actual matrix.
Source: the derived invertibility guarantee for the original `attn @ v`.
No sparse support is fixed; only the triangular mask and diagonal floor remain. -/
theorem causalGramAttention_det_pos {T : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hf : 0 < floor) (hG : G ∈ invertibleGramDomain T cap floor) :
    0 < (causalGramAttention G).det := by
  rw [Matrix.det_of_isLowerTriangular _ (causalGramAttention_lowerTriangular G)]
  exact Finset.prod_pos fun i _ =>
    lt_of_lt_of_le hf (causalGramAttention_diag_floor cap floor G hG i)

/-- A concrete embedding matrix inhabits the normalization and floor constraints.
Source: the one-token sparsemax fixed point, with self weight one. -/
theorem normalizedGramUnit_mem_invertible :
    normalizedGramUnit ∈ invertibleGramDomain 1 1 (1 / 2) := by
  refine ⟨⟨normalizedGramUnit_mem_domain.1, ?_⟩, ?_⟩
  · intro i
    fin_cases i
    exact normalizedGramUnit_mem_domain.2 0
  · intro i
    norm_num [embeddingScores, normalizedGramUnit, Matrix.vecMulVec_apply]

/-- Actual nonzero embeddings inhabit the normalized-matrix premise. -/
example : causalGramAttention normalizedGramUnit =
    Matrix.of (fun i j : Fin 1 => normalizedGramUnit (Sum.inl i) (Sum.inr j)) :=
  causalGramAttention_normalized 1 _ normalizedGramUnit_mem_invertible.1

/-- The diagonal-floor hypothesis has a concrete positive instance. -/
example : (1 / 2 : ℝ) ≤ causalGramAttention normalizedGramUnit 0 0 :=
  causalGramAttention_diag_floor 1 _ _ normalizedGramUnit_mem_invertible _

/-- Both positive-floor and feasibility hypotheses are jointly satisfied. -/
example : 0 < (causalGramAttention normalizedGramUnit).det :=
  causalGramAttention_det_pos 1 _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- The actual attention is nonsingular for every feasible learned Gram.
Source: the structural guarantee permitting exact learned-value recovery. -/
theorem causalGramAttention_det_unit {T : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hf : 0 < floor) (hG : G ∈ invertibleGramDomain T cap floor) :
    IsUnit (causalGramAttention G).det := by
  apply isUnit_iff_ne_zero.mpr
  exact ne_of_gt (causalGramAttention_det_pos cap floor G hf hG)

/-- Nonsingularity is witnessed by a real normalized embedding matrix. -/
example : IsUnit (causalGramAttention normalizedGramUnit).det :=
  causalGramAttention_det_unit 1 _ _ (by norm_num) normalizedGramUnit_mem_invertible

/-- A feasible diagonal floor cannot exceed the embedding entry cap.
Source: the derived linear constraints for the exact learned-value model;
the explicit index ensures this necessary condition is nonvacuous. -/
theorem invertibleGramDomain_floor_le_cap {T : ℕ} (cap floor : ℝ) (G : EmbeddingGram T)
    (hG : G ∈ invertibleGramDomain T cap floor) (i : Fin T) : floor ≤ cap := by
  have hl := hG.2 i
  have hu := (hG.1.1.2 (Sum.inl i) (Sum.inr i)).2
  exact le_trans hl hu

/-- A concrete learned Gram witnesses the floor/cap compatibility premise. -/
example : (1 / 2 : ℝ) ≤ 1 :=
  invertibleGramDomain_floor_le_cap 1 _ _ normalizedGramUnit_mem_invertible (0 : Fin 1)

end Transformer.GPTMini.Sparsemax
