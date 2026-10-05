import Transformer.GPTMini.Sparsemax.GramSupport

/-!
# An exact convex attention restriction with trainable embeddings

Derived architecture from arXiv:1602.02068v2, Eq. (1) and Proposition 1.
Require each selected Gram score row itself to lie in its causal simplex.
These are linear nonnegativity, unit-mass and future-zero constraints on
one learned PSD matrix. Sparsemax then returns that row exactly.

Consequently the exact embedding/attention graph is convex on this new
domain, even when positive supports change. No Q/K embedding coordinates
or supports are frozen. This is a restricted architecture, not an exact
reformulation of unrestricted QKNorm attention. The constraints apply to
the specified finite contexts, and feasibility for arbitrary context
families is not asserted. Small fixed-width recovery and trainable value
mixtures are separate questions. No task loss or FFN is defined here.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Sparsemax fixes every row already in its causal simplex.
Source: sparsemax Eq. (1) and Proposition 1, specialized to threshold zero. -/
theorem sparseWeights_eq_self_of_simplex {T : ℕ} (scores : Fin T → ℝ) (i : Fin T)
    (hs : scores ∈ simplexOn {j : Fin T | j ≤ i}) : sparseWeights scores i = scores := by
  have ht : thresholdWeights scores i 0 = scores := by
    funext j
    by_cases hj : j ≤ i
    · simp only [thresholdWeights, hj, ite_true, sub_zero, max_eq_left (hs.1 j)]
    · simp only [thresholdWeights, hj, ite_false, hs.2.2 j hj]
  have hsum : ∑ j, thresholdWeights scores i 0 j = 1 := by
    rw [ht]
    exact hs.2.1
  rw [← thresholdWeights_eq_sparseWeights scores i 0 hsum]
  exact ht

/-- A sparse basis row inhabits the fixed-point hypothesis. -/
example : sparseWeights (basis (0 : Fin 2)) 1 = basis 0 :=
  sparseWeights_eq_self_of_simplex _ _ (basis_mem_simplex _ _ (by decide))

/-- PSD embedding Grams whose content scores are already causal probabilities.
Source: a new linear normalization restriction before sparsemax Eq. (1),
replacing the earlier assumption that the sparse support stays fixed. -/
def normalizedGramDomain (V : ℕ) {R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) : Set (EmbeddingGram V) :=
  {G | G ∈ embeddingGramDomain V cap ∧
    ∀ r, embeddingScores G (tokens r) (rows r) ∈ simplexOn {j : Fin T | j ≤ rows r}}

/-- The normalized embedding domain is convex with all Q/K blocks variable.
Source: the derived simplex constraints on Gram scores for sparsemax Eq. (1).
No equality of endpoint support sets is required. -/
theorem normalizedGramDomain_convex (V : ℕ) {R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Convex ℝ (normalizedGramDomain V cap tokens rows) := by
  intro G hG H hH a b ha hb hab
  refine ⟨embeddingGramDomain_convex V cap hG.1 hH.1 ha hb hab, ?_⟩
  intro r
  rw [embeddingScores_affine]
  exact simplexOn_convex _ (hG.2 r) (hH.2 r) ha hb hab

/-- Exact attention becomes linear in the learned Gram on this domain.
Source: the derived normalization constraint and sparsemax Eq. (1).
The operator is the original variational sparsemax, not a renamed score row. -/
theorem normalizedGram_routes_eq_scores {V R T : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (hG : G ∈ normalizedGramDomain V cap tokens rows) :
    embeddingRoutes G tokens rows = fun r => embeddingScores G (tokens r) (rows r) := by
  funext r
  exact sparseWeights_eq_self_of_simplex _ _ (hG.2 r)

/-- An exact graph, retaining the actual sparsemax equation on learned variables.
Source: the derived normalized-Gram architecture before sparsemax Eq. (1). -/
def normalizedGramRoutingGraph (V : ℕ) {R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Set (EmbeddingGram V × (Fin R → Fin T → ℝ)) :=
  {p | p.1 ∈ normalizedGramDomain V cap tokens rows ∧
    p.2 = embeddingRoutes p.1 tokens rows}

/-- The exact embedding/attention graph is convex under the new linear
constraints. Source: the derived normalized-Gram restriction of Eq. (1).
Supports may acquire or lose exact zeros along these convex segments. -/
theorem normalizedGramRoutingGraph_convex (V : ℕ) {R T : ℕ} (cap : ℝ)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) :
    Convex ℝ (normalizedGramRoutingGraph V cap tokens rows) := by
  intro p hp q hq a b ha hb hab
  have hm := normalizedGramDomain_convex V cap tokens rows hp.1 hq.1 ha hb hab
  refine ⟨hm, ?_⟩
  change a • p.2 + b • q.2 = embeddingRoutes (a • p.1 + b • q.1) tokens rows
  rw [hp.2, hq.2, normalizedGram_routes_eq_scores cap _ _ _ hp.1,
    normalizedGram_routes_eq_scores cap _ _ _ hq.1,
    normalizedGram_routes_eq_scores cap _ _ _ hm]
  funext r
  exact (embeddingScores_affine _ _ _ _ _ _).symm

/-- A real one-feature Gram with a unit content score.
Source: an inhabited witness for the normalized-Gram restriction. -/
def normalizedGramUnit : EmbeddingGram 1 :=
  Matrix.vecMulVec (fun _ => 1) (fun _ => 1)

/-- Normalization feasibility is proved for a concrete embedding matrix.
Source: the derived normalized-Gram architecture, with one visible token. -/
theorem normalizedGramUnit_mem_domain : normalizedGramUnit ∈
    normalizedGramDomain 1 1 (fun _ : Fin 1 => fun j : Fin 1 => j) (fun _ => 0) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · simpa only [normalizedGramUnit, star_trivial] using
      Matrix.posSemidef_vecMulVec_self_star (fun _ : Sum (Fin 1) (Fin 1) => (1 : ℝ))
  · intro i j
    norm_num [normalizedGramUnit, Matrix.vecMulVec_apply]
  · intro r
    refine ⟨?_, ?_, ?_⟩
    · intro j
      norm_num [embeddingScores, normalizedGramUnit, Matrix.vecMulVec_apply]
    · norm_num [embeddingScores, normalizedGramUnit, Matrix.vecMulVec_apply, Fin.sum_univ_one]
    · intro j hj
      fin_cases j
      exact (hj (by change (0 : Fin 1) ≤ 0; decide)).elim

/-- Concrete nonzero embeddings inhabit the exact-attention hypothesis. -/
example : embeddingRoutes normalizedGramUnit
    (fun _ : Fin 1 => fun j : Fin 1 => j) (fun _ => 0) =
      fun _ : Fin 1 => embeddingScores normalizedGramUnit (fun j : Fin 1 => j) (0 : Fin 1) :=
  normalizedGram_routes_eq_scores 1 _ _ _ normalizedGramUnit_mem_domain

/-- The actual attention inherits the Gram cap on every coordinate.
Source: the new normalization restriction of sparsemax Eq. (1), now a
proved bound on the operator's output rather than a frozen attention row. -/
theorem normalizedGram_weight_bounds {V R T : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (hG : G ∈ normalizedGramDomain V cap tokens rows) (r : Fin R) (j : Fin T) :
    0 ≤ embeddingRoutes G tokens rows r j ∧ embeddingRoutes G tokens rows r j ≤ cap := by
  rw [normalizedGram_routes_eq_scores cap _ _ _ hG]
  exact ⟨(hG.2 r).1 j, (hG.1.2 (Sum.inl (tokens r (rows r))) (Sum.inr (tokens r j))).2⟩

/-- The bound hypotheses are satisfied by a genuine normalized Gram. -/
example : 0 ≤ embeddingRoutes normalizedGramUnit
    (fun _ : Fin 1 => fun j : Fin 1 => j) (fun _ => 0) 0 0 ∧
    embeddingRoutes normalizedGramUnit
      (fun _ : Fin 1 => fun j : Fin 1 => j) (fun _ => 0) 0 0 ≤ 1 :=
  normalizedGram_weight_bounds 1 _ _ _ normalizedGramUnit_mem_domain _ _

/-- Feasible normalized embeddings have genuine finite coordinates.
Source: the exact PSD Gram recovery, applied to the new Eq. (1) restriction.
It does not introduce a rank or feature-width bound. -/
theorem normalizedGram_recovery {V R T : ℕ} (cap : ℝ) (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (hG : G ∈ normalizedGramDomain V cap tokens rows) :
    ∃ D : ℕ, ∃ features : Fin D → Sum (Fin V) (Fin V) → ℝ,
      ∀ i j, G i j = ∑ d, features d i * features d j :=
  embeddingGramDomain_recovery cap G hG.1

/-- Recovery's extra normalization premises hold in an actual example. -/
example : ∃ D : ℕ, ∃ features : Fin D → Sum (Fin 1) (Fin 1) → ℝ,
    ∀ i j, normalizedGramUnit i j = ∑ d, features d i * features d j :=
  normalizedGram_recovery 1 _ _ _ normalizedGramUnit_mem_domain

end Transformer.GPTMini.Sparsemax
