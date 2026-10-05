import Transformer.GPTMini.Sparsemax.EmbeddingGram

/-!
# Shared content scores and free causal rows in Gram coordinates

Derived architecture preceding arXiv:1602.02068v2, §2.1–§2.2. One
bounded PSD matrix supplies every score in every finite context. Token
occurrences share its entries; rows are not given private Q/K parameters.
Both query and key embedding families can change through the Gram matrix.

The joint feasible domain is a product of the bounded embedding cone and
causal simplices. It is convex without prescribing sparsemax supports.
Token identities are fixed inputs; embedding coordinates remain trainable.
The ordinary sparsemax solution is feasible for every learned matrix.
Feasibility alone does not impose the equation defining that solution.
The next modules distinguish the convex projection energy from its exact
optimizer graph. Values, FFN and task loss remain outside this foundation.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Shared learned content scores selected by the tokens in a context.
Source: the derived Gram parameterization before sparsemax Eq. (1).
No normalization or learned multiplicative temperature is included. -/
def embeddingScores {V T : ℕ} (G : EmbeddingGram V) (tokens : Fin T → Fin V)
    (i : Fin T) : Fin T → ℝ :=
  fun j => G (Sum.inl (tokens i)) (Sum.inr (tokens j))

/-- Content scores are affine when both embedding Gram blocks change.
Source: the new Gram architecture preceding arXiv:1602.02068v2, §2.2;
this is not convexity in the original bilinear feature coordinates. -/
theorem embeddingScores_affine {V T : ℕ} (G H : EmbeddingGram V)
    (tokens : Fin T → Fin V) (i : Fin T) (a b : ℝ) :
    embeddingScores (a • G + b • H) tokens i =
      a • embeddingScores G tokens i + b • embeddingScores H tokens i := by
  funext j
  simp only [embeddingScores, Matrix.add_apply, Matrix.smul_apply, Pi.add_apply,
    Pi.smul_apply, smul_eq_mul]

/-- Recovered Gram coordinates produce genuine Q/K scalar products.
Source: the derived content-score lifting of sparsemax Eq. (1).
The recovered query and key columns are distinct learned embeddings. -/
theorem embeddingScores_recovered {V T D : ℕ} (G : EmbeddingGram V)
    (features : Fin D → Sum (Fin V) (Fin V) → ℝ) (tokens : Fin T → Fin V) (i j : Fin T)
    (hf : G = featureGram features) :
    embeddingScores G tokens i j =
      ∑ d, features d (Sum.inl (tokens i)) * features d (Sum.inr (tokens j)) := by
  rw [hf]
  exact featureGram_apply _ _ _

/-- Concrete finite feature columns inhabit exact score recovery. -/
example : embeddingScores (featureGram (fun d : Fin 2 => fun _ => (d.val : ℝ)))
    (fun j : Fin 2 => j) 1 0 =
      ∑ d : Fin 2, (d.val : ℝ) * (d.val : ℝ) :=
  embeddingScores_recovered _ _ _ _ _ rfl

/-- Exact ordinary sparsemax rows for the shared learned embedding matrix.
Source: arXiv:1602.02068v2, Eq. (1), with the repository's causal mask. -/
def embeddingRoutes {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) : Fin R → Fin T → ℝ :=
  fun r => sparseWeights (embeddingScores G (tokens r) (rows r)) (rows r)

/-- All inferred rows belong to their causal simplex.
Source: sparsemax Eq. (1), including arbitrary learned Gram matrices. -/
theorem embeddingRoutes_simplex {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (r : Fin R) :
    embeddingRoutes G tokens rows r ∈ simplexOn {j : Fin T | j ≤ rows r} :=
  (sparseWeights_spec _ _).1

/-- Every row is normalized even though the embedding matrix is trainable.
Source: sparsemax Eq. (1), not an independently imposed row target. -/
theorem embeddingRoutes_row_sum {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (r : Fin R) :
    (∑ j, embeddingRoutes G tokens rows r j) = 1 :=
  (embeddingRoutes_simplex G tokens rows r).2.1

/-- Causal zeros are exact for every learned embedding assignment.
Source: the causal simplex specialization of sparsemax Eq. (1). -/
theorem embeddingRoutes_zero_above {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (r : Fin R) (j : Fin T)
    (hj : rows r < j) : embeddingRoutes G tokens rows r j = 0 :=
  (embeddingRoutes_simplex G tokens rows r).2.2 j (not_le.mpr hj)

/-- A two-position context inhabits the causal exclusion premise. -/
example : embeddingRoutes (0 : EmbeddingGram 2)
    (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 0) 0 1 = 0 :=
  embeddingRoutes_zero_above _ _ _ _ _ (by decide)

/-- Identical visible token codes have identical attention weights.
Source: shared content scores and sparsemax Proposition 1. Occurrence
positions must be included in token codes if they should be distinguished;
the new architecture does not retain the original RoPE transformation. -/
theorem embeddingRoutes_eq_of_token_eq {V R T : ℕ} (G : EmbeddingGram V)
    (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T) (r : Fin R) (j k : Fin T)
    (hj : j ≤ rows r) (hk : k ≤ rows r) (he : tokens r j = tokens r k) :
    embeddingRoutes G tokens rows r j = embeddingRoutes G tokens rows r k := by
  obtain ⟨τ, hτ⟩ := sparseWeights_exists_threshold (embeddingScores G (tokens r) (rows r))
    (rows r)
  change sparseWeights _ _ j = sparseWeights _ _ k
  rw [hτ]
  simp only [thresholdWeights, hj, hk, ite_true, embeddingScores, he]

/-- Repeated tokens at distinct visible positions satisfy these premises. -/
example : embeddingRoutes (0 : EmbeddingGram 1)
    (fun _ : Fin 1 => fun _ : Fin 2 => 0) (fun _ => 1) 0 0 =
      embeddingRoutes (0 : EmbeddingGram 1)
        (fun _ : Fin 1 => fun _ : Fin 2 => 0) (fun _ => 1) 0 1 :=
  embeddingRoutes_eq_of_token_eq _ _ _ _ _ _ (by decide) (by decide) rfl

/-- Embeddings and attention weights are both variables in the domain.
Source: the derived joint domain for sparsemax Eq. (1). No support,
query matrix or key matrix is supplied as a frozen parameter. -/
def gramRoutingDomain (V : ℕ) {R T : ℕ} (cap : ℝ) (rows : Fin R → Fin T) :
    Set (EmbeddingGram V × (Fin R → Fin T → ℝ)) :=
  {p | p.1 ∈ embeddingGramDomain V cap ∧
    ∀ r, p.2 r ∈ simplexOn {j : Fin T | j ≤ rows r}}

/-- The joint feasible domain is convex with unrestricted row supports.
Source: the derived embedding cone and causal simplex product for Eq. (1).
This theorem concerns feasibility, not the sparsemax optimizer graph. -/
theorem gramRoutingDomain_convex (V : ℕ) {R T : ℕ} (cap : ℝ) (rows : Fin R → Fin T) :
    Convex ℝ (gramRoutingDomain V cap rows) := by
  intro p hp q hq a b ha hb hab
  refine ⟨embeddingGramDomain_convex V cap hp.1 hq.1 ha hb hab, ?_⟩
  intro r
  change a • p.2 r + b • q.2 r ∈ simplexOn {j : Fin T | j ≤ rows r}
  exact simplexOn_convex _ (hp.2 r) (hq.2 r) ha hb hab

/-- Every feasible embedding Gram has feasible exact sparsemax rows.
Source: the derived joint domain, using sparsemax Eq. (1) without
assuming any target routes or support pattern. -/
theorem embeddingRoutes_mem_gramRoutingDomain {V R T : ℕ} (cap : ℝ)
    (G : EmbeddingGram V) (tokens : Fin R → Fin T → Fin V) (rows : Fin R → Fin T)
    (hG : G ∈ embeddingGramDomain V cap) :
    (G, embeddingRoutes G tokens rows) ∈ gramRoutingDomain V cap rows :=
  ⟨hG, fun r => embeddingRoutes_simplex G tokens rows r⟩

/-- A shared two-token matrix and its inferred rows satisfy all premises. -/
example : ((0 : EmbeddingGram 2), embeddingRoutes (0 : EmbeddingGram 2)
    (fun _ : Fin 1 => fun j : Fin 2 => j) (fun _ => 1)) ∈
      gramRoutingDomain 2 (3 / 8) (fun _ : Fin 1 => (1 : Fin 2)) :=
  embeddingRoutes_mem_gramRoutingDomain _ _ _ _
    (zero_mem_embeddingGramDomain _ _ (by norm_num))

end Transformer.GPTMini.Sparsemax
