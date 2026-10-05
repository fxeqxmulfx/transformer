import Transformer.GPTMini.Sparsemax.GramValues
import Mathlib.LinearAlgebra.Matrix.Gershgorin

/-!
# Convex learned memory with a structural inverse guarantee

Derived memory architecture for sparsemax arXiv:1602.02068v2, Eq. (1),
and `attn @ v` at `73f8a0b`. A finite dictionary supplies learned Q/K
embeddings and one shared value table. Every dictionary row sees every
memory slot. Normalize its Gram scores linearly and bound each diagonal
below by a floor strictly greater than one half. Gershgorin then proves
invertibility without a triangular support restriction or fixed Q/K.

This changes ordinary token self-attention into attention to a learned
dictionary. Its slots are parameters, not future observations. Context
queries will be data-dependent mixtures of the learned query embeddings.
The score constraints, width freedom and value-coordinate decoder are
explicit architectural choices; no task loss or FFN is selected.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Shared dictionary Q/K scores before the actual sparsemax projection.
Source: the derived memory version of sparsemax Eq. (1). -/
def memoryGramScores {N : ℕ} (G : EmbeddingGram (N + 1)) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  Matrix.of (fun i j => G (Sum.inl i) (Sum.inr j))

/-- Actual variational sparsemax, with every learned memory slot visible.
Source: Eq. (1); the last-slot mask allows the whole dictionary. -/
def memoryGramAttention {N : ℕ} (G : EmbeddingGram (N + 1)) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  Matrix.of (fun i => sparseWeights (memoryGramScores G i) (Fin.last N))

/-- Every actual memory attention row is a full-dictionary probability row.
Source: the feasible simplex in sparsemax Eq. (1), for arbitrary learned G. -/
theorem memoryGramAttention_simplex {N : ℕ} (G : EmbeddingGram (N + 1))
    (i : Fin (N + 1)) : memoryGramAttention G i ∈ simplexOn Set.univ := by
  have hs := (sparseWeights_spec (memoryGramScores G i) (Fin.last N)).1
  exact ⟨hs.1, hs.2.1, fun j hj => False.elim (hj (Set.mem_univ j))⟩

/-- Linear score normalization and diagonal bounds on the bounded PSD Gram.
Source: the new dictionary restriction before sparsemax Eq. (1).
The inverse guarantee requires a floor above one half, proved below. -/
def memoryGramDomain (N : ℕ) (cap floor : ℝ) : Set (EmbeddingGram (N + 1)) :=
  {G | G ∈ embeddingGramDomain (N + 1) cap ∧
    (∀ i, memoryGramScores G i ∈ simplexOn Set.univ) ∧
    ∀ i, floor ≤ memoryGramScores G i i}

/-- Both dictionary embedding families vary over a convex structural domain.
Source: the derived normalization and diagonal constraints for Eq. (1). -/
theorem memoryGramDomain_convex (N : ℕ) (cap floor : ℝ) :
    Convex ℝ (memoryGramDomain N cap floor) := by
  intro G hG H hH a b ha hb hab
  refine ⟨embeddingGramDomain_convex (N + 1) cap hG.1 hH.1 ha hb hab, ?_, ?_⟩
  · intro i
    change a • memoryGramScores G i + b • memoryGramScores H i ∈ simplexOn Set.univ
    exact simplexOn_convex _ (hG.2.1 i) (hH.2.1 i) ha hb hab
  intro i
  change floor ≤ a * memoryGramScores G i i + b * memoryGramScores H i i
  have hg := mul_le_mul_of_nonneg_left (hG.2.2 i) ha
  have hh := mul_le_mul_of_nonneg_left (hH.2.2 i) hb
  calc
    floor = a * floor + b * floor := by rw [← add_mul, hab, one_mul]
    _ ≤ _ := add_le_add hg hh

/-- Actual sparsemax equals the normalized learned cross Gram.
Source: Eq. (1) on the new dictionary domain, with no fixed support. -/
theorem memoryGramAttention_normalized {N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (hG : G ∈ memoryGramDomain N cap floor) :
    memoryGramAttention G = memoryGramScores G := by
  funext i
  have hs := hG.2.1 i
  apply sparseWeights_eq_self_of_simplex
  exact ⟨hs.1, hs.2.1, fun j hj => False.elim (hj (Fin.le_last j))⟩

/-- Actual attention inherits the structural self-weight floor.
Source: the normalized dictionary restriction for the value-coordinate inverse. -/
theorem memoryGramAttention_diag_floor {N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (hG : G ∈ memoryGramDomain N cap floor)
    (i : Fin (N + 1)) : floor ≤ memoryGramAttention G i i := by
  rw [memoryGramAttention_normalized cap floor G hG]
  exact hG.2.2 i

/-- A floor above one half gives strict row diagonal dominance.
Source: the new dictionary restriction; row mass one makes the remaining
absolute row mass strictly smaller than the diagonal. -/
theorem memoryGramAttention_strictDominance {N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) (i : Fin (N + 1)) :
    (∑ j ∈ Finset.univ.erase i, ‖memoryGramAttention G i j‖) <
      ‖memoryGramAttention G i i‖ := by
  have hs := memoryGramAttention_simplex G i
  have hn (j) : ‖memoryGramAttention G i j‖ = memoryGramAttention G i j :=
    Real.norm_of_nonneg (hs.1 j)
  simp_rw [hn]
  have hsum := Finset.sum_erase_add Finset.univ (memoryGramAttention G i) (Finset.mem_univ i)
  rw [hs.2.1] at hsum
  have hd := memoryGramAttention_diag_floor cap floor G hG i
  linarith

/-- The learned memory is nonsingular without fixing either directional support.
Source: Gershgorin's strict-dominance criterion, applied to actual Eq. (1). -/
theorem memoryGramAttention_det_unit {N : ℕ} (cap floor : ℝ)
    (G : EmbeddingGram (N + 1)) (hf : 1 / 2 < floor)
    (hG : G ∈ memoryGramDomain N cap floor) : IsUnit (memoryGramAttention G).det := by
  apply isUnit_iff_ne_zero.mpr
  exact det_ne_zero_of_sum_row_lt_diag (memoryGramAttention_strictDominance cap floor G hf hG)

/-- An actual nonzero one-slot Gram inhabits the new structural domain.
Source: the derived dictionary architecture with normalized score one. -/
theorem normalizedGramUnit_mem_memory : normalizedGramUnit ∈ memoryGramDomain 0 1 (3 / 4) := by
  refine ⟨normalizedGramUnit_mem_domain.1, ?_, ?_⟩
  · intro i
    fin_cases i
    have hs := normalizedGramUnit_mem_domain.2 0
    exact ⟨hs.1, hs.2.1, fun j hj => False.elim (hj (Set.mem_univ j))⟩
  · intro i
    norm_num [memoryGramScores, normalizedGramUnit, Matrix.vecMulVec_apply]

/-- The exact-attention premises are jointly inhabited. -/
example : memoryGramAttention normalizedGramUnit = memoryGramScores normalizedGramUnit :=
  memoryGramAttention_normalized 1 _ _ normalizedGramUnit_mem_memory

/-- A positive concrete diagonal satisfies the floor premise. -/
example : (3 / 4 : ℝ) ≤ memoryGramAttention normalizedGramUnit 0 0 :=
  memoryGramAttention_diag_floor 1 _ _ normalizedGramUnit_mem_memory _

/-- Actual learned scores satisfy the strict-dominance hypotheses. -/
example : (∑ j ∈ Finset.univ.erase (0 : Fin 1), ‖memoryGramAttention normalizedGramUnit 0 j‖) <
    ‖memoryGramAttention normalizedGramUnit 0 0‖ :=
  memoryGramAttention_strictDominance 1 _ _ (by norm_num) normalizedGramUnit_mem_memory _

/-- The inverse theorem has a real nonempty structural instance. -/
example : IsUnit (memoryGramAttention normalizedGramUnit).det :=
  memoryGramAttention_det_unit 1 _ _ (by norm_num) normalizedGramUnit_mem_memory

/-- Actual memory attention is affine as the shared embedding Gram changes.
Source: Eq. (1) under the convex dictionary normalization constraints. -/
theorem memoryGramAttention_affine {N : ℕ} (cap floor : ℝ)
    (G H : EmbeddingGram (N + 1)) (hG : G ∈ memoryGramDomain N cap floor)
    (hH : H ∈ memoryGramDomain N cap floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    memoryGramAttention (a • G + b • H) =
      a • memoryGramAttention G + b • memoryGramAttention H := by
  have hm := memoryGramDomain_convex N cap floor hG hH ha hb hab
  rw [memoryGramAttention_normalized cap floor _ hm,
    memoryGramAttention_normalized cap floor _ hG,
    memoryGramAttention_normalized cap floor _ hH]
  ext i j
  simp only [memoryGramScores, Matrix.of_apply, Matrix.add_apply, Matrix.smul_apply]

/-- The affinity premises are satisfied inside the structural domain. -/
example : memoryGramAttention ((1 / 2 : ℝ) • normalizedGramUnit + (1 / 2 : ℝ) • normalizedGramUnit) =
    (1 / 2 : ℝ) • memoryGramAttention normalizedGramUnit +
      (1 / 2 : ℝ) • memoryGramAttention normalizedGramUnit :=
  memoryGramAttention_affine 1 _ _ _ normalizedGramUnit_mem_memory normalizedGramUnit_mem_memory
    _ _ (by norm_num) (by norm_num) (by norm_num)

end Transformer.GPTMini.Sparsemax
