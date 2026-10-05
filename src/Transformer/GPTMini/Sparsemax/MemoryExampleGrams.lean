import Transformer.GPTMini.Sparsemax.SharedMemoryGeometry

/-!
# Inhabited memory domains and changing two-way sparse supports

Derived embedding witnesses for sparsemax arXiv:1602.02068v2, Eq. (1),
and the common-memory value product at `73f8a0b`. Every dictionary size
has a genuine identity Gram in the structural inverse domain. The two-slot
example changes both query and key feature norms, while actual attention
changes from identity to `[[3/4,1/4],[1/4,3/4]]`. Supports change in both
directions, with no triangular mask or frozen memory embeddings.

All Grams come from explicit feature tables. Fixed small width is not
claimed for their convex interpolants. Memory values are common to all
data contexts and are decoded in the next concrete witness. This is an
architectural feasibility result, not a new training experiment or task loss.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Identity Q/K features for any positive dictionary size.
Source: the derived memory Gram witness preceding Eq. (1). -/
def memoryIdentityFeatures (N : ℕ) (d : Fin (N + 1)) : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ
  | Sum.inl i => if d = i then 1 else 0
  | Sum.inr j => if d = j then 1 else 0

/-- Genuine shared identity embedding Gram, without assuming a factorization.
Source: the derived finite-coordinate memory architecture. -/
def memoryIdentityGram (N : ℕ) : EmbeddingGram (N + 1) := featureGram (memoryIdentityFeatures N)

/-- Identity Gram entries are Kronecker products of the explicit feature columns.
Source: the derived witness; this computes the feature sum, rather than
defining a desired attention matrix as a recovered embedding. -/
theorem memoryIdentityGram_entries (N : ℕ) (i j : Sum (Fin (N + 1)) (Fin (N + 1))) :
    memoryIdentityGram N i j = if Sum.elim id id i = Sum.elim id id j then 1 else 0 := by
  unfold memoryIdentityGram
  rw [featureGram_apply]
  rcases i with i | i <;> rcases j with j | j <;>
    change (∑ d : Fin (N + 1), (if d = i then (1 : ℝ) else 0) *
      (if d = j then 1 else 0)) = if i = j then 1 else 0
  all_goals
    simp only [ite_mul, one_mul, zero_mul]
    exact Fintype.sum_ite_eq' i _

/-- Actual identity feature products give the identity dictionary scores.
Source: the genuine Q/K products preceding sparsemax Eq. (1). -/
theorem memoryIdentityGram_scores (N : ℕ) : memoryGramScores (memoryIdentityGram N) = 1 := by
  ext i j
  change memoryIdentityGram N (Sum.inl i) (Sum.inr j) = (1 : Matrix _ _ ℝ) i j
  rw [memoryIdentityGram_entries, Matrix.one_apply, Sum.elim_inl, Sum.elim_inr]
  rfl

/-- Every positive dictionary size has a feasible bounded identity memory.
Source: the new structural domain; cap at least one and floor at most one
are sufficient for this concrete feature witness. -/
theorem memoryIdentityGram_mem (N : ℕ) (cap floor : ℝ) (hc : 1 ≤ cap) (hf : floor ≤ 1) :
    memoryIdentityGram N ∈ memoryGramDomain N cap floor := by
  refine ⟨⟨featureGram_posSemidef _, ?_⟩, ?_, ?_⟩
  · intro i j
    rw [memoryIdentityGram_entries]
    split_ifs <;> constructor <;> linarith
  · intro i
    rw [memoryIdentityGram_scores]
    have he : (1 : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) i = basis i := by
      funext j
      simp only [Matrix.one_apply, basis, eq_comm]
    rw [he]
    exact basis_mem_simplex Set.univ i (Set.mem_univ i)
  · intro i
    rw [memoryIdentityGram_scores]
    simpa only [Matrix.one_apply, ite_true] using hf

/-- A nonzero two-slot dictionary jointly satisfies both bound hypotheses. -/
example : memoryIdentityGram 1 ∈ memoryGramDomain 1 4 (3 / 4) :=
  memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num)

/-- Arbitrary dictionary size has a nonempty strict-dominance memory domain.
Source: the actual identity feature witness, with cap one and floor three quarters. -/
theorem memoryGramDomain_nonempty (N : ℕ) : (memoryGramDomain N 1 (3 / 4)).Nonempty := by
  exact ⟨memoryIdentityGram N, memoryIdentityGram_mem N _ _ (by norm_num) (by norm_num)⟩

/-- Initial actual two-slot memory embeddings.
Source: the derived shared-memory joint-training witness. -/
def memoryExampleStartGram : EmbeddingGram 2 := memoryIdentityGram 1

/-- Changed query/key embeddings with full, diagonally dominant memory attention.
Source: the derived two-slot witness; Q coordinates double and K columns mix. -/
def memoryExampleStopFeatures (d : Fin 2) : Sum (Fin 2) (Fin 2) → ℝ
  | Sum.inl i => if d = i then 2 else 0
  | Sum.inr j => if d = j then 3 / 8 else 1 / 8

/-- Genuine changed embedding Gram shared by all contexts.
Source: the finite feature witness for the normalized memory architecture. -/
def memoryExampleStopGram : EmbeddingGram 2 := featureGram memoryExampleStopFeatures

/-- The initial memory is feasible at the same cap and floor as the changed one.
Source: the actual identity feature witness in the shared-memory architecture. -/
theorem memoryExampleStartGram_mem : memoryExampleStartGram ∈ memoryGramDomain 1 4 (3 / 4) :=
  memoryIdentityGram_mem _ _ _ (by norm_num) (by norm_num)

/-- Changed memory scores are exact genuine Q/K scalar products.
Source: the two-slot feature witness preceding Eq. (1). -/
theorem memoryExampleStopGram_scores : memoryGramScores memoryExampleStopGram =
    Matrix.of (fun i j => if i = j then 3 / 4 else 1 / 4) := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [memoryGramScores, memoryExampleStopGram,
    featureGram_apply, memoryExampleStopFeatures, Fin.sum_univ_two]

/-- Changed actual embeddings satisfy the same convex PSD, normalization and floor constraints.
Source: the derived cap-four and floor-three-quarters dictionary restriction. -/
theorem memoryExampleStopGram_mem : memoryExampleStopGram ∈ memoryGramDomain 1 4 (3 / 4) := by
  refine ⟨⟨featureGram_posSemidef _, ?_⟩, ?_, ?_⟩
  · intro i j
    rcases i with i | i <;> rcases j with j | j <;> fin_cases i <;> fin_cases j <;>
      norm_num [memoryExampleStopGram, featureGram_apply, memoryExampleStopFeatures, Fin.sum_univ_two]
  · intro i
    rw [memoryExampleStopGram_scores]
    refine ⟨?_, ?_, fun j hj => False.elim (hj (Set.mem_univ j))⟩
    · intro j
      fin_cases i <;> fin_cases j <;> norm_num
    · change (∑ j : Fin 2, if i = j then (3 / 4 : ℝ) else 1 / 4) = 1
      fin_cases i <;> norm_num [Fin.sum_univ_two]
  · intro i
    rw [memoryExampleStopGram_scores]
    norm_num

/-- Actual initial variational sparsemax memory attention is identity.
Source: Eq. (1) on the proved normalized identity Gram. -/
theorem memoryExampleStartGram_attention : memoryGramAttention memoryExampleStartGram = 1 := by
  rw [memoryGramAttention_normalized 4 (3 / 4) _ memoryExampleStartGram_mem]
  exact memoryIdentityGram_scores 1

/-- Actual changed sparsemax memory mixes in both directions.
Source: Eq. (1) on the proved normalized changed Gram. -/
theorem memoryExampleStopGram_attention : memoryGramAttention memoryExampleStopGram =
    Matrix.of (fun i j => if i = j then 3 / 4 else 1 / 4) := by
  rw [memoryGramAttention_normalized 4 (3 / 4) _ memoryExampleStopGram_mem]
  exact memoryExampleStopGram_scores

/-- Both learned embedding families change their genuine squared norms.
Source: the two explicit Q/K feature tables in the shared-memory witness. -/
theorem memoryExample_embedding_norms_change :
    memoryExampleStartGram (Sum.inl 0) (Sum.inl 0) ≠ memoryExampleStopGram (Sum.inl 0) (Sum.inl 0) ∧
    memoryExampleStartGram (Sum.inr 0) (Sum.inr 0) ≠ memoryExampleStopGram (Sum.inr 0) (Sum.inr 0) := by
  norm_num [memoryExampleStartGram, memoryIdentityGram_entries, memoryExampleStopGram,
    featureGram_apply, memoryExampleStopFeatures, Fin.sum_univ_two]

/-- The common learned memory acquires both off-diagonal sparse support entries.
Source: actual variational sparsemax, without a triangular support restriction. -/
theorem memoryExample_support_changes :
    memoryGramAttention memoryExampleStartGram 0 1 = 0 ∧
    memoryGramAttention memoryExampleStartGram 1 0 = 0 ∧
    0 < memoryGramAttention memoryExampleStopGram 0 1 ∧
    0 < memoryGramAttention memoryExampleStopGram 1 0 := by
  rw [memoryExampleStartGram_attention, memoryExampleStopGram_attention]
  norm_num

/-- Actual changed memory has determinant one half, certifying unique common values.
Source: the concrete strict-dominance inverse guarantee after Eq. (1). -/
theorem memoryExampleStopGram_det : (memoryGramAttention memoryExampleStopGram).det = 1 / 2 := by
  rw [Matrix.det_fin_two, memoryExampleStopGram_attention]
  norm_num

end Transformer.GPTMini.Sparsemax
