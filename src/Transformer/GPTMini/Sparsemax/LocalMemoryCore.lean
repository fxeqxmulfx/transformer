import Transformer.GPTMini.Sparsemax.LocalMemoryWeights
import Mathlib.Analysis.Convex.Combination

/-!
# An affine compact Gram with changing local memory support

Derived architecture before arXiv:1602.02068v2, Eq. (1). Mix the genuine
identity and adjacent-swap embedding Grams using the learned edge weights
and residual identity mass. Nonnegative weights summing to one preserve
the bounded PSD and probability-score domain. Identity mass at least the
floor gives the same lower bound on every actual self-weight.

The Gram is affine in N scalar edge coordinates, with no spectral or rank
constraint. Its query and key same-family blocks are identity; a subsequent
diagonal addition makes their squared norms independently trainable. This
fixed orthogonality is an explicit restriction of the new path architecture.
The cross block and actual sparse supports remain variable. Common values
are still decoded globally by the earlier structural memory inverse.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A learned affine combination of actual finite embedding Grams.
Source: the compact memory restriction before arXiv:1602.02068v2, Eq. (1). -/
def localMemoryCore {N : ℕ} (t : Fin N → ℝ) : EmbeddingGram (N + 1) :=
  ∑ e, localMemoryWeights t e • localMemoryAtom e

/-- Core entries are the weighted products of genuine atom feature columns.
Source: the derived Gram mixture preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_apply {N : ℕ} (t : Fin N → ℝ)
    (x y : Sum (Fin (N + 1)) (Fin (N + 1))) :
    localMemoryCore t x y = ∑ e, localMemoryWeights t e * localMemoryAtom e x y := by
  rw [localMemoryCore, Matrix.sum_apply]
  exact Finset.sum_congr rfl fun e he => rfl

/-- All core cross scores are learned mixtures of path permutations.
Source: actual embedding products before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_scores_apply {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1)) :
    memoryGramScores (localMemoryCore t) i j =
      ∑ e, localMemoryWeights t e * (if localMemoryPermutation e i = j then 1 else 0) := by
  change localMemoryCore t (Sum.inl i) (Sum.inr j) = _
  rw [localMemoryCore_apply]
  apply Finset.sum_congr rfl
  intro e he
  have h := congrFun (congrFun (permutationMemoryGram_scores (localMemoryPermutation e)) i) j
  change localMemoryAtom e (Sum.inl i) (Sum.inr j) =
    (if localMemoryPermutation e i = j then 1 else 0) at h
  rw [h]

/-- The nonnegative atom mixture is a bounded PSD memory with probability scores.
Source: the derived convex Gram construction before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_mem_zeroFloor {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ localWeightDomain N floor) :
    localMemoryCore t ∈ memoryGramDomain N 1 0 := by
  exact (memoryGramDomain_convex N 1 0).sum_mem
    (t := Finset.univ) (w := localMemoryWeights t) (z := localMemoryAtom)
    (fun e he => localMemoryWeights_nonneg floor t hf ht e)
    (localMemoryWeights_sum t) (fun e he => localMemoryAtom_mem e)

/-- A positive edge inhabits all atom-mixture feasibility premises. -/
example : localMemoryCore (fun _ : Fin 1 => (1 / 4 : ℝ)) ∈ memoryGramDomain 1 1 0 := by
  apply localMemoryCore_mem_zeroFloor (3 / 4) _ (by norm_num)
  constructor
  · intro e
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- The identity budget implies every required learned self-score floor.
Source: the derived compact inverse domain before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_mem {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ localWeightDomain N floor) :
    localMemoryCore t ∈ memoryGramDomain N 1 floor := by
  have h0 := localMemoryCore_mem_zeroFloor floor t hf ht
  refine ⟨h0.1, h0.2.1, ?_⟩
  intro i
  rw [localMemoryCore_scores_apply, Fintype.sum_option]
  have hid : localMemoryPermutation (none : Option (Fin N)) i = i := rfl
  simp only [hid, ite_true, mul_one]
  have hn : 0 ≤ ∑ e : Fin N, localMemoryWeights t (some e) *
      (if localMemoryPermutation (some e) i = i then (1 : ℝ) else 0) := by
    apply Finset.sum_nonneg
    intro e he
    apply mul_nonneg (ht.1 e)
    split_ifs <;> norm_num
  have hw := localMemoryWeights_identity_floor floor t ht
  linarith

/-- Strict dominance is inhabited with a genuinely changed off-diagonal support. -/
example : localMemoryCore (fun _ : Fin 1 => (1 / 4 : ℝ)) ∈
    memoryGramDomain 1 1 (3 / 4) := by
  apply localMemoryCore_mem _ _ (by norm_num)
  constructor
  · intro e
    norm_num
  · norm_num [Fin.sum_univ_one]

/-- Every query and key norm in the core is one for all edge coordinates.
Source: the normalized genuine atom mixture before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_diagonal {N : ℕ} (t : Fin N → ℝ)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) : localMemoryCore t x x = 1 := by
  rw [localMemoryCore_apply]
  have ha (e : Option (Fin N)) : localMemoryAtom e x x = 1 :=
    permutationMemoryGram_diagonal _ x
  simp_rw [ha, mul_one]
  exact localMemoryWeights_sum t

/-- Both core same-family blocks are identity, independently of learned edge values.
Source: the explicit orthogonality restriction for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_sameSide {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1)) :
    localMemoryCore t (Sum.inl i) (Sum.inl j) = (if i = j then 1 else 0) ∧
    localMemoryCore t (Sum.inr i) (Sum.inr j) = (if i = j then 1 else 0) := by
  constructor
  · rw [localMemoryCore_apply]
    simp_rw [(localMemoryAtom_sameSide _ i j).1]
    rw [← Finset.sum_mul, localMemoryWeights_sum, one_mul]
  · rw [localMemoryCore_apply]
    simp_rw [(localMemoryAtom_sameSide _ i j).2]
    rw [← Finset.sum_mul, localMemoryWeights_sum, one_mul]

/-- Zero learned edges recover the genuine identity embedding Gram.
Source: the identity endpoint of the compact arXiv:1602.02068v2, Eq. (1) family. -/
theorem localMemoryCore_zero (N : ℕ) :
    localMemoryCore (0 : Fin N → ℝ) = memoryIdentityGram N := by
  unfold localMemoryCore
  rw [Fintype.sum_option]
  simp only [localMemoryWeights, Pi.zero_apply, Finset.sum_const_zero, sub_zero,
    one_smul, zero_smul, add_zero]
  unfold localMemoryAtom permutationMemoryGram memoryIdentityGram
  rfl

/-- The compact core Gram is affine in its N learned parameters.
Source: the derived compact embedding chart before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_affine {N : ℕ} (t s : Fin N → ℝ) (a b : ℝ) (hab : a + b = 1) :
    localMemoryCore (a • t + b • s) = a • localMemoryCore t + b • localMemoryCore s := by
  unfold localMemoryCore
  rw [localMemoryWeights_affine t s a b hab]
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_smul, mul_smul,
    Finset.sum_add_distrib, ← Finset.smul_sum]

/-- Different edge endpoints inhabit the compact Gram affinity premise. -/
example : localMemoryCore ((1 / 2 : ℝ) • (0 : Fin 1 → ℝ) +
    (1 / 2 : ℝ) • (fun _ : Fin 1 => (1 / 4 : ℝ))) =
    (1 / 2 : ℝ) • localMemoryCore (0 : Fin 1 → ℝ) +
      (1 / 2 : ℝ) • localMemoryCore (fun _ : Fin 1 => (1 / 4 : ℝ)) :=
  localMemoryCore_affine _ _ _ _ (by norm_num)

end Transformer.GPTMini.Sparsemax
