import Transformer.GPTMini.Sparsemax.LocalMemoryCore

/-!+# Positive embedding Grams from nonnegative doubly stochastic scores

Derived lifting before arXiv:1602.02068v2, Eq. (1). Each nonnegative
cross score weights the outer product of a column joining one query and
one key. The resulting matrix is PSD. Unit row and column sums make both
same-family blocks identity, while the cross block is exactly the scores.

This provides a positivity proof without a global convex-mixture budget.
It is a finite matrix identity, not a rank assumption or a claim that every
nonnegative matrix is a normalized sparsemax input. Subsequent modules
apply it to symmetric local path scores with separate row budgets.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Join one query column and one key column in an explicit real vector.
Source: the derived embedding lift before arXiv:1602.02068v2, Eq. (1). -/
def pairedMemoryColumn {N : ℕ} (i j : Fin (N + 1)) :
    Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ
  | Sum.inl k => if i = k then 1 else 0
  | Sum.inr k => if j = k then 1 else 0

/-- Nonnegative scores weight genuine rank-one embedding Grams.
Source: the derived stochastic-score lifting before arXiv:1602.02068v2, Eq. (1). -/
def bipartiteMemoryGram {N : ℕ} (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) :
    EmbeddingGram (N + 1) :=
  ∑ i, ∑ j, B i j • Matrix.vecMulVec (pairedMemoryColumn i j) (pairedMemoryColumn i j)

/-- The lifting is positive semidefinite for every nonnegative score matrix.
Source: nonnegative outer-product sums in the derived arXiv:1602.02068v2, Eq. (1) lift. -/
theorem bipartiteMemoryGram_posSemidef {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) (hB : ∀ i j, 0 ≤ B i j) :
    (bipartiteMemoryGram B).PosSemidef := by
  unfold bipartiteMemoryGram
  apply Matrix.posSemidef_sum
  intro i hi
  apply Matrix.posSemidef_sum
  intro j hj
  have h : (Matrix.vecMulVec (pairedMemoryColumn i j) (pairedMemoryColumn i j)).PosSemidef := by
    simpa only [Pi.star_def, star_trivial] using
      Matrix.posSemidef_vecMulVec_self_star (pairedMemoryColumn i j)
  exact h.smul (hB i j)

/-- A nonidentity stochastic score table inhabits the positivity premises. -/
example : (bipartiteMemoryGram
    (Matrix.of (fun i j : Fin 2 => if i = j then (3 / 4 : ℝ) else 1 / 4))).PosSemidef := by
  apply bipartiteMemoryGram_posSemidef
  intro i j
  change 0 ≤ if i = j then (3 / 4 : ℝ) else 1 / 4
  split_ifs <;> norm_num

/-- Query-block entries equal row mass on the diagonal and vanish elsewhere.
Source: explicit outer products preceding arXiv:1602.02068v2, Eq. (1). -/
theorem bipartiteMemoryGram_query {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) (i k : Fin (N + 1)) :
    bipartiteMemoryGram B (Sum.inl i) (Sum.inl k) =
      if i = k then ∑ j, B i j else 0 := by
  unfold bipartiteMemoryGram
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.vecMulVec_apply, pairedMemoryColumn]
  by_cases h : i = k
  · subst k
    simp only [mul_ite, mul_one, mul_zero]
    rw [Finset.sum_comm]
    simp only [Fintype.sum_ite_eq', ite_true]
  · have hz (x : Fin (N + 1)) :
        (if x = i then (1 : ℝ) else 0) * (if x = k then 1 else 0) = 0 := by
      by_cases hx : x = i
      · subst x
        simp only [h, ite_true, ite_false, mul_zero]
      · simp only [hx, ite_false, zero_mul]
    simp only [hz, mul_zero, Finset.sum_const_zero, h, ite_false]

/-- Key-block entries equal column mass on the diagonal and vanish elsewhere.
Source: the other embedding family in the derived arXiv:1602.02068v2, Eq. (1) lift. -/
theorem bipartiteMemoryGram_key {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) (j k : Fin (N + 1)) :
    bipartiteMemoryGram B (Sum.inr j) (Sum.inr k) =
      if j = k then ∑ i, B i j else 0 := by
  unfold bipartiteMemoryGram
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.vecMulVec_apply, pairedMemoryColumn]
  by_cases h : j = k
  · subst k
    simp only [mul_ite, mul_one, mul_zero, Fintype.sum_ite_eq', ite_true]
  · have hz (y : Fin (N + 1)) :
        (if y = j then (1 : ℝ) else 0) * (if y = k then 1 else 0) = 0 := by
      by_cases hy : y = j
      · subst y
        simp only [h, ite_true, ite_false, mul_zero]
      · simp only [hy, ite_false, zero_mul]
    simp only [hz, mul_zero, Finset.sum_const_zero, h, ite_false]

/-- The lifted query-key cross block is exactly the supplied score matrix.
Source: actual scalar products before arXiv:1602.02068v2, Eq. (1). -/
theorem bipartiteMemoryGram_cross {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) (i j : Fin (N + 1)) :
    bipartiteMemoryGram B (Sum.inl i) (Sum.inr j) = B i j := by
  unfold bipartiteMemoryGram
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.vecMulVec_apply, pairedMemoryColumn, mul_ite, mul_one, mul_zero,
    Fintype.sum_ite_eq']

/-- Reversing the two embedding copies reverses the score indices.
Source: symmetry of real outer products in the derived arXiv:1602.02068v2, Eq. (1) lift. -/
theorem bipartiteMemoryGram_cross_reverse {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ) (i j : Fin (N + 1)) :
    bipartiteMemoryGram B (Sum.inr j) (Sum.inl i) = B i j := by
  unfold bipartiteMemoryGram
  simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.vecMulVec_apply, pairedMemoryColumn, mul_ite, mul_one, mul_zero,
    Finset.sum_ite_irrel, Finset.sum_const_zero, Fintype.sum_ite_eq']

/-- Distinct cross-score tables give distinct genuine embedding Grams.
Source: exact cross recovery in the derived arXiv:1602.02068v2, Eq. (1) lift. -/
theorem bipartiteMemoryGram_injective (N : ℕ) :
    Function.Injective (bipartiteMemoryGram (N := N)) := by
  intro B C h
  ext i j
  have he := congrFun (congrFun h (Sum.inl i)) (Sum.inr j)
  rw [bipartiteMemoryGram_cross, bipartiteMemoryGram_cross] at he
  exact he

/-- Unit row and column sums produce identity blocks for both embedding families.
Source: the doubly stochastic derived Gram construction before arXiv:1602.02068v2, Eq. (1). -/
theorem bipartiteMemoryGram_sameSide {N : ℕ}
    (B : Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ)
    (hr : ∀ i, ∑ j, B i j = 1) (hc : ∀ j, ∑ i, B i j = 1) (i j : Fin (N + 1)) :
    bipartiteMemoryGram B (Sum.inl i) (Sum.inl j) = (if i = j then 1 else 0) ∧
    bipartiteMemoryGram B (Sum.inr i) (Sum.inr j) = (if i = j then 1 else 0) := by
  rw [bipartiteMemoryGram_query, bipartiteMemoryGram_key, hr, hc]
  exact ⟨rfl, rfl⟩

/-- A changed two-slot table jointly inhabits both normalization hypotheses. -/
example (i j : Fin 2) :
    bipartiteMemoryGram (Matrix.of (fun i j : Fin 2 =>
      if i = j then (3 / 4 : ℝ) else 1 / 4)) (Sum.inl i) (Sum.inl j) =
      (if i = j then 1 else 0) ∧
    bipartiteMemoryGram (Matrix.of (fun i j : Fin 2 =>
      if i = j then (3 / 4 : ℝ) else 1 / 4)) (Sum.inr i) (Sum.inr j) =
      (if i = j then 1 else 0) := by
  apply bipartiteMemoryGram_sameSide
  · intro i
    change (∑ k : Fin 2, if i = k then (3 / 4 : ℝ) else 1 / 4) = 1
    fin_cases i <;> norm_num [Fin.sum_univ_two]
  · intro j
    change (∑ k : Fin 2, if k = j then (3 / 4 : ℝ) else 1 / 4) = 1
    fin_cases j <;> norm_num [Fin.sum_univ_two]

end Transformer.GPTMini.Sparsemax
