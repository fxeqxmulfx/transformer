import Transformer.GPTMini.Sparsemax.LocalIncidentWeights
import Transformer.GPTMini.Sparsemax.LocalMemoryCore

/-!
# Algebra of path scores independently of a global mixture budget

Derived path chart before arXiv:1602.02068v2, Eq. (1). Its scores are
symmetric and have unit row sums for every real edge table. Each diagonal
is exactly one minus the incident edge mass. Separate local budgets and
nonnegative edges therefore make every row a probability vector with the
desired diagonal floor, even when the global identity mass is negative.

These are identities and inequalities for the already defined affine
Gram. No new attention is defined to bypass the actual projection.
Positivity of the full Gram needs a separate proof using these identities.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- All local permutation cross scores are symmetric.
Source: identity and adjacent swaps in the derived arXiv:1602.02068v2, Eq. (1) chart. -/
theorem localMemoryPermutation_score_symmetric {N : ℕ} (e : Option (Fin N))
    (i j : Fin (N + 1)) :
    (localMemoryPermutation e i = j) ↔ (localMemoryPermutation e j = i) := by
  cases e with
  | none => exact eq_comm
  | some e =>
    change Equiv.swap e.castSucc e.succ i = j ↔ Equiv.swap e.castSucc e.succ j = i
    rw [Equiv.swap_apply_eq_iff]
    exact eq_comm

/-- The path score matrix is symmetric for arbitrary real edge coordinates.
Source: the affine symmetric path chart before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_scores_symmetric {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1)) :
    memoryGramScores (localMemoryCore t) i j = memoryGramScores (localMemoryCore t) j i := by
  rw [localMemoryCore_scores_apply, localMemoryCore_scores_apply]
  apply Finset.sum_congr rfl
  intro e he
  simp only [localMemoryPermutation_score_symmetric e i j]

/-- Every path score row has unit mass without a global nonnegativity premise.
Source: normalized affine permutation coefficients preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_scores_rowSum {N : ℕ} (t : Fin N → ℝ) (i : Fin (N + 1)) :
    (∑ j, memoryGramScores (localMemoryCore t) i j) = 1 := by
  simp_rw [localMemoryCore_scores_apply]
  rw [Finset.sum_comm]
  simp only [← Finset.mul_sum, Fintype.sum_ite_eq, mul_one]
  exact localMemoryWeights_sum t

/-- Symmetry gives unit column mass for arbitrary edge coordinates.
Source: the doubly stochastic algebra of the derived arXiv:1602.02068v2, Eq. (1) path chart. -/
theorem localMemoryCore_scores_columnSum {N : ℕ} (t : Fin N → ℝ) (j : Fin (N + 1)) :
    (∑ i, memoryGramScores (localMemoryCore t) i j) = 1 := by
  simp_rw [localMemoryCore_scores_symmetric t _ j]
  exact localMemoryCore_scores_rowSum t j

/-- Diagonal scores spend exactly the edges touching their own slot.
Source: the actual self-score budget before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_scores_diagonal {N : ℕ} (t : Fin N → ℝ) (i : Fin (N + 1)) :
    memoryGramScores (localMemoryCore t) i i = 1 - localIncidentWeight t i := by
  rw [localMemoryCore_scores_apply, Fintype.sum_option]
  change (1 - ∑ e, t e) * (if i = i then (1 : ℝ) else 0) +
    (∑ e, t e * (if Equiv.swap e.castSucc e.succ i = i then (1 : ℝ) else 0)) = _
  have he (e : Fin N) : t e * (if Equiv.swap e.castSucc e.succ i = i then (1 : ℝ) else 0) =
      t e - (if i = e.castSucc ∨ i = e.succ then t e else 0) := by
    have hne : e.castSucc ≠ e.succ := by
      intro h
      have hv := congrArg Fin.val h
      simp only [Fin.val_castSucc, Fin.val_succ] at hv
      omega
    by_cases hl : i = e.castSucc
    · subst i
      simp only [Equiv.swap_apply_left, Ne.symm hne, ite_false, eq_self,
        true_or, ite_true, mul_zero, sub_self]
    · by_cases hr : i = e.succ
      · subst i
        simp only [Equiv.swap_apply_right, hne, ite_false, eq_self,
          or_true, ite_true, mul_zero, sub_self]
      · rw [Equiv.swap_apply_of_ne_of_ne hl hr]
        simp only [eq_self, ite_true, hl, hr, or_self, ite_false, mul_one, sub_zero]
  simp_rw [he]
  rw [Finset.sum_sub_distrib]
  unfold localIncidentWeight
  simp only [ite_true, mul_one]
  ring

/-- Separate budgets imply the actual diagonal floor.
Source: the new local normalization constraints for arXiv:1602.02068v2, Eq. (1). -/
theorem incidentMemory_scores_floor {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (ht : t ∈ incidentMemoryWeightDomain N floor) (i : Fin (N + 1)) :
    floor ≤ memoryGramScores (localMemoryCore t) i i := by
  rw [localMemoryCore_scores_diagonal]
  have hb := ht.2 i
  linarith

/-- A formerly inadmissible three-edge point inhabits the floor premises. -/
example : (3 / 4 : ℝ) ≤ memoryGramScores
    (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) 1 1 :=
  incidentMemory_scores_floor _ _ incidentMemoryExampleWeights_mem 1

/-- All path scores are nonnegative on the separate-budget domain.
Source: probability inputs for the actual arXiv:1602.02068v2, Eq. (1) projection. -/
theorem incidentMemory_scores_nonneg {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) (i j : Fin (N + 1)) :
    0 ≤ memoryGramScores (localMemoryCore t) i j := by
  by_cases h : i = j
  · subst j
    exact hf.trans (incidentMemory_scores_floor floor t ht i)
  · rw [localMemoryCore_scores_apply, Fintype.sum_option]
    change 0 ≤ (1 - ∑ e, t e) * (if i = j then (1 : ℝ) else 0) +
      ∑ e, t e * (if localMemoryPermutation (some e) i = j then (1 : ℝ) else 0)
    simp only [h, ite_false, mul_zero, zero_add]
    apply Finset.sum_nonneg
    intro e he
    apply mul_nonneg (ht.1 e)
    split_ifs <;> norm_num

/-- A nonzero off-diagonal score satisfies all relaxed nonnegativity premises. -/
example : 0 ≤ memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) 1 2 :=
  incidentMemory_scores_nonneg _ _ (by norm_num) incidentMemoryExampleWeights_mem 1 2

/-- Every relaxed score is at most one by its row's unit nonnegative mass.
Source: the normalized sparsemax input rows of the derived arXiv:1602.02068v2, Eq. (1) memory. -/
theorem incidentMemory_scores_le_one {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) (i j : Fin (N + 1)) :
    memoryGramScores (localMemoryCore t) i j ≤ 1 := by
  have h := Finset.single_le_sum
    (fun k hk => incidentMemory_scores_nonneg floor t hf ht i k) (Finset.mem_univ j)
  rw [localMemoryCore_scores_rowSum] at h
  exact h

/-- A changed three-slot row inhabits the normalized-score bound assumptions. -/
example : memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) 1 2 ≤ 1 :=
  incidentMemory_scores_le_one _ _ (by norm_num) incidentMemoryExampleWeights_mem 1 2

/-- The full affine core is symmetric even if some global atom coefficients are negative.
Source: the genuine real permutation Grams before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_symmetric {N : ℕ} (t : Fin N → ℝ)
    (x y : Sum (Fin (N + 1)) (Fin (N + 1))) : localMemoryCore t x y = localMemoryCore t y x := by
  rw [localMemoryCore_apply, localMemoryCore_apply]
  apply Finset.sum_congr rfl
  intro e he
  have ha := (localMemoryAtom_mem e).1.1.isHermitian.apply y x
  change star (localMemoryAtom e x y) = localMemoryAtom e y x at ha
  rw [star_trivial] at ha
  rw [ha]

end Transformer.GPTMini.Sparsemax
