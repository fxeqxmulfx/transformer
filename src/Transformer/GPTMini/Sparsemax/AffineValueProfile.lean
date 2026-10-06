import Transformer.GPTMini.Sparsemax.PeriodicMemoryForward

/-!+# A generated path attention preserving two value features

New compact-value architecture after arXiv:1602.02068v2, Eq. (1).
One learned scalar generates all path edges as `c*(e+1)*(N+1-e)`.
The actual probability-score path preserves constant and affine positional
features. Its action on slot index is `i+c*(N+1-2*i)`, including endpoints.
Consequently the centered positional feature is an eigenvector with
eigenvalue `1-2*c`; subsequent original values use only two coefficients
per output channel and no dictionary-sized inverse or stored value table.

The edge profile and positional features are generated formulas, not
stored per-prototype parameters. Local feasibility and genuine sparsemax
remain the earlier proved conditions, rather than assumed target routes.
This restriction gives affine positional responses, not arbitrary answer
memorization. The prototype records and input encoder are separate costs.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- A single learned scalar generates every possible adjacent path weight.
Source: the new invariant-feature restriction before sparsemax Eq. (1). -/
def affineValueEdges (N : ℕ) (c : ℝ) : Fin (N + 1) → ℝ :=
  fun e => c * (e.val + 1) * (N + 1 - e.val)

/-- Applying the actual score path to any feature is a sum of adjacent feature differences.
Source: identity and swap atoms preceding arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryCore_feature_action {N : ℕ} (t : Fin N → ℝ)
    (f : Fin (N + 1) → ℝ) (i : Fin (N + 1)) :
    (∑ j, memoryGramScores (localMemoryCore t) i j * f j) =
      f i + ∑ e, t e * (f (Equiv.swap e.castSucc e.succ i) - f i) := by
  simp_rw [localMemoryCore_scores_apply, Finset.sum_mul]
  rw [Finset.sum_comm]
  have he (e : Option (Fin N)) :
      (∑ j, localMemoryWeights t e *
        (if localMemoryPermutation e i = j then 1 else 0) * f j) =
      localMemoryWeights t e * f (localMemoryPermutation e i) := by
    simp only [mul_ite, mul_one, mul_zero, zero_mul, ite_mul]
    exact Fintype.sum_ite_eq _ _
  simp_rw [he]
  rw [Fintype.sum_option]
  change (1 - ∑ e, t e) * f i +
    (∑ e, t e * f (Equiv.swap e.castSucc e.succ i)) = _
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib, ← Finset.sum_mul]
  ring

/-- The generated edge factors outgoing to the right have a closed boundary-safe sum.
Source: the quadratic profile of the new sparsemax Eq. (1) restriction. -/
theorem affineValueProfile_right (N : ℕ) (i : Fin (N + 2)) :
    (∑ e : Fin (N + 1), if i = e.castSucc then
      ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) =
      (i.val + 1) * (N + 1 - i.val) := by
  classical
  by_cases hi : i.val < N + 1
  · let e : Fin (N + 1) := ⟨i.val, hi⟩
    have he : i = e.castSucc := Fin.ext rfl
    rw [Fintype.sum_eq_single e]
    · simp only [he, ite_true]
      rfl
    · intro k hk
      have hn : i ≠ k.castSucc := by
        intro h
        exact hk ((Fin.castSucc_injective _).eq_iff.mp (he.symm.trans h)).symm
      simp only [hn, ite_false]
  · have hv : i.val = N + 1 := by omega
    have hz : (∑ e : Fin (N + 1), if i = e.castSucc then
        ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) = 0 := by
      apply Finset.sum_eq_zero
      intro e he
      have hn : i ≠ e.castSucc := by
        intro h
        have hval := congrArg Fin.val h
        simp only [Fin.val_castSucc] at hval
        omega
      simp only [hn, ite_false]
    rw [hz, hv]
    push_cast
    ring

/-- The generated edge factors outgoing to the left have a closed boundary-safe sum.
Source: the same invariant-feature profile before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueProfile_left (N : ℕ) (i : Fin (N + 2)) :
    (∑ e : Fin (N + 1), if i = e.succ then
      ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) =
      i.val * (N + 2 - i.val) := by
  classical
  by_cases hi : i.val = 0
  · have hz : (∑ e : Fin (N + 1), if i = e.succ then
        ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) = 0 := by
      apply Finset.sum_eq_zero
      intro e he
      have hn : i ≠ e.succ := by
        intro h
        have hval := congrArg Fin.val h
        simp only [Fin.val_succ] at hval
        omega
      simp only [hn, ite_false]
    rw [hz, hi]
    norm_num
  · let e : Fin (N + 1) := ⟨i.val - 1, by omega⟩
    have hv : e.val + 1 = i.val := by dsimp [e]; omega
    have he : i = e.succ := Fin.ext hv.symm
    rw [Fintype.sum_eq_single e]
    · simp only [he, ite_true]
      simp only [Fin.val_succ, Nat.cast_add, Nat.cast_one]
      ring
    · intro k hk
      have hn : i ≠ k.succ := by
        intro h
        exact hk ((Fin.succ_injective _).eq_iff.mp (he.symm.trans h)).symm
      simp only [hn, ite_false]

/-- The actual affine path sends slot index to an affine function of that same index.
Source: invariant two-feature generation before sparsemax Eq. (1), proved at both boundaries. -/
theorem affineValueProfile_index (N : ℕ) (c : ℝ) (i : Fin (N + 2)) :
    (∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) i j * (j.val : ℝ)) =
      i.val + c * (N + 1 - 2 * i.val) := by
  rw [localMemoryCore_feature_action]
  have he (e : Fin (N + 1)) :
      affineValueEdges N c e *
        ((Equiv.swap e.castSucc e.succ i).val - (i.val : ℝ)) =
      c * (if i = e.castSucc then ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) -
      c * (if i = e.succ then ((e.val : ℝ) + 1) * (N + 1 - e.val) else 0) := by
    have hn : e.castSucc ≠ e.succ := by
      intro h
      have hv := congrArg Fin.val h
      simp only [Fin.val_castSucc, Fin.val_succ] at hv
      omega
    by_cases hl : i = e.castSucc
    · subst i
      simp only [Equiv.swap_apply_left, Fin.val_succ, Fin.val_castSucc,
        Nat.cast_add, Nat.cast_one, eq_self, hn, ite_true, ite_false]
      unfold affineValueEdges
      ring
    · by_cases hr : i = e.succ
      · subst i
        simp only [Equiv.swap_apply_right, Fin.val_succ, Fin.val_castSucc,
          Nat.cast_add, Nat.cast_one, eq_self, Ne.symm hn, ite_true, ite_false]
        unfold affineValueEdges
        ring
      · rw [Equiv.swap_apply_of_ne_of_ne hl hr]
        simp only [sub_self, mul_zero, hl, hr, ite_false, sub_self]
  simp_rw [he]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    affineValueProfile_right, affineValueProfile_left]
  ring

/-- A nonidentity four-slot profile preserves a nonconstant index feature exactly. -/
example : (∑ j, memoryGramScores (localMemoryCore (affineValueEdges 2 (1 / 48)))
    (1 : Fin 4) j * (j.val : ℝ)) = 1 + (1 / 48 : ℝ) := by
  simpa using affineValueProfile_index 2 (1 / 48) (1 : Fin 4)

end Transformer.GPTMini.Sparsemax
