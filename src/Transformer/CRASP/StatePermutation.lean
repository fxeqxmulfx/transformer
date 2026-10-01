/-
# Attention tables are invariant under permutations of source states

arXiv:2506.16055v3, Appendix B.2 and F, finite-state counting construction.
At one layer, fixing the query and permuting the finite source-state word
preserves every numerator, denominator and fallback sum.
-/

import Transformer.CRASP.PositionalStateCounts

namespace Transformer.CRASP.CountCells

universe u
variable {α : Type u} {p s : ℕ}

/-- The natural-indexed source sum equals the sum over its auxiliary word (F). -/
theorem sum_stateWord (q : ℕ → α) (i : ℕ) (f : α → ℤ) :
    ((stateWord q i).map f).sum = ∑ j ∈ Finset.Icc 1 i, f (q j) := by
  rw [stateWord, List.map_ofFn, List.sum_ofFn]
  refine Finset.sum_bij (fun j _ => j.val + 1) ?_ ?_ ?_ ?_
  · intro j hj
    exact Finset.mem_Icc.mpr ⟨by omega, by have := j.isLt; omega⟩
  · intro j hj t ht heq
    apply Fin.ext
    omega
  · intro j hj
    have hjb := Finset.mem_Icc.mp hj
    refine ⟨⟨j - 1, by omega⟩, Finset.mem_univ _, ?_⟩
    change j - 1 + 1 = j
    omega
  · intro j hj
    rfl

/-- Permuting source states preserves the complete rounded attention value.
Source: arXiv:2506.16055v3, Appendix F, finite query/source enumeration. -/
theorem value_eq_of_perm (b : α) (D N V : α → Fx p s)
    (q q' : ℕ → α) (i i' : ℕ)
    (hperm : (stateWord q i).Perm (stateWord q' i')) :
    value b D N V q i = value b D N V q' i' := by
  have hi : i = i' := by simpa [stateWord] using hperm.length_eq
  subst i'
  have hsum (f : α → Fx p s) : sum b f q i = sum b f q' i := by
    unfold sum
    rw [← sum_stateWord q i (fun a => (f a).m),
      ← sum_stateWord q' i (fun a => (f a).m), (hperm.map (fun a => (f a).m)).sum_eq]
  simp only [value, hsum]

/-- Source permutations occur nontrivially even for two states (Appendix F). -/
example : (stateWord (fun j => decide (j = 1)) 2).Perm
    (stateWord (fun j => decide (j = 2)) 2) := by
  change [true, false].Perm [false, true]
  exact List.Perm.swap _ _ _

end Transformer.CRASP.CountCells
