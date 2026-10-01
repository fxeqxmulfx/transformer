/-
# Exact existential root criteria from coefficient operations

Multiplicity removal and Euclidean chains make the tests independent of
any supplied root or Sturm certificate. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Sturm.SquarefreeCount

noncomputable section
open Polynomial

namespace Transformer.Sturm

/-- A nonzero polynomial has a real root in `(a, b]` exactly when the
constructed variation count decreases. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_root_Ioc_iff {p : Polynomial ℝ} (hp : p ≠ 0)
    {a b : ℝ} (hab : a ≤ b) :
    (∃ x ∈ Set.Ioc a b, p.eval x = 0) ↔
      sturmVar (sturmChain p) b < sturmVar (sturmChain p) a := by
  have hcount := sturmChain_count_Ioc hp hab
  have hpositive : (∃ x ∈ Set.Ioc a b, p.eval x = 0) ↔
      0 < (p.roots.toFinset.filter (fun r => r ∈ Set.Ioc a b)).card := by
    rw [Finset.card_pos]
    simp only [Finset.Nonempty, Finset.mem_filter, Multiset.mem_toFinset, mem_roots hp,
      Polynomial.IsRoot]
    exact ⟨fun ⟨x, hx, hr⟩ => ⟨x, hr, hx⟩, fun ⟨x, hr, hx⟩ => ⟨x, hx, hr⟩⟩
  rw [hpositive]
  omega

/-- A nonzero polynomial has a real root exactly when its variations
at negative and positive infinity differ. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_real_root_iff {p : Polynomial ℝ} (hp : p ≠ 0) :
    (∃ x : ℝ, p.eval x = 0) ↔ sturmVarPosInf (sturmChain p) < sturmVarNegInf (sturmChain p) := by
  have hcount := sturmChain_count hp
  have hpositive : (∃ x : ℝ, p.eval x = 0) ↔ 0 < p.roots.toFinset.card := by
    rw [Finset.card_pos]
    simp only [Finset.Nonempty, Multiset.mem_toFinset, mem_roots hp, Polynomial.IsRoot]
  rw [hpositive]
  omega

/-- Removing the right endpoint changes a finite interval count by one
exactly when that endpoint belongs to the finite set. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem card_filter_Ioc_eq_Ioo (S : Finset ℝ) {a b : ℝ} (hab : a < b) :
    (S.filter (fun x => x ∈ Set.Ioc a b)).card =
      (S.filter (fun x => x ∈ Set.Ioo a b)).card + if b ∈ S then 1 else 0 := by
  by_cases hb : b ∈ S
  · have heq : S.filter (fun x => x ∈ Set.Ioc a b) =
        insert b (S.filter (fun x => x ∈ Set.Ioo a b)) := by
      ext x
      simp only [Finset.mem_filter, Finset.mem_insert, Set.mem_Ioc, Set.mem_Ioo]
      constructor
      · rintro ⟨hx, hax, hxb⟩
        rcases eq_or_lt_of_le hxb with rfl | hxb
        · exact Or.inl rfl
        · exact Or.inr ⟨hx, hax, hxb⟩
      · rintro (rfl | ⟨hx, hax, hxb⟩)
        · exact ⟨hb, hab, le_rfl⟩
        · exact ⟨hx, hax, hxb.le⟩
    have hn : b ∉ S.filter (fun x => x ∈ Set.Ioo a b) := by simp
    rw [heq, Finset.card_insert_of_notMem hn, ite_eq_left hb]
  · have heq : S.filter (fun x => x ∈ Set.Ioc a b) =
        S.filter (fun x => x ∈ Set.Ioo a b) := by
      ext x
      simp only [Finset.mem_filter, Set.mem_Ioc, Set.mem_Ioo]
      constructor
      · rintro ⟨hx, hax, hxb⟩
        have hne : x ≠ b := fun h => hb (h ▸ hx)
        exact ⟨hx, hax, lt_of_le_of_ne hxb hne⟩
      · rintro ⟨hx, hax, hxb⟩
        exact ⟨hx, hax, hxb.le⟩
    rw [heq, ite_eq_right hb, add_zero]

/-- Exact elimination of a real root variable on an open interval.
The zero polynomial is included explicitly; arbitrary repeated roots
and roots at either endpoint are handled. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_root_Ioo_iff (p : Polynomial ℝ) {a b : ℝ} (hab : a < b) :
    (∃ x ∈ Set.Ioo a b, p.eval x = 0) ↔
      p = 0 ∨ sturmVar (sturmChain p) b + (if p.eval b = 0 then 1 else 0) <
        sturmVar (sturmChain p) a := by
  by_cases hp : p = 0
  · simp only [hp, true_or, iff_true]
    exact ⟨(a + b) / 2, by constructor <;> linarith, by simp⟩
  · have hcount := sturmChain_count_Ioc hp hab.le
    rw [card_filter_Ioc_eq_Ioo _ hab] at hcount
    have hendpoint : b ∈ p.roots.toFinset ↔ p.eval b = 0 := by
      simp only [Multiset.mem_toFinset, mem_roots hp, Polynomial.IsRoot]
    simp only [hendpoint] at hcount
    have hpositive : (∃ x ∈ Set.Ioo a b, p.eval x = 0) ↔
        0 < (p.roots.toFinset.filter (fun r => r ∈ Set.Ioo a b)).card := by
      rw [Finset.card_pos]
      simp only [Finset.Nonempty, Finset.mem_filter, Multiset.mem_toFinset, mem_roots hp,
        Polynomial.IsRoot]
      exact ⟨fun ⟨x, hx, hr⟩ => ⟨x, hr, hx⟩, fun ⟨x, hr, hx⟩ => ⟨x, hx, hr⟩⟩
    simp only [hp, false_or, hpositive]
    omega

/-- A repeated-root polynomial, endpoint root, and finite singleton set
jointly witness the interval and nonzero hypotheses. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : (X ^ 4 : Polynomial ℝ) ≠ 0 ∧ (-1 : ℝ) < 0 ∧
    (0 : ℝ) ∈ ({0} : Finset ℝ) := by
  exact ⟨pow_ne_zero _ X_ne_zero, by norm_num, by simp⟩

end Transformer.Sturm
