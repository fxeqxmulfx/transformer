/-
# Counting in `MAJ²`: majorities over lists, and the three masks

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, proof of `thm:tlc_to_majtwo`: a comparison
`t₁ < t₂` of counting terms becomes one majority quantifier over the formulas
that the terms count, padded by `⊤` and `⊥` so that exactly half the mass is
the break-even point.

`Maj2.mass w ξ u l` is the number of pairs `(j, g)` with `g ∈ l` satisfied at
`u ↦ j`.  A majority over a list is a comparison of its mass
(`Maj2.sat_majList`); negating a list complements its mass, and `⊤`/`⊥` add
`|w|` and `0` (`Maj2.sat_cmpList`, which is the whole trick).  The masks
`y ≤ x`, `x ≤ y` and `y = x` turn a sum over all positions into `◁#`, `▷#`
and `1` (`sum_Icc_le`, `sum_Icc_ge`, `sum_Icc_eq`).
-/

import Transformer.CRASP.MajTwoDepthOne

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A sum of indicators over `[s, s+n)` is the length of the filtered
`range'` that `Term.val` counts. -/
theorem sum_Ico_boole_eq_length_filter (f : ℕ → Bool) :
    ∀ s n, ∑ j ∈ Finset.Ico s (s + n), (if f j then 1 else 0) =
      ((List.range' s n).filter f).length
  | s, 0 => by simp
  | s, n + 1 => by
      rw [Finset.sum_eq_sum_Ico_succ_bot (by omega), show s + (n + 1) = s + 1 + n by omega,
        sum_Ico_boole_eq_length_filter f (s + 1) n, List.range'_succ, List.filter_cons]
      cases f s <;> simp [Nat.add_comm]

/-- The mask `j ≤ i` turns the sum over all positions into `◁#` (Appendix E,
proof of `thm:tlc_to_majtwo`). -/
theorem sum_Icc_le (f : ℕ → Bool) {i n : ℕ} (hi : i ≤ n) :
    ∑ j ∈ Finset.Icc 1 n, (if (f j && !decide (i < j)) then 1 else 0) =
      ((List.range' 1 i).filter f).length := by
  rw [← Finset.Ico_add_one_right_eq_Icc, ← Finset.sum_Ico_consecutive _ (by omega : 1 ≤ 1 + i)
    (by omega : 1 + i ≤ n + 1), Finset.sum_eq_zero (s := Finset.Ico (1 + i) (n + 1)) ?_,
    Nat.add_zero, ← sum_Ico_boole_eq_length_filter]
  · refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_Ico] at hj
    simp [show ¬ i < j by omega]
  · intro j hj
    rw [Finset.mem_Ico] at hj
    simp [show i < j by omega]

/-- The mask `i ≤ j` turns the sum over all positions into `▷#`. -/
theorem sum_Icc_ge (f : ℕ → Bool) {i n : ℕ} (hi₁ : 1 ≤ i) (hi : i ≤ n) :
    ∑ j ∈ Finset.Icc 1 n, (if (f j && !decide (j < i)) then 1 else 0) =
      ((List.range' i (n + 1 - i)).filter f).length := by
  rw [← Finset.Ico_add_one_right_eq_Icc, ← Finset.sum_Ico_consecutive _ hi₁
    (by omega : i ≤ n + 1), ← sum_Ico_boole_eq_length_filter,
    show i + (n + 1 - i) = n + 1 by omega, Finset.sum_eq_zero (s := Finset.Ico 1 i), Nat.zero_add]
  · refine Finset.sum_congr rfl fun j hj => ?_
    rw [Finset.mem_Ico] at hj
    simp [show ¬ j < i by omega]
  · intro j hj
    rw [Finset.mem_Ico] at hj
    simp [show j < i by omega]

/-- The mask `j = i` turns the sum over all positions into `1`. -/
theorem sum_Icc_eq {i n : ℕ} (hi₁ : 1 ≤ i) (hi : i ≤ n) :
    ∑ j ∈ Finset.Icc 1 n, (if (!decide (i < j) && !decide (j < i)) then 1 else 0) = 1 := by
  rw [Finset.sum_eq_single_of_mem i (Finset.mem_Icc.mpr ⟨hi₁, hi⟩)]
  · simp
  · intro j _ hj
    have : i < j ∨ j < i := by omega
    rcases this with h | h <;> simp [h]

/-- The hypotheses of the masks are satisfiable: the position `2` of `[1, 3]`. -/
example : ∑ j ∈ Finset.Icc 1 3, (if (!decide (2 < j) && !decide (j < 2)) then 1 else 0) = 1 :=
  sum_Icc_eq (by omega) (by omega)

namespace Maj2

/-- `⊤` at the variable `v`, written `¬(v < v)`. -/
def topv (v : Var) : Maj2 σ := .neg (.lt v v)

/-- The other variable. -/
def _root_.Transformer.CRASP.Var.other : Var → Var
  | .x => .y
  | .y => .x

@[simp] theorem _root_.Transformer.CRASP.Var.other_ne (v : Var) : v.other ≠ v := by
  cases v <;> simp [Var.other]

@[simp] theorem _root_.Transformer.CRASP.Var.ne_other (v : Var) : v ≠ v.other :=
  (Var.other_ne v).symm

/-- `MAJ_u⟨⊤, l⟩`: a majority over a list, with a leading `⊤` so that the
list may be empty (Definition `def:MAJtwo` asks for `m ≥ 1` formulas). -/
def majList (u : Var) (l : List (Maj2 σ)) : Maj2 σ := .maj u l.length (topv u :: l).get

/-- The padded list whose majority is `mass α < mass β` (Appendix E, proof of
`thm:tlc_to_majtwo`): `β`, the negations of `α`, `|β|` copies of `⊤` and
`|α| + 1` copies of `⊥`. -/
def cmpList (u : Var) (α β : List (Maj2 σ)) : List (Maj2 σ) :=
  β ++ α.map .neg ++ List.replicate β.length (topv u) ++ List.replicate (α.length + 1) (.lt u u)

variable [DecidableEq σ]

@[simp] theorem sat_topv (w : List σ) (ξ : Var → ℕ) (v : Var) :
    (topv v : Maj2 σ).sat w ξ = true := by
  simp [topv, sat]

/-- The number of pairs `(j, g)`, `j` a position and `g ∈ l`, with `g`
satisfied at `u ↦ j`. -/
def mass (w : List σ) (ξ : Var → ℕ) (u : Var) (l : List (Maj2 σ)) : ℕ :=
  ∑ j ∈ Finset.Icc 1 w.length,
    (l.map fun g => if g.sat w (Function.update ξ u j) then 1 else 0).sum

theorem mass_append (w : List σ) (ξ : Var → ℕ) (u : Var) (l₁ l₂ : List (Maj2 σ)) :
    mass w ξ u (l₁ ++ l₂) = mass w ξ u l₁ + mass w ξ u l₂ := by
  simp only [mass, List.map_append, List.sum_append, Finset.sum_add_distrib]

theorem mass_map_neg (w : List σ) (ξ : Var → ℕ) (u : Var) (l : List (Maj2 σ)) :
    mass w ξ u (l.map .neg) + mass w ξ u l = l.length * w.length := by
  have h : ∀ (ξ' : Var → ℕ) (l : List (Maj2 σ)),
      ((l.map Maj2.neg).map fun g : Maj2 σ => if g.sat w ξ' then 1 else 0).sum +
        (l.map fun g : Maj2 σ => if g.sat w ξ' then 1 else 0).sum = l.length := by
    intro ξ' l
    induction l with
    | nil => rfl
    | cons g l ih =>
        simp only [List.map_cons, List.sum_cons, List.length_cons, sat]
        rcases Bool.eq_false_or_eq_true (g.sat w ξ') with hg | hg <;>
          simp only [hg, Bool.not_false, Bool.not_true, Bool.false_eq_true, ↓reduceIte] <;> omega
  rw [mass, mass, ← Finset.sum_add_distrib]
  simp only [h, Finset.sum_const, Nat.card_Icc, smul_eq_mul, Nat.add_sub_cancel, Nat.mul_comm]

theorem mass_replicate_topv (w : List σ) (ξ : Var → ℕ) (u : Var) (c : ℕ) :
    mass w ξ u (List.replicate c (topv u)) = c * w.length := by
  simp [mass, Nat.mul_comm]

theorem mass_replicate_lt (w : List σ) (ξ : Var → ℕ) (u : Var) (c : ℕ) :
    mass w ξ u (List.replicate c (.lt u u)) = 0 := by
  simp [mass, sat]

/-- **A majority over a list compares its mass with half of `|w|·(|l|+1)`.** -/
theorem sat_majList (w : List σ) (ξ : Var → ℕ) (u : Var) (l : List (Maj2 σ)) :
    (majList u l).sat w ξ =
      decide (w.length * (l.length + 1) < 2 * (w.length + mass w ξ u l)) := by
  have h : ∀ j, (∑ t : Fin (l.length + 1),
      if ((topv u :: l).get t).sat w (Function.update ξ u j) then 1 else 0) =
        1 + (l.map fun g => if g.sat w (Function.update ξ u j) then 1 else 0).sum := by
    intro j
    rw [Fin.sum_univ_succ]
    simp only [List.get_eq_getElem, Fin.val_zero, List.getElem_cons_zero, sat_topv,
      Fin.val_succ, List.getElem_cons_succ, ↓reduceIte]
    rw [Fin.sum_univ_fun_getElem l fun g => if g.sat w (Function.update ξ u j) then 1 else 0]
  simp only [majList, sat, h, Finset.sum_add_distrib, mass, Finset.sum_const, Nat.card_Icc,
    smul_eq_mul, Nat.mul_one, Nat.add_sub_cancel]
  rfl

/-- **The padded majority decides `mass α < mass β`** (Appendix E, proof of
`thm:tlc_to_majtwo`). -/
theorem sat_cmpList (w : List σ) (ξ : Var → ℕ) (u : Var) (α β : List (Maj2 σ)) :
    (majList u (cmpList u α β)).sat w ξ = decide (mass w ξ u α < mass w ξ u β) := by
  have hn := mass_map_neg w ξ u α
  rw [sat_majList, cmpList]
  simp only [mass_append, mass_replicate_topv, mass_replicate_lt, List.length_append,
    List.length_map, List.length_replicate]
  refine decide_eq_decide.mpr ?_
  constructor <;> intro h <;> nlinarith
