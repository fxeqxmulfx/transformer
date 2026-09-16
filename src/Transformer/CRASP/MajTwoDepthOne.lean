/-
# Closed `MAJ²` formulas of depth one are blind to the order of symbols

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, `def:MAJtwo` and `def:depth_MAJtwo`.

Every atom of `MAJ²` names a variable, so a closed formula has to bind it, and
at depth `1` there is room for exactly one majority quantifier: the body of
that quantifier has depth `0`, mentions only the bound variable, and therefore
reads a position solely through the symbol standing at it.  A single majority
quantifier then counts symbols, and counting cannot tell `ba` from `ab`.

That is what `sat_eq_of_closed_depth_le_one` says, and it is the reason the
first inclusion of `thm:logical_inclusions` needs `k ≥ 1`: at `k = 0` the
language `Q_a` defines — "the last symbol is `a`" — is `TL[◁#,▷#]_0`-definable
and distinguishes the two strings, so no closed `MAJ²_1` formula defines it.
The refutation of the `k = 0` reading is `not_forall_closed_majTwo_of_definable`;
the corrected statement is `CRASP.exists_closed_majTwo_of_definable`.
-/

import Transformer.CRASP.MajTwo

namespace Transformer
namespace CRASP

/-- `ba`, the two-symbol string ending in `a = false`. -/
def wordBA : List Bool := [true, false]

/-- `ab`, the two-symbol string ending in `b = true`. -/
def wordAB : List Bool := [false, true]

/-- The transposition of the two positions of a string of length `2`. -/
def swapPos (i : ℕ) : ℕ := if i = 1 then 2 else 1

@[simp] theorem swapPos_one : swapPos 1 = 2 := rfl

@[simp] theorem swapPos_two : swapPos 2 = 1 := rfl

namespace Maj2

/-- The two strings carry the same symbols in the opposite order: position `i`
of `ba` holds what position `swapPos i` of `ab` holds. -/
theorem getElem?_swapPos (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ 2) :
    wordBA[i - 1]? = wordAB[swapPos i - 1]? := by
  interval_cases i <;> rfl

/-- **A depth-`0` body reads only the symbol at the bound variable.**  If every
free variable of `φ` is sent to `i` on the left and to `swapPos i` on the
right, the two satisfaction values agree: the atoms `Q_σ(v)` see the same
symbol by `getElem?_swapPos`, and the atoms `v < u` are false on both sides
because all free variables share one value. -/
theorem sat_swap_of_depth_eq_zero :
    ∀ (φ : Maj2 Bool), φ.depth = 0 → ∀ i : ℕ, 1 ≤ i → i ≤ 2 →
      ∀ ξ₁ ξ₂ : Var → ℕ, (∀ v, φ.freeIn v = true → ξ₁ v = i ∧ ξ₂ v = swapPos i) →
        φ.sat wordBA ξ₁ = φ.sat wordAB ξ₂ := by
  intro φ
  induction φ with
  | sym a v =>
    intro _ i hi₁ hi₂ ξ₁ ξ₂ hξ
    obtain ⟨h₁, h₂⟩ := hξ v (by simp [freeIn])
    simp only [sat, h₁, h₂, getElem?_swapPos i hi₁ hi₂]
  | lt v u =>
    intro _ i _ _ ξ₁ ξ₂ hξ
    obtain ⟨h₁, _⟩ := hξ v (by simp [freeIn])
    obtain ⟨h₃, _⟩ := hξ u (by simp [freeIn])
    obtain ⟨_, h₂⟩ := hξ v (by simp [freeIn])
    obtain ⟨_, h₄⟩ := hξ u (by simp [freeIn])
    simp [sat, h₁, h₂, h₃, h₄]
  | neg φ ih =>
    intro hd i hi₁ hi₂ ξ₁ ξ₂ hξ
    rw [sat, sat, ih hd i hi₁ hi₂ ξ₁ ξ₂ fun v hv => hξ v (by simpa [freeIn] using hv)]
  | and φ₁ φ₂ ih₁ ih₂ =>
    intro hd i hi₁ hi₂ ξ₁ ξ₂ hξ
    have hd₁ : φ₁.depth = 0 := by simp only [depth] at hd; omega
    have hd₂ : φ₂.depth = 0 := by simp only [depth] at hd; omega
    rw [sat, sat,
      ih₁ hd₁ i hi₁ hi₂ ξ₁ ξ₂ fun v hv => hξ v (by simp [freeIn, hv]),
      ih₂ hd₂ i hi₁ hi₂ ξ₁ ξ₂ fun v hv => hξ v (by simp [freeIn, hv])]
  | maj v m ψ _ =>
    intro hd
    simp only [depth] at hd
    omega

/-- **Closed `MAJ²` formulas of depth at most `1` do not distinguish `ba` from
`ab`.**  The only such formulas are Boolean combinations of single majority
quantifiers whose bodies have depth `0` and are free in the bound variable
alone, so `sat_swap_of_depth_eq_zero` applies inside, and summing over the two
positions absorbs the transposition. -/
theorem sat_eq_of_closed_depth_le_one :
    ∀ (φ : Maj2 Bool), φ.depth ≤ 1 → φ.Closed →
      ∀ ξ₁ ξ₂ : Var → ℕ, φ.sat wordBA ξ₁ = φ.sat wordAB ξ₂ := by
  intro φ
  induction φ with
  | sym a v =>
    intro _ hc _ _
    have := hc v
    simp [freeIn] at this
  | lt v u =>
    intro _ hc _ _
    have := hc v
    simp [freeIn] at this
  | neg φ ih =>
    intro hd hc ξ₁ ξ₂
    rw [sat, sat, ih hd (fun v => hc v) ξ₁ ξ₂]
  | and φ₁ φ₂ ih₁ ih₂ =>
    intro hd hc ξ₁ ξ₂
    have hd₁ : φ₁.depth ≤ 1 := by simp only [depth] at hd; omega
    have hd₂ : φ₂.depth ≤ 1 := by simp only [depth] at hd; omega
    have hc₁ : φ₁.Closed := fun v => by have := hc v; simp [freeIn] at this; exact this.1
    have hc₂ : φ₂.Closed := fun v => by have := hc v; simp [freeIn] at this; exact this.2
    rw [sat, sat, ih₁ hd₁ hc₁ ξ₁ ξ₂, ih₂ hd₂ hc₂ ξ₁ ξ₂]
  | maj v m ψ _ =>
    intro hd hc ξ₁ ξ₂
    have hd₀ : ∀ t : Fin (m + 1), (ψ t).depth = 0 := by
      intro t
      have hle := Finset.le_sup (f := fun t : Fin (m + 1) => (ψ t).depth) (Finset.mem_univ t)
      simp only [depth] at hd
      omega
    have hfree : ∀ (t : Fin (m + 1)) (u : Var), u ≠ v → (ψ t).freeIn u = false := by
      intro t u huv
      have hvu : v ≠ u := fun h => huv h.symm
      have := hc u
      simp only [freeIn, hvu, ite_false, decide_eq_false_iff_not, not_exists,
        Bool.not_eq_true] at this
      exact this t
    have key : ∀ (t : Fin (m + 1)) (i : ℕ), 1 ≤ i → i ≤ 2 →
        (ψ t).sat wordBA (Function.update ξ₁ v i)
          = (ψ t).sat wordAB (Function.update ξ₂ v (swapPos i)) := by
      intro t i hi₁ hi₂
      refine sat_swap_of_depth_eq_zero (ψ t) (hd₀ t) i hi₁ hi₂ _ _ fun u hu => ?_
      by_cases huv : u = v
      · subst huv
        exact ⟨Function.update_self _ _ _, Function.update_self _ _ _⟩
      · rw [hfree t u huv] at hu
        exact absurd hu Bool.noConfusion
    have hpair : ∀ f : ℕ → ℕ, ∑ i ∈ Finset.Icc 1 2, f i = f 1 + f 2 := by
      intro f
      rw [show (Finset.Icc 1 2 : Finset ℕ) = {1, 2} from rfl]
      exact Finset.sum_pair (by norm_num)
    have e₁ : ∑ t : Fin (m + 1), (if (ψ t).sat wordBA (Function.update ξ₁ v 1) then 1 else 0)
        = ∑ t : Fin (m + 1), (if (ψ t).sat wordAB (Function.update ξ₂ v 2) then 1 else 0) :=
      Finset.sum_congr rfl fun t _ => by rw [key t 1 le_rfl one_le_two, swapPos_one]
    have e₂ : ∑ t : Fin (m + 1), (if (ψ t).sat wordBA (Function.update ξ₁ v 2) then 1 else 0)
        = ∑ t : Fin (m + 1), (if (ψ t).sat wordAB (Function.update ξ₂ v 1) then 1 else 0) :=
      Finset.sum_congr rfl fun t _ => by rw [key t 2 one_le_two le_rfl, swapPos_two]
    simp only [sat, show wordBA.length = 2 from rfl, show wordAB.length = 2 from rfl,
      hpair, e₁, e₂, decide_eq_decide]
    omega

end Maj2

/-- **The first inclusion of `thm:logical_inclusions` fails at `k = 0`.**

`Q_b` is a `TL[◁#,▷#]_0` formula, and the language it defines — the strings
whose last symbol is `b` — contains `ab` but not `ba`.  By
`Maj2.sat_eq_of_closed_depth_le_one` no closed `MAJ²_1` formula separates those
two strings, so none defines that language.  This is what pins the hypothesis
`0 < k` in `exists_closed_majTwo_of_definable`. -/
theorem not_forall_closed_majTwo_of_definable :
    ¬ ∀ L : Set (List Bool), Definable L 0 →
        ∃ φ' ∈ MajTwo Bool 1, φ'.Closed ∧ φ'.lang = L := by
  intro h
  obtain ⟨φ', hdepth, hclosed, hlang⟩ :=
    h (Form.sym true).lang ⟨.sym true, ⟨rfl, le_rfl⟩, rfl⟩
  have hAB : φ'.sat wordAB (fun _ => 0) = true := by
    have : wordAB ∈ φ'.lang := by
      rw [hlang]
      show (Form.sym true).sat wordAB wordAB.length = true
      rfl
    exact this
  have hBA : φ'.sat wordBA (fun _ => 0) ≠ true := by
    intro hsat
    have : wordBA ∈ φ'.lang := hsat
    rw [hlang] at this
    exact Bool.noConfusion (this : (Form.sym true).sat wordBA wordBA.length = true)
  exact hBA
    ((Maj2.sat_eq_of_closed_depth_le_one φ' hdepth hclosed (fun _ => 0) (fun _ => 0)).trans hAB)

end CRASP
end Transformer
