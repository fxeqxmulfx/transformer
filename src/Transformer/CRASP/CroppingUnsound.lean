/-
# The cropping lemmas are false as stated

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §4.4 `lem:cropping_oneway` with its proof in Appendix C.2, and
Appendix D `lem:cropping`.

Both cropping lemmas take a family of intervals `I` on which the PNPs of `φ`
are constant and promise a sub-family on which the minimal depth-1
subformulas of `φ` are constant too.  The proof reads a minimal depth-1
subformula as a comparison of affine functions of the prefix vector: "each
`◁#[ψ]` is `α x + β y`", since the depth-0 formula `ψ` is a Boolean
combination of `Q_a`, `Q_b` and PNPs that are constant on `I`.  That describes
the positions *inside* the interval only.  `◁#[ψ]` also counts the positions
before the prefix vector enters `I(n⃗)`, where the PNPs are unconstrained and
which the prefix vector does not determine.

`◁#[Q_a ∧ Π] < 1` with `Π(n⃗, j) = [j = 1]` makes this concrete.  `Π` is
constant on `I(n⃗) = [(1,1), n⃗]`, because a position whose prefix holds an `a`
and a `b` is at least the second.  The formula is its own minimal depth-1
subformula and says "the string does not start with `a`", which no nonempty
interval inside `I(n⃗)` decides: each of its points is the prefix vector of a
string starting with `a` and of one starting with `b`.
-/

import Transformer.CRASP.Middle

namespace Transformer
namespace CRASP

/-- `◁#[Q_a ∧ Π] < 1` with `Π(n⃗, j) = [j = 1]`: from the first position on,
"the string does not start with `a`".  It refutes `lem:cropping_oneway` and
`lem:cropping` (arXiv:2506.16055v3, §4.4 and Appendix D). -/
def Form.firstNotA : Form Bool :=
  .lt (.countL (.and (.sym false) (.pnp fun _ j => decide (j = 1)))) .one

/-- From the first position on, `firstNotA` reads the first letter
(arXiv:2506.16055v3, §2.2, the semantics of `◁#`). -/
theorem Form.sat_firstNotA (w : List Bool) {i : ℕ} (hi : 1 ≤ i) :
    Form.firstNotA.sat w i = decide (w[0]? ≠ some false) := by
  obtain ⟨i, rfl⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
  have h0 : (List.range' 2 i).filter
      (fun j => decide (w[j - 1]? = some false) && decide (j = 1)) = [] := by
    rw [List.filter_eq_nil_iff]
    intro j hj
    rw [List.mem_range'_1] at hj
    simp [show j ≠ 1 by omega]
  by_cases h : w[0]? = some false
  · simp [Form.firstNotA, Form.sat, Term.val, List.range'_succ, h0, h]
  · simp [Form.firstNotA, Form.sat, Term.val, List.range'_succ, h0, h]

/-- `firstNotA` is its own only minimal depth-1 subformula (§2.2). -/
theorem Form.minimalOne_firstNotA : Form.firstNotA.minimalOne = [Form.firstNotA] := by
  simp [Form.firstNotA, Form.minimalOne, Term.depth, Form.depth]

/-- The family `n⃗ ↦ [(1,1), n⃗]` is accommodating (Definition
`def:accommodating`): `s⃗` fits at `n⃗ = s⃗ + 1⃗`. -/
theorem accommodating_one : Accommodating (fun n => ⟨fun _ => 1, n⟩ : IntervalFamily Bool) :=
  fun s => ⟨fun a => s a + 1, fun _ => le_rfl⟩

/-- A position whose prefix vector lies in `[(1,1), n⃗]` is at least the
second: its prefix holds an `a` and a `b` (Definition `def:intervals`). -/
theorem two_le_of_prefixVec_mem {w : List Bool} {i : ℕ} {n : PVec Bool}
    (h : prefixVec w i ∈ (⟨fun _ => 1, n⟩ : Interval Bool)) : 2 ≤ i := by
  have ha : 1 ≤ parikh (w.take i) false := h.1 false
  have hb : 1 ≤ parikh (w.take i) true := h.1 true
  by_contra hi
  obtain rfl | rfl : i = 0 ∨ i = 1 := by omega
  · simp [parikh] at ha
  · rcases w with _ | ⟨x, w⟩
    · simp [parikh] at ha
    · cases x <;> simp [parikh] at ha hb

/-- The PNP of `firstNotA` is constant on `[(1,1), n⃗]`, where it is false
(Definition `def:constant`). -/
theorem pnpsConstantOn_firstNotA :
    PnpsConstantOn Form.firstNotA (fun n => ⟨fun _ => 1, n⟩ : IntervalFamily Bool) := by
  intro ψ hψ n w w' i i' _ _ _ _ _ _ hi hi'
  simp only [Form.firstNotA, Form.pnps, Term.pnps, List.nil_append, List.append_nil,
    List.mem_singleton] at hψ
  subst hψ
  have := two_le_of_prefixVec_mem hi
  have := two_le_of_prefixVec_mem hi'
  simp [Form.sat, show i ≠ 1 by omega, show i' ≠ 1 by omega]

/-- **No accommodating family inside `[(1,1), n⃗]` keeps `firstNotA`
constant.**  A point `v⃗` of such an interval is at least `(1,1)`, and it is
the prefix vector at position `v_a + v_b` of `a^{v_a} b^{v_b} a^{n_a-v_a}
b^{n_b-v_b}` and of `b^{v_b} a^{v_a} a^{n_a-v_a} b^{n_b-v_b}`, which start with
different letters (arXiv:2506.16055v3, §4.4, `lem:cropping_oneway`). -/
theorem not_minimalOneConstantOn_firstNotA {I : IntervalFamily Bool} (hI : Accommodating I)
    (hsub : ∀ n, (I n).Subset ⟨fun _ => 1, n⟩) : ¬ MinimalOneConstantOn Form.firstNotA I := by
  intro h
  obtain ⟨n, hn⟩ := hI 0
  have hlo : (I n).lo ∈ I n := ⟨fun _ => le_rfl, fun a => by simpa using hn a⟩
  obtain ⟨h1, h2⟩ := hsub n _ hlo
  set v := (I n).lo
  have hna : v false ≤ n false := h2 false
  have hnb : v true ≤ n true := h2 true
  obtain ⟨p, hp⟩ : ∃ p, v false = p + 1 := ⟨v false - 1, by have := h1 false; dsimp only at this; omega⟩
  obtain ⟨q, hq⟩ : ∃ q, v true = q + 1 := ⟨v true - 1, by have := h1 true; dsimp only at this; omega⟩
  set r := List.replicate (n false - v false) false ++ List.replicate (n true - v true) true
  have hpar : ∀ c : Bool, parikh (List.replicate (v c) c ++ List.replicate (v !c) (!c) ++ r) = n := by
    intro c
    funext d
    cases c <;> cases d <;> simp [r, parikh] <;> omega
  have hpre : ∀ c : Bool,
      prefixVec (List.replicate (v c) c ++ List.replicate (v !c) (!c) ++ r) (v false + v true) = v := by
    intro c
    rw [prefixVec, List.take_left' (by cases c <;> simp [Nat.add_comm])]
    funext d
    cases c <;> cases d <;> simp [parikh]
  have hlen : ∀ c : Bool,
      v false + v true ≤ (List.replicate (v c) c ++ List.replicate (v !c) (!c) ++ r).length := by
    intro c
    cases c
    · simp [r]
    · simp [r]; omega
  have := h _ (by rw [Form.minimalOne_firstNotA]; exact List.mem_singleton_self _) n _ _ _ _
    (hpar false) (hpar true) (by omega) (hlen false) (by omega) (hlen true)
    (by rw [hpre]; exact hlo) (by rw [hpre]; exact hlo)
  rw [Form.sat_firstNotA _ (by omega), Form.sat_firstNotA _ (by omega)] at this
  simp [hp, hq, List.replicate_succ] at this

/-- The hypotheses above are satisfiable: `1 ≤ i` at the first position, a
prefix `ab` inside `[(1,1), (1,1)]`, and the family `n⃗ ↦ [(1,1), n⃗]` itself. -/
example : 1 ≤ 1 ∧ prefixVec [false, true] 2 ∈ (⟨fun _ => 1, fun _ => 1⟩ : Interval Bool) ∧
    Accommodating (fun n => ⟨fun _ => 1, n⟩ : IntervalFamily Bool) ∧
    ∀ n, ((fun n => ⟨fun _ => 1, n⟩ : IntervalFamily Bool) n).Subset ⟨fun _ => 1, n⟩ :=
  ⟨le_rfl, ⟨fun a => by cases a <;> decide, fun a => by cases a <;> decide⟩, accommodating_one,
    fun _ _ hv => hv⟩

end CRASP
end Transformer
