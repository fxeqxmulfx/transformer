/-
# The previous symbol is definable at no depth

In no paper.  arXiv:2506.16055v3, §1, recalls that "attention heads at lower
layers exhibit lower-order patterns (e.g., each symbol attends to the previous
symbol)".  Without positions no depth suffices for it: `Σ*bΣ`, the strings over
`{a, b}` whose second-to-last symbol is `b` (`secondLast`), is definable in
`TL[◁#]` at no depth (`not_definableL_secondLast`), so no future-masked rounded
transformer recognizes it (`not_recognizes_secondLast`).  Counting to the end
defines it at depth 2 (`definable_secondLast`), so it is, like PARITY
(`CRASP.Parity`, `CRASP.ParityTwoSided`), in `TL[◁#, ▷#]` and not in `TL[◁#]`;
and `Y Q_b` defines it at depth 0 in `TL[◁#]` with the operator `Y` of
Appendix F (`definableY_secondLast`), the logic into which `thm:rtfr_to_TLCly`
translates transformers with ALiBi.

The proof is the lower bound of the depth hierarchy (`CRASP.LowerBound`) cut
short.  After some word `u`, a formula of `TL[◁#]_{k+1}` reads at a position
only its letter and the numbers of `a`s and `b`s since `u`
(`exists_constOnStrip_altPlus`, `Form.sat_eq_of_count_eq`), and `u·aba` and
`u·baa` agree on those at their last position, while only the first has `b`
second to last (`exists_models_iff_secondLast`).  Both words are nonempty, so
no formula agrees with `Σ*bΣ` even on the nonempty strings
(`not_models_iff_secondLast`), which is what a reduction from recall needs
(`CRASP.LatestRecall`).
-/

import Transformer.CRASP.BoundedExists
import Transformer.CRASP.LowerBound
import Transformer.CRASP.Positional
import Transformer.CRASP.Transformers

namespace Transformer
namespace CRASP

/-- `Σ*bΣ` over `{a, b}`, with `b` = `true`: the second-to-last symbol is `b`. -/
def secondLast : Set (List Bool) := {w | ∃ u x, w = u ++ [true, x]}

/-- A string of length `m + 2` is its first `m` symbols followed by the last two. -/
theorem eq_take_append_pair {w : List Bool} {m : ℕ} (h : w.length = m + 2) :
    w = w.take m ++ [w[m], w[m + 1]] := by
  conv_lhs => rw [← List.take_append_drop m w, List.drop_eq_getElem_cons (i := m) (by omega),
    List.drop_eq_getElem_cons (i := m + 1) (by omega), List.drop_eq_nil_of_le (i := m + 1 + 1)
      (by omega)]

/-- The hypothesis of `eq_take_append_pair` is satisfiable: `ba`. -/
example : [true, false] = ([true, false] : List Bool).take 0 ++ [true, false] :=
  eq_take_append_pair (m := 0) rfl

/-- `w ∈ Σ*bΣ` iff `w` has length `m + 2` and holds `b` at index `m`. -/
theorem mem_secondLast_iff (w : List Bool) :
    w ∈ secondLast ↔ ∃ m, w.length = m + 2 ∧ w[m]? = some true := by
  constructor
  · rintro ⟨u, x, rfl⟩
    exact ⟨u.length, by simp, by simp⟩
  · rintro ⟨m, hm, hw⟩
    obtain ⟨_, hb⟩ := List.getElem?_eq_some_iff.mp hw
    have e := eq_take_append_pair hm
    rw [hb] at e
    exact ⟨_, _, e⟩

theorem append_mem_secondLast (u : List Bool) : u ++ [false, true, false] ∈ secondLast :=
  ⟨u ++ [false], false, by simp⟩

theorem append_not_mem_secondLast (u : List Bool) : u ++ [true, false, false] ∉ secondLast := by
  rintro ⟨u', x, h⟩
  rw [show u ++ [true, false, false] = u ++ [true] ++ [false, false] by simp] at h
  have h' := (List.append_inj' h rfl).2
  simp at h'

/-- **A formula of `TL[◁#]` confuses `u·aba` with `u·baa`.**  For each formula of
`TL[◁#]_K` there is `u` after which it holds at the end of `u·aba` iff at the
end of `u·baa`: past a word of `L_{K+1}` it reads only the letter and the
letter counts (`exists_constOnStrip_altPlus`, `Form.sat_eq_of_count_eq`), and
the two end in `a` with two `a`s and one `b`. -/
theorem exists_models_iff_secondLast (K : ℕ) (φ : Form Bool) (hφ : φ ∈ TLCl Bool K) :
    ∃ u : List Bool, φ.models (u ++ [false, true, false]) ↔ φ.models (u ++ [true, false, false]) := by
  obtain ⟨hp, hf, hd⟩ := hφ
  obtain ⟨u, -, hS⟩ := exists_constOnStrip_altPlus K φ.countSubs
    (fun ψ hψ => Form.mem_TLCl_of_mem_countSubs φ ⟨hp, hf, hd.trans K.le_succ⟩ hψ) 2
  have e := Form.sat_eq_of_count_eq u 2 (m := [false, true, false]) (m' := [true, false, false])
    (i := 2) (i' := 2) (by decide) (by decide) (by simp) (by simp) (by decide) (by decide) rfl φ hp
    hf hS
  refine ⟨u, ?_⟩
  have e' : φ.sat (u ++ [false, true, false]) (u ++ [false, true, false]).length =
      φ.sat (u ++ [true, false, false]) (u ++ [true, false, false]).length := by
    rw [List.length_append, List.length_append]
    exact e
  rw [Form.models, Form.models, e']

/-- The hypothesis of `exists_models_iff_secondLast` is satisfiable: `Q_b` has
depth `0`. -/
example : (Form.sym true : Form Bool) ∈ TLCl Bool 0 := ⟨rfl, rfl, le_rfl⟩

/-- **No formula of `TL[◁#]` agrees with `Σ*bΣ` on the nonempty strings**, at
any depth: it would tell `u·aba` from `u·baa` (`exists_models_iff_secondLast`). -/
theorem not_models_iff_secondLast (K : ℕ) (φ : Form Bool) (hφ : φ ∈ TLCl Bool K) :
    ¬ ∀ w : List Bool, w ≠ [] → (φ.models w ↔ w ∈ secondLast) := by
  intro h
  obtain ⟨u, hu⟩ := exists_models_iff_secondLast K φ hφ
  exact append_not_mem_secondLast u ((h _ (by simp)).1
    (hu.1 ((h _ (by simp)).2 (append_mem_secondLast u))))

/-- The hypothesis of `not_models_iff_secondLast` is satisfiable: `Q_b` has
depth `0`. -/
example : ¬ ∀ w : List Bool, w ≠ [] → ((Form.sym true : Form Bool).models w ↔ w ∈ secondLast) :=
  not_models_iff_secondLast 0 _ ⟨rfl, rfl, le_rfl⟩

/-- **`Σ*bΣ` is definable in `TL[◁#]` at no depth.** -/
theorem not_definableL_secondLast (K : ℕ) : ¬ DefinableL secondLast K := by
  rintro ⟨φ, hφ, hL⟩
  exact not_models_iff_secondLast K φ hφ fun w _ => Set.ext_iff.1 hL w

/-- **No future-masked rounded transformer recognizes `Σ*bΣ`**, whatever its
depth, width and precision: by `thm:transformer_equivalence` of
arXiv:2506.16055v3 it would make `Σ*bΣ` definable in `TL[◁#]`, against
`not_definableL_secondLast`. -/
theorem not_recognizes_secondLast {p s d k : ℕ} (T : RTfr (Option Bool) p s d k) :
    ¬ T.Recognizes secondLast :=
  fun hT => not_definableL_secondLast k
    ((definableL_iff_recognizes secondLast k).mpr ⟨p, s, d, T, hT⟩)

/-- `Q_b ∧ ▷#[⊤] = 2`, with `⊤` written `¬(1 < 1)`: the position holds `b`
and is the second to last. -/
def bSecondLast : Form Bool :=
  .and (.sym true) (Form.eq (.countR (.neg (.lt .one .one))) (Term.ofPos 1))

/-- **`Σ*bΣ` is definable in `TL[◁#, ▷#]` at depth 2**, by
`1 ≤ ◁#[Q_b ∧ ▷#[⊤] = 2]`: some position holding `b` has two positions from
it to the end. -/
theorem definable_secondLast : Definable secondLast 2 := by
  refine ⟨Form.exAt bSecondLast, ⟨rfl, by decide⟩, Set.ext fun w => ?_⟩
  change (Form.exAt bSecondLast).sat w w.length = true ↔ w ∈ secondLast
  simp only [Form.sat_exAt, mem_secondLast_iff, bSecondLast, Form.sat, Form.sat_eq, Term.val,
    Term.val_ofPos, lt_self_iff_false, decide_false, Bool.not_false, List.filter_true,
    List.length_range', Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨j, h₁, h₂, hb, hj⟩
    exact ⟨j - 1, by omega, hb⟩
  · rintro ⟨m, hm, hb⟩
    exact ⟨m + 1, by omega, by omega, by simpa using hb, by omega⟩

/-- **`Σ*bΣ` is definable at depth 0 with `Y`**, by `Y Q_b`, in the logic
`TL[◁#]` with the operator `Y` of Appendix F (`app:tlclpos`), into which
`thm:rtfr_to_TLCly` translates transformers with ALiBi: `Y` is what `TL[◁#]`
lacks for it. -/
theorem definableY_secondLast : DefinableY secondLast 0 := by
  refine ⟨.prev (.sym true), ⟨rfl, by decide⟩, Set.ext fun w => ?_⟩
  change (FormP.prev (.sym true)).sat w w.length = true ↔ w ∈ secondLast
  simp only [mem_secondLast_iff, FormP.sat, Bool.and_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨h, hb⟩
    exact ⟨w.length - 2, by omega, by rwa [show w.length - 2 = w.length - 1 - 1 by omega]⟩
  · rintro ⟨m, hm, hb⟩
    exact ⟨by omega, by rwa [hm, show m + 2 - 1 - 1 = m by omega]⟩

end CRASP
end Transformer
