/-
# `TL[◁#]` inside `TL[◁#]^pos`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F (`app:tlclpos`): `TL[◁#]^pos` is `TL[◁#]` with `MOD`
and `Y` added, so a formula of `TL[◁#]` is already one of `TL[◁#]^pos` — and
of both fragments `TL[◁#, MOD]` and `TL[◁#, Y]` — at the same depth.

`Form` carries more than `TL[◁#]`: the future count `▷#` and the Parikh
numerical predicates have no counterpart in `FormP`.  The embedding is
therefore stated for past-only, PNP-free formulas, as the existence of a
positional formula of the same depth that uses neither `MOD` nor `Y` and holds
at the same positions.
-/

import Transformer.CRASP.Positional

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

mutual

/-- A past-only, PNP-free formula has a positional counterpart of the same
depth, with neither `MOD` nor `Y`, satisfied at the same positions. -/
theorem Form.exists_formP : ∀ φ : Form σ, φ.past = true → φ.pnpFree = true →
    ∃ ψ : FormP σ, ψ.depth = φ.depth ∧ ψ.modFree = true ∧ ψ.prevFree = true ∧
      ∀ (w : List σ) (i : ℕ), ψ.sat w i = φ.sat w i
  | .sym a, _, _ => ⟨.sym a, rfl, rfl, rfl, fun _ _ => rfl⟩
  | .lt t₁ t₂, hp, hn => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hn
      obtain ⟨s₁, hd₁, hm₁, hy₁, hv₁⟩ := t₁.exists_termP hp.1 hn.1
      obtain ⟨s₂, hd₂, hm₂, hy₂, hv₂⟩ := t₂.exists_termP hp.2 hn.2
      refine ⟨.lt s₁ s₂, ?_, ?_, ?_, fun w i => ?_⟩
      · rw [FormP.depth, Form.depth, hd₁, hd₂]
      · rw [FormP.modFree, hm₁, hm₂, Bool.and_self]
      · rw [FormP.prevFree, hy₁, hy₂, Bool.and_self]
      · rw [FormP.sat, Form.sat, hv₁, hv₂]
  | .neg φ, hp, hn => by
      obtain ⟨ψ, hd, hm, hy, hs⟩ := φ.exists_formP hp hn
      exact ⟨.neg ψ, hd, hm, hy, fun w i => by rw [FormP.sat, Form.sat, hs]⟩
  | .and φ₁ φ₂, hp, hn => by
      simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hn
      obtain ⟨ψ₁, hd₁, hm₁, hy₁, hs₁⟩ := φ₁.exists_formP hp.1 hn.1
      obtain ⟨ψ₂, hd₂, hm₂, hy₂, hs₂⟩ := φ₂.exists_formP hp.2 hn.2
      refine ⟨.and ψ₁ ψ₂, ?_, ?_, ?_, fun w i => ?_⟩
      · rw [FormP.depth, Form.depth, hd₁, hd₂]
      · rw [FormP.modFree, hm₁, hm₂, Bool.and_self]
      · rw [FormP.prevFree, hy₁, hy₂, Bool.and_self]
      · rw [FormP.sat, Form.sat, hs₁, hs₂]
  | .pnp _, _, hn => absurd hn (by simp [Form.pnpFree])

/-- A past-only, PNP-free term has a positional counterpart of the same depth,
with neither `MOD` nor `Y`, taking the same values. -/
theorem Term.exists_termP : ∀ t : Term σ, t.past = true → t.pnpFree = true →
    ∃ s : TermP σ, s.depth = t.depth ∧ s.modFree = true ∧ s.prevFree = true ∧
      ∀ (w : List σ) (i : ℕ), s.val w i = t.val w i
  | .countL φ, hp, hn => by
      obtain ⟨ψ, hd, hm, hy, hs⟩ := φ.exists_formP hp hn
      refine ⟨.countL ψ, by rw [TermP.depth, Term.depth, hd], hm, hy, fun w i => ?_⟩
      rw [TermP.val, Term.val]
      exact congrArg List.length (List.filter_congr fun j _ => by rw [hs w j])
  | .countR _, hp, _ => absurd hp (by simp [Term.past])
  | .add t₁ t₂, hp, hn => by
      simp only [Term.past, Term.pnpFree, Bool.and_eq_true] at hp hn
      obtain ⟨s₁, hd₁, hm₁, hy₁, hv₁⟩ := t₁.exists_termP hp.1 hn.1
      obtain ⟨s₂, hd₂, hm₂, hy₂, hv₂⟩ := t₂.exists_termP hp.2 hn.2
      refine ⟨.add s₁ s₂, ?_, ?_, ?_, fun w i => ?_⟩
      · rw [TermP.depth, Term.depth, hd₁, hd₂]
      · rw [TermP.modFree, hm₁, hm₂, Bool.and_self]
      · rw [TermP.prevFree, hy₁, hy₂, Bool.and_self]
      · rw [TermP.val, Term.val, hv₁, hv₂]
  | .one, _, _ => ⟨.one, rfl, rfl, rfl, fun _ _ => rfl⟩

end

/-- The hypotheses of `Form.exists_formP` and `Term.exists_termP` are
satisfiable: `Q_a` and `◁#[Q_a]` are past-only and PNP-free. -/
example (a : σ) :
    ((Form.sym a).past = true ∧ (Form.sym a).pnpFree = true) ∧
      ((Term.countL (.sym a)).past = true ∧ (Term.countL (.sym a)).pnpFree = true) :=
  ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩⟩

/-- A language of `TL[◁#]_k` is defined by a `TL[◁#]^pos` formula of depth at
most `k` that uses neither `MOD` nor `Y`. -/
theorem DefinableL.exists_formP {L : Set (List σ)} {k : ℕ} (h : DefinableL L k) :
    ∃ ψ : FormP σ, ψ.depth ≤ k ∧ ψ.modFree = true ∧ ψ.prevFree = true ∧ ψ.lang = L := by
  obtain ⟨φ, ⟨hp, hn, hd⟩, rfl⟩ := h
  obtain ⟨ψ, hψd, hm, hy, hs⟩ := φ.exists_formP hp hn
  refine ⟨ψ, hψd ▸ hd, hm, hy, ?_⟩
  ext w
  show ψ.sat w w.length = true ↔ φ.sat w w.length = true
  rw [hs]

/-- `TL[◁#]_k` languages are `TL[◁#]^pos_k` languages. -/
theorem DefinableL.definablePos {L : Set (List σ)} {k : ℕ} (h : DefinableL L k) :
    DefinablePos L k :=
  let ⟨ψ, hd, _, _, hlang⟩ := h.exists_formP
  ⟨ψ, hd, hlang⟩

/-- `TL[◁#]_k` languages are `TL[◁#, MOD]_k` languages. -/
theorem DefinableL.definableMod {L : Set (List σ)} {k : ℕ} (h : DefinableL L k) :
    DefinableMod L k :=
  let ⟨ψ, hd, _, hy, hlang⟩ := h.exists_formP
  ⟨ψ, ⟨hy, hd⟩, hlang⟩

/-- `TL[◁#]_k` languages are `TL[◁#, Y]_k` languages. -/
theorem DefinableL.definableY {L : Set (List σ)} {k : ℕ} (h : DefinableL L k) :
    DefinableY L k :=
  let ⟨ψ, hd, hm, _, hlang⟩ := h.exists_formP
  ⟨ψ, ⟨hm, hd⟩, hlang⟩

/-- The hypothesis of the four `DefinableL` lemmas is satisfiable: `Σ*` is
defined at depth `0` by `¬(1 < 1)`. -/
example (k : ℕ) : DefinableL (σ := σ) Set.univ k := by
  refine ⟨.neg (.lt .one .one), ⟨rfl, rfl, Nat.zero_le k⟩, ?_⟩
  ext w
  simp [Form.lang, Form.models, Form.sat, Term.val]

end CRASP
end Transformer
