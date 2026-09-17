/-
# Eliminating the sugar of Appendix A.3

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix A.3: the `?`-elimination lemma of Yang & Chiang and
`thm:strict`.

`FormX.elim` translates an extended formula into a plain one, through the
guarded pieces of `CRASP.ExtensionsPieces` for its terms (`TermX.pieces`).
The rewriting rules of the appendix become pieces:

* `◁#`, `▷#` and `1` are one piece under `⊤`;
* `#[φ] + (φ ? 1 : 0) = ◁#[φ] + ▷#[φ]`;
* `◁#_<[φ] + (φ ? 1 : 0) = ◁#[φ]`, except at position `0`, where both counts
  are `0`;
* `▷#_>[φ] + (φ ? 1 : 0) = ▷#[φ]`.

The translation agrees with its source at every position up to the length of
the word, by the counting cases of `CRASP.ExtensionsCounts` (`FormX.sat_elim`) and adds no depth (`FormX.depth_elim_le`), which
gives both lemmas at once (`exists_form_of_formX`).
-/

import Transformer.CRASP.Extensions
import Transformer.CRASP.ExtensionsCounts

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- The plain formula for an extended one (Appendix A.3). -/
def FormX.elim : FormX σ → Form σ
  | .sym a => .sym a
  | .lt t₁ t₂ => ltPieces t₁.pieces t₂.pieces
  | .neg φ => .neg φ.elim
  | .and φ₁ φ₂ => .and φ₁.elim φ₂.elim
  | .pnp π => .pnp π

/-- The guarded pieces of an extended term (Appendix A.3, its rewriting rules
for `#`, `◁#_<` and `▷#_>`). -/
def TermX.pieces : TermX σ → List (Form σ × Term σ × ℕ)
  | .countL φ => [(Form.topAt 0, .countL φ.elim, 0)]
  | .countR φ => [(Form.topAt 0, .countR φ.elim, 0)]
  | .countAll φ => [(.neg φ.elim, .add (.countL φ.elim) (.countR φ.elim), 0),
      (φ.elim, .add (.countL φ.elim) (.countR φ.elim), 1)]
  | .countLStrict φ => [(.neg φ.elim, .countL φ.elim, 0), (.and φ.elim Form.pos, .countL φ.elim, 1),
      (.and φ.elim (.neg Form.pos), .one, 1)]
  | .countRStrict φ => [(.neg φ.elim, .countR φ.elim, 0), (φ.elim, .countR φ.elim, 1)]
  | .cond φ t₁ t₂ => condPieces φ.elim t₁.pieces t₂.pieces
  | .add t₁ t₂ => addPieces t₁.pieces t₂.pieces
  | .one => [(Form.topAt 0, .one, 0)]

end

mutual

/-- The translation adds no depth (Appendix A.3: the sugar "does not affect the
depth of formulas"). -/
theorem FormX.depth_elim_le : ∀ φ : FormX σ, φ.elim.depth ≤ φ.depth
  | .sym _ => le_rfl
  | .lt t₁ t₂ => by
      rw [FormX.elim, FormX.depth]
      exact depth_ltPieces_le
        (fun x hx => (t₁.depth_pieces_le x hx).imp (le_max_of_le_left ·) (le_max_of_le_left ·))
        (fun x hx => (t₂.depth_pieces_le x hx).imp (le_max_of_le_right ·) (le_max_of_le_right ·))
  | .neg φ => by rw [FormX.elim, FormX.depth, Form.depth]; exact φ.depth_elim_le
  | .and φ₁ φ₂ => by
      rw [FormX.elim, FormX.depth, Form.depth]
      exact max_le_max φ₁.depth_elim_le φ₂.depth_elim_le
  | .pnp _ => le_rfl

/-- The pieces of a term add no depth. -/
theorem TermX.depth_pieces_le :
    ∀ t : TermX σ, ∀ x ∈ t.pieces, x.1.depth ≤ t.depth ∧ x.2.1.depth ≤ t.depth
  | .countL φ | .countR φ | .countAll φ | .countLStrict φ | .countRStrict φ => by
      have := φ.depth_elim_le
      simp only [TermX.pieces, TermX.depth, List.mem_cons, List.not_mem_nil, or_false]
      rintro x (rfl | rfl | rfl) <;>
        simp only [Form.depth, Term.depth, Form.depth_topAt, Form.depth_pos] <;> omega
  | .cond φ t₁ t₂ => by
      rw [TermX.pieces, TermX.depth]
      exact depth_condPieces_le (le_max_of_le_left φ.depth_elim_le)
        (fun x hx => (t₁.depth_pieces_le x hx).imp
          (le_max_of_le_right <| le_max_of_le_left ·) (le_max_of_le_right <| le_max_of_le_left ·))
        (fun x hx => (t₂.depth_pieces_le x hx).imp
          (le_max_of_le_right <| le_max_of_le_right ·) (le_max_of_le_right <| le_max_of_le_right ·))
  | .add t₁ t₂ => by
      rw [TermX.pieces, TermX.depth]
      exact depth_addPieces_le
        (fun x hx => (t₁.depth_pieces_le x hx).imp (le_max_of_le_left ·) (le_max_of_le_left ·))
        (fun x hx => (t₂.depth_pieces_le x hx).imp (le_max_of_le_right ·) (le_max_of_le_right ·))
  | .one => by simp [TermX.pieces, TermX.depth, Term.depth]

end

variable [DecidableEq σ]

mutual

/-- **The translation agrees with its source** at every position up to the
length of the word (Appendix A.3). -/
theorem FormX.sat_elim (w : List σ) :
    ∀ (φ : FormX σ) (i : ℕ), i ≤ w.length → φ.elim.sat w i = φ.sat w i
  | .sym _, _, _ => rfl
  | .lt t₁ t₂, i, hi => by
      rw [FormX.elim, FormX.sat, sat_ltPieces (t₁.covers_pieces w i hi) (t₂.covers_pieces w i hi)]
  | .neg φ, i, hi => by rw [FormX.elim, Form.sat, FormX.sat, φ.sat_elim w i hi]
  | .and φ₁ φ₂, i, hi => by
      rw [FormX.elim, Form.sat, FormX.sat, φ₁.sat_elim w i hi, φ₂.sat_elim w i hi]
  | .pnp _, _, _ => rfl

/-- **The pieces of a term represent its value** at every position up to the
length of the word (Appendix A.3). -/
theorem TermX.covers_pieces (w : List σ) :
    ∀ (t : TermX σ) (i : ℕ), i ≤ w.length → Covers t.pieces w i (t.val w i)
  | .countL φ, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_countL (φ.sat_elim w) hi
  | .countR φ, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_countR (φ.sat_elim w) hi
  | .countAll φ, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_countAll (φ.sat_elim w) hi
  | .countLStrict φ, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_countLStrict (φ.sat_elim w) hi
  | .countRStrict φ, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_countRStrict (φ.sat_elim w) hi
  | .cond φ t₁ t₂, i, hi => by
      rw [TermX.pieces, TermX.val, ← φ.sat_elim w i hi]
      exact covers_condPieces (t₁.covers_pieces w i hi) (t₂.covers_pieces w i hi)
  | .add t₁ t₂, i, hi => by
      rw [TermX.pieces, TermX.val]
      exact covers_addPieces (t₁.covers_pieces w i hi) (t₂.covers_pieces w i hi)
  | .one, _, _ => ⟨⟨_, List.mem_singleton_self _, Form.sat_topAt ..⟩, fun x hx _ => by
      rw [TermX.pieces, List.mem_singleton] at hx; subst hx; rfl⟩

end

/-- The hypothesis of `FormX.sat_elim` and `TermX.covers_pieces` is
satisfiable: the last position of a word. -/
example : [true, false].length ≤ [true, false].length := le_rfl

/-- **Appendix A.3, `thm:strict` together with the `?`-elimination lemma of
Yang & Chiang.**  Every formula written with the `?` operator, the unmasked
counting operator `#`, or the strict counting operators `◁#_<` and `▷#_>` can
be converted into one that uses none of them, defining the same language and
having the same depth. -/
theorem exists_form_of_formX (φ : FormX σ) :
    ∃ ψ : Form σ, ψ.lang = φ.lang ∧ ψ.depth = φ.depth :=
  ⟨.and φ.elim (Form.topAt φ.depth), by
    ext w
    simp only [Form.lang, FormX.lang, Form.models, FormX.models, Set.mem_ofPred_eq, Form.sat,
      φ.sat_elim w w.length le_rfl, Form.sat_topAt, Bool.and_true],
    by rw [Form.depth, Form.depth_topAt]; exact max_eq_right φ.depth_elim_le⟩

end CRASP
end Transformer
