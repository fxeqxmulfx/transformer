/-
# A past-only formula reads a prefix, and the formulas under its counts

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.2 and §4.4.

Two facts that every level of the depth-hierarchy argument uses.  A past-only
formula without PNPs does not look ahead: at a position `1 ≤ i ≤ |u|` of
`u ++ v` it has its value on `u` (`Form.sat_append`), since `◁#` counts the
positions up to the current one and `Q_σ` reads the current one.  A PNP does
look ahead, as it reads the Parikh vector of the whole string, and so does
`▷#`.

A level of counting is peeled by looking at the formulas under the counts of a
formula, `Form.countSubs`: those of a formula of `TL[◁#]_{d+1}` lie in
`TL[◁#]_d` (`Form.mem_TLCl_of_mem_countSubs`).
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

mutual

/-- The formulas under the counts of a formula: every `ψ` of a subterm `◁#[ψ]`
or `▷#[ψ]`, at any nesting (§4.4, where a level of counting is peeled by making
the subformulas of the next depth constant). -/
def Form.countSubs : Form σ → List (Form σ)
  | .sym _ => []
  | .lt t₁ t₂ => t₁.countSubs ++ t₂.countSubs
  | .neg φ => φ.countSubs
  | .and φ₁ φ₂ => φ₁.countSubs ++ φ₂.countSubs
  | .pnp _ => []

/-- The formulas under the counts of a term (§4.4). -/
def Term.countSubs : Term σ → List (Form σ)
  | .countL φ => φ :: φ.countSubs
  | .countR φ => φ :: φ.countSubs
  | .add t₁ t₂ => t₁.countSubs ++ t₂.countSubs
  | .one => []

end

mutual

/-- **The formulas under the counts of a formula of `TL[◁#]_{d+1}` lie in
`TL[◁#]_d`** (Definition `def:TLC_depth`: a count adds one to the depth of what
it counts). -/
theorem Form.mem_TLCl_of_mem_countSubs {d : ℕ} {ψ : Form σ} :
    ∀ φ : Form σ, φ ∈ TLCl σ (d + 1) → ψ ∈ φ.countSubs → ψ ∈ TLCl σ d
  | .sym _, _, h => by rw [Form.countSubs] at h; exact absurd h List.not_mem_nil
  | .lt t₁ t₂, ⟨hp, hf, hd⟩, h => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs, List.mem_append] at h
      exact h.elim (t₁.mem_TLCl_of_mem_countSubs hp.1 hf.1 (by omega))
        (t₂.mem_TLCl_of_mem_countSubs hp.2 hf.2 (by omega))
  | .neg φ, ⟨hp, hf, hd⟩, h => by
      rw [Form.past] at hp
      rw [Form.pnpFree] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs] at h
      exact Form.mem_TLCl_of_mem_countSubs φ ⟨hp, hf, hd⟩ h
  | .and φ₁ φ₂, ⟨hp, hf, hd⟩, h => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs, List.mem_append] at h
      exact h.elim (Form.mem_TLCl_of_mem_countSubs φ₁ ⟨hp.1, hf.1, by omega⟩)
        (Form.mem_TLCl_of_mem_countSubs φ₂ ⟨hp.2, hf.2, by omega⟩)
  | .pnp _, _, h => by rw [Form.countSubs] at h; exact absurd h List.not_mem_nil

/-- The formulas under the counts of a past-only term without PNPs of depth at
most `d + 1` lie in `TL[◁#]_d` (Definition `def:TLC_depth`). -/
theorem Term.mem_TLCl_of_mem_countSubs {d : ℕ} {ψ : Form σ} :
    ∀ t : Term σ, t.past = true → t.pnpFree = true → t.depth ≤ d + 1 → ψ ∈ t.countSubs →
      ψ ∈ TLCl σ d
  | .countL φ, hp, hf, hd, h => by
      rw [Term.past] at hp
      rw [Term.pnpFree] at hf
      rw [Term.depth] at hd
      rw [Term.countSubs, List.mem_cons] at h
      rcases h with rfl | h
      · exact ⟨hp, hf, by omega⟩
      · exact Form.mem_TLCl_of_mem_countSubs φ ⟨hp, hf, by omega⟩ h
  | .countR _, hp, _, _, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf, hd, h => by
      rw [Term.past, Bool.and_eq_true] at hp
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      rw [Term.depth] at hd
      rw [Term.countSubs, List.mem_append] at h
      exact h.elim (t₁.mem_TLCl_of_mem_countSubs hp.1 hf.1 (by omega))
        (t₂.mem_TLCl_of_mem_countSubs hp.2 hf.2 (by omega))
  | .one, _, _, _, h => by rw [Term.countSubs] at h; exact absurd h List.not_mem_nil

end

/-- The hypotheses of `Form.mem_TLCl_of_mem_countSubs` and
`Term.mem_TLCl_of_mem_countSubs` are satisfiable: `Q_a` lies under the count of
`◁#[Q_a] < 1`, a formula of `TL[◁#]_1`. -/
example (a : σ) :
    (Form.lt (.countL (.sym a)) .one : Form σ) ∈ TLCl σ (0 + 1) ∧
      Form.sym a ∈ (Form.lt (.countL (.sym a)) .one : Form σ).countSubs := by
  refine ⟨⟨rfl, rfl, by simp [Form.depth, Term.depth]⟩, ?_⟩
  rw [Form.countSubs, Term.countSubs, Term.countSubs]
  exact List.mem_append_left _ (List.mem_cons_self ..)

variable [DecidableEq σ]

mutual

/-- **A past-only formula without PNPs does not look ahead.**  At a position
`1 ≤ i ≤ |u|` of `u ++ v` it has its value on `u` (Definition
`def:TLC_semantics`: `◁#` counts the positions up to the current one, and `Q_σ`
reads the current one). -/
theorem Form.sat_append (u v : List σ) :
    ∀ φ : Form σ, φ.past = true → φ.pnpFree = true → ∀ i, 1 ≤ i → i ≤ u.length →
      φ.sat (u ++ v) i = φ.sat u i
  | .sym _, _, _, i, h₁, h₂ => by
      rw [Form.sat, Form.sat, List.getElem?_append_left (by omega)]
  | .lt t₁ t₂, hp, hf, i, _, h₂ => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.sat, Form.sat, t₁.val_append u v hp.1 hf.1 i h₂, t₂.val_append u v hp.2 hf.2 i h₂]
  | .neg φ, hp, hf, i, h₁, h₂ => by
      rw [Form.past] at hp
      rw [Form.pnpFree] at hf
      rw [Form.sat, Form.sat, φ.sat_append u v hp hf i h₁ h₂]
  | .and φ₁ φ₂, hp, hf, i, h₁, h₂ => by
      rw [Form.past, Bool.and_eq_true] at hp
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.sat, Form.sat, φ₁.sat_append u v hp.1 hf.1 i h₁ h₂,
        φ₂.sat_append u v hp.2 hf.2 i h₁ h₂]
  | .pnp _, _, hf, _, _, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- A past-only term without PNPs does not look ahead either: at `i ≤ |u|` it
has its value on `u` (Definition `def:TLC_semantics`). -/
theorem Term.val_append (u v : List σ) :
    ∀ t : Term σ, t.past = true → t.pnpFree = true → ∀ i, i ≤ u.length →
      t.val (u ++ v) i = t.val u i
  | .countL φ, hp, hf, i, h => by
      rw [Term.past] at hp
      rw [Term.pnpFree] at hf
      rw [Term.val, Term.val, List.filter_congr fun j hj =>
        φ.sat_append u v hp hf j (List.mem_range'_1.mp hj).1
          (by have := (List.mem_range'_1.mp hj).2; omega)]
  | .countR _, hp, _, _, _ => by rw [Term.past] at hp; exact absurd hp Bool.false_ne_true
  | .add t₁ t₂, hp, hf, i, h => by
      rw [Term.past, Bool.and_eq_true] at hp
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      rw [Term.val, Term.val, t₁.val_append u v hp.1 hf.1 i h, t₂.val_append u v hp.2 hf.2 i h]
  | .one, _, _, _, _ => rfl

end

/-- The hypotheses of `Form.sat_append` and `Term.val_append` are satisfiable:
`Q_a` at the only position of `a`. -/
example (a : σ) :
    (Form.sym a : Form σ).past = true ∧ (Form.sym a : Form σ).pnpFree = true ∧ 1 ≤ 1 ∧
      1 ≤ [a].length :=
  ⟨rfl, rfl, le_rfl, le_rfl⟩

end CRASP
end Transformer
