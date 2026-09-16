/-
# Formulas constant on the middles of a frame

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D: `lem:cropping`, and the remark that opens the
appendix, "if the Parikh vector of a word is fixed we can rewrite `▷#` in
terms of `◁#` and `n⃗`".

`lem:cropping` is false as stated (`cropping_unsound`): a count reads the
positions before the interval, and a future count those after it too.  Here
both are fixed.  A *frame* is a prefix `u`, a suffix `v` and a Parikh vector
`(X, Y)`; its *middles* are the words `m` with `X` letters `a` and `Y` letters
`b`, and there is always one (`exists_middle`).  `ConstOnMiddle ψ u v X Y` says
that on `u ++ m ++ v` the value of `ψ` at a position of `m` depends on the
letter alone, and at a position of `u` or `v` does not depend on `m` at all.  A
formula of depth `0` without PNPs is constant on the middles of every frame
(`Form.constOnMiddle_of_depth_eq_zero`).

A level of counting is peeled as in `TL[◁#]`: the formulas under the counts of
a formula of `TL[◁#,▷#]_{d+1}` lie in `TL[◁#,▷#]_d`
(`Form.mem_TLC_of_mem_countSubs`).
-/

import Mathlib.Data.Bool.Count
import Transformer.CRASP.Affine

namespace Transformer
namespace CRASP

/-- The position `|u| + i + 1` of `u ++ m ++ v` holds `m[i]` (Definition
`def:TLC_semantics`, positions numbered from `1`). -/
theorem getElem?_frame {α : Type*} (u m v : List α) {i : ℕ} (hi : i < m.length) :
    (u ++ m ++ v)[u.length + i + 1 - 1]? = some m[i] := by
  rw [List.getElem?_append_left (by rw [List.length_append]; omega),
    getElem?_append_length_add u m hi]

/-- The hypothesis of `getElem?_frame` is satisfiable: `0 < |a|`. -/
example : 0 < [false].length := Nat.one_pos

/-- Outside the middle, `u ++ m ++ v` and `u ++ m' ++ v` hold the same letters
when `m` and `m'` have the same length (Definition `def:TLC_semantics`). -/
theorem getElem?_frame_outside {α : Type*} (u v : List α) {m m' : List α}
    (hm : m.length = m'.length) {j : ℕ} (hj : 1 ≤ j)
    (hj' : j ≤ u.length ∨ u.length + m.length < j) :
    (u ++ m ++ v)[j - 1]? = (u ++ m' ++ v)[j - 1]? := by
  rcases hj' with hj' | hj'
  · rw [List.append_assoc, List.append_assoc, List.getElem?_append_left (by omega),
      List.getElem?_append_left (by omega)]
  · rw [List.getElem?_append_right (by rw [List.length_append]; omega),
      List.getElem?_append_right (by rw [List.length_append]; omega), List.length_append,
      List.length_append, hm]

/-- The hypotheses of `getElem?_frame_outside` are satisfiable: the first
position of `a·a` and of `a·b`. -/
example : [false].length = [true].length ∧ 1 ≤ 1 ∧ (1 ≤ [false].length ∨ [false].length + 1 < 1) :=
  ⟨rfl, le_rfl, Or.inl le_rfl⟩

/-- A formula of depth `0` without PNPs has the same value wherever the same
letter stands (Definitions `def:TLC_semantics` and `def:TLC_depth`). -/
theorem Form.sat_eq_of_pnpFree_depth_eq_zero {σ : Type*} [DecidableEq σ] {w w' : List σ}
    {i i' : ℕ} (h : w[i - 1]? = w'[i' - 1]?) :
    ∀ φ : Form σ, φ.pnpFree = true → φ.depth = 0 → φ.sat w i = φ.sat w' i'
  | .sym _, _, _ => by rw [Form.sat, Form.sat, h]
  | .lt t₁ t₂, _, hd => by
      rw [Form.depth] at hd
      rw [Form.sat, Form.sat, t₁.val_eq_of_depth_eq_zero w w' i i' (by omega),
        t₂.val_eq_of_depth_eq_zero w w' i i' (by omega)]
  | .neg φ, hf, hd => by
      rw [Form.pnpFree] at hf
      rw [Form.depth] at hd
      rw [Form.sat, Form.sat, Form.sat_eq_of_pnpFree_depth_eq_zero h φ hf hd]
  | .and φ₁ φ₂, hf, hd => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at hd
      rw [Form.sat, Form.sat, Form.sat_eq_of_pnpFree_depth_eq_zero h φ₁ hf.1 (by omega),
        Form.sat_eq_of_pnpFree_depth_eq_zero h φ₂ hf.2 (by omega)]
  | .pnp _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The hypotheses of `Form.sat_eq_of_pnpFree_depth_eq_zero` are satisfiable:
`Q_a` at the first positions of `a` and `ab`. -/
example : ([false] : List Bool)[1 - 1]? = ([false, true] : List Bool)[1 - 1]? ∧
    (Form.sym false : Form Bool).pnpFree = true ∧ (Form.sym false : Form Bool).depth = 0 :=
  ⟨rfl, rfl, rfl⟩

/-- A **frame** `(u, v, X, Y)` holds `ψ` constant on its middles.  On
`u ++ m ++ v`, where `m` holds `X` letters `a` (`false`) and `Y` letters `b`
(`true`), the value of `ψ` at the position of `m[i]` is `g m[i]`, and at a
position of `u` or of `v` it is the same for every such `m` (Appendix D,
"constant on `I`", with the positions around the interval and the Parikh
vector of the whole word fixed). -/
def ConstOnMiddle (ψ : Form Bool) (u v : List Bool) (X Y : ℕ) : Prop :=
  (∃ g : Bool → Bool, ∀ m : List Bool, m.count false = X → m.count true = Y →
    ∀ i (hi : i < m.length), ψ.sat (u ++ m ++ v) (u.length + i + 1) = g m[i]) ∧
  ∀ m m' : List Bool, m.count false = X → m.count true = Y → m'.count false = X →
    m'.count true = Y → ∀ j, 1 ≤ j → j ≤ u.length ∨ u.length + X + Y < j →
      ψ.sat (u ++ m ++ v) j = ψ.sat (u ++ m' ++ v) j

/-- A middle of `(X, Y)` has length `X + Y`. -/
theorem length_eq_of_count {m : List Bool} {X Y : ℕ} (hX : m.count false = X)
    (hY : m.count true = Y) : m.length = X + Y := by
  rw [← List.count_false_add_count_true, hX, hY]

/-- The hypotheses of `length_eq_of_count` are satisfiable: `ab` holds one `a`
and one `b`. -/
example : ([false, true] : List Bool).count false = 1 ∧ ([false, true] : List Bool).count true = 1 :=
  ⟨rfl, rfl⟩

/-- **A formula of depth `0` without PNPs is constant on the middles of every
frame**: it reads the letter at its position and nothing else (Definition
`def:TLC_depth`; Appendix C.1, the depth-0 cases of the proof of
`lem:TLCP_commutative`). -/
theorem Form.constOnMiddle_of_depth_eq_zero (u v : List Bool) (X Y : ℕ) (ψ : Form Bool)
    (hf : ψ.pnpFree = true) (hd : ψ.depth = 0) : ConstOnMiddle ψ u v X Y :=
  ⟨⟨fun c => ψ.sat [c] 1, fun m _ _ i hi =>
      Form.sat_eq_of_pnpFree_depth_eq_zero ((getElem?_frame u m v hi).trans rfl) ψ hf hd⟩,
    fun m m' hX hY hX' hY' j hj hj' => Form.sat_eq_of_pnpFree_depth_eq_zero
      (getElem?_frame_outside u v (by rw [length_eq_of_count hX hY, length_eq_of_count hX' hY'])
        hj (by rw [length_eq_of_count hX hY]; omega)) ψ hf hd⟩

/-- The hypotheses of `Form.constOnMiddle_of_depth_eq_zero` are satisfiable:
`Q_a` has depth `0` and no PNPs. -/
example : (Form.sym false : Form Bool).pnpFree = true ∧ (Form.sym false : Form Bool).depth = 0 :=
  ⟨rfl, rfl⟩

/-- **Every Parikh vector has a middle**: `a^X b^Y` (Appendix D). -/
theorem exists_middle (X Y : ℕ) : ∃ m : List Bool, m.count false = X ∧ m.count true = Y :=
  ⟨List.replicate X false ++ List.replicate Y true, by simp [List.count_replicate],
    by simp [List.count_replicate]⟩

section

variable {σ : Type*}

mutual

/-- **The formulas under the counts of a formula of `TL[◁#,▷#]_{d+1}` lie in
`TL[◁#,▷#]_d`** (Definition `def:TLC_depth`: a count adds one to the depth of
what it counts). -/
theorem Form.mem_TLC_of_mem_countSubs {d : ℕ} {ψ : Form σ} :
    ∀ φ : Form σ, φ ∈ TLC σ (d + 1) → ψ ∈ φ.countSubs → ψ ∈ TLC σ d
  | .sym _, _, h => by rw [Form.countSubs] at h; exact absurd h List.not_mem_nil
  | .lt t₁ t₂, ⟨hf, hd⟩, h => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs, List.mem_append] at h
      exact h.elim (t₁.mem_TLC_of_mem_countSubs hf.1 (by omega))
        (t₂.mem_TLC_of_mem_countSubs hf.2 (by omega))
  | .neg φ, ⟨hf, hd⟩, h => by
      rw [Form.pnpFree] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs] at h
      exact Form.mem_TLC_of_mem_countSubs φ ⟨hf, hd⟩ h
  | .and φ₁ φ₂, ⟨hf, hd⟩, h => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.depth] at hd
      rw [Form.countSubs, List.mem_append] at h
      exact h.elim (Form.mem_TLC_of_mem_countSubs φ₁ ⟨hf.1, by omega⟩)
        (Form.mem_TLC_of_mem_countSubs φ₂ ⟨hf.2, by omega⟩)
  | .pnp _, _, h => by rw [Form.countSubs] at h; exact absurd h List.not_mem_nil

/-- The formulas under the counts of a term without PNPs of depth at most
`d + 1` lie in `TL[◁#,▷#]_d` (Definition `def:TLC_depth`). -/
theorem Term.mem_TLC_of_mem_countSubs {d : ℕ} {ψ : Form σ} :
    ∀ t : Term σ, t.pnpFree = true → t.depth ≤ d + 1 → ψ ∈ t.countSubs → ψ ∈ TLC σ d
  | .countL φ, hf, hd, h | .countR φ, hf, hd, h => by
      rw [Term.pnpFree] at hf
      rw [Term.depth] at hd
      rw [Term.countSubs, List.mem_cons] at h
      rcases h with rfl | h
      · exact ⟨hf, by omega⟩
      · exact Form.mem_TLC_of_mem_countSubs φ ⟨hf, by omega⟩ h
  | .add t₁ t₂, hf, hd, h => by
      rw [Term.pnpFree, Bool.and_eq_true] at hf
      rw [Term.depth] at hd
      rw [Term.countSubs, List.mem_append] at h
      exact h.elim (t₁.mem_TLC_of_mem_countSubs hf.1 (by omega))
        (t₂.mem_TLC_of_mem_countSubs hf.2 (by omega))
  | .one, _, _, h => by rw [Term.countSubs] at h; exact absurd h List.not_mem_nil

end

/-- The hypotheses of `Form.mem_TLC_of_mem_countSubs` and
`Term.mem_TLC_of_mem_countSubs` are satisfiable: `Q_a` lies under the count of
`▷#[Q_a] < 1`, a formula of `TL[◁#,▷#]_1`. -/
example (a : σ) :
    (Form.lt (.countR (.sym a)) .one : Form σ) ∈ TLC σ (0 + 1) ∧
      Form.sym a ∈ (Form.lt (.countR (.sym a)) .one : Form σ).countSubs := by
  refine ⟨⟨rfl, by simp [Form.depth, Term.depth]⟩, ?_⟩
  rw [Form.countSubs, Term.countSubs, Term.countSubs]
  exact List.mem_append_left _ (List.mem_cons_self ..)

end

end CRASP
end Transformer
