/-
# Counts around the middle of a frame

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D, the proof of `lem:cropping`: "if the Parikh vector of
a word is fixed we can rewrite `▷#` in terms of `◁#` and `n⃗`".

A future count at `j + 1` and a past count at `j` add up to the past count at
the last position (`Term.val_countR_add_val_countL`).  A past count inside the
middle of a frame is the count up to the middle plus the letters of the middle
on which the counted formula holds (`Term.val_countL_middle`), and outside the
middle it does not depend on the middle (`Term.val_countL_outside`).  So a
formula whose counted formulas are constant on the middles of a frame does not
depend on the middle outside it (`Form.sat_eq_of_constOnMiddle`).
-/

import Transformer.CRASP.Frame

namespace Transformer
namespace CRASP

/-- **A future count at `j + 1` and a past count at `j` add up to the past
count at the last position** (Appendix D: "we can rewrite `▷#` in terms of
`◁#` and `n⃗`"). -/
theorem Term.val_countR_add_val_countL {σ : Type*} [DecidableEq σ] (w : List σ) (ψ : Form σ)
    {j : ℕ} (hj : j ≤ w.length) :
    (Term.countR ψ).val w (j + 1) + (Term.countL ψ).val w j = (Term.countL ψ).val w w.length := by
  have e := List.countP_append (p := fun k => ψ.sat w k) (l₁ := List.range' 1 j)
    (l₂ := List.range' (1 + j) (w.length - j))
  rw [List.range'_append_1, Nat.add_sub_cancel' hj] at e
  rw [Term.val, Term.val, Term.val, ← List.countP_eq_length_filter,
    ← List.countP_eq_length_filter, ← List.countP_eq_length_filter, e, Nat.add_comm j 1,
    show w.length + 1 - (1 + j) = w.length - j by omega]
  omega

/-- The hypothesis of `Term.val_countR_add_val_countL` is satisfiable: `0 ≤ |a|`. -/
example : 0 ≤ [false].length := Nat.zero_le _

/-- A past count up to `a + b` is the count up to `a` and the count over the
next `b` positions (Definition `def:TLC_semantics`). -/
theorem Term.val_countL_add {σ : Type*} [DecidableEq σ] (w : List σ) (ψ : Form σ) (a b : ℕ) :
    (Term.countL ψ).val w (a + b) =
      (Term.countL ψ).val w a + (List.range' 1 b).countP fun j => ψ.sat w (a + j) := by
  rw [Term.val, Term.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
    countP_range'_add]

/-- **Inside the middle, a past count is the count up to the middle and the
letters on which the counted formula holds** (Appendix D, proof of
`lem:cropping`). -/
theorem Term.val_countL_middle {ψ : Form Bool} {u m v : List Bool} {g : Bool → Bool}
    (hg : ∀ i (hi : i < m.length), ψ.sat (u ++ m ++ v) (u.length + i + 1) = g m[i]) {i : ℕ}
    (hi : i ≤ m.length) :
    (Term.countL ψ).val (u ++ m ++ v) (u.length + i) =
      (Term.countL ψ).val (u ++ m ++ v) u.length + (m.take i).countP g := by
  rw [Term.val_countL_add]
  congr 1
  have hlen : (m.take i).length = i := List.length_take_of_le hi
  have h := countP_range'_eq_countP g (m.take i) (fun j => ψ.sat (u ++ m ++ v) (u.length + j))
    fun j a hj ha => by
      rw [List.getElem?_eq_some_iff] at ha
      obtain ⟨hj', rfl⟩ := ha
      rw [List.getElem_take]
      obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
      simpa [Nat.add_assoc] using hg j (by omega)
  rwa [hlen] at h

/-- The hypotheses of `Term.val_countL_middle` are satisfiable: `Q_a` reads the
letter at every position of `a` framed by `ε` and `ε`. -/
example : (∀ i (hi : i < [false].length),
    (Form.sym false : Form Bool).sat ([] ++ [false] ++ []) (([] : List Bool).length + i + 1) =
      (fun c => !c) [false][i]) ∧ 1 ≤ [false].length :=
  ⟨fun i hi => match i, hi with | 0, _ => rfl, le_rfl⟩

/-- **Outside the middle, a past count of a formula constant on the middles
does not depend on the middle** (Appendix D, proof of `lem:cropping`, with the
positions around the interval fixed). -/
theorem Term.val_countL_outside {ψ : Form Bool} {u v : List Bool} {X Y : ℕ}
    (hψ : ConstOnMiddle ψ u v X Y) {m m' : List Bool} (hX : m.count false = X)
    (hY : m.count true = Y) (hX' : m'.count false = X) (hY' : m'.count true = Y) {j : ℕ}
    (hj : j ≤ u.length ∨ u.length + X + Y ≤ j) :
    (Term.countL ψ).val (u ++ m ++ v) j = (Term.countL ψ).val (u ++ m' ++ v) j := by
  obtain ⟨⟨g, hg⟩, hout⟩ := hψ
  have hu : ∀ j ≤ u.length,
      (Term.countL ψ).val (u ++ m ++ v) j = (Term.countL ψ).val (u ++ m' ++ v) j := by
    intro j hj
    rw [Term.val, Term.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
    exact List.countP_congr fun k hk => by
      have := List.mem_range'_1.mp hk
      rw [hout m m' hX hY hX' hY' k this.1 (Or.inl (by omega))]
  rcases hj with hj | hj
  · exact hu j hj
  obtain ⟨k, rfl⟩ : ∃ k, j = u.length + (X + Y) + k := ⟨j - (u.length + X + Y), by omega⟩
  have e : ∀ m'' : List Bool, m''.count false = X → m''.count true = Y →
      (Term.countL ψ).val (u ++ m'' ++ v) (u.length + (X + Y)) =
        (Term.countL ψ).val (u ++ m'' ++ v) u.length + ((g false).toNat * X + (g true).toNat * Y) := by
    intro m'' h₁ h₂
    have hl := length_eq_of_count h₁ h₂
    rw [Term.val_countL_middle (hg m'' h₁ h₂) (i := X + Y) hl.ge, List.take_of_length_le hl.le,
      countP_bool_eq, h₁, h₂]
  rw [Term.val_countL_add (u ++ m ++ v) ψ (u.length + (X + Y)) k,
    Term.val_countL_add (u ++ m' ++ v) ψ (u.length + (X + Y)) k, e m hX hY, e m' hX' hY',
    hu _ le_rfl]
  congr 1
  exact List.countP_congr fun k' hk => by
    have := List.mem_range'_1.mp hk
    rw [hout m m' hX hY hX' hY' _ (by omega) (Or.inr (by omega))]

/-- The hypotheses of `Term.val_countL_outside` are satisfiable: `Q_a` on the
middles `a` of the frame `(ε, ε, 1, 0)`, at position `0`. -/
example : ConstOnMiddle (.sym false) [] [] 1 0 ∧ [false].count false = 1 ∧
    [false].count true = 0 ∧ (0 ≤ ([] : List Bool).length ∨ ([] : List Bool).length + 1 + 0 ≤ 0) :=
  ⟨Form.constOnMiddle_of_depth_eq_zero [] [] 1 0 _ rfl rfl, rfl, rfl, Or.inl le_rfl⟩

/-- **Outside the middle, a term whose counted formulas are constant on the
middles does not depend on the middle** (Appendix D, proof of `lem:cropping`;
a future count is the past count at the last position less the past count
before it). -/
theorem Term.val_eq_of_constOnMiddle {u v : List Bool} {X Y : ℕ} {m m' : List Bool}
    (hX : m.count false = X) (hY : m.count true = Y) (hX' : m'.count false = X)
    (hY' : m'.count true = Y) {j : ℕ} (hj₁ : 1 ≤ j) (hj : j ≤ u.length ∨ u.length + X + Y < j) :
    ∀ t : Term Bool, (∀ ψ ∈ t.countSubs, ConstOnMiddle ψ u v X Y) →
      t.val (u ++ m ++ v) j = t.val (u ++ m' ++ v) j
  | .countL ψ, h => by
      rw [Term.countSubs] at h
      exact Term.val_countL_outside (h ψ (List.mem_cons_self ..)) hX hY hX' hY' (hj.imp id le_of_lt)
  | .countR ψ, h => by
      rw [Term.countSubs] at h
      have hψ := h ψ (List.mem_cons_self ..)
      have hl : (u ++ m ++ v).length = (u ++ m' ++ v).length := by
        simp only [List.length_append, length_eq_of_count hX hY, length_eq_of_count hX' hY']
      have hn : u.length + X + Y ≤ (u ++ m ++ v).length := by
        simp only [List.length_append, length_eq_of_count hX hY]
        omega
      rcases Nat.lt_or_ge (u ++ m ++ v).length j with hjn | hjn
      · rw [Term.val, Term.val, ← hl, show (u ++ m ++ v).length + 1 - j = 0 by omega]
        rfl
      · obtain ⟨j', rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
        have e := Term.val_countR_add_val_countL (u ++ m ++ v) ψ (j := j') (by omega)
        have e' := Term.val_countR_add_val_countL (u ++ m' ++ v) ψ (j := j') (by omega)
        rw [← hl, ← Term.val_countL_outside hψ hX hY hX' hY' (Or.inr hn),
          ← Term.val_countL_outside hψ hX hY hX' hY' (j := j') (by omega)] at e'
        omega
  | .add t₁ t₂, h => by
      rw [Term.countSubs] at h
      rw [Term.val, Term.val,
        Term.val_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj t₁ fun ψ hψ =>
          h ψ (List.mem_append_left _ hψ),
        Term.val_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj t₂ fun ψ hψ =>
          h ψ (List.mem_append_right _ hψ)]
  | .one, _ => rfl

/-- The hypotheses of `Term.val_eq_of_constOnMiddle` are satisfiable: the
middles `a` of the frame `(b, ε, 1, 0)`, at position `1`. -/
example : [false].count false = 1 ∧ [false].count true = 0 ∧ 1 ≤ 1 ∧
    (1 ≤ [true].length ∨ [true].length + 1 + 0 < 1) :=
  ⟨rfl, rfl, le_rfl, Or.inl le_rfl⟩

/-- **Outside the middle, a formula whose counted formulas are constant on the
middles does not depend on the middle** (Appendix D, proof of
`lem:cropping`). -/
theorem Form.sat_eq_of_constOnMiddle {u v : List Bool} {X Y : ℕ} {m m' : List Bool}
    (hX : m.count false = X) (hY : m.count true = Y) (hX' : m'.count false = X)
    (hY' : m'.count true = Y) {j : ℕ} (hj₁ : 1 ≤ j) (hj : j ≤ u.length ∨ u.length + X + Y < j) :
    ∀ φ : Form Bool, φ.pnpFree = true → (∀ ψ ∈ φ.countSubs, ConstOnMiddle ψ u v X Y) →
      φ.sat (u ++ m ++ v) j = φ.sat (u ++ m' ++ v) j
  | .sym _, _, _ => by
      rw [Form.sat, Form.sat, getElem?_frame_outside u v
        ((length_eq_of_count hX hY).trans (length_eq_of_count hX' hY').symm) hj₁
        (by rw [length_eq_of_count hX hY]; omega)]
  | .lt t₁ t₂, _, h => by
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat,
        Term.val_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj t₁ fun ψ hψ =>
          h ψ (List.mem_append_left _ hψ),
        Term.val_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj t₂ fun ψ hψ =>
          h ψ (List.mem_append_right _ hψ)]
  | .neg φ, hf, h => by
      rw [Form.pnpFree] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat, Form.sat_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj φ hf h]
  | .and φ₁ φ₂, hf, h => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat,
        Form.sat_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj φ₁ hf.1 fun ψ hψ =>
          h ψ (List.mem_append_left _ hψ),
        Form.sat_eq_of_constOnMiddle hX hY hX' hY' hj₁ hj φ₂ hf.2 fun ψ hψ =>
          h ψ (List.mem_append_right _ hψ)]
  | .pnp _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The hypotheses of `Form.sat_eq_of_constOnMiddle` are satisfiable: `Q_b`,
with no counted formulas, on the frame `(b, ε, 1, 0)` at position `1`. -/
example : (Form.sym true : Form Bool).pnpFree = true ∧
    ∀ ψ ∈ (Form.sym true : Form Bool).countSubs, ConstOnMiddle ψ [true] [] 1 0 :=
  ⟨rfl, fun ψ hψ => by rw [Form.countSubs] at hψ; exact absurd hψ List.not_mem_nil⟩

end CRASP
end Transformer
