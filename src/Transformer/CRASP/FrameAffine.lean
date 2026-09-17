/-
# Inside the middle of a frame, a count is affine in the letter counts

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D, the proof of `lem:cropping`: with the Parikh vector
fixed, `▷#` is rewritten in terms of `◁#` and `n⃗`, and each minimal depth-1
subformula defines a half-plane.

When the formulas under the counts of a term are constant on the middles of a
frame, the term at a position of the middle holding `c`, with `x` letters `a`
and `y` letters `b` before it in the middle, satisfies
`t + β'·x + γ'·y = K(c) + β·x + γ·y` (`Term.val_affine_frame`): a past count
adds the letters before the position, a future count loses them.  The
coefficients stay natural numbers by keeping the losses on the left.  A formula
over such terms is then determined by the letter and `(x, y)`
(`Form.sat_eq_of_take_count_eq`).
-/

import Transformer.CRASP.FrameCount

namespace Transformer
namespace CRASP

/-- **Inside the middle, a term whose counted formulas are constant on the
middles is affine in the letter counts before the position**, up to the letter
there: `t + β'·x + γ'·y = K(c) + β·x + γ·y` (Appendix D, proof of
`lem:cropping`, where `▷#` is rewritten with `◁#` and `n⃗`). -/
theorem Term.val_affine_frame (u v : List Bool) (X Y : ℕ) :
    ∀ t : Term Bool, (∀ ψ ∈ t.countSubs, ConstOnMiddle ψ u v X Y) →
      ∃ (K : Bool → ℕ) (β γ β' γ' : ℕ), ∀ m : List Bool, m.count false = X →
        m.count true = Y → ∀ i (hi : i < m.length),
          t.val (u ++ m ++ v) (u.length + i + 1) +
              (β' * (m.take i).count false + γ' * (m.take i).count true) =
            K m[i] + (β * (m.take i).count false + γ * (m.take i).count true)
  | .countL ψ, h => by
      rw [Term.countSubs] at h
      obtain ⟨⟨g, hg⟩, hout⟩ := h ψ (List.mem_cons_self ..)
      obtain ⟨m₀, h₀X, h₀Y⟩ := exists_middle X Y
      refine ⟨fun c => (Term.countL ψ).val (u ++ m₀ ++ v) u.length + (g c).toNat,
        (g false).toNat, (g true).toNat, 0, 0, fun m hX hY i hi => ?_⟩
      rw [show u.length + i + 1 = u.length + (i + 1) from Nat.add_assoc ..,
        Term.val_countL_middle (hg m hX hY) (i := i + 1) hi,
        Term.val_countL_outside ⟨⟨g, hg⟩, hout⟩ hX hY h₀X h₀Y (Or.inl le_rfl), List.take_add_one,
        List.getElem?_eq_getElem hi, Option.toList_some, List.countP_append, countP_bool_eq,
        List.countP_singleton]
      beta_reduce
      generalize g m[i] = b
      cases b
      · simp
      · simp
        omega
  | .countR ψ, h => by
      rw [Term.countSubs] at h
      obtain ⟨⟨g, hg⟩, hout⟩ := h ψ (List.mem_cons_self ..)
      obtain ⟨m₀, h₀X, h₀Y⟩ := exists_middle X Y
      refine ⟨fun _ => (Term.countL ψ).val (u ++ m₀ ++ v) (u ++ m₀ ++ v).length -
          (Term.countL ψ).val (u ++ m₀ ++ v) u.length, 0, 0, (g false).toNat, (g true).toNat,
        fun m hX hY i hi => ?_⟩
      have hl : (u ++ m ++ v).length = (u ++ m₀ ++ v).length := by
        simp only [List.length_append, length_eq_of_count hX hY, length_eq_of_count h₀X h₀Y]
      have e := Term.val_countR_add_val_countL (u ++ m ++ v) ψ (j := u.length + i)
        (by simp only [List.length_append]; omega)
      rw [Term.val_countL_middle (hg m hX hY) hi.le, countP_bool_eq,
        Term.val_countL_outside ⟨⟨g, hg⟩, hout⟩ hX hY h₀X h₀Y (Or.inl le_rfl),
        Term.val_countL_outside ⟨⟨g, hg⟩, hout⟩ hX hY h₀X h₀Y (j := (u ++ m ++ v).length)
          (Or.inr (by simp only [List.length_append, length_eq_of_count hX hY]; omega)), hl] at e
      simp only [Nat.zero_mul, Nat.add_zero]
      omega
  | .add t₁ t₂, h => by
      rw [Term.countSubs] at h
      obtain ⟨K₁, β₁, γ₁, β₁', γ₁', h₁⟩ := Term.val_affine_frame u v X Y t₁ fun ψ hψ =>
        h ψ (List.mem_append_left _ hψ)
      obtain ⟨K₂, β₂, γ₂, β₂', γ₂', h₂⟩ := Term.val_affine_frame u v X Y t₂ fun ψ hψ =>
        h ψ (List.mem_append_right _ hψ)
      refine ⟨fun c => K₁ c + K₂ c, β₁ + β₂, γ₁ + γ₂, β₁' + β₂', γ₁' + γ₂',
        fun m hX hY i hi => ?_⟩
      have e₁ := h₁ m hX hY i hi
      have e₂ := h₂ m hX hY i hi
      simp only [Term.val, Nat.add_mul]
      omega
  | .one, _ => ⟨fun _ => 1, 0, 0, 0, 0, fun _ _ _ _ _ => by simp [Term.val]⟩

/-- The hypothesis of `Term.val_affine_frame` is satisfiable: `Q_a`, under the
count of `◁#[Q_a]`, is constant on the middles of every frame. -/
example : ∀ ψ ∈ (Term.countL (.sym false) : Term Bool).countSubs, ConstOnMiddle ψ [] [] 1 1 := by
  intro ψ hψ
  rw [Term.countSubs, Form.countSubs, List.mem_singleton] at hψ
  exact hψ ▸ Form.constOnMiddle_of_depth_eq_zero [] [] 1 1 _ rfl rfl

/-- **Inside the middle, a formula whose counted formulas are constant on the
middles reads the letter and the letter counts before the position** (Appendix
D, proof of `lem:cropping`: such a formula is a Boolean combination of
half-planes and letters). -/
theorem Form.sat_eq_of_take_count_eq {u v : List Bool} {X Y : ℕ} {m m' : List Bool}
    (hX : m.count false = X) (hY : m.count true = Y) (hX' : m'.count false = X)
    (hY' : m'.count true = Y) {i i' : ℕ} (hi : i < m.length) (hi' : i' < m'.length)
    (hx : (m.take i).count false = (m'.take i').count false)
    (hy : (m.take i).count true = (m'.take i').count true) (hc : m[i] = m'[i']) :
    ∀ φ : Form Bool, φ.pnpFree = true → (∀ ψ ∈ φ.countSubs, ConstOnMiddle ψ u v X Y) →
      φ.sat (u ++ m ++ v) (u.length + i + 1) = φ.sat (u ++ m' ++ v) (u.length + i' + 1)
  | .sym _, _, _ => by
      rw [Form.sat, Form.sat, getElem?_frame u m v hi, getElem?_frame u m' v hi', hc]
  | .lt t₁ t₂, _, h => by
      rw [Form.countSubs] at h
      obtain ⟨K₁, β₁, γ₁, β₁', γ₁', h₁⟩ := Term.val_affine_frame u v X Y t₁ fun ψ hψ =>
        h ψ (List.mem_append_left _ hψ)
      obtain ⟨K₂, β₂, γ₂, β₂', γ₂', h₂⟩ := Term.val_affine_frame u v X Y t₂ fun ψ hψ =>
        h ψ (List.mem_append_right _ hψ)
      have e₁ := h₁ m hX hY i hi
      have e₁' := h₁ m' hX' hY' i' hi'
      have e₂ := h₂ m hX hY i hi
      have e₂' := h₂ m' hX' hY' i' hi'
      rw [hx, hy, hc] at e₁ e₂
      rw [Form.sat, Form.sat,
        show t₁.val (u ++ m ++ v) (u.length + i + 1) = t₁.val (u ++ m' ++ v) (u.length + i' + 1) by
          omega,
        show t₂.val (u ++ m ++ v) (u.length + i + 1) = t₂.val (u ++ m' ++ v) (u.length + i' + 1) by
          omega]
  | .neg φ, hf, h => by
      rw [Form.pnpFree] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat, Form.sat_eq_of_take_count_eq hX hY hX' hY' hi hi' hx hy hc φ hf h]
  | .and φ₁ φ₂, hf, h => by
      rw [Form.pnpFree, Bool.and_eq_true] at hf
      rw [Form.countSubs] at h
      rw [Form.sat, Form.sat,
        Form.sat_eq_of_take_count_eq hX hY hX' hY' hi hi' hx hy hc φ₁ hf.1 fun ψ hψ =>
          h ψ (List.mem_append_left _ hψ),
        Form.sat_eq_of_take_count_eq hX hY hX' hY' hi hi' hx hy hc φ₂ hf.2 fun ψ hψ =>
          h ψ (List.mem_append_right _ hψ)]
  | .pnp _, hf, _ => by rw [Form.pnpFree] at hf; exact absurd hf Bool.false_ne_true

/-- The hypotheses of `Form.sat_eq_of_take_count_eq` are satisfiable: the third
positions of `abba` and `baba`, each after one `a` and one `b` and holding `b`. -/
example : ([false, true, true, false] : List Bool).count false = 2 ∧
    ([false, true, true, false] : List Bool).count true = 2 ∧
    ([true, false, true, false] : List Bool).count false = 2 ∧
    ([true, false, true, false] : List Bool).count true = 2 ∧
    (([false, true, true, false] : List Bool).take 2).count false =
      (([true, false, true, false] : List Bool).take 2).count false ∧
    (([false, true, true, false] : List Bool).take 2).count true =
      (([true, false, true, false] : List Bool).take 2).count true ∧
    ([false, true, true, false] : List Bool)[2] = ([true, false, true, false] : List Bool)[2] := by
  decide

end CRASP
end Transformer
