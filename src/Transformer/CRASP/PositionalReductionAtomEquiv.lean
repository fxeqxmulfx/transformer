/-
# `TL[◁#]^pos` reduces to `TL[◁#]`: the `Y`-atoms preserve meaning

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, the `Y^c Q_σ` and `Y^c MOD_m^r` cases of the proof of
`lem:tlclpos_reduction`.

The translation simulates `φ` at the positions `r·i + ρ` of
`f(w) = e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` with `i ∈ [|w|]` and `ρ ∈ [r]`, as the
paper does, and also at `i = 0` with `ρ = r`, the end of the first block.  At
such a position, `Y^c` looks back to `r·i + ρ − c`, which is still on `f(w)`
once `c + 2 ≤ r`, and which carries `wᵢ` when `ρ − c = 1` and `e` otherwise;
at `i = 0` it lands in the first block, which is where `c + 2 ≤ r` rather than
the paper's `c < r` is needed.
-/

import Transformer.CRASP.PositionalReductionAtom
import Transformer.CRASP.Spread
import Transformer.CRASP.YNormalFormEquiv

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- **The letter `Y^c` looks at.**  From a simulated position `r·i + ρ` of
`f(w)`, with `c + 2 ≤ r`, position `r·i + ρ − c` carries `wᵢ` when
`ρ − c = 1` and `e` otherwise (Appendix E, the `Y^c Q_σ` case of the proof of
`lem:tlclpos_reduction`). -/
theorem getElem?_spread_sub (w : List σ) {r i ρ c : ℕ} (hc : c + 2 ≤ r)
    (hadm : (1 ≤ i ∧ i ≤ w.length ∧ 1 ≤ ρ ∧ ρ ≤ r) ∨ (i = 0 ∧ ρ = r)) :
    (spread r w)[r * i + ρ - c - 1]? = some (if ρ = c + 1 then w[i - 1]? else none) := by
  have hr : 1 ≤ r := by omega
  rcases hadm with ⟨hi₁, hi₂, hρ₁, hρ₂⟩ | ⟨rfl, hρ⟩
  · obtain ⟨i, rfl⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
    rcases Nat.lt_or_ge c ρ with hcρ | hcρ
    · rw [show r * (i + 1) + ρ - c - 1 = r * (i + 1) + (ρ - c - 1) by omega,
        getElem?_spread hr w (i + 1) (ρ - c - 1) (by omega), ite_eq_right (by omega : i + 1 ≠ 0),
        Nat.add_sub_cancel, List.getElem?_eq_getElem (by omega)]
      by_cases hρ : ρ = c + 1
      · simp [hρ]
      · simp [hρ, show ρ - c - 1 ≠ 0 by omega]
    · rw [show r * (i + 1) + ρ - c - 1 = r * i + (r + ρ - c - 1) by rw [Nat.mul_add_one]; omega,
        getElem?_spread hr w i (r + ρ - c - 1) (by omega), ite_eq_right (show ρ ≠ c + 1 by omega)]
      rcases Nat.eq_zero_or_pos i with rfl | hi
      · rw [ite_eq_left rfl]
      · rw [ite_eq_right (by omega), List.getElem?_eq_getElem (by omega)]
        simp [show r + ρ - c - 1 ≠ 0 by omega]
  · subst ρ
    rw [show r * 0 + r - c - 1 = r * 0 + (r - c - 1) by omega,
      getElem?_spread hr w 0 (r - c - 1) (by omega), ite_eq_left rfl, ite_eq_right (by omega)]

/-- The hypotheses of `getElem?_spread_sub` are satisfiable: `c = 0`, and the
end `ρ = r = 2` of the first block. -/
example (w : List σ) : 0 + 2 ≤ 2 ∧ ((1 ≤ 0 ∧ 0 ≤ w.length ∧ 1 ≤ 2 ∧ 2 ≤ 2) ∨ (0 = 0 ∧ 2 = 2)) :=
  ⟨le_rfl, Or.inr ⟨rfl, rfl⟩⟩

variable [DecidableEq σ]

/-- **The invariant on `Y`-atoms** (Appendix E, the `Y^c Q_σ` and
`Y^c MOD_m^r` cases of the proof of `lem:tlclpos_reduction`): at a simulated
position `r·i + ρ`, `T_ρ⟦Y^c α⟧` holds at `i` of `w` exactly where `Y^c α`
holds on `f(w)`, once `c + reach ≤ r` and `r` is a multiple of the moduli.
`Y^c MOD_m^s` needs `r·i + ρ − c ≡ r + ρ − c` modulo `m`, which is where
`m ∣ r` is used; for `m = 0` both sides fall short of `s`. -/
theorem FormP.sat_atomTr (w : List σ) {r i ρ : ℕ}
    (hadm : (1 ≤ i ∧ i ≤ w.length ∧ 1 ≤ ρ ∧ ρ ≤ r) ∨ (i = 0 ∧ ρ = r))
    {α : FormP (Option σ)} (hα : YAtomic α) :
    ∀ c : ℕ, c + α.reach ≤ r → α.period ∣ r →
      (FormP.atomTr r ρ c α).sat w i = (FormP.prevN c α).sat (spread r w) (r * i + ρ) := by
  have hri : 1 ≤ i → r ≤ r * i := fun hi => Nat.le_mul_of_pos_right r hi
  have hpos : ∀ c, c + 2 ≤ r → c < r * i + ρ := fun c hc => by
    rcases hadm with ⟨hi, -, -, -⟩ | ⟨rfl, rfl⟩
    · have := hri hi
      omega
    · omega
  induction hα with
  | sym a =>
      intro c hc _
      rw [FormP.reach] at hc
      rw [FormP.sat_prevN, decide_eq_true (Or.inr (hpos c hc)), Bool.true_and, FormP.sat,
        getElem?_spread_sub w hc hadm]
      cases a with
      | none =>
          rw [FormP.atomTr, Form.sat_ofBool]
          by_cases hρ : ρ = c + 1
          · have hi : i - 1 < w.length := by
              rcases hadm with h | h
              · omega
              · omega
            simp [hρ, List.getElem?_eq_getElem hi]
          · simp [hρ]
      | some b =>
          rw [FormP.atomTr]
          by_cases hρ : ρ = c + 1
          · simp [hρ, Form.sat]
          · simp [hρ]
  | mod m s =>
      intro c hc hm
      rw [FormP.reach] at hc
      rw [FormP.period] at hm
      rw [FormP.atomTr, Form.sat_ofBool, FormP.sat_prevN,
        decide_eq_true (Or.inr (hpos c (by omega))), Bool.true_and, FormP.sat]
      refine decide_eq_decide.mpr ?_
      rcases Nat.eq_zero_or_pos m with rfl | hm₀
      · simp only [Nat.mod_zero]
        rcases hadm with ⟨hi, -, -, -⟩ | ⟨rfl, rfl⟩
        · have := hri hi
          omega
        · omega
      · rw [Nat.max_eq_left hm₀] at hm
        obtain ⟨q, rfl⟩ := hm
        rcases hadm with ⟨hi, -, -, -⟩ | ⟨rfl, rfl⟩
        · obtain ⟨i, rfl⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
          rw [show m * q * (i + 1) + ρ - c = m * (q * i) + (m * q + ρ - c) by
            rw [Nat.mul_add_one, Nat.mul_assoc]; omega, Nat.mul_add_mod]
        · rw [show m * q + m * q - c = m * q + (m * q * 0 + m * q - c) by omega, Nat.mul_add_mod]
  | prev _ ih =>
      intro c hc hper
      rw [FormP.reach] at hc
      rw [FormP.period] at hper
      rw [FormP.atomTr, ← FormP.prevN_succ']
      exact ih (c + 1) (by omega) hper

/-- The hypotheses of `FormP.sat_atomTr` are satisfiable: `Q_e` at the end of
the first block, for `r = 2`. -/
example (w : List σ) :
    ((1 ≤ 0 ∧ 0 ≤ w.length ∧ 1 ≤ 2 ∧ 2 ≤ 2) ∨ (0 = 0 ∧ 2 = 2)) ∧
      YAtomic (FormP.sym none : FormP (Option σ)) ∧
      0 + (FormP.sym none : FormP (Option σ)).reach ≤ 2 ∧
      (FormP.sym none : FormP (Option σ)).period ∣ 2 :=
  ⟨Or.inr ⟨rfl, rfl⟩, .sym none, le_rfl, one_dvd 2⟩

end CRASP
end Transformer
