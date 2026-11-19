module

public import Mathlib.Data.Nat.ModEq
public import Transformer.CRASP.LeftQuotient
public import Transformer.CRASP.Spread

/-!
# `p⁻¹φ` reads `φ` after the prefix `p`

In no paper. The formula `p⁻¹φ` of `Transformer.CRASP.LeftQuotient` holds at the position
`i ≥ 1` of `w` iff `φ` holds at the position `|p| + i` of `p ++ w` (`FormP.sat_leftQuot`), so it
accepts a nonempty `w` iff `φ` accepts `p ++ w` (`FormP.models_leftQuot`). A symbol is read at
the same place of `w`; `MOD_m^r` at `|p| + i` is `MOD_m^{r + (m − 1)|p|}` at `i`
(`mod_leftQuot_iff`); `Y φ` at the first position reads the end of `p`
(`FormP.sat_prev_append_length`); a count up to `|p| + i` is the count over `p`
(`TermP.val_append`) plus the count over the first `i` positions of `w`.
-/

@[expose] public section

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

theorem TermP.val_addConst (w : List σ) (i : ℕ) (t : TermP σ) :
    ∀ n, (TermP.addConst n t).val w i = n + t.val w i
  | 0 => (Nat.zero_add _).symm
  | n + 1 => by rw [TermP.addConst, TermP.val, TermP.val_addConst w i t n, TermP.val]; omega

/-- At `i ≥ 1`, `i ≡ r + (m − 1) n (mod m)` iff `n + i ≡ r (mod m)`; for `m = 0`, `i = r − n`
iff `n + i = r`. -/
theorem mod_leftQuot_iff {m n i r : ℕ} (hi : 1 ≤ i) :
    i % m = (r + m * n - n) % m ↔ (n + i) % m = r % m := by
  rcases Nat.eq_zero_or_pos m with rfl | hm
  · simp only [Nat.mod_zero, Nat.zero_mul, Nat.add_zero]
    omega
  · have hn : n ≤ m * n := Nat.le_mul_of_pos_left n hm
    have e : (m - 1) * n = m * n - n := Nat.sub_one_mul m n
    rw [show r + m * n - n = r + (m - 1) * n by omega]
    have key : (n + i + (m - 1) * n) % m = i % m := by
      rw [show n + i + (m - 1) * n = i + m * n by omega, Nat.add_mul_mod_self_left]
    constructor
    · intro h
      exact Nat.ModEq.add_right_cancel' ((m - 1) * n) (key.trans h)
    · intro h
      exact key.symm.trans (Nat.ModEq.add_right ((m - 1) * n) h)

/-- The hypothesis of `mod_leftQuot_iff` is satisfiable: `MOD_2^0` at `2` after one symbol is
`MOD_2^1` at `1`. -/
example : 1 % 2 = (0 + 2 * 1 - 1) % 2 ↔ (1 + 1) % 2 = 0 % 2 := mod_leftQuot_iff le_rfl

/-- `Y φ` at the first position after `p` reads `φ` at the end of `p`. -/
theorem FormP.sat_prev_append_length (p w : List σ) (φ : FormP σ) :
    (FormP.prev φ).sat (p ++ w) (p.length + 1) = (FormP.prev φ).sat p (p.length + 1) := by
  rw [FormP.sat, FormP.sat, Nat.add_sub_cancel]
  rcases Nat.eq_zero_or_pos p.length with h | h
  · simp [h]
  · rw [φ.sat_append p w p.length h le_rfl]

mutual

/-- **`p⁻¹φ` at `i ≥ 1` is `φ` at `|p| + i` of `p ++ w`.** -/
theorem FormP.sat_leftQuot (p w : List σ) :
    ∀ (φ : FormP σ) (i : ℕ), 1 ≤ i → (φ.leftQuot p).sat w i = φ.sat (p ++ w) (p.length + i)
  | .sym a, i, hi => by
      rw [FormP.leftQuot, FormP.sat, FormP.sat, List.getElem?_append_right (by omega),
        show p.length + i - 1 - p.length = i - 1 by omega]
  | .mod m r, i, hi => by
      rw [FormP.leftQuot, FormP.sat, FormP.sat]
      exact decide_eq_decide.mpr (mod_leftQuot_iff hi)
  | .prev φ, i, hi => by
      rcases Nat.lt_or_ge 1 i with h | h
      · have e : (FormP.prev φ).sat (p ++ w) (p.length + i) =
            φ.sat (p ++ w) (p.length + (i - 1)) := by
          rw [FormP.sat, show p.length + i - 1 = p.length + (i - 1) by omega]
          simp [show 1 < p.length + i by omega]
        rw [e, FormP.leftQuot, ← φ.sat_leftQuot p w (i - 1) (by omega)]
        split <;> simp [FormP.sat, h]
      · obtain rfl : i = 1 := by omega
        rw [FormP.sat_prev_append_length, FormP.leftQuot]
        split <;> simp_all [FormP.sat]
  | .lt t₁ t₂, i, _ => by
      rw [FormP.leftQuot, FormP.sat, FormP.sat, t₁.val_leftQuot p w i, t₂.val_leftQuot p w i]
  | .neg φ, i, hi => by rw [FormP.leftQuot, FormP.sat, FormP.sat, φ.sat_leftQuot p w i hi]
  | .and φ₁ φ₂, i, hi => by
      rw [FormP.leftQuot, FormP.sat, FormP.sat, φ₁.sat_leftQuot p w i hi,
        φ₂.sat_leftQuot p w i hi]

/-- `p⁻¹t` at `i` is `t` at `|p| + i` of `p ++ w`. -/
theorem TermP.val_leftQuot (p w : List σ) :
    ∀ (t : TermP σ) (i : ℕ), (t.leftQuot p).val w i = t.val (p ++ w) (p.length + i)
  | .countL φ, i => by
      have e : List.range' 1 (p.length + i) =
          List.range' 1 p.length ++ (List.range' 1 i).map (p.length + ·) := by
        rw [List.map_add_range', ← List.range'_append, Nat.one_mul, Nat.add_comm 1]
      rw [TermP.leftQuot, TermP.val_addConst, ← (TermP.countL φ).val_append p w p.length le_rfl,
        TermP.val, TermP.val, TermP.val, e, List.filter_append, List.length_append,
        List.filter_map, List.length_map]
      congr 2
      refine List.filter_congr fun j hj => ?_
      rw [Function.comp_apply, φ.sat_leftQuot p w j (List.mem_range'_1.mp hj).1]
  | .add t₁ t₂, i => by
      rw [TermP.leftQuot, TermP.val, TermP.val, t₁.val_leftQuot p w i, t₂.val_leftQuot p w i]
  | .one, _ => rfl

end

/-- **`p⁻¹φ` accepts a nonempty `w` iff `φ` accepts `p ++ w`.** -/
theorem FormP.models_leftQuot (p : List σ) {w : List σ} (hw : w ≠ []) (φ : FormP σ) :
    (φ.leftQuot p).models w ↔ φ.models (p ++ w) := by
  rw [FormP.models, FormP.models, FormP.sat_leftQuot p w φ w.length
    (List.length_pos_iff.mpr hw), List.length_append]

/-- The hypotheses of `FormP.sat_leftQuot` and `FormP.models_leftQuot` are satisfiable: `MOD_2^0`
at the first position after one symbol, and `b` after `a`. -/
example : ((FormP.mod 2 0).leftQuot [false]).sat [true] 1 =
      (FormP.mod 2 0).sat ([false] ++ [true]) ([false].length + 1) ∧
    (((FormP.sym true).leftQuot [false]).models [true] ↔
      (FormP.sym true).models ([false] ++ [true])) :=
  ⟨FormP.sat_leftQuot _ _ _ 1 le_rfl, FormP.models_leftQuot [false] (List.cons_ne_nil _ _) _⟩

end CRASP
end Transformer
