/-
# `TL[◁#]^pos` reduces to `TL[◁#]`: counts, block by block

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F, the `◁#[φ]` case of the proof of
`lem:tlclpos_reduction`.

The paper splits a count over `f(w) = e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` up to
`r·i + ρ` into the first block, the blocks of `w₁ ⋯ w_{i−1}`, and the current
block up to `ρ`.  Here the split is into the first block and the blocks of
`w₁ ⋯ wᵢ` in full, which is the count up to `r·i + r`, less the rest
`(r·i + ρ, r·i + r]` of the current block.  `TermP.val_countL_blockEnd` puts
the rest back, and `TermP.val_countL_spread` counts the full blocks: the first
as `e^r` reads it, each later one position of `w` at a time.  Both take as a
hypothesis the formulas `T ρ'` that simulate `φ` at `r·i + ρ'`, which the
induction of `Transformer.CRASP.PositionalReductionEquiv` supplies.
-/

import Transformer.CRASP.BoundedExists
import Transformer.CRASP.PositionalReductionAtom
import Transformer.CRASP.Spread

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A sum of indicators is a count. -/
theorem sum_map_ite_eq_countP {α : Type*} (p : α → Bool) :
    ∀ l : List α, (l.map fun x => if p x = true then 1 else 0).sum = l.countP p
  | [] => rfl
  | a :: l => by
      rw [List.map_cons, List.sum_cons, List.countP_cons, sum_map_ite_eq_countP p l, Nat.add_comm]

variable [DecidableEq σ]

/-- **The rest of the current block.**  The count of `φ` up to `r·i + ρ`, plus
the `ρ' ∈ (ρ, r]` at which `T ρ'` holds, is the count up to the end `r·i + r`
of the block, when `T ρ'` holds at `i` exactly where `φ` holds at `r·i + ρ'`
(Appendix F, proof of `lem:tlclpos_reduction`, the last part of the `◁#[φ]`
case). -/
theorem TermP.val_countL_blockEnd (w : List σ) (φ : FormP (Option σ)) (T : ℕ → Form σ)
    {r i ρ : ℕ} (hρ : ρ ≤ r)
    (hT : ∀ ρ', ρ < ρ' → ρ' ≤ r → (T ρ').sat w i = φ.sat (spread r w) (r * i + ρ')) :
    (TermP.countL φ).val (spread r w) (r * i + ρ) +
        ((List.range' (ρ + 1) (r - ρ)).map T).countP (·.sat w i) =
      (TermP.countL φ).val (spread r w) (r * i + r) := by
  rw [TermP.val, TermP.val, ← List.countP_eq_length_filter, ← List.countP_eq_length_filter,
    show r * i + r = r * i + ρ + (r - ρ) by omega, countP_range'_add _ (r * i + ρ) (r - ρ),
    ← List.map_add_range' (a := ρ) 1 (r - ρ) 1, List.countP_map, List.countP_map]
  congr 1
  refine List.countP_congr fun j hj => ?_
  rw [List.mem_range'_1] at hj
  simp only [Function.comp_apply]
  rw [hT (ρ + j) (by omega) (by omega), Nat.add_assoc]

/-- **The full blocks.**  The count of `φ` up to the end `r·i + r` of block
`i` is the count `C_φ` over the first block, which `φ` reads as `e^r`, plus
the counts up to `i` of the formulas `T ρ'`, when `T ρ'` holds at every
`i' ∈ [i]` exactly where `φ` holds at `r·i' + ρ'` (Appendix F, proof of
`lem:tlclpos_reduction`, the first two parts of the `◁#[φ]` case). -/
theorem TermP.val_countL_spread (w : List σ) (φ : FormP (Option σ)) (T : ℕ → Form σ)
    (r i : ℕ)
    (hT : ∀ i' ρ', 1 ≤ i' → i' ≤ i → 1 ≤ ρ' → ρ' ≤ r →
      (T ρ').sat w i' = φ.sat (spread r w) (r * i' + ρ')) :
    (TermP.countL φ).val (spread r w) (r * i + r) =
      ((List.range' 1 r).filter fun j => φ.sat (List.replicate r none) j).length +
        (((List.range' 1 r).map fun ρ' => Term.countL (T ρ')).map (·.val w i)).sum := by
  induction i with
  | zero =>
      have h0 : (((List.range' 1 r).map fun ρ' => Term.countL (T ρ')).map (·.val w 0)).sum = 0 := by
        simp [Function.comp_def, Term.val]
      rw [h0, Nat.add_zero, Nat.mul_zero, Nat.zero_add, TermP.val]
      exact congrArg List.length (List.filter_congr fun j hj => by
        rw [List.mem_range'_1] at hj
        rw [FormP.sat_spread_of_le r w φ (by omega) (by omega)])
  | succ i ih =>
      have ih := ih fun i' ρ' h₁ h₂ => hT i' ρ' h₁ (by omega)
      rw [TermP.val, ← List.countP_eq_length_filter] at ih ⊢
      rw [Nat.mul_add_one, countP_range'_add _ (r * i + r) r, ih, Nat.add_assoc]
      congr 1
      simp only [List.map_map, Function.comp_def, val_countL_succ, List.sum_map_add,
        sum_map_ite_eq_countP]
      congr 1
      refine List.countP_congr fun j hj => ?_
      rw [List.mem_range'_1] at hj
      rw [hT (i + 1) j (by omega) le_rfl (by omega) (by omega), Nat.mul_add_one]

/-- The hypotheses of `TermP.val_countL_blockEnd` and `TermP.val_countL_spread`
are satisfiable, and not only vacuously: `⊤` simulates `MOD_1^0` everywhere. -/
example (w : List σ) (r i ρ : ℕ) :
    ρ ≤ ρ ∧
      (∀ ρ', ρ < ρ' → ρ' ≤ r → (Form.ofBool true : Form σ).sat w i =
        (FormP.mod 1 0 : FormP (Option σ)).sat (spread r w) (r * i + ρ')) ∧
      ∀ i' ρ', 1 ≤ i' → i' ≤ i → 1 ≤ ρ' → ρ' ≤ r → (Form.ofBool true : Form σ).sat w i' =
        (FormP.mod 1 0 : FormP (Option σ)).sat (spread r w) (r * i' + ρ') := by
  simp [FormP.sat, Nat.mod_one]

end CRASP
end Transformer
