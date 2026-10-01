/-
# Depth-preserving translation from `MAJ²`

arXiv:2506.16055v3, Appendix E, `thm:majtwo_to_tlc`. The induction
represents arbitrary binary formulas by signed rectangles. A majority
becomes a unary temporal comparison; a formula with one free variable is
then read on the diagonal.
-/

import Transformer.CRASP.MajTwoRectMajorityBounds
import Transformer.CRASP.MajTwoRectDiagonal
import Transformer.CRASP.MajTwoFree

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

/-- A rectangle representation agrees at every assignment of valid positions. -/
def MajRects.Represents (R : MajRects σ) (φ : Maj2 σ) : Prop :=
  ∀ (w : List σ) (ξ : Var → ℕ), (∀ v, 1 ≤ ξ v ∧ ξ v ≤ w.length) →
    R.value w (ξ .x) (ξ .y) = if φ.sat w ξ then (1 : ℤ) else 0

/-- The binary normalization required by `thm:majtwo_to_tlc` (Appendix E). -/
theorem Maj2.exists_rectangles (φ : Maj2 σ) :
    ∃ R : MajRects σ, R.Good φ.depth ∧ R.Represents φ := by
  induction φ with
  | sym a v =>
      refine ⟨MajRects.single (MajRect.unary v (.sym a)),
        MajRects.good_single (MajRect.good_unary rfl le_rfl v), ?_⟩
      intro w ξ hξ
      simp only [MajRects.value_single, MajRect.value_unary, Maj2.sat, Form.sat]
      rfl
  | lt v u =>
      refine ⟨MajRects.single (MajRect.order v u),
        MajRects.good_single (MajRect.good_order v u 0), ?_⟩
      intro w ξ hξ
      simp only [MajRects.value_single, MajRect.value_order, Maj2.sat,
        decide_eq_true_eq]
  | neg φ ih =>
      obtain ⟨R, hR, hc⟩ := ih
      refine ⟨R.neg, MajRects.good_neg hR, ?_⟩
      intro w ξ hξ
      rw [MajRects.value_neg, hc w ξ hξ, Maj2.sat]
      cases φ.sat w ξ <;> rfl
  | and φ ψ ihφ ihψ =>
      obtain ⟨R, hR, hcR⟩ := ihφ
      obtain ⟨S, hS, hcS⟩ := ihψ
      refine ⟨R.mul S, MajRects.good_mul
        (MajRects.good_mono hR (le_max_left _ _))
        (MajRects.good_mono hS (le_max_right _ _)), ?_⟩
      intro w ξ hξ
      rw [MajRects.value_mul, hcR w ξ hξ, hcS w ξ hξ, Maj2.sat]
      cases φ.sat w ξ <;> cases ψ.sat w ξ <;> rfl
  | maj v m φ ih =>
      choose R hR hc using ih
      let d := Finset.univ.sup fun t : Fin (m + 1) => (φ t).depth
      have hd : ∀ t, (R t).Good d := fun t => MajRects.good_mono (hR t)
        (Finset.le_sup (f := fun t : Fin (m + 1) => (φ t).depth) (Finset.mem_univ t))
      have hmem := MajRects.majority_mem v R hd
      refine ⟨MajRects.single (MajRect.unary v.other (MajRects.majority v R)),
        MajRects.good_single (MajRect.good_unary hmem.1 ?_ v.other), ?_⟩
      · simpa only [Maj2.depth, Nat.add_comm] using hmem.2
      · intro w ξ hξ
        have hsum : (∑ j ∈ Finset.Icc 1 w.length, ∑ t : Fin (m + 1),
            (R t).value w ((Function.update ξ v j) .x) ((Function.update ξ v j) .y)) =
            ((∑ j ∈ Finset.Icc 1 w.length, ∑ t : Fin (m + 1),
              if (φ t).sat w (Function.update ξ v j) then (1 : ℕ) else 0) : ℤ) := by
          push_cast
          apply Finset.sum_congr rfl
          intro j hj
          apply Finset.sum_congr rfl
          intro t ht
          have hu : ∀ u, 1 ≤ (Function.update ξ v j) u ∧
              (Function.update ξ v j) u ≤ w.length := by
            intro u
            by_cases huv : u = v
            · subst u; simpa using Finset.mem_Icc.mp hj
            · simpa only [Function.update_of_ne huv] using hξ u
          simpa only [Nat.cast_ite, Nat.cast_one, Nat.cast_zero] using hc t w _ hu
        have hsat : (MajRects.majority v R).sat w (ξ v.other) =
            (Maj2.maj v m φ).sat w ξ := by
          rw [MajRects.sat_majority v R w ξ (hξ v.other).1 (hξ v.other).2, hsum]
          simp only [Maj2.sat]
          apply decide_eq_decide.mpr
          norm_cast
        simp only [MajRects.value_single, MajRect.value_unary, hsat]

/-- Valid-position assignments used in the translation have a witness. -/
example : ∀ v : Var, 1 ≤ (fun _ => 1) v ∧ (fun _ => 1) v ≤ [true].length :=
  fun _ => ⟨le_rfl, le_rfl⟩

end Transformer.CRASP
