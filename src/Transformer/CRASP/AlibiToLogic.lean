/-
# The ALiBi-to-logic simulation

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
The accepting-state disjunction closes the layer induction, with the BOS
empty input handled separately from ordinary positions.
-/

import Transformer.CRASP.AlibiStateInduction
import Transformer.CRASP.PeriodicToLogic

namespace Transformer.CRASP

universe u
variable {σ : Type u} [Fintype σ]

/-- The ordinary-position predicate also belongs to the `Y` fragment (F). -/
theorem FormP.onStr_mem_Y (k : ℕ) : (FormP.onStr : FormP σ) ∈ TLClY σ k :=
  ⟨Form.modFree_substPos FormP.sym (fun _ => rfl) _, (FormP.onStr_mem k).2⟩

variable [DecidableEq σ]

/-- **Proposition `thm:rtfr_to_TLCly`.** A finite-alphabet depth-`k`
ALiBi transformer has a depth-`k` formula in `TL[◁#, Y]`.

The paper uses a finite nonzero attention window for positive slopes.
With the downward rounding in Appendix B.1, a distant negative numerator
coefficient can instead have mantissa `−1`, even when its denominator
weight rounds to zero. The proof counts the eventually constant tail and
uses `Y` to correct its recent coefficients. This also handles zero and
negative slopes. The finite-alphabet assumption is from Section 2.3.

Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`, and B.1,
Equation `eq:att`. -/
theorem exists_mem_TLClY_of_alibi {p s d k : ℕ} (a : ℝ)
    (T : PTfr (Option σ) p s d k) (hT : T.pe = .alibi a) :
    ∃ φ ∈ TLClY σ k, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  classical
  obtain ⟨ψ, hb, hs⟩ := AlibiTables.exists_formulas T a hT k
  let A := (Finset.univ : Finset (Fin d → Fx p s)).filter (fun q => 0 < (T.Wout q).val)
  let φ := FormP.any (A.toList.map ψ)
  let e : FormP σ := FormP.truth (decide (T.Accepts (bos ([] : List σ))))
  refine ⟨FormP.or (.and FormP.onStr φ) (.and (.neg FormP.onStr) e),
    FormP.or_mem_Y (FormP.and_mem_Y (FormP.onStr_mem_Y _) (FormP.any_mem_Y _ ?_))
      (FormP.and_mem_Y (FormP.neg_mem_Y (FormP.onStr_mem_Y _))
        (FormP.truth_mem_Y _ _)), ?_⟩
  · rintro χ hχ
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hχ
    exact hb q
  · ext w
    change (FormP.or (.and FormP.onStr φ) (.and (.neg FormP.onStr) e)).sat w w.length =
      true ↔ T.Accepts (bos w)
    rw [FormP.sat_or, FormP.sat, FormP.sat, FormP.sat, FormP.sat_onStr_end]
    by_cases hw : w = []
    · subst w
      simp [e]
    · have hn : 0 < w.length := List.length_pos_iff.mpr hw
      simp only [hn, decide_true, Bool.true_and, Bool.not_true, Bool.false_and,
        Bool.or_false]
      rw [FormP.sat_any, PTfr.Accepts, PTfr.out_bos]
      constructor
      · rintro ⟨χ, hχ, hsat⟩
        obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hχ
        have heq : T.actAt w k w.length = q := of_decide_eq_true (by
          rw [← hs w w.length hn le_rfl q]; exact hsat)
        rw [heq]
        exact (Finset.mem_filter.mp (Finset.mem_toList.mp hq)).2
      · intro hacc
        refine ⟨ψ (T.actAt w k w.length), List.mem_map_of_mem ?_, ?_⟩
        · exact Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hacc⟩)
        · rw [hs w w.length hn le_rfl]
          simp

/-- The encoding hypothesis has a zero-transformer witness (Appendix F). -/
example : ∃ T : PTfr (Option Bool) 1 0 0 0, T.pe = .alibi 1 :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0,
     pe := .alibi 1 }, rfl⟩

end Transformer.CRASP
