/-
# The reverse transformer simulation

arXiv:2506.16055v3, Appendix B.2, `thm:rtfr_to_TLCl`.
Finite activation-state formulas are constructed by induction on layers;
the output predicate is a disjunction of the accepting final states.
-/

import Transformer.CRASP.TransformerNextState

namespace Transformer.CRASP

universe u
variable {σ : Type u} [Fintype σ] [DecidableEq σ]

/-- All states of every layer have formulas of the same depth.
Source: arXiv:2506.16055v3, Appendix B.2, Equation `eq:phi_h`. -/
theorem RTfr.exists_stateFormulas {p s d k : ℕ} (T : RTfr (Option σ) p s d k) (ℓ : ℕ) :
    ∃ ψ : (Fin d → Fx p s) → Form σ,
      (∀ q, ψ q ∈ TLCl σ ℓ) ∧ T.StateDefinition ℓ ψ := by
  induction ℓ with
  | zero => exact ⟨T.initialFormula, T.initialFormula_mem, T.initialFormula_defines⟩
  | succ ℓ ih =>
      obtain ⟨ψ, hb, hs⟩ := ih
      exact ⟨T.nextFormula ℓ ψ, T.nextFormula_mem ℓ hb, T.nextFormula_defines ℓ hs⟩

/-- **Proposition `thm:rtfr_to_TLCl`.** Every future-masked rounded
transformer over a finite alphabet is simulated by a `TL[◁#]` formula of
the same depth.

The former Lean signature omitted finiteness of the alphabet. The manuscript
fixes `Σ = {σ₁, ..., σ_m}` in Section 2.3 (`def:Parikh_map`), and Appendix
B.2's base case is a finite disjunction over `Σ`. Without that assumption
the signature is false, as `RTfr.parityRecognizer_not_definable` proves.

The construction enumerates whole finite activation vectors instead of their
bits. Exact rounding cells express the division step as signed count
comparisons, including the zero-denominator fallback and saturation.

Source: arXiv:2506.16055v3, Appendix B.2, `thm:rtfr_to_TLCl`. -/
theorem exists_mem_TLCl_of_rtfr {p s d k : ℕ} (T : RTfr (Option σ) p s d k) :
    ∃ φ ∈ TLCl σ k, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  classical
  obtain ⟨ψ, hb, hs⟩ := T.exists_stateFormulas k
  let A := (Finset.univ : Finset (Fin d → Fx p s)).filter fun q => 0 < (T.Wout q).val
  refine ⟨Form.any (A.toList.map ψ), Form.any_mem _ ?_, ?_⟩
  · rintro φ hφ
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hφ
    exact hb q
  · ext w
    have hr : RTfr.Readable w w.length :=
      ⟨le_rfl, by by_cases hw : w = []; exact Or.inr hw; exact Or.inl (List.length_pos_iff.mpr hw)⟩
    change (Form.any (A.toList.map ψ)).sat w w.length = true ↔ T.Accepts (bos w)
    rw [Form.sat_any, RTfr.Accepts, RTfr.out_bos]
    constructor
    · rintro ⟨φ, hφ, hsat⟩
      obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hφ
      have heq : T.actAt w k w.length = q := of_decide_eq_true (by
        rw [← hs w w.length hr q]; exact hsat)
      rw [heq]
      exact (Finset.mem_filter.mp (Finset.mem_toList.mp hq)).2
    · intro hacc
      refine ⟨ψ (T.actAt w k w.length), List.mem_map_of_mem ?_, ?_⟩
      · exact Finset.mem_toList.mpr (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hacc⟩)
      · rw [hs w w.length hr]
        simp

end Transformer.CRASP
