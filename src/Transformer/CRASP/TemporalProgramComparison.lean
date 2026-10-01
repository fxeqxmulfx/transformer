/-
# Downward rounding implements the comparison

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The source numerator is an integer difference, its uniform average
has the same sign, and downward rounding preserves negativity.
-/

import Transformer.CRASP.TemporalProgramSources

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- One layer computes every comparison whose count bodies were already correct.
Source: arXiv:2506.16055v3, Appendix B.2, rounded comparison construction. -/
theorem comparison_correct (φ : Form σ) (k ℓ : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true)
    (hcorrect : Correct φ k ℓ) (t u : Term σ) (hm : Form.lt t u ∈ φ.subformulas)
    (hd : (Form.lt t u).depth ≤ ℓ + 1) (w : List σ) (i : ℕ) (hi : i ≤ w.length) :
    readComparison φ (input φ k ℓ ((model φ k).act (bos w) ℓ)
      ⟨i, by rw [length_bos]; omega⟩) (.lt t u) = (Form.lt t u).sat (view w i) i := by
  let idx : Fin (bos w).length := ⟨i, by rw [length_bos]; omega⟩
  let ψ : Node φ := ⟨.lt t u, hm⟩
  have hz : decode φ ((model φ k).act (bos w) ℓ idx) (some (.inr ψ)) = 0 := by
    rw [← state_valid]
    exact state_scratch φ k w ℓ i hi ψ
  have hden : (0 : ℝ) < (RTfr.masked idx).card := by
    exact_mod_cast Finset.card_pos.mpr ⟨idx, RTfr.self_mem_masked idx⟩
  rw [readComparison, dite_eq_left hm]
  rw [input_scratch φ k ℓ _ idx ψ hz, attention_uniform,
    sum_values φ k ℓ hp hf hcorrect t u hm hd w idx, Form.sat]
  apply decide_eq_decide.mpr
  rw [Fx.val_round_neg_iff, div_lt_iff₀ hden, zero_mul, sub_lt_zero]
  norm_cast

/-- The preceding-layer, comparison-bound and index hypotheses have witnesses (B.2). -/
example : Correct (Form.lt (.countL (.sym true)) .one) 1 0 ∧
    (Form.lt (.countL (.sym true)) .one).depth ≤ 1 ∧ 1 ≤ [true].length :=
  ⟨initial_correct _ _ rfl rfl, le_rfl, le_rfl⟩

end Transformer.CRASP.TemporalProgram
