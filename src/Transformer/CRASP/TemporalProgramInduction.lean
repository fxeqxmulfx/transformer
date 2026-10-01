/-
# The complete truth-value induction

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
Every layer adds one level of valid counting comparisons. The output
reads the formula's stored Boolean value at the final token, including BOS
for the empty word.
-/

import Transformer.CRASP.TemporalProgramComparison

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- The truth-value invariant advances by one counting level (Appendix B.2). -/
theorem correct_succ (φ : Form σ) (k ℓ : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true)
    (hcorrect : Correct φ k ℓ) : Correct φ k (ℓ + 1) := by
  intro w i hi ψ hd
  let idx : Fin (bos w).length := ⟨i, by rw [length_bos]; omega⟩
  let H := input φ k ℓ ((model φ k).act (bos w) ℓ) idx
  rw [state_succ φ k w ℓ i hi, read_node]
  change decide ((Fx.ofBool φ.capacity (evaluate φ H ψ.val)).m = 1) = _
  rw [Fx.read_ofBool]
  apply evaluate_eq_sat φ H (view w i) i (ℓ + 1)
  · intro a ha
    rw [read_input_node φ k ℓ _ idx ⟨.sym a, ha⟩, ← state_valid φ k w ℓ idx]
    exact hcorrect w i hi ⟨.sym a, ha⟩ (Nat.zero_le _)
  · intro t u hm hdu
    exact comparison_correct φ k ℓ hp hf hcorrect t u hm hdu w i hi
  · exact ψ.property
  · exact (φ.subformulas_properties hp hf ψ.val ψ.property).2.1
  · exact hd

/-- After `ℓ` layers all formulas of depth at most `ℓ` are correct (B.2). -/
theorem correct_all (φ : Form σ) (k ℓ : ℕ) (hp : φ.past = true) (hf : φ.pnpFree = true) :
    Correct φ k ℓ := by
  induction ℓ with
  | zero => exact initial_correct φ k hp hf
  | succ ℓ ih => exact correct_succ φ k ℓ hp hf ih

omit [DecidableEq σ] in
/-- At end satisfaction, the BOS logical view is the original word (B.2). -/
theorem view_end (w : List σ) : view w w.length = w := by
  cases w <;> simp [view]

/-- The compiled transformer recognizes the formula's language (Appendix B.2). -/
theorem model_recognizes (φ : Form σ) (k : ℕ) (hφ : φ ∈ TLCl σ k) :
    (model φ k).Recognizes φ.lang := by
  intro w
  rw [RTfr.Accepts, RTfr.out_bos]
  change 0 < (Fx.ofBool φ.capacity (read φ (state φ k w k w.length) φ)).val ↔
    φ.sat w w.length = true
  have h := correct_all φ k k hφ.1 hφ.2.1 w w.length le_rfl
    ⟨φ, φ.mem_subformulas⟩ hφ.2.2
  rw [view_end] at h
  rw [h, Fx.val_ofBool]
  cases φ.sat w w.length <;> simp

/-- The fragment, bound and inductive hypotheses have witnesses (Appendix B.2). -/
example : (Form.lt (.countL (.sym true)) .one) ∈ TLCl Bool 1 ∧
    Correct (Form.lt (.countL (.sym true)) .one) 1 0 :=
  ⟨⟨rfl, rfl, le_rfl⟩, initial_correct _ _ rfl rfl⟩

end Transformer.CRASP.TemporalProgram
