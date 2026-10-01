/-
# Uniform attention and scratch coordinates

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The compiled model averages exact integer values. Ordinary memory
coordinates receive zero attention; separate scratch slots receive the
averages used for comparison signs.
-/

import Transformer.CRASP.TemporalProgramBounds

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- Zero query/key projections give the uniform average required by B.2. -/
theorem attention_uniform (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n)
    (c : Fin (dimension φ)) :
    (model φ k).attention ℓ h i c = Fx.round (φ.capacity + 2) 0
      ((∑ j ∈ RTfr.masked i, ((model φ k).WV ℓ (h j) c).val) /
        ((RTfr.masked i).card : ℝ)) := by
  have hcard : ((RTfr.masked i).card : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.card_pos.mpr ⟨i, RTfr.self_mem_masked i⟩).ne'
  simp only [RTfr.attention, model, Fx.val_zero, zero_mul, Finset.sum_const_zero,
    Real.exp_zero, round_one, Fx.val_ofBool, ite_true, Finset.sum_const, nsmul_eq_mul,
    mul_one, hcard, ite_false, one_mul, Fx.round_val]

/-- Boolean-memory coordinates receive no attention contribution (B.2). -/
theorem attention_memory (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) (ψ : Node φ) :
    (model φ k).attention ℓ h i (coordinate φ (some (.inl ψ))) = 0 := by
  rw [attention_uniform]
  have hz : ∀ j : Fin n,
      (model φ k).WV ℓ (h j) (coordinate φ (some (.inl ψ))) = 0 := by
    intro j
    simp [model, encode, values]
  simp [hz]

/-- The BOS flag receives no attention contribution (Appendix B.2). -/
theorem attention_bos (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) :
    (model φ k).attention ℓ h i (coordinate φ none) = 0 := by
  rw [attention_uniform]
  have hz : ∀ j : Fin n, (model φ k).WV ℓ (h j) (coordinate φ none) = 0 := by
    intro j
    simp [model, encode, values]
  simp [hz]

/-- The named features supplied to the feed-forward map after residual addition.
Source: arXiv:2506.16055v3, Appendix B.1, `eq:att` and the residual connection. -/
noncomputable def input (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) : State φ :=
  decode φ fun c => Fx.add ((model φ k).attention ℓ h i c) (h i c)

/-- Memory survives the residual connection unchanged (Appendix B.2). -/
theorem input_memory (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) (ψ : Node φ) :
    input φ k ℓ h i (some (.inl ψ)) = decode φ (h i) (some (.inl ψ)) := by
  simp only [input, decode, attention_memory, Fx.zero_add]

/-- The BOS flag survives the residual connection unchanged (Appendix B.2). -/
theorem input_bos (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) :
    input φ k ℓ h i none = decode φ (h i) none := by
  simp only [input, decode, attention_bos, Fx.zero_add]

/-- A cleared scratch feature contains precisely the attention average (B.2). -/
theorem input_scratch (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) (ψ : Node φ)
    (hz : decode φ (h i) (some (.inr ψ)) = 0) :
    input φ k ℓ h i (some (.inr ψ)) =
      (model φ k).attention ℓ h i (coordinate φ (some (.inr ψ))) := by
  change Fx.add _ (decode φ (h i) (some (.inr ψ))) = _
  rw [hz, Fx.add_zero]

/-- The zero-scratch hypothesis has a witness (Appendix B.2). -/
example : decode (Form.sym true) (fun _ => 0) (some (.inr ⟨.sym true,
    Form.mem_subformulas _⟩)) = 0 := rfl

end Transformer.CRASP.TemporalProgram
