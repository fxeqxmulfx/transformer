/-
# Memory, scratch and BOS invariants of the compiled program

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
Scratch is cleared after every layer and the BOS bit stays fixed.
These invariants separate the comparison average from stored truth values.
-/

import Transformer.CRASP.TemporalProgramAttention

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- Named activation features at a natural-number index of `BOS · w` (B.2). -/
noncomputable def state (φ : Form σ) (k : ℕ) (w : List σ) (ℓ i : ℕ) : State φ :=
  decode φ ((model φ k).actAt w ℓ i)

/-- At a valid position, named state is the original activation (Appendix B.2). -/
theorem state_valid (φ : Form σ) (k : ℕ) (w : List σ) (ℓ : ℕ)
    (i : Fin (bos w).length) :
    state φ k w ℓ i.val = decode φ ((model φ k).act (bos w) ℓ i) := by
  rw [state, RTfr.actAt_valid]

omit [DecidableEq σ] in
/-- A node's Boolean slot is read directly (Appendix B.2). -/
theorem read_node (φ : Form σ) (H : State φ) (ψ : Node φ) :
    read φ H ψ.val = decide ((H (some (.inl ψ))).m = 1) := by
  rcases ψ with ⟨ψ, hψ⟩
  rw [read, dite_eq_left hψ]

/-- Residual input has exactly the old Boolean memory slots (Appendix B.2). -/
theorem read_input_node (φ : Form σ) (k ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin (dimension φ) → Fx (φ.capacity + 2) 0) (i : Fin n) (ψ : Node φ) :
    read φ (input φ k ℓ h i) ψ.val = read φ (decode φ (h i)) ψ.val := by
  rw [read_node, read_node, input_memory]

/-- Initial features are the compiled token embedding (Appendix B.2). -/
theorem state_zero (φ : Form σ) (k : ℕ) (w : List σ) (i : ℕ) (hi : i ≤ w.length) :
    state φ k w 0 i = embedding φ ((bos w)[i]'(by rw [length_bos]; omega)) := by
  have hin : i < (bos w).length := by rw [length_bos]; omega
  rw [state, RTfr.actAt, dite_eq_left hin]
  change decode φ (encode φ (embedding φ ((bos w)[i]))) = _
  rw [decode_encode]

/-- A layer writes the feed-forward features of the residual input (B.2). -/
theorem state_succ (φ : Form σ) (k : ℕ) (w : List σ) (ℓ i : ℕ) (hi : i ≤ w.length) :
    state φ k w (ℓ + 1) i = feedForward φ
      (input φ k ℓ ((model φ k).act (bos w) ℓ) ⟨i, by rw [length_bos]; omega⟩) := by
  have hin : i < (bos w).length := by rw [length_bos]; omega
  rw [state, RTfr.actAt, dite_eq_left hin, RTfr.act, RTfr.layer_eq_attention]
  change decode φ (encode φ (feedForward φ
    (input φ k ℓ ((model φ k).act (bos w) ℓ) ⟨i, hin⟩))) = _
  rw [decode_encode]

/-- All scratch slots are zero before attention at every layer (Appendix B.2). -/
theorem state_scratch (φ : Form σ) (k : ℕ) (w : List σ) (ℓ i : ℕ) (hi : i ≤ w.length)
    (ψ : Node φ) : state φ k w ℓ i (some (.inr ψ)) = 0 := by
  cases ℓ with
  | zero => rw [state_zero φ k w i hi]; rfl
  | succ ℓ => rw [state_succ φ k w ℓ i hi]; rfl

/-- The BOS flag is true exactly at index zero, independently of depth (B.2). -/
theorem state_bos (φ : Form σ) (k : ℕ) (w : List σ) (ℓ i : ℕ) (hi : i ≤ w.length) :
    state φ k w ℓ i none = Fx.ofBool φ.capacity (decide (i = 0)) := by
  have hin : i < (bos w).length := by rw [length_bos]; omega
  induction ℓ with
  | zero =>
      rw [state_zero φ k w i hi]
      cases i with
      | zero => simp [embedding, bos]
      | succ j => simp [embedding, bos]
  | succ ℓ ih =>
      rw [state_succ φ k w ℓ i hi]
      change Fx.ofBool φ.capacity (readBos φ (input φ k ℓ
        ((model φ k).act (bos w) ℓ) ⟨i, hin⟩)) = _
      rw [readBos, input_bos]
      rw [← state_valid φ k w ℓ ⟨i, hin⟩, ih, Fx.read_ofBool]

/-- Every valid index used by these invariants has a witness (Appendix B.2). -/
example : 0 ≤ [true].length ∧ 1 ≤ [true].length := ⟨Nat.zero_le _, le_rfl⟩

end Transformer.CRASP.TemporalProgram
