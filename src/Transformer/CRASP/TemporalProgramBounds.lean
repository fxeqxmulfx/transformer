/-
# Exact integer source values at the compiled precision

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
All value-projection contributions fit a precision chosen once from
the finite syntax, so their inner rounding introduces no error.
-/

import Transformer.CRASP.TemporalProgram

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u}

/-- Every signed source fits the bound read from the finite program (B.2). -/
theorem abs_source_le (φ : Form σ) (H : State φ) (ψ : Node φ) :
    |source φ H ψ| ≤ (φ.capacity : ℤ) := by
  rcases ψ with ⟨ψ, hψ⟩
  have hcap := Form.weightSize_le_capacity hψ
  cases ψ with
  | lt t u =>
      simp only [Form.weightSize, Term.weightSize] at hcap
      change |if readBos φ H then (t.constant : ℤ) - u.constant
        else (t.countBodies.countP (read φ H) : ℤ) - u.countBodies.countP (read φ H)| ≤ _
      split_ifs
      · rw [abs_le]
        constructor <;> omega
      · have ht : t.countBodies.countP (read φ H) ≤ t.countBodies.length := List.countP_le_length
        have hu : u.countBodies.countP (read φ H) ≤ u.countBodies.length := List.countP_le_length
        rw [abs_le]
        constructor <;> omega
  | sym a | neg a | and a b | pnp a => simp [source]

/-- Value-projection mantissas are the exact intended source integers (B.2). -/
theorem m_values_scratch (φ : Form σ) (H : State φ) (ψ : Node φ) :
    (values φ H (some (.inr ψ))).m = source φ H ψ :=
  Fx.m_round_int φ.capacity _ (abs_source_le φ H ψ)

/-- The chosen integer precision has a representable unit (Appendix B.2). -/
theorem round_one (φ : Form σ) :
    Fx.round (φ.capacity + 2) 0 1 = Fx.ofBool φ.capacity true := by
  simpa only [Fx.val_ofBool, ite_true] using
    Fx.round_val (φ.capacity + 2) 0 (Fx.ofBool φ.capacity true)

end Transformer.CRASP.TemporalProgram
