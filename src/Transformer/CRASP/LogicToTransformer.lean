/-
# The forward transformer simulation

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The finite parallel program compiles a temporal formula into a genuine
future-masked rounded transformer with the same depth bound.
-/

import Transformer.CRASP.TemporalProgramInduction

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]

/-- **Proposition `thm:TLCl_to_rtfr`.** Every `TL[◁#]` formula of depth at
most `k` is simulated by a future-masked rounded transformer of depth `k`.

The construction stores Boolean subformula values, uses zero query/key
projections for uniform attention, and places signed integer comparisons in
separate scratch features. A precision bound read from the finite syntax
makes every value-projection contribution exact; downward rounding tests its
sign. The residual and the empty word are handled explicitly.

Source: arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`. -/
theorem exists_rtfr_of_mem_TLCl (k : ℕ) (φ : Form σ) (hφ : φ ∈ TLCl σ k) :
    ∃ (p s d : ℕ) (T : RTfr (Option σ) p s d k), T.Recognizes φ.lang :=
  ⟨φ.capacity + 2, 0, TemporalProgram.dimension φ, TemporalProgram.model φ k,
    TemporalProgram.model_recognizes φ k hφ⟩

/-- The forward-simulation hypothesis has a witness (Appendix B.2). -/
example (a : σ) (k : ℕ) : (Form.sym a : Form σ) ∈ TLCl σ k :=
  ⟨rfl, rfl, Nat.zero_le k⟩

end Transformer.CRASP
