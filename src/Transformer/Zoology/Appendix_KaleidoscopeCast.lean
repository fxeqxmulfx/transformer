/-
# Equality transport for recursive hierarchy dimensions

Arora et al., arXiv:2312.04927v1, Appendix `def: kaleidoscope`.
Expansion associates the binary dimensions as (k+p)+e, whereas a grid
associates them as k+(p+e). Equality transport changes neither coefficients,
hierarchy width, nor scalar matrix action.
-/

import Transformer.Zoology.Appendix_KaleidoscopeGrid

namespace Transformer.Zoology

/-- Reassociate a hierarchy's depth using an actual dimension equality.
Source: Appendix `def: kaleidoscope`, unchanged expanded scalar matrix. -/
def Kaleidoscope.cast {a b : ℕ} (h : a = b) (K : Kaleidoscope a) :
    Kaleidoscope b := h ▸ K

/-- Equality transport preserves the number of BB* factors.
Source: Appendix `def: kaleidoscope`, expansion does not change w. -/
theorem Kaleidoscope.cast_width {a b : ℕ} (h : a = b) (K : Kaleidoscope a) :
    (K.cast h).width = K.width := by
  subst b
  rfl

/-- Equality transport only reindexes inputs and outputs by the same
coordinate-preserving Fin cast. Source: Appendix `def: kaleidoscope`. -/
theorem Kaleidoscope.cast_apply {a b : ℕ} (h : a = b) (K : Kaleidoscope a)
    (v : Fin (butterflyWidth a) → ℝ) :
    (K.cast h).apply (fun j => v (Fin.cast (congrArg butterflyWidth h).symm j)) =
      fun j => K.apply v (Fin.cast (congrArg butterflyWidth h).symm j) := by
  subst b
  rfl

/-- Expansion and layout have exactly equal scalar dimensions.
Source: Appendix `def: kaleidoscope`, expanded dimension 2^e nd. -/
theorem butterflyGridExpandedWidth (p k e : ℕ) :
    butterflyWidth ((k + p) + e) = butterflyWidth (k + (p + e)) :=
  congrArg butterflyWidth (Nat.add_assoc k p e)

/-- Dimension equality is satisfiable for a genuinely expanded 2 × 2 grid.
Source: Appendix `lmm: kaleido-coyote`, expansion two. -/
example : ((1 : ℕ) + 1) + 1 = 1 + (1 + 1) := rfl

end Transformer.Zoology
