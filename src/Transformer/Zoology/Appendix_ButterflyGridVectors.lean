/-
# Vector and matrix coordinates in the row-major grid

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`
and `def: kaleidoscope`. Encoding and decoding preserve every coordinate,
including the coordinate basis used to define ordinary matrix transpose.
-/

import Transformer.Zoology.Appendix_ButterflyGridToggle

namespace Transformer.Zoology

/-- Read an n × d input as its ordinary row-major scalar vector.
Source: Appendix `prop: butterfly-hyena`, the vector x associated with u. -/
def butterflyGridEncode {p k : ℕ}
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    Fin (butterflyWidth (k + p)) → ℝ :=
  fun j => u (butterflyGridUnflatten p k j).1 (butterflyGridUnflatten p k j).2

/-- Read a scalar vector in the original n × d input layout.
Source: Appendix `prop: butterfly-hyena`, row-major reconstruction. -/
def butterflyGridDecode {p k : ℕ} (v : Fin (butterflyWidth (k + p)) → ℝ) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  fun i q => v (butterflyGridFlatten p k i q)

/-- The grid layout loses no input coordinates.
Source: Appendix `prop: butterfly-hyena`, exact row-major representation. -/
theorem butterflyGridDecode_encode {p k : ℕ}
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    butterflyGridDecode (butterflyGridEncode u) = u := by
  funext i q
  simp only [butterflyGridDecode, butterflyGridEncode,
    butterflyGridUnflatten_flatten]

/-- The scalar representation loses no vector coordinates.
Source: Appendix `prop: butterfly-hyena`, bijective row-major representation. -/
theorem butterflyGridEncode_decode {p k : ℕ}
    (v : Fin (butterflyWidth (k + p)) → ℝ) :
    butterflyGridEncode (butterflyGridDecode (p := p) (k := k) v) = v := by
  funext j
  simp only [butterflyGridEncode, butterflyGridDecode,
    butterflyGridFlatten_unflatten]

/-- Equality to a grid coordinate is exactly equality to its scalar index.
Source: Appendix `def: kaleidoscope`, basis coordinates for matrix entries. -/
theorem butterflyGridUnflatten_eq_iff (p k : ℕ)
    (j : Fin (butterflyWidth (k + p))) (c : ButterflyGrid p k) :
    butterflyGridUnflatten p k j = c ↔
      j = butterflyGridFlatten p k c.1 c.2 := by
  constructor
  · intro h
    rw [← butterflyGridFlatten_unflatten p k j, h]
  · intro h
    rw [h, butterflyGridUnflatten_flatten]

/-- Encoding a grid basis vector gives the actual scalar coordinate basis.
Source: Appendix `def: kaleidoscope`, matrix entries and transpose. -/
theorem butterflyGridEncode_basis {p k : ℕ} (c : ButterflyGrid p k) :
    butterflyGridEncode (butterflyGridBasis c) =
      fun j => if j = butterflyGridFlatten p k c.1 c.2 then 1 else 0 := by
  funext j
  change (if butterflyGridUnflatten p k j = c then (1 : ℝ) else 0) = _
  simp only [butterflyGridUnflatten_eq_iff]

end Transformer.Zoology
