/-
# Coyote layers with kaleidoscope-constrained projection weights

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat`,
`def: kaleidoscope`, and Proposition `prop: single-baseconv`. The article
restricts theoretical linear projections to K-matrices. This module joins
the matrix representation to the Coyote layer and computes its exact stored
coefficient count at power-of-two feature width.
-/

import Transformer.Zoology.Appendix_KaleidoscopeMatrix
import Transformer.Zoology.Section4_Coyote

namespace Transformer.Zoology

/-- The Coyote row-vector orientation of an expanded K-matrix: input index
first, output index second. Source: Appendix `def: W-kmat`. -/
def kaleidoscopeWeight {k e : ℕ} (K : ExpandedKaleidoscope k e) :
    Fin (butterflyWidth k) → Fin (butterflyWidth k) → ℝ :=
  fun input output => K.matrix output input

/-- The rowwise Coyote projection using a K weight is exactly application
of that K operator to every row. Source: Appendix `def: W-kmat`. -/
theorem linearProjection_kaleidoscope {n k e : ℕ}
    (u : RealSequence n (butterflyWidth k))
    (K : ExpandedKaleidoscope k e) (i : Fin n)
    (q : Fin (butterflyWidth k)) :
    linearProjection u (kaleidoscopeWeight K) i q = K.apply (u i) q := by
  rw [K.apply_eq_matrix]
  simp [linearProjection, kaleidoscopeWeight, mul_comm]

/-- A Coyote layer whose linear weight is represented by an expanded
K-matrix, rather than an arbitrary dense matrix. The remaining three
arrays are the convolution filter and two biases. Source: Appendix
`def: W-kmat` and equation `eq:simplified_hyena`. -/
structure KCoyoteParameters (n k e : ℕ) where
  weight : ExpandedKaleidoscope k e
  filter : RealSequence n (butterflyWidth k)
  bias₁ : RealSequence n (butterflyWidth k)
  bias₂ : RealSequence n (butterflyWidth k)

/-- Forget the compressed representation and obtain ordinary Coyote
parameters with the same mathematical action. Source: Appendix
`def: W-kmat`. -/
def KCoyoteParameters.toParameters {n k e : ℕ}
    (p : KCoyoteParameters n k e) :
    CoyoteParameters n (butterflyWidth k) := {
  weight := kaleidoscopeWeight p.weight
  filter := p.filter
  bias₁ := p.bias₁
  bias₂ := p.bias₂
}

/-- A K-constrained Coyote layer uses the actual K operator rowwise.
Source: Appendix `def: W-kmat` and equation `eq:simplified_hyena`. -/
theorem kCoyoteLayer_eq {n k e : ℕ} (p : KCoyoteParameters n k e)
    (u : RealSequence n (butterflyWidth k)) :
    coyoteLayer p.toParameters u =
      fun i q => (p.weight.apply (u i) q + p.bias₁ i q) *
        (causalConvolution u p.filter i q + p.bias₂ i q) := by
  funext i q
  simp only [coyoteLayer, KCoyoteParameters.toParameters]
  rw [linearProjection_kaleidoscope]

/-- Count one stored real coefficient for each filter and bias entry and
for each butterfly factor coefficient. Source: Appendix Proposition
`prop: single-baseconv`, parameter accounting. -/
def KCoyoteParameters.parameterCount {n k e : ℕ}
    (p : KCoyoteParameters n k e) : ℕ :=
  3 * n * butterflyWidth k + p.weight.inner.parameterCount

/-- The exact count for the chosen power-of-two K representation is
`3n·2^k + 4w(k+e)·2^(k+e)`, where `w` is the number of `BB*` factors.
Source: Appendix Proposition `prop: single-baseconv`, before its
asymptotic simplification under polylogarithmic `w` and expansion. -/
theorem kCoyote_parameterCount_eq {n k e : ℕ}
    (p : KCoyoteParameters n k e) :
    p.parameterCount = 3 * n * butterflyWidth k +
      p.weight.inner.width *
        (4 * (k + e) * butterflyWidth (k + e)) := by
  unfold KCoyoteParameters.parameterCount
  rw [p.weight.inner.parameterCount_eq]

end Transformer.Zoology
