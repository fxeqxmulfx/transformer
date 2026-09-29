/-
# K-constrained linear operations without sequence workspace

Arora et al., arXiv:2312.04927v1, Appendix `lmm:primitives` and
`def: W-kmat`. Shared butterfly projections, arbitrary diagonal gates,
and channel-dependent cyclic shifts each use one layer of the original
n × d shape. Every projection is an explicit width-one K matrix.
-/

import Transformer.Zoology.Appendix_SingleLevelButterfly

namespace Transformer.Zoology

/-- One shared butterfly projection followed by a position-dependent gate.
Source: Appendix `lmm:primitives`, linear and elementwise operations. -/
def inPlaceProjectionParameters {n k : ℕ} (tree : ButterflyTree k)
    (a : RealSequence n (butterflyWidth k)) : KCoyoteParameters n k 1 := {
  weight := butterflyTreeK tree
  filter := fun _ _ => 0
  bias₁ := fun _ _ => 0
  bias₂ := a
}

/-- The one-layer projection uses no additional token or feature slots.
Source: Appendix `lmm:primitives`, exact primitive semantics. -/
theorem inPlaceProjection_correct {n k : ℕ} (tree : ButterflyTree k)
    (a u : RealSequence n (butterflyWidth k)) :
    coyoteLayerCyclic (inPlaceProjectionParameters tree a).toParameters u =
      fun i q => a i q * tree.apply (u i) q := by
  funext i q
  simp [coyoteLayerCyclic, inPlaceProjectionParameters, KCoyoteParameters.toParameters,
    linearProjection_kaleidoscope, butterflyTreeK_apply, cyclicConvolution, mul_comm]

/-- Arbitrary diagonal map at the original shape.
Source: Appendix `lmm:primitives`, fixed elementwise gate. -/
def inPlaceDiagonalParameters {n k : ℕ} (a : RealSequence n (butterflyWidth k)) :
    KCoyoteParameters n k 1 := inPlaceProjectionParameters (identityButterflyTree k) a

/-- The diagonal layer multiplies each input by its prescribed coefficient.
Source: Appendix `lmm:primitives`, elementwise gating. -/
theorem inPlaceDiagonal_correct {n k : ℕ}
    (a u : RealSequence n (butterflyWidth k)) :
    coyoteLayerCyclic (inPlaceDiagonalParameters a).toParameters u =
      fun i q => a i q * u i q := by
  rw [inPlaceDiagonalParameters, inPlaceProjection_correct]
  simp only [identityButterflyTree_apply]

/-- Independently shift each feature column by its fixed cyclic offset.
Source: Appendix `lmm:primitives`, allowed cyclic convolution. -/
def inPlaceShiftParameters {n k : ℕ} (s : Fin (butterflyWidth k) → Fin n) :
    KCoyoteParameters n k 1 := {
  weight := zeroExpandedKaleidoscope k
  filter := fun j q => if j = s q then 1 else 0
  bias₁ := fun _ _ => 1
  bias₂ := fun _ _ => 0
}

/-- A column shift reads exactly the prescribed original coordinate.
Source: Appendix `lmm:primitives`, impulse filter. -/
theorem inPlaceShift_correct {n k : ℕ} (s : Fin (butterflyWidth k) → Fin n)
    (u : RealSequence n (butterflyWidth k)) :
    coyoteLayerCyclic (inPlaceShiftParameters s).toParameters u =
      fun i q => u (i - s q) q := by
  funext i q
  simp [coyoteLayerCyclic, inPlaceShiftParameters, KCoyoteParameters.toParameters,
    linearProjection_kaleidoscope, zeroExpandedKaleidoscope_apply,
    cyclicConvolution, ite_mul]

end Transformer.Zoology
