/-
# K-constrained versions of basic Coyote primitives

Arora et al., arXiv:2312.04927v1, Appendix `lmm:primitives` and
`def: W-kmat`. The convolution and fixed-gate constructions use zero
linear weights. We prove these weights have an explicit expanded K-matrix
representation and preserve the functional constructions.
-/

import Transformer.Zoology.Appendix_KCoyoteZero
import Transformer.Zoology.Appendix_Shift

namespace Transformer.Zoology

/-- K-constrained parameters for a columnwise causal convolution.
Source: Appendix `lmm:primitives`, convolution construction. -/
def kConvolutionParameters {n k : ℕ}
    (h : RealSequence n (butterflyWidth k)) : KCoyoteParameters n k 1 := {
  weight := zeroExpandedKaleidoscope k
  filter := h
  bias₁ := fun _ _ => 1
  bias₂ := fun _ _ => 0
}

/-- A K-constrained Coyote layer computes any allowed columnwise causal
convolution. Source: Appendix `lmm:primitives`, convolution case. -/
theorem kCoyote_realizes_convolution {n k : ℕ}
    (u h : RealSequence n (butterflyWidth k)) :
    coyoteLayer (kConvolutionParameters h).toParameters u =
      causalConvolution u h := by
  funext i q
  rw [kCoyoteLayer_eq]
  simp [kConvolutionParameters, zeroExpandedKaleidoscope_apply]

/-- K-constrained parameters for multiplication by a fixed
position-dependent factor. Source: Appendix `lmm:primitives`, gating case. -/
def kFixedGateParameters {n k : ℕ} [NeZero n]
    (factor : RealSequence n (butterflyWidth k)) :
    KCoyoteParameters n k 1 := {
  weight := zeroExpandedKaleidoscope k
  filter := impulse 0
  bias₁ := factor
  bias₂ := fun _ _ => 0
}

/-- A K-constrained Coyote layer gates an input by any fixed factor.
Source: Appendix `lmm:primitives`, gating case. -/
theorem kCoyote_realizes_fixed_gate {n k : ℕ} [NeZero n]
    (u factor : RealSequence n (butterflyWidth k)) :
    coyoteLayer (kFixedGateParameters factor).toParameters u =
      fun i q => factor i q * u i q := by
  funext i q
  rw [kCoyoteLayer_eq]
  simp [kFixedGateParameters, zeroExpandedKaleidoscope_apply,
    causalConvolution_impulse, shiftDown_zero]

/-- These zero-weight constructions use one `BB*` factor at one additional
butterfly level, so their stored coefficient count is explicit.
Source: Appendix Proposition `prop: single-baseconv`, parameter count
for the `lmm:primitives` constructions. -/
theorem kConvolution_parameterCount {n k : ℕ}
    (h : RealSequence n (butterflyWidth k)) :
    (kConvolutionParameters h).parameterCount =
      3 * n * butterflyWidth k +
        4 * (k + 1) * butterflyWidth (k + 1) := by
  rw [kCoyote_parameterCount_eq]
  simp [kConvolutionParameters, zeroExpandedKaleidoscope,
    Kaleidoscope.width]

end Transformer.Zoology
