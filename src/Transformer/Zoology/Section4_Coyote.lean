/-
# The Coyote gated-convolution operator

Arora et al., arXiv:2312.04927v1, §4, equation `eq: coyote-recursion`,
and Appendix `sec:simplified-hyena-append`, equation `eq:simplified_hyena`.
The finite sequence model uses causal columnwise convolution.  The paper
also permits cyclic convolution; that variant is represented separately.
-/

import Transformer.Zoology.Section3_MQAR

open scoped BigOperators

namespace Transformer.Zoology

/-- A real sequence with `n` token positions and `d` feature coordinates.
Source: §4, equation `eq: coyote-recursion`. -/
abbrev RealSequence (n d : ℕ) := Fin n → Fin d → ℝ

/-- A shared linear projection across token positions.  Source: §4,
equation `eq: coyote-recursion`. -/
def linearProjection {n d : ℕ} (u : RealSequence n d)
    (W : Fin d → Fin d → ℝ) : RealSequence n d :=
  fun i q => ∑ k, u i k * W k q

/-- Columnwise causal convolution.  For `k ≤ i`, subtraction in `Fin n`
agrees with ordinary subtraction, so the term reads `u[i-k,q]`.
Source: §4, equation `eq: coyote-recursion`. -/
def causalConvolution {n d : ℕ} (u h : RealSequence n d) :
    RealSequence n d :=
  fun i q => ∑ k : Fin n, if k ≤ i then h k q * u (i - k) q else 0

/-- Columnwise cyclic convolution, also allowed in Appendix
`sec:simplified-hyena-append`, below Algorithm `algo: simp-hyena`. -/
def cyclicConvolution {n d : ℕ} (u h : RealSequence n d) :
    RealSequence n d :=
  fun i q => ∑ k : Fin n, h k q * u (i - k) q

/-- The four parameter families in one Coyote layer.  Source: §4,
equation `eq: coyote-recursion`. -/
structure CoyoteParameters (n d : ℕ) where
  weight : Fin d → Fin d → ℝ
  filter : RealSequence n d
  bias₁ : RealSequence n d
  bias₂ : RealSequence n d

/-- A Coyote layer with causal convolution.  Source: §4,
equation `eq: coyote-recursion`. -/
def coyoteLayer {n d : ℕ} (p : CoyoteParameters n d)
    (u : RealSequence n d) : RealSequence n d :=
  fun i q => (linearProjection u p.weight i q + p.bias₁ i q) *
    (causalConvolution u p.filter i q + p.bias₂ i q)

/-- A Coyote layer with cyclic convolution, as permitted in Appendix
`sec:simplified-hyena-append`. -/
def coyoteLayerCyclic {n d : ℕ} (p : CoyoteParameters n d)
    (u : RealSequence n d) : RealSequence n d :=
  fun i q => (linearProjection u p.weight i q + p.bias₁ i q) *
    (cyclicConvolution u p.filter i q + p.bias₂ i q)

/-- The special parameters realizing a linear projection via the Coyote
operator.  Source: Appendix `lmm:primitives`, first construction. -/
def linearCoyoteParameters {n d : ℕ} (W : Fin d → Fin d → ℝ) :
    CoyoteParameters n d := {
  weight := W
  filter := fun _ _ => 0
  bias₁ := fun _ _ => 0
  bias₂ := fun _ _ => 1
}

/-- The Coyote layer implements any allowed rowwise linear projection.
Source: Appendix `lmm:primitives`, first construction. -/
theorem coyote_realizes_linear {n d : ℕ} (u : RealSequence n d)
    (W : Fin d → Fin d → ℝ) :
    coyoteLayer (linearCoyoteParameters W) u = linearProjection u W := by
  funext i q
  simp [coyoteLayer, linearCoyoteParameters, causalConvolution]

/-- The special parameters realizing convolution via Coyote.
Source: Appendix `lmm:primitives`, second construction. -/
def convolutionCoyoteParameters {n d : ℕ} (h : RealSequence n d) :
    CoyoteParameters n d := {
  weight := fun _ _ => 0
  filter := h
  bias₁ := fun _ _ => 1
  bias₂ := fun _ _ => 0
}

/-- The Coyote layer implements any columnwise causal convolution.
Source: Appendix `lmm:primitives`, second construction. -/
theorem coyote_realizes_convolution {n d : ℕ} (u h : RealSequence n d) :
    coyoteLayer (convolutionCoyoteParameters h) u =
      causalConvolution u h := by
  funext i q
  simp [coyoteLayer, convolutionCoyoteParameters, linearProjection]

/-- Causal convolution by the zero filter vanishes.
Source: §4, equation `eq: coyote-recursion`. -/
theorem causalConvolution_zero {n d : ℕ} (u : RealSequence n d) :
    causalConvolution u (fun _ _ => 0) = fun _ _ => 0 := by
  funext i q
  simp [causalConvolution]

end Transformer.Zoology
