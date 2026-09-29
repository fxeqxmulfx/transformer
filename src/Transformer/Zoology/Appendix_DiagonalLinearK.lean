/-
# K-constrained cancellation construction

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat`,
`prop: butterfly-hyena`, and `prop: single-baseconv`. The shared projection
is supplied as an actual expanded K matrix. The other two weights are
explicit zero K matrices; parameter storage is counted, rather than dense
weights being silently assumed to satisfy the restriction.
-/

import Transformer.Zoology.Appendix_DiagonalLinearSimulation

namespace Transformer.Zoology

/-- Store the shared projection and cancellation biases in one K layer.
Source: Appendix `def: W-kmat` and `lmm:primitives`. -/
def diagonalLinearKBranch {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1)
    (a b : RealSequence n (butterflyWidth k)) : KCoyoteParameters (9 * n) k 1 := {
  weight := K
  filter := impulse 0
  bias₁ := diagonalLinearBias₁ a b
  bias₂ := diagonalLinearBias₂ b
}

/-- The K-constrained version of the exact three-layer cancellation stack.
Source: Appendix `prop: butterfly-hyena` and `def: W-kmat`. -/
def diagonalLinearKNetwork {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1)
    (a b : RealSequence n (butterflyWidth k)) : CyclicKCoyoteNetwork (9 * n) k 1 := {
  layers := [zeroWeightCyclicKParameters shiftSumCopyParameters,
    diagonalLinearKBranch K a b,
    zeroWeightCyclicKParameters diagonalLinearCollectParameters]
}

/-- Decoding gives precisely the ordinary cancellation construction.
Source: Appendix `def: W-kmat`, semantics of the supplied shared weight. -/
theorem diagonalLinearKNetwork_toNetwork {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1) (a b : RealSequence n (butterflyWidth k)) :
    (diagonalLinearKNetwork K a b).toNetwork =
      diagonalLinearNetwork (kaleidoscopeWeight K) a b := by
  change ({layers :=
    [(zeroWeightCyclicKParameters shiftSumCopyParameters).toParameters,
      (diagonalLinearKBranch K a b).toParameters,
      (zeroWeightCyclicKParameters diagonalLinearCollectParameters).toParameters]} :
      CyclicCoyoteNetwork (9 * n) (butterflyWidth k)) = _
  rw [zeroWeightCyclicKParameters_correct shiftSumCopyParameters rfl,
    zeroWeightCyclicKParameters_correct diagonalLinearCollectParameters rfl]
  rfl

/-- The K network computes the exact diagonal-plus-projection map and
clears its workspace. Source: Appendix `prop: butterfly-hyena` and
`lem: stacking-layers`, cancellation construction. -/
theorem diagonalLinearKNetwork_correct {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1) (a b u : RealSequence n (butterflyWidth k)) :
    (diagonalLinearKNetwork K a b).run (shiftSumPad u) =
      shiftSumPad (diagonalLinearApply (kaleidoscopeWeight K) a b u) := by
  rw [← ((diagonalLinearKNetwork K a b).toNetwork_correct _).1,
    diagonalLinearKNetwork_toNetwork, diagonalLinearNetwork_correct]

/-- Exactly three layers, with no extra feature coordinates.
Source: Appendix `prop: butterfly-hyena`, constant-depth component. -/
theorem diagonalLinearKNetwork_layerCount {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1) (a b : RealSequence n (butterflyWidth k)) :
    (diagonalLinearKNetwork K a b).layerCount = 3 := rfl

/-- Count all stored scalars, including the supplied K matrix's factors.
Source: Appendix `prop: single-baseconv` and `def: W-kmat`. -/
theorem diagonalLinearKNetwork_parameterCount {n k : ℕ} [NeZero n]
    (K : ExpandedKaleidoscope k 1) (a b : RealSequence n (butterflyWidth k)) :
    (diagonalLinearKNetwork K a b).parameterCount =
      81 * n * butterflyWidth k +
        (K.inner.width + 2) * (4 * (k + 1) * butterflyWidth (k + 1)) := by
  simp only [CyclicKCoyoteNetwork.parameterCount, diagonalLinearKNetwork,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
    zeroWeightCyclicKParameters_parameterCount]
  rw [kCoyote_parameterCount_eq]
  simp only [diagonalLinearKBranch]
  ring

end Transformer.Zoology
