/-
# Actual hierarchy-width bounds of the in-place compiler

Arora et al., arXiv:2312.04927v1, Appendix `def: W-kmat` and
`lmm: kaleido-coyote`. Every stored projection has width one and expansion
two, including routing layers. The predicate inspects actual layer weights.
-/

import Transformer.Zoology.Appendix_InPlaceButterflyProgram

noncomputable section

namespace Transformer.Zoology

/-- Every actual stored projection has hierarchy width one.
Source: Appendix `def: W-kmat`, restriction on each Coyote layer. -/
def CyclicKCoyoteNetwork.HasWidthOne {n k : ℕ} (N : CyclicKCoyoteNetwork n k 1) : Prop :=
  ∀ p ∈ N.layers, p.weight.inner.width = 1

/-- Concatenation retains the actual width bound of both layer lists.
Source: Appendix `lem: stacking-layers`, preservation of allowed weights. -/
theorem CyclicKCoyoteNetwork.hasWidthOne_append {n k : ℕ}
    (A B : CyclicKCoyoteNetwork n k 1) :
    (A.append B).HasWidthOne ↔ A.HasWidthOne ∧ B.HasWidthOne := by
  simp [CyclicKCoyoteNetwork.HasWidthOne, CyclicKCoyoteNetwork.append,
    List.mem_append, or_imp, forall_and]

/-- Fixed pair projections use a single BB* representation.
Source: Appendix `def: W-kmat`, actual represented weight. -/
theorem featurePairProjectionNetwork_hasWidthOne {n k : ℕ} (t : Fin k) (M : PairMatrix) :
    (featurePairProjectionNetwork (n := n) t M).HasWidthOne := by
  simp [CyclicKCoyoteNetwork.HasWidthOne, featurePairProjectionNetwork,
    inPlaceProjectionParameters, butterflyTreeK, Kaleidoscope.width]

/-- Diagonal gates also store width-one projections.
Source: Appendix `def: W-kmat`, identity projection. -/
theorem featurePairDiagonalNetwork_hasWidthOne {n k : ℕ} (t : Fin k)
    (a b : RealSequence n (butterflyWidth k)) :
    (featurePairDiagonalNetwork t a b).HasWidthOne := by
  simp [CyclicKCoyoteNetwork.HasWidthOne, featurePairDiagonalNetwork,
    inPlaceDiagonalParameters, inPlaceProjectionParameters, butterflyTreeK, Kaleidoscope.width]

/-- Conjugated shears use only allowed width-one projections.
Source: Appendix `def: W-kmat`, primitive composition. -/
theorem featurePairConjugateShearNetwork_hasWidthOne {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) :
    (featurePairConjugateShearNetwork t upper a).HasWidthOne := by
  simp [featurePairConjugateShearNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    featurePairDiagonalNetwork_hasWidthOne, featurePairUnitShearNetwork,
    featurePairProjectionNetwork_hasWidthOne]

/-- Every variable shear retains width one throughout its six layers.
Source: Appendix `def: W-kmat`, primitive composition. -/
theorem featurePairShearNetwork_hasWidthOne {n k : ℕ} (t : Fin k) (upper : Bool)
    (a : RealSequence n (butterflyWidth k)) :
    (featurePairShearNetwork t upper a).HasWidthOne := by
  simp [featurePairShearNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    featurePairConjugateShearNetwork_hasWidthOne]

/-- Every arbitrary feature block is compiled with actual width-one weights.
Source: Appendix `def: W-kmat`, no dense-weight relaxation. -/
theorem featurePairMatrixNetwork_hasWidthOne {n k : ℕ} (t : Fin k)
    (M : Fin n → Fin (butterflyWidth k) → PairMatrix) :
    (featurePairMatrixNetwork t M).HasWidthOne := by
  simp [featurePairMatrixNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    featurePairShearNetwork_hasWidthOne, featurePairDiagonalNetwork_hasWidthOne]

/-- Conditional feature swaps use only allowed K projections.
Source: Appendix `def: W-kmat`, routing is inside the source model. -/
theorem controlledFeatureSwapNetwork_hasWidthOne {n k : ℕ} (t : Fin k) (g : Fin n → Bool) :
    (controlledFeatureSwapNetwork t g).HasWidthOne := by
  simp [controlledFeatureSwapNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    featurePairProjectionNetwork_hasWidthOne, featurePairDiagonalNetwork_hasWidthOne]

/-- Bit exchanges retain width-one weights, including both zero projections.
Source: Appendix `def: W-kmat`, permitted convolution primitives. -/
theorem rowFeatureExchangeNetwork_hasWidthOne {p k : ℕ} (r : Fin p) (f : Fin k) :
    (rowFeatureExchangeNetwork r f).HasWidthOne := by
  rw [rowFeatureExchangeNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    CyclicKCoyoteNetwork.hasWidthOne_append]
  refine ⟨⟨?_, controlledFeatureSwapNetwork_hasWidthOne _ _⟩, ?_⟩ <;>
    simp [CyclicKCoyoteNetwork.HasWidthOne, inPlaceShiftParameters,
      zeroExpandedKaleidoscope, Kaleidoscope.width]

/-- Row factors retain the same constant K restriction as feature factors.
Source: Appendix `def: W-kmat`, complete same-shape factor compiler. -/
theorem inPlaceRowFactorNetwork_hasWidthOne {p k : ℕ} (r : Fin p) (f : Fin k)
    (a b : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (inPlaceRowFactorNetwork r f a b).HasWidthOne := by
  simp [inPlaceRowFactorNetwork, CyclicKCoyoteNetwork.hasWidthOne_append,
    rowFeatureExchangeNetwork_hasWidthOne, featurePairMatrixNetwork_hasWidthOne]

/-- Each axis case meets the actual K-width restriction.
Source: Appendix `def: W-kmat`, all compiled projections. -/
theorem BinaryButterflyStage.compileInPlace_hasWidthOne {p k : ℕ}
    (s : BinaryButterflyStage p k) (f : Fin k) : (s.compileInPlace f).HasWidthOne := by
  rcases s with ⟨axis, a, b⟩
  cases axis <;> simp [BinaryButterflyStage.compileInPlace,
    inPlaceRowFactorNetwork_hasWidthOne, featurePairMatrixNetwork_hasWidthOne]

/-- All projections in a complete product have width one and expansion two.
Source: Appendix `def: W-kmat`, the source's polylogarithmic bounds. -/
theorem compileInPlaceButterflyStages_hasWidthOne {p k : ℕ} (f : Fin k)
    (stages : List (BinaryButterflyStage p k)) :
    (compileInPlaceButterflyStages f stages).HasWidthOne := by
  induction stages with
  | nil => simp [compileInPlaceButterflyStages, CyclicKCoyoteNetwork.HasWidthOne]
  | cons s ss ih =>
      simp [compileInPlaceButterflyStages, CyclicKCoyoteNetwork.hasWidthOne_append,
        s.compileInPlace_hasWidthOne, ih]

end Transformer.Zoology
