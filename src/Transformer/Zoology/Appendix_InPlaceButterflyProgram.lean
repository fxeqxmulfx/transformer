/-
# Original-shape K compilation of every butterfly product

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`
and `lmm: kaleido-coyote`. A feature factor takes 25 layers; a row factor
takes 35 layers via bit exchange. Composition uses exactly n rows and d
features. A chosen feature digit witnesses the source condition d ≥ 2.
-/

import Transformer.Zoology.Appendix_RowFeatureConjugacy

noncomputable section

namespace Transformer.Zoology

/-- Compile either axis on the original n × d coordinates.
Source: Appendix `prop: butterfly-hyena`, in-place constant-depth factor. -/
def BinaryButterflyStage.compileInPlace {p k : ℕ} (s : BinaryButterflyStage p k)
    (f : Fin k) : CyclicKCoyoteNetwork (butterflyWidth p) k 1 :=
  match s.axis with
  | .inl r => inPlaceRowFactorNetwork r f s.main s.off
  | .inr t => featurePairMatrixNetwork t (featureStagePairMatrix t s.main s.off)

/-- Every compiled factor has exact action, including singular coefficients.
Source: Appendix `prop: butterfly-hyena`, part (1). -/
theorem BinaryButterflyStage.compileInPlace_correct {p k : ℕ}
    (s : BinaryButterflyStage p k) (f : Fin k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (s.compileInPlace f).run u = s.apply u := by
  rcases s with ⟨axis, a, b⟩
  cases axis with
  | inl r => exact inPlaceRowFactorNetwork_correct r f a b u
  | inr t =>
      rw [BinaryButterflyStage.compileInPlace, featurePairMatrixNetwork_correct,
        featureStagePairMatrix_correct]
      rfl

/-- Uniform constant-depth bound at the exact input shape.
Source: Appendix `prop: butterfly-hyena`, factor depth O(1). -/
theorem BinaryButterflyStage.compileInPlace_layerCount {p k : ℕ}
    (s : BinaryButterflyStage p k) (f : Fin k) :
    (s.compileInPlace f).layerCount ≤ 35 := by
  rcases s with ⟨axis, a, b⟩
  cases axis with
  | inl r => rw [BinaryButterflyStage.compileInPlace, inPlaceRowFactorNetwork_layerCount]
  | inr t => rw [BinaryButterflyStage.compileInPlace, featurePairMatrixNetwork_layerCount]; omega

/-- Stack factor networks on the same original coordinates.
Source: Appendix `lem: stacking-layers`, no intermediate padding. -/
def compileInPlaceButterflyStages {p k : ℕ} (f : Fin k) :
    List (BinaryButterflyStage p k) → CyclicKCoyoteNetwork (butterflyWidth p) k 1
  | [] => {layers := []}
  | s :: ss => (s.compileInPlace f).append (compileInPlaceButterflyStages f ss)

/-- The original-shape network realizes the entire factor product.
Source: Appendix `prop: butterfly-hyena`, part (2), and `lmm: kaleido-coyote`. -/
theorem compileInPlaceButterflyStages_correct {p k : ℕ} (f : Fin k)
    (stages : List (BinaryButterflyStage p k))
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    (compileInPlaceButterflyStages f stages).run u = applyBinaryButterflyStages stages u := by
  induction stages generalizing u with
  | nil => rfl
  | cons s ss ih =>
      rw [compileInPlaceButterflyStages, CyclicKCoyoteNetwork.run_append,
        s.compileInPlace_correct, ih]
      rfl

/-- Constant depth per factor yields the logarithmic product bound.
Source: Appendix `prop: butterfly-hyena` and `lmm: kaleido-coyote`. -/
theorem compileInPlaceButterflyStages_layerCount {p k : ℕ} (f : Fin k)
    (stages : List (BinaryButterflyStage p k)) :
    (compileInPlaceButterflyStages f stages).layerCount ≤ 35 * stages.length := by
  induction stages with
  | nil => rfl
  | cons s ss ih =>
      rw [compileInPlaceButterflyStages, CyclicKCoyoteNetwork.layerCount_append,
        List.length_cons]
      have hs := s.compileInPlace_layerCount f
      omega

end Transformer.Zoology
