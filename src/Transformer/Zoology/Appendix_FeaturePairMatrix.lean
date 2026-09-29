/-
# Every variable feature butterfly block uses exactly n × d

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`
and `lmm: kaleido-coyote`. Universal scalar factorization and six-layer
variable shears give a 25-layer implementation of arbitrary pair matrices,
including singular blocks. No input coordinate is used as scratch space.
-/

import Transformer.Zoology.Appendix_FeaturePairShears

noncomputable section

namespace Transformer.Zoology

/-- Compile the universal five-factor decomposition at every feature pair.
Source: Appendix `lmm: kaleido-coyote`, in-place alternative construction. -/
def featurePairMatrixNetwork {n k : ℕ} (t : Fin k)
    (M : Fin n → Fin (butterflyWidth k) → PairMatrix) : CyclicKCoyoteNetwork n k 1 :=
  ((((featurePairShearNetwork t false (fun i q => -(M i q).columnPrep)).append
    (featurePairShearNetwork t true (fun i q =>
      ((M i q).b + (M i q).rowPrep * (M i q).d) / (M i q).pivot))).append
    (featurePairDiagonalNetwork t (fun i q => (M i q).pivot)
      (fun i q => (M i q).d -
        ((M i q).c + (M i q).columnPrep * (M i q).d) / (M i q).pivot *
          ((M i q).b + (M i q).rowPrep * (M i q).d)))).append
    (featurePairShearNetwork t false (fun i q =>
      ((M i q).c + (M i q).columnPrep * (M i q).d) / (M i q).pivot))).append
    (featurePairShearNetwork t true (fun i q => -(M i q).rowPrep))

/-- The constructed network computes every independently parameterized
2 × 2 block exactly. Source: Appendix `prop: butterfly-hyena`, feature factors. -/
theorem featurePairMatrixNetwork_correct {n k : ℕ} (t : Fin k)
    (M : Fin n → Fin (butterflyWidth k) → PairMatrix)
    (u : RealSequence n (butterflyWidth k)) :
    (featurePairMatrixNetwork t M).run u =
      featurePairRun t (fun i q v => (M i q).apply v) u := by
  simp only [featurePairMatrixNetwork, CyclicKCoyoteNetwork.run_append,
    featurePairShearNetwork_correct, featurePairDiagonalNetwork_correct,
    featurePairRun_comp, pairShear]
  apply congrArg (fun F => featurePairRun t F u)
  funext i q v
  exact ((M i q).factorization v).symm

/-- Arbitrary feature-pair matrices use exactly 25 layers.
Source: Appendix `prop: butterfly-hyena`, constant-depth factor bound. -/
theorem featurePairMatrixNetwork_layerCount {n k : ℕ} (t : Fin k)
    (M : Fin n → Fin (butterflyWidth k) → PairMatrix) :
    (featurePairMatrixNetwork t M).layerCount = 25 := by
  simp [featurePairMatrixNetwork, CyclicKCoyoteNetwork.layerCount_append,
    featurePairShearNetwork_layerCount, featurePairDiagonalNetwork_layerCount]

/-- Extract the actual four entries of any feature-axis binary factor.
Source: Appendix `def: butterfly`, upper and lower diagonal entries. -/
def featureStagePairMatrix {n k : ℕ} (t : Fin k)
    (a b : RealSequence n (butterflyWidth k))
    (i : Fin n) (q : Fin (butterflyWidth k)) : PairMatrix := {
  a := a i q
  b := b i q
  c := b i (butterflyToggleIndex k t.val q)
  d := a i (butterflyToggleIndex k t.val q)
}

/-- Extracting the four entries recovers the factor's exact scalar action.
Source: Appendix `def: butterfly`, arbitrary row-dependent coefficients. -/
theorem featureStagePairMatrix_correct {n k : ℕ} (t : Fin k)
    (a b u : RealSequence n (butterflyWidth k)) :
    featurePairRun t (fun i q v => (featureStagePairMatrix t a b i q).apply v) u =
      fun i q => a i q * u i q + b i q * u i (butterflyToggleIndex k t.val q) := by
  funext i q
  cases h : butterflyToggleUpper k t.val q <;>
    simp [featurePairRun, featurePairRead, butterflyPairUpper, h,
      featureStagePairMatrix, PairMatrix.apply,
      butterflyToggleIndex_involutive k t.val q, add_comm]

end Transformer.Zoology
