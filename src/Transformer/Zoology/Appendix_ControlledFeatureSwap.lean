/-
# Conditional feature swaps with no temporary storage

Arora et al., arXiv:2312.04927v1, Appendix `lmm:primitives` and
`prop: butterfly-hyena`. Hadamard mixing, a sign gate, and inverse mixing
exchange feature partners only on selected rows. This is a three-layer
K-constrained permutation of the original coordinates.
-/

import Transformer.Zoology.Appendix_FeaturePairMatrix

noncomputable section

namespace Transformer.Zoology

/-- Conditional partner exchange at selected rows.
Source: Appendix `prop: butterfly-hyena`, alternative in-place routing. -/
def controlledFeatureSwapNetwork {n k : ℕ} (t : Fin k) (g : Fin n → Bool) :
    CyclicKCoyoteNetwork n k 1 :=
  ((featurePairProjectionNetwork t {a := 1, b := 1, c := 1, d := -1}).append
    (featurePairDiagonalNetwork t (fun _ _ => 1)
      (fun i _ => if g i then -1 else 1))).append
    (featurePairProjectionNetwork t {a := 1/2, b := 1/2, c := 1/2, d := -1/2})

/-- Hadamard-sign-Hadamard implements the selected pair swap exactly.
Source: Appendix `lmm:primitives`, projection and diagonal gate composition. -/
theorem controlledFeatureSwapNetwork_pair_correct {n k : ℕ} (t : Fin k)
    (g : Fin n → Bool) (u : RealSequence n (butterflyWidth k)) :
    (controlledFeatureSwapNetwork t g).run u =
      featurePairRun t (fun i _ v => if g i then (v.2, v.1) else v) u := by
  simp only [controlledFeatureSwapNetwork, CyclicKCoyoteNetwork.run_append,
    featurePairProjectionNetwork_correct, featurePairDiagonalNetwork_correct,
    featurePairRun_comp]
  apply congrArg (fun F => featurePairRun t F u)
  funext i q v
  cases g i <;> apply Prod.ext <;> simp [PairMatrix.apply] <;> ring

/-- The selected rows read their feature partners; all other rows retain
the original values. Source: Appendix `prop: butterfly-hyena`, exact permutation. -/
theorem controlledFeatureSwapNetwork_correct {n k : ℕ} (t : Fin k)
    (g : Fin n → Bool) (u : RealSequence n (butterflyWidth k)) :
    (controlledFeatureSwapNetwork t g).run u =
      fun i q => if g i then u i (butterflyToggleIndex k t.val q) else u i q := by
  rw [controlledFeatureSwapNetwork_pair_correct]
  funext i q
  cases hg : g i <;> cases hq : butterflyToggleUpper k t.val q <;>
    simp [featurePairRun, featurePairRead, butterflyPairUpper, hg, hq,
      butterflyToggleIndex_involutive k t.val q]

/-- Conditional partner swaps require exactly three layers.
Source: Appendix `prop: butterfly-hyena`, constant-depth routing. -/
theorem controlledFeatureSwapNetwork_layerCount {n k : ℕ} (t : Fin k) (g : Fin n → Bool) :
    (controlledFeatureSwapNetwork t g).layerCount = 3 := by
  simp [controlledFeatureSwapNetwork, CyclicKCoyoteNetwork.layerCount_append,
    featurePairProjectionNetwork_layerCount, featurePairDiagonalNetwork_layerCount]

end Transformer.Zoology
