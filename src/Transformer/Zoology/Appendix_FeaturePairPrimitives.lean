/-
# Fixed pair projections and variable pair diagonals in place

Arora et al., arXiv:2312.04927v1, Appendix `lmm:primitives`,
`def: butterfly`, and `def: W-kmat`. These one-layer K networks retain
exactly the input n × d shape. Pair coefficients can vary with every row.
-/

import Transformer.Zoology.Appendix_FeaturePairAction

namespace Transformer.Zoology

/-- One-layer fixed pair matrix, with an actual butterfly K weight.
Source: Appendix `lmm:primitives`, shared linear projection. -/
def featurePairProjectionNetwork {n k : ℕ} (t : Fin k) (M : PairMatrix) :
    CyclicKCoyoteNetwork n k 1 := {
  layers := [inPlaceProjectionParameters (featurePairTree t M) (fun _ _ => 1)]
}

/-- The fixed shared matrix acts on every pair at the original shape.
Source: Appendix `lmm:primitives`, exact matrix semantics. -/
theorem featurePairProjectionNetwork_correct {n k : ℕ} (t : Fin k) (M : PairMatrix)
    (u : RealSequence n (butterflyWidth k)) :
    (featurePairProjectionNetwork t M).run u =
      featurePairRun t (fun _ _ v => M.apply v) u := by
  change coyoteLayerCyclic
    (inPlaceProjectionParameters (featurePairTree t M) (fun _ _ => 1)).toParameters u = _
  rw [inPlaceProjection_correct]
  funext i q
  simp only [one_mul, featurePairTree_apply, featurePairRun, featurePairRead]

/-- One-layer independently parameterized diagonal at every pair.
Source: Appendix `lmm:primitives`, arbitrary fixed elementwise coefficients. -/
def featurePairDiagonalNetwork {n k : ℕ} (t : Fin k)
    (a b : RealSequence n (butterflyWidth k)) : CyclicKCoyoteNetwork n k 1 := {
  layers := [inPlaceDiagonalParameters (fun i q =>
    if butterflyToggleUpper k t.val q then a i (butterflyPairUpper k t q)
    else b i (butterflyPairUpper k t q))]
}

/-- The pair diagonal gate retains both original coordinates.
Source: Appendix `lmm:primitives`, diagonal action. -/
theorem featurePairDiagonalNetwork_correct {n k : ℕ} (t : Fin k)
    (a b u : RealSequence n (butterflyWidth k)) :
    (featurePairDiagonalNetwork t a b).run u =
      featurePairRun t (fun i q v => (a i q * v.1, b i q * v.2)) u := by
  change coyoteLayerCyclic (inPlaceDiagonalParameters (fun i q =>
    if butterflyToggleUpper k t.val q then a i (butterflyPairUpper k t q)
    else b i (butterflyPairUpper k t q))).toParameters u = _
  rw [inPlaceDiagonal_correct]
  funext i q
  cases h : butterflyToggleUpper k t.val q <;>
    simp [featurePairRun, featurePairRead, butterflyPairUpper, h,
      butterflyToggleIndex_involutive k t.val q]

/-- Choose one of the two fixed unit shears.
Source: Appendix `def: butterfly`, valid two-coordinate factor. -/
def featurePairUnitShearNetwork {n k : ℕ} (t : Fin k) (upper : Bool) :
    CyclicKCoyoteNetwork n k 1 :=
  featurePairProjectionNetwork t
    {a := 1, b := if upper then 1 else 0, c := if upper then 0 else 1, d := 1}

/-- Upper or lower pair shear as a scalar function.
Source: Appendix `prop: butterfly-hyena`, in-place proof construction. -/
def pairShear (upper : Bool) (a : ℝ) : ℝ × ℝ → ℝ × ℝ :=
  if upper then pairUpperShear a else pairLowerShear a

/-- The fixed unit shear is implemented by one K-constrained layer.
Source: Appendix `lmm:primitives`, fixed linear projection. -/
theorem featurePairUnitShearNetwork_correct {n k : ℕ} (t : Fin k) (upper : Bool)
    (u : RealSequence n (butterflyWidth k)) :
    (featurePairUnitShearNetwork t upper).run u =
      featurePairRun t (fun _ _ v => pairShear upper 1 v) u := by
  cases upper <;>
    simp [featurePairUnitShearNetwork, featurePairProjectionNetwork_correct,
      PairMatrix.apply, pairShear, pairUpperShear, pairLowerShear, add_comm]

/-- Each primitive uses exactly one layer.
Source: Appendix `lmm:primitives`, constant-depth operations. -/
theorem featurePairProjectionNetwork_layerCount {n k : ℕ} (t : Fin k) (M : PairMatrix) :
    (featurePairProjectionNetwork (n := n) t M).layerCount = 1 := rfl

/-- Each variable diagonal also uses exactly one layer.
Source: Appendix `lmm:primitives`, constant-depth operations. -/
theorem featurePairDiagonalNetwork_layerCount {n k : ℕ} (t : Fin k)
    (a b : RealSequence n (butterflyWidth k)) :
    (featurePairDiagonalNetwork t a b).layerCount = 1 := rfl

end Transformer.Zoology
