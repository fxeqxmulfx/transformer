/-
# Pairwise linear actions in the unchanged feature layout

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`prop: butterfly-hyena`. A canonical pair read makes composition explicit
even when every row and every pair has different scalar coefficients.
-/

import Transformer.Zoology.Appendix_PairMatrixFactorization

namespace Transformer.Zoology

/-- Selecting the canonical upper endpoint twice changes nothing.
Source: Appendix `def: butterfly`, pair indexing. -/
theorem butterflyPairUpper_idem (k : ℕ) (t : Fin k)
    (q : Fin (butterflyWidth k)) :
    butterflyPairUpper k t (butterflyPairUpper k t q) = butterflyPairUpper k t q := by
  change (if butterflyToggleUpper k t.val (butterflyPairUpper k t q) then
    butterflyPairUpper k t q
    else butterflyToggleIndex k t.val (butterflyPairUpper k t q)) = _
  rw [butterflyPairUpper_orientation]
  rfl

/-- Read both coordinates of the feature pair in upper/lower order.
Source: Appendix `def: butterfly`, two-coordinate block action. -/
def featurePairRead {n k : ℕ} (t : Fin k)
    (u : RealSequence n (butterflyWidth k)) (i : Fin n)
    (q : Fin (butterflyWidth k)) : ℝ × ℝ :=
  (u i (butterflyPairUpper k t q),
    u i (butterflyToggleIndex k t.val (butterflyPairUpper k t q)))

/-- Execute an independently parameterized function on every feature pair.
Source: Appendix `def: butterfly`, all diagonal blocks may differ. -/
def featurePairRun {n k : ℕ} (t : Fin k)
    (F : Fin n → Fin (butterflyWidth k) → (ℝ × ℝ → ℝ × ℝ))
    (u : RealSequence n (butterflyWidth k)) : RealSequence n (butterflyWidth k) :=
  fun i q => if butterflyToggleUpper k t.val q then
    (F i (butterflyPairUpper k t q) (featurePairRead t u i q)).1
  else (F i (butterflyPairUpper k t q) (featurePairRead t u i q)).2

/-- Both outputs of a pair use its same function and its same input pair.
Source: Appendix `def: butterfly`, exact pair semantics. -/
theorem featurePairRead_run {n k : ℕ} (t : Fin k)
    (F : Fin n → Fin (butterflyWidth k) → (ℝ × ℝ → ℝ × ℝ))
    (u : RealSequence n (butterflyWidth k)) (i : Fin n)
    (q : Fin (butterflyWidth k)) :
    featurePairRead t (featurePairRun t F u) i q =
      F i (butterflyPairUpper k t q) (featurePairRead t u i q) := by
  apply Prod.ext <;>
    simp [featurePairRead, featurePairRun, butterflyPairUpper_orientation,
      butterflyPairUpper_toggle, butterflyPairUpper_idem, butterflyToggleUpper_toggle]

/-- Pairwise execution composes without any temporary coordinates.
Source: Appendix `lem: stacking-layers`, pairwise specialization. -/
theorem featurePairRun_comp {n k : ℕ} (t : Fin k)
    (F G : Fin n → Fin (butterflyWidth k) → (ℝ × ℝ → ℝ × ℝ))
    (u : RealSequence n (butterflyWidth k)) :
    featurePairRun t G (featurePairRun t F u) =
      featurePairRun t (fun i q v => G i q (F i q v)) u := by
  funext i q
  simp only [featurePairRun, featurePairRead_run]

/-- The identity pair action retains every original scalar coordinate.
Source: Appendix `def: butterfly`, identity blocks. -/
theorem featurePairRun_id {n k : ℕ} (t : Fin k)
    (u : RealSequence n (butterflyWidth k)) :
    featurePairRun t (fun _ _ v => v) u = u := by
  funext i q
  cases h : butterflyToggleUpper k t.val q <;>
    simp [featurePairRun, featurePairRead, butterflyPairUpper, h,
      butterflyToggleIndex_involutive k t.val q]

/-- Fixed pair matrix as a shared actual butterfly projection.
Source: Appendix `def: W-kmat`, admissible fixed linear weight. -/
def featurePairTree {k : ℕ} (t : Fin k) (M : PairMatrix) : ButterflyTree k :=
  singleLevelButterfly k t.val
    (fun q => if butterflyToggleUpper k t.val q then M.a else M.d)
    (fun q => if butterflyToggleUpper k t.val q then M.b else M.c)

/-- The shared tree has precisely the ordinary pair matrix action.
Source: Appendix `def: butterfly`, four arbitrary diagonals. -/
theorem featurePairTree_apply {k : ℕ} (t : Fin k) (M : PairMatrix)
    (v : Fin (butterflyWidth k) → ℝ) :
    (featurePairTree t M).apply v =
      fun q => if butterflyToggleUpper k t.val q then
        (M.apply (v (butterflyPairUpper k t q),
          v (butterflyToggleIndex k t.val (butterflyPairUpper k t q)))).1
      else (M.apply (v (butterflyPairUpper k t q),
          v (butterflyToggleIndex k t.val (butterflyPairUpper k t q)))).2 := by
  rw [featurePairTree, singleLevelButterfly_apply k t.val t.isLt]
  funext q
  cases h : butterflyToggleUpper k t.val q <;>
    simp [butterflyPairUpper, h, PairMatrix.apply,
      butterflyToggleIndex_involutive k t.val q, add_comm]

end Transformer.Zoology
