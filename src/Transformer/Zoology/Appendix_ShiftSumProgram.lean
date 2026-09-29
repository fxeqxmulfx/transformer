/-
# Stacking shifted-diagonal factors without accumulating workspace

Arora et al., arXiv:2312.04927v1, Appendix `prop: butterfly-hyena`,
`lmm: kaleido-coyote`, and `lem: stacking-layers`. Every factor clears its
extra rows, so a product of L factors shares one ninefold sequence workspace
and has exactly 3L layers. All weights have explicit K representations.
-/

import Transformer.Zoology.Appendix_BlockButterflyCoyote

namespace Transformer.Zoology

/-- One factor given by its three diagonals and cyclic offsets.
Source: Appendix `eq: butterfly-split`, corrected shifted representation. -/
structure ShiftSumStage (n d : ℕ) where
  coefficient : Fin 3 → RealSequence n d
  shift : Fin 3 → Fin n

/-- Apply the three shifted diagonal branches of a factor.
Source: Appendix `eq: butterfly-split`. -/
def ShiftSumStage.apply {n d : ℕ} (s : ShiftSumStage n d)
    (u : RealSequence n d) : RealSequence n d :=
  shiftedDiagonalSum s.coefficient s.shift u

/-- Evaluate a list of factors in matrix multiplication order.
Source: Appendix `def: butterfly` and `def: kaleidoscope`, products. -/
def applyShiftSumStages {n d : ℕ} (stages : List (ShiftSumStage n d))
    (u : RealSequence n d) : RealSequence n d :=
  stages.foldl (fun state s => s.apply state) u

/-- Compile all factors into one K-constrained network on shared rows.
Source: Appendix `lem: stacking-layers` and `prop: butterfly-hyena`. -/
def compileShiftSumStages {n k : ℕ} [NeZero n] :
    List (ShiftSumStage n (butterflyWidth k)) → CyclicKCoyoteNetwork (9 * n) k 1
  | [] => {layers := []}
  | s :: ss => (shiftSumKNetwork s.coefficient s.shift).append
      (compileShiftSumStages ss)

/-- A factor product is realized exactly, with cleared shared workspace.
Source: Appendix `lem: stacking-layers`, with explicit factor construction. -/
theorem compileShiftSumStages_correct {n k : ℕ} [NeZero n]
    (stages : List (ShiftSumStage n (butterflyWidth k)))
    (u : RealSequence n (butterflyWidth k)) :
    (compileShiftSumStages stages).run (shiftSumPad u) =
      shiftSumPad (applyShiftSumStages stages u) := by
  induction stages generalizing u with
  | nil => rfl
  | cons s ss ih =>
      rw [compileShiftSumStages, CyclicKCoyoteNetwork.run_append,
        shiftSumKNetwork_correct, ih]
      rfl

/-- Each factor contributes exactly three layers.
Source: Appendix `prop: butterfly-hyena`, composition component. -/
theorem compileShiftSumStages_layerCount {n k : ℕ} [NeZero n]
    (stages : List (ShiftSumStage n (butterflyWidth k))) :
    (compileShiftSumStages stages).layerCount = 3 * stages.length := by
  induction stages with
  | nil => rfl
  | cons s ss ih =>
      rw [compileShiftSumStages, CyclicKCoyoteNetwork.layerCount_append,
        shiftSumKNetwork_layerCount, ih, List.length_cons]
      omega

/-- Exact scalar storage for the whole product; the sequence workspace
does not grow with the number of factors.
Source: Appendix `prop: single-baseconv` and `lem: stacking-layers`. -/
theorem compileShiftSumStages_parameterCount {n k : ℕ} [NeZero n]
    (stages : List (ShiftSumStage n (butterflyWidth k))) :
    (compileShiftSumStages stages).parameterCount =
      stages.length * (3 * (27 * n * butterflyWidth k +
        4 * (k + 1) * butterflyWidth (k + 1))) := by
  induction stages with
  | nil => simp [compileShiftSumStages, CyclicKCoyoteNetwork.parameterCount]
  | cons s ss ih =>
      rw [compileShiftSumStages, CyclicKCoyoteNetwork.parameterCount_append,
        shiftSumKNetwork_parameterCount, ih, List.length_cons]
      ring

/-- Externally shaped model for a product of shifted-diagonal factors.
Source: Appendix `prop: butterfly-hyena`, workspace shape and composition. -/
def shiftSumProgramModel {n k : ℕ} [NeZero n]
    (stages : List (ShiftSumStage n (butterflyWidth k))) :
    PaddedCyclicCoyoteModel n (butterflyWidth k) := {
  innerLength := 9 * n
  innerWidth := butterflyWidth k
  lengthBound := by omega
  widthBound := le_refl _
  network := (compileShiftSumStages stages).toNetwork
}

/-- Zero padding, execution, and cropping preserve the exact product.
Source: Appendix `prop: butterfly-hyena` and `lem: stacking-layers`. -/
theorem shiftSumProgramModel_correct {n k : ℕ} [NeZero n]
    (stages : List (ShiftSumStage n (butterflyWidth k)))
    (u : RealSequence n (butterflyWidth k)) :
    (shiftSumProgramModel stages).run u = applyShiftSumStages stages u := by
  have hpad : (padSequence u : RealSequence (9 * n) (butterflyWidth k)) =
      shiftSumPad u := by
    funext i q
    simp [padSequence, shiftSumPad]
  unfold PaddedCyclicCoyoteModel.run shiftSumProgramModel
  rw [hpad, ((compileShiftSumStages stages).toNetwork_correct _).1,
    compileShiftSumStages_correct]
  funext i q
  simp [cropSequence, shiftSumPad]

end Transformer.Zoology
