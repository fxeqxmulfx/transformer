/-
# Both axes of a row-major butterfly factor

Arora et al., arXiv:2312.04927v1, Appendix `def: butterfly` and
`prop: butterfly-hyena`. For power-of-two sequence and feature dimensions,
every binary pairing lies either inside a feature row or along a feature
column. Both cases use three K-constrained layers on the same workspace.
-/

import Transformer.Zoology.Appendix_ButterflyToggleShift

namespace Transformer.Zoology

/-- One binary butterfly factor with arbitrary diagonals at every entry.
The axis identifies the paired row or feature digit. Source: Appendix
`def: butterfly`, four arbitrary diagonals of each two-half block. -/
structure BinaryButterflyStage (p k : ℕ) where
  axis : Fin p ⊕ Fin k
  main : RealSequence (butterflyWidth p) (butterflyWidth k)
  off : RealSequence (butterflyWidth p) (butterflyWidth k)

/-- Apply the factor in the original sequence/feature layout.
Source: Appendix `def: butterfly` and `eq: butterfly-split`. -/
def BinaryButterflyStage.apply {p k : ℕ} (s : BinaryButterflyStage p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    RealSequence (butterflyWidth p) (butterflyWidth k) :=
  fun i q => s.main i q * u i q + s.off i q *
    match s.axis with
    | .inl t => u (butterflyToggleIndex p t.val i) q
    | .inr t => u i (butterflyToggleIndex k t.val q)

/-- The sequence-axis factor's masked main and opposite-shift diagonals.
Source: Appendix `eq: butterfly-split`, upper/lower block masks. -/
def binaryButterflyRowCoefficients {p k : ℕ} (t : Fin p)
    (main off : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    Fin 3 → RealSequence (butterflyWidth p) (butterflyWidth k) :=
  fun a i q => if a = 0 then main i q else if a = 1 then
    if butterflyToggleUpper p t.val i then 0 else off i q
  else if butterflyToggleUpper p t.val i then off i q else 0

/-- Off-diagonal reads retain the correct positive and negative offsets.
Source: Appendix `eq: butterfly-split`, corrected shifted display. -/
def binaryButterflyRowShifts {p : ℕ} (t : Fin p) : Fin 3 → Fin (butterflyWidth p) :=
  fun a => if a = 0 then 0 else if a = 1 then butterflyToggleOffset p t
    else -butterflyToggleOffset p t

/-- The sequence-axis case is exactly the masked three-shift construction.
Source: Appendix `eq: butterfly-split` and `prop: butterfly-hyena`. -/
theorem binaryButterflyRow_apply {p k : ℕ} (t : Fin p)
    (main off u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    shiftedDiagonalSum (binaryButterflyRowCoefficients t main off)
      (binaryButterflyRowShifts t) u =
        ({axis := .inl t, main := main, off := off} : BinaryButterflyStage p k).apply u := by
  funext i q
  simp only [BinaryButterflyStage.apply, butterflyToggleIndex_shift]
  by_cases hu : butterflyToggleUpper p t.val i = true <;>
    simp [shiftedDiagonalSum, Fin.sum_univ_three, binaryButterflyRowCoefficients,
      binaryButterflyRowShifts, hu, sub_neg_eq_add]

/-- Compile either axis into three layers with shared 9n sequence workspace
and unchanged feature width. Source: Appendix `prop: butterfly-hyena`. -/
def BinaryButterflyStage.compile {p k : ℕ} (s : BinaryButterflyStage p k) :
    CyclicKCoyoteNetwork (9 * butterflyWidth p) k 1 :=
  match s.axis with
  | .inl t => shiftSumKNetwork (binaryButterflyRowCoefficients t s.main s.off)
      (binaryButterflyRowShifts t)
  | .inr t => diagonalLinearKNetwork (butterflyToggleK k t.val) s.main s.off

/-- Both row-major cases compute the original factor exactly and leave
zero workspace. Source: Appendix `prop: butterfly-hyena`, corrected proof. -/
theorem BinaryButterflyStage.compile_correct {p k : ℕ} (s : BinaryButterflyStage p k)
    (u : RealSequence (butterflyWidth p) (butterflyWidth k)) :
    s.compile.run (shiftSumPad u) = shiftSumPad (s.apply u) := by
  rcases s with ⟨axis, main, off⟩
  cases axis with
  | inl t =>
      rw [BinaryButterflyStage.compile, shiftSumKNetwork_correct,
        binaryButterflyRow_apply]
  | inr t =>
      rw [BinaryButterflyStage.compile, diagonalLinearKNetwork_correct]
      congr 1
      funext i q
      simp [diagonalLinearApply, BinaryButterflyStage.apply,
        butterflyToggleK_linearProjection]

/-- Either axis takes exactly three layers, independent of block size.
Source: Appendix `prop: butterfly-hyena`, constant-depth bound. -/
theorem BinaryButterflyStage.compile_layerCount {p k : ℕ} (s : BinaryButterflyStage p k) :
    s.compile.layerCount = 3 := by
  rcases s with ⟨axis, main, off⟩
  cases axis with
  | inl t => exact shiftSumKNetwork_layerCount _ _
  | inr t => exact diagonalLinearKNetwork_layerCount _ _ _

/-- Exact storage, including every compressed weight, filter, and bias.
Source: Appendix `def: W-kmat` and `prop: single-baseconv`. -/
theorem BinaryButterflyStage.compile_parameterCount {p k : ℕ}
    (s : BinaryButterflyStage p k) :
    s.compile.parameterCount = 81 * butterflyWidth p * butterflyWidth k +
      12 * (k + 1) * butterflyWidth (k + 1) := by
  rcases s with ⟨axis, main, off⟩
  cases axis with
  | inl t =>
      rw [BinaryButterflyStage.compile, shiftSumKNetwork_parameterCount]
      ring
  | inr t =>
      rw [BinaryButterflyStage.compile, diagonalLinearKNetwork_parameterCount,
        butterflyToggleK_width]
      ring

/-- Every stored weight has hierarchy width one and common expansion two;
this verifies the actual K restriction, not only a scalar storage count.
Source: Appendix `def: W-kmat`, polylogarithmic hierarchy bounds. -/
theorem BinaryButterflyStage.compile_weightWidths {p k : ℕ}
    (s : BinaryButterflyStage p k) :
    (s.compile.layers.map fun layer => layer.weight.inner.width) = [1, 1, 1] := by
  rcases s with ⟨axis, main, off⟩
  cases axis <;>
    simp [BinaryButterflyStage.compile, shiftSumKNetwork, shiftSumNetwork,
      diagonalLinearKNetwork, zeroWeightCyclicKParameters, diagonalLinearKBranch,
      zeroExpandedKaleidoscope, butterflyToggleK, butterflyTreeK, Kaleidoscope.width]

end Transformer.Zoology
