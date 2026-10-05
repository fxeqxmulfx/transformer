import Transformer.GPTMini.Sparsemax.ProjectionUpdate
import Transformer.GPTMini.Sparsemax.QKProjectedError

/-!
# A nonstandard input row with certified normalized sparse attention

Concrete derived instance for arXiv:1602.02068v2, §2.2 and §2.5,
and the shared projections in `Attention.forward` at `73f8a0b`.
The first two input vectors are `(2, 1, 0)` and `(1, 2, 0)`;
the third is `(0, 0, 1)`. Their inverse is constructed explicitly.
Queries and keys come from actual shared matrices, and the usual
normalized dot product recovers the certified anchored scores.

The row retains the third exact sparse zero and the earlier finite
squared-error correction. This supplies inhabited nonstandard-input
premises for the general derivative and local-minimum results.
It is an unrotated single-row example before XSA and output projection.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

/-- Two mixed prefix coordinates and a separate ordinary coordinate.
Source context: the derived §2.5 input accessibility condition. -/
def mixedRowInputs : Fin 3 → (Fin 3 → ℝ) :=
  fun n f => if n = 2 then basis 2 f else if f = 2 then 0 else if n = f then 2 else 1

/-- The actual inverse input matrix, with the third coordinate unchanged.
Source context: the derived shared-projection example for §2.5. -/
def mixedRowDecoder : (Fin 3 → ℝ) →L[ℝ] (Fin 3 → ℝ) :=
  ContinuousLinearMap.pi fun n => if n = 2 then ContinuousLinearMap.proj 2 else
    (2 / 3 : ℝ) • ContinuousLinearMap.proj n -
      (1 / 3 : ℝ) • ContinuousLinearMap.proj (1 - n)

/-- The concrete inverse decodes each actual input exactly.
Source: the derived matrix accessibility condition for §2.5 of
arXiv:1602.02068v2 and shared projections at `73f8a0b`. -/
theorem mixedRowDecoder_coordinates : ∀ n, mixedRowDecoder (mixedRowInputs n) = basis n := by
  intro n
  funext f
  fin_cases n <;> fin_cases f <;>
    norm_num [mixedRowDecoder, mixedRowInputs, basis, ContinuousLinearMap.pi_apply,
      smul_apply, sub_apply, ContinuousLinearMap.proj_apply]

/-- The mixed input row meets the general independence condition.
Source context: arXiv:1602.02068v2, §2.5, derived accessibility. -/
theorem mixedRowInputs_independent : LinearIndependent ℝ mixedRowInputs :=
  inputIndependent_of_decoder mixedRowInputs mixedRowDecoder mixedRowDecoder_coordinates

/-- An actual query matrix selecting the ordinary input's third coordinate.
Source: the linear query projection at `73f8a0b`, in the derived example. -/
def mixedQKQueries : Fin 3 → EucSpace 2 := fun f => basis 2 f • qkQueryExample

/-- Actual shared key columns for any finite anchor parameters.
Source: the derived decoder lift for §2.5 of arXiv:1602.02068v2,
before QKNorm and sparsemax at `73f8a0b`. -/
def mixedQKKeys (parameters : Fin 2 → ℝ) : Fin 3 → EucSpace 2 :=
  inputKeyLift mixedRowDecoder (qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
    parameters (fun _ : Fin 1 => 0))

/-- The actual query projection gives the certified unit query for this row.
Source: shared query matrix at `73f8a0b` in the derived nonstandard example. -/
theorem mixedQKQueries_row : projectionEvaluation mixedRowInputs mixedQKQueries 2 = qkQueryExample := by
  have hi : mixedRowInputs 2 = basis 2 := by
    funext f
    norm_num [mixedRowInputs]
  rw [projectionEvaluation_apply, hi, projection_basis_column]
  norm_num [mixedQKQueries, basis]

/-- The actual key projection gives the complete certified unit-key family.
Source: the derived input lift for §2.5 of arXiv:1602.02068v2,
evaluated by the shared matrix at `73f8a0b`. -/
theorem mixedQKKeys_row (parameters : Fin 2 → ℝ) :
    projectionEvaluation mixedRowInputs (mixedQKKeys parameters) =
      qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8) parameters (fun _ : Fin 1 => 0) := by
  unfold mixedQKKeys
  exact projectionEvaluation_inputKeyLift mixedRowInputs mixedRowDecoder mixedRowDecoder_coordinates _

/-- The projected key vectors remain unit vectors for all finite anchor parameters.
Source: the derived normalized key chart for §2.5 of arXiv:1602.02068v2,
evaluated by the actual shared matrix at `73f8a0b`. -/
theorem mixedQKKeys_norm (parameters : Fin 2 → ℝ) (n : Fin 3) :
    ‖projectionEvaluation mixedRowInputs (mixedQKKeys parameters) n‖ = 1 := by
  rw [mixedQKKeys_row]
  exact qkAnchoredKeys_norm qkQueryExample qkTransverseExample (1 / 8) parameters
    (fun _ : Fin 1 => 0) qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) n

/-- Actual projected and normalized scores equal the certified sparse anchor scores.
Source: `Attention.forward` at `73f8a0b`, with the derived restrictions
for §2.2 and §2.5 of arXiv:1602.02068v2 on the mixed input row. -/
theorem mixedQKScores_eq (parameters : Fin 2 → ℝ) :
    projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000)
      mixedQKQueries (mixedQKKeys parameters) mixedRowInputs 2 =
        anchoredScores (1 / 8) parameters (fun _ : Fin 1 => 0) := by
  rw [projectedQKScores_evaluation, mixedQKQueries_row, mixedQKKeys_row]
  change qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
    parameters (fun _ : Fin 1 => 0) = _
  exact qkAnchoredScores_eq qkQueryExample qkTransverseExample (1 / 1000000) (1 / 8)
    parameters (fun _ : Fin 1 => 0) qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num)

/-- The actual mixed-input row retains its visible third exact zero.
Source: arXiv:1602.02068v2, §2.2, derived normalized anchor scores. -/
theorem mixedQKProjection_initial : sparseWeights
    (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000)
      mixedQKQueries (mixedQKKeys (fun _ => 0)) mixedRowInputs 2) 2 =
      (fun n : Fin 3 => if n = 2 then 0 else 1 / 2) := by
  rw [mixedQKScores_eq]
  exact anchoredScores_sparse_example

/-- A wrong scalar readout inhabits the ordinary task premises on the mixed row.
Source context: the derived §2.5 example and the value sum at `73f8a0b`. -/
theorem mixedQKReadout_initial : frozenValueReadout
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
      (1 / 1000000) mixedQKQueries (mixedQKKeys (fun _ => 0)) mixedRowInputs 2) 2) = (1 / 2 : ℝ) := by
  rw [mixedQKScores_eq]
  exact anchoredScalarReadout_sparse_example

/-- The complete mixed-input correction starts with positive output loss.
Source: the derived correction for §2.2 and §2.5 of arXiv:1602.02068v2,
through the actual shared projection at `73f8a0b`. -/
theorem mixedQKCorrection_initial_loss : projectedSquaredLoss correctionAnchorValues
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) mixedRowInputs 2 (1 / 2 : ℝ)
    (mixedQKQueries, mixedQKKeys (fun _ => 0)) = 1 / 256 := by
  unfold projectedSquaredLoss
  rw [mixedQKScores_eq]
  exact correctionAnchorLoss_initial

/-- Changing the actual key matrix reaches zero loss on nonstandard inputs.
Source: the derived correction for §2.2 and §2.5 of arXiv:1602.02068v2,
with the actual shared matrix, normalization and sparsemax at `73f8a0b`. -/
theorem mixedQKCorrection_zero_loss : projectedSquaredLoss correctionAnchorValues
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) mixedRowInputs 2 (1 / 2 : ℝ)
    (mixedQKQueries, mixedQKKeys correctionAnchorParameters) = 0 := by
  unfold projectedSquaredLoss
  rw [mixedQKScores_eq]
  exact correctionAnchorLoss_final

/-- The corrected actual matrix retains the third exact sparse zero.
Source: the derived finite correction for §2.2 of arXiv:1602.02068v2,
with shared projections and QKNorm at `73f8a0b`. -/
theorem mixedQKProjection_final : sparseWeights
    (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000)
      mixedQKQueries (mixedQKKeys correctionAnchorParameters) mixedRowInputs 2) 2 =
      (fun n : Fin 3 => if n = 2 then 0 else if n = 0 then 7 / 16 else 9 / 16) := by
  rw [mixedQKScores_eq]
  exact correctionAnchorProjection

end Transformer.GPTMini.Sparsemax
