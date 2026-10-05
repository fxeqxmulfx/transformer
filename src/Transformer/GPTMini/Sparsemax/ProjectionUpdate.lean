import Transformer.GPTMini.Sparsemax.ProjectionLift

/-!
# Local shared projection updates at the actual current matrix

Derived from the matrix accessibility condition for arXiv:1602.02068v2,
§2.5, and the shared linear Q/K projections at `73f8a0b`. A desired
key-vector row is lifted as a change from the current projection, rather
than replacing the current matrix. The update starts at that matrix,
is differentiable and realizes the desired keys exactly when the input
decoder separates the considered inputs.

The construction preserves the current matrix's action on inputs killed
by the decoder. Repeated updates compose exactly: the last desired key
row is realized without losing the original matrix's unseen component.
These properties permit derivative and local-minimum transport at actual
parameter points, rather than only at specially chosen matrix lifts.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall
open scoped BigOperators

/-- An affine change from the actual current shared projection matrix.
Source: the derived accessibility lift for §2.5 of arXiv:1602.02068v2
and the linear projections at `73f8a0b`; all current columns are retained. -/
def inputProjectionUpdate {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ)) (columns : Fin F → EucSpace h)
    (keys : Fin T → EucSpace h) : Fin F → EucSpace h :=
  columns + inputKeyLift decoder (keys - projectionEvaluation inputs columns)

/-- Requesting the current keys starts at the actual matrix itself.
Source: the derived local matrix lift for §2.5 of arXiv:1602.02068v2. -/
theorem inputProjectionUpdate_center {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ)) (columns : Fin F → EucSpace h) :
    inputProjectionUpdate inputs decoder columns (projectionEvaluation inputs columns) = columns := by
  rw [inputProjectionUpdate, sub_self, map_zero, add_zero]

/-- The local affine update realizes every requested key vector exactly.
Source: the derived coordinate decoder for §2.5 of arXiv:1602.02068v2,
evaluated by the shared matrix at `73f8a0b`. -/
theorem inputProjectionUpdate_realizes {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) (columns : Fin F → EucSpace h)
    (keys : Fin T → EucSpace h) :
    projectionEvaluation inputs (inputProjectionUpdate inputs decoder columns keys) = keys := by
  rw [inputProjectionUpdate, map_add, projectionEvaluation_inputKeyLift inputs decoder hd]
  abel

/-- Nonstandard decoded inputs satisfy the actual-update realization premise.
Source context: arXiv:1602.02068v2, §2.5, derived shared matrix update. -/
example : projectionEvaluation mixedInputs
    (inputProjectionUpdate mixedInputs mixedInputDecoder (fun _ : Fin 2 => qkQueryExample)
      (fun _ : Fin 2 => qkTransverseExample)) = (fun _ => qkTransverseExample) :=
  inputProjectionUpdate_realizes mixedInputs mixedInputDecoder mixedInputDecoder_coordinates _ _

/-- The update is differentiable at every desired key row.
Source: the derived matrix lift for §2.5 of arXiv:1602.02068v2;
only ordinary continuous linear maps and affine translations are used. -/
theorem inputProjectionUpdate_differentiableAt {F T h : ℕ}
    (inputs : Fin T → (Fin F → ℝ)) (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (columns : Fin F → EucSpace h) (keys : Fin T → EucSpace h) :
    DifferentiableAt ℝ (inputProjectionUpdate inputs decoder columns) keys := by
  have hd := ((hasFDerivAt_id (𝕜 := ℝ) keys).sub_const
    (projectionEvaluation inputs columns)).differentiableAt
  have hl := (inputKeyLift decoder).differentiableAt.comp keys hd
  unfold inputProjectionUpdate
  simpa only [Function.comp_def, id_eq, Pi.add_def] using (differentiableAt_const columns).add hl

/-- Subsequent updates retain the same unseen matrix component.
Source: the derived shared-projection update for §2.5 of arXiv:1602.02068v2;
the actual new key row is used, not a stale initial-row subtraction. -/
theorem inputProjectionUpdate_compose {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) (columns : Fin F → EucSpace h)
    (firstKeys nextKeys : Fin T → EucSpace h) :
    inputProjectionUpdate inputs decoder (inputProjectionUpdate inputs decoder columns firstKeys)
      nextKeys = inputProjectionUpdate inputs decoder columns nextKeys := by
  rw [inputProjectionUpdate, inputProjectionUpdate_realizes inputs decoder hd]
  unfold inputProjectionUpdate
  rw [map_sub, map_sub, map_sub]
  abel

/-- Two nonstandard-row updates inhabit every composition premise.
Source context: arXiv:1602.02068v2, §2.5, derived matrix update. -/
example : inputProjectionUpdate mixedInputs mixedInputDecoder
    (inputProjectionUpdate mixedInputs mixedInputDecoder (fun _ : Fin 2 => 0)
      (fun _ : Fin 2 => qkQueryExample)) (fun _ : Fin 2 => qkTransverseExample) =
      inputProjectionUpdate mixedInputs mixedInputDecoder (fun _ : Fin 2 => 0)
        (fun _ : Fin 2 => qkTransverseExample) :=
  inputProjectionUpdate_compose mixedInputs mixedInputDecoder mixedInputDecoder_coordinates _ _ _

/-- Inputs unseen by the decoder keep their actual projected vectors.
Source: the derived coordinate update for §2.5 of arXiv:1602.02068v2,
with the same shared matrix action as `Attention.forward` at `73f8a0b`. -/
theorem inputProjectionUpdate_preserves_kernel {F T h : ℕ}
    (inputs : Fin T → (Fin F → ℝ)) (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (columns : Fin F → EucSpace h) (keys : Fin T → EucSpace h) (x : Fin F → ℝ)
    (hx : decoder x = 0) :
    frozenValueReadout (inputProjectionUpdate inputs decoder columns keys) x =
      frozenValueReadout columns x := by
  have he : frozenValueReadout (inputProjectionUpdate inputs decoder columns keys) x =
      frozenValueReadout columns x +
        frozenValueReadout (inputKeyLift decoder (keys - projectionEvaluation inputs columns)) x := by
    simp only [frozenValueReadout_apply, inputProjectionUpdate, Pi.add_apply,
      smul_add, Finset.sum_add_distrib]
  rw [he, inputKeyLift_action, hx, map_zero, add_zero]

/-- A nonzero third input is unseen by a decoder of the first two coordinates.
Source context: the derived §2.5 update, illustrating an actual untouched
matrix direction when the ambient dimension exceeds the decoded row size. -/
example : frozenValueReadout
    (inputProjectionUpdate (fun n : Fin 2 => basis (Fin.castAdd 1 n))
      (ContinuousLinearMap.pi fun n : Fin 2 => ContinuousLinearMap.proj (Fin.castAdd 1 n))
      (fun _ : Fin 3 => qkQueryExample) (fun _ : Fin 2 => qkTransverseExample)) (basis 2) =
      frozenValueReadout (fun _ : Fin 3 => qkQueryExample) (basis 2) := by
  apply inputProjectionUpdate_preserves_kernel
  funext n
  fin_cases n <;> norm_num [ContinuousLinearMap.pi_apply, basis]

/-- The actual QKNorm scores evaluate the requested new keys after an update.
Source: shared projections and normalization at `73f8a0b`, through
the derived local lift for §2.5 of arXiv:1602.02068v2. -/
theorem projectedQKScores_inputProjectionUpdate {F T h : ℕ} (alpha eps : ℝ)
    (queries columns : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) (keys : Fin T → EucSpace h) (i : Fin T) :
    projectedQKScores alpha eps queries (inputProjectionUpdate inputs decoder columns keys) inputs i =
      (fun n => score alpha eps (projectionEvaluation inputs queries i) (keys n)) := by
  rw [projectedQKScores_evaluation, inputProjectionUpdate_realizes inputs decoder hd]

/-- The actual normalized-score update premise is inhabited on nonstandard inputs.
Source context: arXiv:1602.02068v2, §2.5, derived local matrix lift. -/
example : projectedQKScores 0 (1 / 1000000) (fun _ : Fin 2 => qkQueryExample)
    (inputProjectionUpdate mixedInputs mixedInputDecoder (fun _ : Fin 2 => 0)
      (fun _ : Fin 2 => qkTransverseExample)) mixedInputs 1 =
      (fun _ => score 0 (1 / 1000000)
        (projectionEvaluation mixedInputs (fun _ : Fin 2 => qkQueryExample) 1) qkTransverseExample) :=
  projectedQKScores_inputProjectionUpdate _ _ _ _ _ _ mixedInputDecoder_coordinates _ _

/-- Requested key rows near the current row give matrices near the current matrix.
Source: the ordinary local lift for §2.5 of arXiv:1602.02068v2;
this continuity is used to transport local-minimum claims. -/
theorem inputProjectionUpdate_continuousAt {F T h : ℕ}
    (inputs : Fin T → (Fin F → ℝ)) (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (columns : Fin F → EucSpace h) (keys : Fin T → EucSpace h) :
    ContinuousAt (inputProjectionUpdate inputs decoder columns) keys :=
  (inputProjectionUpdate_differentiableAt inputs decoder columns keys).continuousAt

end Transformer.GPTMini.Sparsemax
