import Transformer.GPTMini.Sparsemax.InputDecoder

/-!
# Actual projection changes realizing arbitrary independent input rows

Derived accessibility construction for arXiv:1602.02068v2, §2.5,
and the shared linear query/key matrices in `Attention.forward` at
`73f8a0b`. Evaluation applies one matrix to all inputs. A coordinate
decoder gives a continuous linear lift of any desired key-vector row
back into the columns of that shared matrix. The right-inverse identity
is proved for the actual finite weighted sums.

The decoder's existence follows from input independence in `InputDecoder`.
No standard-basis input, orthogonality or equality of input and context
dimensions is assumed. The rank condition is explicit: dependent inputs
cannot in general admit arbitrary independent key changes.
These linear constructions precede normalization, sparsemax and task loss.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall
open scoped BigOperators

/-- Evaluate one actual matrix on the entire fixed input family.
Source: the linear projections in `Attention.forward` at `73f8a0b`,
with columns as parameters and the ordinary finite matrix-vector sum. -/
def projectionEvaluation {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ)) :
    (Fin F → EucSpace h) →L[ℝ] (Fin T → EucSpace h) :=
  ContinuousLinearMap.pi fun n =>
    ∑ f : Fin F, inputs n f • (ContinuousLinearMap.proj f)

/-- Lift desired key vectors to a shared matrix using input coordinates.
Derived construction for §2.5 of arXiv:1602.02068v2 and the linear
Q/K projections at `73f8a0b`; the decoder is a genuine linear map. -/
def inputKeyLift {F T h : ℕ} (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ)) :
    (Fin T → EucSpace h) →L[ℝ] (Fin F → EucSpace h) :=
  ContinuousLinearMap.pi fun f =>
    ∑ n : Fin T, decoder (basis f) n • (ContinuousLinearMap.proj n)

/-- Matrix evaluation is the existing weighted sum of its columns.
Source: linear projections in `Attention.forward` at `73f8a0b`,
as represented by the value aggregation's existing linear operator. -/
theorem projectionEvaluation_apply {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (columns : Fin F → EucSpace h) (n : Fin T) :
    projectionEvaluation inputs columns n = frozenValueReadout columns (inputs n) := by
  rw [frozenValueReadout_apply]
  simp only [projectionEvaluation, ContinuousLinearMap.pi_apply, sum_apply,
    smul_apply, ContinuousLinearMap.proj_apply]

/-- Each lifted column is the desired row aggregated with the decoder's coefficients.
Source: the derived coordinate lift for §2.5 of arXiv:1602.02068v2,
before the shared normalized projections at `73f8a0b`. -/
theorem inputKeyLift_apply {F T h : ℕ} (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (keys : Fin T → EucSpace h) (f : Fin F) :
    inputKeyLift decoder keys f = frozenValueReadout keys (decoder (basis f)) := by
  rw [frozenValueReadout_apply]
  simp only [inputKeyLift, ContinuousLinearMap.pi_apply, sum_apply,
    smul_apply, ContinuousLinearMap.proj_apply]

/-- The lifted matrix acts on every input as the decoded weighted key sum.
Source: the derived input-decoder lift for §2.5 of arXiv:1602.02068v2;
the actual matrix action at `73f8a0b` is retained. -/
theorem inputKeyLift_action {F T h : ℕ} (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (keys : Fin T → EucSpace h) (x : Fin F → ℝ) :
    frozenValueReadout (inputKeyLift decoder keys) x =
      frozenValueReadout keys (decoder x) := by
  classical
  have hx : (∑ f : Fin F, x f • basis f) = x := by
    funext j
    simp [basis, smul_eq_mul]
  calc
    frozenValueReadout (inputKeyLift decoder keys) x =
        ∑ f : Fin F, x f • frozenValueReadout keys (decoder (basis f)) := by
      rw [frozenValueReadout_apply]
      simp only [inputKeyLift_apply]
    _ = frozenValueReadout keys (decoder (∑ f : Fin F, x f • basis f)) := by
      simp only [map_sum, map_smul]
    _ = frozenValueReadout keys (decoder x) := by rw [hx]

/-- Coordinate decoding realizes every desired key vector exactly.
Source: the derived accessibility restriction for §2.5 of
arXiv:1602.02068v2 and the actual shared projection at `73f8a0b`. -/
theorem projectionEvaluation_inputKeyLift {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) (keys : Fin T → EucSpace h) :
    projectionEvaluation inputs (inputKeyLift decoder keys) = keys := by
  funext n
  rw [projectionEvaluation_apply, inputKeyLift_action, hd n, projection_basis_column]

/-- Nonstandard inputs and their explicit decoder inhabit the realization premise.
Source context: arXiv:1602.02068v2, §2.5, derived shared matrix lift. -/
example : projectionEvaluation mixedInputs
    (inputKeyLift mixedInputDecoder (fun _ : Fin 2 => qkQueryExample)) =
      (fun _ : Fin 2 => qkQueryExample) :=
  projectionEvaluation_inputKeyLift mixedInputs mixedInputDecoder mixedInputDecoder_coordinates _

/-- A proved coordinate decoder makes shared matrix evaluation surjective.
Source: the derived matrix-accessibility condition for §2.5 of
arXiv:1602.02068v2; this permits finite changes and local parameter lifts. -/
theorem projectionEvaluation_surjective {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) :
    Function.Surjective (projectionEvaluation (h := h) inputs) := by
  intro keys
  exact ⟨inputKeyLift decoder keys, projectionEvaluation_inputKeyLift inputs decoder hd keys⟩

/-- The surjectivity hypothesis is satisfied by a nonorthogonal input matrix.
Source context: arXiv:1602.02068v2, §2.5, derived input coordinates. -/
example : Function.Surjective (projectionEvaluation (h := 2) mixedInputs) :=
  projectionEvaluation_surjective mixedInputs mixedInputDecoder mixedInputDecoder_coordinates

/-- Independent inputs alone give arbitrary key rows through a shared matrix.
Source: the derived input restriction for §2.5 of arXiv:1602.02068v2
and the actual linear Q/K projections at `73f8a0b`; decoder existence is proved. -/
theorem projectionEvaluation_surjective_of_independent {F T h : ℕ}
    (inputs : Fin T → (Fin F → ℝ)) (hi : LinearIndependent ℝ inputs) :
    Function.Surjective (projectionEvaluation (h := h) inputs) := by
  obtain ⟨decoder, hd⟩ := inputDecoder_exists inputs hi
  exact projectionEvaluation_surjective inputs decoder hd

/-- The rank-only hypothesis has a concrete nonstandard instance.
Source context: arXiv:1602.02068v2, §2.5, derived matrix accessibility. -/
example : Function.Surjective (projectionEvaluation (h := 2) mixedInputs) :=
  projectionEvaluation_surjective_of_independent mixedInputs mixedInputs_independent

/-- The actual normalized scores depend on the evaluated matrix columns.
Source: shared Q/K projections and QKNorm in `Attention.forward` at
`73f8a0b`, before the score path of arXiv:1602.02068v2, §2.5. -/
theorem projectedQKScores_evaluation {F T h : ℕ} (alpha eps : ℝ)
    (queries keys : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ)) (i : Fin T) :
    projectedQKScores alpha eps queries keys inputs i =
      (fun n => score alpha eps (projectionEvaluation inputs queries i)
        (projectionEvaluation inputs keys n)) := by
  funext n
  rw [projectedQKScores, projectionEvaluation_apply, projectionEvaluation_apply]

/-- The lifted shared matrix realizes the desired keys inside the actual QKNorm scores.
Source: the derived input decoder for §2.5 of arXiv:1602.02068v2,
composed with the shared projections and normalization at `73f8a0b`. -/
theorem projectedQKScores_inputKeyLift {F T h : ℕ} (alpha eps : ℝ)
    (queries : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin T → ℝ))
    (hd : ∀ n, decoder (inputs n) = basis n) (keys : Fin T → EucSpace h) (i : Fin T) :
    projectedQKScores alpha eps queries (inputKeyLift decoder keys) inputs i =
      (fun n => score alpha eps (projectionEvaluation inputs queries i) (keys n)) := by
  rw [projectedQKScores_evaluation, projectionEvaluation_inputKeyLift inputs decoder hd]

/-- Nonstandard inputs inhabit the actual normalized-score lift premise.
Source context: arXiv:1602.02068v2, §2.5, shared Q/K at `73f8a0b`. -/
example : projectedQKScores 0 (1 / 1000000) (fun _ : Fin 2 => qkQueryExample)
    (inputKeyLift mixedInputDecoder (fun _ : Fin 2 => qkTransverseExample)) mixedInputs 1 =
      (fun n => score 0 (1 / 1000000)
        (projectionEvaluation mixedInputs (fun _ : Fin 2 => qkQueryExample) 1)
        ((fun _ : Fin 2 => qkTransverseExample) n)) :=
  projectedQKScores_inputKeyLift _ _ _ _ _ mixedInputDecoder_coordinates _ _

end Transformer.GPTMini.Sparsemax
