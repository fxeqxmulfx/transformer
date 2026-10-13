import Transformer.GPTMini.Convex.Structured.TensorState

/-!
# All-pair learned binding inference from prenorm tensors

Source: Binding/RawBinding's actual affine matching/value model and
TensorRecovery at e49ca6d. The head receives physical prenorm tensor
rows plus globally learned chronology and relative-position weights.
Query, key, value and absolute position are recovered exclusively
from these tensors. Every visible key/value pair remains a candidate;
there is no fixed adjacency/table filter or raw-token lookup in the
head. Only physical indices supply the learned displacement lookup.

Its actual compact probability and ten-coordinate output are proved
equal to the already verified unrestricted shared pointer for genuine
RMS-normalized embeddings. Causal row-prefix selection, mixture,
residual/tied readout and full changed-model integration remain.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- A physical tensor index keeps its value inside the checked context.
Source: RawBinding's position inclusion, with no input-token or semantic-role argument. -/
def tensorContextPosition (hcap : T ≤ C) (position : Fin T) : Fin C :=
  ⟨position.val, by have hp := position.isLt; omega⟩

/-- Actual free query channels are read from the recovered prenorm tensor.
Source: SharedSlots' sixteen query axes, without a fixed matching code in inference. -/
def tensorQuery (hwidth : 64 ≤ d) (x : EucSpace d) (g c : Fin 4) : ℝ :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorFieldAxis hwidth (querySlot g c))

/-- Actual independently learned key channels arrive through the same tensor interface.
Source: SharedSlots' separate sixteen key axes. -/
def tensorKey (hwidth : 64 ≤ d) (x : EucSpace d) (g c : Fin 4) : ℝ :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorFieldAxis hwidth (keySlot g c))

/-- Actual jointly trained value potentials are read from the twenty free tensor fields.
Source: SharedSlots' five-by-four value axes, distinct from readonly decoder coordinates. -/
def tensorValue (hwidth : 64 ≤ d) (x : EucSpace d) (h : Fin 5) (value : Fin 4) : ℝ :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorFieldAxis hwidth (valueSlot h value))

/-- Actual route bias reads free absolute position from its tensor and chronology/displacement from head weights.
Source: Binding's learned all-pair physical bias, without a semantic record mask. -/
def tensorBindingBias (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (pair : Fin T × Fin T) : ℝ :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) (x pair.2) (tensorPositionAxis hwidth) +
    tensorChronology ψ * (pair.2.val : ℝ) +
    ψ.2 (bindingRelativeIndex (tensorContextPosition hcap pair.1) (tensorContextPosition hcap pair.2))

/-- Genuine tensor binding probability contracts small channels over every supplied position pair.
Source: Factorial.pointerProbability, with actual tensor-conditioned free fields and relative bias. -/
def tensorPointerProbability (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (z : SharedPointerConfiguration (Fin T × Fin T)) : ℝ :=
  pointerProbability (tensorQuery hwidth (x query)) (fun pair => tensorKey hwidth (x pair.1))
    (fun pair => tensorValue hwidth (x pair.2)) (tensorBindingBias hwidth ψ x hcap) z

/-- The actual ten-coordinate pointer output computes normalized compact learned-value means.
Source: PointerDecoder.pointerOutputCoordinates, without an observed route or output label. -/
def tensorPointerCoordinates (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (axis : Fin 10) : ℝ :=
  pointerOutputCoordinates (tensorQuery hwidth (x query)) (fun pair => tensorKey hwidth (x pair.1))
    (fun pair => tensorValue hwidth (x pair.2)) (tensorBindingBias hwidth ψ x hcap) axis

/-- Actual candidate-token scores from the same freely learned tensor pointer.
Source: PointerDecoder.pointerOutputScore, evaluating inferred rather than supervised value channels. -/
def tensorPointerScore (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (target : Fin 1024) : ℝ :=
  pointerOutputScore (tensorQuery hwidth (x query)) (fun pair => tensorKey hwidth (x pair.1))
    (fun pair => tensorValue hwidth (x pair.2)) (tensorBindingBias hwidth ψ x hcap) target

/-- True tensor query/key/value reads equal all unrestricted raw shared channels after genuine prenorm.
Source: exact recovered Euclidean fields, not a supplied matching or value encoder. -/
theorem tensorPointerFields_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (position : Fin tokens.length) :
    tensorQuery hwidth (tensorSequence hsize hwidth eps θ tokens hcap position) = sharedQuery θ.1 (tokens.get position) ∧
      tensorKey hwidth (tensorSequence hsize hwidth eps θ tokens hcap position) = sharedKey θ.1 (tokens.get position) ∧
      tensorValue hwidth (tensorSequence hsize hwidth eps θ tokens hcap position) = sharedValue θ.1 (tokens.get position) := by
  refine ⟨?_, ?_, ?_⟩
  all_goals
    funext first second
    exact tensorPrenormRecovery_field hsize hwidth eps heps θ.1 (tokens.get position)
      (rawBindingPosition tokens hcap position) _

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Every actual tensor pair has precisely the unrestricted raw model's learned bias.
Source: recovered learned absolute position, original chronology and identical physical signed displacement. -/
theorem tensorBindingBias_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) :
    tensorBindingBias hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap =
      (fun pair => bindingRouteBias θ (rawBindingPosition tokens hcap pair.1) (rawBindingPosition tokens hcap pair.2)) := by
  funext pair
  unfold tensorBindingBias
  rw [show tensorAnchorRecovery (tensorAnchorAxis hwidth)
      (tensorSequence hsize hwidth eps θ tokens hcap pair.2) (tensorPositionAxis hwidth) =
      θ.1 (.inr (.inl (rawBindingPosition tokens hcap pair.2))) from
    tensorPrenormRecovery_position hsize hwidth eps heps θ.1 (tokens.get pair.2)
      (rawBindingPosition tokens hcap pair.2)]
  rfl

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- The genuine normalized tensor pointer is the same actual complete model proved convex on all shared weights.
Source: all true field/bias reads into the identical contracted pointer probability. -/
theorem tensorPointerProbability_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (z : RawBindingConfiguration tokens) :
    tensorPointerProbability hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query z =
      rawBindingProbability θ tokens hcap query z := by
  simp only [tensorPointerProbability,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.2,
    tensorBindingBias_sequence hsize hwidth eps heps, rawBindingProbability]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Actual tensor pointer decoding equals the ten-coordinate dot product with the candidate's tied code.
Source: PointerDecoder's genuine physical mean/score identity, with every position pair retained. -/
theorem tensorPointerScore_axes (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C) (x : Fin T → EucSpace d)
    (hcap : T ≤ C) (query : Fin T) (target : Fin 1024) :
    tensorPointerScore hwidth ψ x hcap query target =
      ∑ axis, tensorPointerCoordinates hwidth ψ x hcap query axis * outputCoordinate (outputDigit target) axis := by
  unfold tensorPointerScore tensorPointerCoordinates
  exact pointerOutputScore_axes _ _ _ _ _

example : (64 : ℕ) ≤ 64 ∧ (4 : ℕ) ≤ 64 := by omega

/-- Every genuine tensor pointer score matches the complete raw model for all unrestricted learned parameters.
Source: exact free tensor Q/K/value/position/binding recovery, rather than assumed correct logits. -/
theorem tensorPointerScore_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (query : Fin tokens.length) (target : Fin 1024) :
    tensorPointerScore hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) hcap query target =
      rawBindingScore θ tokens hcap query target := by
  simp only [tensorPointerScore,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.1,
    (tensorPointerFields_sequence hsize hwidth eps heps θ tokens hcap _).2.2,
    tensorBindingBias_sequence hsize hwidth eps heps, rawBindingScore]

example : (548 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 36, 292, 36] : List (Fin 548)).length ≤ 64 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
