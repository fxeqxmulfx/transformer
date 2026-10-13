import Transformer.GPTMini.Convex.Structured.TensorRecovery
import Mathlib.Data.List.FinRange

/-!
# Learned state/value inference from genuine prenorm tensors

Source: MarkovChain/MarkovEmissions' actual compact state propagation
and TensorRecovery's exact protected-coordinate recovery at e49ca6d.
The head below receives ordinary `Fin T → EucSpace d` prenorm tensors.
It has no token table, raw IDs, labels or semantic reference transition.
Its learned global parameters contain chronology, initial state,
conditional value and mixture logits plus relative binding potentials.
Token-conditioned transitions are read exclusively from input tensors.

The actual RMS-normalized embedding sequence is proved to produce
exactly the already verified shared learned state/value computation,
for every free parameter assignment. Compact ten-coordinate values
are coupled to the true scalar decoder. Residual/tied readout and
causal row-prefix selection remain further integration obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.GPTMini.Semantics
open scoped BigOperators Classical
noncomputable section

variable {V C d T : ℕ}

/-- Head-global coordinates exclude the token table and positional embedding rows.
Source: SharedSlots' actual chronology, six initial logits, 120 value logits and two branch logits. -/
abbrev TensorGlobal := Unit ⊕ (Fin 6 ⊕ ((Fin 6 × (Fin 5 × Fin 4)) ⊕ Fin 2))

/-- Actual tensor head weights contain only globally shared logits and relative displacement potentials.
Source: Binding's compact learned parameter layout; token/absolute-position fields arrive through tensors. -/
abbrev TensorHeadParameters (C : ℕ) := (TensorGlobal → ℝ) × (Fin (C + (C - 1)) → ℝ)

/-- Split genuine free head weights from the jointly learned embedding/position parameter vector.
Source: actual BindingParameters coordinates, with no duplicate per-occurrence token weights. -/
def tensorHeadParameters (θ : BindingParameters V C) : TensorHeadParameters C :=
  (fun field => θ.1 (.inr (.inr field)), θ.2)

/-- Actual learned initial logits supplied to the tensor state head.
Source: the six free global initial-state coordinates. -/
def tensorInitial (ψ : TensorHeadParameters C) (state : Fin 6) : ℝ := ψ.1 (.inr (.inl state))

/-- Actual jointly learned conditional output-channel logits.
Source: the complete six-by-five-by-four free global value table. -/
def tensorEmission (ψ : TensorHeadParameters C) (state : Fin 6) (h : Fin 5) (value : Fin 4) : ℝ :=
  ψ.1 (.inr (.inr (.inl (state, (h, value)))))

/-- Actual learned mixture softmax, shared by both tensor heads.
Source: MixedHeads' two free global branch logits. -/
def tensorHeadWeight (ψ : TensorHeadParameters C) (head : Fin 2) : ℝ :=
  stateRow (fun branch => ψ.1 (.inr (.inr (.inr branch)))) head

/-- Chronology remains a free global coefficient read by the binding tensor head.
Source: SharedPointer's single learned physical-position coefficient. -/
def tensorChronology (ψ : TensorHeadParameters C) : ℝ := ψ.1 (.inl ())

/-- Every global tensor-head read retains the corresponding original unrestricted shared weight.
Source: the true coordinate split, without supplying probabilities or a correct state. -/
theorem tensorHeadParameters_reads (θ : BindingParameters V C) :
    tensorInitial (tensorHeadParameters θ) = (fun state => sharedInitialRead state θ.1) ∧
      tensorEmission (tensorHeadParameters θ) = (fun state h value => sharedEmissionRead state h value θ.1) ∧
      tensorHeadWeight (tensorHeadParameters θ) = mixedHeadWeight θ ∧
      tensorChronology (tensorHeadParameters θ) = sharedChronologyRead θ.1 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- True learned transition logits are recovered from the head's actual prenorm tensor input alone.
Source: TensorRecovery's internal ratio operation and SharedSlots' 36 transition axes. -/
def tensorTransition (hwidth : 64 ≤ d) (x : EucSpace d) (previous next : Fin 6) : ℝ :=
  tensorAnchorRecovery (tensorAnchorAxis hwidth) x (tensorFieldAxis hwidth (transitionSlot previous next))

/-- The real prenorm tensor sequence from actual raw vocabulary tokens and learned positions.
Source: TensorEmbedding and GPTMini.RMSNorm; physical indexing is preserved in the checked context. -/
def tensorSequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C)
    (position : Fin tokens.length) : EucSpace d :=
  rmsNormEps eps (tensorInput hsize hwidth θ.1 (tokens.get position) (rawBindingPosition tokens hcap position))

/-- The actual tensor-conditioned matrix is exactly the raw free transition table after genuine prenorm.
Source: the proved whole-tensor recovery, with no prepared transition or reference-state premise. -/
theorem tensorTransition_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (position : Fin tokens.length) :
    tensorTransition hwidth (tensorSequence hsize hwidth eps θ tokens hcap position) =
      (fun previous next => sharedTransitionRead (tokens.get position) previous next θ.1) := by
  funext previous next
  exact tensorPrenormRecovery_transition hsize hwidth eps heps θ.1 (tokens.get position)
    (rawBindingPosition tokens hcap position) previous next

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 21, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- True compact state/channel expectation is computed solely from input tensors and free head-global weights.
Source: MarkovEmissions.markovValueMean; actual chronological matrix propagation, without enumerating state histories. -/
def tensorStateMean (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (h : Fin 5) (code : Fin 4 → ℝ) : ℝ :=
  markovValueMean (tensorInitial ψ) (tensorTransition hwidth) (List.ofFn x) (tensorEmission ψ) h code

/-- Actual ten-coordinate state-head output uses the same inferred small-channel means as the compact decoder.
Source: OutputCodes' five signed-axis pairs and the true tensorStateMean above. -/
def tensorStateCoordinates (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (axis : Fin 10) : ℝ :=
  if axis.val % 2 = 0 then tensorStateMean hwidth ψ x ⟨axis.val / 2, by omega⟩ quarterX
  else tensorStateMean hwidth ψ x ⟨axis.val / 2, by omega⟩ quarterY

/-- A candidate vocabulary token is scored against the actual inferred tensor state/value means.
Source: MarkovEmissions and OutputMargins, with no desired answer argument in inference. -/
def tensorStateScore (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (target : Fin 1024) : ℝ :=
  ∑ h, (tensorStateMean hwidth ψ x h quarterX * quarterX (outputDigit target h) +
    tensorStateMean hwidth ψ x h quarterY * quarterY (outputDigit target h))

/-- The scalar decoder really is the dot product of all ten actual tensor-head coordinates and the tied code.
Source: the physical even/odd coordinate layout, not an abstract presumed logit identity. -/
theorem tensorStateScore_axes (hwidth : 64 ≤ d) (ψ : TensorHeadParameters C)
    (x : Fin T → EucSpace d) (target : Fin 1024) :
    tensorStateScore hwidth ψ x target =
      ∑ axis, tensorStateCoordinates hwidth ψ x axis * outputCoordinate (outputDigit target) axis := by
  unfold tensorStateScore
  rw [Fin.sum_univ_five]
  simp only [Fin.sum_univ_succ, tensorStateCoordinates, outputCoordinate]
  norm_num
  ring

example : (64 : ℕ) ≤ 128 := by omega

/-- The actual tensor state computation equals the verified raw learned state computation for all shared weights.
Source: chronological foldl over physical tensor rows and exact RMS transition recovery, not a semantic encoder assumption. -/
theorem tensorStateMean_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (h : Fin 5) (code : Fin 4 → ℝ) :
    tensorStateMean hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) h code =
      markovValueMean (fun state => sharedInitialRead state θ.1)
        (fun token previous next => sharedTransitionRead token previous next θ.1) tokens
        (fun state group value => sharedEmissionRead state group value θ.1) h code := by
  have hrun : markovRun (tensorInitial (tensorHeadParameters θ)) (tensorTransition hwidth)
      (List.ofFn (tensorSequence hsize hwidth eps θ tokens hcap)) =
      markovRun (fun state => sharedInitialRead state θ.1)
        (fun token previous next => sharedTransitionRead token previous next θ.1) tokens := by
    unfold markovRun markovPropagate
    rw [List.ofFn_eq_map, List.foldl_map]
    simp_rw [tensorTransition_sequence hsize hwidth eps heps]
    conv_rhs => rw [← List.ofFn_get tokens, List.ofFn_eq_map, List.foldl_map]
    rfl
  simp only [tensorStateMean, markovValueMean, hrun, (tensorHeadParameters_reads θ).2.1]

example : (36 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 128 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 9, 10] : List (Fin 36)).length ≤ 128 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

/-- Every actual tensor state-head candidate score equals its proved shared-parameter counterpart.
Source: the full learned state/value mean identity for both signed decoder axes and every output digit. -/
theorem tensorStateScore_sequence (hsize : V ≤ 1024) (hwidth : 64 ≤ d) (eps : ℝ) (heps : 0 < eps)
    (θ : BindingParameters V C) (tokens : List (Fin V)) (hcap : tokens.length ≤ C) (target : Fin 1024) :
    tensorStateScore hwidth (tensorHeadParameters θ) (tensorSequence hsize hwidth eps θ tokens hcap) target =
      sharedMarkovScore θ.1 tokens target := by
  simp only [tensorStateScore, tensorStateMean_sequence hsize hwidth eps heps,
    sharedMarkovScore, markovOutputScore]

example : (68 : ℕ) ≤ 1024 ∧ (64 : ℕ) ≤ 64 ∧ (0 : ℝ) < 1 / 100000 ∧
    ([1, 22, 18] : List (Fin 68)).length ≤ 19 := by
  exact ⟨by omega, by omega, by norm_num, by decide⟩

end
end Transformer.GPTMini.Convex.Structured
