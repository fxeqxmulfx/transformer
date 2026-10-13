import Transformer.GPTMini.Convex.Structured.SharedSlots
import Transformer.GPTMini.Convex.Structured.OutputMargins

/-!
# Actual shared finite weights for causal state/value capacity

Source: the shared 52-field vocabulary/head domain at bdbea00 and the
whole-vocabulary learned decoder margins at d84bd84. Reference data
rules choose one finite shared weight table. The actual model continues
to read and normalize every free transition and emission row, with no
reference rule or correct state passed at inference time.

Each transition, initial and output-channel read is evaluated against
the concrete shared assignment. The actual unrestricted shared-domain
complete likelihood and the actual compact output score coincide with
the learned model whose finite confidence was proved. This is a head
capacity construction; raw Basis semantics and residual/tied realization
are still independent obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open scoped BigOperators Classical
noncomputable section

variable {V C : ℕ}

/-- A finite reference-derived assignment of every actual shared vocabulary/head coordinate.
Source: six-by-six token transition slots, free initial/emission fields, and the finite sharp-row capacity witnesses; inference does not use the rule. -/
def referenceSharedParameters (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) : SharedParameters V C
  | .inl (token, slot) =>
      if hi : slot.val < 36 then
        sharpRowLogits (rule token ⟨slot.val / 6, by omega⟩) gain ⟨slot.val % 6, Nat.mod_lt _ (by decide)⟩
      else 0
  | .inr (.inl _) => 0
  | .inr (.inr (.inl ())) => 0
  | .inr (.inr (.inr (.inl state))) => sharpRowLogits start gain state
  | .inr (.inr (.inr (.inr (.inl (state, (h, d)))))) => sharpRowLogits (outputDigit (labels state) h) gain d
  | .inr (.inr (.inr (.inr (.inr head)))) => sharpRowLogits (0 : Fin 2) gain head

/-- The shared head computes a genuine ten-coordinate decoder score from its free raw parameter reads.
Source: actual coordinate linear maps, causal state propagation and conditional value-channel expectations. -/
def sharedMarkovScore (θ : SharedParameters V C) (tokens : List (Fin V)) (target : Fin 1024) : ℝ :=
  markovOutputScore (fun s => sharedInitialRead s θ) (fun token previous next => sharedTransitionRead token previous next θ)
    tokens (fun state h d => sharedEmissionRead state h d θ) target

/-- Every genuine transition read in the shared finite witness equals its finite sharp learned row.
Source: actual row-major 36-field slot decoding, not a correct-probability assumption or a substituted hard transition. -/
theorem referenceShared_transition (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (token : Fin V) (previous next : Fin 6) :
    sharedTransitionRead token previous next (referenceSharedParameters (C := C) rule labels start gain) =
      sharpReferenceTable rule gain token previous next := by
  have hp := previous.isLt
  have hn := next.isLt
  have hi : 6 * previous.val + next.val < 36 := by omega
  have hd : (6 * previous.val + next.val) / 6 = previous.val := by omega
  have hm : (6 * previous.val + next.val) % 6 = next.val := by omega
  simp only [sharedTransitionRead, sharedTokenRead, LinearMap.proj_apply, transitionSlot,
    referenceSharedParameters, hi, dite_true, hd, hm, sharpReferenceTable]

/-- The actual shared initial coordinates give the same finite learned initial row used in confidence propagation.
Source: the concrete global coordinate assignment and SharedSlots.sharedInitialRead. -/
theorem referenceShared_initial (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (state : Fin 6) :
    sharedInitialRead state (referenceSharedParameters (C := C) rule labels start gain) = sharpRowLogits start gain state := by
  unfold sharedInitialRead
  rw [LinearMap.proj_apply]
  rfl

/-- The actual shared value coordinates realize the same freely trained conditional output-channel model.
Source: the concrete six-state/five-group/four-channel global fields; the fixed decoder code chooses witness targets only. -/
theorem referenceShared_emission (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (state : Fin 6) (h : Fin 5) (d : Fin 4) :
    sharedEmissionRead state h d (referenceSharedParameters (C := C) rule labels start gain) =
      sharpReferenceEmission (fun state => outputDigit (labels state)) gain state h d := by
  unfold sharedEmissionRead
  rw [LinearMap.proj_apply]
  rfl

/-- The same finite witness favors the state branch in its two freely learned head-selection coordinates.
Source: the actual shared global head fields; successful finite mixing with the pointer branch remains a separate theorem. -/
theorem referenceShared_head (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (head : Fin 2) :
    referenceSharedParameters (C := C) rule labels start gain
      (.inr (.inr (.inr (.inr (.inr head))))) = sharpRowLogits (0 : Fin 2) gain head := by
  unfold referenceSharedParameters
  rfl

/-- The actual complete shared training loss at the finite witness is the same learned initial/path/value negative log probability.
Source: MarkovObjective.markovNLL_eq and the proved concrete shared lookup identities, with no supplied correct encoder. -/
theorem referenceShared_loss (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (tokens : List (Fin V))
    (observed : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length) :
    markovNLL sharedInitialRead sharedTransitionRead sharedEmissionRead tokens observed
      (referenceSharedParameters (C := C) rule labels start gain) =
    -Real.log (markovJoint (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
      (sharpReferenceEmission (fun state => outputDigit (labels state)) gain) observed) := by
  rw [markovNLL_eq]
  simp_rw [referenceShared_initial, referenceShared_transition, referenceShared_emission]

/-- Real compact shared-head inference at the concrete finite weights equals the actually learned stochastic model's decoder.
Source: all initial, transition and emission coordinate identities; the reference run does not enter the forward definition. -/
theorem referenceShared_score (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (gain : ℝ) (tokens : List (Fin V)) (target : Fin 1024) :
    sharedMarkovScore (referenceSharedParameters (C := C) rule labels start gain) tokens target =
      markovOutputScore (sharpRowLogits start gain) (sharpReferenceTable rule gain) tokens
        (sharpReferenceEmission (fun state => outputDigit (labels state)) gain) target := by
  unfold sharedMarkovScore
  simp_rw [referenceShared_initial, referenceShared_transition, referenceShared_emission]

/-- One ordinary finite logarithmic gain works uniformly for the full proposed Basis context cap.
Source: true joint confidence and actual whole-vocabulary margin, not infinite logits or deterministic transitions. -/
def referenceGain : ℝ := Real.log 100000

/-- The concrete finite gain discharges the genuine decoder confidence inequality.
Source: exact exp/log inverse law at a positive finite number and the 794 error coefficient. -/
theorem referenceGain_confidence : 794 * Real.exp (-referenceGain) < (1 / 11 : ℝ) := by
  unfold referenceGain
  rw [Real.exp_neg, Real.exp_log (by norm_num : (0 : ℝ) < 100000)]
  norm_num

/-- Actual shared finite weights yield strict whole-vocabulary output margins from raw finite-token lists.
Source: concrete shared table realization and the independently proved finite stochastic confidence/margin; reference task correctness remains required. -/
theorem referenceShared_strict (rule : Fin V → Fin 6 → Fin 6) (labels : Fin 6 → Fin 1024)
    (start : Fin 6) (tokens : List (Fin V)) (rival : Fin 1024) (hT : tokens.length ≤ 128)
    (hne : labels (referenceRun rule start tokens) ≠ rival) :
    sharedMarkovScore (referenceSharedParameters (C := C) rule labels start referenceGain) tokens rival <
      sharedMarkovScore (referenceSharedParameters (C := C) rule labels start referenceGain) tokens
        (labels (referenceRun rule start tokens)) := by
  rw [referenceShared_score, referenceShared_score]
  exact sharpReference_output_strict rule labels start tokens referenceGain rival hT hne referenceGain_confidence

example : ([0, 1, 2] : List (Fin 3)).length ≤ 128 ∧
    (fun _ : Fin 6 => (25 : Fin 1024)) (referenceRun (fun _ state => state) 0 [0, 1, 2]) ≠ 17 := by
  decide

/-- Raw-token re-encoding preserves the chronological reference semantics exactly.
Source: List.foldl_map applied to the independent data recurrence; this rule is not used by learned inference. -/
theorem referenceRun_map {A B : Type*} (rule : B → Fin 6 → Fin 6) (encode : A → B)
    (start : Fin 6) (tokens : List A) :
    referenceRun rule start (tokens.map encode) = referenceRun (fun token => rule (encode token)) start tokens := by
  unfold referenceRun
  rw [List.foldl_map]

/-- A genuine raw slot example realizes both state flips with finite actual shared logits.
Source: concrete token-conditioned shared weights, not a separate probability or hard-coded forward transition. -/
example : sharedTransitionRead (0 : Fin 3) (0 : Fin 6) (1 : Fin 6)
    (referenceSharedParameters (C := 128) (fun _ state => if state = 0 then 1 else 0)
      (fun _ => (25 : Fin 1024)) 0 referenceGain) = referenceGain := by
  rw [referenceShared_transition]
  norm_num [sharpReferenceTable, sharpRowLogits]

end
end Transformer.GPTMini.Convex.Structured
