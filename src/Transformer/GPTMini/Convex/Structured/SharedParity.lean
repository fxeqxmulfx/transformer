import Transformer.GPTMini.Convex.Structured.SharedInterface
import Transformer.GPTMini.Convex.Structured.ParityReference
import Transformer.Basis.Encoding.Tasks

/-!
# Actual finite learned causal head solves raw Basis parity

Source: Basis's no-scratchpad bit grammar at cbafbe9, the independent
reference semantics at 91299ae and actual finite shared stochastic
weights at d72d3b3. One common finite raw vocabulary/global parameter
assignment is used for every prompt and answer-before-EOS prefix.
Inference reads learned rows, propagates all states and values and
greedily decodes their actual means. It does not call parity semantics.

The standalone shared head's real List Int -> List Int callback solves
both complete parity grammars and freely generates the label then EOS.
Raw data-derived state/channel targets give a globally convex likelihood
on the entire free shared parameter space. This does not yet realize
the full prenorm/residual/tied block or prove AdamW/finite-precision
training convergence; complete depth and recall remain open.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.GPTMini.TokenInterface
open scoped Classical
noncomputable section

/-- Actual parity output IDs as genuine tokens of its full 68-token vocabulary.
Source: Basis.Parity's EVEN/ODD/EOS/PAD IDs; these select witness values and training channels only. -/
def paritySharedLabel (state : Fin 6) : Fin 68 :=
  if state = 2 then 24 else if state = 3 then 25 else if state = 4 then 17 else 0

/-- Finite witness output labels preserve the independent raw data semantics exactly.
Source: actual parity vocabulary IDs, including EOS after the correct supplied label. -/
theorem paritySharedLabel_id (state : Fin 6) : (paritySharedLabel state).val = parityReferenceOutput state := by
  fin_cases state <;> norm_num [paritySharedLabel, parityReferenceOutput, evenToken, oddToken, eos, pad]

/-- Reference data/capacity transitions on genuine finite raw tokens, without a provided count or phase.
Source: the independently verified raw integer six-state rule; it is absent from actual learned inference. -/
def paritySharedRule (token : Fin 68) : Fin 6 → Fin 6 := parityReferenceRule (token.val : ℤ)

/-- Checked finite/raw token serialization gives exactly the same independent chronological data run.
Source: referenceRun_map and Basis.decodeTokens, with every physical bit, delimiter and answer retained. -/
theorem parityShared_run (tokens : List (Fin 68)) :
    referenceRun paritySharedRule 0 tokens = referenceRun parityReferenceRule 0 (decodeTokens tokens) := by
  unfold decodeTokens
  rw [referenceRun_map]
  rfl

/-- One actual finite shared weight table covers every valid parity prefix in both modes.
Source: concrete learned transition/initial/value assignments and gain log(100000), independent of the specific input bits. -/
def paritySharedParameters : SharedParameters 68 19 :=
  referenceSharedParameters paritySharedRule (fun state => vocabularyCode (by decide) (paritySharedLabel state)) 0 referenceGain

/-- Real parity prediction calls only the freely learned stochastic head and original checked integer decoder.
Source: sharedMarkovFunction on the actual 68-token/19-position domain; no data rule or task interpreter enters its forward call. -/
def paritySharedFunction : Tokens → Tokens := sharedMarkovFunction (by decide) (by decide) paritySharedParameters

/-- The actual finite learned head predicts the independently required next token at every raw parity supervision position.
Source: independently proved raw reference parity, genuine shared finite-weight margins, complete encoding/domain checks and actual greedy decoding. -/
theorem parityShared_next (tokens : Tokens) (hprefix : ParityPrefix 19 tokens) :
    sharedMarkovNext (by decide) (by decide) paritySharedParameters tokens = parityNext tokens := by
  obtain ⟨head, tail, he, hcap⟩ := Encoding.parity_inputs .easy tokens hprefix
  have h128 : (head :: tail).length ≤ 128 := by
    simp only [List.length_cons]
    omega
  rw [sharedMarkovNext_of_encode (by decide) (by decide) paritySharedParameters tokens head tail he hcap]
  unfold paritySharedParameters
  rw [referenceShared_best (by decide) (by decide) paritySharedRule paritySharedLabel 0 (head :: tail) h128,
    paritySharedLabel_id, parityShared_run, decode_encode he]
  exact parityReference_next 19 tokens hprefix

example : ParityPrefix 19 [1, 22, 18, 25] :=
  ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩

/-- The standalone actual learned-head callback solves the complete raw parity task in either Basis mode.
Source: derived raw next answers above; this is not a full GPTMini-stack theorem and assumes no correct state, logits or encoder. -/
theorem parityShared_solves (mode : Mode) : SolvesTask paritySharedFunction mode .parity := by
  apply (solvesTask_extend_iff _ mode .parity).mpr
  intro tokens hprefix
  exact parityShared_next tokens hprefix

/-- Two real autoregressive calls use the first predicted label as input and then emit EOS for every allowed bit word.
Source: complete genuine standalone head correctness and Basis's two-call parity generation theorem, with one shared weight assignment. -/
theorem parityShared_twice (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    paritySharedFunction (paritySharedFunction (parityPrompt bits)) = parityPrompt bits ++ [parityLabel bits, eos] :=
  solvesTask_parity_twice paritySharedFunction .easy (parityShared_solves .easy) bits hmin hmax

example : 1 ≤ ([true, false, true] : List Bool).length ∧ ([true, false, true] : List Bool).length ≤ 16 := by decide

/-- State/channel labels for the genuine complete-likelihood loss are computed causally from raw training data.
Source: the independently verified parity rule and physical referencePath; these targets are never arguments of inference. -/
def paritySharedTargets (tokens : List (Fin 68)) : MarkovConfiguration (Fin 6) (Fin 5) (Fin 4) tokens.length :=
  (0, (referencePath paritySharedRule 0 tokens,
    outputDigit (vocabularyCode (by decide) (paritySharedLabel (referenceRun paritySharedRule 0 tokens)))))

/-- The exact training loss on raw data-derived complete parity labels is globally convex in all free shared weights.
Source: sharedMarkovNLL_convex, using the actual free transition/initial/value coordinate maps and no frozen encoder parameters. -/
theorem parityShared_training_convex (tokens : List (Fin 68)) :
    ConvexOn ℝ Set.univ (markovNLL (sharedInitialRead (V := 68) (C := 19))
      sharedTransitionRead sharedEmissionRead tokens (paritySharedTargets tokens)) :=
  sharedMarkovNLL_convex tokens (paritySharedTargets tokens)

/-- The output-channel part of actual data supervision agrees with the independently checked raw Basis answer.
Source: exact finite/raw reference equality and the complete parity reference-next theorem, not an encoder-correctness premise. -/
theorem paritySharedTargets_answer (tokens : List (Fin 68)) (hprefix : ParityPrefix 19 (decodeTokens tokens)) :
    (paritySharedTargets tokens).2.2 = outputDigit
      ⟨(parityNext (decodeTokens tokens)).toNat, by
        have h := paritySharedLabel_id (referenceRun paritySharedRule 0 tokens)
        rw [parityShared_run, parityReference_next 19 _ hprefix] at h
        omega⟩ := by
  have h := paritySharedLabel_id (referenceRun paritySharedRule 0 tokens)
  rw [parityShared_run, parityReference_next 19 _ hprefix] at h
  dsimp only [paritySharedTargets]
  apply congrArg outputDigit
  apply Fin.ext
  change (paritySharedLabel (referenceRun paritySharedRule 0 tokens)).val = (parityNext (decodeTokens tokens)).toNat
  rw [parityShared_run]
  omega

example : ParityPrefix 19 (decodeTokens [1, 22, 18, (25 : Fin 68)]) :=
  ⟨[true], by decide, by decide, by decide, Or.inr rfl⟩

/-- The full raw odd-parity and supplied-answer control is computed by the same genuine learned stochastic head.
Source: universal next-token correctness on two literal valid prefixes and the unchanged append-one public contract. -/
theorem parityShared_odd_control :
    paritySharedFunction [1, 22, 21, 21, 18] = [1, 22, 21, 21, 18, 25] ∧
    paritySharedFunction [1, 22, 21, 21, 18, 25] = [1, 22, 21, 21, 18, 25, 17] := by
  have hp : ParityPrefix 19 [1, 22, 21, 21, 18] :=
    ⟨[true, false, false], by decide, by decide, by decide, Or.inl rfl⟩
  have ha : ParityPrefix 19 [1, 22, 21, 21, 18, 25] :=
    ⟨[true, false, false], by decide, by decide, by decide, Or.inr rfl⟩
  constructor
  · have h := parityShared_next _ hp
    norm_num [parityNext, parityBody, parityLabel, bos, zeroBit, oneBit, sep, oddToken] at h
    exact congrArg (fun token : ℤ => [1, 22, 21, 21, 18] ++ [token]) h
  · have h := parityShared_next _ ha
    norm_num [parityNext, parityBody, parityLabel, bos, zeroBit, oneBit, sep, oddToken, eos] at h
    exact congrArg (fun token : ℤ => [1, 22, 21, 21, 18, 25] ++ [token]) h

/-- The actual standalone head preserves length even for raw inputs outside the parity grammar.
Source: the same checked append-one inference interface, with no task-validity assumption. -/
theorem parityShared_length (tokens : Tokens) : (paritySharedFunction tokens).length = tokens.length + 1 :=
  sharedMarkovFunction_length _ _ _ _

/-- All original integer tokens remain in order for every real standalone-head call.
Source: the actual checked append-one interface, including invalid and overlong raw inputs. -/
theorem parityShared_prefix (tokens : Tokens) : (paritySharedFunction tokens).take tokens.length = tokens :=
  sharedMarkovFunction_prefix _ _ _ _

end
end Transformer.GPTMini.Convex.Structured
