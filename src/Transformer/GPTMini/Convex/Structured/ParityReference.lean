import Transformer.GPTMini.Convex.Structured.MarkovTeacher
import Transformer.Basis.Parity

/-!
# Six-state parity data semantics on actual integer tokens

Source: Basis.Parity's raw no-scratchpad grammar at cbafbe9 and
arXiv:2310.16028v1, §5.2 and Appendix B. States 0/1 represent even/odd
bit prefixes, 2/3 the post-SEP label position, 4 completion, and 5 other
transitions. This new reference rule generates data state targets and
chooses finite capacity witness logits only. Learned inference still
reads all free transition rows; it never calls this reference rule.

The independent chronological reference computation is proved equal
to the true bit count modulo two and to Basis's actual next label/EOS
on every validated prefix. The learned stochastic head's confidence,
shared raw parameter assignment, output-channel decoding and actual
residual/integer model coupling remain separate proof obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical

/-- Reference data transitions use actual token IDs and six finite states; they are not inference code.
Source: Bits.prompt/Parity.solve at cbafbe9; BOS resets the reference, bit/SEP/answer phases are explicit. -/
def parityReferenceRule (token : ℤ) (state : Fin 6) : Fin 6 :=
  if token = bos then 0
  else if token = zeroBit then if state = 0 then 0 else if state = 1 then 1 else 5
  else if token = oneBit then if state = 0 then 1 else if state = 1 then 0 else 5
  else if token = sep then if state = 0 then 2 else if state = 1 then 3 else 5
  else if token = evenToken then if state = 2 then 4 else 5
  else if token = oddToken then if state = 3 then 4 else 5
  else 5

/-- Bit-phase data states encode count parity, independently of the learned model's hidden state probabilities.
Source: Parity.solve's count/XOR semantics; only this finite state target is used in supervision. -/
def parityReferencePhase (count : ℕ) : Fin 6 := if count % 2 = 1 then 1 else 0

/-- Actual raw BOS begins the reference bit phase from every prior data state.
Source: the explicit reference reset transition, outside learned inference. -/
theorem parityReference_bos (state : Fin 6) : parityReferenceRule bos state = 0 := by
  simp only [parityReferenceRule, ite_true]

/-- A raw zero leaves count parity unchanged in the reference data states.
Source: the actual ZERO token and finite transition table, rather than a count supplied to learned inference. -/
theorem parityReference_zero (count : ℕ) :
    parityReferenceRule zeroBit (parityReferencePhase count) = parityReferencePhase count := by
  by_cases h : count % 2 = 1 <;>
    norm_num [parityReferenceRule, parityReferencePhase, bos, zeroBit, oneBit, h]

/-- A raw one flips count parity for every integer count, with the modulo arithmetic derived.
Source: the actual ONE token and finite table, matching Parity.solve's XOR step. -/
theorem parityReference_one (count : ℕ) :
    parityReferenceRule oneBit (parityReferencePhase count) = parityReferencePhase (count + 1) := by
  by_cases h : count % 2 = 1
  · have hn : ¬(count + 1) % 2 = 1 := by omega
    norm_num [parityReferenceRule, parityReferencePhase, bos, zeroBit, oneBit, h, hn]
  · have hn : (count + 1) % 2 = 1 := by omega
    norm_num [parityReferenceRule, parityReferencePhase, bos, zeroBit, oneBit, h, hn]

/-- The actual bit token list drives the independent reference recurrence to its counted parity state.
Source: raw Bits.prompt serialization and chronological induction over every bit, including arbitrary starting count. -/
theorem parityReference_bits (count : ℕ) (bits : List Bool) :
    referenceRun parityReferenceRule (parityReferencePhase count) (bitTokens bits) =
      parityReferencePhase (count + bits.count true) := by
  induction bits generalizing count with
  | nil => rfl
  | cons bit bits ih =>
      cases bit with
      | false =>
          rw [List.count_cons]
          change referenceRun parityReferenceRule (parityReferenceRule zeroBit (parityReferencePhase count))
            (bitTokens bits) = parityReferencePhase (count + (bits.count true + 0))
          rw [parityReference_zero, ih]
          congr 1
      | true =>
          rw [List.count_cons]
          change referenceRun parityReferenceRule (parityReferenceRule oneBit (parityReferencePhase count))
            (bitTokens bits) = parityReferencePhase (count + (bits.count true + 1))
          rw [parityReference_one, ih]
          congr 1
          omega

/-- Raw SEP changes counted bit phase into the corresponding reference answer-label phase.
Source: no-scratchpad Bits.prompt delimiter and the actual six-state transition table. -/
theorem parityReference_sep (count : ℕ) :
    parityReferenceRule sep (parityReferencePhase count) = if count % 2 = 1 then 3 else 2 := by
  by_cases h : count % 2 = 1 <;>
    norm_num [parityReferenceRule, parityReferencePhase, bos, zeroBit, oneBit, sep, h]

/-- The entire raw BOS/bit/SEP prompt has the right independent reference answer phase.
Source: actual integer serialization, raw reset, counted bit induction and physical SEP transition. -/
theorem parityReference_prompt (bits : List Bool) :
    referenceRun parityReferenceRule 0 (parityPrompt bits) = if bits.count true % 2 = 1 then 3 else 2 := by
  unfold parityPrompt
  change referenceRun parityReferenceRule (parityReferenceRule bos 0) (bitTokens bits ++ [sep]) = _
  rw [parityReference_bos, referenceRun_append]
  have hphase : (0 : Fin 6) = parityReferencePhase 0 := by decide
  rw [hphase, parityReference_bits]
  simpa only [referenceRun, List.foldl_cons, List.foldl_nil, Nat.zero_add] using parityReference_sep (bits.count true)

/-- The correct actual supplied answer takes the reference to completion on the second supervised prefix.
Source: Parity.solve's EVEN/ODD then EOS output, with label consistency derived from the raw bit count. -/
theorem parityReference_answer (bits : List Bool) :
    referenceRun parityReferenceRule 0 (parityPrompt bits ++ [parityLabel bits]) = 4 := by
  rw [referenceRun_append, parityReference_prompt]
  change parityReferenceRule (parityLabel bits) (if bits.count true % 2 = 1 then 3 else 2) = 4
  by_cases h : bits.count true % 2 = 1 <;>
    norm_num [parityReferenceRule, parityLabel, bos, zeroBit, oneBit, sep, evenToken, oddToken, h]

/-- Reference data states map to actual output IDs; this chooses value-channel targets and witness logits only.
Source: Basis parity vocabulary and completion convention, not a fixed output during learned inference. -/
def parityReferenceOutput (state : Fin 6) : ℤ :=
  if state = 2 then evenToken else if state = 3 then oddToken else if state = 4 then eos else pad

/-- The raw reference prompt computation predicts the independently counted actual parity label.
Source: proved full reference recurrence, without any learned probability or correct-output premise. -/
theorem parityReference_prompt_output (bits : List Bool) :
    parityReferenceOutput (referenceRun parityReferenceRule 0 (parityPrompt bits)) = parityLabel bits := by
  rw [parityReference_prompt]
  by_cases h : bits.count true % 2 = 1 <;>
    norm_num [parityReferenceOutput, parityLabel, h]

/-- The raw reference continuation predicts actual EOS after the counted correct label.
Source: complete answer-phase transition, covering the second call as well as the first. -/
theorem parityReference_answer_output (bits : List Bool) :
    parityReferenceOutput (referenceRun parityReferenceRule 0 (parityPrompt bits ++ [parityLabel bits])) = eos := by
  rw [parityReference_answer]
  norm_num [parityReferenceOutput]

/-- Both supervised raw Basis parity positions have the correct independent reference output.
Source: Basis.Parity's actual valid-prefix grammar and checked next-token semantics; this is data semantics, not a learned-model solver claim. -/
theorem parityReference_next (bound : ℕ) (tokens : Tokens) (hprefix : ParityPrefix bound tokens) :
    parityReferenceOutput (referenceRun parityReferenceRule 0 tokens) = parityNext tokens := by
  obtain ⟨bits, hlength, _, _, hshape⟩ := hprefix
  have hbits : bits ≠ [] := by
    intro hnil
    subst bits
    norm_num at hlength
  rcases hshape with hprompt | hanswer
  · subst tokens
    rw [parityReference_prompt_output]
    simp only [parityNext, parityPrompt, parityBody_tokens, ite_eq_right hbits, ite_true]
  · subst tokens
    rw [parityReference_answer_output]
    simp only [parityPrompt, List.cons_append, List.append_assoc, List.nil_append,
      parityNext, parityBody_tokens, ite_eq_right hbits, ite_true]

example : ParityPrefix 20 [1, 22, 21, 21, 18] := parityPrefix_prompt_example

/-- Actual raw even-parity and supplied-answer prefixes require different finite data phases and output IDs.
Source: the full reference computation on both no-scratchpad supervised positions, without an external phase input. -/
example : parityReferenceOutput (referenceRun parityReferenceRule 0 [1, 22, 21, 22, 18]) = 24 ∧
    parityReferenceOutput (referenceRun parityReferenceRule 0 [1, 22, 21, 22, 18, 24]) = 17 := by decide

end Transformer.GPTMini.Convex.Structured
