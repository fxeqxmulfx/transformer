import Transformer.Basis.Basic
import Transformer.CRASP.Parity

/-!
# Basis parity as two integer-list continuations

Source: arXiv:2310.16028v1, §5.2 and Appendix B, and Parity.solve in
synthetic/bits.py at cbafbe9. Basis uses no scratchpad or index hints and
1 to 16 bits. The first supervised prefix is BOS, bits, SEP. One call
appends EVEN or ODD; a second call appends EOS, preserving the first answer.

The decoder checks actual integer IDs rather than counting arbitrary
tokens. Invalid inputs, including an incorrect supplied answer, yield
PAD. This total-function convention replaces Python's input exceptions;
claims about task correctness use well-formed supervised prefixes.
-/

namespace Transformer.Basis

/-- Serialize raw bits with the actual vocabulary IDs.
Source: Bits.prompt in synthetic/bits.py at cbafbe9, without index hints. -/
def bitTokens (bits : List Bool) : Tokens :=
  bits.map fun b => if b then oneBit else zeroBit

/-- Parse bits up to SEP, retaining the complete continuation after SEP.
Source: Bits.inputs and Answered.body; all non-bit body tokens are rejected. -/
def parityBody : Tokens → Option (List Bool × Tokens)
  | [] => none
  | token :: rest =>
      if token = sep then some ([], rest)
      else if token = zeroBit then (parityBody rest).map fun r => (false :: r.1, r.2)
      else if token = oneBit then (parityBody rest).map fun r => (true :: r.1, r.2)
      else none

/-- A raw no-scratchpad prompt, including BOS and SEP.
Source: Bits.prompt at cbafbe9; the mathematical word contains neither delimiter. -/
def parityPrompt (bits : List Bool) : Tokens := bos :: (bitTokens bits ++ [sep])

/-- The benchmark parity label, obtained by counting ones modulo two.
Source: Parity.solve's repeated XOR, equivalent to membership in CRASP.parity. -/
def parityLabel (bits : List Bool) : ℤ :=
  if bits.count true % 2 = 1 then oddToken else evenToken

/-- The next supervised integer answer, followed by EOS on the correct continuation.
Source: Parity.solve and the new total raw-token adapter; no scratchpad is inserted. -/
def parityNext : Tokens → ℤ
  | [] => pad
  | head :: body =>
      if head = bos then
        match parityBody body with
        | some (bits, []) => if bits = [] then pad else parityLabel bits
        | some (bits, [answer]) =>
            if bits = [] then pad else if answer = parityLabel bits then eos else pad
        | _ => pad
      else pad

/-- Parity with the same append-one-token signature as depth, recall and GPTMini.
Source: the new Basis continuation convention; the full answer needs two calls. -/
def parityFunction : Tokens → Tokens := extend parityNext

/-- Basis supervises the prompt and its correct one-token answer continuation.
Source: domain/basis.py's 1..16-bit task, bounded by the model's raw-token context. -/
def ParityPrefix (bound : ℕ) (tokens : Tokens) : Prop :=
  ∃ bits : List Bool, 1 ≤ bits.length ∧ bits.length ≤ 16 ∧ tokens.length ≤ bound ∧
    (tokens = parityPrompt bits ∨ tokens = parityPrompt bits ++ [parityLabel bits])

/-- Raw parsing preserves every bit and the entire post-SEP continuation.
Source: the new checked serialization adapter, for arbitrary bit words and suffixes. -/
theorem parityBody_tokens (bits : List Bool) (rest : Tokens) :
    parityBody (bitTokens bits ++ sep :: rest) = some (bits, rest) := by
  induction bits with
  | nil => simp [bitTokens, parityBody]
  | cons b bits ih =>
      simp only [bitTokens, oneBit, zeroBit, sep] at ih
      cases b <;> simp [bitTokens, parityBody, oneBit, zeroBit, sep, ih]

/-- The integer label is ODD exactly for the independently defined parity language.
Source: CRASP.parity and arXiv:2310.16028v1 §5.2; Odd is equivalent to remainder one. -/
theorem parityLabel_odd_iff (bits : List Bool) :
    parityLabel bits = oddToken ↔ bits ∈ Transformer.CRASP.parity := by
  change (if bits.count true % 2 = 1 then oddToken else evenToken) = oddToken ↔
    Odd (bits.count true)
  rw [Nat.odd_iff]
  split_ifs with h
  · exact iff_of_true rfl h
  · exact iff_of_false (by norm_num [evenToken, oddToken]) h

/-- The first call appends exactly the parity label of the actual bit word.
Source: Parity.solve's no-scratchpad answer, with the raw BOS/SEP layout. -/
theorem parityFunction_prompt (bits : List Bool) (hbits : bits ≠ []) :
    parityFunction (parityPrompt bits) = parityPrompt bits ++ [parityLabel bits] := by
  simp only [parityFunction, extend, parityNext, parityPrompt,
    parityBody_tokens bits [], ite_eq_right hbits, ↓reduceIte]

example : ([true, false] : List Bool) ≠ [] := by decide

/-- The second call emits EOS after the correct answer, rather than another label.
Source: Parity.solve returns (EVEN/ODD, EOS); the input answer is retained. -/
theorem parityFunction_answer (bits : List Bool) (hbits : bits ≠ []) :
    parityFunction (parityPrompt bits ++ [parityLabel bits]) =
      parityPrompt bits ++ [parityLabel bits, eos] := by
  simp only [parityFunction, extend, parityPrompt, List.cons_append, List.append_assoc]
  simp only [List.nil_append, parityNext, parityBody_tokens, ite_eq_right hbits, ↓reduceIte]

example : ([true, false, true] : List Bool) ≠ [] := by decide

/-- Two calls generate the complete no-scratchpad answer and EOS for every bit word.
Source: the exact autoregressive interpretation of the Basis parity answer. -/
theorem parityFunction_twice (bits : List Bool) (hbits : bits ≠ []) :
    parityFunction (parityFunction (parityPrompt bits)) =
      parityPrompt bits ++ [parityLabel bits, eos] := by
  rw [parityFunction_prompt bits hbits, parityFunction_answer bits hbits]

example : ([true] : List Bool) ≠ [] := by decide

/-- The requested length contract also holds for invalid inputs.
Source: the common answer-append interface, not a property of the parity language. -/
theorem parityFunction_length (tokens : Tokens) :
    (parityFunction tokens).length = tokens.length + 1 := extend_length _ _

/-- Delimiters consume two context positions in addition to the actual bits.
Source: Bits.prompt prepends BOS and appends SEP; Basis's 16-bit limit is not 16 raw tokens. -/
theorem parityPrompt_length (bits : List Bool) :
    (parityPrompt bits).length = bits.length + 2 := by
  simp [parityPrompt, bitTokens]

/-- Odd parity and EOS are two actual integer-token predictions.
Source: a three-bit no-scratchpad Basis example. -/
theorem parity_odd_example :
    parityFunction [1, 22, 21, 21, 18] = [1, 22, 21, 21, 18, 25] ∧
    parityFunction [1, 22, 21, 21, 18, 25] = [1, 22, 21, 21, 18, 25, 17] := by decide

/-- An even number of ones receives EVEN, not ODD.
Source: the same Basis oracle on a different raw bit prefix. -/
theorem parity_even_example :
    parityFunction [1, 22, 21, 22, 18] = [1, 22, 21, 22, 18, 24] := by decide

/-- An arbitrary non-bit cannot silently be treated as zero.
Source: Bits.inputs validates its complete bit alphabet. -/
theorem parity_invalid_example : parityNext [1, 9, 18] = pad := by decide

/-- Zero bits are invalid, even though the mathematical empty word has even parity.
Source: Answered.body rejects empty problems; the total adapter returns PAD instead. -/
theorem parity_empty_problem : parityNext [1, 18] = pad := by decide

/-- A wrong supplied answer is outside the adapter's correct-continuation grammar.
Source: the explicit PAD convention; this case is not a Python teacher-forced prefix. -/
theorem parity_wrong_answer : parityNext [1, 22, 18, 24] = pad := by decide

/-- The prompt part of the supervised 1..16-bit grammar is inhabited.
Source: Basis parity's exact minimum and maximum bit lengths. -/
theorem parityPrefix_prompt_example : ParityPrefix 20 [1, 22, 21, 21, 18] := by
  refine ⟨[true, false, false], by decide, by decide, by decide, Or.inl ?_⟩
  rfl

/-- The second supervised position is included, so proving only the first label is insufficient.
Source: Answered.example/targets supervises both the parity answer and EOS. -/
theorem parityPrefix_answer_example : ParityPrefix 20 [1, 22, 21, 21, 18, 25] := by
  refine ⟨[true, false, false], by decide, by decide, by decide, Or.inr ?_⟩
  rfl

end Transformer.Basis
