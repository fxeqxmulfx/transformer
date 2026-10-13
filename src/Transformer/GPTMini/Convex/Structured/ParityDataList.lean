import Transformer.GPTMini.Convex.Structured.LocalConfidence

/-!
# Actual finite parity token lists and chronological path composition

Source: Basis.Parity's raw no-scratchpad grammar and ParityReference's
independent counted data semantics at 842cc59/91299ae. These lists keep
every physical BOS/bit/SEP/answer token and decode to exactly the original
integer inputs. They are data/proof helpers, never learned inference.

The true conditional learned path on a concatenated raw word factors
at the actual reference boundary state. This holds for unrestricted
transition logits; the reference rule only selects which data path's
actual probability is being bounded. Subsequent finite-row certificates
must derive every selected factor from learned probabilities, and the
real tensor model must still be coupled to its actual mixed decoder.
In particular, the finite lists introduce no new tokens and no teacher
input to the model. They are an exact representation of existing data
for proofs about the actual learned probability of that data path.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Raw bit IDs in the true 68-token parity vocabulary, for data certificates only.
Source: Basis.bitTokens's ZERO=21 and ONE=22, without an inferred phase or count input. -/
def parityFiniteBits (bits : List Bool) : List (Fin 68) := bits.map (fun bit => if bit then 22 else 21)

/-- The exact original BOS/bit/SEP input, checked as finite raw IDs.
Source: Basis.parityPrompt's physical serialization, with no scratchpad or hidden data state. -/
def parityFinitePrompt (bits : List Bool) : List (Fin 68) := 1 :: (parityFiniteBits bits ++ [18])

/-- The independent raw-data label, represented as its actual finite vocabulary ID.
Source: Basis.parityLabel's counted data answer; this is absent from learned inference. -/
def parityFiniteLabel (bits : List Bool) : Fin 68 := if bits.count true % 2 = 1 then 25 else 24

/-- The actual second supervised input reuses the correct data answer after the unchanged prompt.
Source: Basis.Parity's label-before-EOS grammar, not a new inference interface. -/
def parityFiniteCompletion (bits : List Bool) : List (Fin 68) := parityFinitePrompt bits ++ [parityFiniteLabel bits]

/-- Finite bit serialization retains precisely every original integer bit token.
Source: true per-bit vocabulary bounds and Basis.bitTokens, proved by physical list induction. -/
theorem parityFiniteBits_decode (bits : List Bool) : decodeTokens (parityFiniteBits bits) = bitTokens bits := by
  induction bits with
  | nil => rfl
  | cons bit bits ih =>
      cases bit with
      | false =>
          change zeroBit :: decodeTokens (parityFiniteBits bits) = zeroBit :: bitTokens bits
          rw [ih]
      | true =>
          change oneBit :: decodeTokens (parityFiniteBits bits) = oneBit :: bitTokens bits
          rw [ih]

/-- The whole finite prompt is exactly the original raw integer prompt after checked decoding.
Source: actual physical BOS/SEP serialization and the independent bit-list identity, not an assumed encoder feature. -/
theorem parityFinitePrompt_decode (bits : List Bool) : decodeTokens (parityFinitePrompt bits) = parityPrompt bits := by
  unfold parityFinitePrompt parityPrompt
  unfold decodeTokens
  rw [List.map_cons, List.map_append]
  change bos :: (decodeTokens (parityFiniteBits bits) ++ [sep]) = _
  rw [parityFiniteBits_decode]

/-- Finite data answer IDs are the original counted integer answer at every bit word.
Source: actual EVEN/ODD IDs and Basis.parityLabel's independent modulo-two criterion. -/
theorem parityFiniteLabel_decode (bits : List Bool) : ((parityFiniteLabel bits).val : ℤ) = parityLabel bits := by
  by_cases h : bits.count true % 2 = 1
  · norm_num [parityFiniteLabel, parityLabel, h, oddToken]
  · norm_num [parityFiniteLabel, parityLabel, h, evenToken]

/-- The finite label-before-EOS input decodes to the unchanged correct raw second input.
Source: exact finite prompt/answer identities and original concatenation order. -/
theorem parityFiniteCompletion_decode (bits : List Bool) :
    decodeTokens (parityFiniteCompletion bits) = parityPrompt bits ++ [parityLabel bits] := by
  unfold parityFiniteCompletion decodeTokens
  rw [List.map_append]
  change decodeTokens (parityFinitePrompt bits) ++ [((parityFiniteLabel bits).val : ℤ)] = _
  rw [parityFinitePrompt_decode, parityFiniteLabel_decode]

/-- Every finite raw bit makes the same independent counted reference transition as the integer grammar.
Source: ParityReference's proved ZERO/ONE count laws, with the true 68-token IDs retained. -/
theorem parityFiniteBits_step (count : ℕ) (bit : Bool) :
    paritySharedRule (if bit then 22 else 21) (parityReferencePhase count) =
      parityReferencePhase (count + if bit then 1 else 0) := by
  cases bit with
  | false =>
      change parityReferenceRule zeroBit (parityReferencePhase count) = parityReferencePhase (count + 0)
      simpa only [Nat.add_zero] using parityReference_zero count
  | true =>
      change parityReferenceRule oneBit (parityReferencePhase count) = parityReferencePhase (count + 1)
      exact parityReference_one count

/-- Actual finite bit lists give the independently counted reference endpoint from either bit-phase start.
Source: true per-bit transitions and chronological induction; this does not replace the learned stochastic state recurrence. -/
theorem parityFiniteBits_run (count : ℕ) (bits : List Bool) :
    referenceRun paritySharedRule (parityReferencePhase count) (parityFiniteBits bits) =
      parityReferencePhase (count + bits.count true) := by
  induction bits generalizing count with
  | nil => rfl
  | cons bit bits ih =>
      change referenceRun paritySharedRule
        (paritySharedRule (if bit then 22 else 21) (parityReferencePhase count)) (parityFiniteBits bits) = _
      rw [parityFiniteBits_step, ih, List.count_cons]
      cases bit with
      | false =>
          change parityReferencePhase (count + 0 + bits.count true) = parityReferencePhase (count + (bits.count true + 0))
          congr 1
      | true =>
          change parityReferencePhase (count + 1 + bits.count true) = parityReferencePhase (count + (bits.count true + 1))
          congr 1
          omega

/-- The finite prompt reaches its actual raw data label phase, including the true SEP transition.
Source: exact finite/raw serialization and ParityReference.parityReference_prompt. -/
theorem parityFinitePrompt_run (bits : List Bool) :
    referenceRun paritySharedRule 0 (parityFinitePrompt bits) = if bits.count true % 2 = 1 then 3 else 2 := by
  rw [parityShared_run, parityFinitePrompt_decode]
  exact parityReference_prompt bits

/-- The finite label-before-EOS input reaches completion only through its real correct label token.
Source: exact second-input serialization and the independent counted label/completion transition. -/
theorem parityFiniteCompletion_run (bits : List Bool) :
    referenceRun paritySharedRule 0 (parityFiniteCompletion bits) = 4 := by
  rw [parityShared_run, parityFiniteCompletion_decode]
  exact parityReference_answer bits

/-- Actual selected data-path probabilities compose through their true boundary state for arbitrary learned transition logits.
Source: conditionalStatePath's original positive chronological factors and referencePath's physical post-token histories. -/
theorem referencePath_probability_append {A S : Type*} [Fintype S] (transition : A → S → S → ℝ)
    (rule : A → S → S) (start : S) (left right : List A) :
    conditionalStatePath transition start (left ++ right) (referencePath rule start (left ++ right)) =
      conditionalStatePath transition start left (referencePath rule start left) *
        conditionalStatePath transition (referenceRun rule start left) right
          (referencePath rule (referenceRun rule start left) right) := by
  induction left generalizing start with
  | nil =>
      change conditionalStatePath transition start right (referencePath rule start right) =
        1 * conditionalStatePath transition start right (referencePath rule start right)
      rw [one_mul]
  | cons token left ih =>
      change stateRow (transition token start) (rule token start) *
        conditionalStatePath transition (rule token start) (left ++ right) (referencePath rule (rule token start) (left ++ right)) =
          (stateRow (transition token start) (rule token start) * conditionalStatePath transition (rule token start) left
            (referencePath rule (rule token start) left)) * conditionalStatePath transition
              (referenceRun rule (rule token start) left) right (referencePath rule (referenceRun rule (rule token start) left) right)
      rw [ih]
      ring

/-- Both actual finite supervised input forms have their original raw IDs and independently computed data endpoints.
Source: explicit one-ONE data control, including the genuine ODD input before EOS. -/
example : decodeTokens (parityFinitePrompt [true]) = [1, 22, 18] ∧
    decodeTokens (parityFiniteCompletion [true]) = [1, 22, 18, 25] ∧
    referenceRun paritySharedRule 0 (parityFinitePrompt [true]) = 3 ∧
    referenceRun paritySharedRule 0 (parityFiniteCompletion [true]) = 4 := by decide

end
end Transformer.GPTMini.Convex.Structured
