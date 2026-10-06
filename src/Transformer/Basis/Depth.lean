import Transformer.Basis.Basic
import Transformer.CRASP.PositionalDepth

/-!
# Basis depth as an integer-list continuation

New adapter for AlternatingBlocks.targets in synthetic/prefixes.py at
cbafbe9, and arXiv:2506.16055v3, §2.4 and Appendix F's language E_k.
The executable test uses the proved alternating-subsequence criterion
instead of the Python oracle's run counter. This keeps exactly the same
language, including neutral letters and rejection of a first active b.
This uses CRASP.altPlus_eq's correction to equation altsingle: the even-a
exponent is k/2 rather than the manuscript's printed k. The E_k block
language itself is unchanged.

The body is decoded from the actual integer IDs; BOS is required. Invalid
tokens yield PAD rather than an exception. The valid-prefix grammar is
broader than the benchmark's sampling distribution, whose neutral fraction,
minimum lengths, and positive/negative pairing are not modeled here.
-/

namespace Transformer.Basis

open scoped Classical

/-- Decode the actual depth alphabet, retaining neutral positions.
Source: vocabulary.py and AlternatingBlocks.targets at cbafbe9. -/
def depthLetters : Tokens → Option (List (Option Bool))
  | [] => some []
  | t :: rest =>
      let decoded := depthLetters rest
      if t = letterA then decoded.map (List.cons (some false))
      else if t = letterB then decoded.map (List.cons (some true))
      else if t = neutral then decoded.map (List.cons none)
      else none

/-- Serialize a mathematical depth word with the benchmark's actual token IDs.
Source: the new inverse vocabulary adapter to CRASP.altPlusNeutral. -/
def depthBody (w : List (Option Bool)) : Tokens :=
  w.map fun t => match t with
    | none => neutral
    | some false => letterA
    | some true => letterB

/-- The actual integer next answer for a depth-k prefix.
Source: AlternatingBlocks.targets; the equivalent subsequence test is §2.4's altPlus_eq. -/
def depthNext (k : ℕ) : Tokens → ℤ
  | [] => pad
  | head :: body =>
      if head = bos then
        match depthLetters body with
        | none => pad
        | some w =>
            if (Transformer.CRASP.altList false k).Sublist w.reduceOption ∧
                ¬(Transformer.CRASP.altList true k).Sublist w.reduceOption then accept else reject
      else pad

/-- Basis depth with the requested List Int to List Int signature.
Source: the new answer-append interface at a supervised prefix. -/
def depthFunction (k : ℕ) : Tokens → Tokens := extend (depthNext k)

/-- Valid depth prefixes have BOS, at least one body symbol, and fit the context.
Source: prefixes.py's input grammar; distribution-specific restrictions are omitted explicitly. -/
def DepthPrefix (bound : ℕ) (tokens : Tokens) : Prop :=
  ∃ w : List (Option Bool), w ≠ [] ∧ w.length + 1 ≤ bound ∧ tokens = bos :: depthBody w

/-- Serialization and raw vocabulary decoding preserve every neutral position.
Source: the new exact integer representation of Appendix F's alphabet. -/
theorem depthLetters_body (w : List (Option Bool)) : depthLetters (depthBody w) = some w := by
  induction w with
  | nil => rfl
  | cons t w ih =>
      simp only [depthBody, neutral, letterA, letterB] at ih
      cases t with
      | none => simp [depthBody, depthLetters, neutral, letterA, letterB, ih]
      | some b => cases b <;> simp [depthBody, depthLetters, neutral, letterA, letterB, ih]

/-- The executable integer test reduces to the actual alternating-subsequence criterion.
Source: the new vocabulary adapter and §2.4's test over the neutral-free word. -/
theorem depthNext_body (k : ℕ) (w : List (Option Bool)) :
    depthNext k (bos :: depthBody w) =
      if (Transformer.CRASP.altList false k).Sublist w.reduceOption ∧
          ¬(Transformer.CRASP.altList true k).Sublist w.reduceOption then accept else reject := by
  simp only [depthNext, depthLetters_body, ↓reduceIte]

/-- The integer oracle accepts exactly E_k, with the source's positive-k hypothesis.
Source: arXiv:2506.16055v3, §2.4 altPlus_eq and Appendix F's neutral-letter language.
This connects an executable subsequence test to the independent block-language definition. -/
theorem depthNext_accept_iff (k : ℕ) (hk : 0 < k) (w : List (Option Bool)) :
    depthNext k (bos :: depthBody w) = accept ↔ w ∈ Transformer.CRASP.altPlusNeutral k := by
  rw [depthNext_body]
  have he : w ∈ Transformer.CRASP.altPlusNeutral k ↔
      (Transformer.CRASP.altList false k).Sublist w.reduceOption ∧
        ¬(Transformer.CRASP.altList true k).Sublist w.reduceOption := by
    rw [Transformer.CRASP.altPlusNeutral, Set.mem_preimage,
      Transformer.CRASP.altPlus_eq false k hk]
    rfl
  rw [he]
  split_ifs with h
  · exact iff_of_true rfl h
  · exact iff_of_false (by norm_num [reject, accept]) h

example : 0 < 2 := by omega

/-- The full integer-list result appends the correct language label.
Source: the new Basis interface and the independently proved E_k equivalence above. -/
theorem depthFunction_semantics (k : ℕ) (hk : 0 < k) (w : List (Option Bool)) :
    depthFunction k (bos :: depthBody w) =
      (bos :: depthBody w) ++
        [if w ∈ Transformer.CRASP.altPlusNeutral k then accept else reject] := by
  unfold depthFunction extend
  congr 1
  have he := depthNext_accept_iff k hk w
  rw [depthNext_body] at he ⊢
  split_ifs <;> simp_all [accept, reject]

example : 0 < 4 := by omega

/-- Every depth input, valid or invalid, obeys the requested length contract.
Source: the new total integer-list continuation wrapper. -/
theorem depthFunction_length (k : ℕ) (tokens : Tokens) :
    (depthFunction k tokens).length = tokens.length + 1 := extend_length _ _

/-- A supervised E_2 prefix with a neutral letter is accepted.
Source: Basis easy depth and the actual vocabulary IDs. -/
theorem depth_easy_example :
    depthFunction 2 [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 16] := by decide

/-- The same prefix is rejected by E_4; the task mode is an explicit argument.
Source: Basis hard depth; a shared token prefix can have different task labels. -/
theorem depth_hard_example :
    depthFunction 4 [1, 9, 11, 9, 10] = [1, 9, 11, 9, 10, 15] := by decide

/-- Invalid alphabet tokens receive the documented total-function fallback.
Source: the new interface's explicit deviation from Python's ValueError. -/
theorem depth_invalid_example : depthFunction 2 [1, -1] = [1, -1, 0] := by decide

/-- Legal supervised prefixes occur within the actual easy context bound.
Source: Basis's 64-token training context and its prefix-label supervision. -/
theorem depthPrefix_example : DepthPrefix 64 [1, 9, 11, 9, 10] := by
  refine ⟨[some false, none, some false, some true], by decide, by decide, ?_⟩
  rfl

/-- A word starting with b is rejected even if it later contains two a/b runs.
Source: Section 2.4 requires the first active block to contain a. -/
theorem depth_wrong_start : depthFunction 2 [1, 10, 9, 10] = [1, 10, 9, 10, 15] := by
  decide

/-- Neutral letters do not start or split an active block.
Source: Appendix F defines E_k by deleting all neutral letters. -/
theorem depth_neutral_example :
    depthFunction 2 [1, 11, 9, 11, 10, 11] = [1, 11, 9, 11, 10, 11, 16] := by
  decide

end Transformer.Basis
