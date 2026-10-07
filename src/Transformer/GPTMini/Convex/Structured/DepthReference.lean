import Transformer.GPTMini.Convex.Structured.DepthScan

/-!
# Raw six-state depth data semantics for both Basis modes

Source: arXiv:2506.16055v3, §2.4 and Appendix F's E_k, actual
Basis.depthBody/depthNext at cbafbe9 and the independent compressed-word
criterion in DepthCompression. The six-state reference recurrence
receives physical integer BOS/A/B/neutral tokens. It has no external
run count, prepared word or label argument.

Its endpoint and finite vocabulary label are derived equal to the
independent Basis answer for every valid prefix, for k=2 and k=4.
The reference generates data state/channel targets and finite learned
parameter witnesses only. Actual inference still mixes every free
finite stochastic transition/value row, and never calls this rule.
Learned head and genuine tensor-block coupling remain subsequent steps.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis Transformer.CRASP
open scoped Classical

/-- Each nonsaturated positive state corresponds exactly to the independent required compressed alternating word.
Source: actual idempotent reference scan, two-letter chain canonical form, chronological alternating counts and wrong-start rejection. -/
theorem depthReference_state_iff (k : ℕ) (hk : 0 < k) (hcap : k < 5) (word : List Bool) :
    referenceRun depthLetterReference 0 word = depthReferencePhase k ↔
      word.destutter (· ≠ ·) = altList false k := by
  have hread := (referenceRun_destutter depthLetterReference depthLetterReference_idempotent 0 word).symm
  constructor
  · intro hs
    rw [hread] at hs
    let compact := word.destutter (· ≠ ·)
    obtain ⟨phase, hc⟩ := alternatingChain_canonical compact (List.isChain_destutter (· ≠ ·) word)
    have hm : referenceRun depthLetterReference 0 (altList phase compact.length) = depthReferencePhase k := by
      rw [← hc]
      exact hs
    cases phase with
    | false =>
        rw [depthReference_initial] at hm
        have hv := congrArg Fin.val hm
        change min compact.length 5 = min k 5 at hv
        have hn : compact.length = k := by omega
        change compact = altList false k
        rw [hc, hn]
    | true =>
        by_cases hz : compact.length = 0
        · rw [hz] at hm
          change (0 : Fin 6) = depthReferencePhase k at hm
          have hv := congrArg Fin.val hm
          change 0 = min k 5 at hv
          omega
        · obtain ⟨length, hn⟩ : ∃ length, compact.length = length + 1 := ⟨compact.length - 1, by omega⟩
          rw [hn, depthReference_wrong_start] at hm
          have hv := congrArg Fin.val hm
          change 5 = min k 5 at hv
          omega
  · intro hc
    rw [hread, hc, depthReference_initial]

example : (0 : ℕ) < 2 ∧ (2 : ℕ) < 5 := by omega

/-- Reference endpoint acceptance is the original independent alternating-subsequence criterion.
Source: the full true scan/compression equivalence and Basis's A_k minus B_k language, without a supplied correct encoded state. -/
theorem depthReference_criterion (k : ℕ) (hk : 0 < k) (hcap : k < 5) (word : List Bool) :
    referenceRun depthLetterReference 0 word = depthReferencePhase k ↔
      (altList false k).Sublist word ∧ ¬(altList true k).Sublist word := by
  rw [depthReference_state_iff k hk hcap word]
  exact (depthCompression_criterion k hk word).symm

example : (0 : ℕ) < 4 ∧ (4 : ℕ) < 5 := by omega

/-- The independent data recurrence consumes true integer BOS/A/B/neutral IDs, preserving every physical position.
Source: Basis vocabulary.py and the six-state a-first data transitions; invalid alphabet tokens enter rejection, outside valid-prefix claims. -/
def depthRawReference (token : ℤ) (state : Fin 6) : Fin 6 :=
  if token = bos then 0 else if token = neutral then state
  else if token = letterA then depthLetterReference false state
  else if token = letterB then depthLetterReference true state else 5

/-- Actual raw BOS begins the independent depth data scan from state zero.
Source: the physical BOS reset, not an external prepared count or hidden-state input. -/
theorem depthRawReference_bos (state : Fin 6) : depthRawReference bos state = 0 := by
  unfold depthRawReference
  rw [ite_eq_left rfl]

/-- Actual neutral tokens preserve the entire independent data state.
Source: Appendix F's neutral deletion semantics and the actual vocabulary ID 11. -/
theorem depthRawReference_neutral (state : Fin 6) : depthRawReference neutral state = state := by
  norm_num [depthRawReference, bos, neutral]

/-- Actual raw a tokens invoke the data table's a transition.
Source: Basis's A=9 serialization and the explicit six-state recurrence. -/
theorem depthRawReference_a (state : Fin 6) : depthRawReference letterA state = depthLetterReference false state := by
  norm_num [depthRawReference, bos, neutral, letterA]

/-- Actual raw b tokens invoke the data table's b transition.
Source: Basis's B=10 serialization, retaining wrong-start rejection and all run changes. -/
theorem depthRawReference_b (state : Fin 6) : depthRawReference letterB state = depthLetterReference true state := by
  norm_num [depthRawReference, bos, neutral, letterA, letterB]

/-- The real chronological integer body scan equals the active-letter scan after deleting actual neutral positions.
Source: the exact raw serialization and explicit neutral/A/B transitions, proved inductively for every supplied start state. -/
theorem depthRawReference_word (start : Fin 6) (word : List (Option Bool)) :
    referenceRun depthRawReference start (depthBody word) = referenceRun depthLetterReference start word.reduceOption := by
  induction word generalizing start with
  | nil => rfl
  | cons letter word ih =>
      cases letter with
      | none =>
          change referenceRun depthRawReference (depthRawReference neutral start) (depthBody word) =
            referenceRun depthLetterReference start word.reduceOption
          rw [depthRawReference_neutral]
          exact ih start
      | some bit =>
          cases bit with
          | false =>
              change referenceRun depthRawReference (depthRawReference letterA start) (depthBody word) =
                referenceRun depthLetterReference (depthLetterReference false start) word.reduceOption
              rw [depthRawReference_a]
              exact ih _
          | true =>
              change referenceRun depthRawReference (depthRawReference letterB start) (depthBody word) =
                referenceRun depthLetterReference (depthLetterReference true start) word.reduceOption
              rw [depthRawReference_b]
              exact ih _

/-- The complete raw BOS/body prefix reaches the independently derived depth scan endpoint.
Source: genuine raw reset and body induction, with no count, encoded word or desired label passed to the raw recurrence. -/
theorem depthRawReference_body (word : List (Option Bool)) :
    referenceRun depthRawReference 0 (bos :: depthBody word) = referenceRun depthLetterReference 0 word.reduceOption := by
  change referenceRun depthRawReference (depthRawReference bos 0) (depthBody word) = _
  rw [depthRawReference_bos]
  exact depthRawReference_word 0 word

/-- Actual vocabulary accept/reject IDs select the reference state's training output channels and finite witness emissions.
Source: Basis's 36-token depth vocabulary and the two task-dependent positive target run counts; this is not learned inference. -/
def depthReferenceLabel (k : ℕ) (state : Fin 6) : Fin 36 := if state = depthReferencePhase k then 16 else 15

/-- Actual raw reference labels equal the independently checked Basis answer on every serialized depth word.
Source: original raw parser/subsequence semantics, exact integer scan and derived finite-state acceptance equivalence, with k restricted to one through four. -/
theorem depthRawReference_answer (k : ℕ) (hk : 0 < k) (hcap : k < 5) (word : List (Option Bool)) :
    ((depthReferenceLabel k (referenceRun depthRawReference 0 (bos :: depthBody word))).val : ℤ) =
      depthNext k (bos :: depthBody word) := by
  rw [depthRawReference_body, depthNext_body]
  have he := depthReference_criterion k hk hcap word.reduceOption
  unfold depthReferenceLabel
  split_ifs with hstate hcriterion hcriterion
  · rfl
  · exact False.elim (hcriterion (he.mp hstate))
  · exact False.elim (hstate (he.mpr hcriterion))
  · rfl

example : (0 : ℕ) < 2 ∧ (2 : ℕ) < 5 := by omega

/-- Every actual validated raw depth prefix has the right independent reference output ID.
Source: Basis.DepthPrefix's full neutral-preserving grammar and the proved raw state/answer relation; learned inference remains separate. -/
theorem depthRawReference_next (k bound : ℕ) (hk : 0 < k) (hcap : k < 5) (tokens : Tokens)
    (hprefix : DepthPrefix bound tokens) :
    ((depthReferenceLabel k (referenceRun depthRawReference 0 tokens)).val : ℤ) = depthNext k tokens := by
  obtain ⟨word, _, _, rfl⟩ := hprefix
  exact depthRawReference_answer k hk hcap word

example : (0 : ℕ) < 4 ∧ (4 : ℕ) < 5 ∧ DepthPrefix 128 [1, 9, 10, 9, 10] :=
  ⟨by omega, by omega, [some false, some true, some false, some true], by decide, by decide, rfl⟩

end Transformer.GPTMini.Convex.Structured
