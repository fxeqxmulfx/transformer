import Transformer.GPTMini.Convex.Structured.DepthCompression
import Transformer.GPTMini.Convex.Structured.MarkovTeacher

/-!
# Six-state chronological depth data scan

Source: arXiv:2506.16055v3, §2.4's alternating runs and the independent
run/subsequence equivalence in DepthCompression. States 0 through 4
track a valid a-first prefix with that many runs; state 5 rejects a
wrong first active b or at least five runs. This transition rule creates
training targets and finite weight witnesses only, never hard inference.

Repeated-letter transitions are proved idempotent, and arbitrary such
chronological scans preserve their result under actual run compression.
The scan of a true alternating word is derived for every finite length.
Raw neutral/BOS serialization, Basis labels and the actual freely learned
stochastic head remain subsequent coupling obligations.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.CRASP
open scoped Classical

/-- Explicit six-state data transitions on the two active letters, with no external run count.
Source: a-first alternating-block semantics; the freely learned model uses finite softmax rows instead of this data rule. -/
def depthLetterReference (letter : Bool) (state : Fin 6) : Fin 6 :=
  if letter then
    if state = 0 then 5 else if state = 1 then 2 else if state = 3 then 4 else state
  else
    if state = 0 then 1 else if state = 2 then 3 else if state = 4 then 5 else state

/-- Consecutive repeated active letters do not change the data run state after the first occurrence.
Source: every explicit transition of the six-state reference table, including the absorbing rejection state. -/
theorem depthLetterReference_idempotent (letter : Bool) (state : Fin 6) :
    depthLetterReference letter (depthLetterReference letter state) = depthLetterReference letter state := by
  cases letter <;> fin_cases state <;> norm_num [depthLetterReference]

/-- State five remains rejecting for either next active letter.
Source: the actual finite table; wrong starts and excessive alternation can never return to an accepting state. -/
theorem depthLetterReference_dead (letter : Bool) : depthLetterReference letter 5 = 5 := by
  cases letter <;> norm_num [depthLetterReference]

/-- An actual chronological data scan with idempotent letters keeps its result after run compression with a known first letter.
Source: List.destutter' removes only consecutive equal symbols, and each real repeated transition is algebraically idempotent. -/
theorem referenceRun_destutter_head {A S : Type*} [DecidableEq A] (rule : A → S → S)
    (hid : ∀ token state, rule token (rule token state) = rule token state)
    (token : A) (start : S) (tail : List A) :
    referenceRun rule start (tail.destutter' (· ≠ ·) token) = referenceRun rule start (token :: tail) := by
  induction tail generalizing token start with
  | nil => rfl
  | cons next tail ih =>
      by_cases he : token = next
      · subst next
        rw [List.destutter'_cons_neg tail (by exact fun h => h rfl), ih]
        change referenceRun rule (rule token start) tail = referenceRun rule (rule token (rule token start)) tail
        rw [hid]
      · rw [List.destutter'_cons_pos tail he]
        change referenceRun rule (rule token start) (tail.destutter' (· ≠ ·) next) =
          referenceRun rule (rule token start) (next :: tail)
        exact ih next (rule token start)

example : ∀ token state : Fin 3, (fun (_ : Fin 3) (s : Fin 3) => s) token
    ((fun (_ : Fin 3) (s : Fin 3) => s) token state) = (fun (_ : Fin 3) (s : Fin 3) => s) token state := by
  intro token state
  rfl

/-- Actual idempotent-letter chronological scans preserve their endpoint under genuine run compression.
Source: the proved first-letter compression invariant and the real empty/nonempty raw list split. -/
theorem referenceRun_destutter {A S : Type*} [DecidableEq A] (rule : A → S → S)
    (hid : ∀ token state, rule token (rule token state) = rule token state)
    (start : S) (tokens : List A) :
    referenceRun rule start (tokens.destutter (· ≠ ·)) = referenceRun rule start tokens := by
  cases tokens with
  | nil => rfl
  | cons token tail =>
      rw [List.destutter_cons']
      exact referenceRun_destutter_head rule hid token start tail

example : ∀ token state : Fin 2, (fun t (_ : Fin 2) => t) token
    ((fun t (_ : Fin 2) => t) token state) = (fun t (_ : Fin 2) => t) token state := by
  intro token state
  rfl

/-- Saturating run-count data state, used to prove the reference table rather than supplied to learned inference.
Source: six physical states, with all counts at least five in the absorbing rejecting state. -/
def depthReferencePhase (count : ℕ) : Fin 6 := ⟨min count 5, by omega⟩

/-- The next truly alternating letter advances the data run state, including its saturation boundary.
Source: the explicit six-state reference table, count parity of the alternating word and state-five absorption. -/
theorem depthReference_advance (count : ℕ) :
    depthLetterReference (decide (count % 2 = 1)) (depthReferencePhase count) = depthReferencePhase (count + 1) := by
  by_cases hc : count ≤ 4
  · interval_cases count <;> norm_num [depthLetterReference, depthReferencePhase]
  · have hm : min count 5 = 5 := by omega
    have hn : min (count + 1) 5 = 5 := by omega
    change depthLetterReference _ ⟨min count 5, _⟩ = ⟨min (count + 1) 5, _⟩
    simp only [hm, hn]
    exact depthLetterReference_dead _

/-- Consecutive run counts flip the actual alternating word phase.
Source: elementary count parity, independent of a learned or supplied encoded state. -/
theorem depthReference_flip (count : ℕ) :
    (!(decide (count % 2 = 1))) = decide ((count + 1) % 2 = 1) := by
  by_cases h : count % 2 = 1
  · have hn : ¬(count + 1) % 2 = 1 := by omega
    simp only [h, hn, decide_true, decide_false, Bool.not_true]
  · have hn : (count + 1) % 2 = 1 := by omega
    simp only [h, hn, decide_true, decide_false, Bool.not_false]

/-- Every finite continuation of a true alternating word gives exactly the saturated chronological reference state.
Source: real list induction, actual per-letter transitions and independently derived parity phase flips. -/
theorem depthReference_alternating (count length : ℕ) :
    referenceRun depthLetterReference (depthReferencePhase count) (altList (decide (count % 2 = 1)) length) =
      depthReferencePhase (count + length) := by
  induction length generalizing count with
  | zero => rfl
  | succ length ih =>
      rw [altList_succ]
      change referenceRun depthLetterReference
        (depthLetterReference (decide (count % 2 = 1)) (depthReferencePhase count))
        (altList (!(decide (count % 2 = 1))) length) = depthReferencePhase (count + (length + 1))
      rw [depthReference_advance, depthReference_flip, ih]
      congr 1
      omega

/-- Starting from the actual initial state, every a-first alternating word gives its true saturated run count.
Source: the proved chronological alternating continuation at count zero, without a count supplied to inference. -/
theorem depthReference_initial (length : ℕ) :
    referenceRun depthLetterReference 0 (altList false length) = depthReferencePhase length := by
  have h := depthReference_alternating 0 length
  have hz : depthReferencePhase 0 = 0 := by rfl
  rw [hz] at h
  simpa using h

/-- Every actual active-letter continuation from rejection stays rejected.
Source: the genuine absorbing transition table and chronological list induction. -/
theorem depthReference_dead_run (tokens : List Bool) : referenceRun depthLetterReference 5 tokens = 5 := by
  induction tokens with
  | nil => rfl
  | cons token tokens ih =>
      change referenceRun depthLetterReference (depthLetterReference token 5) tokens = 5
      rw [depthLetterReference_dead]
      exact ih

/-- A nonempty word whose first active letter is b cannot return to a valid a-first run state.
Source: the actual start-state transition and proved state-five absorption, not a task label chosen in forward inference. -/
theorem depthReference_wrong_start (length : ℕ) :
    referenceRun depthLetterReference 0 (altList true (length + 1)) = 5 := by
  rw [altList_succ]
  change referenceRun depthLetterReference (depthLetterReference true 0) (altList false length) = 5
  have hfirst : depthLetterReference true 0 = 5 := by norm_num [depthLetterReference]
  rw [hfirst]
  exact depthReference_dead_run _

end Transformer.GPTMini.Convex.Structured
