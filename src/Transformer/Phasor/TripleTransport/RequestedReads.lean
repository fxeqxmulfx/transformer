module
public import Transformer.Phasor.TripleTransport.SharedRowReads

/-!
# Evaluating the final read only at requested output positions

The Phasor causal store must consume every embedding. Its final projection
at one position cannot change either the shared backbone or any later write.
A next-token caller can therefore ask for the logits at the last real
position of each packed sequence, including rows with padding.
-/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.RequestedReads
variable {A S V : Type*}

/-- Compute every causal output from the updated state. The flags are
external requests for output positions and are ignored by the state update. -/
def dense (step : S → A → S) (emit : S → A → V) (s : S) : List (Bool × A) → List (Bool × V)
  | [] => []
  | (b, x) :: xs => let t := step s x; (b, emit t x) :: dense step emit t xs

/-- Advance the same state at every position and compute a projection only
when requested. In a memory read the state includes the full prefix store. -/
def selected (step : S → A → S) (emit : S → A → V) (s : S) : List (Bool × A) → List V
  | [] => []
  | (b, x) :: xs => let t := step s x
                     if b then emit t x :: selected step emit t xs else selected step emit t xs

/-- The ordered requested output values agree with gathering those values
from a complete dense forward. This applies to the classifier logits too. -/
theorem gathered (step : S → A → S) (emit : S → A → V) (s : S) (xs : List (Bool × A)) :
    selected step emit s xs =
      (dense step emit s xs).filterMap (fun p => if p.1 then some p.2 else none) := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨b, a⟩ := x
    cases b <;> simp [dense, selected, ih]

/-- Unrequested positions still update the store. The final emission sees
precisely the state from the requested prefix, for any following padding. -/
theorem last_real (step : S → A → S) (emit : S → A → V) (s : S) (xs : List A)
    (last : A) (pad : List A) :
    selected step emit s (xs.map (fun x => (false, x)) ++ (true, last) :: pad.map (fun x => (false, x))) =
      [emit (step (xs.foldl step s) last) last] := by
  have skipped (st : S) (tail : List A) :
      selected step emit st (tail.map (fun a => (false, a))) = [] := by
    induction tail generalizing st with
    | nil => rfl
    | cons a tail ih => simp [selected, ih]
  induction xs generalizing s with
  | nil => simp [selected, skipped]
  | cons a xs ih => simpa [selected] using ih (step s a)

example : selected (· + ·) (fun (s a : ℕ) => (s, a)) 1
    [(false, 2), (false, 4), (true, 8), (false, 1000)] = [(15, 8)] := by decide
end Transformer.Phasor.TripleTransport.RequestedReads
