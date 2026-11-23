module

public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-!
# Reusing an associative read on identical dense input prefixes

Phasor lookup evaluation, consecutive rows of a packed batch. The raw
endpoint vote depends only on the input prefix and fixed weights. A
contextual gate still acts separately on each row's hidden vector.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.SharedRowReads

variable {A H V Z : Type*}

/-- Use the preceding row's raw vote through the common prefix and replay
the current row's vote after its first different embedding. The hidden
argument always belongs to the current row. -/
def cached (read : List A → V) (gate : H → V → Z) (previous current : List A)
    (hidden : ℕ → H) (shared t : ℕ) : Z :=
  gate (hidden t) (if t < shared then read (previous.take (t+1))
    else read (current.take (t+1)))

/-- One equality of input prefixes suffices at all earlier positions.
Rows may diverge anywhere after that prefix, even if they coincide again. -/
theorem agrees (read : List A → V) (gate : H → V → Z) (previous current : List A)
    (hidden : ℕ → H) (shared : ℕ) (equal : previous.take shared = current.take shared)
    (t : ℕ) : cached read gate previous current hidden shared t =
      gate (hidden t) (read (current.take (t+1))) := by
  unfold cached
  split
  next h =>
    have bound : t+1 ≤ shared := h
    have sub := congrArg (List.take (t+1)) equal
    simpa only [List.take_take, Nat.min_eq_left bound] using
      congrArg (fun xs => gate (hidden t) (read xs)) sub
  next => rfl

example : ∀ t, cached List.sum (fun (h v : Nat) => h+v) [2,5,7,1] [2,5,9,1]
    (fun i => 3*i) 2 t = 3*t + ([2,5,9,1].take (t+1)).sum :=
  agrees _ _ _ _ _ _ (by decide)

/-- Comparing only the embedding at a later position can reuse a wrong vote. -/
example : ([2,5,7,1] : List Nat)[3]? = ([2,5,9,1] : List Nat)[3]? ∧
    cached List.sum (fun (_ v : Nat) => v) [2,5,7,1] [2,5,9,1] (fun _ => 0) 4 3 ≠
      ([2,5,9,1] : List Nat).sum := by decide

end Transformer.Phasor.TripleTransport.SharedRowReads
