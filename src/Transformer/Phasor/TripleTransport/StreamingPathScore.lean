module

public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-!
# Frozen scoring of the repeated associative read

In the Phasor lookup experiment, each causal store enumerates all of its
path witnesses in a fixed order. Training retains the witnesses for the
parameter adjoint. The frozen scorer can update each endpoint's vote as it
visits a witness. The update below is arbitrary, so the result also
applies to rounded, order-sensitive addition.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.StreamingPathScore

variable {S X W V Y : Type*}

/-- The inner lists are the ordered witnesses of each successive read.
Visiting them directly gives the same endpoint accumulator as replaying
the saved, flattened list, even when addresses repeat. -/
theorem ordered (apply : V → W → V) (groups : List (List W)) (initial : V) :
    groups.foldl (fun v row => row.foldl apply v) initial =
      groups.flatten.foldl apply initial := by
  induction groups generalizing initial with
  | nil => rfl
  | cons row rows ih =>
    rw [List.foldl_cons, List.flatten_cons, List.foldl_append, ih]

/-- Run the causal state update before scoring each position. -/
def scan (step : S → X → S) (read : S → Y) (s : S) : List X → List Y
  | [] => []
  | x :: xs => let next := step s x; read next :: scan step read next xs

/-- Every position of every prefix yields the same finalized read, for
any state transition, witness enumeration and final dense projection.
The frozen pass may discard all witnesses before the next position. -/
theorem every_output (step : S → X → S) (groups : S → List (List W))
    (apply : V → W → V) (initial : V) (finish : S → V → Y) (s : S) (xs : List X) :
    scan step (fun k => finish k ((groups k).foldl (fun v row => row.foldl apply v) initial)) s xs =
      (scan step (fun k => (k, (groups k).flatten)) s xs).map
        (fun p => finish p.1 (p.2.foldl apply initial)) := by
  induction xs generalizing s with
  | nil => rfl
  | cons x xs ih =>
    simp only [scan, List.map_cons]
    rw [ih, ordered]

example : [[1, 2], [], [3, 2]].foldl (fun v row => row.foldl (fun a b => 10*a+b) v) 4 =
    [1, 2, 3, 2].foldl (fun a b => 10*a+b) 4 :=
  ordered _ [[1, 2], [], [3, 2]] _

end Transformer.Phasor.TripleTransport.StreamingPathScore
