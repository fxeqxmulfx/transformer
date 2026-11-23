module
public import Transformer.Phasor.TripleTransport.BlockReads

/-! The retained training tape for the Phasor read. A worker owns complete
sequences and returns the ordered frames for its slice. The shared-weight
pullback accumulates the same floating-point additions in the same order. -/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.TrainingRows
variable {S F G : Type*}

/-- Each input sequence emits its whole causal tape, with its own empty
initial store. A block carries no state from one example to the next. -/
def grouped (frames : S → List F) (blocks : List (List S)) : List F :=
  (blocks.map (fun xs => xs.flatMap frames)).flatten

/-- Joining worker results in input order preserves every frame, including the
addresses, hidden gates and the ordered sequence of per-position witnesses. -/
theorem frames_equal (frames : S → List F) (blocks : List (List S)) :
    grouped frames blocks = blocks.flatten.flatMap frames := by
  induction blocks with
  | nil => rfl
  | cons xs rest ih =>
    simpa only [grouped,List.map_cons,List.flatten_cons,List.flatMap_append] using
      congrArg (xs.flatMap frames ++ ·) ih

/-- The same shared-parameter adjoint follows for any accumulator; rounding,
clipping or other order-dependent operations need no algebraic assumptions. -/
theorem gradient_equal (frames : S → List F) (blocks : List (List S))
    (add : G → F → G) (initial : G) :
    (grouped frames blocks).foldl add initial = (blocks.flatten.flatMap frames).foldl add initial := by
  rw [frames_equal]

example : (grouped (fun n:ℕ => [n,n+1]) [[1,3],[],[2]]).foldl (fun g x=>10*g+x) 0 =
    123423 := by decide
end Transformer.Phasor.TripleTransport.TrainingRows
