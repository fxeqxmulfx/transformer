module

public import Transformer.Phasor.TripleTransport.PairClasses

/-!
# Ranked bitmap storage for pair-class candidate masks

Source: “Indexing the pair-class images” in the triple-transport experiment.
Each present key stores one candidate set. The bitmap determines whether
the key is present and locates the set in the packed array.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.MaskRank
variable {V : Type*}

/-- The number of stored records strictly before a logical key. -/
def rank (xs : List (Option V)) (i : ℕ) : ℕ := ((xs.take i).filterMap id).length

def read (xs : List (Option V)) (i : ℕ) : Option V :=
  if xs[i]?.join.isSome then (xs.filterMap id)[rank xs i]? else none

/-- A membership bit and the number of preceding set bits recover exactly
the original optional value, including keys outside the allocated range. -/
theorem packed (xs : List (Option V)) (i : ℕ) : read xs i = xs[i]?.join := by
  induction xs generalizing i with
  | nil => simp [read]
  | cons x xs ih =>
    cases i with
    | zero => cases x <;> simp [read,rank]
    | succ i => cases x <;> simpa [read,rank] using ih i

/-- Complete earlier bitmap words contribute a constant offset. Within
the new word only the bits before the query contribute. -/
theorem block_rank (xs ys : List (Option V)) (i : ℕ) :
    rank (xs++ys) (xs.length+i) = (xs.filterMap id).length+rank ys i := by
  simp [rank,List.take_length_add_append,List.filterMap_append]

example : read [none,some (7:ℕ),none,some 3] 3 = some 3 := by
  rw [packed]; decide

example : read [none,some (7:ℕ),none,some 3] 12 = none := by
  rw [packed]; decide

example : rank ([some true,none,some false] ++ [none,some false]) (3+1) =
    ([some true,none,some false].filterMap id).length + rank [none,some false] 1 :=
  block_rank _ _ _

end Transformer.Phasor.TripleTransport.MaskRank
