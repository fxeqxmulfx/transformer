module

public import Mathlib.Logic.Equiv.Fin.Basic
public import Transformer.Phasor.TripleTransport.PairClasses

/-!
# Ranked bitmap storage for pair-class candidate masks

Source: “Indexing the pair-class images” in the triple-transport experiment.
Each present key stores one candidate set. The bitmap determines whether
the key is present and locates the set in the packed array.
The section “Storing image classes in query order” groups the compact source
classes by query before scanning candidate masks.
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

namespace Transformer.Phasor.TripleTransport.MaskRank.QueryClasses

variable {n : ℕ} {R I V S : Type*}

/-- The native flat array stores its minor coordinate contiguously. -/
abbrev address (major minor : Fin n) : Fin (n * n) := finProdFinEquiv (major, minor)

/-- Store the compact image class with the query as the major coordinate.
The component table supplies its root with the anchor as the major coordinate. -/
def prepare (roots : Fin (n * n) → R) (imageClass : R → Option I) : Array (Option I) :=
  Array.ofFn fun index : Fin (n * n) =>
    let p := finProdFinEquiv.symm index
    imageClass (roots (address p.2 p.1))

def read (roots : Fin (n * n) → R) (imageClass : R → Option I) (query anchor : Fin n) : Option I :=
  (prepare roots imageClass)[(address query anchor).val]'(by
    simpa [prepare] using (address query anchor).isLt)

/-- Transposing the two finite coordinates and ranking during construction
recovers the original optional image class at every anchor and query. -/
theorem correct (roots : Fin (n * n) → R) (imageClass : R → Option I) (q k : Fin n) :
    read roots imageClass q k = imageClass (roots (address k q)) := by
  simp only [read, prepare, Array.getElem_ofFn]
  change imageClass (roots (address (finProdFinEquiv.symm (finProdFinEquiv (q,k))).2
    (finProdFinEquiv.symm (finProdFinEquiv (q,k))).1)) = _
  rw [Equiv.symm_apply_apply]

/-- For any ordered constraint scan, both layouts supply exactly the same
classes. Its accumulator may track candidate intersections, support counts
and an absorbing state for an empty intersection. -/
theorem scan (roots : Fin (n * n) → R) (imageClass : R → Option I) (q : Fin n)
    (anchors : List (Fin n × V)) (step : S → V → Option I → S) (start : S) :
    anchors.foldl (fun s a => step s a.2 (read roots imageClass q a.1)) start =
    anchors.foldl (fun s a => step s a.2 (imageClass (roots (address a.1 q)))) start := by
  apply congrArg (fun reduce : S → Fin n × V → S => anchors.foldl reduce start)
  funext s a
  rw [correct]

private def exampleRoots (i : Fin (3 * 3)) := i.val
private def exampleClass (i : ℕ) : Option ℕ := if i < 7 then some (i + 1) else none

example : [((0 : Fin 3), true), (2, true), (1, false)].foldl
    (fun s a => (read exampleRoots exampleClass 1 a.1, a.2) :: s) [] =
    [(some 5, false), (none, true), (some 2, true)] := by decide

end Transformer.Phasor.TripleTransport.MaskRank.QueryClasses
