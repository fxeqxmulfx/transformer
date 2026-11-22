module

public import Transformer.Phasor.ClosedRelations

/-!
# Sorted source rows for a causal relation store

Source: the relation storage trial in `experiments/phasor_value_transport/README.md`.
Each source has a sparse row of relation labels and current destinations.
Inserting an edge preserves strict order and replaces its old destination.
Reading the rows after a finite write stream gives the same partial function
as applying the writes to one table of source–relation pairs.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*}

/-- Replace a relation's destination in its sorted row, or insert it at
the unique position that preserves label order. -/
def insertRelation (h : ℕ) (v : β) : List (ℕ × β) → List (ℕ × β)
  | [] => [(h,v)]
  | (j,x) :: tail =>
    if h < j then (h,v) :: (j,x) :: tail else if h = j then (h,v) :: tail
    else (j,x) :: insertRelation h v tail

private theorem lookupRow_cons (q j : ℕ) (x : β) (tail : List (ℕ × β)) :
    ((j,x) :: tail).lookup q = if q = j then some x else tail.lookup q := by
  by_cases eq : q = j
  · simp [eq, List.lookup]
  · simp [List.lookup, beq_false_of_ne eq, eq]

/-- Insertion updates exactly one queried label, including a rewrite from
one destination to another. This lookup identity even holds on unsorted rows. -/
theorem lookup_insertRelation (h q : ℕ) (v : β) (row : List (ℕ × β)) :
    (insertRelation h v row).lookup q = if q = h then some v else row.lookup q := by
  induction row with
  | nil => simp [insertRelation, lookupRow_cons]
  | cons pair tail ih =>
    obtain ⟨j,x⟩ := pair
    by_cases before : h < j
    · by_cases hit : q = h <;> simp [insertRelation, before, lookupRow_cons, hit]
    · by_cases same : h = j
      · subst j
        by_cases hit : q = h <;> simp [insertRelation, lookupRow_cons, hit]
      · by_cases old : q = j
        · subst q
          simp [insertRelation, before, same, Ne.symm same]
        · simp [insertRelation, before, same, old, lookupRow_cons, ih]

private theorem below_insertRelation (lo h : ℕ) (v : β) (row : List (ℕ × β)) (lt : lo < h)
    (lower : ∀ a ∈ row, lo < a.1) : ∀ a ∈ insertRelation h v row, lo < a.1 := by
  induction row with
  | nil => simpa [insertRelation] using lt
  | cons pair tail ih =>
    obtain ⟨j,x⟩ := pair
    have first : lo < j := lower (j,x) (by simp)
    have rest : ∀ a ∈ tail, lo < a.1 := fun a ha => lower a (by simp [ha])
    intro a ha
    rw [insertRelation] at ha
    split at ha
    · rcases List.mem_cons.mp ha with rfl | old
      · exact lt
      · exact lower a old
    · split at ha
      · rcases List.mem_cons.mp ha with rfl | old
        · exact lt
        · exact rest a old
      · rcases List.mem_cons.mp ha with rfl | later
        · exact first
        · exact ih rest a later

/-- Strictly ordered labels give a unique position for binary search.
Both adding a new key and replacing an existing value preserve this invariant. -/
theorem insertRelation_ordered (h : ℕ) (v : β) (row : List (ℕ × β))
    (ordered : row.Pairwise (fun a b => a.1 < b.1)) :
    (insertRelation h v row).Pairwise (fun a b => a.1 < b.1) := by
  induction row with
  | nil => simp [insertRelation]
  | cons pair tail ih =>
    obtain ⟨j,x⟩ := pair
    obtain ⟨lower, tailOrder⟩ := List.pairwise_cons.mp ordered
    by_cases before : h < j
    · rw [insertRelation, ite_eq_left before, List.pairwise_cons]
      refine ⟨?_, ordered⟩
      intro a ha
      rcases List.mem_cons.mp ha with rfl | later
      · exact before
      · exact lt_trans before (lower a later)
    · by_cases same : h = j
      · subst h
        simpa [insertRelation, List.pairwise_cons] using List.pairwise_cons.mpr ⟨lower, tailOrder⟩
      · rw [insertRelation, ite_eq_right before, ite_eq_right same, List.pairwise_cons]
        exact ⟨below_insertRelation j h v tail (by omega) lower, ih tailOrder⟩

/-- Interpret source-indexed rows as the partial function used by the
closed-relation vote. An unwritten label yields `none`. -/
def readRelationRows (rows : α → List (ℕ × β)) (key : α × ℕ) : Option β :=
  (rows key.1).lookup key.2

/-- A causal triple writes only the row belonging to its source. -/
def writeRelationRow [DecidableEq α] (rows : α → List (ℕ × β)) (edge : (α × ℕ) × β) :
    α → List (ℕ × β) :=
  Function.update rows edge.1.1 (insertRelation edge.1.2 edge.2 (rows edge.1.1))

/-- Row writes and flat writes denote the same table for all sources and
relation labels, independently of earlier writes. -/
theorem readRelationRows_write [DecidableEq α] (rows : α → List (ℕ × β)) (edge : (α × ℕ) × β) :
    readRelationRows (writeRelationRow rows edge) =
      Function.update (readRelationRows rows) edge.1 (some edge.2) := by
  obtain ⟨⟨s,h⟩,v⟩ := edge
  funext ⟨a,q⟩
  by_cases same : a = s <;> simp [readRelationRows, writeRelationRow,
    Function.update_apply, same, lookup_insertRelation]

/-- For any ordered stream of writes, both stores agree as entire partial
functions. Consequently repeated overwrites give the same two-hop inputs
and the same support counts at every prefix of the stream. -/
theorem readRelationRows_fold [DecidableEq α] (rows : α → List (ℕ × β))
    (writes : List ((α × ℕ) × β)) :
    readRelationRows (writes.foldl writeRelationRow rows) =
      writes.foldl (fun (t : α × ℕ → Option β) edge => Function.update t edge.1 (some edge.2))
        (readRelationRows rows) := by
  induction writes generalizing rows with
  | nil => rfl
  | cons edge tail ih => simp only [List.foldl, ih, readRelationRows_write]

example : (insertRelation 2 8 [(1,4),(2,5),(3,6)]).Pairwise (fun a b => a.1 < b.1) :=
  insertRelation_ordered 2 8 _ (by decide)

example : readRelationRows ([((0,2),4),((1,2),9),((0,2),7)].foldl writeRelationRow
    (fun _ : ℕ => [])) (0,2) = some 7 := by decide

end Transformer.Phasor
