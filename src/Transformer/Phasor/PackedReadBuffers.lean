module

public import Transformer.Phasor.PairValues

/-!
# Packing the forward addresses and the stable sparse transpose

Source: the packed-transfer trial in `experiments/phasor_value_transport/README.md`.
The forward pass uploads table addresses followed by its gain evidence.
The transpose uploads its offset directory followed by a flat list of the
rows that own each address. The final allocation may retain an arbitrary
unused suffix from an earlier minibatch.
-/

@[expose] public section

namespace Transformer.Phasor

/-- Table addresses occupy the first region of the forward upload. Gain
evidence starts exactly at the configured capacity of that first region. -/
theorem packedForward_entries (keys evidence : List ℕ) (i j : ℕ) (hi : i < keys.length) :
    (keys ++ evidence)[i]? = keys[i]? ∧
      (keys ++ evidence)[keys.length + j]? = evidence[j]? := by
  refine ⟨List.getElem?_append_left hi, ?_⟩
  rw [List.getElem?_append_right (by omega)]
  simp

/-- Offset at each address boundary, including the past-the-end sentinel. -/
def ownerOffsets (rows : List (List ℕ)) : List ℕ :=
  (List.range (rows.length + 1)).map (fun i => (rows.take i).flatten.length)

/-- Store all boundaries and then the owners in their accumulation order. -/
def ownerBuffer (rows : List (List ℕ)) (unused : List ℕ) : List ℕ :=
  ownerOffsets rows ++ (rows.flatten ++ unused)

/-- The kernel uses the two consecutive directory entries at an address to
read its half-open owner interval from the second region. -/
def readOwnerInterval (buffer : List ℕ) (addresses k : ℕ) : Option (List ℕ) := do
  let lo ← buffer[k]?
  let hi ← buffer[k+1]?
  return (buffer.drop (addresses + 1 + lo)).take (hi - lo)

/-- The entry at each valid directory index counts all preceding owners. -/
theorem ownerBuffer_boundary (rows : List (List ℕ)) (unused : List ℕ) (i : ℕ)
    (hi : i ≤ rows.length) :
    (ownerBuffer rows unused)[i]? = some (rows.take i).flatten.length := by
  have bound : i < (ownerOffsets rows).length := by
    simp only [ownerOffsets, List.length_map, List.length_range]
    omega
  rw [ownerBuffer, List.getElem?_append_left bound, ownerOffsets, List.getElem?_map,
    List.getElem?_range (by omega)]
  rfl

/-- Decoding any address gives exactly its ordered owners. Empty intervals,
repeated owners, and a stale tail of arbitrary size are all allowed. -/
theorem readOwnerInterval_exact (before after : List (List ℕ)) (owners unused : List ℕ) :
    let rows := before ++ owners :: after
    readOwnerInterval (ownerBuffer rows unused) rows.length before.length = some owners := by
  let rows := before ++ owners :: after
  have lo : rows.take before.length = before := List.take_left
  have hi : rows.take (before.length + 1) = before ++ [owners] := by
    rw [show rows = (before ++ [owners]) ++ after by simp [rows]]
    simpa only [List.length_append, List.length_singleton] using
      (List.take_left (l₁ := before ++ [owners]) (l₂ := after))
  have size : (ownerOffsets rows).length = rows.length + 1 := by simp [ownerOffsets]
  change readOwnerInterval (ownerBuffer rows unused) rows.length before.length = some owners
  rw [readOwnerInterval, ownerBuffer_boundary _ _ _ (by simp [rows]),
    ownerBuffer_boundary _ _ _ (by simp [rows]), lo, hi]
  change some (((ownerBuffer rows unused).drop
    (rows.length + 1 + before.flatten.length)).take
      ((before ++ [owners]).flatten.length - before.flatten.length)) = some owners
  simp only [List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil,
    List.length_append, Nat.add_sub_cancel_left]
  unfold ownerBuffer
  rw [← size, ← List.drop_drop, List.drop_left]
  simp [rows, List.append_assoc]

/-- The packed kernel therefore performs the same ordered reduction even
when rounding makes a change in summation order observable. -/
theorem packedOwner_fold {α : Type*} (step : α → ℕ → α) (initial : α)
    (before after : List (List ℕ)) (owners unused : List ℕ) :
    let rows := before ++ owners :: after
    (readOwnerInterval (ownerBuffer rows unused) rows.length before.length).map
      (fun indices => indices.foldl step initial) = some (owners.foldl step initial) := by
  simp only [readOwnerInterval_exact, Option.map_some]

example : ([4, 8, 2] ++ [7, 3] : List ℕ)[2]? = some 2 ∧
    ([4, 8, 2] ++ [7, 3] : List ℕ)[4]? = some 3 :=
  packedForward_entries [4, 8, 2] [7, 3] 2 1 (by decide)

example : (ownerBuffer [[2, 2, 0], [], [1]] [91, 93])[2]? = some 3 :=
  ownerBuffer_boundary [[2, 2, 0], [], [1]] [91, 93] 2 (by decide)

example : readOwnerInterval (ownerBuffer [[2, 2, 0], [], [1]] [91, 93]) 3 1 = some [] :=
  readOwnerInterval_exact [[2, 2, 0]] [[1]] [] [91, 93]

end Transformer.Phasor
