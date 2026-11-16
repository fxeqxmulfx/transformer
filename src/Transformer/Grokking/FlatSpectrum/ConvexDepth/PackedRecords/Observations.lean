module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords.Basic

/-!
# Original observation order and constraints under packed states

Source: `review/flat_spectrum.md`, CM68. Precompute every fixed program's
state, then pack the resulting observations. Record IDs and multiplicity are
unchanged, including duplicates and every actual short minibatch.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords

variable {J X I : Type*} {s c : ℕ}

/-- Materialize every fixed input-program result at one original observation,
CM68. No candidate or original record is removed by this map. -/
def states (program : J → X → Fin s) (r : X × Fin c) : Row J s c :=
  (fun j => program j r.1, r.2)

/-- Read a precomputed state by its original program index, CM68. -/
def coordinate (j : J) (x : J → Fin s) : Fin s := x j

/-- Each original record ID recovers its exact ordered state vector and raw
class after packing, CM68. No uniqueness premise on observations is needed. -/
theorem indexed_roundtrip (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (rows : I → Row J s c) :
    (fun i => decode (encode (rows i)) (encode_valid hs hc (rows i))) = rows := by
  funext i
  exact decode_encode hs hc (rows i)

/-- Any ordered list of record IDs preserves every occurrence, CM68.
In particular an original batch tail is not padded, resampled or deduplicated. -/
theorem ordered_roundtrip (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (rows : I → Row J s c) (ids : List I) :
    ids.map (fun i => decode (encode (rows i)) (encode_valid hs hc (rows i))) =
      ids.map rows := by
  rw [indexed_roundtrip hs hc rows]

/-- Precomputation preserves the first observed raw decoder label at each
state, CM68. Thus observations and conflict checks see the same full records. -/
theorem lookup_states (program : J → X → Fin s) (samples : List (X × Fin c))
    (j : J) (z : Fin s) :
    LatentPrograms.lookup (coordinate j) (samples.map (states program)) z =
      LatentPrograms.lookup (program j) samples z := by
  induction samples with
  | nil => rfl
  | cons t rest ih =>
    change (if program j t.1 = z then some t.2 else
      LatentPrograms.lookup (coordinate j) (rest.map (states program)) z) =
      (if program j t.1 = z then some t.2 else LatentPrograms.lookup (program j) rest z)
    rw [ih]

/-- The complete executable consistency filter retains the same ordered
candidate list after state precomputation, CM68; no top-k operation is hidden. -/
theorem retained_states (program : J → X → Fin s) (order : List J)
    (samples : List (X × Fin c)) :
    LatentPrograms.retained coordinate order (samples.map (states program)) =
      LatentPrograms.retained program order samples := by
  unfold LatentPrograms.retained
  apply List.filter_congr
  intro j _
  rw [List.all_map]
  apply congrArg (fun f : X × Fin c → Bool => samples.all f)
  funext t
  change decide (LatentPrograms.lookup (coordinate j) (samples.map (states program))
    (program j t.1) = some t.2) =
    decide (LatentPrograms.lookup (program j) samples (program j t.1) = some t.2)
  rw [lookup_states]

variable [Fintype J]

/-- All full joint anchor equations and mass bounds survive state
precomputation, CM68. Both sets retain every unknown decoder coordinate. -/
theorem domain_states (program : J → X → Fin s) (order : List J)
    (samples : List (X × Fin c)) (δ R : ℝ) :
    BoundedLatent.domain coordinate order (samples.map (states program)) δ R =
      BoundedLatent.domain program order samples δ R := by
  ext q
  constructor
  · rintro ⟨h, lower, upper⟩
    refine ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, ?_⟩, lower, upper⟩
    intro j t ht d
    exact h.2.2.2.2 j (states program t) (List.mem_map_of_mem ht) d
  · rintro ⟨h, lower, upper⟩
    refine ⟨⟨h.1, h.2.1, h.2.2.1, h.2.2.2.1, ?_⟩, lower, upper⟩
    intro j t ht d
    obtain ⟨r, hr, rfl⟩ := List.mem_map.mp ht
    exact h.2.2.2.2 j r hr d

/-- Packed decoding preserves precisely the original full feasible set,
CM68, including cases whose observations have repeated values. -/
theorem domain_roundtrip (hs : s ≤ UInt16.size) (hc : c ≤ UInt16.size)
    (program : J → X → Fin s) (order : List J) (rows : I → X × Fin c)
    (ids : List I) (δ R : ℝ) :
    BoundedLatent.domain coordinate order
      (ids.map (fun i => decode (encode (states program (rows i)))
        (encode_valid hs hc (states program (rows i))))) δ R =
      BoundedLatent.domain program order (ids.map rows) δ R := by
  rw [ordered_roundtrip hs hc]
  have h := domain_states program order (ids.map rows) δ R
  simpa only [List.map_map, Function.comp_def] using h

end Transformer.Grokking.FlatSpectrum.ConvexDepth.PackedRecords
