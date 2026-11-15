module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Transitions

/-!
# Exact count constraints for a streamed candidate join

Source: `review/flat_spectrum.md`, CM62. A fixed retained anchor bank can be
indexed while the other side is streamed. Both index orientations test the
actual inclusive-count comparison on every known observed prefix. Batching
does not require retaining the whole query bank, nor make a truncated anchor
complete. These statements concern observed fits, not unseen answers.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Streaming

variable {A I J K : Type*}

/-- Match the known comparison truth at every original observed prefix
(CM62). Gates and raw-label orientation must first determine these known bits;
unknown comparison values must not be inserted as observations. -/
def CountFit (observed : Finset (List A)) (labels : List A → Bool)
    (left right extra : List A → Bool) (offset : ℤ) : Prop :=
  ∀ w ∈ observed, decide ((prefixCount left w : ℤ) <
    (prefixCount right w : ℤ) + (prefixCount extra w : ℤ) + offset) = labels w

/-- A threshold index needs only the zero and length-plus-one sentinels
outside the actual count range (CM62). Clipping preserves membership exactly. -/
theorem count_threshold_clamp (count length : ℕ) (hcount : count ≤ length) (t : ℤ) :
    t ≤ (count : ℤ) ↔ max 0 (min t ((length : ℤ) + 1)) ≤ count := by
  omega

/-- CM58's actual inclusive counts satisfy the clipping premise, at every
internal predicate depth (CM62). This does not assert a bound on AST count. -/
theorem prefix_threshold_clamp (p : List A → Bool) (w : List A) (t : ℤ) :
    t ≤ (prefixCount p w : ℤ) ↔
      max 0 (min t ((w.length : ℤ) + 1)) ≤ prefixCount p w :=
  count_threshold_clamp _ _ (prefixCount_le_length p w) t

/-- Exact threshold when the indexed bank contains left counts (CM62).
The inequality direction includes the strict-comparison integer correction. -/
theorem comparison_left_threshold (a b e k : ℤ) (wanted : Bool) :
    decide (a < b + e + k) = wanted ↔
      if wanted then a ≤ b + e + k - 1 else b + e + k ≤ a := by
  cases wanted with
  | false =>
    change decide (a < b + e + k) = false ↔ b + e + k ≤ a
    rw [decide_eq_false_iff_not]
    omega
  | true =>
    change decide (a < b + e + k) = true ↔ a ≤ b + e + k - 1
    rw [decide_eq_true_eq]
    omega

/-- Exact threshold when the indexed bank contains right counts (CM62).
This packages CM59's two equivalences for the same Boolean query interface. -/
theorem comparison_right_threshold (a b e k : ℤ) (wanted : Bool) :
    decide (a < b + e + k) = wanted ↔
      if wanted then a - e - k + 1 ≤ b else b ≤ a - e - k := by
  cases wanted with
  | false =>
    change decide (a < b + e + k) = false ↔ b ≤ a - e - k
    rw [decide_eq_false_iff_not]
    exact comparison_right_upper a b e k
  | true =>
    change decide (a < b + e + k) = true ↔ a - e - k + 1 ≤ b
    rw [decide_eq_true_eq]
    exact comparison_right_lower a b e k

/-- Intersecting all left-bank thresholds is equivalent to the actual
observed inclusive-count fit, not only a necessary range test (CM62). -/
theorem countFit_left_bounds (observed : Finset (List A)) (labels : List A → Bool)
    (left right extra : List A → Bool) (offset : ℤ) :
    CountFit observed labels left right extra offset ↔
      ∀ w ∈ observed, if labels w then
        (prefixCount left w : ℤ) ≤ prefixCount right w + prefixCount extra w + offset - 1
      else (prefixCount right w : ℤ) + prefixCount extra w + offset ≤ prefixCount left w := by
  constructor
  · intro h w hw
    exact (comparison_left_threshold _ _ _ _ _).mp (h w hw)
  · intro h w hw
    exact (comparison_left_threshold _ _ _ _ _).mpr (h w hw)

/-- The right-bank index has the same exact fit semantics (CM62). Every
known prefix is intersected; a heuristic subset is not substituted. -/
theorem countFit_right_bounds (observed : Finset (List A)) (labels : List A → Bool)
    (left right extra : List A → Bool) (offset : ℤ) :
    CountFit observed labels left right extra offset ↔
      ∀ w ∈ observed, if labels w then
        (prefixCount left w : ℤ) - prefixCount extra w - offset + 1 ≤ prefixCount right w
      else (prefixCount right w : ℤ) ≤ prefixCount left w - prefixCount extra w - offset := by
  constructor
  · intro h w hw
    exact (comparison_right_threshold _ _ _ _ _).mp (h w hw)
  · intro h w hw
    exact (comparison_right_threshold _ _ _ _ _).mpr (h w hw)

/-- Streaming every right candidate preserves existence of a fit with the
fixed left anchor bank (CM62). The union must cover the actual query stream;
this theorem does not assert that a truncated anchor contains a target. -/
theorem stream_right_candidates [DecidableEq J]
    (anchors : Finset I) (stages : Finset K) (batch : K → Finset J)
    (observed : Finset (List A)) (labels : List A → Bool)
    (left : I → List A → Bool) (right : J → List A → Bool)
    (extra : List A → Bool) (offset : ℤ) :
    (∃ a ∈ anchors, ∃ b ∈ stages.biUnion batch,
      CountFit observed labels (left a) (right b) extra offset) ↔
    ∃ s ∈ stages, ∃ a ∈ anchors, ∃ b ∈ batch s,
      CountFit observed labels (left a) (right b) extra offset := by
  constructor
  · rintro ⟨a, ha, b, hb, hfit⟩
    obtain ⟨s, hs, hb⟩ := Finset.mem_biUnion.mp hb
    exact ⟨s, hs, a, ha, b, hb, hfit⟩
  · rintro ⟨s, hs, a, ha, b, hb, hfit⟩
    exact ⟨a, ha, b, Finset.mem_biUnion.mpr ⟨s, hs, hb⟩, hfit⟩

/-- The dual streaming orientation keeps a fixed right anchor bank (CM62).
Chunks may overlap; they are search candidates, not deduplicated CE records. -/
theorem stream_left_candidates [DecidableEq I]
    (anchors : Finset J) (stages : Finset K) (batch : K → Finset I)
    (observed : Finset (List A)) (labels : List A → Bool)
    (left : I → List A → Bool) (right : J → List A → Bool)
    (extra : List A → Bool) (offset : ℤ) :
    (∃ a ∈ stages.biUnion batch, ∃ b ∈ anchors,
      CountFit observed labels (left a) (right b) extra offset) ↔
    ∃ s ∈ stages, ∃ a ∈ batch s, ∃ b ∈ anchors,
      CountFit observed labels (left a) (right b) extra offset := by
  constructor
  · rintro ⟨a, ha, b, hb, hfit⟩
    obtain ⟨s, hs, ha⟩ := Finset.mem_biUnion.mp ha
    exact ⟨s, hs, a, ha, b, hb, hfit⟩
  · rintro ⟨s, hs, a, ha, b, hb, hfit⟩
    exact ⟨a, Finset.mem_biUnion.mpr ⟨s, hs, ha⟩, b, hb, hfit⟩

example : decide ((-2 : ℤ) < -3 + 2 + 0) = true ↔ (-2 : ℤ) ≤ -3 + 2 + 0 - 1 :=
  comparison_left_threshold (-2) (-3) 2 0 true

example : decide ((-2 : ℤ) < -3 + 2 + 0) = true ↔ (-2 : ℤ) - 2 - 0 + 1 ≤ -3 :=
  comparison_right_threshold (-2) (-3) 2 0 true

example : (-7 : ℤ) ≤ (2 : ℕ) ↔ max 0 (min (-7) ((3 : ℤ) + 1)) ≤ (2 : ℕ) :=
  count_threshold_clamp 2 3 (by decide) (-7)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Streaming
