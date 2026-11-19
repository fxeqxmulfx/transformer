module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows.Domain

/-!
# The complete Euclidean geometry in reduced coordinates

Source: `review/convex/moments.md`, CM65. Removing an anchored row eliminates
one affine copy of its program mass, not its contribution to distance. Exact
equivalence of feasible predictions does not justify changing optimizer geometry.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J S C : Type*} [Fintype J] [Fintype S] [Fintype C] [DecidableEq C]

/-- States with an actual observed dictionary entry, CM65. Each distinct
state counts once, regardless of its multiplicity in the CE observations. -/
abbrev Known (labels : J → S → Option C) (j : J) := {s : S // labels j s ≠ none}

/-- Squared Euclidean distance in every original joint coordinate, CM65. -/
def fullSquaredDistance (a b : Parameters J S C) : ℝ :=
  (∑ j, (a.1 j - b.1 j) ^ 2) + ∑ j, ∑ s, ∑ c, (a.2 j s c - b.2 j s c) ^ 2

/-- Unweighted distance in only the stored reduced coordinates, CM65.
This generally differs from the original Euclidean distance. -/
def reducedSquaredDistance (labels : J → S → Option C) (a b : Reduced labels) : ℝ :=
  (∑ j, (a.1 j - b.1 j) ^ 2) + ∑ j, ∑ s, ∑ c, (a.2 j s c - b.2 j s c) ^ 2

omit [Fintype J] [Fintype S] in
/-- A known row contributes exactly one squared mass difference to the
full geometry, CM65; zero entries at other raw classes add no hidden terms. -/
theorem known_row_distance (labels : J → S → Option C) (a b : Reduced labels)
    (j : J) (s : Known labels j) :
    (∑ c, ((expand labels a).2 j s.1 c - (expand labels b).2 j s.1 c) ^ 2) =
      (a.1 j - b.1 j) ^ 2 := by
  cases h : labels j s.1 with
  | none => exact False.elim (s.2 h)
  | some y =>
    simp_rw [expand_known labels a j s.1 y h, expand_known labels b j s.1 y h]
    calc
      _ = ∑ c, if c = y then (a.1 j - b.1 j) ^ 2 else 0 := by
        apply Finset.sum_congr rfl
        intro c hc
        split_ifs <;> simp
      _ = _ := by simp

omit [Fintype J] in
/-- Partitioning a complete decoder block into unknown and observed states
preserves all squared terms, CM65. This count is over states, not records. -/
theorem decoder_distance (labels : J → S → Option C) (a b : Reduced labels) (j : J) :
    (∑ s, ∑ c, ((expand labels a).2 j s c - (expand labels b).2 j s c) ^ 2) =
      (∑ s : Unknown labels j, ∑ c, (a.2 j s c - b.2 j s c) ^ 2) +
        (Fintype.card (Known labels j) : ℝ) * (a.1 j - b.1 j) ^ 2 := by
  rw [← Fintype.sum_subtype_add_sum_subtype (fun s => labels j s = none)]
  congr 1
  · apply Finset.sum_congr rfl
    intro s hs
    apply Finset.sum_congr rfl
    intro c hc
    rw [expand_unknown, expand_unknown]
  · simp_rw [known_row_distance]
    simp

/-- The original squared distance pulls back to a weighted mass metric
and every unknown raw-class coordinate, CM65. The mass weight is one plus
the number of anchored states; replacing it by one changes the geometry. -/
theorem full_distance_reduced (labels : J → S → Option C) (a b : Reduced labels) :
    fullSquaredDistance (expand labels a) (expand labels b) =
      ∑ j, ((1 + (Fintype.card (Known labels j) : ℝ)) * (a.1 j - b.1 j) ^ 2 +
        ∑ s : Unknown labels j, ∑ c, (a.2 j s c - b.2 j s c) ^ 2) := by
  unfold fullSquaredDistance
  change (∑ j, (a.1 j - b.1 j) ^ 2) +
    (∑ j, ∑ s, ∑ c, ((expand labels a).2 j s c - (expand labels b).2 j s c) ^ 2) = _
  simp_rw [decoder_distance]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- The exact amount omitted by unweighted reduced distance is the sum of
anchored mass-copy distances, CM65. No such term vanishes merely by not storing it. -/
theorem distance_difference (labels : J → S → Option C) (a b : Reduced labels) :
    fullSquaredDistance (expand labels a) (expand labels b) =
      reducedSquaredDistance labels a b +
        ∑ j, (Fintype.card (Known labels j) : ℝ) * (a.1 j - b.1 j) ^ 2 := by
  rw [full_distance_reduced]
  unfold reducedSquaredDistance
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  ring

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedRows
