module

public import Transformer.Grokking.Convex.ConvexDepth.ObservedRows.Geometry

/-!
# Exact reduced projection for an arbitrary full proposal

Source: `review/convex/moments.md`, CM66. An unconstrained gradient proposal
need not satisfy the observed anchors. Completing its known-row squares gives
a weighted reduced target plus a constant independent of the optimized point.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ObservedRows

open scoped BigOperators
open LatentPrograms (Parameters)

variable {J S C : Type*} [Fintype J] [Fintype S] [Fintype C] [DecidableEq C]

/-- The original mass coordinate and all its anchored copies, CM66. -/
def massWeight (labels : J → S → Option C) (j : J) : ℝ :=
  1 + Fintype.card (Known labels j)

/-- Select the proposal coordinate at a recorded class, CM66. No selected
value at an unknown row is used in the known-state sum. -/
def anchorProposal (labels : J → S → Option C) (v : Parameters J S C) (j : J) (s : S) : ℝ :=
  match labels j s with
  | none => 0
  | some y => v.2 j s y

/-- Linear coefficient of the full projection objective after eliminating
known rows, CM66. Off-label proposal coordinates enter only the constant term. -/
def massLinear (labels : J → S → Option C) (v : Parameters J S C) (j : J) : ℝ :=
  v.1 j + ∑ s : Known labels j, anchorProposal labels v j s.1

/-- The exact reduced proposal keeps every unknown raw-class entry and
averages the original mass with its known-label proposal coordinates, CM66. -/
noncomputable def projectionTarget (labels : J → S → Option C) (v : Parameters J S C) : Reduced labels :=
  (fun j => massLinear labels v j / massWeight labels j, fun j s c => v.2 j s.1 c)

/-- The actual induced weighted distance, CM66; the complete joint
Euclidean projection is not the unweighted stored-coordinate projection. -/
def weightedDistance (labels : J → S → Option C) (a b : Reduced labels) : ℝ :=
  ∑ j, (massWeight labels j * (a.1 j - b.1 j) ^ 2 +
    ∑ s : Unknown labels j, ∑ c, (a.2 j s c - b.2 j s c) ^ 2)

/-- All eliminated proposal squares, including off-label entries, CM66.
This remainder is independent of the point being optimized. -/
noncomputable def projectionRemainder (labels : J → S → Option C) (v : Parameters J S C) : ℝ :=
  ∑ j, (v.1 j ^ 2 + (∑ s : Known labels j, ∑ c, v.2 j s.1 c ^ 2) -
    massLinear labels v j ^ 2 / massWeight labels j)

omit [Fintype J] [Fintype C] [DecidableEq C] in
/-- Every mass has at least its own coordinate, CM66; completing its
quadratic square never divides by zero, even without an observed state. -/
theorem massWeight_pos (labels : J → S → Option C) (j : J) : 0 < massWeight labels j := by
  unfold massWeight
  positivity

omit [Fintype J] [Fintype S] in
/-- A single known row's distance to an arbitrary proposal, CM66. The
proposal may have nonzero coordinates at every other raw class. -/
theorem known_proposal_distance (labels : J → S → Option C) (r : Reduced labels)
    (v : Parameters J S C) (j : J) (s : Known labels j) :
    (∑ c, ((expand labels r).2 j s.1 c - v.2 j s.1 c) ^ 2) =
      r.1 j ^ 2 - 2 * r.1 j * anchorProposal labels v j s.1 + ∑ c, v.2 j s.1 c ^ 2 := by
  cases h : labels j s.1 with
  | none => exact False.elim (s.2 h)
  | some y =>
    rw [anchorProposal, h]
    simp_rw [expand_known labels r j s.1 y h]
    calc
      _ = ∑ c, ((if c = y then r.1 j ^ 2 - 2 * r.1 j * v.2 j s.1 c else 0) +
          v.2 j s.1 c ^ 2) := by
        apply Finset.sum_congr rfl
        intro c hc
        split_ifs <;> ring
      _ = _ := by rw [Finset.sum_add_distrib]; simp

omit [Fintype J] in
/-- The known decoder rows have exactly their counted quadratic weight
and selected proposal sum, CM66; no diagonal or raw class is dropped. -/
theorem known_proposal_sum (labels : J → S → Option C) (r : Reduced labels)
    (v : Parameters J S C) (j : J) :
    (∑ s : Known labels j, ∑ c, ((expand labels r).2 j s.1 c - v.2 j s.1 c) ^ 2) =
      (Fintype.card (Known labels j) : ℝ) * r.1 j ^ 2 -
        2 * r.1 j * (∑ s : Known labels j, anchorProposal labels v j s.1) +
          ∑ s : Known labels j, ∑ c, v.2 j s.1 c ^ 2 := by
  simp_rw [known_proposal_distance]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp

/-- Exact completion of all projection squares, CM66, for every full
proposal and every reduced point. No feasibility or gradient assumption is used. -/
theorem projection_distance (labels : J → S → Option C) (v : Parameters J S C)
    (r : Reduced labels) :
    fullSquaredDistance (expand labels r) v =
      weightedDistance labels r (projectionTarget labels v) + projectionRemainder labels v := by
  unfold fullSquaredDistance weightedDistance projectionRemainder
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j hj
  rw [← Fintype.sum_subtype_add_sum_subtype (fun s => labels j s = none)]
  rw [known_proposal_sum]
  have hu : (∑ s : Unknown labels j, ∑ c,
      ((expand labels r).2 j s.1 c - v.2 j s.1 c) ^ 2) =
      ∑ s : Unknown labels j, ∑ c, (r.2 j s c - v.2 j s.1 c) ^ 2 := by
    simp_rw [expand_unknown]
  rw [hu]
  have hw := ne_of_gt (massWeight_pos labels j)
  change (r.1 j - v.1 j) ^ 2 + _ = _
  dsimp only [projectionTarget]
  unfold massLinear
  have hw' : (1 + (Fintype.card (Known labels j) : ℝ)) ≠ 0 := hw
  unfold massWeight
  field_simp
  ring

/-- The parameter-independent remainder cancels from every pairwise
projection comparison, CM66. Weighted and full objectives have the same order. -/
theorem projection_comparison (labels : J → S → Option C) (v : Parameters J S C)
    (a b : Reduced labels) :
    fullSquaredDistance (expand labels a) v ≤ fullSquaredDistance (expand labels b) v ↔
      weightedDistance labels a (projectionTarget labels v) ≤
        weightedDistance labels b (projectionTarget labels v) := by
  rw [projection_distance, projection_distance, add_le_add_iff_right]

variable {X : Type*} [DecidableEq S]

/-- Exact minimizers correspond on the complete original and reduced
feasible domains, CM66. Membership is part of both sides of this equivalence. -/
theorem projection_minimizer_iff (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) (v : Parameters J S C)
    (r : Reduced (dictionary program samples)) :
    (expand (dictionary program samples) r ∈ BoundedLatent.domain program order samples δ R ∧
      ∀ q ∈ BoundedLatent.domain program order samples δ R,
        fullSquaredDistance (expand (dictionary program samples) r) v ≤ fullSquaredDistance q v) ↔
    (r ∈ domain (dictionary program samples) (LatentPrograms.retained program order samples) δ R ∧
      ∀ s ∈ domain (dictionary program samples) (LatentPrograms.retained program order samples) δ R,
        weightedDistance (dictionary program samples) r (projectionTarget (dictionary program samples) v) ≤
          weightedDistance (dictionary program samples) s (projectionTarget (dictionary program samples) v)) := by
  rw [expand_mem_iff]
  constructor
  · rintro ⟨hr, hmin⟩
    refine ⟨hr, fun s hs => ?_⟩
    exact (projection_comparison _ v r s).mp
      (hmin _ (expand_mem program order samples hs))
  · rintro ⟨hr, hmin⟩
    refine ⟨hr, fun q hq => ?_⟩
    have h := (projection_comparison _ v r (restrict (dictionary program samples) q)).mpr
      (hmin _ (restrict_mem program order samples hq))
    rwa [expand_restrict program order samples hq] at h

end Transformer.Grokking.Convex.ConvexDepth.ObservedRows
