import Transformer.FrankWolfe.Section5_Metastability
import Mathlib.Analysis.Convex.Strict.Extreme
import Mathlib.Analysis.InnerProductSpace.Convex

/-!
# A genuine acute segment for the first-phase clustering audit

ArXiv:2508.09628v1, §5, `lem: first.phase`, assumptions i)–iii).
The actual one-dimensional unit ball is the segment with endpoints
minus one and one. These are exactly its extreme points, have equal
norm, and satisfy the original separation condition with `c₀ = 1`.
At either vertex all nonzero directions into the segment have angle
zero, so the original uniform acute-cone condition holds as well.

The scalar embedding below is the actual coordinate embedding into
`EucSpace 1`. Its norm, inner product and algebra are computed directly.
Every vector in that space is recovered from its sole coordinate.
The actual convex-hull identity and extreme-point property are proved
for the segment itself. These facts supply genuine geometric inputs
for checking the printed minimum-distance clustering time.
-/

open Real MeasureTheory Set
open scoped BigOperators
namespace Transformer.FrankWolfe

/-- The actual scalar-coordinate embedding for the segment construction.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
noncomputable def firstPhasePoint (r : ℝ) : EucSpace 1 := EuclideanSpace.single (0 : Fin 1) r

/-- The actual embedded scalar is its sole coordinate.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
@[simp] theorem firstPhasePoint_apply (r : ℝ) (i : Fin 1) : firstPhasePoint r i = r := by
  have hi : i = 0 := Subsingleton.elim _ _
  subst i
  simp [firstPhasePoint]

/-- Every actual one-dimensional vector is recovered from its coordinate.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem firstPhasePoint_coord (x : EucSpace 1) : firstPhasePoint (x 0) = x := by
  ext i
  have hi : i = 0 := Subsingleton.elim _ _
  simp [hi]

/-- The actual norm is the absolute value of the original scalar.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
@[simp] theorem norm_firstPhasePoint (r : ℝ) : ‖firstPhasePoint r‖ = |r| := by
  simp [firstPhasePoint, PiLp.norm_single]

/-- The actual Euclidean inner product is the product of the scalars.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
@[simp] theorem inner_firstPhasePoint (r s : ℝ) :
    inner (𝕜 := ℝ) (firstPhasePoint r) (firstPhasePoint s) = r * s := by
  simp [firstPhasePoint, EuclideanSpace.inner_single_left]

/-- The actual scalar embedding preserves subtraction.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem firstPhasePoint_sub (r s : ℝ) : firstPhasePoint r - firstPhasePoint s = firstPhasePoint (r - s) := by
  ext i
  simp

/-- The actual scalar embedding preserves real scaling.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem firstPhasePoint_smul (r s : ℝ) : r • firstPhasePoint s = firstPhasePoint (r * s) := by
  ext i
  simp

/-- The actual unit segment, represented by its genuine closed ball.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
noncomputable def firstPhaseSegment : Set (EucSpace 1) := Metric.closedBall 0 1

/-- The two actual extreme endpoints of the segment.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
noncomputable def firstPhaseVertices : Idx 2 → EucSpace 1 := ![firstPhasePoint (-1), firstPhasePoint 1]

/-- The original segment condition is the actual scalar bound.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
@[simp] theorem firstPhasePoint_mem_segment (r : ℝ) :
    firstPhasePoint r ∈ firstPhaseSegment ↔ |r| ≤ 1 := by
  simp [firstPhaseSegment]

/-- Actual segment membership bounds its original coordinate.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem firstPhaseSegment_coord {x : EucSpace 1} (hx : x ∈ firstPhaseSegment) :
    -1 ≤ x 0 ∧ x 0 ≤ 1 := by
  rw [← firstPhasePoint_coord x, firstPhasePoint_mem_segment, abs_le] at hx
  simpa using hx

/-- The actual unit segment is exactly the convex hull of the two endpoints.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem firstPhaseSegment_hull :
    firstPhaseSegment = convexHull ℝ (range firstPhaseVertices) := by
  apply Set.Subset.antisymm
  · intro x hx
    have hcoord := firstPhaseSegment_coord hx
    have hleft := subset_convexHull ℝ (range firstPhaseVertices) (mem_range_self (0 : Idx 2))
    have hright := subset_convexHull ℝ (range firstPhaseVertices) (mem_range_self (1 : Idx 2))
    have hc := (convex_convexHull ℝ (range firstPhaseVertices)) hleft hright
      (show 0 ≤ (1 - x 0) / 2 by linarith) (show 0 ≤ (1 + x 0) / 2 by linarith)
      (show (1 - x 0) / 2 + (1 + x 0) / 2 = 1 by ring)
    convert hc using 1
    ext i
    have hi : i = 0 := Subsingleton.elim _ _
    simp [firstPhaseVertices, hi]
    ring
  · apply convexHull_min
    · rintro x ⟨i, rfl⟩
      fin_cases i <;> simp [firstPhaseVertices]
    · exact convex_closedBall _ _

/-- The original vertex-list predicate holds for the actual two endpoints.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem isVertexList_firstPhaseSegment : IsVertexList firstPhaseSegment firstPhaseVertices := by
  constructor
  · intro i j hij
    fin_cases i <;> fin_cases j <;> try rfl
    all_goals
      have heq := congrArg (fun z : EucSpace 1 => z 0) hij
      norm_num [firstPhaseVertices] at heq
  · apply Set.Subset.antisymm
    · rintro x ⟨i, rfl⟩
      apply StrictConvexSpace.sphere_subset_extremePoints_closedBall 0 (by norm_num : (1 : ℝ) ≠ 0)
      fin_cases i <;> simp [firstPhaseVertices]
    · rw [firstPhaseSegment_hull]
      exact extremePoints_convexHull_subset

/-- Every original polytope hypothesis holds for this actual acute segment.
Source: arXiv:2508.09628v1, §5, `lem: first.phase`, geometric audit. -/
theorem conePolytope_firstPhaseSegment : ConePolytope firstPhaseSegment firstPhaseVertices 1 := by
  refine ⟨firstPhaseSegment_hull, isVertexList_firstPhaseSegment, ?_, ?_, one_pos, ?_⟩
  · intro i
    refine ⟨0, by linarith [Real.pi_pos], ?_⟩
    intro x hx y hy hxi hyi
    have hxc := firstPhaseSegment_coord hx
    have hyc := firstPhaseSegment_coord hy
    have hxpoint := firstPhasePoint_coord x
    have hypoint := firstPhasePoint_coord y
    have hnotx (r : ℝ) (he : firstPhaseVertices i = firstPhasePoint r) : x 0 ≠ r := by
      intro hc
      apply hxi
      rw [he, ← hxpoint, hc]
    have hnoty (r : ℝ) (he : firstPhaseVertices i = firstPhasePoint r) : y 0 ≠ r := by
      intro hc
      apply hyi
      rw [he, ← hypoint, hc]
    rw [← hxpoint, ← hypoint]
    fin_cases i
    · change InnerProductGeometry.angle (firstPhasePoint (x 0) - firstPhasePoint (-1))
        (firstPhasePoint (y 0) - firstPhasePoint (-1)) ≤ 0
      rw [firstPhasePoint_sub, firstPhasePoint_sub]
      have hxpos : 0 < x 0 - -1 := sub_pos.mpr (lt_of_le_of_ne hxc.1 (hnotx (-1) rfl).symm)
      have hypos : 0 < y 0 - -1 := sub_pos.mpr (lt_of_le_of_ne hyc.1 (hnoty (-1) rfl).symm)
      rw [← mul_one (x 0 - -1), ← mul_one (y 0 - -1), ← firstPhasePoint_smul, ← firstPhasePoint_smul,
        InnerProductGeometry.angle_smul_left_of_pos _ _ hxpos,
        InnerProductGeometry.angle_smul_right_of_pos _ _ hypos,
        InnerProductGeometry.angle_self (by intro h; have := congrArg (fun z : EucSpace 1 => z 0) h; simp at this)]
    · change InnerProductGeometry.angle (firstPhasePoint (x 0) - firstPhasePoint 1)
        (firstPhasePoint (y 0) - firstPhasePoint 1) ≤ 0
      rw [firstPhasePoint_sub, firstPhasePoint_sub]
      have hxneg : x 0 - 1 < 0 := sub_neg.mpr (lt_of_le_of_ne hxc.2 (hnotx 1 rfl))
      have hyneg : y 0 - 1 < 0 := sub_neg.mpr (lt_of_le_of_ne hyc.2 (hnoty 1 rfl))
      rw [← mul_one (x 0 - 1), ← mul_one (y 0 - 1), ← firstPhasePoint_smul, ← firstPhasePoint_smul,
        InnerProductGeometry.angle_smul_left_of_neg _ _ hxneg,
        InnerProductGeometry.angle_smul_right_of_neg _ _ hyneg,
        InnerProductGeometry.angle_self (by intro h; have := congrArg (fun z : EucSpace 1 => z 0) h; simp at this)]
  · intro i j
    fin_cases i <;> fin_cases j <;> simp [firstPhaseVertices]
  · intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [firstPhaseVertices]

/-- A genuine interior point and the complete actual acute polytope jointly
realize the membership and original geometric hypotheses of this calculation. -/
example : firstPhasePoint 0 ∈ firstPhaseSegment ∧
    ConePolytope firstPhaseSegment firstPhaseVertices 1 :=
  ⟨by simp, conePolytope_firstPhaseSegment⟩

end Transformer.FrankWolfe
