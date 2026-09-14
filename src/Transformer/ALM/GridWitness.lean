/-
# The whole run's verdict, and that merging the heads does not soften it

`Transformer.ALM.GuardSep` says what one query earns by passing
`grid.rs::off_the_grid`.  The runtime does not ask about one query: every head
carries a `GridWitness`, `observe` folds each query into it, `merge` combines
the per-head records into the run's, and the report is `is_clean`, which reads
one field — `count == 0`.  Between "each query passed" and "the run is clean"
sit a fold and a merge, and neither had been said to preserve anything.

They do, and the two statements are the file's point.  `clean_merge` — a merged
witness is clean exactly when both of its halves are, so a dirty query in one
head cannot be averaged away by a clean one in another.  `clean_iff_worst_le_one`
— the counter and the high-water mark of `grid_ratio` threshold the same thing,
so reporting the ratio instead of the verdict, which is what `grid_ratio`'s
docstring claims makes the cost of the query scale visible, loses no
information at all.

And the margin of safety is readable off the same number:
`strict_guard_of_worst_lt_one` turns a run whose worst ratio is below `1` into
the strict guard at each of its queries, `binade_of_worst_le_half` into the
factor of two the docstring calls "one binade from the wall", and
`run_retrieval_of_worst_lt_one` runs that into the answer through
`Transformer.ALM.ScoreGap`.

Source: `todo3.md` §4 and §4a; `vm-rs/alm-hull/src/grid.rs`, `grid_ratio`,
`GridWitness::observe`, `merge` and `is_clean`.
-/

import Transformer.ALM.ScoreGap

namespace Transformer
namespace ALM

/-! ### One query, as `observe` sees it -/

/-- A query whose spacing `u` at the winning score exceeds the margin `m` it
had to beat: `off_the_grid(score, margin)`, with the zero margin excluded
because it is an exact tie and not a crossing. -/
def OffGrid (u m : ℝ) : Prop := m ≠ 0 ∧ |m| < u

/-- How much of the available separation the query used, `grid_ratio`. -/
noncomputable def gridRatio (u m : ℝ) : ℝ := if m = 0 then 0 else u / |m|

lemma gridRatio_nonneg {u m : ℝ} (hu : 0 ≤ u) : 0 ≤ gridRatio u m := by
  unfold gridRatio
  split
  · exact le_refl 0
  · exact div_nonneg hu (abs_nonneg m)

/-- **The ratio carries the verdict.**  `off_the_grid` is `grid_ratio > 1`, on
the nose and including the zero-margin case, which is why the high-water mark
of the ratio can stand in for the counter. -/
theorem offGrid_iff_one_lt_gridRatio (u m : ℝ) : OffGrid u m ↔ 1 < gridRatio u m := by
  unfold OffGrid gridRatio
  by_cases hm : m = 0
  · rw [if_pos hm]
    constructor
    · rintro ⟨h, -⟩
      exact absurd hm h
    · intro h
      exact absurd h (by norm_num)
  · rw [if_neg hm, lt_div_iff₀ (abs_pos.mpr hm), one_mul]
    exact ⟨fun h => h.2, fun h => ⟨hm, h⟩⟩

/-! ### The fold `observe` performs -/

lemma foldr_max_nonneg (l : List ℝ) : 0 ≤ l.foldr max 0 := by
  induction l with
  | nil => exact le_refl 0
  | cons a t ih => exact le_trans ih (le_max_right _ _)

lemma foldr_max_append (a b : List ℝ) :
    (a ++ b).foldr max 0 = max (a.foldr max 0) (b.foldr max 0) := by
  induction a with
  | nil => simp [max_eq_right (foldr_max_nonneg b)]
  | cons x t ih =>
      simp only [List.cons_append, List.foldr_cons, ih]
      rw [max_assoc]

lemma le_foldr_max {l : List ℝ} {x : ℝ} (hx : x ∈ l) : x ≤ l.foldr max 0 := by
  induction l with
  | nil => exact absurd hx (List.not_mem_nil)
  | cons a t ih =>
      rcases List.mem_cons.mp hx with rfl | ht
      · exact le_max_left _ _
      · exact le_trans (ih ht) (le_max_right _ _)

lemma foldr_max_le {l : List ℝ} {c : ℝ} (hc : 0 ≤ c) (h : ∀ x ∈ l, x ≤ c) :
    l.foldr max 0 ≤ c := by
  induction l with
  | nil => exact hc
  | cons a t ih =>
      exact max_le (h a (List.mem_cons_self ..)) (ih fun x hx => h x (List.mem_cons_of_mem _ hx))

open scoped Classical in
/-- `GridWitness::count` over a run: the queries `off_the_grid` fired on. -/
noncomputable def runCount (l : List (ℝ × ℝ)) : ℕ :=
  l.countP fun p => decide (OffGrid p.1 p.2)

/-- `GridWitness::worst` over a run: the high-water mark of `grid_ratio`,
starting from the zero `Default` gives it. -/
noncomputable def runWorst (l : List (ℝ × ℝ)) : ℝ :=
  (l.map fun p => gridRatio p.1 p.2).foldr max 0

/-! ### What `merge` preserves -/

/-- The counter is additive over the merge, which is `List.countP` and nothing
more. -/
theorem runCount_append (l₁ l₂ : List (ℝ × ℝ)) :
    runCount (l₁ ++ l₂) = runCount l₁ + runCount l₂ := by
  unfold runCount
  exact List.countP_append

/-- And the high-water mark is a maximum over it. -/
theorem runWorst_append (l₁ l₂ : List (ℝ × ℝ)) :
    runWorst (l₁ ++ l₂) = max (runWorst l₁) (runWorst l₂) := by
  unfold runWorst
  rw [List.map_append, foldr_max_append]

/-- **A merged witness is clean exactly when its halves are.**  One head's
crossing cannot be absorbed by another head's clean run: `is_clean` on the
merge is `is_clean` on both. -/
theorem clean_merge (l₁ l₂ : List (ℝ × ℝ)) :
    runCount (l₁ ++ l₂) = 0 ↔ runCount l₁ = 0 ∧ runCount l₂ = 0 := by
  rw [runCount_append]
  omega

/-! ### What `is_clean` says about each query -/

/-- The run's verdict is the conjunction of the per-query ones, so
`Transformer.ALM.GuardSep` applies at every query of a clean run. -/
theorem clean_iff_forall (l : List (ℝ × ℝ)) :
    runCount l = 0 ↔ ∀ p ∈ l, ¬ OffGrid p.1 p.2 := by
  classical
  simp [runCount, List.countP_eq_zero]

/-- **The counter and the high-water mark threshold the same thing.**  So
`grid_ratio`'s claim — that reporting the ratio rather than the verdict makes
the cost of the query scale visible — costs nothing: the verdict is recoverable
from the ratio, and the distance to the wall is not recoverable from the
verdict. -/
theorem clean_iff_worst_le_one (l : List (ℝ × ℝ)) :
    runCount l = 0 ↔ runWorst l ≤ 1 := by
  rw [clean_iff_forall]
  constructor
  · intro h
    refine foldr_max_le (by norm_num) fun x hx => ?_
    obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hx
    exact not_lt.mp fun hlt => h q hq ((offGrid_iff_one_lt_gridRatio _ _).mpr hlt)
  · intro h q hq hoff
    have hle : gridRatio q.1 q.2 ≤ runWorst l :=
      le_foldr_max (List.mem_map.mpr ⟨q, hq, rfl⟩)
    exact absurd ((offGrid_iff_one_lt_gridRatio _ _).mp hoff) (by linarith)

/-! ### And what the margin of safety buys -/

/-- A run whose worst ratio is strictly below `1` passed the guard *strictly*
at each of its queries, which is the room `cmp_of_guard` asks for and the
shipped comparison does not insist on. -/
theorem strict_guard_of_worst_lt_one {l : List (ℝ × ℝ)} (hw : runWorst l < 1)
    {u m : ℝ} (hmem : (u, m) ∈ l) (hm : m ≠ 0) : u < |m| := by
  have hle : gridRatio u m ≤ runWorst l := le_foldr_max (List.mem_map.mpr ⟨(u, m), hmem, rfl⟩)
  rw [gridRatio, if_neg hm, div_le_iff₀ (abs_pos.mpr hm)] at hle
  nlinarith [abs_pos.mpr hm]

/-- And one whose worst ratio is at most `1/2` is a whole binade from the wall,
which is how `grid_ratio`'s docstring reads the number. -/
theorem binade_of_worst_le_half {l : List (ℝ × ℝ)} (hw : runWorst l ≤ 1 / 2)
    {u m : ℝ} (hmem : (u, m) ∈ l) (hm : m ≠ 0) : 2 * u ≤ |m| := by
  have hle : gridRatio u m ≤ runWorst l := le_foldr_max (List.mem_map.mpr ⟨(u, m), hmem, rfl⟩)
  rw [gridRatio, if_neg hm, div_le_iff₀ (abs_pos.mpr hm)] at hle
  nlinarith [abs_nonneg m]

/-- **From the run's report to a query's answer.**  A run that came within a
factor of two of the wall answers each of its separated queries exactly: the
witness gives the spacing, `Transformer.ALM.ScoreGap` gives the separation, and
`cmp_of_guard` gives the comparison.  This is what `is_clean` is for. -/
theorem run_retrieval_of_worst_lt_one {l : List (ℝ × ℝ)} (hw : runWorst l < 1)
    {p : ℕ} {E : ℤ} {σ best second best' second' δ₁ δ₂ : ℝ}
    (hmem : (ulpOf p E, σ) ∈ l) (hσ : 0 < σ) (hgap : 1 ≤ keyGap σ best second)
    (h₁ : |best' - best| ≤ δ₁) (h₂ : |second' - second| ≤ δ₂)
    (hδ₁ : δ₁ ≤ ulpOf p E / 2) (hδ₂ : δ₂ ≤ ulpOf p E / 2) :
    second' < best' := by
  have hguard := strict_guard_of_worst_lt_one hw hmem (ne_of_gt hσ)
  rw [abs_of_pos hσ] at hguard
  exact retrieval_of_keyGap hσ hguard hgap h₁ h₂ hδ₁ hδ₂

/-! ### The hypotheses are satisfiable -/

/-- A two-query run one binade from the wall, and the crossing that would ruin
it: spacings of `1` against margins of `±4` are clean with a ratio of `1/4`,
while a spacing of `8` against the same margin is not.  These witness
`clean_merge`, `clean_iff_forall`, `clean_iff_worst_le_one`,
`strict_guard_of_worst_lt_one` and `binade_of_worst_le_half`. -/
example : runCount [((1 : ℝ), (4 : ℝ)), (1, -4)] = 0 ∧
    runWorst [((1 : ℝ), (4 : ℝ)), (1, -4)] = 1 / 4 ∧ OffGrid 8 4 := by
  have h4 : |(4 : ℝ)| = 4 := abs_of_pos (by norm_num)
  have hn4 : |(-4 : ℝ)| = 4 := by rw [abs_of_neg (by norm_num)]; norm_num
  refine ⟨?_, ?_, ⟨by norm_num, by rw [h4]; norm_num⟩⟩
  · rw [clean_iff_forall]
    intro q hq
    rcases List.mem_cons.mp hq with rfl | hq'
    · rintro ⟨-, h⟩
      norm_num [h4] at h
    · rcases List.mem_cons.mp hq' with rfl | hq''
      · rintro ⟨-, h⟩
        norm_num [hn4] at h
      · exact absurd hq'' List.not_mem_nil
  · simp only [runWorst, List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, gridRatio,
      if_neg (by norm_num : (4 : ℝ) ≠ 0), if_neg (by norm_num : (-4 : ℝ) ≠ 0), h4, hn4]
    norm_num

/-- And the query the last theorem answers: a spacing of `2^-52` against a unit
margin, a full key step of gap, and a half-ulp of rounding on each score.
These are the hypotheses of `run_retrieval_of_worst_lt_one`. -/
example : ((ulpOf 53 0, (1 : ℝ)) ∈ [(ulpOf 53 0, (1 : ℝ))]) ∧ (0 : ℝ) < 1 ∧
    runWorst [(ulpOf 53 0, (1 : ℝ))] < 1 ∧ 1 ≤ keyGap 1 1 0 := by
  refine ⟨List.mem_singleton_self _, by norm_num, ?_, ?_⟩
  · simp only [runWorst, List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, gridRatio,
      if_neg (by norm_num : (1 : ℝ) ≠ 0), abs_of_pos (by norm_num : (0 : ℝ) < 1), ulpOf]
    norm_num
  · simp only [keyGap]
    norm_num

end ALM
end Transformer
