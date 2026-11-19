module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.UniformRows.Basic

/-!
# Uniform proposals and the complete decoder projection

Source: `review/convex/moments.md`, CM69. Every unknown coordinate remains in
CM65/66's original domain. Replacing an unknown row by its mean is a feasible
comparison point; it improves distance to a uniform proposal by exactly that
row's variance. Hence every exact minimizer has uniform unknown rows under
this explicit proposal hypothesis, including in full original coordinates.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.UniformRows

open scoped BigOperators
open ObservedRows

variable {C : Type*} [Fintype C] [Nonempty C]
variable {J S : Type*} [Fintype J] [Fintype S]

/-- The same masses with each unknown row set to its mean, CM69. This is a
comparison point and generally differs from the original feasible point. -/
noncomputable def balanced (labels : J → S → Option C) (r : Reduced labels) : Reduced labels :=
  (r.1, fun j _s _c => r.1 j / Fintype.card C)

/-- All raw-class entries of each unknown proposal row agree, CM69.
Different rows may have different values, and known proposals are arbitrary. -/
def ClassUniform (labels : J → S → Option C) (v : Reduced labels) : Prop :=
  ∀ j s c d, v.2 j s c = v.2 j s d

/-- All unknown-row squared deviations, with no omitted coordinate, CM69. -/
noncomputable def variance (labels : J → S → Option C) (r : Reduced labels) : ℝ :=
  ∑ j, ∑ s, ∑ c, (r.2 j s c - r.1 j / Fintype.card C) ^ 2

omit [Fintype S] in
/-- Balancing preserves every full reduced constraint and mass bound,
CM69. It does not assert that every feasible point was already balanced. -/
theorem balanced_mem (labels : J → S → Option C) (order : List J) (δ R : ℝ)
    (r : Reduced labels) (hr : r ∈ domain labels order δ R) :
    balanced labels r ∈ domain labels order δ R := by
  refine ⟨hr.1, ?_, ?_, hr.2.2.2.1, hr.2.2.2.2.1, hr.2.2.2.2.2⟩
  · intro j s c
    exact div_nonneg (hr.1 j) (le_of_lt (card_pos (C := C)))
  · intro j s
    exact sum_mean (r.1 j)

omit [Nonempty C] in
/-- The total unknown-coordinate variance is nonnegative, CM69. -/
theorem variance_nonneg (labels : J → S → Option C) (r : Reduced labels) :
    0 ≤ variance labels r := by
  exact Finset.sum_nonneg fun j hj =>
    Finset.sum_nonneg fun s hs => square_remainder_nonneg (r.2 j s) (r.1 j)

omit [Nonempty C] in
/-- Zero total variance identifies every unknown coordinate, CM69. -/
theorem variance_zero_iff (labels : J → S → Option C) (r : Reduced labels) :
    variance labels r = 0 ↔ ∀ j s c, r.2 j s c = r.1 j / Fintype.card C := by
  unfold variance
  rw [Finset.sum_eq_zero_iff_of_nonneg fun j hj =>
    Finset.sum_nonneg fun s hs => square_remainder_nonneg (r.2 j s) (r.1 j)]
  simp only [Finset.mem_univ, forall_const]
  apply forall_congr'
  intro j
  rw [Finset.sum_eq_zero_iff_of_nonneg fun s hs => square_remainder_nonneg (r.2 j s) (r.1 j)]
  simp only [Finset.mem_univ, forall_const]
  apply forall_congr'
  intro s
  exact square_remainder_zero_iff (r.2 j s) (r.1 j)

/-- The complete CM66 weighted objective splits into the balanced
candidate's objective and all variances, CM69, for a uniform proposal only. -/
theorem distance_decomposition (labels : J → S → Option C) (r v : Reduced labels)
    (hr : ∀ j s, ∑ c, r.2 j s c = r.1 j) (hv : ClassUniform labels v) :
    weightedDistance labels r v = weightedDistance labels (balanced labels r) v + variance labels r := by
  let c₀ : C := Classical.choice (inferInstance : Nonempty C)
  have row (j : J) (s : Unknown labels j) :
      (∑ c, (r.2 j s c - v.2 j s c) ^ 2) =
        (∑ c, ((balanced labels r).2 j s c - v.2 j s c) ^ 2) +
          ∑ c, (r.2 j s c - r.1 j / Fintype.card C) ^ 2 := by
    have hu (c : C) := hv j s c c₀
    simp_rw [hu]
    rw [square_decomposition (r.2 j s) (r.1 j) (v.2 j s c₀) (hr j s)]
    simp [balanced, add_comm]
  unfold weightedDistance variance
  simp_rw [row, Finset.sum_add_distrib]
  simp only [balanced]
  ring

/-- Every exact minimizer over the unchanged full reduced domain has
uniform unknown rows when the proposal does, CM69. No existence theorem is claimed. -/
theorem minimizer_uniform (labels : J → S → Option C) (order : List J) (δ R : ℝ)
    (r v : Reduced labels) (hr : r ∈ domain labels order δ R) (hv : ClassUniform labels v)
    (hmin : ∀ q ∈ domain labels order δ R, weightedDistance labels r v ≤ weightedDistance labels q v) :
    ∀ j s c, r.2 j s c = r.1 j / Fintype.card C := by
  have h := hmin (balanced labels r) (balanced_mem labels order δ R r hr)
  rw [distance_decomposition labels r v hr.2.2.1 hv] at h
  apply (variance_zero_iff labels r).mp
  have hn := variance_nonneg labels r
  linarith

omit [Nonempty C] in
/-- The resulting mass-only quadratic retains every anchored-copy weight
and every unknown row's class-count contribution, CM69. -/
theorem balanced_objective (labels : J → S → Option C) (r v : Reduced labels)
    (c₀ : C) (hv : ClassUniform labels v) :
    weightedDistance labels (balanced labels r) v =
      ∑ j, (massWeight labels j * (r.1 j - v.1 j) ^ 2 +
        ∑ s : Unknown labels j, (Fintype.card C : ℝ) *
          (r.1 j / Fintype.card C - v.2 j s c₀) ^ 2) := by
  unfold weightedDistance
  apply Finset.sum_congr rfl
  intro j hj
  congr 1
  apply Finset.sum_congr rfl
  intro s hs
  simp_rw [(fun c => hv j s c c₀)]
  simp [balanced]

omit [Nonempty C] in
/-- CM66's weighted squared distance is nonnegative on arbitrary points,
CM69; feasibility is unnecessary for this fact. -/
theorem weightedDistance_nonneg (labels : J → S → Option C) (r v : Reduced labels) :
    0 ≤ weightedDistance labels r v := by
  apply Finset.sum_nonneg
  intro j hj
  exact add_nonneg (mul_nonneg (le_of_lt (massWeight_pos labels j)) (sq_nonneg _))
    (Finset.sum_nonneg fun s hs => Finset.sum_nonneg fun c hc => sq_nonneg _)

variable {X : Type*} [DecidableEq S] [DecidableEq C]

/-- The same conclusion in every original joint coordinate, CM69. The
full proposal may violate anchors and mass constraints. Only its unknown
raw-class rows must be uniform; the feasible set remains the whole CM64 set. -/
theorem full_minimizer_uniform (program : J → X → S) (order : List J)
    (samples : List (X × C)) (δ R : ℝ) (q v : LatentPrograms.Parameters J S C)
    (hq : q ∈ BoundedLatent.domain program order samples δ R)
    (hv : ∀ j (s : Unknown (dictionary program samples) j) c d,
      v.2 j s.1 c = v.2 j s.1 d)
    (hmin : ∀ t ∈ BoundedLatent.domain program order samples δ R,
      fullSquaredDistance q v ≤ fullSquaredDistance t v) :
    ∀ j (s : Unknown (dictionary program samples) j) c,
      q.2 j s.1 c = q.1 j / Fintype.card C := by
  let labels := dictionary program samples
  let r := restrict labels q
  have he : expand labels r = q := expand_restrict program order samples hq
  have hh : expand labels r ∈ BoundedLatent.domain program order samples δ R ∧
      ∀ t ∈ BoundedLatent.domain program order samples δ R,
        fullSquaredDistance (expand labels r) v ≤ fullSquaredDistance t v := by
    rw [he]
    exact ⟨hq, hmin⟩
  have hm := (projection_minimizer_iff program order samples δ R v r).mp hh
  have hu : ClassUniform labels (projectionTarget labels v) := by
    intro j s c d
    exact hv j s c d
  exact minimizer_uniform labels (LatentPrograms.retained program order samples) δ R
    r (projectionTarget labels v) hm.1 hu hm.2

end Transformer.Grokking.FlatSpectrum.ConvexDepth.UniformRows
