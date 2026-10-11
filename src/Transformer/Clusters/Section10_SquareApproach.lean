/-
# A genuine attention trajectory approaches its limiting hull from outside

ArXiv:2305.05465v6, §10, Claim `cl:gamma'12` and Step 2'.
The source defines `Sδ` by restricting a distance neighborhood to the
limiting polytope `K`. An actual token may never belong to `K` at finite
time, even though it converges to a vertex. Its energy and every competing
score can then decrease, contrary to the two proposed positive bounds.

Use the actual square solution with `Q = K = V = I₂`, already verified
directly from the exponential attention weights. Its normalized amplitude
strictly decreases. A corner lies outside every later square hull and thus
outside their intersection. The projected limit below is nonempty and
equals that intersection; no empty-set Hausdorff convention is used.
-/

import Transformer.Clusters.Section4_SquareDynamics
import Transformer.Clusters.Section10_ProjectedHullLimit
import Transformer.Clusters.Section10_Remainder
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

/-- The normalized square has strictly negative amplitude derivative.
Source: arXiv:2305.05465v6, §10, the genuine square test of Step 2'. -/
theorem squareAmplitude_derivative_neg (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) :
    Real.exp (-t) * (squareScalarDrift (u t) - u t) < 0 := by
  have hut := squareScalar_solution_pos u hu h0 ht
  have hlt : squareScalarDrift (u t) < u t := by
    unfold squareScalarDrift
    apply (div_lt_iff₀ (squareScalar_denominator_pos (u t))).mpr
    nlinarith [mul_pos hut (Real.exp_pos (-(u t) ^ 2))]
  exact mul_neg_of_pos_of_neg (Real.exp_pos _) (sub_neg.mpr hlt)

example : ∃ u : ℝ → ℝ, (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧
    u 0 = 2 ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu, h0, zero_le_one⟩

/-- Strict decrease holds at every pair of distinct future times.
Source: arXiv:2305.05465v6, §10, the square test of the restricted `Sδ`. -/
theorem strictAntiOn_squareAmplitude (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2) :
    StrictAntiOn (squareAmplitude u) (Set.Ici 0) := by
  apply strictAntiOn_of_hasDerivWithinAt_neg (convex_Ici (0 : ℝ))
    (continuous_squareAmplitude u hu).continuousOn
    (fun t _ => (hasDerivAt_squareAmplitude (hu t)).hasDerivWithinAt)
  exact fun t ht => squareAmplitude_derivative_neg u hu h0 (interior_subset ht)

example : ∃ u : ℝ → ℝ, (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧
    u 0 = 2 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu, h0⟩

/-- Every point in the square hull has first coordinate at most its
amplitude. Source: arXiv:2305.05465v6, §10, the actual limiting hull. -/
theorem first_coord_le_of_mem_square_hull (r : ℝ) (hr : 0 ≤ r) (x : EucSpace 2)
    (hx : x ∈ tokenHull (squareCorner r)) : x 0 ≤ r := by
  have hconv : Convex ℝ {y : EucSpace 2 | y 0 ≤ r} := by
    rintro a ha b hb c e hc he hce
    change c * a 0 + e * b 0 ≤ r
    change a 0 ≤ r at ha
    change b 0 ≤ r at hb
    nlinarith [mul_le_mul_of_nonneg_left ha hc, mul_le_mul_of_nonneg_left hb he]
  apply convexHull_min ?_ hconv hx
  rintro _ ⟨i, rfl⟩
  change squareCorner r i 0 ≤ r
  rw [squareCorner_apply_zero]
  fin_cases i <;> norm_num [squareSignX]
  all_goals linarith

example : (0 : ℝ) ≤ 2 ∧ squareCorner 2 3 ∈ tokenHull (squareCorner 2) :=
  ⟨by norm_num, mem_tokenHull _ 3⟩

/-- The positive corner is outside the actual future-hull intersection
at every finite future time. Source: arXiv:2305.05465v6, §10, Step 2'. -/
theorem squareCorner_not_mem_future_tokenHull (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) : squareCorner (squareAmplitude u t) 3 ∉
      {x | ∀ s : ℝ, 0 ≤ s → x ∈ tokenHull (squareCorner (squareAmplitude u s))} := by
  intro hmem
  have hs : 0 ≤ t + 1 := by linarith
  have hlt := strictAntiOn_squareAmplitude u hu h0 ht hs (by linarith : t < t + 1)
  have hcoord := first_coord_le_of_mem_square_hull (squareAmplitude u (t + 1))
    (by linarith [one_le_squareAmplitude u hu h0 hs]) _ (hmem (t + 1) hs)
  norm_num [squareCorner_apply_zero, squareSignX] at hcoord
  linarith

example : ∃ u : ℝ → ℝ, (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧
    u 0 = 2 ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu, h0, zero_le_one⟩

/-- Every competing score at a square corner is nonpositive.
Source: arXiv:2305.05465v6, §10, `a_j` evaluated on the actual square. -/
theorem projScore_square_nonpos (r : ℝ) (i j : Idx 4) :
    projScore (1 : ParamMatrix 2) 1 (squareCorner r) i j ≤ 0 := by
  simp only [projScore, one_apply_eq_self, inner_sub_right, inner_squareCorner_same]
  fin_cases i <;> fin_cases j <;> norm_num [squareSignX, squareSignY]
  all_goals nlinarith [sq_nonneg r]

/-- The actual square energy has a nonpositive derivative at every future
time. Source: arXiv:2305.05465v6, §10, the quantity bounded in Step 2'. -/
theorem hasDerivAt_norm_sq_square_nonpos (u : ℝ → ℝ)
    (hu : ∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) (h0 : u 0 = 2)
    {t : ℝ} (ht : 0 ≤ t) (i : Idx 4) :
    ∃ D : ℝ, HasDerivAt (fun s => ‖squareCorner (squareAmplitude u s) i‖ ^ 2) D t ∧
      D ≤ 0 := by
  have hfun : (fun s => ‖squareCorner (squareAmplitude u s) i‖ ^ 2) =
      fun s => 2 * squareAmplitude u s ^ 2 := by
    funext s
    exact norm_squareCorner_sq _ _
  let r' := Real.exp (-t) * (squareScalarDrift (u t) - u t)
  refine ⟨2 * (2 * squareAmplitude u t * r'), ?_, ?_⟩
  · rw [hfun]
    simpa [r'] using ((hasDerivAt_squareAmplitude (hu t)).pow 2).const_mul 2
  · exact mul_nonpos_of_nonneg_of_nonpos (by norm_num) (mul_nonpos_of_nonneg_of_nonpos
      (by linarith [one_le_squareAmplitude u hu h0 ht])
      (squareAmplitude_derivative_neg u hu h0 ht).le)

example : ∃ u : ℝ → ℝ, (∀ t : ℝ, HasDerivAt u (squareScalarDrift (u t)) t) ∧
    u 0 = 2 ∧ (0 : ℝ) ≤ 1 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  exact ⟨u, hu, h0, zero_le_one⟩

/-- An actual good-multiplicity trajectory has a genuine nonempty limit
polytope, yet its positive corner is always outside it, with nonpositive
scores and energy derivative. Source: arXiv:2305.05465v6, §10,
`cl:gamma'12` and Step 2'; the fixed triple is good by `isGoodTripleMulti_one`,
its root by `isAttentionRoot_id`, and its projection by `isProjOnto_one`. -/
theorem exists_square_outside_limit_polytope :
    ∃ (Z : ℝ → Idx 4 → EucSpace 2) (S : Finset (⊤ : Submodule ℝ (EucSpace 2))),
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z ∧ S.Nonempty ∧
      polytopeIn ⊤ S = {x | ∀ t : ℝ, 0 ≤ t → x ∈ (1 : ParamMatrix 2) '' tokenHull (Z t)} ∧
      Tendsto (fun t => Metric.hausdorffDist ((1 : ParamMatrix 2) '' tokenHull (Z t))
        (polytopeIn ⊤ S)) atTop (nhds 0) ∧
      (∀ t : ℝ, 0 ≤ t → Z t 3 ∉ polytopeIn ⊤ S) ∧
      (∀ (t : ℝ) (j : Idx 4), projScore (1 : ParamMatrix 2) 1 (Z t) 3 j ≤ 0) ∧
      ∀ t : ℝ, 0 ≤ t → ∃ D : ℝ, HasDerivAt (fun s => ‖Z s 3‖ ^ 2) D t ∧ D ≤ 0 := by
  obtain ⟨u, h0, hu⟩ := exists_squareScalar_solution 2
  let Z : ℝ → Idx 4 → EucSpace 2 := fun t => squareCorner (squareAmplitude u t)
  have hZ : RescaledDynamics (1 : ParamMatrix 2) 1 1 Z := rescaledDynamics_square u hu
  obtain ⟨S, hS, _, hHull, hHaus⟩ := hZ.exists_projected_hull_polytope_limit 1 1 1 1
    ⊤ ⊥ 1 (isGoodTripleMulti_one 2).2 (isProjOnto_one 2) Z (by norm_num)
  refine ⟨Z, S, hZ, hS, hHull, hHaus, ?_, fun t j => projScore_square_nonpos _ 3 j, ?_⟩
  · intro t ht hmem
    rw [hHull] at hmem
    apply squareCorner_not_mem_future_tokenHull u hu h0 ht
    simpa only [one_apply_eq_self, Set.image_id'] using hmem
  · exact fun t ht => hasDerivAt_norm_sq_square_nonpos u hu h0 ht 3

end Transformer.Clusters
