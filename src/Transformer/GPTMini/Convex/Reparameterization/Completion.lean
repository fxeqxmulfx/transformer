import Transformer.GPTMini.Convex.Reparameterization.Boundary

/-!
# A small convex likelihood completion of the finite head control

New construction using Boyd and Vandenberghe (2004), §3.5, and the sequential
Bernoulli probabilities in GPTMini.Convex.Likelihood. Give each of two
query rows two unrestricted utilities. Every positive normalized three-
category prediction table is attained, with convex category likelihoods.
Four scalar utilities suffice to cover both original head witnesses and
the entire positive simplex product, by enlarging the prediction class.

This shows the scalar-head obstruction does not require a larger parameter
count in this tiny control. It does require a different operator/class.
Utilities are independent per query, values are fixed category vectors,
and the category decision order is fixed. This is not a shared learned
token encoder, a variable-context attention block, joint value training,
or the requested compact drop-in model. A per-query categorical table
grows with the number of query/category pairs when generalized directly.
-/

noncomputable section

namespace Transformer.GPTMini.Convex.Reparameterization

open scoped BigOperators
open Likelihood

/-- A finite log-odds utility yields the exact Bernoulli ratio.
Source: the new sequential completion's inverse coordinates, derived from §3.5. -/
theorem routeSigmoid_log_odds (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
    routeSigmoid (Real.log (u / v)) = u / (u + v) := by
  unfold routeSigmoid
  rw [Real.exp_neg, Real.exp_log (div_pos hu hv)]
  have hs : u + v ≠ 0 := by positivity
  field_simp

example : (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 4 := by norm_num

/-- The complementary finite log-odds branch yields the other Bernoulli ratio.
Source: the new completion's exact inverse, without zero or infinite utilities. -/
theorem routeSigmoid_neg_log_odds (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
    routeSigmoid (-Real.log (u / v)) = v / (u + v) := by
  have h := routeSigmoid_complement (Real.log (u / v))
  rw [routeSigmoid_log_odds u v hu hv] at h
  have hs : u + v ≠ 0 := by positivity
  have he : u / (u + v) + v / (u + v) = 1 := by
    field_simp
  linarith

example : (0 : ℝ) < 1 / 3 ∧ (0 : ℝ) < 2 / 3 := by norm_num

/-- Every interior three-category probability row has finite sequential utility coordinates.
Source: the new §3.5 completion; the formula is verified through its actual sigmoid branches. -/
theorem threeRoute_from_probability (u v w : ℝ) (hu : 0 < u) (hv : 0 < v)
    (hw : 0 < w) (hsum : u + v + w = 1) (c : Fin 3) :
    threeRouteProbability (Real.log (u / (v + w)), Real.log (v / w)) c =
      if c = 0 then u else if c = 1 then v else w := by
  have hvw : 0 < v + w := by positivity
  have hs : u + (v + w) = 1 := by linarith
  unfold threeRouteProbability
  split_ifs
  · rw [routeSigmoid_log_odds u (v + w) hu hvw, hs, div_one]
  · rw [routeSigmoid_neg_log_odds u (v + w) hu hvw,
      routeSigmoid_log_odds v w hv hw, hs, div_one]
    field_simp
  · rw [routeSigmoid_neg_log_odds u (v + w) hu hvw,
      routeSigmoid_neg_log_odds v w hv hw, hs, div_one]
    field_simp

example :
    (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 / 2 : ℝ) + 1 / 4 + 1 / 4 = 1 := by norm_num

/-- Four unrestricted scalar utilities, two per query row, define the changed finite operator.
Source: the new shared-memory completion of the sequential §3.5 likelihood control. -/
def sharedRouteProbability (θ : Fin 2 → ℝ × ℝ) : SharedProbability :=
  fun r c => threeRouteProbability (θ r) c

/-- Every entry of the changed operator is positive for all jointly trained utilities.
Source: the actual sequential sigmoid probabilities, with no constraint projection. -/
theorem sharedRouteProbability_pos (θ : Fin 2 → ℝ × ℝ) (r : Fin 2) (c : Fin 3) :
    0 < sharedRouteProbability θ r c := by
  exact threeRouteProbability_pos (θ r) c

/-- The changed operator is exactly normalized for every parameter assignment.
Source: the actual sequential probability branches and their complementary mass identities. -/
theorem sharedRouteProbability_sum (θ : Fin 2 → ℝ × ℝ) (r : Fin 2) :
    ∑ c, sharedRouteProbability θ r c = 1 := by
  exact threeRouteProbability_sum (θ r)

/-- All category likelihoods are jointly convex in the four unrestricted row utilities.
Source: the new finite completion and §3.5; the full probability class is enlarged. -/
theorem sharedRouteNLL_convex (r : Fin 2) (c : Fin 3) :
    ConvexOn ℝ Set.univ (fun θ : Fin 2 → ℝ × ℝ =>
      -Real.log (sharedRouteProbability θ r c)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  exact (threeRouteNLL_convex c).2 (Set.mem_univ (x r)) (Set.mem_univ (y r)) ha hb hab

/-- Explicit inverse coordinates of a strictly positive normalized finite prediction table.
Source: the new completion's sequential conditional odds, not a defining success predicate. -/
def routeCoordinates (p : SharedProbability) (r : Fin 2) : ℝ × ℝ :=
  (Real.log (p r 0 / (p r 1 + p r 2)), Real.log (p r 1 / p r 2))

/-- The actual changed operator reconstructs every positive normalized table from its coordinates.
Source: the new exact finite §3.5 completion; the statement is not restricted to original heads. -/
theorem routeCoordinates_reconstruct (p : SharedProbability) (hpos : ∀ r c, 0 < p r c)
    (hsum : ∀ r, ∑ c, p r c = 1) : sharedRouteProbability (routeCoordinates p) = p := by
  funext r c
  have hs : p r 0 + p r 1 + p r 2 = 1 := by
    simpa only [Fin.sum_univ_three] using hsum r
  have he := threeRoute_from_probability (p r 0) (p r 1) (p r 2)
    (hpos r 0) (hpos r 1) (hpos r 2) hs c
  change threeRouteProbability (Real.log (p r 0 / (p r 1 + p r 2)),
    Real.log (p r 1 / p r 2)) c = p r c
  rw [he]
  fin_cases c <;> norm_num

example :
    (∀ r c, 0 < meanProbability r c) ∧ (∀ r, ∑ c, meanProbability r c = 1) := by
  exact ⟨meanProbability_pos, meanProbability_sum⟩

/-- The changed operator covers every interior finite probability table.
Source: the new explicit inverse proof; arbitrary probability tables are not postulated attainable. -/
theorem sharedRouteProbability_covers (p : SharedProbability) (hpos : ∀ r c, 0 < p r c)
    (hsum : ∀ r, ∑ c, p r c = 1) :
    ∃ θ : Fin 2 → ℝ × ℝ, sharedRouteProbability θ = p := by
  exact ⟨routeCoordinates p, routeCoordinates_reconstruct p hpos hsum⟩

example :
    (∀ r c, 0 < meanProbability r c) ∧ (∀ r, ∑ c, meanProbability r c = 1) := by
  exact ⟨meanProbability_pos, meanProbability_sum⟩

/-- The convex finite completion covers the first physical scalar-head witness.
Source: the new §3.5 completion and the actual softmax endpoint's positivity and normalization. -/
theorem sharedRouteProbability_covers_left :
    ∃ θ : Fin 2 → ℝ × ℝ, sharedRouteProbability θ = leftProbability := by
  exact sharedRouteProbability_covers leftProbability
    (scalarHeadProbability_pos leftQueries leftKeys) (scalarHeadProbability_sum leftQueries leftKeys)

/-- The convex finite completion covers the second physical scalar-head witness.
Source: the new §3.5 completion and the actual second softmax endpoint. -/
theorem sharedRouteProbability_covers_right :
    ∃ θ : Fin 2 → ℝ × ℝ, sharedRouteProbability θ = rightProbability := by
  exact sharedRouteProbability_covers rightProbability
    (scalarHeadProbability_pos rightQueries rightKeys)
    (scalarHeadProbability_sum rightQueries rightKeys)

/-- Every original free scalar head is exactly covered by the changed finite operator.
Source: the new completion of §3's physical head class via its actual positive probability rows. -/
theorem sharedRouteProbability_covers_head (q : Fin 2 → ℝ) (k : Fin 3 → ℝ) :
    ∃ θ : Fin 2 → ℝ × ℝ, sharedRouteProbability θ = scalarHeadProbability q k := by
  exact sharedRouteProbability_covers (scalarHeadProbability q k)
    (scalarHeadProbability_pos q k) (scalarHeadProbability_sum q k)

/-- Using log probabilities as logits retains ordinary log-sum-exp cross entropy.
Source: the new completion's exact normalization; this explicitly specifies the finite readout. -/
theorem sharedRoute_crossEntropy_eq (θ : Fin 2 → ℝ × ℝ) (r : Fin 2) (c : Fin 3) :
    Real.log (∑ j, Real.exp (Real.log (sharedRouteProbability θ r j))) -
      Real.log (sharedRouteProbability θ r c) = -Real.log (sharedRouteProbability θ r c) := by
  have he : ∀ j, Real.exp (Real.log (sharedRouteProbability θ r j)) =
      sharedRouteProbability θ r j := fun j => Real.exp_log (sharedRouteProbability_pos θ r j)
  simp only [he]
  rw [sharedRouteProbability_sum, Real.log_one, zero_sub]

/-- The explicit finite log-probability readout has jointly convex ordinary cross entropy.
Source: the new completion and §3.5; no learned value mixture or downstream FFN is included. -/
theorem sharedRoute_crossEntropy_convex (r : Fin 2) (c : Fin 3) :
    ConvexOn ℝ Set.univ (fun θ : Fin 2 → ℝ × ℝ =>
      Real.log (∑ j, Real.exp (Real.log (sharedRouteProbability θ r j))) -
        Real.log (sharedRouteProbability θ r c)) := by
  simp_rw [sharedRoute_crossEntropy_eq]
  exact sharedRouteNLL_convex r c

/-- Convex likelihoods escape the obstruction by admitting predictions outside the scalar class.
Source: the new finite completion's actual coverage of the consistent arithmetic midpoint. -/
theorem sharedRouteProbability_enlarges_class :
    ∃ θ : Fin 2 → ℝ × ℝ, ¬ScalarHeadClass (sharedRouteProbability θ) := by
  obtain ⟨θ, hθ⟩ := sharedRouteProbability_covers meanProbability
    meanProbability_pos meanProbability_sum
  refine ⟨θ, ?_⟩
  rw [hθ]
  exact meanProbability_not_scalarHead

end Transformer.GPTMini.Convex.Reparameterization
