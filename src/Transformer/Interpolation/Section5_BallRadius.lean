/-
# A common radius and ordered centers for each small carrying cap

ArXiv:2411.04551v3, §5, proof of `cl: balls`. For a geometric
scale `0 < k ≤ 1`, use carrying radius `k/1000` and a center sweep of
angular length `k/8`. The first center is at chord distance between
`k/16` and `k/8` from the endpoint. Radius bounds two carrying radii
on either side give masses zero and one, so continuity gives mass `1/M`.

The resulting common radius lies strictly above the carrying radius
and below `k/4`. Nesting on the carrier makes the actual mass function
antitone in angular time. Its remaining prescribed cumulative masses
give strictly decreasing center times by the proved quantile theorem.
Every mass is measured under the original probability.
-/

import Transformer.Interpolation.Section5_BallQuantiles
import Transformer.Interpolation.Section5_BallSweep
import Transformer.Interpolation.Section5_SphericalBallLaws

open Real MeasureTheory Metric Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The first original ball has mass `1/M`, with a common radius
that both contains the small carrier at the final endpoint and stays
small enough to avoid the other measures. Source: arXiv:2411.04551v3,
§5, `cl: balls`, choosing `rⁱ` between the two zero/one mass bounds. -/
theorem exists_small_cap_sweep_radius (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (p v : SSphere d) (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0)
    (S : Set (SSphere d)) (hS : ν Sᶜ = 0)
    (hboundary : ∀ y : SSphere d, ∀ r : ℝ, ν (sphere y r) = 0)
    (k : ℝ) (hk : 0 < k) (hk1 : k ≤ 1)
    (hsmall : ∀ x ∈ S, dist x p ≤ k / 1000) (M : ℕ) (hM : 0 < M) :
    ∃ r : ℝ, k / 1000 < r ∧ r ≤ k / 4 ∧
      ν (ball (sweepPoint p v hperp (k / 8)) r) = ENNReal.ofReal ((M : ℝ)⁻¹) := by
  let q := sweepPoint p v hperp (k / 8)
  let D := dist q p
  have hDlo : k / 16 ≤ D := by
    have h := half_le_dist_sweepPoint_zero p v hperp
      (a := k / 8) (by positivity) (by linarith)
    change (k / 8) / 2 ≤ D at h
    linarith
  have hDhi : D ≤ k / 8 := by
    simpa only [sweepPoint_zero, sub_zero, abs_of_pos (by positivity : 0 < k / 8)] using
      dist_sweepPoint_le_abs p v hperp (k / 8) 0
  have hlo : ν (ball q (D - 2 * (k / 1000))) = 0 := by
    apply measure_zero_of_disjoint_carrier ν S hS
    intro x hx hball
    have hb : dist x q < D - 2 * (k / 1000) := hball
    have ht := dist_triangle q x p
    rw [dist_comm q x] at ht
    change D ≤ dist x q + dist x p at ht
    linarith [hsmall x hx]
  have hhi : ν (ball q (D + 2 * (k / 1000))) = 1 := by
    apply measure_one_of_contains_carrier ν S hS
    intro x hx
    have ht := dist_triangle x p q
    rw [dist_comm p q] at ht
    change dist x q < D + 2 * (k / 1000)
    change dist x q ≤ dist x p + D at ht
    linarith [hsmall x hx]
  obtain ⟨r, hr, hmass⟩ := exists_ball_mass_between ν q (hboundary q)
    (r₀ := D - 2 * (k / 1000)) (r₁ := D + 2 * (k / 1000))
    (q := (M : ℝ)⁻¹) (by linarith) (by rw [measureReal_def, hlo]; positivity)
    (by rw [measureReal_def, hhi, ENNReal.toReal_one]
        exact inv_le_one_of_one_le₀ (by exact_mod_cast hM))
  exact ⟨r, by linarith [hr.1], by linarith [hr.2], hmass⟩

/-- All cumulative masses of one small original probability are
attained by equal-radius actual balls at strictly ordered centers.
Source: arXiv:2411.04551v3, §5, `cl: balls`, selecting `sₘⁱ` after `rⁱ`.
The radius bounds and actual parameter interval remain in the conclusion. -/
theorem exists_small_cap_sweep_quantiles (ν : Measure (SSphere d)) [IsProbabilityMeasure ν]
    (p v : SSphere d) (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0)
    (S : Set (SSphere d)) (hS : ν Sᶜ = 0)
    (hboundary : ∀ y : SSphere d, ∀ r : ℝ, ν (sphere y r) = 0)
    (k : ℝ) (hk : 0 < k) (hk1 : k ≤ 1)
    (hsmall : ∀ x ∈ S, dist x p ≤ k / 1000) (M : ℕ) (hM : 0 < M) :
    ∃ (r : ℝ) (t : Fin M → ℝ), k / 1000 < r ∧ r ≤ k / 4 ∧
      (∀ m, t m ∈ Icc 0 (k / 8)) ∧ StrictAnti t ∧
      ∀ m, ν (ball (sweepPoint p v hperp (t m)) r) =
        ENNReal.ofReal (((m : ℕ) + 1 : ℝ) / M) := by
  obtain ⟨r, hrlo, hrhi, hmass⟩ := exists_small_cap_sweep_radius ν p v hperp S hS hboundary k hk hk1 hsmall M hM
  have hr : 0 < r := lt_trans (by positivity : 0 < k / 1000) hrlo
  have hr1 : r ≤ 1 := by linarith
  let f : ℝ → ℝ := fun t => ν.real (ball (sweepPoint p v hperp t) r)
  have hpath : Continuous (fun t : ℝ => (sweepPoint p v hperp t, r)) :=
    (continuous_sweepPoint p v hperp).prodMk continuous_const
  have hc : Continuous f := Continuous.comp
    (f := fun t : ℝ => (sweepPoint p v hperp t, r))
    (g := fun z : SSphere d × ℝ => ν.real (ball z.1 z.2))
    (continuous_ball_mass ν hboundary) hpath
  have hp (x : SSphere d) (hx : x ∈ S) : x ∈ ball p r :=
    lt_of_le_of_lt (hsmall x hx) hrlo
  have h0 : f 0 = 1 := by
    dsimp [f]
    rw [sweepPoint_zero, measureReal_def,
      measure_one_of_contains_carrier ν S hS (ball p r) (fun x hx => hp x hx), ENNReal.toReal_one]
  have ha : f (k / 8) = (M : ℝ)⁻¹ := by
    dsimp [f]
    rw [measureReal_def, hmass, ENNReal.toReal_ofReal (by positivity)]
  have hmono : AntitoneOn f (Icc 0 (k / 8)) := by
    intro s hs t ht hst
    change ν.real (ball (sweepPoint p v hperp t) r) ≤ ν.real (ball (sweepPoint p v hperp s) r)
    apply measureReal_mono_on_carrier ν S hS
    intro x hx hxt
    exact sweep_ball_nested_on_endpoint_ball p v hperp hr hr1 hs.1 hst
      (by linarith [ht.2, Real.pi_gt_three]) x (hp x hx) hxt
  obtain ⟨t, ht, heq, hanti⟩ := exists_strictly_ordered_mass_quantiles M hM
    (by positivity : 0 ≤ k / 8) f hc.continuousOn hmono h0 ha
  refine ⟨r, t, hrlo, hrhi, ht, hanti, fun m => ?_⟩
  rw [← heq m]
  exact (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm

/-- The actual successive sweep balls have exactly the prescribed
shell mass. Source: arXiv:2411.04551v3, §5, property 1 of `cl: balls`.
The ambient balls are only nested on the original carrier; the proof
uses that proved geometric fact before subtracting actual cumulative masses. -/
theorem sweep_shell_mass (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (p v : SSphere d) (hperp : inner (𝕜 := ℝ) (p : EucSpace d) (v : EucSpace d) = 0)
    (S : Set (SSphere d)) (hS : ν Sᶜ = 0)
    {r s t : ℝ} (hr : 0 < r) (hr1 : r ≤ 1) (hs : 0 ≤ s) (hst : s ≤ t) (ht : t < Real.pi)
    (hcap : ∀ x ∈ S, x ∈ ball p r) (M m : ℕ)
    (hA : ν (ball (sweepPoint p v hperp s) r) = ENNReal.ofReal (((m : ℝ) + 1) / M))
    (hB : ν (ball (sweepPoint p v hperp t) r) = ENNReal.ofReal ((m : ℝ) / M)) :
    ν (ball (sweepPoint p v hperp s) r \ ball (sweepPoint p v hperp t) r) =
      ENNReal.ofReal ((M : ℝ)⁻¹) := by
  apply measure_shell_eq_inv_nat ν S hS _ _ measurableSet_ball _ M m hA hB
  intro x hx hxt
  exact sweep_ball_nested_on_endpoint_ball p v hperp hr hr1 hs hst ht x (hcap x hx) hxt

/-- The actual uniform restriction to a small nonempty circle cap
realizes every radius and quantile input jointly, including two
distinct requested masses. Source: arXiv:2411.04551v3, §5, `cl: balls`. -/
example : ∃ (ν : Perspective.ProbSphere 2) (p v : SSphere 2)
    (S : Set (SSphere 2)),
    inner (𝕜 := ℝ) (p : EucSpace 2) (v : EucSpace 2) = 0 ∧
      (ν : Measure (SSphere 2)) Sᶜ = 0 ∧
      (∀ y : SSphere 2, ∀ r : ℝ, (ν : Measure (SSphere 2)) (sphere y r) = 0) ∧
      (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧
      (∀ x ∈ S, dist x p ≤ (1 : ℝ) / 1000) ∧ (0 : ℕ) < 2 := by
  obtain ⟨ν, hν, hcap⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) (1 / 1000) (by norm_num)
  obtain ⟨v, hv⟩ := exists_unit_tangent le_rfl (basePoint 1)
  exact ⟨ν, basePoint 1, v, closedBall (basePoint 1) (1 / 1000), hv, hcap, hν,
    one_pos, le_rfl, fun x hx => hx, by decide⟩

/-- The construction produces actual equal-radius balls of masses
one half and one, at distinct ordered centers under a diffuse law.
Source: arXiv:2411.04551v3, §5, `cl: balls`, the nontrivial two-ball case. -/
example : ∃ (ν : Perspective.ProbSphere 2) (p v : SSphere 2)
    (hperp : inner (𝕜 := ℝ) (p : EucSpace 2) (v : EucSpace 2) = 0)
    (r : ℝ) (t : Fin 2 → ℝ),
    (1 / 1000 : ℝ) < r ∧ r ≤ 1 / 4 ∧ (∀ m, t m ∈ Icc 0 (1 / 8)) ∧
      StrictAnti t ∧ ∀ m, (ν : Measure (SSphere 2)) (ball (sweepPoint p v hperp (t m)) r) =
        ENNReal.ofReal (((m : ℕ) + 1 : ℝ) / 2) := by
  obtain ⟨ν, hν, hcap⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) (1 / 1000) (by norm_num)
  obtain ⟨v, hv⟩ := exists_unit_tangent le_rfl (basePoint 1)
  obtain ⟨r, t, h⟩ := exists_small_cap_sweep_quantiles (ν : Measure (SSphere 2))
    (basePoint 1) v hv (closedBall (basePoint 1) (1 / 1000)) hcap hν 1 one_pos le_rfl
    (fun x hx => hx) 2 (by decide)
  exact ⟨ν, basePoint 1, v, hv, r, t, h⟩

end Transformer.Interpolation
