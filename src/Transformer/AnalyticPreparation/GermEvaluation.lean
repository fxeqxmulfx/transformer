/-
# Real analytic preparation: GermEvaluation

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.PolynomialEvaluation

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- On a sufficiently small common neighborhood, evaluating the weighted
moving-coefficient sequence recovers the analytic function represented by the
ambient formal multilinear series.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventually_eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesAt
    {n : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hp : HasFPowerSeriesAt f p 0) (hr0 : 0 < r)
    (hrp : (r : ℝ≥0∞) < p.radius) :
    (fun x : Ambient n ↦
      evalL1PowerSeries ((weightedCoefficientSeries p r).sum x.1)
        ((r : ℝ)⁻¹ * x.2)) =ᶠ[𝓝 0] f := by
  obtain ⟨ρ, hpρ⟩ := hp
  have hzq : ∀ᶠ x : Ambient n in 𝓝 0,
      x.1 ∈ Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius := by
    have hball := Metric.eball_mem_nhds (0 : Base n)
      (radius_weightedCoefficientSeries_pos p r hrp)
    exact continuousAt_fst.eventually hball
  have hzw : ∀ᶠ x : Ambient n in 𝓝 0,
      (‖(x.1, (0 : ℝ))‖₊ + ‖((0 : Base n), x.2)‖₊ : ℝ≥0∞) < ρ := by
    let s : Ambient n → ℝ≥0∞ := fun x ↦
      (‖(x.1, (0 : ℝ))‖₊ : ℝ≥0∞) +
        (‖((0 : Base n), x.2)‖₊ : ℝ≥0∞)
    have hs : ContinuousAt s 0 := by
      dsimp only [s]
      fun_prop
    have hopen : Set.Iio ρ ∈ 𝓝 (s 0) := by
      apply Iio_mem_nhds
      simpa [s, ambient_zero_eq] using hpρ.r_pos
    simpa only [s] using hs.eventually hopen
  have hw : ∀ᶠ x : Ambient n in 𝓝 0, ‖x.2‖ < r := by
    have hball := Metric.ball_mem_nhds (0 : ℝ) (by exact_mod_cast hr0)
    have hsnd : ∀ᶠ x : Ambient n in 𝓝 0, x.2 ∈ Metric.ball (0 : ℝ) (r : ℝ) :=
      continuousAt_snd.eventually hball
    simpa only [Metric.mem_ball, dist_zero_right] using hsnd
  filter_upwards [hzq, hzw, hw] with x hxq hxρ hxw
  exact eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesOnBall
    p r hpρ hr0 hrp x.1 x.2 hxq hxρ hxw

end Transformer.AnalyticPreparation
