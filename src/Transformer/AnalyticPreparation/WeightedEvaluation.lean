/-
# Real analytic preparation: WeightedEvaluation

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.ScalarEvaluation

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_weightedCoefficientSeries_eq {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr0 : 0 < r) (hrp : (r : ℝ≥0∞) < p.radius)
    (z : Base n) (w : ℝ)
    (hzq : z ∈ Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius)
    (hzw : (‖(z, (0 : ℝ))‖₊ + ‖((0 : Base n), w)‖₊ : ℝ≥0∞) < p.radius)
    (hw : ‖w‖ < r) :
    evalL1PowerSeries ((weightedCoefficientSeries p r).sum z) ((r : ℝ)⁻¹ * w) =
      p.sum (z, w) := by
  have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
  have hscaled : ‖(r : ℝ)⁻¹ * w‖ < 1 := by
    rw [norm_mul, norm_inv, Real.norm_of_nonneg r.coe_nonneg]
    exact (inv_mul_lt_one₀ (by exact_mod_cast hr0)).2 (by exact_mod_cast hw)
  rw [evalL1PowerSeries_eq_tsum _ hscaled]
  have hzp : (z, (0 : ℝ)) ∈ Metric.eball (0 : Ambient n) p.radius := by
    rw [mem_eball_zero_iff]
    exact (show (‖(z, (0 : ℝ))‖₊ : ℝ≥0∞) ≤
        ‖(z, (0 : ℝ))‖₊ + ‖((0 : Base n), w)‖₊ by
      exact le_add_right le_rfl).trans_lt hzw
  calc
    (∑' k : ℕ, (weightedCoefficientSeries p r).sum z k * ((r : ℝ)⁻¹ * w) ^ k) =
        ∑' k : ℕ, p.changeOrigin (z, 0) k (fun _ ↦ lastDirection n) * w ^ k := by
      apply tsum_congr
      intro k
      rw [weightedCoefficientSeries_sum_apply_lastTaylorCoefficient p r z k hrp hzq hzp]
      simp only [lastTaylorCoefficient]
      let c : ℝ := p.changeOrigin (z, 0) k (fun _ ↦ lastDirection n)
      change ((r : ℝ) ^ k * c) * ((r : ℝ)⁻¹ * w) ^ k = c * w ^ k
      rw [mul_pow]
      calc
        ((r : ℝ) ^ k * c) * ((r : ℝ)⁻¹ ^ k * w ^ k) =
            (((r : ℝ) ^ k * (r : ℝ)⁻¹ ^ k) * c) * w ^ k := by ring
        _ = c * w ^ k := by rw [← mul_pow]; simp [hrC]
    _ = (p.changeOrigin (z, 0)).sum (0, w) := by
      unfold FormalMultilinearSeries.sum
      apply tsum_congr
      intro k
      rw [show ((0 : Base n), w) = w • lastDirection n by
        ext <;> simp [lastDirection]]
      rw [(p.changeOrigin (z, 0) k).map_smul_univ]
      simp [Finset.prod_const, smul_eq_mul, mul_comm]
    _ = p.sum (z, w) := by
      rw [p.changeOrigin_eval hzw]
      congr 1
      ext <;> simp

/-- If `p` represents `f` on a ball, weighted evaluation reconstructs `f` on
the explicit common convergence region.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_weightedCoefficientSeries_eq_of_hasFPowerSeriesOnBall {n : ℕ}
    {f : Ambient n → ℝ} (p : FormalMultilinearSeries ℝ (Ambient n) ℝ)
    (r : ℝ≥0) {ρ : ℝ≥0∞}
    (hp : HasFPowerSeriesOnBall f p 0 ρ)
    (hr0 : 0 < r) (hrp : (r : ℝ≥0∞) < p.radius)
    (z : Base n) (w : ℝ)
    (hzq : z ∈ Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius)
    (hzw : (‖(z, (0 : ℝ))‖₊ + ‖((0 : Base n), w)‖₊ : ℝ≥0∞) < ρ)
    (hw : ‖w‖ < r) :
    evalL1PowerSeries ((weightedCoefficientSeries p r).sum z) ((r : ℝ)⁻¹ * w) =
      f (z, w) := by
  rw [eval_weightedCoefficientSeries_eq p r hr0 hrp z w hzq
    (hzw.trans_le hp.r_le) hw]
  symm
  have hball : (z, w) ∈ Metric.eball (0 : Ambient n) ρ := by
    rw [mem_eball_zero_iff]
    have hpair : (‖(z, w)‖₊ : ℝ≥0∞) ≤
      ‖(z, (0 : ℝ))‖₊ + ‖((0 : Base n), w)‖₊ := by
      rw [show (z, w) = (z, (0 : ℝ)) + ((0 : Base n), w) by ext <;> simp]
      exact_mod_cast nnnorm_add_le (z, (0 : ℝ)) ((0 : Base n), w)
    exact hpair.trans_lt hzw
  simpa only [zero_add] using hp.sum hball

end Transformer.AnalyticPreparation
