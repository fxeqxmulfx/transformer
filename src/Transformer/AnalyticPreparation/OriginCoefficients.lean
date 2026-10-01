/-
# Real analytic preparation: OriginCoefficients

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.SequenceScaling

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma originWeightedCoeffs_low_eq_zero {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) (R : ℝ≥0)
    (hR : (R : ℝ≥0∞) < p.radius) {k : ℕ} (hk : k < d) :
    originWeightedCoeffs p R hR k = 0 := by
  rw [originWeightedCoeffs_apply,
    ((exactOrderInLastVariable_iff_lastTaylorCoefficients p hp).mp horder).1 k hk,
    mul_zero]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma originWeightedCoeffs_top_ne_zero {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) (R : ℝ≥0) (hR0 : 0 < R)
    (hR : (R : ℝ≥0∞) < p.radius) :
    originWeightedCoeffs p R hR d ≠ 0 := by
  rw [originWeightedCoeffs_apply]
  have hRC : (R : ℝ) ≠ 0 := by
    simp only [ne_eq, NNReal.coe_eq_zero]
    exact hR0.ne'
  exact mul_ne_zero (pow_ne_zero _ hRC)
    ((exactOrderInLastVariable_iff_lastTaylorCoefficients p hp).mp horder).2

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma mul_radius_lt_of_le_one {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (R t : ℝ≥0)
    (hR : (R : ℝ≥0∞) < p.radius) (ht : t ≤ 1) :
    ((t * R : ℝ≥0) : ℝ≥0∞) < p.radius := by
  apply lt_of_le_of_lt _ hR
  exact_mod_cast mul_le_of_le_one_left (show 0 ≤ R from bot_le) ht

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma scaleSeq_originWeightedCoeffs {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (R t : ℝ≥0)
    (hR : (R : ℝ≥0∞) < p.radius) (ht : t ≤ 1) :
    scaleSeq t ht (originWeightedCoeffs p R hR) =
      originWeightedCoeffs p (t * R) (mul_radius_lt_of_le_one p R t hR ht) := by
  apply lp.ext
  funext k
  simp only [scaleSeq_apply, originWeightedCoeffs_apply]
  push_cast
  rw [mul_pow]
  ring

noncomputable def normalizedOriginCoeffs {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : OriginSeq :=
  ((originWeightedCoeffs p r hr d)⁻¹) • originWeightedCoeffs p r hr

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] lemma normalizedOriginCoeffs_apply {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d k : ℕ) :
    normalizedOriginCoeffs p r hr d k =
      (((r : ℝ) ^ d * lastTaylorCoefficient p d 0)⁻¹) *
        ((r : ℝ) ^ k * lastTaylorCoefficient p k 0) := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_radius_normalizedOrigin_close_half {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) :
    ∃ (r : ℝ≥0) (hr : (r : ℝ≥0∞) < p.radius),
      0 < r ∧ originWeightedCoeffs p r hr d ≠ 0 ∧
        ‖normalizedOriginCoeffs p r hr d - monomialSeq d‖ < (1 : ℝ) / 2 := by
  rcases ENNReal.lt_iff_exists_nnreal_btwn.1 hp.radius_pos with ⟨R, hR0E, hR⟩
  have hR0 : 0 < R := by exact_mod_cast hR0E
  let B : OriginSeq := originWeightedCoeffs p R hR
  have hBlow : ∀ j < d, B j = 0 := by
    intro j hj
    exact originWeightedCoeffs_low_eq_zero p hp horder R hR hj
  have hBd : B d ≠ 0 := originWeightedCoeffs_top_ne_zero p hp horder R hR0 hR
  obtain ⟨t, ht, ht0, ht1, hclose⟩ :=
    exists_scale_normalized_close_half B d hBlow hBd
  let r : ℝ≥0 := t * R
  have hr : (r : ℝ≥0∞) < p.radius := mul_radius_lt_of_le_one p R t hR ht
  have hr0 : 0 < r := mul_pos ht0 hR0
  have hscale : scaleSeq t ht B = originWeightedCoeffs p r hr := by
    dsimp only [B, r, hr]
    exact scaleSeq_originWeightedCoeffs p R t hR ht
  have hnorm : normalizedScale t ht B d = normalizedOriginCoeffs p r hr d := by
    unfold normalizedScale normalizedOriginCoeffs
    rw [hscale]
  refine ⟨r, hr, hr0,
    originWeightedCoeffs_top_ne_zero p hp horder r hr0 hr, ?_⟩
  rw [← hnorm]
  exact hclose

/-- Normalize any analytic coefficient map by a fixed scalar. -/
noncomputable def normalizedCoefficientMap {n : ℕ}
    (C : Base n → OriginSeq) (denom : ℝ) : Base n → OriginSeq :=
  fun z ↦ denom⁻¹ • C z

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_normalizedCoefficientMap {n : ℕ}
    (C : Base n → OriginSeq) (denom : ℝ) (hC : AnalyticAt ℝ C 0) :
    AnalyticAt ℝ (normalizedCoefficientMap C denom) 0 := by
  change AnalyticAt ℝ (denom⁻¹ • C) 0
  exact hC.const_smul

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventually_norm_normalizedCoefficientMap_sub_monomial_lt_one {n d : ℕ}
    (C : Base n → OriginSeq) (denom : ℝ) (hC : AnalyticAt ℝ C 0)
    (hclose : ‖normalizedCoefficientMap C denom 0 - monomialSeq d‖ < (1 : ℝ) / 2) :
    ∀ᶠ z in 𝓝 (0 : Base n),
      ‖normalizedCoefficientMap C denom z - monomialSeq d‖ < 1 := by
  have hhalf : (1 : ℝ) / 2 < 1 := by norm_num
  have hmem : normalizedCoefficientMap C denom 0 ∈
      Metric.ball (monomialSeq d) 1 := by
    rw [Metric.mem_ball, dist_eq_norm]
    exact hclose.trans hhalf
  have hopen : Metric.ball (monomialSeq d) 1 ∈
      𝓝 (normalizedCoefficientMap C denom 0) :=
    Metric.isOpen_ball.mem_nhds hmem
  filter_upwards [(analyticAt_normalizedCoefficientMap C denom hC).continuousAt.eventually hopen]
    with z hz
  simpa [Metric.mem_ball, dist_eq_norm] using hz

/-- The analytic weighted coefficient map normalized by its degree-`d`
coefficient at the base origin. -/
noncomputable def analyticNormalizedCoefficientMap {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Base n → OriginSeq :=
  normalizedCoefficientMap (weightedCoefficientSeries p r).sum
    (originWeightedCoeffs p r hr d)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticNormalizedCoefficientMap_zero {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) :
    analyticNormalizedCoefficientMap p r hr d 0 = normalizedOriginCoeffs p r hr d := by
  have hzq : (0 : Base n) ∈
      Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius := by
    simpa only [Metric.mem_eball, edist_self] using
      radius_weightedCoefficientSeries_pos p r hr
  have hzp : ((0 : Base n), (0 : ℝ)) ∈
      Metric.eball (0 : Ambient n) p.radius := by
    rw [← ambient_zero_eq n]
    have hp0 : (0 : ℝ≥0∞) < p.radius :=
      (show (0 : ℝ≥0∞) ≤ r by simp).trans_lt hr
    simpa only [Metric.mem_eball, edist_self] using hp0
  have hC0 : (weightedCoefficientSeries p r).sum (0 : Base n) =
      originWeightedCoeffs p r hr := by
    apply lp.ext
    funext k
    exact weightedCoefficientSeries_sum_apply_lastTaylorCoefficient
      p r 0 k hr hzq hzp
  unfold analyticNormalizedCoefficientMap normalizedCoefficientMap normalizedOriginCoeffs
  rw [hC0]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_analyticNormalizedCoefficientMap {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) :
    AnalyticAt ℝ (analyticNormalizedCoefficientMap p r hr d) 0 := by
  exact analyticAt_normalizedCoefficientMap _ _
    (analyticAt_weightedCoefficientSeries_sum p r hr)

end Transformer.AnalyticPreparation
