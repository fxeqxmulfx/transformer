/-
# Real analytic preparation: NormalizedMaps

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.OriginCoefficients

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Exact distinguished order supplies a normalized analytic coefficient map
that is within `1/2` of the monomial at the origin and remains within `1` on
a neighborhood of the base origin.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_normalizedAnalyticCoefficientMap {n d : ℕ} {f : Ambient n → ℝ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (hp : HasFPowerSeriesAt f p 0)
    (horder : ExactOrderInLastVariable f d) :
    ∃ (r : ℝ≥0) (hr : (r : ℝ≥0∞) < p.radius),
      0 < r ∧ originWeightedCoeffs p r hr d ≠ 0 ∧
      AnalyticAt ℝ (analyticNormalizedCoefficientMap p r hr d) 0 ∧
      ‖analyticNormalizedCoefficientMap p r hr d 0 - monomialSeq d‖ < (1 : ℝ) / 2 ∧
      ∀ᶠ z in 𝓝 (0 : Base n),
        ‖analyticNormalizedCoefficientMap p r hr d z - monomialSeq d‖ < 1 := by
  obtain ⟨r, hr, hr0, htop, hclose⟩ :=
    exists_radius_normalizedOrigin_close_half p hp horder
  have hzero : analyticNormalizedCoefficientMap p r hr d 0 =
      normalizedOriginCoeffs p r hr d :=
    analyticNormalizedCoefficientMap_zero p r hr d
  have hclose' :
      ‖analyticNormalizedCoefficientMap p r hr d 0 - monomialSeq d‖ < (1 : ℝ) / 2 := by
    rw [hzero]
    exact hclose
  have han := analyticAt_analyticNormalizedCoefficientMap p r hr d
  refine ⟨r, hr, hr0, htop, han, hclose', ?_⟩
  exact eventually_norm_normalizedCoefficientMap_sub_monomial_lt_one
    (weightedCoefficientSeries p r).sum (originWeightedCoeffs p r hr d)
    (analyticAt_weightedCoefficientSeries_sum p r hr) hclose'

end Transformer.AnalyticPreparation
