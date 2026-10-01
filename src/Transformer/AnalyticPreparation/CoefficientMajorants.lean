/-
# Real analytic preparation: CoefficientMajorants

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.MovingCoefficients
import Mathlib.Analysis.Normed.Lp.lpSpace

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- The ordinary (Archimedean) majorant needed to sum the weighted coordinate
series.  This is extracted from Mathlib's `changeOriginSeries_summable_aux₁`;
no ultrametric inequality is used.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem summable_nnnorm_changeOriginSeries_mul_pow {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (l : ℕ) :
    Summable (fun k : ℕ ↦ ‖p.changeOriginSeries k l‖₊ * r ^ k) := by
  rcases ENNReal.lt_iff_exists_add_pos_lt.1 hr with ⟨s, hs, hrs⟩
  have hsr : (s + r : ℝ≥0∞) < p.radius := by simpa [add_comm] using hrs
  let I := Σ k : ℕ, {t : Finset (Fin (k + l)) // t.card = l}
  let J := Σ k m : ℕ, {t : Finset (Fin (k + m)) // t.card = m}
  let e : I → J := fun q ↦ ⟨q.1, l, q.2⟩
  have he : Function.Injective e := by
    rintro ⟨k, t⟩ ⟨k', t'⟩ h
    dsimp only [e] at h
    cases h
    rfl
  have hbig : Summable (fun q : J ↦
      ‖p (q.1 + q.2.1)‖₊ * s ^ q.2.1 * r ^ q.1) :=
    p.changeOriginSeries_summable_aux₁ hsr
  have hfixedS : Summable (fun q : I ↦
      ‖p (q.1 + l)‖₊ * s ^ l * r ^ q.1) := by
    simpa [Function.comp_def, e] using NNReal.summable_comp_injective hbig he
  have hspow : s ^ l ≠ 0 := pow_ne_zero l hs.ne'
  have hfixed : Summable (fun q : I ↦ ‖p (q.1 + l)‖₊ * r ^ q.1) := by
    simpa [mul_assoc, mul_left_comm, mul_comm, hspow] using
      hfixedS.mul_right (s ^ l)⁻¹
  have houter : Summable (fun k : ℕ ↦
      ∑' t : {t : Finset (Fin (k + l)) // t.card = l},
        ‖p (k + l)‖₊ * r ^ k) :=
    (NNReal.summable_sigma.1 hfixed).2
  refine NNReal.summable_of_le (fun k ↦ ?_) houter
  calc
    ‖p.changeOriginSeries k l‖₊ * r ^ k ≤
        (∑' _ : {t : Finset (Fin (k + l)) // t.card = l}, ‖p (k + l)‖₊) * r ^ k := by
      gcongr
      exact p.nnnorm_changeOriginSeries_le_tsum k l
    _ = ∑' _ : {t : Finset (Fin (k + l)) // t.card = l},
        ‖p (k + l)‖₊ * r ^ k := by rw [NNReal.tsum_mul_right]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem summable_weightedCoordinateSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (l : ℕ) :
    Summable (fun k : ℕ ↦ weightedCoordinateSeries p r k l) := by
  refine Summable.of_nnnorm_bounded
    (summable_nnnorm_changeOriginSeries_mul_pow p r hr l) (fun k ↦ ?_)
  exact nnnorm_weightedCoordinateSeries_le p r k l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem summable_nnnorm_weightedCoordinateSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (l : ℕ) :
    Summable (fun k : ℕ ↦ ‖weightedCoordinateSeries p r k l‖₊) := by
  refine NNReal.summable_of_le (fun k ↦ ?_)
    (summable_nnnorm_changeOriginSeries_mul_pow p r hr l)
  exact nnnorm_weightedCoordinateSeries_le p r k l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem nnnorm_weightedCoefficientSeries_le {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (l : ℕ) :
    ‖weightedCoefficientSeries p r l‖₊ ≤
      ∑' k : ℕ, ‖p.changeOriginSeries k l‖₊ * r ^ k := by
  unfold weightedCoefficientSeries
  calc
    ‖∑' k : ℕ, weightedCoordinateSeries p r k l‖₊ ≤
        ∑' k : ℕ, ‖weightedCoordinateSeries p r k l‖₊ :=
      nnnorm_tsum_le (summable_nnnorm_weightedCoordinateSeries p r hr l)
    _ ≤ ∑' k : ℕ, ‖p.changeOriginSeries k l‖₊ * r ^ k := by
      apply Summable.tsum_le_tsum
      · intro k
        exact nnnorm_weightedCoordinateSeries_le p r k l
      · exact summable_nnnorm_weightedCoordinateSeries p r hr l
      · exact summable_nnnorm_changeOriginSeries_mul_pow p r hr l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem weightedCoefficientSeries_apply {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (l : ℕ) (v : Fin l → Base n) (j : ℕ) :
    weightedCoefficientSeries p r l v j =
      (r : ℝ) ^ j * (p.changeOriginSeries j l)
        (fun i ↦ baseInclusion n (v i)) (fun _ ↦ lastDirection n) := by
  let ev : ((Base n)[×l]→L[ℝ] WeightedSeq) →L[ℝ] ℝ :=
    (lp.evalCLM ℝ (fun _ : ℕ ↦ ℝ) 1 j).comp
      (ContinuousMultilinearMap.apply ℝ (fun _ : Fin l ↦ Base n) WeightedSeq v)
  have hsum := (summable_weightedCoordinateSeries p r hr l).hasSum.mapL ev
  change ev (∑' k : ℕ, weightedCoordinateSeries p r k l) = _
  rw [← hsum.tsum_eq]
  change (∑' k : ℕ, (lp.single 1 k ((r : ℝ) ^ k *
    (p.changeOriginSeries k l) (fun i ↦ baseInclusion n (v i))
      (fun _ ↦ lastDirection n)) : WeightedSeq) j) = _
  rw [tsum_eq_single j]
  · exact Pi.single_eq_same _ _
  · intro k hkj
    exact Pi.single_eq_of_ne' hkj _

end Transformer.AnalyticPreparation
