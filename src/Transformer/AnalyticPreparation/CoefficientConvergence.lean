/-
# Real analytic preparation: CoefficientConvergence

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.CoefficientMajorants
import Mathlib.Analysis.Normed.Lp.lpSpace

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem summable_changeOrigin_weighted_majorant {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (s r : ℝ≥0)
    (hsr : (s + r : ℝ≥0∞) < p.radius) :
    Summable (fun l : ℕ ↦
      (∑' k : ℕ, ‖p.changeOriginSeries k l‖₊ * r ^ k) * s ^ l) := by
  let J := Σ k m : ℕ, {t : Finset (Fin (k + m)) // t.card = m}
  let K := Σ m k : ℕ, {t : Finset (Fin (k + m)) // t.card = m}
  let e : K → J := fun q ↦ ⟨q.2.1, q.1, q.2.2⟩
  have he : Function.Injective e := by
    rintro ⟨m, k, t⟩ ⟨m', k', t'⟩ h
    dsimp only [e] at h
    cases h
    rfl
  have hbig : Summable (fun q : J ↦
      ‖p (q.1 + q.2.1)‖₊ * s ^ q.2.1 * r ^ q.1) :=
    p.changeOriginSeries_summable_aux₁ hsr
  have hswap : Summable (fun q : K ↦
      ‖p (q.2.1 + q.1)‖₊ * s ^ q.1 * r ^ q.2.1) := by
    simpa [Function.comp_def, e] using NNReal.summable_comp_injective hbig he
  have hr : (r : ℝ≥0∞) < p.radius := by
    exact (show (r : ℝ≥0∞) ≤ s + r by simp).trans_lt hsr
  have hK := NNReal.summable_sigma.1 hswap
  refine NNReal.summable_of_le (fun l ↦ ?_) hK.2
  have hKl := NNReal.summable_sigma.1 (hK.1 l)
  have hleft : Summable (fun k : ℕ ↦
      (‖p.changeOriginSeries k l‖₊ * r ^ k) * s ^ l) :=
    (summable_nnnorm_changeOriginSeries_mul_pow p r hr l).mul_right (s ^ l)
  calc
    (∑' k : ℕ, ‖p.changeOriginSeries k l‖₊ * r ^ k) * s ^ l =
        ∑' k : ℕ, (‖p.changeOriginSeries k l‖₊ * r ^ k) * s ^ l := by
      rw [NNReal.tsum_mul_right]
    _ ≤ ∑' k : ℕ, ∑' t : {t : Finset (Fin (k + l)) // t.card = l},
        ‖p (k + l)‖₊ * s ^ l * r ^ k := by
      apply Summable.tsum_le_tsum
      · intro k
        calc
          (‖p.changeOriginSeries k l‖₊ * r ^ k) * s ^ l ≤
              ((∑' _ : {t : Finset (Fin (k + l)) // t.card = l}, ‖p (k + l)‖₊) *
                r ^ k) * s ^ l := by
            gcongr
            exact p.nnnorm_changeOriginSeries_le_tsum k l
          _ = ∑' _ : {t : Finset (Fin (k + l)) // t.card = l},
              ‖p (k + l)‖₊ * s ^ l * r ^ k := by
            calc
              ((∑' _ : {t : Finset (Fin (k + l)) // t.card = l}, ‖p (k + l)‖₊) *
                  r ^ k) * s ^ l =
                  (∑' _ : {t : Finset (Fin (k + l)) // t.card = l},
                    ‖p (k + l)‖₊ * r ^ k) * s ^ l := by
                exact congrArg (fun x ↦ x * s ^ l)
                  (NNReal.tsum_mul_right
                    (fun _ : {t : Finset (Fin (k + l)) // t.card = l} ↦ ‖p (k + l)‖₊)
                    (r ^ k)).symm
              _ = ∑' _ : {t : Finset (Fin (k + l)) // t.card = l},
                    (‖p (k + l)‖₊ * r ^ k) * s ^ l := by
                exact (NNReal.tsum_mul_right
                  (fun _ : {t : Finset (Fin (k + l)) // t.card = l} ↦
                    ‖p (k + l)‖₊ * r ^ k) (s ^ l)).symm
              _ = ∑' _ : {t : Finset (Fin (k + l)) // t.card = l},
                    ‖p (k + l)‖₊ * s ^ l * r ^ k := by
                apply tsum_congr
                intro t
                ac_rfl
      · exact hleft
      · exact hKl.2
    _ = ∑' q : Σ k : ℕ, {t : Finset (Fin (k + l)) // t.card = l},
        ‖p (q.1 + l)‖₊ * s ^ l * r ^ q.1 := by
      exact (Summable.tsum_sigma'
        (f := fun q : Σ k : ℕ, {t : Finset (Fin (k + l)) // t.card = l} ↦
          ‖p (q.1 + l)‖₊ * s ^ l * r ^ q.1) hKl.1 (hK.1 l)).symm

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem le_radius_weightedCoefficientSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (s r : ℝ≥0)
    (hsr : (s + r : ℝ≥0∞) < p.radius) :
    (s : ℝ≥0∞) ≤ (weightedCoefficientSeries p r).radius := by
  have hr : (r : ℝ≥0∞) < p.radius :=
    (show (r : ℝ≥0∞) ≤ s + r by simp).trans_lt hsr
  apply FormalMultilinearSeries.le_radius_of_summable_nnnorm
  refine NNReal.summable_of_le (fun l ↦ ?_)
    (summable_changeOrigin_weighted_majorant p s r hsr)
  gcongr
  exact nnnorm_weightedCoefficientSeries_le p r hr l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem radius_weightedCoefficientSeries_pos {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) :
    0 < (weightedCoefficientSeries p r).radius := by
  rcases ENNReal.lt_iff_exists_add_pos_lt.1 hr with ⟨s, hs, hrs⟩
  have hsr : (s + r : ℝ≥0∞) < p.radius := by simpa [add_comm] using hrs
  exact (show (0 : ℝ≥0∞) < s by simpa using hs).trans_le
    (le_radius_weightedCoefficientSeries p s r hsr)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_weightedCoefficientSeries_sum {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) :
    AnalyticAt ℝ (weightedCoefficientSeries p r).sum 0 :=
  ((weightedCoefficientSeries p r).hasFPowerSeriesOnBall
    (radius_weightedCoefficientSeries_pos p r hr)).analyticAt

/-- On the common convergence neighborhood, the `j`-th coordinate of the
`ℓ¹`-valued analytic sum is exactly the weighted moving Taylor coefficient.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem weightedCoefficientSeries_sum_apply {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (z : Base n) (j : ℕ)
    (hr : (r : ℝ≥0∞) < p.radius)
    (hzq : z ∈ Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius)
    (hzp : baseInclusion n z ∈ Metric.eball (0 : Ambient n) p.radius) :
    (weightedCoefficientSeries p r).sum z j =
      (r : ℝ) ^ j *
        p.changeOrigin (baseInclusion n z) j (fun _ ↦ lastDirection n) := by
  let coord : WeightedSeq →L[ℝ] ℝ := lp.evalCLM ℝ (fun _ : ℕ ↦ ℝ) 1 j
  have hqsum := ((weightedCoefficientSeries p r).summable hzq).hasSum.mapL coord
  have hzpj : baseInclusion n z ∈
      Metric.eball (0 : Ambient n) (p.changeOriginSeries j).radius := by
    exact mem_eball_zero_iff.mpr
      ((mem_eball_zero_iff.mp hzp).trans_le (p.le_changeOriginSeries_radius j))
  have hpsum : HasSum
      (fun l : ℕ ↦ (p.changeOriginSeries j l)
        (fun _ ↦ baseInclusion n z) (fun _ ↦ lastDirection n))
      (p.changeOrigin (baseInclusion n z) j (fun _ ↦ lastDirection n)) := by
    simpa [FormalMultilinearSeries.changeOrigin, FormalMultilinearSeries.sum, evalLast] using
      ((p.changeOriginSeries j).summable hzpj).hasSum.mapL (evalLast n j)
  change coord ((weightedCoefficientSeries p r).sum z) = _
  calc
    coord ((weightedCoefficientSeries p r).sum z) =
        ∑' l : ℕ, coord (weightedCoefficientSeries p r l (fun _ ↦ z)) :=
      hqsum.tsum_eq.symm
    _ = ∑' l : ℕ, (r : ℝ) ^ j * (p.changeOriginSeries j l)
        (fun _ ↦ baseInclusion n z) (fun _ ↦ lastDirection n) := by
      congr 1
      funext l
      change weightedCoefficientSeries p r l (fun _ ↦ z) j = _
      exact weightedCoefficientSeries_apply p r hr l (fun _ ↦ z) j
    _ = (r : ℝ) ^ j *
        p.changeOrigin (baseInclusion n z) j (fun _ ↦ lastDirection n) :=
      (hpsum.mul_left ((r : ℝ) ^ j)).tsum_eq

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem weightedCoefficientSeries_sum_apply_lastTaylorCoefficient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (z : Base n) (j : ℕ)
    (hr : (r : ℝ≥0∞) < p.radius)
    (hzq : z ∈ Metric.eball (0 : Base n) (weightedCoefficientSeries p r).radius)
    (hzp : (z, 0) ∈ Metric.eball (0 : Ambient n) p.radius) :
    (weightedCoefficientSeries p r).sum z j =
      (r : ℝ) ^ j * lastTaylorCoefficient p j z := by
  simpa [lastTaylorCoefficient] using
    weightedCoefficientSeries_sum_apply p r z j hr hzq hzp

end Transformer.AnalyticPreparation
