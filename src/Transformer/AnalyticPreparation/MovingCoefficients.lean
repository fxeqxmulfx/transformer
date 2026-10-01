/-
# Real analytic preparation: MovingCoefficients

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.DirectionalCoefficients
import Mathlib.Analysis.Normed.Lp.lpSpace

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

noncomputable abbrev WeightedSeq := lp (fun _ : ℕ ↦ ℝ) 1

noncomputable def baseInclusion (n : ℕ) : Base n →L[ℝ] Ambient n :=
  ContinuousLinearMap.inl ℝ (Base n) ℝ

noncomputable def evalLast (n k : ℕ) :
    ((Ambient n)[×k]→L[ℝ] ℝ) →L[ℝ] ℝ :=
  ContinuousMultilinearMap.apply ℝ (fun _ : Fin k ↦ Ambient n) ℝ
    (fun _ ↦ lastDirection n)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem norm_lastDirection (n : ℕ) : ‖lastDirection n‖ = 1 := by
  simp [lastDirection]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem baseInclusion_apply (n : ℕ) (z : Base n) :
    baseInclusion n z = (z, 0) := rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_baseInclusion_le (n : ℕ) : ‖baseInclusion n‖ ≤ 1 := by
  exact ContinuousLinearMap.opNorm_le_bound (baseInclusion n) (by norm_num) (fun z ↦ by
    simp [baseInclusion])

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_evalLast_le (n k : ℕ) : ‖evalLast n k‖ ≤ 1 := by
  exact ContinuousLinearMap.opNorm_le_bound (evalLast n k) (by norm_num) (fun q ↦ by
    simpa [evalLast, norm_lastDirection] using
      q.le_opNorm (fun _ ↦ lastDirection n))

noncomputable def lastCoefficientSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (k : ℕ) :
    FormalMultilinearSeries ℝ (Base n) ℝ :=
  (evalLast n k).compFormalMultilinearSeries
    ((p.changeOriginSeries k).compContinuousLinearMap (baseInclusion n))

noncomputable def weightedSingleCLM (r : ℝ≥0) (k : ℕ) :
    ℝ →L[ℝ] WeightedSeq :=
  (lp.singleContinuousLinearMap ℝ (fun _ : ℕ ↦ ℝ) 1 k).comp
    ((r : ℝ) ^ k • ContinuousLinearMap.id ℝ ℝ)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem weightedSingleCLM_apply (r : ℝ≥0) (k : ℕ) (z : ℝ) :
    weightedSingleCLM r k z = lp.single 1 k ((r : ℝ) ^ k * z) := by
  rfl

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_weightedSingleCLM_le (r : ℝ≥0) (k : ℕ) :
    ‖weightedSingleCLM r k‖ ≤ (r : ℝ) ^ k := by
  exact ContinuousLinearMap.opNorm_le_bound (weightedSingleCLM r k)
    (pow_nonneg r.coe_nonneg k) (fun z ↦ by
      rw [weightedSingleCLM_apply, lp.norm_single (by norm_num), norm_mul, norm_pow,
        Real.norm_of_nonneg r.coe_nonneg])

noncomputable def weightedCoordinateSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (k : ℕ) :
    FormalMultilinearSeries ℝ (Base n) WeightedSeq :=
  (weightedSingleCLM r k).compFormalMultilinearSeries (lastCoefficientSeries p k)

noncomputable def weightedCoefficientSeries {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) :
    FormalMultilinearSeries ℝ (Base n) WeightedSeq :=
  fun l ↦ ∑' k : ℕ, weightedCoordinateSeries p r k l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_lastCoefficientSeries_le {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (k l : ℕ) :
    ‖lastCoefficientSeries p k l‖ ≤ ‖p.changeOriginSeries k l‖ := by
  unfold lastCoefficientSeries
  rw [ContinuousLinearMap.compFormalMultilinearSeries_apply]
  have hpre :
      ‖((p.changeOriginSeries k).compContinuousLinearMap (baseInclusion n)) l‖ ≤
        ‖p.changeOriginSeries k l‖ := by
    calc
      _ ≤ ‖p.changeOriginSeries k l‖ * ‖baseInclusion n‖ ^ l :=
        (p.changeOriginSeries k).norm_compContinuousLinearMap_le (baseInclusion n) l
      _ ≤ ‖p.changeOriginSeries k l‖ * 1 ^ l := by
        exact mul_le_mul_of_nonneg_left
          (pow_le_pow_left₀ (norm_nonneg (baseInclusion n)) (norm_baseInclusion_le n) l)
          (norm_nonneg (p.changeOriginSeries k l))
      _ = ‖p.changeOriginSeries k l‖ := by simp
  calc
    ‖(evalLast n k).compContinuousMultilinearMap
        (((p.changeOriginSeries k).compContinuousLinearMap (baseInclusion n)) l)‖ ≤
        ‖evalLast n k‖ *
          ‖((p.changeOriginSeries k).compContinuousLinearMap (baseInclusion n)) l‖ :=
      ContinuousLinearMap.norm_compContinuousMultilinearMap_le _ _
    _ ≤ 1 * ‖p.changeOriginSeries k l‖ :=
      mul_le_mul (norm_evalLast_le n k) hpre
        (norm_nonneg (((p.changeOriginSeries k).compContinuousLinearMap
          (baseInclusion n)) l)) (by norm_num)
    _ = ‖p.changeOriginSeries k l‖ := one_mul _

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem norm_weightedCoordinateSeries_le {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (k l : ℕ) :
    ‖weightedCoordinateSeries p r k l‖ ≤
      (r : ℝ) ^ k * ‖p.changeOriginSeries k l‖ := by
  unfold weightedCoordinateSeries
  rw [ContinuousLinearMap.compFormalMultilinearSeries_apply]
  calc
    ‖(weightedSingleCLM r k).compContinuousMultilinearMap
        (lastCoefficientSeries p k l)‖ ≤
        ‖weightedSingleCLM r k‖ * ‖lastCoefficientSeries p k l‖ :=
      ContinuousLinearMap.norm_compContinuousMultilinearMap_le _ _
    _ ≤ (r : ℝ) ^ k * ‖p.changeOriginSeries k l‖ :=
      mul_le_mul (norm_weightedSingleCLM_le r k) (norm_lastCoefficientSeries_le p k l)
        (norm_nonneg _) (pow_nonneg r.coe_nonneg k)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem nnnorm_weightedCoordinateSeries_le {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (k l : ℕ) :
    ‖weightedCoordinateSeries p r k l‖₊ ≤
      ‖p.changeOriginSeries k l‖₊ * r ^ k := by
  apply NNReal.coe_le_coe.mp
  push_cast
  simpa [mul_comm] using norm_weightedCoordinateSeries_le p r k l

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem weightedCoordinateSeries_apply {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0) (k l : ℕ)
    (v : Fin l → Base n) :
    weightedCoordinateSeries p r k l v =
      lp.single 1 k ((r : ℝ) ^ k *
        (p.changeOriginSeries k l) (fun i ↦ baseInclusion n (v i))
          (fun _ ↦ lastDirection n)) := by
  rfl

end Transformer.AnalyticPreparation
