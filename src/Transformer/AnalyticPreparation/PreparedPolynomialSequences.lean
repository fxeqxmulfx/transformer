/-
# Real analytic preparation: PreparedPolynomialSequences

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d, PreparationUniqueness.lean.
The scalar field is real; only the required division infrastructure is copied.
Apache-2.0 licenses: third_party/classical-complex-wpt/LICENSE and
third_party/LocalComplexGeometry/LICENSE.
-/

import Transformer.AnalyticPreparation.PreparedFunctions

open Filter
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
namespace Transformer.AnalyticPreparation

/-- The normalized weighted low-degree tail of a prepared polynomial.  Its
`i`-th coordinate is `r^i / r^d * a_i(z)`; adding the shifted constant
sequence gives the coefficients of `r^{-d} P(z,rw)`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
noncomputable def preparedTailSeq {n : ℕ} (r : ℝ≥0) (d : ℕ)
    (a : Fin d → Base n → ℝ) (z : Base n) : L1Sequence :=
  ∑ i : Fin d, (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) •
    lp.single 1 (i : ℕ) (a i z)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem preparedTailSeq_apply_fin {n d : ℕ} (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (z : Base n) (i : Fin d) :
    preparedTailSeq r d a z (i : ℕ) =
      (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) * a i z := by
  classical
  change (lp.evalCLM ℝ (fun _ : ℕ ↦ ℝ) 1 (i : ℕ))
    (∑ j : Fin d, (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (j : ℕ)) •
      lp.single 1 (j : ℕ) (a j z)) = _
  rw [map_sum, Finset.sum_eq_single i]
  · rw [map_smul]
    change (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) *
      ((lp.single 1 (i : ℕ) (a i z) : L1Sequence) (i : ℕ)) = _
    rw [lp.single_apply_self]
  · intro j hj hji
    have hval : (j : ℕ) ≠ (i : ℕ) := by
      exact fun h ↦ hji (Fin.ext h)
    rw [map_smul]
    change _ * ((lp.single 1 (j : ℕ) (a j z) : L1Sequence) (i : ℕ)) = 0
    have hs : ((lp.single 1 (j : ℕ) (a j z) : L1Sequence) (i : ℕ)) = 0 :=
      lp.single_apply_ne (E := fun _ : ℕ ↦ ℝ) 1 (j : ℕ) (a j z) hval.symm
    rw [hs, mul_zero]
  · simp

/-- Prepared tails have no coefficients in degrees at least `d`.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem seqHighShift_preparedTailSeq {n d : ℕ} (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (z : Base n) :
    seqHighShift d (preparedTailSeq r d a z) = 0 := by
  classical
  apply lp.ext
  funext k
  change (lp.evalCLM ℝ (fun _ : ℕ ↦ ℝ) 1 (k + d))
    (∑ i : Fin d, (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) •
      lp.single 1 (i : ℕ) (a i z)) = 0
  rw [map_sum]
  apply Finset.sum_eq_zero
  intro i hi
  have hne : (i : ℕ) ≠ k + d := by omega
  rw [map_smul]
  change _ * ((lp.single 1 (i : ℕ) (a i z) : L1Sequence) (k + d)) = 0
  have hs : ((lp.single 1 (i : ℕ) (a i z) : L1Sequence) (k + d)) = 0 :=
    lp.single_apply_ne (E := fun _ : ℕ ↦ ℝ) 1 (i : ℕ) (a i z) hne.symm
  rw [hs, mul_zero]

/-- The prepared tail varies analytically with the base point.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_preparedTailSeq {n d : ℕ} (r : ℝ≥0)
    (a : Fin d → Base n → ℝ) (ha : ∀ i, AnalyticAt ℝ (a i) 0) :
    AnalyticAt ℝ (preparedTailSeq r d a) 0 := by
  classical
  have hsum : AnalyticAt ℝ
      (∑ i : Fin d, fun z ↦ (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) •
        (lp.singleContinuousLinearMap ℝ (fun _ : ℕ ↦ ℝ) 1 (i : ℕ)) (a i z)) 0 := by
    apply Finset.univ.analyticAt_sum
    intro i hi
    let single : ℝ →L[ℝ] L1Sequence :=
      lp.singleContinuousLinearMap ℝ (fun _ : ℕ ↦ ℝ) 1 (i : ℕ)
    have hs : AnalyticAt ℝ (fun z ↦ single (a i z)) 0 :=
      (single.analyticAt (a i 0)).comp (f := a i) (ha i)
    have hscaled : AnalyticAt ℝ
        (((((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ))) •
          (fun z ↦ single (a i z))) 0 := hs.const_smul
    have heq : (((((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ))) •
          (fun z ↦ single (a i z))) =
        (fun z ↦ ((((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ))) • single (a i z)) := by
      rfl
    change AnalyticAt ℝ
      (fun z ↦ ((((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ))) • single (a i z)) 0
    rw [← heq]
    exact hscaled
  change AnalyticAt ℝ (fun z ↦ preparedTailSeq r d a z) 0
  have heq : (fun z ↦ preparedTailSeq r d a z) =
      ∑ i : Fin d, fun z ↦ (((r : ℝ) ^ d)⁻¹ * (r : ℝ) ^ (i : ℕ)) •
        (lp.singleContinuousLinearMap ℝ (fun _ : ℕ ↦ ℝ) 1 (i : ℕ)) (a i z) := by
    funext z
    simp only [preparedTailSeq, Finset.sum_apply,
      lp.singleContinuousLinearMap_apply]
  rw [heq]
  exact hsum

/-- Since all prepared coefficients vanish at the base origin, every fixed
positive weighted tail is small on a sufficiently small base neighborhood.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eventually_norm_preparedTailSeq_lt_one {n d : ℕ} (r : ℝ≥0)
    (a : Fin d → Base n → ℝ)
    (ha : ∀ i, AnalyticAt ℝ (a i) 0) (ha0 : ∀ i, a i 0 = 0) :
    ∀ᶠ z in nhds (0 : Base n), ‖preparedTailSeq r d a z‖ < 1 := by
  have hzero : preparedTailSeq r d a 0 = 0 := by
    classical
    unfold preparedTailSeq
    simp [ha0]
  have hcont := (analyticAt_preparedTailSeq r a ha).continuousAt
  have hopen : Metric.ball (0 : L1Sequence) 1 ∈
      nhds (preparedTailSeq r d a 0) := by
    rw [hzero]
    exact Metric.isOpen_ball.mem_nhds (by simp)
  filter_upwards [hcont.eventually hopen] with z hz
  simpa [Metric.mem_ball, dist_zero_right] using hz

/-- Evaluation of the normalized prepared-polynomial coefficient sequence.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem eval_preparedPolynomialSeq {n d : ℕ} (r : ℝ≥0) (hr : 0 < r)
    (a : Fin d → Base n → ℝ) (z : Base n) {w : ℝ} (hw : ‖w‖ < 1) :
    evalL1PowerSeries (monomialSeq d + preparedTailSeq r d a z) w =
      (((r : ℝ) ^ d)⁻¹) * preparedPolynomial d a (z, (r : ℝ) * w) := by
  rw [evalL1PowerSeries_add,
    evalL1PowerSeries_monomialSeq d hw]
  rw [evalL1PowerSeries_eq_sum_fin_of_highShift_eq_zero d _
    (seqHighShift_preparedTailSeq r a z) hw]
  rw [preparedPolynomial]
  simp only [preparedTailSeq_apply_fin, mul_pow]
  rw [mul_add, Finset.mul_sum]
  congr 1
  · have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    rw [← mul_assoc, inv_mul_cancel₀ (pow_ne_zero d hrC), one_mul]
  · apply Finset.sum_congr rfl
    intro i hi
    ring


end Transformer.AnalyticPreparation
