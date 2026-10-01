/-
# Real analytic preparation: PreparedFunctions

Adapted from Bochao Kong's classical-complex-wpt, revision
b4a7273fe5c9752753c52e10494097569089642d:
https://github.com/BochaoKong/classical-complex-wpt
Only the existence proof and its coefficient infrastructure are copied.
The scalar field is real; no complex root or complex differentiation result
is used. Apache-2.0 license: third_party/classical-complex-wpt/LICENSE.
-/

import Transformer.AnalyticPreparation.GermEvaluation

open Filter Finset
open scoped BigOperators ENNReal NNReal Topology

noncomputable section
set_option maxHeartbeats 5000000

namespace Transformer.AnalyticPreparation

variable {A I J : Type*} [AddCommMonoid A] [Finset.HasAntidiagonal A]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
@[simp] theorem evalL1PowerSeries_zero (a : L1Sequence) :
    evalL1PowerSeries a 0 = a 0 := by
  rw [evalL1PowerSeries_eq_tsum a (by norm_num)]
  rw [tsum_eq_single 0]
  · simp
  · intro k hk
    simp [zero_pow hk]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem evalL1PowerSeries_add (a b : L1Sequence) (w : ℝ) :
    evalL1PowerSeries (a + b) w =
      evalL1PowerSeries a w + evalL1PowerSeries b w := by
  change l1EvalOperator w (a + b) = _
  exact map_add (l1EvalOperator w) a b

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
lemma weighted_scaled_pow (r : ℝ) (hr : r ≠ 0) {d i : ℕ} (hi : i ≤ d)
    (w : ℝ) :
    r ^ (d - i) * w ^ i = r ^ d * (r⁻¹ * w) ^ i := by
  rw [pow_sub₀ r hr hi, mul_pow, inv_pow]
  ring

/-- Lower coefficients of the reconstructed distinguished polynomial. -/
noncomputable def preparationCoefficient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) (i : Fin d) : Base n → ℝ :=
  fun z ↦ -((r : ℝ) ^ (d - (i : ℕ))) * normalizedPreparationRemainder p r hr d z i

/-- Evaluation of the normalized division quotient in the original
distinguished variable. -/
noncomputable def preparationQuotientEval {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Ambient n → ℝ :=
  fun x ↦ evalL1PowerSeries (normalizedPreparationQuotient p r hr d x.1)
    ((r : ℝ)⁻¹ * x.2)

/-- The reconstructed analytic unit. -/
noncomputable def preparationUnit {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) : Ambient n → ℝ :=
  fun x ↦ (originWeightedCoeffs p r hr d * (r : ℝ)⁻¹ ^ d) *
    (preparationQuotientEval p r hr d x)⁻¹

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_preparationCoefficient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hq : AnalyticAt ℝ (normalizedPreparationRemainder p r hr d) 0)
    (i : Fin d) : AnalyticAt ℝ (preparationCoefficient p r hr d i) 0 := by
  have hi : AnalyticAt ℝ (fun z : Base n ↦ normalizedPreparationRemainder p r hr d z i) 0 :=
    by
      change AnalyticAt ℝ
        (fun z : Base n ↦ coefficientEval (i : ℕ)
          (normalizedPreparationRemainder p r hr d z)) 0
      simpa [Function.comp_def] using
        ((coefficientEval (i : ℕ)).analyticAt _).comp hq
  exact analyticAt_const.mul hi

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparationCoefficient_zero {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hrem : normalizedPreparationRemainder p r hr d 0 = 0)
    (i : Fin d) : preparationCoefficient p r hr d i 0 = 0 := by
  simp [preparationCoefficient, hrem]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparedPolynomial_preparationCoefficient {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) (hr0 : 0 < r)
    (z : Base n) (w : ℝ) :
    preparedPolynomial d (preparationCoefficient p r hr d) (z, w) =
      (r : ℝ) ^ d *
        ((r : ℝ)⁻¹ ^ d * w ^ d -
          ∑ i : Fin d, normalizedPreparationRemainder p r hr d z i *
            ((r : ℝ)⁻¹ * w) ^ (i : ℕ)) := by
  have hrC : (r : ℝ) ≠ 0 := by exact_mod_cast hr0.ne'
  have hrpow : (r : ℝ) ^ d * (r : ℝ)⁻¹ ^ d = 1 := by
    rw [← mul_pow, mul_inv_cancel₀ hrC, one_pow]
  have hlead : (r : ℝ) ^ d * ((r : ℝ)⁻¹ ^ d * w ^ d) = w ^ d := by
    rw [← mul_assoc, hrpow, one_mul]
  rw [preparedPolynomial]
  simp only [preparationCoefficient]
  rw [mul_sub, hlead, Finset.mul_sum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  have hs := weighted_scaled_pow (r : ℝ) hrC i.isLt.le w
  calc
    (-((r : ℝ) ^ (d - (i : ℕ))) *
          normalizedPreparationRemainder p r hr d z i) * w ^ (i : ℕ) =
        -(normalizedPreparationRemainder p r hr d z i *
          ((r : ℝ) ^ (d - (i : ℕ)) * w ^ (i : ℕ))) := by ring
    _ = -(normalizedPreparationRemainder p r hr d z i *
          ((r : ℝ) ^ d * ((r : ℝ)⁻¹ * w) ^ (i : ℕ))) := by rw [hs]
    _ = -((r : ℝ) ^ d *
          (normalizedPreparationRemainder p r hr d z i *
            ((r : ℝ)⁻¹ * w) ^ (i : ℕ))) := by ring

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_preparationQuotientEval {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hq : AnalyticAt ℝ (normalizedPreparationQuotient p r hr d) 0) :
    AnalyticAt ℝ (preparationQuotientEval p r hr d) 0 := by
  exact Transformer.AnalyticPreparation.AnalyticAt.evalL1PowerSeries hq (r : ℝ)⁻¹

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparationQuotientEval_zero {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ) :
    preparationQuotientEval p r hr d 0 =
      normalizedPreparationQuotient p r hr d 0 0 := by
  simp [preparationQuotientEval, ambient_zero_eq]

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analyticAt_preparationUnit {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hq : AnalyticAt ℝ (normalizedPreparationQuotient p r hr d) 0)
    (hq0 : normalizedPreparationQuotient p r hr d 0 0 = 1) :
    AnalyticAt ℝ (preparationUnit p r hr d) 0 := by
  have hQ := analyticAt_preparationQuotientEval p r hr d hq
  have hQ0 : preparationQuotientEval p r hr d 0 ≠ 0 := by
    rw [preparationQuotientEval_zero, hq0]
    exact one_ne_zero
  exact analyticAt_const.mul (hQ.inv hQ0)

/-- Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem preparationUnit_zero_ne {n : ℕ}
    (p : FormalMultilinearSeries ℝ (Ambient n) ℝ) (r : ℝ≥0)
    (hr : (r : ℝ≥0∞) < p.radius) (d : ℕ)
    (hr0 : 0 < r) (htop : originWeightedCoeffs p r hr d ≠ 0)
    (hq0 : normalizedPreparationQuotient p r hr d 0 0 = 1) :
    preparationUnit p r hr d 0 ≠ 0 := by
  rw [preparationUnit, preparationQuotientEval_zero, hq0, inv_one, mul_one]
  exact mul_ne_zero htop (pow_ne_zero _ (inv_ne_zero (by exact_mod_cast hr0.ne')))

end Transformer.AnalyticPreparation
