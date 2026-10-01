/-
# A stable unbiased alignment-sampling model on a genuine quadratic

Scoped result for the future direction in arXiv:2602.15322v1, Section 3:
use the instantaneous score as the survival probability and rescale by
its reciprocal. The paper reports instability in its neural-network
experiments. That is not an impossibility theorem. The result below proves
conditional objective contraction on a quadratic under a uniform step bound;
it makes no claim to solve stability of that scheme for large-model training.
-/

import Transformer.Magma.Section3_DampingBounds
import Transformer.Optimization.Basic

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

/-- Exact expected loss of unbiased masking on x^2/2, using its actual
gradient. Source: arXiv:2602.15322v1, Section 3, unbiased alternative,
and Section 5, normalized SGD update. -/
theorem unbiased_quadratic_expected_loss (p rate x : ℝ) (hp : p ≠ 0) :
    maskExpectation p (fun mask : Unit → Bool =>
      quadratic (x - rate * maskCoefficient p (mask ()) * gradient quadratic x)) =
        (1 - 2 * rate + rate ^ 2 / p) * quadratic x := by
  rw [maskExpectation_coordinate p ()
    (fun bit => quadratic (x - rate * maskCoefficient p bit * gradient quadratic x))]
  simp [maskCoefficient, quadratic_gradient, quadratic]
  field_simp
  ring

/-- Unbiased quadratic masking has a genuine probability-domain witness.
Source: arXiv:2602.15322v1, Section 3, unbiased alternative. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- A positive fixed-temperature score permits an explicitly stable
unbiased variant on the actual quadratic: choose rate at most the minimum
possible sigmoid survival. This is a precise partial result for the open
research direction, not a universal neural-network convergence theorem.
Source: arXiv:2602.15322v1, Section 3, final unbiased-masking paragraph. -/
theorem unbiased_alignment_sampling_quadratic_descent (τ rate x momentum : ℝ)
    (hτ : 0 < τ) (hrate : 0 < rate) (hstep : rate ≤ Real.sigmoid (-1 / τ)) :
    let p := Real.sigmoid (cosine momentum (gradient quadratic x) / τ)
    maskExpectation p (fun mask : Unit → Bool =>
      quadratic (x - rate * maskCoefficient p (mask ()) * gradient quadratic x)) ≤
        (1 - rate) * quadratic x := by
  dsimp only
  let p := Real.sigmoid (cosine momentum (gradient quadratic x) / τ)
  have hp : 0 < p := Real.sigmoid_pos _
  have hlo : Real.sigmoid (-1 / τ) ≤ p :=
    Real.sigmoid_le (div_le_div_of_nonneg_right
      (abs_le.mp (cosine_abs_le_one momentum (gradient quadratic x))).1 hτ.le)
  have hrp := hstep.trans hlo
  rw [unbiased_quadratic_expected_loss p rate x hp.ne']
  have hcoef : 1 - 2 * rate + rate ^ 2 / p ≤ 1 - rate := by
    have hsq : rate ^ 2 / p ≤ rate := by
      apply (div_le_iff₀ hp).mpr
      nlinarith [mul_le_mul_of_nonneg_left hrp hrate.le]
    linarith
  exact mul_le_mul_of_nonneg_right hcoef (by unfold quadratic; positivity)

/-- All strict-domain and step hypotheses of the unbiased stable example
hold at a positive unit temperature.
Source: arXiv:2602.15322v1, Section 3, unbiased-masking future direction. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < Real.sigmoid (-1) / 2 ∧
    Real.sigmoid (-1) / 2 ≤ Real.sigmoid (-1 / 1) := by
  have h := Real.sigmoid_pos (-1)
  norm_num only [div_one]
  constructor
  · norm_num
  constructor <;> linarith

end Transformer.Magma
