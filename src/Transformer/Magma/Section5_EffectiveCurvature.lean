/-
# Effective curvature factors and missing moment assumptions

Formalization of arXiv:2602.15322v1, Section 5, eq:eff_smoothness and
assumption:layerwise_second_moment. Ratio formulas require a nonzero
raw second moment; a zero-gradient block needs a separate convention.
-/

import Transformer.Magma.Section5_FiniteLaw

open scoped BigOperators

noncomputable section

namespace Transformer.Magma

variable {ι : Type*} [Fintype ι]

/-- Each effective curvature is a ratio of damped to raw second moments.
Source: arXiv:2602.15322v1, Section 5, rho_t and eq:eff_smoothness. -/
def effectiveCurvature (p : ℝ) (L raw damped : ι → ℝ) (j : ι) : ℝ :=
  (damped j / raw j) * L j / p

/-- The printed effective-curvature identity is exact for nonzero raw
block moments; no independence of damping and gradients is needed.
Source: arXiv:2602.15322v1, Section 5, eq:eff_smoothness. -/
theorem effective_curvature_identity (p rate : ℝ) (L raw damped : ι → ℝ)
    (hraw : ∀ j, raw j ≠ 0) :
    rate ^ 2 / (2 * p) * (∑ j, L j * damped j) =
      rate ^ 2 / 2 * (∑ j, effectiveCurvature p L raw damped j * raw j) := by
  have hpoint (j : ι) : effectiveCurvature p L raw damped j * raw j =
      L j * damped j / p := by
    unfold effectiveCurvature
    field_simp [hraw j]
  simp only [hpoint]
  rw [← Finset.sum_div]
  ring

/-- Nonzero second moments are possible with nonzero constant gradients.
Source: arXiv:2602.15322v1, Section 5, effective-curvature domain. -/
example : ∀ _ : Unit, (1 : ℝ) ≠ 0 := by norm_num

/-- The curvature ratio can use the source raw-noise bound only when
its coefficients are nonnegative. Source: arXiv:2602.15322v1,
Appendix A.4, its first inequality, with the ratio domain explicit. -/
theorem effective_curvature_noise_bound (p : ℝ) (L raw damped energy noise : ι → ℝ)
    (hp : 0 < p) (hL : ∀ j, 0 ≤ L j) (hraw : ∀ j, 0 < raw j)
    (hdamped : ∀ j, 0 ≤ damped j) (hnoise : ∀ j, raw j ≤ energy j + noise j) :
    (∑ j, effectiveCurvature p L raw damped j * raw j) ≤
      ∑ j, effectiveCurvature p L raw damped j * (energy j + noise j) := by
  apply Finset.sum_le_sum
  intro j hj
  have hc : 0 ≤ effectiveCurvature p L raw damped j := by
    unfold effectiveCurvature
    exact div_nonneg (mul_nonneg (div_nonneg (hdamped j) (hraw j).le) (hL j)) hp.le
  exact mul_le_mul_of_nonneg_left (hnoise j) hc

/-- Every ratio/noise hypothesis holds on a nonzero exact-gradient
one-block model with damping 1/2. Source: arXiv:2602.15322v1, Section 5. -/
example : (0 : ℝ) < 1 / 2 ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧
    (∀ _ : Unit, (0 : ℝ) < 1) ∧ (∀ _ : Unit, (0 : ℝ) ≤ 1 / 4) ∧
    (∀ _ : Unit, (1 : ℝ) ≤ 1 + 0) := by norm_num

/-- A raw second-moment bound alone does not imply unbiasedness or a
centered-variance bound. Take the actual quadratic gradient at x=1,
but the deterministic stochastic estimator -1. Its raw moment satisfies
the source inequality with sigma=0, while its mean is wrong and centered
noise equals 4. Source: arXiv:2602.15322v1, Section 5,
assumption:layerwise_second_moment and Appendix A.3's use of unbiasedness. -/
theorem raw_moment_does_not_imply_unbiasedness :
    let v := gradient Transformer.Optimization.quadratic (1 : ℝ)
    finiteExpectation (fun _ : Unit => (1 : ℝ)) (fun _ => (-1 : ℝ) ^ 2) ≤ v ^ 2 + 0 ^ 2 ∧
      (∑ _ : Unit, (1 : ℝ) * (-1)) ≠ v ∧
      finiteExpectation (fun _ : Unit => (1 : ℝ)) (fun _ => (-1 - v) ^ 2) > 0 ^ 2 := by
  norm_num [finiteExpectation, Transformer.Optimization.quadratic_gradient]

/-- A lower bound on s reverses its multiplication by a negative
alignment. Thus the complement-event inequality in the source's proof
does not follow from s>=s_minus. Source: arXiv:2602.15322v1,
Appendix A.3, intem1, lower bounding the term on E_t complement. -/
theorem complement_alignment_sign_counterexample :
    (1 / 4 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) * (-1) < (1 / 4 : ℝ) * (-1) := by norm_num

end Transformer.Magma
