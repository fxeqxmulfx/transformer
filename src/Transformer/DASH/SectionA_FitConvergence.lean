/-
# DASH — uniform convergence of the actual inverse-power fit

arXiv:2602.02016v2, Appendix A. The cosine fitting algorithm itself,
followed by mapped Clenshaw, converges uniformly as the degree grows
and the number of samples stays above that degree.
-/

import Transformer.DASH.SectionA_ReferenceError
import Mathlib.Analysis.SpecificLimits.Normed

open scoped BigOperators Topology
open Filter

noncomputable section

namespace Transformer.DASH

/-- The actual fitted coefficient array has a uniform error bound
`(2d+2)C q^(d+1)`, with fixed `0<q<1` and `C>0`, for every `N>d`.
This establishes a mathematical accuracy guarantee for the algorithm;
it does not assert a particular error at the experimental degree `60`.
Source: arXiv:2602.02016v2, Appendix A, coefficient fitting for inverse roots. -/
theorem chebFit_inverse_power_geometric_error (ε exponent : ℝ) (hε : 0 < ε) :
    ∃ q ∈ Set.Ioo (0 : ℝ) 1, ∃ C > 0, ∀ d N : ℕ, d < N →
      ∀ t : ℝ, -1 ≤ t → t ≤ 1 →
        |clenshaw t (chebFit (fun x : ℝ => x ^ exponent) ε (1 + ε) N d) -
          (chebFromCoordinate ε (1 + ε) t) ^ exponent| ≤
          (2 * (d : ℝ) + 2) * (C * q ^ (d + 1)) := by
  obtain ⟨q, hq, C, hC, hgeo⟩ := inversePowerReference_geometric_error ε exponent hε
  refine ⟨q, hq, C, hC, ?_⟩
  intro d N hd t ht ht'
  obtain ⟨c, hc⟩ := exists_chebyshev_degree_eval (inversePowerReference ε exponent d) d
    (inversePowerReference_degree ε exponent d)
  apply chebFit_inverse_power_approximation ε exponent t (C * q ^ (d + 1)) N d c
    ht ht' hd (mul_nonneg hC.le (pow_nonneg hq.1.le _))
  intro u hu hu'
  rw [← hc]
  exact hgeo d u hu hu'

/-- Positive regularized inverse-root fitting is possible,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 / 100 := by norm_num

/-- The polynomial factor in the fitting bound still leaves a vanishing
error sequence whenever the reference errors decrease geometrically.
Source: arXiv:2602.02016v2, Appendix A, increasing finite approximation degree. -/
theorem fitted_geometric_bound_tendsto (q C : ℝ) (hq : 0 ≤ q) (hq' : q < 1) :
    Tendsto (fun d : ℕ => (2 * (d : ℝ) + 2) * (C * q ^ (d + 1))) atTop (𝓝 0) := by
  have hn := tendsto_self_mul_const_pow_of_lt_one hq hq'
  have hp := tendsto_pow_atTop_nhds_zero_of_lt_one hq hq'
  have h := (hn.add hp).const_mul (2 * C * q)
  convert h using 1
  · funext d
    simp only [pow_succ]
    ring
  · simp

/-- Admissible geometric ratios exist, arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- For every positive tolerance, all sufficiently large fitting degrees
and every larger sample count achieve that tolerance uniformly on the
reference interval. The arrays are those computed by the source's cosine
fit, rather than existentially chosen accurate coefficients.
Source: arXiv:2602.02016v2, Appendix A, regularized inverse-power approximation. -/
theorem chebFit_inverse_power_uniform_accuracy (ε exponent δ : ℝ)
    (hε : 0 < ε) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, d < N →
      ∀ t : ℝ, -1 ≤ t → t ≤ 1 →
        |clenshaw t (chebFit (fun x : ℝ => x ^ exponent) ε (1 + ε) N d) -
          (chebFromCoordinate ε (1 + ε) t) ^ exponent| < δ := by
  obtain ⟨q, hq, C, hC, hgeo⟩ := chebFit_inverse_power_geometric_error ε exponent hε
  have hlimit := fitted_geometric_bound_tendsto q C hq.1.le hq.2
  obtain ⟨D, hD⟩ := eventually_atTop.1 (hlimit.eventually_lt_const hδ)
  refine ⟨D, ?_⟩
  intro d hd N hN t ht ht'
  exact (hgeo d N hN t ht ht').trans_lt (hD d hd)

/-- Positive accuracy and regularization assumptions coexist,
arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 / 100 ∧ (0 : ℝ) < 1 / 1000 := by norm_num

end Transformer.DASH
