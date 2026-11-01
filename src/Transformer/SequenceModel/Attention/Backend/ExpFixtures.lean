module

public import Transformer.SequenceModel.Attention.Backend.Causal

/-!
# Scalar activation and two-token SDPA boxes for the captured backend

Source: `review/flat_spectrum.md`, *Concrete backend contracts*, K6;
`review/backend_contract_arithmetic.py`. The common rational witness is
the degree-eleven polynomial at -1/2. The SDPA fixture has actual scaled
scores [0,1/2], two distinct value vectors, and the original causal mask.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend

open GPTMini Stability Landscape.Neural
open scoped BigOperators

/-- The exactly evaluated exponential polynomial used by the capture checker. -/
def halfExp : ℝ := 49583642701 / 81749606400

/-- K6: the stored rational exponential witness equals the specified polynomial. -/
theorem halfExp_polynomial : halfExp = expPolynomial (-(1 / 2)) 12 := by
  norm_num [halfExp, expPolynomial, Finset.sum_range_succ]

/-- K6: the analytic error of this rational witness is below 1e-10. -/
theorem halfExp_error : |halfExp - Real.exp (-(1 / 2))| ≤ 1 / 10000000000 := by
  have h := exponential_error (-(1 / 2)) halfExp 12 (by norm_num) (by decide) 0
    (by rw [← halfExp_polynomial]; norm_num)
  exact h.trans (by norm_num)

/-- K6: the observed scalar SiLU output box implies an absolute error below 2e-6. -/
theorem silu_fixture (z : ℝ) (hz : |z - (1 / 2) / (1 + halfExp)| ≤ 1 / 1000000) :
    |z - silu (1 / 2)| ≤ 2 / 1000000 := by
  have h := silu_residual_error (1 / 2) z halfExp (1 / 1000000) (1 / 10000000000)
    (by norm_num [halfExp]) halfExp_error hz
  exact h.trans (by norm_num)

/-- The two visible values, matching the dimension-four source SDPA input. -/
def sdpaValues : Fin 2 → EucSpace 4 :=
  ![EuclideanSpace.single 0 1, EuclideanSpace.single 1 1]

/-- Scores from queries [1,0,0,0], keys [0,0,0,0] and [1,0,0,0], scaled by 1/sqrt(4). -/
def sdpaScores : Fin 2 → Fin 2 → ℝ := !![0, 1 / 2; 0, 1 / 2]

/-- Rational normalized-average witness, independent of the captured SDPA outputs. -/
def sdpaReference : EucSpace 4 :=
  WithLp.toLp 2 ![1 - 1 / (1 + halfExp), 1 / (1 + halfExp), 0, 0]

/-- K6: the SDPA output box implies the actual source causal-average
contract, with Euclidean error at most 3e-6. -/
theorem sdpa_fixture (z : EucSpace 4)
    (hz : ∀ c, |z c - sdpaReference c| ≤ 1 / 1000000) :
    ‖z - ∑ j, causalWeight sdpaScores 1 j • sdpaValues j‖ ≤ 3 / 1000000 := by
  let a : Fin 2 → ℝ := ![halfExp, 1]
  let b : Fin 2 → ℝ := ![Real.exp (-(1 / 2)), 1]
  have hA : 0 < ∑ j, a j := by norm_num [a, halfExp, Fin.sum_univ_succ]
  have hB : 0 < ∑ j, b j := by simp [b, Fin.sum_univ_succ]; positivity
  have he : ∑ j, |a j - b j| ≤ 1 / 10000000000 := by
    simpa [a, b, Fin.sum_univ_succ] using halfExp_error
  have hv (j) : ‖sdpaValues j‖ ≤ 1 := by fin_cases j <;> simp [sdpaValues]
  have href : (∑ j, (a j / (∑ k, a k)) • sdpaValues j) = sdpaReference := by
    ext c
    fin_cases c <;> norm_num [a, sdpaValues, sdpaReference, halfExp, Fin.sum_univ_succ]
  have hm : ‖z - ∑ j, (a j / (∑ k, a k)) • sdpaValues j‖ ≤ 2 / 1000000 := by
    rw [href]
    have h := norm_le_coordinates (z - sdpaReference) (1 / 1000000) (by norm_num) hz
    norm_num at h ⊢
    exact h
  have hw (j : Fin 2) : b j / (∑ k, b k) = causalWeight sdpaScores 1 j := by
    have hp := Real.exp_pos (1 / 2)
    have hi : Finset.Iic (1 : Fin 2) = Finset.univ := by decide
    fin_cases j <;> simp [b, causalWeight, sdpaScores, hi, Fin.sum_univ_succ, Real.exp_neg] <;>
      field_simp
  have h := average_residual_error a b sdpaValues z hA hB
    (by intro j; fin_cases j <;> simp [b, Real.exp_nonneg]) (1 / 10000000000) 1
    (2 / 1000000) (by norm_num) he hv hm
  simp only [hw] at h
  exact h.trans (by norm_num [a, halfExp, Fin.sum_univ_succ])

example := silu_fixture (311229676 / 1000000000) (by norm_num [halfExp])
example := sdpa_fixture sdpaReference (by intro c; simp)

end Transformer.SequenceModel.Attention.Backend
