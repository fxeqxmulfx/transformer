/-
# The cap-exit variance estimate fails at the initial time

ArXiv:2410.06833v1, §5, proof of `thm: metastability MF`, Step 2,
`eq: v.small` claims `V(T_*(q,λ)) ≤ exp(-λβ)` for `8ε < λ < γ`.
The stopping set instead uses the threshold `2 exp(-λβ)`.

Use `β = 1000`, `ε = 1/10000`, `λ = log(15000)/1000`, one cap
and a stationary Dirac law at its center on the circle. Its genuine cap
minimizer starts on the boundary, so `η(0) = 1-ε` and `V(0) = ε`.
Time zero is in the stopping set and every time in that set is nonnegative.
Thus `T_* = 0`, whereas `V(T_*) = 1/10000 > 1/15000 = exp(-λβ)`.

The single-cap convention is the existing `alphaDist_one`: `α = 0`
when there are no pairs of distinct caps. The lower bound on `γ` below
also verifies its asymptotic nondegeneracy for fixed `ε` and large `β`.
The full counterexample, including every hypothesis of the old
`variance_small` statement, is in `MeanFieldCapExit`.
-/

import Transformer.Metastability.Section5_DiracCaps
import Transformer.Metastability.DirectProofWitness
import Mathlib.Analysis.Complex.ExponentialBounds

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Metastability

open Perspective

/-- A numerical lower bound needed for `8ε < λ` in the counterexample to `eq: v.small`.

Source: arXiv:2410.06833v1, §5. -/
theorem log_15000_gt_one : (1 : ℝ) < Real.log 15000 := by
  apply (Real.lt_log_iff_exp_lt (by norm_num)).mpr
  exact Real.exp_one_lt_three.trans (by norm_num)

/-- A numerical upper bound on the logarithmic correction in `eq: gamma.mf`.

Source: arXiv:2410.06833v1, §5. -/
theorem log_20000_lt_fifteen : Real.log 20000 < (15 : ℝ) := by
  apply (Real.log_lt_iff_lt_exp (by norm_num)).mpr
  have hp : (2 : ℝ) ^ 15 < Real.exp 1 ^ 15 :=
    pow_lt_pow_left₀ Real.exp_one_gt_two (by norm_num) (by norm_num)
  have he : Real.exp (15 : ℝ) = Real.exp 1 ^ 15 := by
    simp [← Real.exp_nat_mul]
  rw [he]
  exact lt_trans (by norm_num : (20000 : ℝ) < 2 ^ 15) hp

/-- The claimed upper bound in `eq: v.small` equals `1/15000` for the chosen parameters.

Source: arXiv:2410.06833v1, §5. -/
theorem exp_neg_variance_counter_lambda :
    Real.exp (-(Real.log 15000 / 1000) * 1000) = (1 / 15000 : ℝ) := by
  have h : -(Real.log 15000 / 1000) * 1000 = -Real.log 15000 := by ring
  rw [h, Real.exp_neg, Real.exp_log (by norm_num)]
  norm_num

/-- The counterexample satisfies the complete interval condition `8ε < λ < γ` of `eq: v.small`.

Source: arXiv:2410.06833v1, §5. -/
theorem varianceCounterLambda_mem (d : ℕ) (w : SSphere d) :
    8 * (1 / 10000 : ℝ) < Real.log 15000 / 1000 ∧
      Real.log 15000 / 1000 < γβ 1 1000 (αDist d 1 (fun _ => w) (1 / 10000)) (1 / 10000) := by
  rw [alphaDist_one, γβ]
  have hl := log_15000_gt_one
  have hu := log_20000_lt_fifteen
  have hl' : Real.log 15000 < Real.log 20000 := Real.log_lt_log (by norm_num) (by norm_num)
  norm_num
  constructor <;> linarith

/-- The counterexample also has `γ(β) > 1/2` for every `β ≥ 1000`.
Thus the asymptotic `γ(β) = Ω(1)` requirement of `def: init_measure_MF` holds in its regime.

Source: arXiv:2410.06833v1, §5. -/
theorem variance_counter_gamma_lower_bound (d : ℕ) (w : SSphere d) (β : ℝ)
    (hβ : 1000 ≤ β) :
    (1 / 2 : ℝ) < γβ 1 β (αDist d 1 (fun _ => w) (1 / 10000)) (1 / 10000) := by
  have hi : β⁻¹ ≤ (1 / 1000 : ℝ) := by
    simpa using inv_anti₀ (by norm_num : (0 : ℝ) < 1000) hβ
  have hm : β⁻¹ * Real.log 20000 ≤ (1 / 1000 : ℝ) * 15 :=
    mul_le_mul hi (le_of_lt log_20000_lt_fifteen) (Real.log_nonneg (by norm_num))
      (by norm_num)
  rw [alphaDist_one, γβ]
  norm_num
  linarith

/-- The stopping threshold preceding `eq: v.small` is already met at time zero.

Source: arXiv:2410.06833v1, §5. -/
theorem zero_mem_varianceCounterExitSet :
    (0 : ℝ) ∈ capExitSet 2 1000 (Real.log 15000 / 1000) (1 / 10000) 1
      (fun _ => varianceCounterCentre) (diracProb 2 varianceCounterCentre)
      (diracFlow 2 varianceCounterCentre) 0
      (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary) := by
  refine ⟨zero_mem_escapeWindow_diracFlow 2 1 _ _ (by norm_num), ?_⟩
  rw [capMin_diracFlow_zero 2 _ _ _ varianceCounterBoundary_coordinate,
    capVariance_diracFlow_zero 2 _ _ _ (by norm_num) varianceCounterBoundary_coordinate,
    exp_neg_variance_counter_lambda]
  have he : Real.exp (-(1 - (1 - (1 / 10000 : ℝ))) * 1000) ≤ 1 :=
    Real.exp_le_one_iff.mpr (by norm_num)
  calc
    (1 - (1 / 10000 : ℝ)) * (1 / 10000) * Real.exp (-(1 - (1 - 1 / 10000)) * 1000)
      ≤ (1 - (1 / 10000 : ℝ)) * (1 / 10000) * 1 :=
        mul_le_mul_of_nonneg_left he (by norm_num)
    _ ≤ 2 * (1 / 15000 : ℝ) := by norm_num

/-- The stopping time preceding `eq: v.small` is exactly zero, with a nonempty stopping set.

Source: arXiv:2410.06833v1, §5. -/
theorem varianceCounterExitTime_zero :
    sInf (capExitSet 2 1000 (Real.log 15000 / 1000) (1 / 10000) 1
      (fun _ => varianceCounterCentre) (diracProb 2 varianceCounterCentre)
      (diracFlow 2 varianceCounterCentre) 0
      (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary)) = 0 := by
  exact IsLeast.csInf_eq ⟨zero_mem_varianceCounterExitSet, fun t ht => ht.1.1⟩

/-- The conclusion of `eq: v.small` fails strictly for the explicit stationary Dirac solution.

Source: arXiv:2410.06833v1, §5. -/
theorem varianceCounter_strict_failure :
    Real.exp (-(Real.log 15000 / 1000) * 1000) <
      capVariance 2 (diracProb 2 varianceCounterCentre) (diracFlow 2 varianceCounterCentre)
        varianceCounterCentre (1 / 10000)
        (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary)
        (sInf (capExitSet 2 1000 (Real.log 15000 / 1000) (1 / 10000) 1
          (fun _ => varianceCounterCentre) (diracProb 2 varianceCounterCentre)
          (diracFlow 2 varianceCounterCentre) 0
          (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary))) := by
  rw [varianceCounterExitTime_zero, capVariance_diracFlow_zero 2 _ _ _ (by norm_num)
    varianceCounterBoundary_coordinate, exp_neg_variance_counter_lambda]
  norm_num

/-- What the stopping threshold preceding `eq: v.small` actually implies:
`V(t) ≤ 2 exp(-cβ) / (η(t) exp(-(1-η(t))β))` when `η(t) > 0`.
The factor and exponential in the denominator cannot be dropped; the strict
counterexample above shows why. This is a conditional algebraic consequence
of membership in the stopping set, not an assertion that any unproved exit
claim supplies that membership.

Source: arXiv:2410.06833v1, §5. -/
theorem variance_le_of_mem_capExitSet (d k : ℕ) (β c ε t : ℝ)
    (w : Idx k → SSphere d) (μ₀ : ProbSphere d) (Φ : ℝ → SSphere d → SSphere d)
    (q : Idx k) (x : ℝ → SSphere d)
    (ht : t ∈ capExitSet d β c ε k w μ₀ Φ q x)
    (hη : 0 < capMin d Φ (w q) ε t) :
    capVariance d μ₀ Φ (w q) ε x t ≤
      2 * Real.exp (-c * β) /
        (capMin d Φ (w q) ε t * Real.exp (-(1 - capMin d Φ (w q) ε t) * β)) := by
  apply (le_div_iff₀ (mul_pos hη (Real.exp_pos _))).mpr
  calc
    capVariance d μ₀ Φ (w q) ε x t *
        (capMin d Φ (w q) ε t * Real.exp (-(1 - capMin d Φ (w q) ε t) * β))
      = capMin d Φ (w q) ε t * capVariance d μ₀ Φ (w q) ε x t *
          Real.exp (-(1 - capMin d Φ (w q) ε t) * β) := by ring
    _ ≤ 2 * Real.exp (-c * β) := ht.2

/-- The lower bound on `γ` has an attained temperature hypothesis. -/
example : (1000 : ℝ) ≤ 1000 ∧ (1 / 2 : ℝ) <
    γβ 1 1000 (αDist 2 1 (fun _ => varianceCounterCentre) (1 / 10000)) (1 / 10000) :=
  ⟨le_rfl, variance_counter_gamma_lower_bound 2 _ 1000 le_rfl⟩

/-- Both hypotheses of the valid threshold bound hold for the counterexample. -/
example : (0 : ℝ) ∈ capExitSet 2 1000 (Real.log 15000 / 1000) (1 / 10000) 1
    (fun _ => varianceCounterCentre) (diracProb 2 varianceCounterCentre)
    (diracFlow 2 varianceCounterCentre) 0
    (fun t => diracFlow 2 varianceCounterCentre t varianceCounterBoundary) ∧
    0 < capMin 2 (diracFlow 2 varianceCounterCentre) varianceCounterCentre (1 / 10000) 0 := by
  refine ⟨zero_mem_varianceCounterExitSet, ?_⟩
  rw [capMin_diracFlow_zero 2 _ _ _ varianceCounterBoundary_coordinate]
  norm_num

end Transformer.Metastability
