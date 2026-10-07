/-
# Numerical separation of the stopping threshold from early Dirac trajectories

ArXiv:2410.06833v1, §5, `claim: de sortie de cap` in Step 2 of the
proof of `thm: metastability MF`. For one stationary Dirac cap take
`β = 1000000`, `ε = 1/10000` and `c = 801/1000000`.
The claimed bound on `T_*` is `4ε exp(1) < 1`.

On the entire interval `[0,1]`, the genuine cap trajectory has
`η ≥ 9999/10000`, `1/1000000 ≤ V = 1-η ≤ 1/10000`.
Thus `η V exp(-V β) ≥ exp(-100)/2000000 > 2 exp(-801)`:
the stopping threshold has not yet been met. All inequalities below
are proved over real numbers, without rounded exponential evaluations.

In fact the claimed bound is less than `3/2500 = 0.0012`.
The comparison with time one therefore has a substantial strict margin.

The estimate `exp(100) ≥ 11^10` supplies the exact comparison between
the two small exponentials in the stopping inequality.

The remaining geometric and measure hypotheses, and nonemptiness of
the stopping set, are established in the adjoining modules. These
numerical facts do not assume the source's derivative estimate.
-/

import Transformer.Metastability.Section5_DiracCapEvolution
import Mathlib.Analysis.Complex.ExponentialBounds

open Real
namespace Transformer.Metastability

/-- Bounds on the actual cap minimum and variance throughout `[0,1]`.
The lower variance bound comes from the explicit positive denominator.

Source: arXiv:2410.06833v1, §5, auxiliary to `claim: de sortie de cap`. -/
theorem capExitCounter_deficit_bounds (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    1 / 1000000 ≤ 1 - diracAlong t (1 - 1 / 10000) ∧
    1 - diracAlong t (1 - 1 / 10000) ≤ 1 / 10000 ∧
    9999 / 10000 ≤ diracAlong t (1 - 1 / 10000) ∧
    diracAlong t (1 - 1 / 10000) ≤ 1 := by
  have hc : (1 - 1 / 10000 : ℝ) ∈ Set.Icc (-1 : ℝ) 1 := by norm_num
  have hf := diracAlong_forward_bounds t _ hc ht.1
  have he : Real.exp (2 * t) ≤ 9 := by
    have hmono := Real.exp_le_exp.mpr (show 2 * t ≤ 2 by linarith [ht.2])
    have hsq : Real.exp 2 = Real.exp 1 ^ 2 := by
      simp [← Real.exp_nat_mul]
    rw [hsq] at hmono
    nlinarith [Real.exp_pos 1, Real.exp_one_lt_three]
  have hD : diracDen t (1 - 1 / 10000) ≤ 200 := by
    dsimp [diracDen]
    nlinarith
  have hd := diracDen_pos t _ hc
  refine ⟨?_, by linarith [hf.1], by linarith [hf.1], hf.2⟩
  rw [one_sub_diracAlong t _ hc, le_div_iff₀ hd]
  nlinarith

/-- A coarse exact exponential estimate used to separate the stopping thresholds.

Source: arXiv:2410.06833v1, §5, auxiliary to `claim: de sortie de cap`. -/
theorem capExitCounter_exp_hundred : (4000000 : ℝ) < Real.exp 100 := by
  have hten : (11 : ℝ) ≤ Real.exp 10 := by
    have h := Real.add_one_le_exp 10
    norm_num at h
    exact h
  have hp := pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 11) hten 10
  have hexp : Real.exp 100 = Real.exp 10 ^ 10 := by
    have h := Real.exp_nat_mul 10 10
    norm_num at h
    exact h
  rw [← hexp] at hp
  norm_num at hp
  linarith

/-- The proposed threshold is strictly below the early-time lower bound.

Source: arXiv:2410.06833v1, §5, auxiliary to `claim: de sortie de cap`. -/
theorem capExitCounter_neg_exp :
    2 * Real.exp (-801) < (1 / 2000000 : ℝ) * Real.exp (-100) := by
  have hinv : Real.exp (-100) < (1 / 4000000 : ℝ) := by
    rw [Real.exp_neg]
    have h := (inv_lt_inv₀ (Real.exp_pos 100)
      (by norm_num : (0 : ℝ) < 4000000)).mpr capExitCounter_exp_hundred
    norm_num at h ⊢
    exact h
  have he2 : Real.exp (-200) = Real.exp (-100) ^ 2 := by
    have h := Real.exp_nat_mul (-100) 2
    norm_num at h
    exact h
  have hmono : Real.exp (-801) ≤ Real.exp (-200) := Real.exp_le_exp.mpr (by norm_num)
  rw [he2] at hmono
  have hm := mul_lt_mul_of_pos_right hinv (Real.exp_pos (-100))
  nlinarith

/-- No time in `[0,1]` satisfies the source's stopping inequality for these parameters.
The variance here is the exact variance about the transported cap boundary.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_threshold_before_one (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    2 * Real.exp (-(801 / 1000000 : ℝ) * 1000000) <
      diracAlong t (1 - 1 / 10000) * (1 - diracAlong t (1 - 1 / 10000)) *
        Real.exp (-(1 - diracAlong t (1 - 1 / 10000)) * 1000000) := by
  obtain ⟨hlo, hhi, hClo, hChi⟩ := capExitCounter_deficit_bounds t ht
  have hC : (1 / 2 : ℝ) ≤ diracAlong t (1 - 1 / 10000) := by linarith
  have hprod : (1 / 2000000 : ℝ) ≤
      diracAlong t (1 - 1 / 10000) * (1 - diracAlong t (1 - 1 / 10000)) := by
    have h := mul_le_mul hC hlo (by norm_num : (0 : ℝ) ≤ 1 / 1000000)
      (by linarith : 0 ≤ diracAlong t (1 - 1 / 10000))
    norm_num at h ⊢
    exact h
  have hexp : Real.exp (-100) ≤
      Real.exp (-(1 - diracAlong t (1 - 1 / 10000)) * 1000000) :=
    Real.exp_le_exp.mpr (by linarith)
  have hle := mul_le_mul hprod hexp (Real.exp_pos (-100)).le (by linarith)
  norm_num only [neg_div, div_mul_cancel₀, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true] at *
  exact capExitCounter_neg_exp.trans_le hle

/-- The claimed bound is strictly below `0.0012`, by `exp(1) < 3`.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_claimed_bound_lt_twelve_thousandths :
    4 * (1 / 10000 : ℝ) / 1 *
      Real.exp (((801 / 1000000) - 8 * (1 / 10000)) * 1000000) < 3 / 2500 := by
  norm_num
  nlinarith [Real.exp_one_lt_three]

/-- The claimed upper bound on `T_*` is smaller than the interval on which
we have proved that the stopping condition fails.

Source: arXiv:2410.06833v1, §5, `claim: de sortie de cap`. -/
theorem capExitCounter_claimed_bound_lt_one :
    4 * (1 / 10000 : ℝ) / 1 *
      Real.exp (((801 / 1000000) - 8 * (1 / 10000)) * 1000000) < 1 := by
  exact capExitCounter_claimed_bound_lt_twelve_thousandths.trans (by norm_num)

/-- Positive time satisfies the interval hypotheses of both early-time estimates. -/
example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧
    1 / 1000000 ≤ 1 - diracAlong (1 / 2) (1 - 1 / 10000) ∧
    1 - diracAlong (1 / 2) (1 - 1 / 10000) ≤ 1 / 10000 ∧
    9999 / 10000 ≤ diracAlong (1 / 2) (1 - 1 / 10000) ∧
    diracAlong (1 / 2) (1 - 1 / 10000) ≤ 1 :=
  ⟨by norm_num, capExitCounter_deficit_bounds (1 / 2) (by norm_num)⟩

/-- The same interior time gives a strict violation of the stopping inequality. -/
example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧
    2 * Real.exp (-(801 / 1000000 : ℝ) * 1000000) <
      diracAlong (1 / 2) (1 - 1 / 10000) *
        (1 - diracAlong (1 / 2) (1 - 1 / 10000)) *
        Real.exp (-(1 - diracAlong (1 / 2) (1 - 1 / 10000)) * 1000000) :=
  ⟨by norm_num, capExitCounter_threshold_before_one (1 / 2) (by norm_num)⟩

end Transformer.Metastability
