/-
# Double-sum algebra for the actual angular energy

ArXiv:2410.06833v1, §3.2, `𝖤_β` and `lem: PL.borjan`.
Cancelling `exp β` identifies the energy with its normalized pair-potential
sum. Oddness of the true interaction potential identifies the linear
term of a pairwise quadratic bound with the actual angular gradient.
-/

import Transformer.Metastability.Section3_ScalarTaylor
import Transformer.Metastability.AngularEnergy

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **The true energy is the normalized pair-potential double sum.**
Source: arXiv:2410.06833v1, §3.2, the definition of `𝖤_β`. -/
theorem angularEβ_eq_pairPotential (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : β ≠ 0)
    (Θ : Idx n → ℝ) :
    angularEβ n β Θ = (1 / (2 * (n : ℝ) ^ 2)) *
      ∑ i : Idx n, ∑ j : Idx n, angularPairPotential β (Θ i - Θ j) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  unfold angularEβ
  rw [Finset.mul_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.mul_sum, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  unfold angularPairPotential
  rw [mul_sub, mul_one, Real.exp_sub]
  field_simp

/-- **The linear term of the pairwise bound is the actual gradient pairing.**
Oddness swaps the second-coordinate contribution into the first, giving
the factor `2` in the double sum. Source: arXiv:2410.06833v1, §3.2,
the gradient of `𝖤_β` in `lem: PL.borjan`. -/
theorem pair_linear_eq_angularGrad (n : ℕ) (hn : 0 < n) (β : ℝ)
    (Θ v : Idx n → ℝ) :
    (1 / (2 * (n : ℝ) ^ 2)) * ∑ i : Idx n, ∑ j : Idx n,
        -Causal.h_pot β (Θ i - Θ j) * (v i - v j) =
      ∑ i : Idx n, angularGrad n β Θ i * v i := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  let S := ∑ i : Idx n, ∑ j : Idx n, Causal.h_pot β (Θ i - Θ j) * v i
  have hswap : (∑ i : Idx n, ∑ j : Idx n, Causal.h_pot β (Θ i - Θ j) * v j) = -S := by
    rw [Finset.sum_comm]
    calc
      _ = ∑ j : Idx n, ∑ i : Idx n, -(Causal.h_pot β (Θ j - Θ i) * v j) := by
        apply Finset.sum_congr rfl
        intro j _
        apply Finset.sum_congr rfl
        intro i _
        rw [show Θ i - Θ j = -(Θ j - Θ i) by ring, Causal.h_pot_odd, neg_mul]
      _ = -S := by simp only [Finset.sum_neg_distrib, S]
  have hlin : (∑ i : Idx n, ∑ j : Idx n,
      -Causal.h_pot β (Θ i - Θ j) * (v i - v j)) = -2 * S := by
    calc
      _ = ∑ i : Idx n, ∑ j : Idx n,
          (-(Causal.h_pot β (Θ i - Θ j) * v i) + Causal.h_pot β (Θ i - Θ j) * v j) := by
        apply Finset.sum_congr rfl
        intro i _
        apply Finset.sum_congr rfl
        intro j _
        ring
      _ = -S + -S := by
        simp only [Finset.sum_add_distrib, Finset.sum_neg_distrib, hswap, S]
      _ = -2 * S := by ring
  have hgrad : (∑ i : Idx n, angularGrad n β Θ i * v i) = -(1 / (n : ℝ) ^ 2) * S := by
    calc
      _ = ∑ i : Idx n, -(1 / (n : ℝ) ^ 2) *
          ∑ j : Idx n, Causal.h_pot β (Θ i - Θ j) * v i := by
        apply Finset.sum_congr rfl
        intro i _
        rw [← Finset.sum_mul]
        simp only [angularGrad, Causal.h_pot, mul_comm]
        ring
      _ = _ := by rw [← Finset.mul_sum]
  rw [hlin, hgrad]
  field_simp

/-- The complete algebraic input restrictions above hold at `n=2`, `β=1`.
The tuples are arbitrary, so no dynamical or cap hypothesis is hidden. -/
example : 0 < 2 ∧ (1 : ℝ) ≠ 0 ∧ (0 : ℝ) ≤ 1 :=
  ⟨by norm_num, one_ne_zero, zero_le_one⟩

/-- **The double sum of squared pair differences is at most `4n` times
the coordinate-square sum.** Source: arXiv:2410.06833v1, §3.2, the
Euclidean gradient norm and pairwise Hessian form of `lem: PL.borjan`.
The constant is deliberately a loose bound, sufficient for the PL estimate. -/
theorem sum_pair_sq_le (n : ℕ) (v : Idx n → ℝ) :
    (∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) ≤
      4 * (n : ℝ) * ∑ i : Idx n, (v i) ^ 2 := by
  calc
    _ ≤ ∑ i : Idx n, ∑ j : Idx n, (2 * (v i) ^ 2 + 2 * (v j) ^ 2) := by
      apply Finset.sum_le_sum
      intro i _
      apply Finset.sum_le_sum
      intro j _
      nlinarith [sq_nonneg (v i + v j)]
    _ = _ := by
      simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, nsmul_eq_mul]
      simp_rw [← Finset.mul_sum]
      ring

/-- **The normalized quadratic pair sum is controlled by the coordinate
square sum.** Source: arXiv:2410.06833v1, §3.2, the Hessian contribution
to the energy discrepancy in `lem: PL.borjan`. -/
theorem normalized_pair_sq_le (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (v : Idx n → ℝ) :
    (1 / (2 * (n : ℝ) ^ 2)) * (1 + β) * (∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) ≤
      2 * (1 + β) * ∑ i : Idx n, (v i) ^ 2 := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (show 1 ≤ n by omega)
  have hsum : 0 ≤ ∑ i : Idx n, (v i) ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hscaled := mul_le_mul_of_nonneg_left (sum_pair_sq_le n v)
    (by positivity : 0 ≤ (1 / (2 * (n : ℝ) ^ 2)) * (1 + β))
  have heq : (1 / (2 * (n : ℝ) ^ 2)) * (1 + β) * (4 * (n : ℝ) * ∑ i : Idx n, (v i) ^ 2) =
      (2 * (1 + β) / (n : ℝ)) * ∑ i : Idx n, (v i) ^ 2 := by field_simp; ring
  rw [heq] at hscaled
  have hcoef : 2 * (1 + β) / (n : ℝ) ≤ 2 * (1 + β) := by
    apply (div_le_iff₀ hnR).mpr
    simpa only [mul_one] using mul_le_mul_of_nonneg_left hn1 (by linarith : 0 ≤ 2 * (1 + β))
  exact hscaled.trans (mul_le_mul_of_nonneg_right hcoef hsum)

/-- **The linear gradient pairing is bounded by half the sum of the two
coordinate-square norms.** Source: arXiv:2410.06833v1, §3.2, the
Euclidean norms in `lem: PL.borjan`; this is the coordinate Young inequality. -/
theorem sum_mul_le_half_squares (n : ℕ) (g v : Idx n → ℝ) :
    (∑ i : Idx n, g i * v i) ≤
      (1 / 2 : ℝ) * ∑ i : Idx n, (g i) ^ 2 + (1 / 2 : ℝ) * ∑ i : Idx n, (v i) ^ 2 := by
  calc
    _ ≤ ∑ i : Idx n, ((1 / 2 : ℝ) * (g i) ^ 2 + (1 / 2 : ℝ) * (v i) ^ 2) :=
      Finset.sum_le_sum fun i _ => by nlinarith [sq_nonneg (g i - v i)]
    _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- **The linear part of the actual pair-energy discrepancy has the
coordinate Young bound.** Source: arXiv:2410.06833v1, §3.2, the
gradient-norm estimate of `lem: PL.borjan`. -/
theorem pair_linear_le_half_squares (n : ℕ) (hn : 0 < n) (β : ℝ) (Θ v : Idx n → ℝ) :
    (1 / (2 * (n : ℝ) ^ 2)) * ∑ i : Idx n, ∑ j : Idx n,
        -Causal.h_pot β (Θ i - Θ j) * (v i - v j) ≤
      (1 / 2 : ℝ) * ∑ i : Idx n, (angularGrad n β Θ i) ^ 2 +
        (1 / 2 : ℝ) * ∑ i : Idx n, (v i) ^ 2 := by
  rw [pair_linear_eq_angularGrad n hn]
  exact sum_mul_le_half_squares n (angularGrad n β Θ) v

end Transformer.Metastability
