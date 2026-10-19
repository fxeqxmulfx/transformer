module

public import Transformer.Metastability.Section3_SharpTaylor
public import Transformer.Metastability.Section3_EnergyAlgebra

/-!
# A global energy remainder bound uniform in temperature

ArXiv:2410.06833v1, §3.2, `eq: otto.attention` and the pairwise
Hessian representation preceding `eq: Ht.first.lb`. The exact identity
for the squared pair differences gives the quadratic coefficient
`1/(2n)`, including for the two-token system. This estimate concerns
the actual exponential energy and its actual angular gradient.
The pair-square identity separates relative motion from common
rotation: the square of the sum is subtracted explicitly. The bound
therefore remains valid for arbitrary lifted angle representatives.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **The exact squared pair-difference identity.**
Source: arXiv:2410.06833v1, §3.2, the pairwise Hessian quadratic form
in the proof of `lem: PL.borjan`. -/
theorem sum_pair_sq_eq (n : ℕ) (v : Idx n → ℝ) :
    (∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) =
      2 * (n : ℝ) * ∑ i : Idx n, (v i) ^ 2 - 2 * (∑ i : Idx n, v i) ^ 2 := by
  have hcross : (∑ i : Idx n, ∑ j : Idx n, 2 * v i * v j) =
      2 * (∑ i : Idx n, v i) ^ 2 := by
    simp_rw [mul_assoc, ← Finset.mul_sum]
    rw [← Finset.sum_mul]
    ring
  calc
    _ = ∑ i : Idx n, ∑ j : Idx n, ((v i) ^ 2 + (v j) ^ 2 - 2 * v i * v j) := by
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ = _ := by
      simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, hcross,
        Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      rw [← Finset.mul_sum]
      ring

/-- **The sharp upper bound on the pair-square sum.**
Source: arXiv:2410.06833v1, §3.2, the same Hessian form;
the discarded term is the square of the sum of the displacements. -/
theorem sum_pair_sq_le_two (n : ℕ) (v : Idx n → ℝ) :
    (∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) ≤
      2 * (n : ℝ) * ∑ i : Idx n, (v i) ^ 2 := by
  rw [sum_pair_sq_eq]
  linarith [sq_nonneg (∑ i : Idx n, v i)]

/-- **Normalization gives the coefficient `1/(2n)`.**
Source: arXiv:2410.06833v1, §3.2, the normalized `𝖤_β`
and the quadratic remainder in `eq: otto.attention`. -/
theorem normalized_pair_half_sq_le (n : ℕ) (hn : 0 < n) (v : Idx n → ℝ) :
    (1 / (2 * (n : ℝ) ^ 2)) * ((1 / 2 : ℝ) *
        ∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) ≤
      (1 / (2 * (n : ℝ))) * ∑ i : Idx n, (v i) ^ 2 := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  calc
    _ ≤ (1 / (2 * (n : ℝ) ^ 2)) * ((1 / 2 : ℝ) *
        (2 * (n : ℝ) * ∑ i : Idx n, (v i) ^ 2)) := by
      gcongr
      exact sum_pair_sq_le_two n v
    _ = _ := by field_simp

/-- The normalization input is satisfied at `n=2`, including for
nonzero displacements with opposite signs. -/
example : 0 < 2 ∧ (∑ i : Idx 2, (![1, -1] : Idx 2 → ℝ) i) = 0 := by
  norm_num [Fin.sum_univ_two]

/-- **The global absolute Taylor bound for the actual angular energy.**
Source: arXiv:2410.06833v1, §3.2, `𝖤_β`, its angular gradient,
and `eq: otto.attention`. Unlike the earlier loose PL bound, this
coefficient is independent of temperature and retains `1/(2n)`. -/
theorem angularEβ_remainder_half (n : ℕ) (hn : 0 < n) (β : ℝ) (hβ : 0 < β)
    (Θ v : Idx n → ℝ) :
    |angularEβ n β (Θ + v) - angularEβ n β Θ -
        ∑ i : Idx n, angularGrad n β Θ i * v i| ≤
      (1 / (2 * (n : ℝ))) * ∑ i : Idx n, (v i) ^ 2 := by
  let C : ℝ := 1 / (2 * (n : ℝ) ^ 2)
  have hC : 0 ≤ C := by dsimp only [C]; positivity
  let R (i j : Idx n) :=
    angularPairPotential β ((Θ + v) i - (Θ + v) j) -
      angularPairPotential β (Θ i - Θ j) + Causal.h_pot β (Θ i - Θ j) * (v i - v j)
  have hR (i j : Idx n) : |R i j| ≤ (1 / 2 : ℝ) * (v i - v j) ^ 2 := by
    have h := angularPairPotential_remainder_half β hβ (Θ i - Θ j)
      ((Θ + v) i - (Θ + v) j)
    have heq : (Θ + v) i - (Θ + v) j - (Θ i - Θ j) = v i - v j := by
      simp only [Pi.add_apply]
      ring
    simpa only [R, heq] using h
  have heq : angularEβ n β (Θ + v) - angularEβ n β Θ -
      ∑ i : Idx n, angularGrad n β Θ i * v i = C * ∑ i : Idx n, ∑ j : Idx n, R i j := by
    rw [angularEβ_eq_pairPotential n hn β hβ.ne',
      angularEβ_eq_pairPotential n hn β hβ.ne', ← pair_linear_eq_angularGrad n hn]
    dsimp only [R, C]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, neg_mul,
      Finset.sum_neg_distrib]
    ring
  rw [heq, abs_mul, abs_of_nonneg hC]
  calc
    C * |∑ i : Idx n, ∑ j : Idx n, R i j| ≤
        C * ∑ i : Idx n, ∑ j : Idx n, |R i j| := by
      apply mul_le_mul_of_nonneg_left _ hC
      exact (Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum fun _ _ => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ C * ∑ i : Idx n, ∑ j : Idx n, (1 / 2 : ℝ) * (v i - v j) ^ 2 := by
      apply mul_le_mul_of_nonneg_left _ hC
      exact Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hR i j
    _ = C * ((1 / 2 : ℝ) * ∑ i : Idx n, ∑ j : Idx n, (v i - v j) ^ 2) := by
      simp only [Finset.mul_sum]
    _ ≤ _ := normalized_pair_half_sq_le n hn v

/-- **The reverse energy discrepancy uses the gradient at its target.**
Source: arXiv:2410.06833v1, §3.2, the lower inequality of (H1)
used in `eq: otto.attention`. This is the lower half of Taylor's
formula at `V`, rearranged to bound `𝖤_β(V)-𝖤_β(U)`. -/
theorem angularEβ_gain_le_target_gradient (n : ℕ) (hn : 0 < n)
    (β : ℝ) (hβ : 0 < β) (U V : Idx n → ℝ) :
    angularEβ n β V - angularEβ n β U ≤
      (∑ i : Idx n, angularGrad n β V i * (V i - U i)) +
        (1 / (2 * (n : ℝ))) * ∑ i : Idx n, (V i - U i) ^ 2 := by
  have h := (abs_le.mp (angularEβ_remainder_half n hn β hβ V (U - V))).1
  have heq : V + (U - V) = U := by abel
  rw [heq] at h
  have hlin : (∑ i : Idx n, angularGrad n β V i * (V i - U i)) =
      -(∑ i : Idx n, angularGrad n β V i * (U - V) i) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Pi.sub_apply]
    ring
  have hsq : (∑ i : Idx n, ((U - V) i) ^ 2) =
      ∑ i : Idx n, (V i - U i) ^ 2 := by
    apply Finset.sum_congr rfl
    intro i _
    simp only [Pi.sub_apply]
    ring
  rw [hlin, ← hsq]
  linarith

/-- All inputs of both actual-energy estimates hold at `n=2`, `β=1`.
The angle tuples may be arbitrary. -/
example : 0 < 2 ∧ (0 : ℝ) < 1 := ⟨by norm_num, one_pos⟩

end Transformer.Metastability
