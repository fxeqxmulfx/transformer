module

public import Mathlib.Analysis.Complex.Polynomial.Basic
public import Transformer.Weil.NewtonIdentity
public import Transformer.Weil.Transfer

/-!
# Rationality: `S_n = −∑_ω ω^n`, and the bound from the extensions

Let `w` be a weight multiplicative on the monic polynomials with `c_n = 0` for `n ≥ m ≥ 1`. Then
the `L`-function `L(T) = ∑_{k < m} c_k T^k` is a polynomial with `c_0 = 1`. Its reverse
`L^*(X) = X^{m − 1} L(1/X) = ∑_{k < m} c_k X^{m − 1 − k}` is monic of degree `m − 1`, so over `ℂ`
it is `∏_ω (X − ω)` over `m − 1` roots `ω`, and `L(T) = ∏_ω (1 − ω T)`.

The reverse of `L^*` has the coefficients `c_b`, so the power sums `p_n = ∑_ω ω^n` satisfy
Newton's recursion `p_n = −n c_n − ∑_{1 ≤ b < n} c_b p_{n − b}`, while
`n c_n = ∑_{1 ≤ d ≤ n} S_d c_{n − d}`. By induction on `n`, `S_n = −p_n` for `n ≥ 1`.

So a bound `|S_d| ≤ M R^d` for all large `d` gives `|ω| ≤ R` for every root (`Transfer`), and
then `|S_1| ≤ (m − 1) R`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K] [Fintype K]

/-- The reverse `L^*(X) = ∑_{k < m} c_k X^{m − 1 − k}` of the `L`-function, cut at degree `m`. -/
noncomputable def lReverse (w : K[X] → ℂ) (m : ℕ) : ℂ[X] :=
  ∑ k ∈ range m, C (lCoeff w k) * X ^ (m - 1 - k)

theorem coeff_lReverse (w : K[X] → ℂ) (m n : ℕ) :
    (lReverse w m).coeff n = if n < m then lCoeff w (m - 1 - n) else 0 := by
  rw [lReverse, finsetSum_coeff]
  simp only [coeff_C_mul_X_pow]
  split_ifs with hn
  · rw [sum_eq_single (m - 1 - n)
      (fun k hk hkn => ite_eq_right_iff.mpr fun h => absurd
        (by have := mem_range.mp hk; omega) hkn)
      (fun h => absurd (mem_range.mpr (by omega)) h)]
    exact ite_eq_left (by omega)
  · exact sum_eq_zero fun k hk => ite_eq_right_iff.mpr fun h => absurd h
      (by have := mem_range.mp hk; omega)

theorem natDegree_lReverse_le (w : K[X] → ℂ) (m : ℕ) : (lReverse w m).natDegree ≤ m - 1 :=
  natDegree_le_iff_coeff_eq_zero.mpr fun N hN => by
    rw [coeff_lReverse]
    exact ite_eq_right (by omega)

section Rational

variable {w : K[X] → ℂ} (hw : IsMonicMul w) {m : ℕ} (hm : 1 ≤ m)

include hw hm

theorem coeff_lReverse_sub_one : (lReverse w m).coeff (m - 1) = 1 := by
  rw [coeff_lReverse, ite_eq_left (show m - 1 < m by omega), Nat.sub_self, lCoeff_zero hw]

theorem lReverse_monic : (lReverse w m).Monic :=
  monic_of_natDegree_le_of_coeff_eq_one _ (natDegree_lReverse_le w m)
    (coeff_lReverse_sub_one hw hm)

theorem natDegree_lReverse : (lReverse w m).natDegree = m - 1 :=
  natDegree_eq_of_le_of_coeff_ne_zero (natDegree_lReverse_le w m)
    (by rw [coeff_lReverse_sub_one hw hm]; exact one_ne_zero)

theorem card_roots_lReverse : (lReverse w m).roots.card = m - 1 := by
  rw [IsAlgClosed.card_roots_eq_natDegree, natDegree_lReverse hw hm]

variable (hvan : ∀ n, m ≤ n → lCoeff w n = 0)

include hvan

/-- The reverse of `L^*` has the coefficients `c_b` of the `L`-function. -/
theorem coeff_reverse_lReverse (b : ℕ) : (lReverse w m).reverse.coeff b = lCoeff w b := by
  rw [coeff_reverse, natDegree_lReverse hw hm]
  rcases le_or_gt b (m - 1) with hb | hb
  · rw [revAt_le hb, coeff_lReverse, ite_eq_left (show m - 1 - b < m by omega)]
    congr 1
    omega
  · rw [revAt_eq_self_of_lt hb, coeff_lReverse, ite_eq_right (show ¬b < m by omega),
      hvan b (by omega)]

/-- **Rationality**: `S_n = −p_n(L^*)` for `n ≥ 1`. -/
theorem lPowerSum_eq_neg_powerSum {n : ℕ} (hn : n ≠ 0) :
    lPowerSum w n = -powerSum (lReverse w m) n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  have hnewton := natCast_mul_lCoeff hw n
  rw [← Ico_add_one_right_eq_Icc, sum_Ico_succ_top (by omega : 1 ≤ n), Nat.sub_self,
    lCoeff_zero hw, mul_one] at hnewton
  have hrefl : ∑ d ∈ Ico 1 n, lPowerSum w d * lCoeff w (n - d) =
      ∑ b ∈ Ico 1 n, lCoeff w b * lPowerSum w (n - b) := by
    have h := sum_Ico_reflect (fun d => lPowerSum w d * lCoeff w (n - d)) 1 (Nat.le_succ n)
    rw [Nat.add_sub_cancel_left, Nat.add_sub_cancel] at h
    rw [← h]
    refine sum_congr rfl fun b hb => ?_
    rw [Nat.sub_sub_self (mem_Ico.mp hb).2.le, mul_comm]
  have hsum : ∑ b ∈ Ico 1 n, (lReverse w m).reverse.coeff b * powerSum (lReverse w m) (n - b) =
      -∑ b ∈ Ico 1 n, lCoeff w b * lPowerSum w (n - b) := by
    rw [← sum_neg_distrib]
    refine sum_congr rfl fun b hb => ?_
    have hb' := mem_Ico.mp hb
    rw [coeff_reverse_lReverse hw hm hvan, ih (n - b) (by omega) (by omega)]
    ring
  rw [powerSum_eq (lReverse_monic hw hm) hn, hsum, coeff_reverse_lReverse hw hm hvan]
  linear_combination -hnewton - hrefl

/-- `S_n = −∑_ω ω^n` over the `m − 1` roots `ω` of `L^*` in `ℂ`, for `n ≥ 1`. -/
theorem lPowerSum_eq_neg_sum_roots {n : ℕ} (hn : n ≠ 0) :
    lPowerSum w n = -((lReverse w m).roots.map (· ^ n)).sum := by
  rw [lPowerSum_eq_neg_powerSum hw hm hvan hn, ← powerSum_multiset_prod_X_sub_C,
    ← (IsAlgClosed.splits (lReverse w m)).eq_prod_roots_of_monic (lReverse_monic hw hm)]

variable {M R : ℝ} (hR : 0 ≤ R) {n₀ : ℕ} (hS : ∀ d, n₀ ≤ d → ‖lPowerSum w d‖ ≤ M * R ^ d)

include hR hS

/-- **The roots from the extensions**: if `|S_d| ≤ M R^d` for all `d ≥ n₀`, every root of `L^*`
has `|ω| ≤ R`. -/
theorem norm_le_of_mem_roots_lReverse {ω : ℂ} (hω : ω ∈ (lReverse w m).roots) : ‖ω‖ ≤ R :=
  norm_le_of_norm_sum_pow_le hR (M := M) (n₀ := n₀ + 1) (fun n hn => by
    rw [← norm_neg, ← lPowerSum_eq_neg_sum_roots hw hm hvan (by omega)]
    exact hS n (by omega)) hω

/-- **The bound over `K`**: if `|S_d| ≤ M R^d` for all `d ≥ n₀`, then `|S_1| ≤ (m − 1) R`. -/
theorem norm_lPowerSum_one_le : ‖lPowerSum w 1‖ ≤ ((m : ℝ) - 1) * R := by
  rw [lPowerSum_eq_neg_sum_roots hw hm hvan one_ne_zero, norm_neg]
  simp only [pow_one, Multiset.map_id']
  calc ‖(lReverse w m).roots.sum‖ ≤ ((lReverse w m).roots.map (‖·‖)).sum :=
        norm_multiset_sum_le _
    _ ≤ ((lReverse w m).roots.map (‖·‖)).card • R :=
        Multiset.sum_le_card_nsmul _ _ fun x hx => by
          obtain ⟨ω, hω, rfl⟩ := Multiset.mem_map.mp hx
          exact norm_le_of_mem_roots_lReverse hw hm hvan hR hS hω
    _ = ((m : ℝ) - 1) * R := by
        rw [Multiset.card_map, card_roots_lReverse hw hm, nsmul_eq_mul, Nat.cast_sub hm,
          Nat.cast_one]

end Rational

/-- The weight of `y² − y = x` over `𝔽₂`, `ψ` the standard character: `m = 1`. -/
theorem isMonicMul_addWeight_X :
    IsMonicMul (addWeight (ZMod.stdAddChar : AddChar (ZMod 2) ℂ) X) :=
  isMonicMul_addWeight _ X

/-- Its `L`-function is `1`: `c_n = 0` for `n ≥ 1`. -/
theorem lCoeff_addWeight_X_eq_zero {n : ℕ} (hn : 1 ≤ n) :
    lCoeff (addWeight (ZMod.stdAddChar : AddChar (ZMod 2) ℂ) X) n = 0 :=
  lCoeff_eq_zero stdAddChar_ne_one (by rw [natDegree_X]; exact one_ne_zero)
    (by rw [natDegree_X, Nat.cast_one]; exact one_ne_zero) (by rwa [natDegree_X])

/-- So `S_d = 0` for `d ≥ 1`: `L^* = 1` has no roots. -/
theorem lPowerSum_addWeight_X_eq_zero {d : ℕ} (hd : d ≠ 0) :
    lPowerSum (addWeight (ZMod.stdAddChar : AddChar (ZMod 2) ℂ) X) d = 0 := by
  have h0 := card_roots_lReverse isMonicMul_addWeight_X (le_refl 1)
  rw [Nat.sub_self, Multiset.card_eq_zero] at h0
  rw [lPowerSum_eq_neg_sum_roots isMonicMul_addWeight_X le_rfl
    (fun _ => lCoeff_addWeight_X_eq_zero) hd, h0, Multiset.map_zero, Multiset.sum_zero, neg_zero]

example := lCoeff_addWeight_X_eq_zero le_rfl

example := lPowerSum_addWeight_X_eq_zero one_ne_zero

example := coeff_lReverse_sub_one isMonicMul_addWeight_X le_rfl

example := lReverse_monic isMonicMul_addWeight_X le_rfl

example := card_roots_lReverse isMonicMul_addWeight_X le_rfl

example := natDegree_lReverse isMonicMul_addWeight_X le_rfl

example := coeff_reverse_lReverse isMonicMul_addWeight_X le_rfl
  (fun _ => lCoeff_addWeight_X_eq_zero) 1

example := lPowerSum_eq_neg_powerSum isMonicMul_addWeight_X le_rfl
  (fun _ => lCoeff_addWeight_X_eq_zero) one_ne_zero

example := lPowerSum_eq_neg_sum_roots isMonicMul_addWeight_X le_rfl
  (fun _ => lCoeff_addWeight_X_eq_zero) one_ne_zero

example := norm_lPowerSum_one_le isMonicMul_addWeight_X le_rfl
  (fun _ => lCoeff_addWeight_X_eq_zero) zero_le_one (M := 0) (n₀ := 1)
  fun d hd => by rw [lPowerSum_addWeight_X_eq_zero (by omega), norm_zero, zero_mul]

example (ω : ℂ) (hω : ω ∈ (lReverse (addWeight ZMod.stdAddChar (X : (ZMod 2)[X])) 1).roots) :=
  norm_le_of_mem_roots_lReverse isMonicMul_addWeight_X le_rfl
    (fun _ => lCoeff_addWeight_X_eq_zero) zero_le_one (M := 0) (n₀ := 1)
    (fun d hd => by rw [lPowerSum_addWeight_X_eq_zero (by omega), norm_zero, zero_mul]) hω

end Transformer.Weil
