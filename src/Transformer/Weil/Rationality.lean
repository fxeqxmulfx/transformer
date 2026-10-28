module

public import Mathlib.Analysis.Complex.Polynomial.Basic
public import Transformer.Weil.NewtonIdentity

/-!
# Rationality: `S_n = −∑_ω ω^n`

For `ψ ≠ 1` and `f` of degree `m ≥ 1` prime to the characteristic, `c_n = 0` for `n ≥ m`, so the
`L`-function `L(T) = ∑_{k < m} c_k T^k` is a polynomial with `c_0 = 1`. Its reverse
`L^*(X) = X^{m − 1} L(1/X) = ∑_{k < m} c_k X^{m − 1 − k}` is monic of degree `m − 1`, so over `ℂ`
it is `∏_ω (X − ω)` over `m − 1` roots `ω`, and `L(T) = ∏_ω (1 − ω T)`.

The reverse of `L^*` has the coefficients `c_b`, so the power sums `p_n = ∑_ω ω^n` satisfy
Newton's recursion `p_n = −n c_n − ∑_{1 ≤ b < n} c_b p_{n − b}`, while
`n c_n = ∑_{1 ≤ d ≤ n} S_d c_{n − d}`. By induction on `n`, `S_n = −p_n` for `n ≥ 1`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K] [Fintype K]

/-- The reverse `L^*(X) = ∑_{k < m} c_k X^{m − 1 − k}` of the `L`-function, `m = deg f`. -/
noncomputable def lReverse (ψ : AddChar K ℂ) (f : K[X]) : ℂ[X] :=
  ∑ k ∈ range f.natDegree, C (lCoeff ψ f k) * X ^ (f.natDegree - 1 - k)

theorem coeff_lReverse (ψ : AddChar K ℂ) (f : K[X]) (n : ℕ) :
    (lReverse ψ f).coeff n =
      if n < f.natDegree then lCoeff ψ f (f.natDegree - 1 - n) else 0 := by
  rw [lReverse, finsetSum_coeff]
  simp only [coeff_C_mul_X_pow]
  split_ifs with hn
  · rw [sum_eq_single (f.natDegree - 1 - n)
      (fun k hk hkn => ite_eq_right_iff.mpr fun h => absurd
        (by have := mem_range.mp hk; omega) hkn)
      (fun h => absurd (mem_range.mpr (by omega)) h)]
    exact ite_eq_left (by omega)
  · exact sum_eq_zero fun k hk => ite_eq_right_iff.mpr fun h => absurd h
      (by have := mem_range.mp hk; omega)

theorem natDegree_lReverse_le (ψ : AddChar K ℂ) (f : K[X]) :
    (lReverse ψ f).natDegree ≤ f.natDegree - 1 :=
  natDegree_le_iff_coeff_eq_zero.mpr fun N hN => by
    rw [coeff_lReverse]
    exact ite_eq_right (by omega)

theorem coeff_lReverse_natDegree_sub_one (ψ : AddChar K ℂ) {f : K[X]} (hm : 1 ≤ f.natDegree) :
    (lReverse ψ f).coeff (f.natDegree - 1) = 1 := by
  rw [coeff_lReverse, ite_eq_left (show f.natDegree - 1 < f.natDegree by omega), Nat.sub_self,
    lCoeff_zero]

theorem lReverse_monic (ψ : AddChar K ℂ) {f : K[X]} (hm : 1 ≤ f.natDegree) :
    (lReverse ψ f).Monic :=
  monic_of_natDegree_le_of_coeff_eq_one _ (natDegree_lReverse_le ψ f)
    (coeff_lReverse_natDegree_sub_one ψ hm)

example := coeff_lReverse_natDegree_sub_one (K := ZMod 2) 1 (f := X) (by rw [natDegree_X])

example := lReverse_monic (K := ZMod 2) 1 (f := X) (by rw [natDegree_X])

theorem natDegree_lReverse (ψ : AddChar K ℂ) {f : K[X]} (hm : 1 ≤ f.natDegree) :
    (lReverse ψ f).natDegree = f.natDegree - 1 :=
  natDegree_eq_of_le_of_coeff_ne_zero (natDegree_lReverse_le ψ f)
    (by rw [coeff_lReverse_natDegree_sub_one ψ hm]; exact one_ne_zero)

example := natDegree_lReverse (K := ZMod 2) 1 (f := X) (by rw [natDegree_X])

/-- The reverse of `L^*` has the coefficients `c_b` of the `L`-function. -/
theorem coeff_reverse_lReverse {ψ : AddChar K ℂ} (hψ : ψ ≠ 1) {f : K[X]} (hm0 : 1 ≤ f.natDegree)
    (hm : (f.natDegree : K) ≠ 0) (b : ℕ) : (lReverse ψ f).reverse.coeff b = lCoeff ψ f b := by
  rw [coeff_reverse, natDegree_lReverse ψ hm0]
  rcases le_or_gt b (f.natDegree - 1) with hb | hb
  · rw [revAt_le hb, coeff_lReverse, ite_eq_left (show f.natDegree - 1 - b < f.natDegree by omega)]
    congr 1
    omega
  · rw [revAt_eq_self_of_lt hb, coeff_lReverse, ite_eq_right (show ¬b < f.natDegree by omega),
      lCoeff_eq_zero (n := b) hψ (by omega) hm (by omega)]

example := coeff_reverse_lReverse (K := ZMod 2) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero
    (ZMod.injective_stdAddChar (h.trans (AddChar.map_zero_eq_one _).symm))⟩)
  (f := X) (by rw [natDegree_X]) (by rw [natDegree_X, Nat.cast_one]; exact one_ne_zero) 1

/-- **Rationality**: `S_n = −p_n(L^*)` for `n ≥ 1`. -/
theorem lPowerSum_eq_neg_powerSum {ψ : AddChar K ℂ} (hψ : ψ ≠ 1) {f : K[X]}
    (hm0 : 1 ≤ f.natDegree) (hm : (f.natDegree : K) ≠ 0) {n : ℕ} (hn : n ≠ 0) :
    lPowerSum ψ f n = -powerSum (lReverse ψ f) n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  have hnewton := natCast_mul_lCoeff ψ f n
  rw [← Ico_add_one_right_eq_Icc, sum_Ico_succ_top (by omega : 1 ≤ n), Nat.sub_self, lCoeff_zero,
    mul_one] at hnewton
  have hrefl : ∑ d ∈ Ico 1 n, lPowerSum ψ f d * lCoeff ψ f (n - d) =
      ∑ b ∈ Ico 1 n, lCoeff ψ f b * lPowerSum ψ f (n - b) := by
    have h := sum_Ico_reflect (fun d => lPowerSum ψ f d * lCoeff ψ f (n - d)) 1 (Nat.le_succ n)
    rw [Nat.add_sub_cancel_left, Nat.add_sub_cancel] at h
    rw [← h]
    refine sum_congr rfl fun b hb => ?_
    rw [Nat.sub_sub_self (mem_Ico.mp hb).2.le, mul_comm]
  have hsum : ∑ b ∈ Ico 1 n, (lReverse ψ f).reverse.coeff b * powerSum (lReverse ψ f) (n - b) =
      -∑ b ∈ Ico 1 n, lCoeff ψ f b * lPowerSum ψ f (n - b) := by
    rw [← sum_neg_distrib]
    refine sum_congr rfl fun b hb => ?_
    have hb' := mem_Ico.mp hb
    rw [coeff_reverse_lReverse hψ hm0 hm, ih (n - b) (by omega) (by omega)]
    ring
  rw [powerSum_eq (lReverse_monic ψ hm0) hn, hsum, coeff_reverse_lReverse hψ hm0 hm]
  linear_combination -hnewton - hrefl

example := lPowerSum_eq_neg_powerSum (K := ZMod 2) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero
    (ZMod.injective_stdAddChar (h.trans (AddChar.map_zero_eq_one _).symm))⟩)
  (f := X) (by rw [natDegree_X]) (by rw [natDegree_X, Nat.cast_one]; exact one_ne_zero)
  one_ne_zero

/-- `S_n = −∑_ω ω^n` over the `m − 1` roots `ω` of `L^*` in `ℂ`, for `n ≥ 1`. -/
theorem lPowerSum_eq_neg_sum_roots {ψ : AddChar K ℂ} (hψ : ψ ≠ 1) {f : K[X]}
    (hm0 : 1 ≤ f.natDegree) (hm : (f.natDegree : K) ≠ 0) {n : ℕ} (hn : n ≠ 0) :
    lPowerSum ψ f n = -((lReverse ψ f).roots.map (· ^ n)).sum := by
  rw [lPowerSum_eq_neg_powerSum hψ hm0 hm hn, ← powerSum_multiset_prod_X_sub_C,
    ← (IsAlgClosed.splits (lReverse ψ f)).eq_prod_roots_of_monic (lReverse_monic ψ hm0)]

example := lPowerSum_eq_neg_sum_roots (K := ZMod 2) (ψ := ZMod.stdAddChar)
  (AddChar.ne_one_iff.mpr ⟨1, fun h => one_ne_zero
    (ZMod.injective_stdAddChar (h.trans (AddChar.map_zero_eq_one _).symm))⟩)
  (f := X) (by rw [natDegree_X]) (by rw [natDegree_X, Nat.cast_one]; exact one_ne_zero)
  one_ne_zero

theorem card_roots_lReverse (ψ : AddChar K ℂ) {f : K[X]} (hm : 1 ≤ f.natDegree) :
    (lReverse ψ f).roots.card = f.natDegree - 1 := by
  rw [IsAlgClosed.card_roots_eq_natDegree, natDegree_lReverse ψ hm]

example := card_roots_lReverse (K := ZMod 2) 1 (f := X) (by rw [natDegree_X])

end Transformer.Weil
