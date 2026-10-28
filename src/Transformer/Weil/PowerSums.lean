module

public import Mathlib.Algebra.BigOperators.NatAntidiagonal
public import Mathlib.Algebra.Field.Rat
public import Mathlib.Algebra.Order.BigOperators.Group.LocallyFinite
public import Mathlib.Algebra.Polynomial.Derivative
public import Mathlib.Algebra.Polynomial.Reverse
public import Mathlib.RingTheory.PowerSeries.Inverse
public import Mathlib.Tactic.LinearCombination

/-!
# Power sums of the roots of a polynomial, from its coefficients

Let `g ∈ K[X]` have degree `n` and reverse `A = X^n g(1/X)`. When `g = ∏_i (X − β_i)`,
`A = ∏_i (1 − β_i X)` and `−X A′/A = ∑_i β_i X/(1 − β_i X) = ∑_{j ≥ 1} (∑_i β_i^j) X^j`. So the
power sums `p_j(g) = ∑_i β_i^j` are defined for every `g` by its coefficients, as the
coefficients of `−X A′/A`, with `p_0(g) = n`:

* Newton's recursion `p_j = −j A_j − ∑_{1 ≤ b < j} A_b p_{j − b}` for monic `g` and `j ≥ 1`;
* `p_j(g h) = p_j(g) + p_j(h)`, `p_j(X − β) = β^j`, and `p_j` commutes with field maps.

They define the sum `∑_i f(β_i)` over the roots of `g` without a splitting field.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K]

/-- `−X A′/A` for the reverse `A` of `g`: the generating series of the power sums of its
roots. -/
noncomputable def newtonSeries (g : K[X]) : PowerSeries K :=
  ((-(X * derivative g.reverse) : K[X]) : PowerSeries K) * (g.reverse : PowerSeries K)⁻¹

/-- The power sums `p_j = ∑_i β_i^j` of the roots of `g`: `p_0 = deg g`, and for `j ≥ 1` the
coefficient of `X^j` in `−X A′/A`. -/
noncomputable def powerSum (g : K[X]) (j : ℕ) : K :=
  if j = 0 then g.natDegree else PowerSeries.coeff j (newtonSeries g)

theorem powerSum_zero (g : K[X]) : powerSum g 0 = g.natDegree := by
  simp [powerSum]

theorem powerSum_of_ne_zero (g : K[X]) {j : ℕ} (hj : j ≠ 0) :
    powerSum g j = PowerSeries.coeff j (newtonSeries g) := by
  simp [powerSum, hj]

example := powerSum_of_ne_zero (K := ℚ) X one_ne_zero

/-- `(−X A′/A) · A = −X A′` for `g ≠ 0`: the reverse `A` has constant coefficient `lc g ≠ 0`. -/
theorem newtonSeries_mul_reverse {g : K[X]} (hg : g ≠ 0) :
    newtonSeries g * (g.reverse : PowerSeries K) =
      ((-(X * derivative g.reverse) : K[X]) : PowerSeries K) := by
  rw [newtonSeries, mul_assoc, PowerSeries.inv_mul_cancel, mul_one]
  rw [Polynomial.constantCoeff_coe, coeff_zero_reverse]
  exact leadingCoeff_ne_zero.mpr hg

example := newtonSeries_mul_reverse (K := ℚ) X_ne_zero

theorem coeff_zero_newtonSeries (g : K[X]) : PowerSeries.coeff 0 (newtonSeries g) = 0 := by
  rw [PowerSeries.coeff_zero_eq_constantCoeff_apply, newtonSeries, map_mul,
    Polynomial.constantCoeff_coe, coeff_neg, coeff_X_mul_zero, neg_zero, zero_mul]

/-- **Newton's recursion**: for monic `g` and `j ≥ 1`,
`p_j = −j A_j − ∑_{1 ≤ b < j} A_b p_{j − b}`, with `A = g.reverse`. -/
theorem powerSum_eq {g : K[X]} (hg : g.Monic) {j : ℕ} (hj : j ≠ 0) :
    powerSum g j = -(j * g.reverse.coeff j) -
      ∑ b ∈ Ico 1 j, g.reverse.coeff b * powerSum g (j - b) := by
  obtain ⟨i, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hj
  have e := congrArg (PowerSeries.coeff (i + 1)) (newtonSeries_mul_reverse hg.ne_zero)
  rw [mul_comm, PowerSeries.coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ
    (fun b a => PowerSeries.coeff b (g.reverse : PowerSeries K) *
      PowerSeries.coeff a (newtonSeries g)), sum_range_succ, range_eq_Ico,
    sum_eq_sum_Ico_succ_bot (Nat.succ_pos i)] at e
  simp only [Polynomial.coeff_coe, coeff_zero_reverse, hg.leadingCoeff, one_mul, Nat.sub_zero,
    Nat.sub_self, coeff_zero_newtonSeries, mul_zero, add_zero, zero_add, coeff_neg, coeff_X_mul,
    coeff_derivative] at e
  have hs : ∑ b ∈ Ico 1 (i + 1), g.reverse.coeff b * powerSum g (i + 1 - b) =
      ∑ b ∈ Ico 1 (i + 1), g.reverse.coeff b * PowerSeries.coeff (i + 1 - b) (newtonSeries g) :=
    sum_congr rfl fun b hb => by
      rw [powerSum_of_ne_zero g (by have := (mem_Ico.mp hb).2; omega)]
  rw [powerSum_of_ne_zero g hj, hs, Nat.succ_eq_add_one, Nat.cast_succ]
  linear_combination e

example := powerSum_eq (K := ℚ) monic_X one_ne_zero

/-- `−X A′/A` is additive in `g`: `(g h)ʳ = gʳ hʳ` and `(gʳ hʳ)′ = gʳ′ hʳ + gʳ hʳ′`. -/
theorem newtonSeries_mul {g h : K[X]} (hg : g ≠ 0) (hh : h ≠ 0) :
    newtonSeries (g * h) = newtonSeries g + newtonSeries h := by
  have hA : ((g.reverse : PowerSeries K) * (h.reverse : PowerSeries K)) ≠ 0 := by
    rw [← Polynomial.coe_mul, Ne, Polynomial.coe_eq_zero_iff]
    exact mul_ne_zero (fun h0 => hg (reverse_eq_zero.mp h0)) fun h0 => hh (reverse_eq_zero.mp h0)
  refine mul_right_cancel₀ hA ?_
  calc newtonSeries (g * h) * ((g.reverse : PowerSeries K) * (h.reverse : PowerSeries K))
      = ((-(X * derivative (g.reverse * h.reverse)) : K[X]) : PowerSeries K) := by
        rw [← Polynomial.coe_mul, ← reverse_mul_of_domain]
        exact newtonSeries_mul_reverse (mul_ne_zero hg hh)
    _ = ((-(X * derivative g.reverse) : K[X]) : PowerSeries K) * (h.reverse : PowerSeries K) +
        ((-(X * derivative h.reverse) : K[X]) : PowerSeries K) * (g.reverse : PowerSeries K) := by
        rw [derivative_mul]
        push_cast
        ring
    _ = (newtonSeries g + newtonSeries h) *
        ((g.reverse : PowerSeries K) * (h.reverse : PowerSeries K)) := by
        rw [← newtonSeries_mul_reverse hg, ← newtonSeries_mul_reverse hh]
        ring

example := newtonSeries_mul (K := ℚ) X_ne_zero X_ne_zero

/-- `p_j(g h) = p_j(g) + p_j(h)` for `g, h ≠ 0`. -/
theorem powerSum_mul {g h : K[X]} (hg : g ≠ 0) (hh : h ≠ 0) (j : ℕ) :
    powerSum (g * h) j = powerSum g j + powerSum h j := by
  rcases eq_or_ne j 0 with rfl | hj
  · rw [powerSum_zero, powerSum_zero, powerSum_zero, natDegree_mul hg hh, Nat.cast_add]
  · rw [powerSum_of_ne_zero _ hj, powerSum_of_ne_zero _ hj, powerSum_of_ne_zero _ hj,
      newtonSeries_mul hg hh, map_add]

example := powerSum_mul (K := ℚ) X_ne_zero X_ne_zero 1

theorem powerSum_one (j : ℕ) : powerSum (1 : K[X]) j = 0 := by
  have h := powerSum_mul (one_ne_zero (α := K[X])) one_ne_zero j
  rw [mul_one] at h
  linear_combination -h

theorem coeff_reverse_X_sub_C_one (β : K) : (X - C β).reverse.coeff 1 = -β := by
  rw [coeff_reverse, natDegree_X_sub_C, revAt_le le_rfl, Nat.sub_self]
  simp

theorem coeff_reverse_X_sub_C_of_two_le (β : K) {b : ℕ} (hb : 2 ≤ b) :
    (X - C β).reverse.coeff b = 0 := by
  rw [coeff_reverse, natDegree_X_sub_C, revAt_eq_self_of_lt (by omega), coeff_sub,
    coeff_X_of_ne_one (by omega), coeff_C_of_ne_zero (by omega), sub_zero]

example := coeff_reverse_X_sub_C_of_two_le (1 : ℚ) le_rfl

/-- `p_j(X − β) = β^j`. -/
theorem powerSum_X_sub_C (β : K) (j : ℕ) : powerSum (X - C β) j = β ^ j := by
  induction j using Nat.strong_induction_on with
  | _ j ih =>
  rcases eq_or_ne j 0 with rfl | hj
  · rw [powerSum_zero, natDegree_X_sub_C, pow_zero, Nat.cast_one]
  rw [powerSum_eq (monic_X_sub_C β) hj]
  rcases eq_or_ne j 1 with rfl | hj1
  · simp
  obtain ⟨k, rfl⟩ : ∃ k, j = k + 1 := ⟨j - 1, by omega⟩
  rw [coeff_reverse_X_sub_C_of_two_le β (by omega), sum_eq_single_of_mem 1
    (mem_Ico.mpr ⟨le_rfl, by omega⟩) fun b hb hb1 => by
      rw [coeff_reverse_X_sub_C_of_two_le β (by have := (mem_Ico.mp hb).1; omega), zero_mul],
    coeff_reverse_X_sub_C_one, Nat.add_sub_cancel, ih k (by omega)]
  ring

/-- `p_j(∏_{β ∈ s} (X − β)) = ∑_{β ∈ s} β^j`. -/
theorem powerSum_multiset_prod_X_sub_C (s : Multiset K) (j : ℕ) :
    powerSum (s.map fun β => X - C β).prod j = (s.map (· ^ j)).sum := by
  induction s using Multiset.induction_on with
  | empty => simp [powerSum_one]
  | cons β t ih =>
    rw [Multiset.map_cons, Multiset.prod_cons, powerSum_mul (X_sub_C_ne_zero β)
      (monic_multiset_prod_of_monic _ _ fun β _ => monic_X_sub_C β).ne_zero, powerSum_X_sub_C,
      ih, Multiset.map_cons, Multiset.sum_cons]

/-- `p_j` commutes with field maps. -/
theorem powerSum_map {L : Type*} [Field L] (φ : K →+* L) {g : K[X]} (hg : g.Monic) (j : ℕ) :
    powerSum (g.map φ) j = φ (powerSum g j) := by
  induction j using Nat.strong_induction_on with
  | _ j ih =>
  rcases eq_or_ne j 0 with rfl | hj
  · rw [powerSum_zero, powerSum_zero, natDegree_map_eq_of_injective φ.injective, map_natCast]
  have hA (b : ℕ) : (g.map φ).reverse.coeff b = φ (g.reverse.coeff b) := by
    rw [coeff_reverse, coeff_reverse, natDegree_map_eq_of_injective φ.injective, coeff_map]
  rw [powerSum_eq (hg.map φ) hj, powerSum_eq hg hj, map_sub, map_neg, map_mul, map_natCast,
    map_sum, hA]
  congr 1
  refine sum_congr rfl fun b hb => ?_
  rw [hA, map_mul, ih _ (by have := mem_Ico.mp hb; omega)]

example := powerSum_map (RingHom.id ℚ) monic_X 1

end Transformer.Weil
