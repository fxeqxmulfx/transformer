module

public import Transformer.Weil.FieldSum
public import Transformer.Weil.MulWeight

/-!
# `S_d` of `χ(f(x))` is the sum of `χ(N f(x))` over the extension of degree `d`

Let `L` be an extension of degree `d` of the finite field `K` with `q` elements, and let `x ∈ L`
have minimal polynomial `P` of degree `k`. Over `L`, `P = ∏_{i < k} (X − x^{q^i})`, so
`Res(P, f) = ∏_{i < k} f(x^{q^i}) = ∏_{i < k} f(x)^{q^i}`, while
`N_{L/K} f(x) = ∏_{i < d} f(x)^{q^i}`. These factors have period `k`, and `k ∣ d`, so
`N_{L/K} f(x) = Res(P, f)^{d/k}`, and `χ(N_{L/K} f(x)) = w(P)^{d/k}` for the weight
`w(P) = χ(Res(P, f))`. By `FieldSum`, `∑_{x ∈ L} χ(N_{L/K} f(x)) = S_d`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- A product over `k e` terms of a sequence of period `k` is the `e`-th power of the product
over one period. -/
theorem prod_range_mul_of_periodic {M : Type*} [CommMonoid M] {g : ℕ → M} {k : ℕ}
    (hg : ∀ i, g (i + k) = g i) (e : ℕ) :
    ∏ i ∈ range (k * e), g i = (∏ i ∈ range k, g i) ^ e := by
  have hper (e i : ℕ) : g (k * e + i) = g i := by
    induction e with
    | zero => simp
    | succ e ih => rw [mul_add, mul_one, add_right_comm, hg, ih]
  induction e with
  | zero => simp
  | succ e ih =>
    rw [mul_add, mul_one, prod_range_add, ih, pow_succ]
    simp only [hper]

example := prod_range_mul_of_periodic (M := ℕ) (g := fun _ => 2) (k := 1) (fun _ => rfl) 2

variable {K L : Type*} [Field K] [Fintype K] [Field L] [Algebra K L] [Finite L]

/-- `Res(P, f) = ∏_{i < k} f(x)^{q^i}` in `L`, for the minimal polynomial `P` of `x`. -/
theorem algebraMap_resultant_minpoly (f : K[X]) (x : L) :
    algebraMap K L (resultant (minpoly K x) f) =
      ∏ i ∈ range (minpoly K x).natDegree, aeval x f ^ Fintype.card K ^ i := by
  have hdeg : ((minpoly K x).map (algebraMap K L)).natDegree = (minpoly K x).natDegree :=
    natDegree_map _
  have hfL : (f.map (algebraMap K L)).natDegree ≤ f.natDegree := natDegree_map_le
  calc algebraMap K L (resultant (minpoly K x) f)
      = resultant ((minpoly K x).map (algebraMap K L)) (f.map (algebraMap K L))
          ((minpoly K x).map (algebraMap K L)).natDegree f.natDegree := by
        rw [hdeg, resultant_map_map]
    _ = ∏ i ∈ range (minpoly K x).natDegree, resultant (X - C (x ^ Fintype.card K ^ i))
          (f.map (algebraMap K L)) (X - C (x ^ Fintype.card K ^ i)).natDegree f.natDegree := by
        rw [map_minpoly_eq_prod]
        exact resultant_prod_left _ _ _ _
          (by simp only [leadingCoeff_X_sub_C, prod_const_one]; exact one_ne_zero) hfL
    _ = _ := prod_congr rfl fun i _ => by
        rw [natDegree_X_sub_C, resultant_X_sub_C_left (hg := hfL), eval_map_algebraMap,
          aeval_pow_card_pow]

/-- **The norm through the resultant**: `N_{L/K} f(x) = Res(P, f)^{d/k}` for `x` with minimal
polynomial `P` of degree `k`, and `d = [L : K]`. -/
theorem norm_aeval_eq_resultant_pow (f : K[X]) (x : L) :
    Algebra.norm K (aeval x f) =
      resultant (minpoly K x) f ^ (Module.finrank K L / (minpoly K x).natDegree) := by
  have hx := IsIntegral.of_finite K x
  obtain ⟨e, he⟩ := minpoly.degree_dvd hx
  apply (algebraMap K L).injective
  rw [FiniteField.algebraMap_norm_eq_prod_pow, map_pow, algebraMap_resultant_minpoly,
    Nat.card_eq_fintype_card, he, Nat.mul_div_cancel_left _ (minpoly.natDegree_pos hx)]
  refine prod_range_mul_of_periodic (g := fun i => aeval x f ^ Fintype.card K ^ i)
    (fun i => ?_) e
  rw [pow_add, pow_mul', ← aeval_pow_card_pow f x, (pow_card_pow_eq_self_iff x _).mpr dvd_rfl]

/-- `χ(N_{L/K} f(x)) = w(P)^{d/k}` for the weight `w(P) = χ(Res(P, f))`. -/
theorem mulChar_norm_aeval (χ : MulChar K ℂ) (f : K[X]) (x : L) :
    χ (Algebra.norm K (aeval x f)) =
      mulWeight χ f (minpoly K x) ^ (Module.finrank K L / (minpoly K x).natDegree) := by
  rw [norm_aeval_eq_resultant_pow, map_pow, mulWeight]

/-- **`S_d` is the sum of `χ(N f(x))` over `L`**: `∑_{x ∈ L} χ(N_{L/K} f(x)) = S_d` for
`d = [L : K]` and the weight `w(P) = χ(Res(P, f))`. -/
theorem sum_mulChar_norm_aeval [Fintype L] (χ : MulChar K ℂ) (f : K[X]) :
    ∑ x : L, χ (Algebra.norm K (aeval x f)) = lPowerSum (mulWeight χ f) (Module.finrank K L) :=
  sum_eq_lPowerSum _ fun x => mulChar_norm_aeval χ f x

end Transformer.Weil
