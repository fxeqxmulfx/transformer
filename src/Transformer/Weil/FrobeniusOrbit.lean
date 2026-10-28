module

public import Mathlib.FieldTheory.Finite.Extension
public import Mathlib.FieldTheory.Finite.Trace
public import Mathlib.FieldTheory.IntermediateField.Adjoin.Basic
public import Mathlib.RingTheory.Coprime.Lemmas
public import Transformer.Weil.RootSum

/-!
# Frobenius orbits and the trace of `f(x)`

Let `L` be a finite extension of degree `d` of a finite field `K` with `q` elements, and let
`x ∈ L` have minimal polynomial `P` of degree `k`. Then:

* `x^{q^l} = x` exactly when `k ∣ l`, since `P ∣ X^{q^l} − X` exactly when `k ∣ l`;
* the conjugates `x^{q^i}`, `i < k`, are distinct: `x^{q^i} = x^{q^j}` with `i < j < k` gives
  `x^{q^l} = x` for `l = i + k − j`, and `0 < l < k`;
* so `P = ∏_{i < k} (X − x^{q^i})` over `L`, and `rootSum f P = ∑_{i < k} f(x)^{q^i}`;
* `Tr_{L/K} f(x) = ∑_{i < d} f(x)^{q^i} = (d/k) · rootSum f P`, since `i ↦ f(x)^{q^i}` has
  period `k` and `k ∣ d`.

Hence `ψ(Tr_{L/K} f(x)) = λ(P)^{d/k}` for the character `λ(P) = ψ(rootSum f P)` of the
`L`-function.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- A sum over `k e` terms of a sequence of period `k` is `e` times the sum over one period. -/
theorem sum_range_mul_of_periodic {M : Type*} [AddCommMonoid M] {g : ℕ → M} {k : ℕ}
    (hg : ∀ i, g (i + k) = g i) (e : ℕ) :
    ∑ i ∈ range (k * e), g i = e • ∑ i ∈ range k, g i := by
  have hper (e i : ℕ) : g (k * e + i) = g i := by
    induction e with
    | zero => simp
    | succ e ih => rw [mul_add, mul_one, add_right_comm, hg, ih]
  induction e with
  | zero => simp
  | succ e ih =>
    rw [mul_add, mul_one, sum_range_add, ih, succ_nsmul]
    simp only [hper]

example := sum_range_mul_of_periodic (M := ℕ) (g := fun _ => 1) (k := 1) (fun _ => rfl) 2

variable {K L : Type*} [Field K] [Fintype K] [Field L] [Algebra K L]

/-- `f(x)^{q^i} = f(x^{q^i})` for `f ∈ K[X]` and `q = #K`. -/
theorem aeval_pow_card_pow (f : K[X]) (x : L) (i : ℕ) :
    aeval (x ^ Fintype.card K ^ i) f = aeval x f ^ Fintype.card K ^ i := by
  induction i with
  | zero => simp
  | succ i ih =>
    rw [pow_succ, pow_mul, pow_mul, ← ih, ← FiniteField.frobeniusAlgHom_apply K L,
      aeval_algHom_apply, FiniteField.frobeniusAlgHom_apply]

variable [Finite L]

/-- `x^{q^l} = x` exactly when the degree of the minimal polynomial of `x` divides `l`. -/
theorem pow_card_pow_eq_self_iff (x : L) (l : ℕ) :
    x ^ Fintype.card K ^ l = x ↔ (minpoly K x).natDegree ∣ l := by
  have hi := minpoly.irreducible (IsIntegral.of_finite K x)
  rw [hi.natDegree_dvd_iff_dvd_X_pow_card_pow_sub_X, Nat.card_eq_fintype_card]
  constructor
  · intro h
    exact minpoly.dvd K x (by simp [h])
  · intro h
    simpa [sub_eq_zero] using aeval_eq_zero_of_dvd_aeval_eq_zero h (minpoly.aeval K x)

/-- The conjugates `x^{q^i}`, `i < k`, are distinct. -/
theorem pow_card_pow_injOn (x : L) :
    Set.InjOn (fun i => x ^ Fintype.card K ^ i) (range (minpoly K x).natDegree) := by
  have key (i j : ℕ) (hij : i < j) (hj : j < (minpoly K x).natDegree) :
      x ^ Fintype.card K ^ i ≠ x ^ Fintype.card K ^ j := by
    intro h
    have hl : x ^ Fintype.card K ^ (i + ((minpoly K x).natDegree - j)) = x := by
      rw [pow_add, pow_mul, h, ← pow_mul, ← pow_add, Nat.add_sub_cancel' hj.le,
        (pow_card_pow_eq_self_iff x _).mpr dvd_rfl]
    have := Nat.le_of_dvd (by omega) ((pow_card_pow_eq_self_iff x _).mp hl)
    omega
  intro i hi j hj h
  simp only [coe_range, Set.mem_Iio] at hi hj
  rcases lt_trichotomy i j with hij | hij | hij
  · exact absurd h (key i j hij hj)
  · exact hij
  · exact absurd h.symm (key j i hij hi)

/-- Over `L`, the minimal polynomial of `x` is `∏_{i < k} (X − x^{q^i})`. -/
theorem map_minpoly_eq_prod (x : L) :
    (minpoly K x).map (algebraMap K L) =
      ∏ i ∈ range (minpoly K x).natDegree, (X - C (x ^ Fintype.card K ^ i)) := by
  have hx := IsIntegral.of_finite K x
  refine eq_of_monic_of_dvd_of_natDegree_le
    (monic_prod_of_monic _ _ fun i _ => monic_X_sub_C _) ((minpoly.monic hx).map _) ?_ ?_
  · refine prod_dvd_of_coprime (fun i hi j hj hij => ?_) fun i _ => ?_
    · exact isCoprime_X_sub_C_of_isUnit_sub
        (sub_ne_zero.mpr fun h => hij (pow_card_pow_injOn x hi hj h)).isUnit
    · rw [dvd_iff_isRoot, IsRoot.def, eval_map_algebraMap, aeval_pow_card_pow, minpoly.aeval,
        zero_pow (pow_ne_zero _ Fintype.card_ne_zero)]
  · rw [natDegree_map, natDegree_prod_of_monic _ _ fun i _ => monic_X_sub_C _]
    simp only [natDegree_X_sub_C, sum_const, card_range, smul_eq_mul, mul_one, le_refl]

/-- `rootSum f P = ∑_{i < k} f(x)^{q^i}` in `L`, for the minimal polynomial `P` of `x`. -/
theorem algebraMap_rootSum_minpoly (f : K[X]) (x : L) :
    algebraMap K L (rootSum f (minpoly K x)) =
      ∑ i ∈ range (minpoly K x).natDegree, aeval x f ^ Fintype.card K ^ i := by
  rw [← rootSum_map (algebraMap K L) f (minpoly.monic (IsIntegral.of_finite K x)),
    map_minpoly_eq_prod, rootSum_prod_X_sub_C]
  exact sum_congr rfl fun i _ => by rw [eval_map_algebraMap, aeval_pow_card_pow]

/-- **The trace through the minimal polynomial**: `Tr_{L/K} f(x) = (d/k) · rootSum f P` for `x`
with minimal polynomial `P` of degree `k`, and `d = [L : K]`. -/
theorem trace_aeval_eq_nsmul_rootSum (f : K[X]) (x : L) :
    Algebra.trace K L (aeval x f) =
      (Module.finrank K L / (minpoly K x).natDegree) • rootSum f (minpoly K x) := by
  have hx := IsIntegral.of_finite K x
  obtain ⟨e, he⟩ := minpoly.degree_dvd hx
  apply (algebraMap K L).injective
  rw [FiniteField.algebraMap_trace_eq_sum_pow, map_nsmul, algebraMap_rootSum_minpoly,
    Nat.card_eq_fintype_card, he, Nat.mul_div_cancel_left _ (minpoly.natDegree_pos hx)]
  refine sum_range_mul_of_periodic (g := fun i => aeval x f ^ Fintype.card K ^ i)
    (fun i => ?_) e
  rw [pow_add, pow_mul', ← aeval_pow_card_pow f x, (pow_card_pow_eq_self_iff x _).mpr dvd_rfl]

/-- `ψ(Tr_{L/K} f(x)) = λ(P)^{d/k}` for `λ(P) = ψ(rootSum f P)`. -/
theorem addChar_trace_aeval {M : Type*} [Monoid M] (ψ : AddChar K M) (f : K[X]) (x : L) :
    ψ (Algebra.trace K L (aeval x f)) =
      ψ (rootSum f (minpoly K x)) ^ (Module.finrank K L / (minpoly K x).natDegree) := by
  rw [trace_aeval_eq_nsmul_rootSum, AddChar.map_nsmul_eq_pow]

end Transformer.Weil
