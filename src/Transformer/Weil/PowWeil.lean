module

public import Transformer.Weil.NormSum
public import Transformer.Weil.PowCharSum
public import Transformer.Weil.WeilBound

/-!
# Weil's bound for `∑_x χ(f(x))`, `χ` of order prime to `deg f`

Let `χ ≠ 1` be a multiplicative character of `𝔽_p` of order `d`, `p − 1 = d r`, and let
`f ∈ 𝔽_p[X]` have degree `m ≥ 1` prime to `d`. Then `|∑_{x ∈ 𝔽_p} χ(f(x))| ≤ (m − 1) √p`:

* `χ(v) = 1` gives `v^r = 1`: `v = g^j` for a generator `g` of `𝔽_p^×`, and `χ(g)^j = 1` makes
  `χ^j = 1`, so `d ∣ j`;
* on the field `L` with `p^k` elements, `χ_L = χ ∘ N_{L/𝔽_p}` has `χ_L^d = 1`, and `χ_L(u) = 1`
  gives `u^c = 1` for `c = (1 + p + ⋯ + p^{k − 1}) r`, where `d c + 1 = p^k`; so `NormSum` and
  `PowCharSum` give `|S_k| ≤ 5 d m √p^k` once `p^k ≥ 64`, `p^k ≥ 8 d m` and `p^k ≥ d²`;
* `χ^m ≠ 1` as `d` is prime to `m`, so `χ(t)^m ≠ 1` for some `t` and `c_n = 0` for `n ≥ m`
  (`MulWeight`);
* `Rationality` turns the bounds on `S_k` into `|S_1| ≤ (m − 1) √p`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

section Kernel

variable {K : Type*} [Field K] [Fintype K]

/-- **The kernel of a character of order `d`**: for `q − 1 = d r`, `χ(v) = 1` gives `v^r = 1`.
With `v = g^j` for a generator `g` of `K^×`, `χ(g)^j = 1` makes `χ^j = 1`, so `d ∣ j`. -/
theorem pow_eq_one_of_mulChar_eq_one {χ : MulChar K ℂ} {r : ℕ}
    (hr : Fintype.card K - 1 = orderOf χ * r) {v : K} (hv : χ v = 1) : v ^ r = 1 := by
  classical
  have hv0 : v ≠ 0 := by
    rintro rfl
    rw [MulChar.map_zero] at hv
    exact zero_ne_one hv
  obtain ⟨g, hg⟩ := IsCyclic.exists_monoid_generator (α := Kˣ)
  obtain ⟨j, hj⟩ := (Submonoid.mem_powers_iff _ _).mp (hg (Units.mk0 v hv0))
  have hgj : χ (g : K) ^ j = 1 := by
    rw [← map_pow, ← Units.val_pow_eq_pow_val, hj, Units.val_mk0, hv]
  have hχj : χ ^ j = 1 := by
    ext w
    obtain ⟨i, hi⟩ := (Submonoid.mem_powers_iff _ _).mp (hg w)
    rw [MulChar.pow_apply_coe, MulChar.one_apply_coe, ← hi, Units.val_pow_eq_pow_val, map_pow,
      ← pow_mul, mul_comm, pow_mul, hgj, one_pow]
  obtain ⟨t, ht⟩ := orderOf_dvd_of_pow_eq_one hχj
  have h1 : Units.mk0 v hv0 ^ r = 1 := by
    rw [← hj, ← pow_mul, ht, show orderOf χ * t * r = (orderOf χ * r) * t by ring, ← hr,
      ← Fintype.card_units, pow_mul, pow_card_eq_one, one_pow]
  have h2 := congrArg Units.val h1
  rwa [Units.val_pow_eq_pow_val, Units.val_mk0, Units.val_one] at h2

example := pow_eq_one_of_mulChar_eq_one (K := ZMod 3) (χ := 1) (r := 2)
  (by rw [ZMod.card, orderOf_one]) (v := 1) (map_one _)

variable {L : Type*} [Field L] [Algebra K L] [Finite L]

/-- The lift `χ_L = χ ∘ N_{L/K}` of a multiplicative character of `K` to `L`. -/
noncomputable def normMulChar (χ : MulChar K ℂ) : MulChar L ℂ where
  toFun y := χ (Algebra.norm K y)
  map_one' := by simp only [map_one]
  map_mul' x y := by simp only [map_mul]
  map_nonunit' y hy := by
    rw [isUnit_iff_ne_zero, not_not] at hy
    simp only [hy, Algebra.norm_zero, MulChar.map_zero]

theorem normMulChar_apply (χ : MulChar K ℂ) (y : L) :
    normMulChar χ y = χ (Algebra.norm K y) := rfl

/-- `χ_L^d = 1` for `χ` of order `d`. -/
theorem normMulChar_pow_orderOf (χ : MulChar K ℂ) : normMulChar (L := L) χ ^ orderOf χ = 1 := by
  ext u
  have h := MulChar.pow_apply_coe χ (orderOf χ) (Units.map (Algebra.norm K : L →* K) u)
  rw [pow_orderOf_eq_one, MulChar.one_apply_coe, Units.coe_map] at h
  rw [MulChar.pow_apply_coe, MulChar.one_apply_coe, normMulChar_apply]
  exact h.symm

/-- **The kernel of `χ_L`**: for `q − 1 = d r`, `d` the order of `χ`, `χ_L(u) = 1` gives
`u^c = 1` for `c = (1 + q + ⋯ + q^{k − 1}) r`, `k = [L : K]`: `N_{L/K} u = u^{1 + q + ⋯}`. -/
theorem pow_eq_one_of_normMulChar_eq_one {χ : MulChar K ℂ} {r : ℕ}
    (hr : Fintype.card K - 1 = orderOf χ * r) {u : L} (hu : normMulChar χ u = 1) :
    u ^ ((∑ i ∈ range (Module.finrank K L), Fintype.card K ^ i) * r) = 1 := by
  have h := congrArg (algebraMap K L) (pow_eq_one_of_mulChar_eq_one hr hu)
  rwa [map_pow, FiniteField.algebraMap_norm_eq_prod_pow, prod_pow_eq_pow_sum, ← pow_mul,
    Nat.card_eq_fintype_card, map_one] at h

example := pow_eq_one_of_normMulChar_eq_one (K := ZMod 3) (L := ZMod 3) (χ := 1) (r := 2)
  (by rw [ZMod.card, orderOf_one]) (u := 1) (map_one _)

end Kernel

variable {p : ℕ} [Fact p.Prime]

/-- `|S_k| ≤ 5 d m √p^k` for `χ(f(x))`, `χ ≠ 1` of order `d` prime to `m = deg f ≥ 1`, once
`p^k ≥ 64`, `p^k ≥ 8 d m` and `p^k ≥ d²`: `PowCharSum` over the field with `p^k` elements. -/
theorem norm_lPowerSum_mulChar_le_of_coprime {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {f : (ZMod p)[X]} (hm : 1 ≤ f.natDegree) (hdm : Nat.Coprime (orderOf χ) f.natDegree)
    {k : ℕ} (hk : k ≠ 0) (h64 : 64 ≤ p ^ k) (h8 : 8 * orderOf χ * f.natDegree ≤ p ^ k)
    (hd2 : orderOf χ * orderOf χ ≤ p ^ k) :
    ‖lPowerSum (mulWeight χ f) k‖ ≤ 5 * orderOf χ * f.natDegree * √p ^ k := by
  classical
  let _ : Fintype (GaloisField p k) := Fintype.ofFinite _
  have hcard : Fintype.card (GaloisField p k) = p ^ k := by
    rw [← Nat.card_eq_fintype_card, GaloisField.card p k hk]
  obtain ⟨r, hr⟩ := MulChar.orderOf_dvd_card_sub_one (χ := χ)
  have hd : 2 ≤ orderOf χ := by
    have h0 := MulChar.orderOf_pos χ
    have h1 : orderOf χ ≠ 1 := fun h => hχ (orderOf_eq_one_iff.mp h)
    omega
  have hq : orderOf χ * ((∑ i ∈ range k, p ^ i) * r) + 1 = Fintype.card (GaloisField p k) := by
    have hgeom := geom_sum_mul_add (p - 1) k
    rw [Nat.sub_add_cancel (Fact.out : p.Prime).one_lt.le,
      show p - 1 = orderOf χ * r by rw [← hr, ZMod.card]] at hgeom
    rw [hcard, ← hgeom]
    ring
  have hb := norm_sum_mulChar_eval_le_of_coprime (F := GaloisField p k) (χ := normMulChar χ)
    (c := (∑ i ∈ range k, p ^ i) * r) hd (normMulChar_pow_orderOf χ) (fun u hu => by
      have h := pow_eq_one_of_normMulChar_eq_one hr hu
      rwa [GaloisField.finrank p hk, ZMod.card] at h) hq
    (f := f.map (algebraMap (ZMod p) (GaloisField p k))) (by rwa [natDegree_map])
    (by rwa [natDegree_map]) (by rwa [hcard]) (by rwa [hcard, natDegree_map]) (by rwa [hcard])
  rw [natDegree_map, hcard, Nat.cast_pow, sqrt_pow_eq_pow_sqrt (Nat.cast_nonneg _)] at hb
  have hS := sum_mulChar_norm_aeval (L := GaloisField p k) χ f
  rw [GaloisField.finrank p hk] at hS
  rw [← hS]
  refine Eq.trans_le (congrArg norm (sum_congr rfl fun x _ => ?_)) hb
  rw [normMulChar_apply, eval_map_algebraMap]

/-- **Weil's bound for a character of order prime to `deg f`**: for `χ ≠ 1` a multiplicative
character of `𝔽_p` of order `d` and `f ∈ 𝔽_p[X]` of degree `m ≥ 1` prime to `d`,
`|∑_{x ∈ 𝔽_p} χ(f(x))| ≤ (m − 1) √p`. -/
theorem norm_sum_mulChar_eval_le_of_coprime_orderOf {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {f : (ZMod p)[X]} (hm : 1 ≤ f.natDegree) (hdm : Nat.Coprime (orderOf χ) f.natDegree) :
    ‖∑ x : ZMod p, χ (f.eval x)‖ ≤ ((f.natDegree : ℝ) - 1) * √p := by
  have h1 := sum_mulChar_norm_aeval (L := ZMod p) χ f
  rw [Module.finrank_self] at h1
  simp only [Algebra.norm_self, MonoidHom.id_apply, coe_aeval_eq_eval] at h1
  rw [h1]
  have hχm : χ ^ f.natDegree ≠ 1 := fun h =>
    hχ (orderOf_eq_one_iff.mp (hdm.eq_one_of_dvd (orderOf_dvd_of_pow_eq_one h)))
  obtain ⟨t, ht⟩ := MulChar.ne_one_iff.mp hχm
  rw [MulChar.pow_apply_coe] at ht
  have hpow (k : ℕ) : k < p ^ k := Nat.lt_pow_self (Fact.out : p.Prime).one_lt
  have hdd (k : ℕ) (hk : 8 * orderOf χ * f.natDegree + orderOf χ * orderOf χ + 64 ≤ k) :
      8 * orderOf χ * f.natDegree ≤ p ^ k ∧ orderOf χ * orderOf χ ≤ p ^ k ∧ 64 ≤ p ^ k := by
    have := hpow k
    omega
  exact norm_lPowerSum_one_le (isMonicMul_mulWeight _ f) hm
    (fun n hn => lCoeff_mulWeight_eq_zero t.ne_zero ht hn) (Real.sqrt_nonneg _)
    (n₀ := 8 * orderOf χ * f.natDegree + orderOf χ * orderOf χ + 64) fun k hk =>
      norm_lPowerSum_mulChar_le_of_coprime hχ hm hdm (by omega) (hdd k hk).2.2 (hdd k hk).1
        (hdd k hk).2.1

section Examples

attribute [local instance] fact_prime_71

/-- `𝔽₇₁` has a multiplicative character of order `5`, as `5 ∣ 70`. -/
theorem exists_mulChar_zmod71_orderOf_eq_five :
    ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 :=
  MulChar.exists_mulChar_orderOf (ZMod 71) (by rw [ZMod.card]; norm_num)
    (Complex.isPrimitiveRoot_exp 5 (by norm_num))

/-- For a character `χ` of order `5` of `𝔽₇₁` and `f = X² + 1`: `|S_2| ≤ 50 · 71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    ‖lPowerSum (mulWeight χ (X ^ 2 + C 1)) 2‖ ≤ 5 * 5 * 2 * √71 ^ 2 := by
  obtain ⟨χ, hχ⟩ := exists_mulChar_zmod71_orderOf_eq_five
  have hdeg : (X ^ 2 + C 1 : (ZMod 71)[X]).natDegree = 2 := natDegree_X_pow_add_C
  refine ⟨χ, hχ, ?_⟩
  have h := norm_lPowerSum_mulChar_le_of_coprime (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (f := X ^ 2 + C 1) (by rw [hdeg]; norm_num)
    (by rw [hχ, hdeg]; norm_num) (k := 2) (by norm_num) (by norm_num) (by rw [hχ, hdeg]; norm_num)
    (by rw [hχ]; norm_num)
  rw [hχ, hdeg] at h
  exact_mod_cast h

/-- **Weil's bound for a character `χ` of order `5` of `𝔽₇₁`** and `f = X² + 1`:
`|∑_x χ(x² + 1)| ≤ √71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    ‖∑ x : ZMod 71, χ (x ^ 2 + 1)‖ ≤ √71 := by
  obtain ⟨χ, hχ⟩ := exists_mulChar_zmod71_orderOf_eq_five
  have hdeg : (X ^ 2 + C 1 : (ZMod 71)[X]).natDegree = 2 := natDegree_X_pow_add_C
  refine ⟨χ, hχ, ?_⟩
  have h := norm_sum_mulChar_eval_le_of_coprime_orderOf (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (f := X ^ 2 + C 1) (by rw [hdeg]; norm_num)
    (by rw [hχ, hdeg]; norm_num)
  rw [hdeg] at h
  norm_num at h
  exact h

end Examples

end Transformer.Weil
