module

public import Transformer.Weil.PowExtension
public import Transformer.Weil.PowRoot
public import Transformer.Weil.PowUnit

/-!
# Weil's bound for `∑_x χ(f(x))` through an irreducible factor `P^e`, `e` prime to the order

For `χ ≠ 1` of order `d` on `𝔽_p` and `f = P^e Q` with `P` monic irreducible of degree `s`,
`P ∤ Q` and `e` prime to `d`: `|∑_{x ∈ 𝔽_p} χ(f(x))| ≤ (m − 1) √p`, `m = deg f`. Unlike
`PowWeil`, `d` may share a factor with `m`.

* In the field `L` with `p^k` elements, `s ∣ k`, `f = (X − α)^e G` with `G(α) ≠ 0`
  (`PowExtension`). With `N > m` divisible by `d`, `PowRoot` bounds
  `S_k = ∑_{x ∈ L} χ_L(f(x))` by `5 d N √p^k` once `p^k ≥ max(64, 8 d N, d²)`.
* `c_n = 0` for `n ≥ m` (`PowUnit`), as `χ^e ≠ 1`.
* `Rationality` along the multiples `k = s j` (`PowExtension`) turns these bounds into
  `|S_1| ≤ (m − 1) √p`, and `S_1 = ∑_x χ(f(x))`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- `|S_k| ≤ 5 d N √p^k` for `χ(f(x))`, `χ ≠ 1` of order `d`, `f = P^e Q` with `P` irreducible,
`P ∤ Q` and `e` prime to `d`, for `N > deg f` divisible by `d`, once `deg P ∣ k`, `p^k ≥ 64`,
`8 d N ≤ p^k` and `d² ≤ p^k`. -/
theorem norm_lPowerSum_mulChar_le_of_pow_mul {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {P Q : (ZMod p)[X]} (hP : Irreducible P) (hPQ : ¬P ∣ Q) {e : ℕ}
    (he : Nat.Coprime (orderOf χ) e) {N : ℕ} (hdN : orderOf χ ∣ N)
    (hN : (P ^ e * Q).natDegree < N) {k : ℕ} (hk : k ≠ 0) (hs : P.natDegree ∣ k)
    (h64 : 64 ≤ p ^ k) (h8 : 8 * orderOf χ * N ≤ p ^ k)
    (hd2 : orderOf χ * orderOf χ ≤ p ^ k) :
    ‖lPowerSum (mulWeight χ (P ^ e * Q)) k‖ ≤ 5 * orderOf χ * N * √p ^ k := by
  classical
  let _ : Fintype (GaloisField p k) := Fintype.ofFinite _
  have hcard : Fintype.card (GaloisField p k) = p ^ k := by
    rw [← Nat.card_eq_fintype_card, GaloisField.card p k hk]
  have hd : 2 ≤ orderOf χ := by
    have h0 := MulChar.orderOf_pos χ
    have h1 : orderOf χ ≠ 1 := fun h => hχ (orderOf_eq_one_iff.mp h)
    omega
  obtain ⟨c, hq, hker⟩ := exists_normMulChar_eq_one_imp_pow_eq_one χ hk
  obtain ⟨α, G, hfG, hG⟩ := exists_map_pow_mul_eq_X_sub_C_pow_mul hP hPQ e hk hs
  have hdeg : ((X - C α) ^ e * G).natDegree = (P ^ e * Q).natDegree := by
    rw [← hfG, natDegree_map]
  have hb := norm_sum_mulChar_eval_le_of_root (F := GaloisField p k) (χ := normMulChar χ) hd
    (normMulChar_pow_orderOf χ) hker (by rw [hcard]; exact hq) hG he hdN (by rwa [hdeg])
    (by rwa [hcard]) (by rwa [hcard]) (by rwa [hcard])
  rw [hcard, Nat.cast_pow, sqrt_pow_eq_pow_sqrt (Nat.cast_nonneg _)] at hb
  have hS := sum_mulChar_norm_aeval (L := GaloisField p k) χ (P ^ e * Q)
  rw [GaloisField.finrank p hk] at hS
  rw [← hS]
  refine Eq.trans_le (congrArg norm (sum_congr rfl fun x _ => ?_)) hb
  rw [normMulChar_apply, ← eval_map_algebraMap, hfG]

/-- **Weil's bound through an irreducible factor of multiplicity prime to the order**: for
`χ ≠ 1` of order `d` on `𝔽_p` and `f = P^e Q` with `P` monic irreducible, `P ∤ Q` and `e` prime
to `d`, `|∑_{x ∈ 𝔽_p} χ(f(x))| ≤ (m − 1) √p`, `m = deg f`. -/
theorem norm_sum_mulChar_eval_le_of_pow_mul {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {P Q : (ZMod p)[X]} (hPm : P.Monic) (hP : Irreducible P) (hPQ : ¬P ∣ Q) {e : ℕ}
    (he : Nat.Coprime (orderOf χ) e) :
    ‖∑ x : ZMod p, χ ((P ^ e * Q).eval x)‖ ≤ (((P ^ e * Q).natDegree : ℝ) - 1) * √p := by
  have h1 := sum_mulChar_norm_aeval (L := ZMod p) χ (P ^ e * Q)
  rw [Module.finrank_self] at h1
  simp only [Algebra.norm_self, MonoidHom.id_apply, coe_aeval_eq_eval] at h1
  rw [h1]
  have hd : 2 ≤ orderOf χ := by
    have h0 := MulChar.orderOf_pos χ
    have h1 : orderOf χ ≠ 1 := fun h => hχ (orderOf_eq_one_iff.mp h)
    omega
  have he0 : e ≠ 0 := by
    rintro rfl
    rw [Nat.coprime_zero_right] at he
    omega
  have hχe : χ ^ e ≠ 1 := fun h =>
    hχ (orderOf_eq_one_iff.mp (he.eq_one_of_dvd (orderOf_dvd_of_pow_eq_one h)))
  have hQ0 : Q ≠ 0 := fun h => hPQ (h ▸ dvd_zero P)
  have hs := hP.natDegree_pos
  have hm : 1 ≤ (P ^ e * Q).natDegree := by
    rw [natDegree_mul (pow_ne_zero _ hP.ne_zero) hQ0, natDegree_pow]
    have := Nat.mul_pos (Nat.pos_of_ne_zero he0) hs
    omega
  have hN : (P ^ e * Q).natDegree < orderOf χ * ((P ^ e * Q).natDegree + 1) :=
    (Nat.lt_succ_self _).trans_le (Nat.le_mul_of_pos_left _ (by omega))
  have hpow (k : ℕ) : k < p ^ k := Nat.lt_pow_self (Fact.out : p.Prime).one_lt
  exact norm_lPowerSum_one_le_of_mul (isMonicMul_mulWeight _ _) hm
    (fun n hn => lCoeff_mulWeight_eq_zero_of_pow_mul hPm hP hPQ hχe hn) (Real.sqrt_nonneg _)
    hs.ne' (M := 5 * orderOf χ * (orderOf χ * ((P ^ e * Q).natDegree + 1) : ℕ))
    (n₀ := 8 * orderOf χ * (orderOf χ * ((P ^ e * Q).natDegree + 1)) +
      orderOf χ * orderOf χ + 64) fun j hj => by
      have hk := hpow (P.natDegree * j)
      have hj' : j ≤ P.natDegree * j := Nat.le_mul_of_pos_left j hs
      exact norm_lPowerSum_mulChar_le_of_pow_mul hχ hP hPQ he (dvd_mul_right _ _) hN
        (Nat.mul_ne_zero hs.ne' (by omega)) (dvd_mul_right _ _) (by omega) (by omega)
        (by omega)

section Examples

attribute [local instance] fact_prime_71

/-- For a character `χ` of order `5` of `𝔽₇₁` and `f = X (X⁴ + 1)`, of degree `5`:
`|S_2| ≤ 750 · 71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    ‖lPowerSum (mulWeight χ (X ^ 1 * (X ^ 4 + C 1))) 2‖ ≤ 5 * 5 * 30 * √71 ^ 2 := by
  obtain ⟨χ, hχ⟩ := exists_mulChar_zmod71_orderOf_eq_five
  have hdeg : (X ^ 1 * (X ^ 4 + C 1) : (ZMod 71)[X]).natDegree = 5 := by
    rw [natDegree_mul (pow_ne_zero _ X_ne_zero) (X_pow_add_C_ne_zero (by norm_num) 1),
      natDegree_X_pow, natDegree_X_pow_add_C]
  refine ⟨χ, hχ, ?_⟩
  have h := norm_lPowerSum_mulChar_le_of_pow_mul (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) irreducible_X (Q := X ^ 4 + C 1)
    (by rw [X_dvd_iff]; simp) (e := 1) (by rw [hχ]; norm_num) (N := 30) (by rw [hχ]; norm_num)
    (by rw [hdeg]; norm_num) (k := 2) (by norm_num) (by rw [natDegree_X]; norm_num)
    (by norm_num) (by rw [hχ]; norm_num) (by rw [hχ]; norm_num)
  rw [hχ] at h
  exact_mod_cast h

/-- **Weil's bound for a character `χ` of order `5` of `𝔽₇₁`** and `f = X (X⁴ + 1)`, whose
degree `5` is the order: `|∑_x χ(x (x⁴ + 1))| ≤ 4 √71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    ‖∑ x : ZMod 71, χ (x * (x ^ 4 + 1))‖ ≤ 4 * √71 := by
  obtain ⟨χ, hχ⟩ := exists_mulChar_zmod71_orderOf_eq_five
  have hdeg : (X ^ 1 * (X ^ 4 + C 1) : (ZMod 71)[X]).natDegree = 5 := by
    rw [natDegree_mul (pow_ne_zero _ X_ne_zero) (X_pow_add_C_ne_zero (by norm_num) 1),
      natDegree_X_pow, natDegree_X_pow_add_C]
  refine ⟨χ, hχ, ?_⟩
  have h := norm_sum_mulChar_eval_le_of_pow_mul (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) monic_X irreducible_X
    (Q := X ^ 4 + C 1) (by rw [X_dvd_iff]; simp) (e := 1) (by rw [hχ]; norm_num)
  rw [hdeg] at h
  simp only [eval_mul, eval_pow, eval_add, eval_X, eval_C, pow_one] at h
  exact h.trans_eq (by norm_num)

end Examples

end Transformer.Weil
