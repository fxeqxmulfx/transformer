module

public import Mathlib.FieldTheory.Finite.GaloisField
public import Mathlib.RingTheory.AdjoinRoot
public import Transformer.Weil.MulMod
public import Transformer.Weil.NormSum

/-!
# `c_n = 0` for `χ(f(x))`, `f` with an irreducible factor `P^e`, `χ^e ≠ 1`

A constant `u = t` gives `χ(Res(f, t)) = χ(t)^m` (`MulWeight`), useless when the order of `χ`
divides `m = deg f`. A unit built at an irreducible factor works for every order.

Let `f = P^e Q` with `P` monic irreducible, `P ∤ Q` and `χ^e ≠ 1`, and take `t ∈ K` with
`χ(t)^e ≠ 1`. The norm of `L = K[X]/(P)` onto `K` is surjective, so `t = N_{L/K}(y)` with
`y = v(α)`, `α` the class of `X`. With `a P + b Q = 1` put `u = 1 + Q b (v − 1)`: then
`u(α) = y` and `u ≡ 1` modulo `Q`. So `Res(P, u) = t`, `Res(Q, u) = λ^N` for the leading
coefficient `λ` of `Q`, which is that of `f`, and `χ(Res(f, u)) = χ(t)^e χ(λ)^N ≠ χ(λ)^N`.
`MulMod` then gives `c_n = 0` for every `n ≥ deg f`. This is `QuadraticUnit` with the norm's
surjectivity in place of Euler's criterion.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {K : Type*} [Field K] [Fintype K]

/-- For `P` monic irreducible, `P ∤ Q` and `χ^e ≠ 1`, a `u` prime to `f = P^e Q` with
`χ(Res(f, u)) ≠ χ(λ)^N`, `deg u ≤ N`, `λ` the leading coefficient of `f`. -/
theorem exists_isCoprime_mulChar_resultant {χ : MulChar K ℂ} {P Q : K[X]} {e : ℕ}
    (hPm : P.Monic) (hP : Irreducible P) (hPQ : ¬P ∣ Q) (he : χ ^ e ≠ 1) :
    ∃ u : K[X], ∃ N : ℕ, IsCoprime u (P ^ e * Q) ∧ u.natDegree ≤ N ∧
      χ (resultant (P ^ e * Q) u (P ^ e * Q).natDegree N) ≠ χ (P ^ e * Q).leadingCoeff ^ N := by
  classical
  have : Fact (Irreducible P) := ⟨hP⟩
  have : Module.Finite K (AdjoinRoot P) := (AdjoinRoot.powerBasis hP.ne_zero).finite
  have : Finite (AdjoinRoot P) := Module.finite_of_finite K
  obtain ⟨t, ht⟩ := MulChar.ne_one_iff.mp he
  rw [MulChar.pow_apply_coe] at ht
  obtain ⟨y, hy⟩ := FiniteField.norm_surjective K (AdjoinRoot P) (t : K)
  obtain ⟨v, rfl⟩ := AdjoinRoot.mk_surjective y
  obtain ⟨a, b, hab⟩ := hP.coprime_iff_not_dvd.mpr hPQ
  have hQ0 : Q ≠ 0 := fun h => hPQ (h ▸ dvd_zero P)
  have hu : AdjoinRoot.mk P (1 + Q * (b * (v - 1))) = AdjoinRoot.mk P v := by
    have h1 := congrArg (AdjoinRoot.mk P) hab
    rw [map_add, map_mul, map_mul, AdjoinRoot.mk_self, mul_zero, zero_add, map_one] at h1
    rw [map_add, map_one, map_mul, map_mul, map_sub, map_one]
    linear_combination (AdjoinRoot.mk P v - 1) * h1
  have huN : (1 + Q * (b * (v - 1))).natDegree ≤ (b * (v - 1)).natDegree + Q.natDegree :=
    (natDegree_add_le _ _).trans (max_le (by rw [natDegree_one]; exact Nat.zero_le _)
      (natDegree_mul_le.trans (add_comm _ _).le))
  refine ⟨1 + Q * (b * (v - 1)), (b * (v - 1)).natDegree + Q.natDegree, ?_, huN, ?_⟩
  · refine IsCoprime.mul_right (IsCoprime.pow_right ?_) ⟨1, -(b * (v - 1)), by ring⟩
    refine (hP.coprime_iff_not_dvd.mpr fun hdvd => ?_).symm
    have h0 : AdjoinRoot.mk P v = 0 := by rw [← hu]; exact AdjoinRoot.mk_eq_zero.mpr hdvd
    rw [h0, Algebra.norm_zero] at hy
    exact t.ne_zero hy.symm
  have hPu : resultant P (1 + Q * (b * (v - 1))) P.natDegree
      ((b * (v - 1)).natDegree + Q.natDegree) = Algebra.norm K (AdjoinRoot.mk P v) := by
    obtain ⟨k, hk⟩ := Nat.exists_eq_add_of_le huN
    rw [hk, resultant_add_right_deg _ _ _ _ _ le_rfl, coeff_natDegree, hPm.leadingCoeff, one_pow,
      one_mul, ← hu, ← AdjoinRoot.aeval_eq, norm_aeval_eq_resultant_pow,
      AdjoinRoot.minpoly_root hP.ne_zero, hPm.leadingCoeff, inv_one, C_1, mul_one,
      (AdjoinRoot.powerBasis hP.ne_zero).finrank, AdjoinRoot.powerBasis_dim,
      Nat.div_self hP.natDegree_pos, pow_one]
  have hQu : resultant Q (1 + Q * (b * (v - 1))) Q.natDegree
      ((b * (v - 1)).natDegree + Q.natDegree) = Q.leadingCoeff ^ ((b * (v - 1)).natDegree +
        Q.natDegree) := by
    rw [resultant_add_mul_right _ _ _ _ _ le_rfl le_rfl, resultant_one_right, coeff_natDegree]
  have hc : χ Q.leadingCoeff ^ ((b * (v - 1)).natDegree + Q.natDegree) ≠ 0 :=
    pow_ne_zero _ ((Ne.isUnit (leadingCoeff_ne_zero.mpr hQ0)).map χ).ne_zero
  rw [natDegree_mul (pow_ne_zero _ hP.ne_zero) hQ0, resultant_mul_left _ _ _ _ huN,
    resultant_pow_left _ _ _ _ (by rw [hPm.leadingCoeff, one_pow]; exact one_ne_zero) huN, map_mul,
    map_pow, hPu, hy, hQu, map_pow, leadingCoeff_mul, leadingCoeff_pow, hPm.leadingCoeff, one_pow,
    one_mul]
  exact fun h => ht (mul_right_cancel₀ hc (h.trans (one_mul _).symm))

/-- **`L` is a polynomial of degree `< m` for `χ(f(x))`, `f = P^e Q`, `χ^e ≠ 1`**: for `P`
monic irreducible and `P ∤ Q`, `c_n = 0` for every `n ≥ deg f`. -/
theorem lCoeff_mulWeight_eq_zero_of_pow_mul {χ : MulChar K ℂ} {P Q : K[X]} {e : ℕ}
    (hPm : P.Monic) (hP : Irreducible P) (hPQ : ¬P ∣ Q) (he : χ ^ e ≠ 1) {n : ℕ}
    (hn : (P ^ e * Q).natDegree ≤ n) : lCoeff (mulWeight χ (P ^ e * Q)) n = 0 := by
  obtain ⟨u, N, hu, hN, hχ⟩ := exists_isCoprime_mulChar_resultant hPm hP hPQ he
  have hQ0 : Q ≠ 0 := fun h => hPQ (h ▸ dvd_zero P)
  exact lCoeff_mulWeight_eq_zero_of_isCoprime (mul_ne_zero (pow_ne_zero _ hP.ne_zero) hQ0) hu hN
    hχ hn

/-- The Legendre symbol of `𝔽₃` as a character with values in `ℂ` is not trivial. -/
theorem quadraticChar_zmod3_ringHomComp_ne_one :
    (quadraticChar (ZMod 3)).ringHomComp (Int.castRingHom ℂ) ≠ 1 :=
  (MulChar.ringHomComp_eq_one_iff Int.cast_injective).not.mpr
    (quadraticChar_ne_one (by rw [ZMod.ringChar_zmod_n]; norm_num))

/-- `X + 1` is not divisible by `X`. -/
theorem not_X_dvd_X_add_one {R : Type*} [CommRing R] [Nontrivial R] :
    ¬(X : R[X]) ∣ X + C 1 := by
  rw [X_dvd_iff]
  simp

example := exists_isCoprime_mulChar_resultant (K := ZMod 3) (e := 1) monic_X irreducible_X
  not_X_dvd_X_add_one (by rw [pow_one]; exact quadraticChar_zmod3_ringHomComp_ne_one)

/-- For the Legendre symbol of `𝔽₃` and `f = X (X + 1)`, of even degree: `c_2 = 0`. -/
example := lCoeff_mulWeight_eq_zero_of_pow_mul (K := ZMod 3) (e := 1) monic_X irreducible_X
  not_X_dvd_X_add_one (by rw [pow_one]; exact quadraticChar_zmod3_ringHomComp_ne_one) (n := 2)
  (by rw [pow_one, natDegree_mul X_ne_zero (X_add_C_ne_zero 1), natDegree_X, natDegree_X_add_C])

end Transformer.Weil
