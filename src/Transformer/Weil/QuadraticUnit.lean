module

public import Mathlib.RingTheory.AdjoinRoot
public import Transformer.Weil.IndependenceRoot
public import Transformer.Weil.MulMod
public import Transformer.Weil.NormSum

/-!
# `c_n = 0` for `χ₂(f(x))`, `f` not a constant times a square

Euler's criterion through the norm: `χ_L(y) = χ_K(N_{L/K} y)` for a finite extension `L` of a
finite field `K` of odd order `q`, since `(q^d − 1)/2 = (1 + q + ⋯ + q^{d − 1}) (q − 1)/2`.

If `f` is not `λ g²`, then `f = P^e Q` with `P` monic irreducible, `e` odd and `P ∤ Q`
(`IndependenceRoot`). In `L = K[X]/(P)` take a non-square `y = v(α)`, and with `a P + b Q = 1`
put `u = 1 + Q b (v − 1)`: then `u(α) = y` and `u ≡ 1` modulo `Q`. So `Res(P, u) = N_{L/K}(y)`
is a non-square, `Res(Q, u) = λ^N` for the leading coefficient `λ` of `Q`, which is that of `f`,
and `χ₂(Res(f, u)) = −χ₂(λ)^N`. `MulMod` then gives `c_n = 0` for every `n ≥ deg f`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- **Euler's criterion through the norm**: `χ_L(y) = χ_K(N_{L/K} y)` for `K` of odd
characteristic. -/
theorem quadraticChar_norm {K L : Type*} [Field K] [Fintype K] [DecidableEq K] [Field L]
    [Fintype L] [DecidableEq L] [Algebra K L] (hK : ringChar K ≠ 2) (y : L) :
    quadraticChar K (Algebra.norm K y) = quadraticChar L y := by
  have hL : ringChar L ≠ 2 := by rwa [← Algebra.ringChar_eq K L]
  refine Int.cast_injOn_of_ringChar_ne_two hL (quadraticChar_isQuadratic K _)
    (quadraticChar_isQuadratic L y) ?_
  obtain ⟨a, ha⟩ : ∃ a, Fintype.card K = 2 * a + 1 :=
    ⟨Fintype.card K / 2, by have := FiniteField.odd_card_of_char_ne_two hK; omega⟩
  have hgeom := geom_sum_mul_add (2 * a) (Module.finrank K L)
  rw [quadraticChar_eq_pow_of_char_ne_two' hL, ← map_intCast (algebraMap K L),
    quadraticChar_eq_pow_of_char_ne_two' hK, map_pow, FiniteField.algebraMap_norm_eq_prod_pow,
    prod_pow_eq_pow_sum, ← pow_mul, Module.card_eq_pow_finrank (K := K) (V := L),
    Nat.card_eq_fintype_card, ha, ← hgeom, show (2 * a + 1) / 2 = a by omega]
  congr 1
  generalize ∑ i ∈ range (Module.finrank K L), (2 * a + 1) ^ i = s
  rw [show s * (2 * a) = 2 * (s * a) by ring]
  omega

example := quadraticChar_norm (K := ZMod 3) (L := ZMod 3)
  (by rw [ZMod.ringChar_zmod_n]; norm_num) 2

variable {K : Type*} [Field K] [Fintype K] [DecidableEq K]

/-- For `P` monic irreducible, `e` odd and `P ∤ Q`, a `u` prime to `f = P^e Q` with
`χ₂(Res(f, u)) = −χ₂(λ)^N`, `deg u ≤ N`, `λ` the leading coefficient of `Q`. -/
theorem exists_isCoprime_quadraticChar_resultant (hK : ringChar K ≠ 2) {P Q : K[X]} {e : ℕ}
    (hPm : P.Monic) (hP : Irreducible P) (he : Odd e) (hPQ : ¬P ∣ Q) :
    ∃ u : K[X], ∃ N : ℕ, IsCoprime u (P ^ e * Q) ∧ u.natDegree ≤ N ∧
      quadraticChar K (resultant (P ^ e * Q) u (P ^ e * Q).natDegree N) =
        -quadraticChar K Q.leadingCoeff ^ N := by
  classical
  have : Fact (Irreducible P) := ⟨hP⟩
  have : Module.Finite K (AdjoinRoot P) := (AdjoinRoot.powerBasis hP.ne_zero).finite
  have : Finite (AdjoinRoot P) := Module.finite_of_finite K
  let : Fintype (AdjoinRoot P) := Fintype.ofFinite _
  have hL : ringChar (AdjoinRoot P) ≠ 2 := by rwa [← Algebra.ringChar_eq K (AdjoinRoot P)]
  obtain ⟨y, hy⟩ := quadraticChar_exists_neg_one hL
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
    rw [h0, quadraticChar_zero] at hy
    exact absurd hy (by norm_num)
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
  rw [natDegree_mul (pow_ne_zero _ hP.ne_zero) hQ0, resultant_mul_left _ _ _ _ huN,
    resultant_pow_left _ _ _ _ (by rw [hPm.leadingCoeff, one_pow]; exact one_ne_zero) huN,
    map_mul, map_pow, hPu, quadraticChar_norm hK, hy, hQu, map_pow, he.neg_one_pow, neg_one_mul]

example := exists_isCoprime_quadraticChar_resultant (K := ZMod 3)
  (by rw [ZMod.ringChar_zmod_n]; norm_num) monic_X irreducible_X odd_one (Q := 1)
  (by rw [X_dvd_iff]; simp)

/-- **`L` is a polynomial of degree `< m` for `χ₂(f(x))`, `f` not a constant times a square**:
`c_n = 0` for every `n ≥ deg f`. -/
theorem lCoeff_mulWeight_quadraticChar_eq_zero_of_ne (hK : ringChar K ≠ 2) {f : K[X]}
    (hf : ∀ (c : K) (g : K[X]), f ≠ C c * g ^ 2) {n : ℕ} (hn : f.natDegree ≤ n) :
    lCoeff (mulWeight ((quadraticChar K).ringHomComp (Int.castRingHom ℂ)) f) n = 0 := by
  obtain ⟨P, Q, e, hPm, hP, he, hPQ, rfl⟩ := exists_monic_irreducible_odd_pow_mul hf
  obtain ⟨u, N, hu, hN, hχ⟩ := exists_isCoprime_quadraticChar_resultant hK hPm hP he hPQ
  have hQ0 : Q ≠ 0 := fun h => hPQ (h ▸ dvd_zero P)
  have hc : (quadraticChar K Q.leadingCoeff : ℂ) ^ N ≠ 0 := pow_ne_zero _ (Int.cast_ne_zero.mpr
    (quadraticChar_eq_zero_iff.not.mpr (leadingCoeff_ne_zero.mpr hQ0)))
  refine lCoeff_mulWeight_eq_zero_of_isCoprime (mul_ne_zero (pow_ne_zero _ hP.ne_zero) hQ0) hu hN
    ?_ hn
  rw [MulChar.ringHomComp_apply, MulChar.ringHomComp_apply, hχ, leadingCoeff_mul,
    leadingCoeff_pow, hPm.leadingCoeff, one_pow, one_mul]
  simp only [Int.coe_castRingHom, Int.cast_neg, Int.cast_pow]
  intro h
  exact hc (by linear_combination -h / 2)

example := lCoeff_mulWeight_quadraticChar_eq_zero_of_ne (K := ZMod 3)
  (by rw [ZMod.ringChar_zmod_n]; norm_num) (f := X)
  (ne_C_mul_sq_of_odd (by rw [natDegree_X]; exact odd_one)) (n := 1) (by rw [natDegree_X])

end Transformer.Weil
