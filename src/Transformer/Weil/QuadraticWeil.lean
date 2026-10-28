module

public import Transformer.Weil.Hyperelliptic
public import Transformer.Weil.NormSum
public import Transformer.Weil.Rationality
public import Transformer.Weil.WeilBound

/-!
# Weil's bound for `∑_x χ₂(f(x))`

For `p` odd, `χ₂` the Legendre symbol of `𝔽_p` and `f ∈ 𝔽_p[X]` of odd degree `m`,
`|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p`:

* Euler's criterion through the norm: `χ_L(y) = χ_K(N_{L/K} y)` for a finite extension `L` of a
  finite field `K` of odd order `q`, since `(q^d − 1)/2 = (1 + q + ⋯ + q^{d − 1}) (q − 1)/2`;
* so `S_d = ∑_{x ∈ L} χ_L(f(x))` for `L` with `p^d` elements (`NormSum`), and Stepanov's bound
  (`Hyperelliptic`) gives `|S_d| ≤ 7 m √p^d` once `p^d ≥ 64` and `p^d ≥ 8 m`;
* `χ₂(t)^m = −1` for a non-square `t`, since `m` is odd, so `c_n = 0` for `n ≥ m` (`MulWeight`);
* `Rationality` turns the bounds on `S_d` into `|S_1| ≤ (m − 1) √p`, and `S_1 = ∑_x χ₂(f(x))`.
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

/-- `c_n = 0` for `n ≥ m` for `χ₂(f(x))`, `m` odd: `χ₂(t)^m = −1` for a non-square `t`. -/
theorem lCoeff_mulWeight_quadraticChar_eq_zero {K : Type*} [Field K] [Fintype K] [DecidableEq K]
    (hK : ringChar K ≠ 2) {f : K[X]} (hf : Odd f.natDegree) {n : ℕ} (hn : f.natDegree ≤ n) :
    lCoeff (mulWeight ((quadraticChar K).ringHomComp (Int.castRingHom ℂ)) f) n = 0 := by
  obtain ⟨t, ht⟩ := quadraticChar_exists_neg_one hK
  refine lCoeff_mulWeight_eq_zero (t := t) (fun h => ?_) ?_ hn
  · rw [h, quadraticChar_zero] at ht
    exact absurd ht (by norm_num)
  · rw [MulChar.ringHomComp_apply, ht, map_neg, map_one, hf.neg_one_pow]
    norm_num

example := lCoeff_mulWeight_quadraticChar_eq_zero (K := ZMod 3)
  (by rw [ZMod.ringChar_zmod_n]; norm_num) (f := X) (by rw [natDegree_X]; exact odd_one)
  (n := 1) (by rw [natDegree_X])

variable {p : ℕ} [Fact p.Prime]

/-- `|S_d| ≤ 7 m √p^d` for `χ₂(f(x))` once `p^d ≥ 64` and `p^d ≥ 8 m`: Stepanov's bound over the
field with `p^d` elements. -/
theorem norm_lPowerSum_quadraticChar_le (hp : p ≠ 2) {f : (ZMod p)[X]} (hf : Odd f.natDegree)
    {d : ℕ} (hd : d ≠ 0) (h64 : 64 ≤ p ^ d) (h8 : 8 * f.natDegree ≤ p ^ d) :
    ‖lPowerSum (mulWeight ((quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ)) f) d‖ ≤
      7 * f.natDegree * √p ^ d := by
  classical
  let _ : Fintype (GaloisField p d) := Fintype.ofFinite _
  have hcard : Fintype.card (GaloisField p d) = p ^ d := by
    rw [← Nat.card_eq_fintype_card, GaloisField.card p d hd]
  have hK : ringChar (ZMod p) ≠ 2 := by rwa [ZMod.ringChar_zmod_n]
  have hL : ringChar (GaloisField p d) ≠ 2 := by
    rwa [← Algebra.ringChar_eq (ZMod p) (GaloisField p d)]
  have hS := sum_mulChar_norm_aeval (L := GaloisField p d)
    ((quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ)) f
  rw [GaloisField.finrank p hd] at hS
  have hsum : ∑ x : GaloisField p d, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ)
      (Algebra.norm (ZMod p) (aeval x f)) = ((∑ x, quadraticChar (GaloisField p d)
        ((f.map (algebraMap (ZMod p) (GaloisField p d))).eval x) : ℤ) : ℂ) := by
    push_cast
    refine sum_congr rfl fun x _ => ?_
    rw [MulChar.ringHomComp_apply, quadraticChar_norm hK, eval_map_algebraMap]
    rfl
  have hb := abs_sum_quadraticChar_le hL (f := f.map (algebraMap (ZMod p) (GaloisField p d)))
    (by rw [natDegree_map]; exact hf.pos)
    (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd (by rwa [natDegree_map])) (by rwa [hcard])
    (by rwa [hcard, natDegree_map])
  rw [natDegree_map, hcard, Nat.cast_pow, sqrt_pow_eq_pow_sqrt (Nat.cast_nonneg _)] at hb
  rw [← hS, hsum, Complex.norm_intCast]
  exact hb

example := norm_lPowerSum_quadraticChar_le (p := 3) (by norm_num) (f := X)
  (by rw [natDegree_X]; exact odd_one) (d := 4) (by norm_num) (by norm_num)
  (by rw [natDegree_X]; norm_num)

/-- **Weil's bound for the Legendre symbol**: for `p` odd and `f ∈ 𝔽_p[X]` of odd degree `m`,
`|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p`. -/
theorem norm_sum_quadraticChar_eval_le (hp : p ≠ 2) {f : (ZMod p)[X]} (hf : Odd f.natDegree) :
    ‖∑ x : ZMod p, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (f.eval x)‖ ≤
      ((f.natDegree : ℝ) - 1) * √p := by
  have hK : ringChar (ZMod p) ≠ 2 := by rwa [ZMod.ringChar_zmod_n]
  have h1 := sum_mulChar_norm_aeval (L := ZMod p)
    ((quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ)) f
  rw [Module.finrank_self] at h1
  simp only [Algebra.norm_self, MonoidHom.id_apply, coe_aeval_eq_eval] at h1
  rw [h1]
  have hpow (d : ℕ) : d < p ^ d := Nat.lt_pow_self (Fact.out : p.Prime).one_lt
  exact norm_lPowerSum_one_le (isMonicMul_mulWeight _ f) hf.pos
    (fun n hn => lCoeff_mulWeight_quadraticChar_eq_zero hK hf hn) (Real.sqrt_nonneg _)
    (n₀ := 8 * f.natDegree + 64) fun d hd => norm_lPowerSum_quadraticChar_le hp hf (by omega)
      (by have := hpow d; omega) (by have := hpow d; omega)

example := norm_sum_quadraticChar_eval_le (p := 3) (by norm_num) (f := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩)

end Transformer.Weil
