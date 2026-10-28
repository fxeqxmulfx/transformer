module

public import Transformer.Weil.Hyperelliptic
public import Transformer.Weil.QuadraticUnit
public import Transformer.Weil.Rationality
public import Transformer.Weil.WeilBound

/-!
# Weil's bound for `∑_x χ₂(f(x))`

For `p` odd, `χ₂` the Legendre symbol of `𝔽_p` and `f ∈ 𝔽_p[X]` of degree `m` that is not a
constant times a square, `|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p`; odd `m` is a special case:

* by Euler's criterion through the norm (`QuadraticUnit`), `S_d = ∑_{x ∈ L} χ_L(f(x))` for `L`
  with `p^d` elements (`NormSum`);
* over `L`, `f(X) R² = f(Y) S²` still forces `S = 0`, since `f` has a root of odd multiplicity
  (`IndependenceRoot`), so Stepanov's bound (`Hyperelliptic`) gives `|S_d| ≤ 7 m √p^d` once
  `p^d ≥ 64` and `p^d ≥ 8 m`;
* `c_n = 0` for `n ≥ m` (`QuadraticUnit`);
* `Rationality` turns the bounds on `S_d` into `|S_1| ≤ (m − 1) √p`, and `S_1 = ∑_x χ₂(f(x))`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- A polynomial that is not a constant times a square has positive degree. -/
theorem natDegree_pos_of_ne_C_mul_sq {K : Type*} [Field K] {f : K[X]}
    (hf : ∀ (c : K) (g : K[X]), f ≠ C c * g ^ 2) : 0 < f.natDegree :=
  Nat.pos_of_ne_zero fun h => hf (f.coeff 0) 1
    (by rw [one_pow, mul_one]; exact eq_C_of_natDegree_eq_zero h)

example := natDegree_pos_of_ne_C_mul_sq (K := ZMod 3) (f := X)
  (ne_C_mul_sq_of_odd (by rw [natDegree_X]; exact odd_one))

variable {p : ℕ} [Fact p.Prime]

/-- `|S_d| ≤ 7 m √p^d` for `χ₂(f(x))`, `f` not a constant times a square, once `p^d ≥ 64` and
`p^d ≥ 8 m`: Stepanov's bound over the field with `p^d` elements. -/
theorem norm_lPowerSum_quadraticChar_le (hp : p ≠ 2) {f : (ZMod p)[X]}
    (hf : ∀ (c : ZMod p) (g : (ZMod p)[X]), f ≠ C c * g ^ 2) {d : ℕ} (hd : d ≠ 0)
    (h64 : 64 ≤ p ^ d) (h8 : 8 * f.natDegree ≤ p ^ d) :
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
    (by rw [natDegree_map]; exact natDegree_pos_of_ne_C_mul_sq hf)
    (fun _ _ => eq_zero_of_C_mul_sq_eq_of_ne_C_mul_sq hf) (by rwa [hcard])
    (by rwa [hcard, natDegree_map])
  rw [natDegree_map, hcard, Nat.cast_pow, sqrt_pow_eq_pow_sqrt (Nat.cast_nonneg _)] at hb
  rw [← hS, hsum, Complex.norm_intCast]
  exact hb

/-- `X (X + 1)`, of even degree, is not a constant times a square: `0` is a simple root. -/
theorem X_mul_X_add_one_ne_C_mul_sq {K : Type*} [Field K] (c : K) (g : K[X]) :
    X * (X + C 1) ≠ C c * g ^ 2 := by
  refine ne_C_mul_sq_of_rootMultiplicity (a := 0) ?_ c g
  rw [rootMultiplicity_mul (mul_ne_zero X_ne_zero (X_add_C_ne_zero 1)),
    show (X : K[X]) = X - C 0 by simp, rootMultiplicity_X_sub_C_self,
    rootMultiplicity_eq_zero (by simp)]
  exact odd_one

example := norm_lPowerSum_quadraticChar_le (p := 3) (by norm_num) (f := X * (X + C 1))
  X_mul_X_add_one_ne_C_mul_sq (d := 4) (by norm_num) (by norm_num)
  (by rw [natDegree_mul X_ne_zero (X_add_C_ne_zero 1), natDegree_X, natDegree_X_add_C]; norm_num)

/-- **Weil's bound for the Legendre symbol**: for `p` odd and `f ∈ 𝔽_p[X]` of degree `m` that is
not a constant times a square, `|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p`. -/
theorem norm_sum_quadraticChar_eval_le_of_ne (hp : p ≠ 2) {f : (ZMod p)[X]}
    (hf : ∀ (c : ZMod p) (g : (ZMod p)[X]), f ≠ C c * g ^ 2) :
    ‖∑ x : ZMod p, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (f.eval x)‖ ≤
      ((f.natDegree : ℝ) - 1) * √p := by
  have hK : ringChar (ZMod p) ≠ 2 := by rwa [ZMod.ringChar_zmod_n]
  have h1 := sum_mulChar_norm_aeval (L := ZMod p)
    ((quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ)) f
  rw [Module.finrank_self] at h1
  simp only [Algebra.norm_self, MonoidHom.id_apply, coe_aeval_eq_eval] at h1
  rw [h1]
  have hpow (d : ℕ) : d < p ^ d := Nat.lt_pow_self (Fact.out : p.Prime).one_lt
  exact norm_lPowerSum_one_le (isMonicMul_mulWeight _ f) (natDegree_pos_of_ne_C_mul_sq hf)
    (fun n hn => lCoeff_mulWeight_quadraticChar_eq_zero_of_ne hK hf hn) (Real.sqrt_nonneg _)
    (n₀ := 8 * f.natDegree + 64) fun d hd => norm_lPowerSum_quadraticChar_le hp hf (by omega)
      (by have := hpow d; omega) (by have := hpow d; omega)

example := norm_sum_quadraticChar_eval_le_of_ne (p := 3) (by norm_num) (f := X * (X + C 1))
  X_mul_X_add_one_ne_C_mul_sq

/-- **Weil's bound for the Legendre symbol**, odd degree: for `p` odd and `f ∈ 𝔽_p[X]` of odd
degree `m`, `|∑_{x ∈ 𝔽_p} χ₂(f(x))| ≤ (m − 1) √p`. -/
theorem norm_sum_quadraticChar_eval_le (hp : p ≠ 2) {f : (ZMod p)[X]} (hf : Odd f.natDegree) :
    ‖∑ x : ZMod p, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (f.eval x)‖ ≤
      ((f.natDegree : ℝ) - 1) * √p :=
  norm_sum_quadraticChar_eval_le_of_ne hp (ne_C_mul_sq_of_odd hf)

example := norm_sum_quadraticChar_eval_le (p := 3) (by norm_num) (f := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩)

end Transformer.Weil
