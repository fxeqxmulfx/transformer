module

public import Transformer.Weil.PowCharSum

/-!
# The character sum for `y^d = f(x)` through a root of multiplicity prime to `d`

`PowCharSum` needs the order `d` of `χ` prime to `deg f`. A root `α` of `f` of multiplicity `e`
prime to `d` serves as well, through the substitution `x = α + 1/y`, which sends `α` to
infinity. Write `f = (X − α)^e G` with `G(α) ≠ 0`, and take `N > deg f` divisible by `d`. The
reflection `f̃ = X^N f(α + 1/X)` of `f(X + α) = X^e G(X + α)` is a polynomial of degree `N − e`
with `f̃(0) = 0` and `f̃(y) = y^N f(α + 1/y)` for `y ≠ 0`. As `χ(y)^N = 1`, the bijection
`y ↦ α + 1/y` (`0 ↦ α`, with `0⁻¹ = 0`) gives `∑_x χ(f(x)) = ∑_y χ(f̃(y))`: the term `x = α` is
`χ(0) = 0`, and so is `y = 0`. Now `N − e` is prime to `d`, and `PowCharSum` gives
`|∑_x χ(f(x))| ≤ 5 d (N − e) √q ≤ 5 d N √q`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

section Reflect

variable {F : Type*} [Field F]

/-- `f(X + α) = X^e G(X + α)` for `f = (X − α)^e G`. -/
theorem comp_X_add_C_pow_mul (α : F) (e : ℕ) (G : F[X]) :
    ((X - C α) ^ e * G).comp (X + C α) = X ^ e * G.comp (X + C α) := by
  rw [mul_comp, pow_comp, sub_comp, X_comp, C_comp, add_sub_cancel_right]

/-- `deg ((X − α)^e G) = e + deg G` for `G ≠ 0`. -/
theorem natDegree_X_sub_C_pow_mul (α : F) (e : ℕ) {G : F[X]} (hG : G ≠ 0) :
    ((X - C α) ^ e * G).natDegree = e + G.natDegree := by
  rw [natDegree_mul (pow_ne_zero _ (X_sub_C_ne_zero α)) hG, natDegree_pow, natDegree_X_sub_C,
    mul_one]

/-- **The degree after `x = α + 1/y`**: for `f = (X − α)^e G` with `G(α) ≠ 0` and `deg f ≤ N`,
the reflection `X^N f(α + 1/X)` of `f(X + α)` has degree `N − e`. -/
theorem natDegree_reflect_comp_X_add_C {α : F} {e : ℕ} {G : F[X]} (hG : G.eval α ≠ 0) {N : ℕ}
    (hN : ((X - C α) ^ e * G).natDegree ≤ N) :
    (reflect N (((X - C α) ^ e * G).comp (X + C α))).natDegree = N - e := by
  have hdeg := natDegree_X_sub_C_pow_mul α e (G := G) fun h => hG (by rw [h, eval_zero])
  have hcomp : (X ^ e * G.comp (X + C α)).natDegree = ((X - C α) ^ e * G).natDegree := by
    rw [← comp_X_add_C_pow_mul, natDegree_comp, natDegree_X_add_C, mul_one]
  rw [comp_X_add_C_pow_mul]
  refine natDegree_eq_of_le_of_coeff_ne_zero
    (natDegree_le_iff_coeff_eq_zero.mpr fun j hj => ?_) ?_
  · rw [coeff_reflect]
    rcases le_or_gt j N with hjN | hjN
    · rw [revAt_le hjN, coeff_X_pow_mul', ite_eq_right (by omega)]
    · rw [revAt_eq_self_of_lt hjN]
      exact coeff_eq_zero_of_natDegree_lt (by omega)
  · rw [coeff_reflect, revAt_le (by omega), Nat.sub_sub_self (by omega), coeff_X_pow_mul',
      ite_eq_left le_rfl, Nat.sub_self, coeff_zero_eq_eval_zero, eval_comp, eval_add, eval_X,
      eval_C, zero_add]
    exact hG

/-- `χ(y)^N = 1` for `y ≠ 0` when `χ^d = 1` and `d ∣ N`. -/
theorem mulChar_pow_eq_one_of_dvd {χ : MulChar F ℂ} {d N : ℕ} (hχ : χ ^ d = 1) (hdN : d ∣ N)
    {y : F} (hy : y ≠ 0) : χ y ^ N = 1 := by
  obtain ⟨k, rfl⟩ := hdN
  have h := MulChar.pow_apply_coe χ d (Units.mk0 y hy)
  rw [hχ, MulChar.one_apply_coe, Units.val_mk0] at h
  rw [pow_mul, ← h, one_pow]

variable [Fintype F]

/-- **The substitution `x = α + 1/y`**: if `f(α) = 0`, `deg f < N` and `χ(y)^N = 1` for `y ≠ 0`,
then `∑_x χ(f(x)) = ∑_y χ(f̃(y))` for the reflection `f̃ = X^N f(α + 1/X)` of `f(X + α)`. -/
theorem sum_mulChar_eval_reflect_comp {χ : MulChar F ℂ} {N : ℕ}
    (hN : ∀ y : F, y ≠ 0 → χ y ^ N = 1) {f : F[X]} {α : F} (hα : f.eval α = 0)
    (hfN : f.natDegree < N) :
    ∑ y, χ ((reflect N (f.comp (X + C α))).eval y) = ∑ x, χ (f.eval x) := by
  have hcomp : (f.comp (X + C α)).natDegree = f.natDegree := by
    rw [natDegree_comp, natDegree_X_add_C, mul_one]
  refine Fintype.sum_equiv ((Equiv.inv F).trans (Equiv.addRight α)) _ _ fun y => ?_
  simp only [Equiv.trans_apply, Equiv.inv_apply, Equiv.coe_addRight]
  rcases eq_or_ne y 0 with rfl | hy
  · rw [inv_zero, zero_add, hα, ← coeff_zero_eq_eval_zero, coeff_reflect,
      revAt_le (Nat.zero_le N), Nat.sub_zero, coeff_eq_zero_of_natDegree_lt (by omega)]
  · have : Invertible y⁻¹ := invertibleOfNonzero (inv_ne_zero hy)
    have h := eval₂_reflect_mul_pow (RingHom.id F) y⁻¹ N (f.comp (X + C α)) (by omega)
    rw [eval₂_id, eval₂_id, invOf_eq_inv, inv_inv, eval_comp, eval_add, eval_X, eval_C] at h
    rw [← h, map_mul, map_pow, hN _ (inv_ne_zero hy), mul_one]

end Reflect

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- **The character sum through a root**: for `χ^d = 1`, `d ≥ 2`, `χ(u) = 1 ⇒ u^c = 1`,
`d c + 1 = q = |F|`, and `f = (X − α)^e G` with `G(α) ≠ 0` and `e` prime to `d`: for
`N > deg f` divisible by `d` with `q ≥ 64`, `8 d N ≤ q` and `d² ≤ q`,
`‖∑_x χ(f(x))‖ ≤ 5 d N √q`. -/
theorem norm_sum_mulChar_eval_le_of_root {χ : MulChar F ℂ} {d c : ℕ} (hd : 2 ≤ d)
    (hχ : χ ^ d = 1) (hker : ∀ u, χ u = 1 → u ^ c = 1) (hq : d * c + 1 = Fintype.card F)
    {α : F} {e : ℕ} {G : F[X]} (hG : G.eval α ≠ 0) (he : Nat.Coprime d e) {N : ℕ}
    (hdN : d ∣ N) (hN : ((X - C α) ^ e * G).natDegree < N) (h64 : 64 ≤ Fintype.card F)
    (h8 : 8 * d * N ≤ Fintype.card F) (hd2 : d * d ≤ Fintype.card F) :
    ‖∑ x, χ (((X - C α) ^ e * G).eval x)‖ ≤ 5 * d * N * √(Fintype.card F) := by
  have he0 : e ≠ 0 := by
    rintro rfl
    rw [Nat.coprime_zero_right] at he
    omega
  have hfdeg := natDegree_X_sub_C_pow_mul α e (G := G) fun h => hG (by rw [h, eval_zero])
  have heN : e ≤ N := by omega
  have hα : ((X - C α) ^ e * G).eval α = 0 := by
    rw [eval_mul, eval_pow, eval_sub, eval_X, eval_C, sub_self, zero_pow he0, zero_mul]
  have hcop : Nat.Coprime d (N - e) := by
    have h1 : Nat.gcd d (N - e) ∣ e := by
      have h2 := Nat.dvd_sub ((Nat.gcd_dvd_left d (N - e)).trans hdN) (Nat.gcd_dvd_right d (N - e))
      rwa [Nat.sub_sub_self heN] at h2
    exact Nat.eq_one_of_dvd_one (he.gcd_eq_one ▸ Nat.dvd_gcd (Nat.gcd_dvd_left d (N - e)) h1)
  have hdeg := natDegree_reflect_comp_X_add_C hG hN.le
  rw [← sum_mulChar_eval_reflect_comp (fun y hy => mulChar_pow_eq_one_of_dvd hχ hdN hy) hα hN]
  have hb := norm_sum_mulChar_eval_le_of_coprime hd hχ hker hq
    (f := reflect N (((X - C α) ^ e * G).comp (X + C α))) (by rw [hdeg]; omega)
    (by rwa [hdeg]) h64 (by rw [hdeg]; exact (Nat.mul_le_mul_left _ (Nat.sub_le N e)).trans h8)
    hd2
  rw [hdeg] at hb
  refine hb.trans ?_
  gcongr
  exact Nat.sub_le N e

section Examples

attribute [local instance] fact_prime_71

example := comp_X_add_C_pow_mul (F := ℚ) 1 2 X

example := natDegree_X_sub_C_pow_mul (F := ℚ) 1 2 X_ne_zero

example := natDegree_reflect_comp_X_add_C (F := ℚ) (α := 0) (e := 1) (G := X + C 1) (by simp)
  (N := 2) (by rw [natDegree_X_sub_C_pow_mul _ _ (X_add_C_ne_zero 1), natDegree_X_add_C])

example := mulChar_pow_eq_one_of_dvd (F := ZMod 71)
  (χ := (quadraticChar (ZMod 71)).ringHomComp (Int.castRingHom ℂ)) (d := 2) (N := 4)
  ((quadraticChar_isQuadratic _).comp _).sq_eq_one (by norm_num) (y := 1) one_ne_zero

example := sum_mulChar_eval_reflect_comp (F := ZMod 71)
  (χ := (quadraticChar (ZMod 71)).ringHomComp (Int.castRingHom ℂ)) (N := 2)
  (fun _ hy => mulChar_pow_eq_one_of_dvd ((quadraticChar_isQuadratic _).comp _).sq_eq_one
    dvd_rfl hy) (f := X) (α := 0) (by simp) (by simp)

/-- For the Legendre symbol of `𝔽₇₁` and `f = X (X + 1)`, of even degree:
`|∑_x χ₂(x (x + 1))| ≤ 40 √71`. -/
example := norm_sum_mulChar_eval_le_of_root (F := ZMod 71)
  (χ := (quadraticChar (ZMod 71)).ringHomComp (Int.castRingHom ℂ)) (d := 2) (c := 35) le_rfl
  ((quadraticChar_isQuadratic _).comp _).sq_eq_one
  (fun u hu => by
    rw [MulChar.ringHomComp_apply, eq_intCast, Int.cast_eq_one] at hu
    have h := quadraticChar_eq_pow_of_char_ne_two' (F := ZMod 71)
      (by rw [ZMod.ringChar_zmod_n]; decide) u
    rw [hu, ZMod.card] at h
    simpa using h.symm)
  (by rw [ZMod.card]) (α := 0) (e := 1) (G := X + C 1) (by simp) (by norm_num) (N := 4)
  (by norm_num)
  (by rw [natDegree_X_sub_C_pow_mul _ _ (X_add_C_ne_zero 1), natDegree_X_add_C]; norm_num)
  (by rw [ZMod.card]; norm_num) (by rw [ZMod.card]; norm_num) (by rw [ZMod.card]; norm_num)

end Examples

end Transformer.Weil
