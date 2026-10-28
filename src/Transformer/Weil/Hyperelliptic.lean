module

public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic
public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Weil.Stepanov

/-!
# Weil's bound for `y² = f(x)`

For `f` of degree `m ≥ 1` such that `f(X) R(X, Y)² = f(Y) S(X, Y)²` forces `S = 0` (odd `m`, or
a root of odd multiplicity), take `ℓ = ⌊√q⌋`, `K = ⌊(q − m)/2⌋`, `J = ⌊ℓ/2⌋ + 2m`: Stepanov's
conditions hold once `q ≥ 64` and `8 m ≤ q`, and `#{x : f(x)^{(q−1)/2} = a} ≤ q/2 + 3 m √q` for
`a ≠ 0`. By Euler's criterion, and as `f` has at most `m` roots, `|∑_x χ₂(f(x))| ≤ 7 m √q`; so
`|#{y² = f(x)} − q| ≤ 7 m √q`, as the curve has `q + ∑_x χ₂(f(x))` affine points: the Riemann
hypothesis for `y² = f(x)`, with `7m` in place of Weil's `m − 1`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- Stepanov's dimension count for `ℓ = ⌊√q⌋`, `K = ⌊(q − m)/2⌋`, `J = ⌊ℓ/2⌋ + 2m`, `q` odd,
`m ≥ 1`, `q ≥ 64`, `8 m ≤ q`: `ℓ (K + J + (ℓ − 1)(m − 1)) < 2 J K`. -/
theorem stepanov_params {q m : ℕ} (hq : Odd q) (hm : 1 ≤ m) (h64 : 64 ≤ q) (h8 : 8 * m ≤ q) :
    q.sqrt * ((q - m) / 2 + (q.sqrt / 2 + 2 * m) + (q.sqrt - 1) * (m - 1)) <
      2 * ((q.sqrt / 2 + 2 * m) * ((q - m) / 2)) := by
  obtain ⟨t, ht⟩ := hq
  have hℓ2 : q.sqrt * q.sqrt ≤ q := Nat.sqrt_le q
  have hℓ8 : 8 ≤ q.sqrt := Nat.le_sqrt.mpr (by omega)
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, q.sqrt = ℓ := ⟨_, rfl⟩
  rw [hℓ] at hℓ2 hℓ8 ⊢
  have h8ℓ : 8 * ℓ ≤ q := (Nat.mul_le_mul_right _ hℓ8).trans hℓ2
  obtain ⟨J, hJ⟩ : ∃ J, ℓ / 2 + 2 * m = J := ⟨_, rfl⟩
  obtain ⟨K, hK⟩ : ∃ K, (q - m) / 2 = K := ⟨_, rfl⟩
  rw [hJ, hK]
  have hKlo : (q : ℤ) - m - 1 ≤ 2 * K := by omega
  have hKhi : 2 * (K : ℤ) ≤ q - m := by omega
  have hJlo : (ℓ : ℤ) - 1 + 4 * m ≤ 2 * J := by omega
  have hsq : (ℓ : ℤ) * ℓ ≤ q := by exact_mod_cast hℓ2
  have hℓm : 4 * (m : ℤ) + 4 * ℓ ≤ q := by omega
  have pa := mul_nonneg (by omega : (0 : ℤ) ≤ J) (sub_nonneg.mpr hKlo)
  have pb := mul_nonneg (by omega : (0 : ℤ) ≤ ℓ) (sub_nonneg.mpr hKhi)
  have pc := mul_nonneg (sub_nonneg.mpr hJlo) (by omega : (0 : ℤ) ≤ q - m - 1 - ℓ)
  have pd := mul_nonneg (sub_nonneg.mpr hsq) (by omega : (0 : ℤ) ≤ 2 * m - 1)
  have pe := mul_nonneg (by omega : (0 : ℤ) ≤ m) (sub_nonneg.mpr hℓm)
  have pf := mul_nonneg (by omega : (0 : ℤ) ≤ m - 1) (by omega : (0 : ℤ) ≤ ℓ)
  have pg := mul_nonneg (by omega : (0 : ℤ) ≤ m) (by omega : (0 : ℤ) ≤ m - 1)
  zify [(by omega : 1 ≤ ℓ), hm]
  linarith

example := stepanov_params (q := 65) (m := 1) ⟨32, rfl⟩ le_rfl (by norm_num) (by norm_num)

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- **Stepanov's bound**, explicit: for `q = |F|` odd, `q ≥ 64`, `f` of degree `m ≥ 1` with
`8 m ≤ q` such that `f(X) R² = f(Y) S²` forces `S = 0`, and `a ≠ 0`,
`#{x : f(x)^{(q−1)/2} = a} ≤ q/2 + 3 m √q`. -/
theorem card_pow_eq_le_sqrt {f : F[X]} (hm : 1 ≤ f.natDegree)
    (hsq : ∀ R S : F[X][X], C f * R ^ 2 = f.map C * S ^ 2 → S = 0) (hq : Odd (Fintype.card F))
    (h64 : 64 ≤ Fintype.card F) (h8 : 8 * f.natDegree ≤ Fintype.card F) {a : F} (ha : a ≠ 0) :
    (#{x | f.eval x ^ (Fintype.card F / 2) = a} : ℝ) ≤
      Fintype.card F / 2 + 3 * f.natDegree * √(Fintype.card F) := by
  have hℓ2 := Nat.sqrt_le (Fintype.card F)
  have hℓ3 := Nat.lt_succ_sqrt (Fintype.card F)
  have hℓ8 : 8 ≤ (Fintype.card F).sqrt := Nat.le_sqrt.mpr (by omega)
  have h8ℓ := (Nat.mul_le_mul_right _ hℓ8).trans hℓ2
  have hcore := mul_card_pow_eq_le hm hsq hq ha (ℓ := (Fintype.card F).sqrt)
    (J := (Fintype.card F).sqrt / 2 + 2 * f.natDegree) (K := (Fintype.card F - f.natDegree) / 2)
    (by omega) (by omega) (stepanov_params hq hm h64 h8)
  obtain ⟨t, ht⟩ := hq
  set q := Fintype.card F
  set m := f.natDegree
  set N := #{x | f.eval x ^ (q / 2) = a}
  obtain ⟨ℓ, hℓ⟩ : ∃ ℓ, q.sqrt = ℓ := ⟨_, rfl⟩
  rw [hℓ] at hcore hℓ2 hℓ3 hℓ8
  obtain ⟨J, hJ⟩ : ∃ J, ℓ / 2 + 2 * m = J := ⟨_, rfl⟩
  obtain ⟨K, hK⟩ : ∃ K, (q - m) / 2 = K := ⟨_, rfl⟩
  obtain ⟨c, hc⟩ : ∃ c, q / 2 = c := ⟨_, rfl⟩
  rw [hJ, hK, hc] at hcore
  have hcR : (ℓ : ℝ) * N ≤ (K : ℝ) - 1 + c * m + q * ((J : ℝ) - 1) := by
    have := (Nat.cast_le (α := ℝ)).mpr hcore
    push_cast [Nat.cast_sub (by omega : 1 ≤ K), Nat.cast_sub (by omega : 1 ≤ J)] at this
    exact this
  have eK : 2 * (K : ℝ) + m ≤ q := by exact_mod_cast (by omega : 2 * K + m ≤ q)
  have ec : 2 * (c : ℝ) + 1 = q := by exact_mod_cast (by omega : 2 * c + 1 = q)
  have eJ : 2 * (J : ℝ) ≤ ℓ + 4 * m := by exact_mod_cast (by omega : 2 * J ≤ ℓ + 4 * m)
  set s := √(q : ℝ)
  have hs2 : s * s = q := Real.mul_self_sqrt (Nat.cast_nonneg _)
  have hs8 : (8 : ℝ) ≤ s :=
    Real.le_sqrt_of_sq_le (by norm_num; exact_mod_cast (by omega : 64 ≤ q))
  have hsℓ : s < ℓ + 1 := by
    rw [Real.sqrt_lt' (by positivity), sq]
    exact_mod_cast hℓ3
  have ecm : 2 * (c : ℝ) * m + m = q * m := by rw [← ec]; ring
  have eqJ : (q : ℝ) * (2 * J) ≤ q * (ℓ + 4 * m) := mul_le_mul_of_nonneg_left eJ (by positivity)
  have hm0 : (0 : ℝ) ≤ m := Nat.cast_nonneg _
  have P1 := mul_nonneg (mul_nonneg hm0 (Real.sqrt_nonneg (q : ℝ))) (by linarith : 0 ≤ ℓ + 1 - s)
  have P2 := mul_nonneg (mul_nonneg hm0 (Real.sqrt_nonneg (q : ℝ))) (by linarith : 0 ≤ s - 6)
  have hms : (m : ℝ) * s * s = m * q := by rw [mul_assoc, hs2]
  have key : (ℓ : ℝ) * (2 * N) ≤ ℓ * (q + 6 * m * s) := by
    have hq0 : (0 : ℝ) ≤ q := Nat.cast_nonneg _
    linarith
  have hℓ0 : (0 : ℝ) < ℓ := by exact_mod_cast (by omega : 0 < ℓ)
  linarith [le_of_mul_le_mul_left key hℓ0]

/-- **Weil's bound for the quadratic character**, with Stepanov's constant: for
`ringChar F ≠ 2`, `q = |F| ≥ 64` and `f` of degree `m ≥ 1` with `8 m ≤ q` such that
`f(X) R² = f(Y) S²` forces `S = 0`, `|∑_x χ₂(f(x))| ≤ 7 m √q`. -/
theorem abs_sum_quadraticChar_le (hF : ringChar F ≠ 2) {f : F[X]} (hm : 1 ≤ f.natDegree)
    (hsq : ∀ R S : F[X][X], C f * R ^ 2 = f.map C * S ^ 2 → S = 0)
    (h64 : 64 ≤ Fintype.card F) (h8 : 8 * f.natDegree ≤ Fintype.card F) :
    |((∑ x, quadraticChar F (f.eval x) : ℤ) : ℝ)| ≤ 7 * f.natDegree * √(Fintype.card F) := by
  have hq : Odd (Fintype.card F) := Nat.odd_iff.mpr (FiniteField.odd_card_of_char_ne_two hF)
  have hc0 : Fintype.card F / 2 ≠ 0 := by omega
  have hne : (-1 : F) ≠ 1 := Ring.neg_one_ne_one_of_char_ne_two hF
  have h0ne : (0 : F) ≠ -1 := (neg_ne_zero.mpr one_ne_zero).symm
  have hf0 : f ≠ 0 := ne_zero_of_natDegree_gt hm
  have hpt (x : F) : quadraticChar F (f.eval x) =
      (if f.eval x ^ (Fintype.card F / 2) = 1 then 1 else 0) -
        (if f.eval x ^ (Fintype.card F / 2) = -1 then 1 else 0) := by
    by_cases h0 : f.eval x = 0
    · rw [h0, quadraticChar_zero, zero_pow hc0, ite_eq_right zero_ne_one, ite_eq_right h0ne, sub_zero]
    · rw [quadraticChar_eq_pow_of_char_ne_two hF h0]
      rcases FiniteField.pow_dichotomy hF h0 with h | h
      · rw [h, ite_eq_left rfl, ite_eq_left rfl, ite_eq_right hne.symm, sub_zero]
      · rw [h, ite_eq_right hne, ite_eq_right hne, ite_eq_left rfl, zero_sub]
  have hcount (x : F) : ((if f.eval x = 0 then 1 else 0) +
      (if f.eval x ^ (Fintype.card F / 2) = 1 then 1 else 0) +
        (if f.eval x ^ (Fintype.card F / 2) = -1 then 1 else 0) : ℕ) = 1 := by
    by_cases h0 : f.eval x = 0
    · rw [ite_eq_left h0, h0, zero_pow hc0, ite_eq_right zero_ne_one, ite_eq_right h0ne]
    · rw [ite_eq_right h0]
      rcases FiniteField.pow_dichotomy hF h0 with h | h
      · rw [h, ite_eq_left rfl, ite_eq_right hne.symm]
      · rw [h, ite_eq_right hne, ite_eq_left rfl]
  have hsum : ∑ x, quadraticChar F (f.eval x) =
      (#{x | f.eval x ^ (Fintype.card F / 2) = 1} : ℤ) -
        #{x | f.eval x ^ (Fintype.card F / 2) = -1} := by
    rw [Finset.sum_congr rfl fun x _ => hpt x, Finset.sum_sub_distrib, Finset.sum_boole,
      Finset.sum_boole]
  have hpart : #{x | f.eval x = 0} + #{x | f.eval x ^ (Fintype.card F / 2) = 1} +
      #{x | f.eval x ^ (Fintype.card F / 2) = -1} = Fintype.card F := by
    rw [Finset.card_filter, Finset.card_filter, Finset.card_filter, ← Finset.sum_add_distrib,
      ← Finset.sum_add_distrib, Finset.sum_congr rfl fun x _ => hcount x, Finset.sum_const,
      Finset.card_univ, smul_eq_mul, mul_one]
  have hroots : #{x | f.eval x = 0} ≤ f.natDegree :=
    card_le_degree_of_subset_roots fun x hx => (mem_roots hf0).mpr (Finset.mem_filter.mp hx).2
  have h1 := card_pow_eq_le_sqrt hm hsq hq h64 h8 one_ne_zero
  have h2 := card_pow_eq_le_sqrt hm hsq hq h64 h8 (neg_ne_zero.mpr one_ne_zero)
  have hpartR : (#{x | f.eval x = 0} : ℝ) + #{x | f.eval x ^ (Fintype.card F / 2) = 1} +
      #{x | f.eval x ^ (Fintype.card F / 2) = -1} = Fintype.card F := by exact_mod_cast hpart
  have hrootsR : (#{x | f.eval x = 0} : ℝ) ≤ f.natDegree := by exact_mod_cast hroots
  have hs1 : (f.natDegree : ℝ) ≤ f.natDegree * √(Fintype.card F) := le_mul_of_one_le_right
    (Nat.cast_nonneg _) (Real.one_le_sqrt.mpr (by exact_mod_cast (by omega : 1 ≤ Fintype.card F)))
  rw [hsum]
  push_cast
  rw [abs_le]
  constructor <;> linarith

/-- **The Riemann hypothesis for `y² = f(x)`**, `f` of degree `m ≥ 1` such that
`f(X) R² = f(Y) S²` forces `S = 0`, `ringChar F ≠ 2`, `q = |F| ≥ 64`, `8 m ≤ q`:
`|#{(x, y) : y² = f(x)} − q| ≤ 7 m √q`. -/
theorem abs_card_sq_eq_sub_le (hF : ringChar F ≠ 2) {f : F[X]} (hm : 1 ≤ f.natDegree)
    (hsq : ∀ R S : F[X][X], C f * R ^ 2 = f.map C * S ^ 2 → S = 0)
    (h64 : 64 ≤ Fintype.card F) (h8 : 8 * f.natDegree ≤ Fintype.card F) :
    |(#{xy : F × F | xy.2 ^ 2 = f.eval xy.1} : ℝ) - Fintype.card F| ≤
      7 * f.natDegree * √(Fintype.card F) := by
  have hrow (x : F) : ((∑ y : F, if y ^ 2 = f.eval x then 1 else 0 : ℕ) : ℤ) =
      quadraticChar F (f.eval x) + 1 := by
    rw [← quadraticChar_card_sqrts hF, Set.toFinset_ofPred, Finset.card_filter]
  have hcount : (#{xy : F × F | xy.2 ^ 2 = f.eval xy.1} : ℤ) =
      ∑ x, quadraticChar F (f.eval x) + Fintype.card F := by
    rw [Finset.card_filter, Fintype.sum_prod_type, Nat.cast_sum,
      Finset.sum_congr rfl fun x _ => hrow x, Finset.sum_add_distrib, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul, mul_one]
  have h := abs_sum_quadraticChar_le hF hm hsq h64 h8
  have hcountR : (#{xy : F × F | xy.2 ^ 2 = f.eval xy.1} : ℝ) =
      ((∑ x, quadraticChar F (f.eval x) : ℤ) : ℝ) + Fintype.card F := by exact_mod_cast hcount
  rwa [hcountR, add_sub_cancel_right]

section Examples

local instance fact_prime_67 : Fact (Nat.Prime 67) := ⟨by norm_num⟩

example := card_pow_eq_le_sqrt (F := ZMod 67) (f := X) (by simp)
  (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd (by simp)) (by rw [ZMod.card]; exact ⟨33, rfl⟩)
  (by simp) (by simp) (a := 1) one_ne_zero

example := abs_sum_quadraticChar_le (F := ZMod 67) (by rw [ZMod.ringChar_zmod_n]; decide)
  (f := X) (by simp) (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd (by simp)) (by simp) (by simp)

example := abs_card_sq_eq_sub_le (F := ZMod 67) (by rw [ZMod.ringChar_zmod_n]; decide)
  (f := X) (by simp) (fun _ _ => eq_zero_of_C_mul_sq_eq_of_odd (by simp)) (by simp) (by simp)

end Examples

end Transformer.Weil
