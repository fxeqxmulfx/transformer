module

public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Tactic.NormNum.Prime
public import Transformer.Weil.PowStepanov

/-!
# Stepanov's bound for `y^d = f(x)`, explicit

For `d c + 1 = q = |F|`, `f ∈ F[X]` of degree `m ≥ 1` prime to `d` and `a ≠ 0`, take
`ℓ = ⌊√q⌋`, `K = ⌊(q − (d − 1) m)/d⌋` and `J = ⌊ℓ/d⌋ + 2 m` in `PowStepanov`. Once `q ≥ 64`,
`8 d m ≤ q` and `d² ≤ q`, the conditions hold (`ℓ ≤ c + 1` as `d (ℓ − 1) ≤ ℓ (ℓ − 1) < q`), and
`d ℓ · #{x : f(x)^c = a} ≤ q ℓ + 3 d m q`. As `3 √q ≤ 4 ℓ`,
`#{x : f(x)^c = a} ≤ q/d + 4 m √q`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- **Stepanov's bound for `y^d = f(x)`**, explicit: for `d c + 1 = q = |F|`, `q ≥ 64`, `f` of
degree `m ≥ 1` prime to `d`, `8 d m ≤ q`, `d² ≤ q` and `a ≠ 0`,
`#{x : f(x)^c = a} ≤ q/d + 4 m √q`. -/
theorem card_pow_eq_le_of_coprime {f : F[X]} (hm : 1 ≤ f.natDegree) {d c : ℕ}
    (hq : d * c + 1 = Fintype.card F) (hdm : Nat.Coprime d f.natDegree)
    (h64 : 64 ≤ Fintype.card F) (h8 : 8 * d * f.natDegree ≤ Fintype.card F)
    (hd2 : d * d ≤ Fintype.card F) {a : F} (ha : a ≠ 0) :
    (#{x | f.eval x ^ c = a} : ℝ) ≤
      Fintype.card F / d + 4 * f.natDegree * √(Fintype.card F) := by
  have hd : 1 ≤ d := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, zero_mul, zero_add] at hq
    omega
  have hs2 := Nat.sqrt_le (Fintype.card F)
  have hs3 := Nat.lt_succ_sqrt (Fintype.card F)
  have hs8 : 8 ≤ (Fintype.card F).sqrt := Nat.le_sqrt.mpr (by omega)
  have hds : d ≤ (Fintype.card F).sqrt := Nat.le_sqrt.mpr hd2
  have hdm8 : 8 * (d * f.natDegree) ≤ Fintype.card F := by rw [← mul_assoc]; exact h8
  have hdm1 : (d - 1) * f.natDegree ≤ d * f.natDegree := Nat.mul_le_mul_right _ (Nat.sub_le d 1)
  have hdd : d ≤ d * f.natDegree := Nat.le_mul_of_pos_right d hm
  have hKq := Nat.mul_div_le (Fintype.card F - (d - 1) * f.natDegree) d
  have hsc : (Fintype.card F).sqrt ≤ c + 1 := by
    obtain ⟨t, ht⟩ : ∃ t, (Fintype.card F).sqrt = t + 1 := ⟨(Fintype.card F).sqrt - 1, by omega⟩
    rw [ht] at hs2 hds ⊢
    have h1 : d * t ≤ (t + 1) * t := Nat.mul_le_mul_right t hds
    have h2 : d * t ≤ d * c := by linarith
    have := Nat.le_of_mul_le_mul_left h2 (by omega : 0 < d)
    omega
  have hcore := mul_card_pow_eq_le_of_coprime hm hq hdm ha (ℓ := (Fintype.card F).sqrt)
    (J := (Fintype.card F).sqrt / d + 2 * f.natDegree)
    (K := (Fintype.card F - (d - 1) * f.natDegree) / d)
    (by rw [Nat.mul_sub_one]; omega) hsc (pow_stepanov_params hd hm h64 h8)
  have hK1 : 1 ≤ (Fintype.card F - (d - 1) * f.natDegree) / d :=
    (Nat.le_div_iff_mul_le (by omega)).mpr (by omega)
  set q := Fintype.card F
  set m := f.natDegree
  set N := #{x | f.eval x ^ c = a}
  obtain ⟨s, hs⟩ : ∃ s, q.sqrt = s := ⟨_, rfl⟩
  rw [hs] at hcore hs2 hs3 hs8 hds
  have hdJ := Nat.mul_div_le s d
  obtain ⟨J, hJ⟩ : ∃ J, s / d + 2 * m = J := ⟨_, rfl⟩
  obtain ⟨K, hK⟩ : ∃ K, (q - (d - 1) * m) / d = K := ⟨_, rfl⟩
  rw [hJ, hK] at hcore
  rw [hK] at hKq hK1
  have hJ1 : 1 ≤ J := by
    rw [← hJ]
    have := Nat.zero_le (s / d)
    omega
  have hcR : (s : ℝ) * N ≤ (K : ℝ) - 1 + (d - 1) * (c * m) + q * ((J : ℝ) - 1) := by
    have h := (Nat.cast_le (α := ℝ)).mpr hcore
    push_cast [Nat.cast_sub hK1, Nat.cast_sub hd, Nat.cast_sub hJ1] at h
    exact h
  have eK : (d : ℝ) * K + (d - 1) * m ≤ q := by
    have h := (Nat.cast_le (α := ℝ)).mpr (by omega : d * K + (d - 1) * m ≤ q)
    push_cast [Nat.cast_sub hd] at h
    exact h
  have eJ : (d : ℝ) * J ≤ s + 2 * (d * m) := by
    have h := (Nat.cast_le (α := ℝ)).mpr
      (by rw [← hJ, mul_add, mul_left_comm]; omega : d * J ≤ s + 2 * (d * m))
    push_cast at h
    exact h
  have ec : (d : ℝ) * c + 1 = q := by exact_mod_cast hq
  have hd1 : (1 : ℝ) ≤ d := by exact_mod_cast hd
  have hm1 : (1 : ℝ) ≤ m := by exact_mod_cast hm
  have hq0 : (0 : ℝ) ≤ q := Nat.cast_nonneg _
  set σ := √(q : ℝ)
  have hσ0 : 0 ≤ σ := Real.sqrt_nonneg _
  have hσ2 : σ * σ = q := Real.mul_self_sqrt (Nat.cast_nonneg _)
  have hσ8 : (8 : ℝ) ≤ σ := Real.le_sqrt_of_sq_le (by norm_num; exact_mod_cast h64)
  have hσs : σ < s + 1 := by
    rw [Real.sqrt_lt' (by positivity), sq]
    exact_mod_cast hs3
  have hcm : (d : ℝ) * ((d - 1) * (c * m)) = (d - 1) * m * (q - 1) := by rw [← ec]; ring
  have hσq : (d : ℝ) * m * σ * σ = d * m * q := by rw [mul_assoc _ σ σ, hσ2]
  have P1 := mul_le_mul_of_nonneg_left hcR (by positivity : (0 : ℝ) ≤ d)
  have P2 := mul_le_mul_of_nonneg_left eJ (by positivity : (0 : ℝ) ≤ q)
  have P3 := mul_nonneg (by linarith : (0 : ℝ) ≤ d - 1) (by positivity : (0 : ℝ) ≤ m)
  have P4 := mul_nonneg (by positivity : (0 : ℝ) ≤ q) (by linarith : (0 : ℝ) ≤ m - 1)
  have P5 := mul_nonneg (by positivity : (0 : ℝ) ≤ q) (by linarith : (0 : ℝ) ≤ d - 1)
  have P6 := mul_nonneg (mul_nonneg (by positivity : (0 : ℝ) ≤ d * m) hσ0)
    (by linarith : (0 : ℝ) ≤ 4 * s - 3 * σ)
  have key : (s : ℝ) * (d * N) ≤ s * (q + 4 * d * m * σ) := by linarith
  have hdN := le_of_mul_le_mul_left key (by exact_mod_cast (by omega : 0 < s))
  have hd0 : (0 : ℝ) < d := by linarith
  rw [div_add' _ _ _ hd0.ne', le_div_iff₀ hd0]
  linarith

section Examples

local instance fact_prime_71 : Fact (Nat.Prime 71) := ⟨by norm_num⟩

example := card_pow_eq_le_of_coprime (F := ZMod 71) (f := X) (by simp) (d := 5) (c := 14)
  (by rw [ZMod.card]) (by simp) (by simp) (by simp) (by simp) (a := 1) one_ne_zero

end Examples

end Transformer.Weil
