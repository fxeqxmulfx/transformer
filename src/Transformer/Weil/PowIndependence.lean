module

public import Transformer.Weil.TraceIndependence

/-!
# Stepanov's independence lemma for `y^d = f(x)`

Let `f` have degree `m ≥ 1`, `d` prime to `m` and `d c + 1 = q`. If
`∑_{i < d} ∑_{j < J} r_{ij} f^{c i} X^{q j} = 0` with `deg r_{ij} < K` and
`d (K − 1) + (d − 1) m < q`, then every `r_{ij}` is `0`. A nonzero `h_i = ∑_j r_{ij} X^{q j}`
has degree `N = q a + e` with `e < K`, and
`d (N + i c m) + (d − 1) m = q (d a + i m) + d e + (d − 1 − i) m` with
`d e + (d − 1 − i) m < q`. So equal degrees of `h_i f^{c i}` and `h_{i'} f^{c i'}` give
`d e + (d − 1 − i) m = d e' + (d − 1 − i') m`, hence `(d − 1 − i) m ≡ (d − 1 − i') m` modulo `d`,
and `i = i'` as `d` is prime to `m`. The nonzero terms have distinct degrees
(`eq_zero_of_sum_mul_pow_mul_X_pow_eq_zero_of_inj`). For `d = 2` and odd `m` this is the case
of `Independence`; here no second variable is needed.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- `d (N + i c m) + (d − 1) m ≡ d (N mod q) + (d − 1 − i) m` modulo `q`, and the right side is
below `q`, for `d c + 1 = q`, `i < d`, `N mod q < K` and `d (K − 1) + (d − 1) m < q`. -/
theorem mul_add_mul_mod_eq {d c q m K N i : ℕ} (hq : d * c + 1 = q) (hi : i < d)
    (hK : d * (K - 1) + (d - 1) * m < q) (hN : N % q < K) :
    (d * (N + i * (c * m)) + (d - 1) * m) % q = d * (N % q) + (d - 1 - i) * m := by
  obtain ⟨t, rfl⟩ : ∃ t, d = i + t + 1 := ⟨d - 1 - i, by omega⟩
  rw [show i + t + 1 - 1 = i + t by omega] at *
  rw [show i + t - i = t by omega]
  have hlt : (i + t + 1) * (N % q) + t * m < q := by
    have h1 : (i + t + 1) * (N % q) ≤ (i + t + 1) * (K - 1) :=
      Nat.mul_le_mul_left _ (by omega)
    have h2 : t * m ≤ (i + t) * m := Nat.mul_le_mul_right _ (by omega)
    omega
  have e : (i + t + 1) * (N + i * (c * m)) + (i + t) * m =
      q * ((i + t + 1) * (N / q) + i * m) + ((i + t + 1) * (N % q) + t * m) := by
    conv_lhs => rw [← Nat.div_add_mod N q]
    subst hq
    ring
  rw [e, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]

example := mul_add_mul_mod_eq (d := 2) (c := 1) (q := 3) (m := 1) (K := 1) (N := 0) (i := 1)
  rfl (by norm_num) (by norm_num) (by norm_num)

variable {F : Type*} [Field F]

/-- **Independence for `f^c`**: if `d c + 1 = q`, `d` is prime to `m = deg f ≥ 1` and
`∑_{i < d} ∑_{j < J} r_{ij} f^{c i} X^{q j} = 0` with `deg r_{ij} < K` and
`d (K − 1) + (d − 1) m < q`, then `r = 0`. -/
theorem eq_zero_of_sum_mul_pow_pow_mul_X_pow_eq_zero {f : F[X]} {d c q J K : ℕ}
    (hq : d * c + 1 = q) (hdm : Nat.Coprime d f.natDegree) (hm : 1 ≤ f.natDegree)
    (hK : d * (K - 1) + (d - 1) * f.natDegree < q) {r : Fin d → Fin J → F[X]}
    (hr : ∀ i j, (r i j).natDegree < K)
    (h : ∑ i, ∑ j, r i j * (f ^ c) ^ (i : ℕ) * X ^ (q * (j : ℕ)) = 0) : r = 0 := by
  rcases Nat.eq_zero_or_pos d with hd0 | hd
  · subst hd0
    exact funext fun i => i.elim0
  have hf0 : f ≠ 0 := ne_zero_of_natDegree_gt hm
  have hKq : K ≤ q := by
    have := Nat.le_mul_of_pos_left (K - 1) hd
    omega
  refine eq_zero_of_sum_mul_pow_mul_X_pow_eq_zero_of_inj (pow_ne_zero c hf0) hKq hr
    (fun i i' N N' hN hN' heq => ?_) h
  rw [natDegree_pow] at heq
  have h1 := mul_add_mul_mod_eq hq i.2 hK hN
  rw [heq, mul_add_mul_mod_eq hq i'.2 hK hN'] at h1
  have hmod : (d - 1 - i) * f.natDegree ≡ (d - 1 - i') * f.natDegree [MOD d] := by
    have h2 := congrArg (· % d) h1
    simp only [Nat.mul_add_mod] at h2
    exact h2.symm
  have h3 := Nat.ModEq.eq_of_lt_of_lt (Nat.ModEq.cancel_right_of_coprime hdm hmod)
    (by omega) (by omega)
  exact Fin.ext (by omega)

example := eq_zero_of_sum_mul_pow_pow_mul_X_pow_eq_zero (F := ZMod 3) (f := X) (d := 2) (c := 1)
  (q := 3) (J := 1) (K := 1) rfl (by rw [natDegree_X]; norm_num) (by rw [natDegree_X])
  (by rw [natDegree_X]; norm_num) (r := 0) (fun _ _ => by simp) (by simp)

end Transformer.Weil
