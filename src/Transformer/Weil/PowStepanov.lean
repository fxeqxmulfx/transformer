module

public import Transformer.Weil.PowIndependence
public import Transformer.Weil.PowSystem

/-!
# Stepanov's bound for `y^d = f(x)`

Let `F` have `q` elements, `d c + 1 = q`, and `f ∈ F[X]` of degree `m ≥ 1` with `d` prime to `m`.
For `a ≠ 0` and parameters with `ℓ ≤ c + 1`, `d (K − 1) + (d − 1) m < q` and
`ℓ (K + J + (ℓ − 1)(m − 1)) < d J K`, Siegel's lemma gives `r_{ij} ∈ F[X]_{< K}`, `i < d`,
`j < J`, not all `0`, with `T_k(r) = 0` for every `k < ℓ` (`PowSystem`). Then
`R = ∑ r_{ij} f^{c i} X^{q j}` is nonzero (`PowIndependence`) and vanishes to order `ℓ` on
`{x : f(x)^c = a}`, so `ℓ · #{x : f(x)^c = a} ≤ deg R ≤ K − 1 + (d − 1) c m + q (J − 1)`.
The dimension count holds for `ℓ = ⌊√q⌋`, `K = ⌊(q − (d − 1) m)/d⌋` and `J = ⌊ℓ/d⌋ + 2 m` once
`q ≥ 64` and `8 d m ≤ q` (`pow_stepanov_params`).

Reference: W. M. Schmidt, *Equations over Finite Fields*, Ch. I. The proofs here are written from
the mathematics.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

/-- Stepanov's dimension count for `y^d = f(x)`: for `ℓ = ⌊√q⌋`, `K = ⌊(q − (d − 1) m)/d⌋`,
`J = ⌊ℓ/d⌋ + 2 m`, `d, m ≥ 1`, `q ≥ 64` and `8 d m ≤ q`, `ℓ (K + J + (ℓ − 1)(m − 1)) < d J K`.
Times `d`, with `X = d J ≥ ℓ + A`, `A = 2 d m − d + 1`, and `Y = d K ≥ q − (d − 1) m − d + 1`, it
reads `(X − ℓ)(Y − ℓ) > ℓ² + d ℓ (ℓ − 1)(m − 1)`, with room `d m q / 4`. -/
theorem pow_stepanov_params {q d m : ℕ} (hd : 1 ≤ d) (hm : 1 ≤ m) (h64 : 64 ≤ q)
    (h8 : 8 * d * m ≤ q) :
    q.sqrt * ((q - (d - 1) * m) / d + (q.sqrt / d + 2 * m) + (q.sqrt - 1) * (m - 1)) <
      d * (q.sqrt / d + 2 * m) * ((q - (d - 1) * m) / d) := by
  have hs2 : q.sqrt * q.sqrt ≤ q := Nat.sqrt_le q
  have hs8 : 8 ≤ q.sqrt := Nat.le_sqrt.mpr (by omega)
  obtain ⟨s, hs⟩ : ∃ s, q.sqrt = s := ⟨_, rfl⟩
  rw [hs] at hs2 hs8 ⊢
  have h8s : 8 * s ≤ q := (Nat.mul_le_mul_right _ hs8).trans hs2
  have hdm1 : (d - 1) * m ≤ d * m := Nat.mul_le_mul_right _ (Nat.sub_le d 1)
  have hdm8 : 8 * (d * m) ≤ q := by rw [← mul_assoc]; exact h8
  have hst := Nat.div_add_mod s d
  have hsu := Nat.mod_lt s (by omega : 0 < d)
  have hKt := Nat.div_add_mod (q - (d - 1) * m) d
  have hKu := Nat.mod_lt (q - (d - 1) * m) (by omega : 0 < d)
  obtain ⟨t, ht⟩ : ∃ t, s / d = t := ⟨_, rfl⟩
  obtain ⟨K, hK⟩ : ∃ K, (q - (d - 1) * m) / d = K := ⟨_, rfl⟩
  obtain ⟨u, hu⟩ : ∃ u, s % d = u := ⟨_, rfl⟩
  obtain ⟨w, hw⟩ : ∃ w, (q - (d - 1) * m) % d = w := ⟨_, rfl⟩
  rw [ht, hu] at hst
  rw [hu] at hsu
  rw [hK, hw] at hKt
  rw [hw] at hKu
  rw [ht, hK]
  have est : (d : ℤ) * t + u = s := by exact_mod_cast hst
  have eKt : (d : ℤ) * K + w = q - (d - 1) * m := by
    have h := (Nat.cast_inj (R := ℤ)).mpr hKt
    push_cast [Nat.cast_sub (hdm1.trans (by omega) : (d - 1) * m ≤ q), Nat.cast_sub hd] at h
    exact h
  have ed : (d : ℤ) ≤ d * m := by exact_mod_cast Nat.le_mul_of_pos_right d hm
  have em : (m : ℤ) ≤ d * m := by exact_mod_cast Nat.le_mul_of_pos_left m hd
  have eq8 : 8 * ((d : ℤ) * m) ≤ q := by exact_mod_cast hdm8
  have es8 : 8 * (s : ℤ) ≤ q := by exact_mod_cast h8s
  have es2 : (s : ℤ) * s ≤ q := by exact_mod_cast hs2
  have ed1 : (1 : ℤ) ≤ d := by exact_mod_cast hd
  have em1 : (1 : ℤ) ≤ m := by exact_mod_cast hm
  have eu : (u : ℤ) + 1 ≤ d := by exact_mod_cast (hsu : u + 1 ≤ d)
  have ew : (w : ℤ) + 1 ≤ d := by exact_mod_cast (hKu : w + 1 ≤ d)
  have F1 := mul_nonneg (by linarith : (0 : ℤ) ≤ d * t - s + d - 1)
    (by linarith : (0 : ℤ) ≤ d * K - s)
  have F2 := mul_nonneg (by linarith : (0 : ℤ) ≤ 2 * d * m - d + 1)
    (by linarith : (0 : ℤ) ≤ d * K - (q - (d - 1) * m - d + 1))
  have F3 := mul_nonneg (by linarith : (0 : ℤ) ≤ q - s * s) (by linarith : (0 : ℤ) ≤ 1 + d * (m - 1))
  have F4 := mul_nonneg (mul_nonneg (by positivity : (0 : ℤ) ≤ d) (by positivity : (0 : ℤ) ≤ s))
    (by linarith : (0 : ℤ) ≤ m - 1)
  have F5 := mul_nonneg (by positivity : (0 : ℤ) ≤ d * m) (by linarith : (0 : ℤ) ≤ q - 8 * (d * m))
  have F6 := mul_nonneg (by positivity : (0 : ℤ) ≤ d * m) (by linarith : (0 : ℤ) ≤ q - 8 * s)
  have F7 := mul_nonneg (by linarith : (0 : ℤ) ≤ d - 1) (by linarith : (0 : ℤ) ≤ d * m - m + d - 1)
  have F8 := mul_nonneg (by positivity : (0 : ℤ) ≤ d * m) (by linarith : (0 : ℤ) ≤ d * m + m - d + 1)
  have F9 := mul_nonneg (by linarith : (0 : ℤ) ≤ d - 1) (by positivity : (0 : ℤ) ≤ s)
  have F0 := mul_pos (by linarith : (0 : ℤ) < d * m) (by linarith : (0 : ℤ) < q)
  have hlt : d * (s * (K + (t + 2 * m) + (s - 1) * (m - 1))) < d * (d * (t + 2 * m) * K) := by
    zify [hm, (by omega : 1 ≤ s)]
    linarith
  exact Nat.lt_of_mul_lt_mul_left hlt

example := pow_stepanov_params (q := 64) (d := 1) (m := 1) le_rfl le_rfl le_rfl (by norm_num)

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F]

/-- **Stepanov's bound for `y^d = f(x)`**: for `d c + 1 = q = |F|`, `f` of degree `m ≥ 1` prime
to `d`, `a ≠ 0` and parameters with `d (K − 1) + (d − 1) m < q`, `ℓ ≤ c + 1` and
`ℓ (K + J + (ℓ − 1)(m − 1)) < d J K`,
`ℓ · #{x : f(x)^c = a} ≤ K − 1 + (d − 1) c m + q (J − 1)`. -/
theorem mul_card_pow_eq_le_of_coprime {f : F[X]} (hm : 1 ≤ f.natDegree) {d c : ℕ}
    (hq : d * c + 1 = Fintype.card F) (hdm : Nat.Coprime d f.natDegree) {a : F} (ha : a ≠ 0)
    {ℓ J K : ℕ} (hK : d * (K - 1) + (d - 1) * f.natDegree < Fintype.card F) (hℓ : ℓ ≤ c + 1)
    (hD : ℓ * (K + J + (ℓ - 1) * (f.natDegree - 1)) < d * J * K) :
    ℓ * #{x | f.eval x ^ c = a} ≤
      K - 1 + (d - 1) * (c * f.natDegree) + Fintype.card F * (J - 1) := by
  have hf0 : f ≠ 0 := ne_zero_of_natDegree_gt hm
  have hq2 : 1 < Fintype.card F := Fintype.one_lt_card
  have hd : 0 < d := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, zero_mul, zero_add] at hq
    omega
  have hc0 : c ≠ 0 := fun h0 => by
    rw [h0, mul_zero, zero_add] at hq
    omega
  have hcd : c ≤ d * c := Nat.le_mul_of_pos_left c hd
  have hK1 : 1 ≤ K := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hD
  choose G hG using fun i : ℕ => exists_hasseDeriv_pow_eq hf0 (c * i)
  obtain ⟨v, hv0, hv⟩ := exists_ne_zero_forall_eq_zero
    (fun k : Fin ℓ => powMap f G a d J K k)
    (fun k v => natDegree_powMap_lt hm (fun i l => (hG i l).2) a hK1 k.2 v)
    (by rw [finrank_traceDomain]; exact hD)
  set R := ∑ ij, (v ij : F[X]) * (f ^ c) ^ (ij.1 : ℕ) * X ^ (Fintype.card F * (ij.2 : ℕ))
    with hRdef
  have hR : R ≠ 0 := by
    intro hR0
    have h0 := eq_zero_of_sum_mul_pow_pow_mul_X_pow_eq_zero hq hdm hm hK
      (r := fun i j => (v (i, j) : F[X]))
      (fun i j => natDegree_lt_of_mem_degreeLT hK1 (v (i, j)).2)
      (by rw [← hR0, hRdef, Fintype.sum_prod_type])
    exact hv0 (funext fun ij => Subtype.ext (congrFun (congrFun h0 ij.1) ij.2))
  have hvan : ∀ x ∈ ({x | f.eval x ^ c = a} : Finset F), ∀ k < ℓ,
      (hasseDeriv k R).eval x = 0 := by
    intro x hx k hk
    have hx' : f.eval x ^ c = a := (mem_filter.mp hx).2
    have hfx : f.eval x ≠ 0 := fun h0 => ha (by rw [← hx', h0, zero_pow hc0])
    have h := eval_pow_mul_hasseDeriv_powAux (by omega : k < Fintype.card F) (by omega : k ≤ c)
      (fun i l => (hG i l).1) (fun ij => (v ij : F[X])) hx'
    have hvk : powMap f G a d J K k v = 0 := hv ⟨k, hk⟩
    rw [← powMap_apply, hvk, eval_zero, eval_mul, eval_pow] at h
    exact (mul_eq_zero.mp h).resolve_left (pow_ne_zero _ hfx)
  have hdeg := natDegree_traceAux_le (f ^ c) (Fintype.card F) hK1 v
  rw [natDegree_pow] at hdeg
  exact (mul_card_le_natDegree hR _ hvan).trans hdeg

example := mul_card_pow_eq_le_of_coprime (F := ZMod 3) (f := X) (by simp) (d := 2) (c := 1)
  (by rw [ZMod.card]) (by rw [natDegree_X]; exact Nat.coprime_one_right 2) (a := 1) one_ne_zero
  (ℓ := 1) (J := 2) (K := 1) (by rw [natDegree_X, ZMod.card]; norm_num) (by norm_num)
  (by rw [natDegree_X]; norm_num)

end Transformer.Weil
