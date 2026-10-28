module

public import Transformer.Weil.LinearSystem

/-!
# Stepanov's bound for the fibers of `f^{(q − 1)/2}`

Let `q = |F|` be odd, `c = (q − 1)/2`, `f ∈ F[X]` of odd degree `m`, and `a ≠ 0`. For
parameters `ℓ, J, K` with `m + 2K ≤ q + 1`, `ℓ ≤ c + 1` and
`ℓ (K + J + (ℓ − 1)(m − 1)) < 2 J K`, Siegel's lemma gives `r_j, s_j` of degree `< K`, not all
`0`, with `T_k(r, s) = 0` for every `k < ℓ`. The auxiliary polynomial
`R = ∑_{j < J} (r_j + s_j f^c) X^{q j}` is then nonzero (independence) and, for `k < ℓ`,
`f^k E^{(k)} R` vanishes at every `x` with `f(x)^c = a`, where `f(x) ≠ 0`. So `R` vanishes to
order `ℓ` on that set, and `ℓ · #{x : f(x)^c = a} ≤ deg R ≤ K − 1 + c m + q (J − 1)`.

Reference: S. A. Stepanov, *On the number of points of a hyperelliptic curve over a finite prime
field* (1969); W. M. Schmidt, *Equations over Finite Fields*, Ch. I. The proof here is written
from the mathematics.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {F : Type*} [Field F] [Fintype F]

/-- The auxiliary polynomial `R = ∑_j (r_j + s_j f^c) X^{q j}` has degree at most
`K − 1 + c m + q (J − 1)` when the `r_j`, `s_j` lie in `F[X]_{< K}`. -/
theorem natDegree_auxiliary_le (f : F[X]) {J K : ℕ} (hK : 1 ≤ K)
    (v : (Fin J → degreeLT F K) × (Fin J → degreeLT F K)) :
    (∑ j, ((v.1 j : F[X]) + (v.2 j : F[X]) * f ^ (Fintype.card F / 2)) *
      X ^ (Fintype.card F * (j : ℕ))).natDegree ≤
      K - 1 + Fintype.card F / 2 * f.natDegree + Fintype.card F * (J - 1) := by
  refine natDegree_sum_le_of_forall_le _ _ fun j _ => natDegree_mul_le.trans
    (add_le_add ((natDegree_add_le _ _).trans (max_le ?_ ?_)) ?_)
  · have := natDegree_lt_of_mem_degreeLT hK (v.1 j).2
    omega
  · refine natDegree_mul_le.trans ?_
    have := natDegree_lt_of_mem_degreeLT hK (v.2 j).2
    have h2 : (f ^ (Fintype.card F / 2)).natDegree ≤ Fintype.card F / 2 * f.natDegree :=
      natDegree_pow_le
    omega
  · exact (natDegree_X_pow_le _).trans (Nat.mul_le_mul_left _ (by have := j.2; omega))

example := natDegree_auxiliary_le (F := ZMod 3) X (J := 1) le_rfl 0

variable [DecidableEq F]

/-- **Stepanov's bound**: for `q = |F|` odd, `f` of odd degree `m`, `a ≠ 0` and parameters with
`m + 2K ≤ q + 1`, `ℓ ≤ (q − 1)/2 + 1` and `ℓ (K + J + (ℓ − 1)(m − 1)) < 2 J K`,
`ℓ · #{x : f(x)^{(q−1)/2} = a} ≤ K − 1 + (q − 1)/2 · m + q (J − 1)`. -/
theorem mul_card_pow_eq_le {f : F[X]} (hf : Odd f.natDegree) (hq : Odd (Fintype.card F))
    {a : F} (ha : a ≠ 0) {ℓ J K : ℕ} (hK : f.natDegree + 2 * K ≤ Fintype.card F + 1)
    (hℓ : ℓ ≤ Fintype.card F / 2 + 1)
    (hD : ℓ * (K + J + (ℓ - 1) * (f.natDegree - 1)) < 2 * (J * K)) :
    ℓ * #{x | f.eval x ^ (Fintype.card F / 2) = a} ≤
      K - 1 + Fintype.card F / 2 * f.natDegree + Fintype.card F * (J - 1) := by
  have hf0 : f ≠ 0 := fun h0 => by simp [h0] at hf
  have hm : 1 ≤ f.natDegree := hf.pos
  have hq2 : 1 < Fintype.card F := Fintype.one_lt_card
  have hJ : 1 ≤ J := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hD
  have hK1 : 1 ≤ K := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hD
  obtain ⟨G, hG⟩ := exists_hasseDeriv_pow_eq hf0 (Fintype.card F / 2)
  obtain ⟨v, hv0, hv⟩ := exists_ne_zero_forall_eq_zero
    (fun k : Fin ℓ => stepanovMap f G a J K k)
    (fun k v => natDegree_stepanovMap_lt hm (fun l => (hG l).2) a hK1 k.2 v)
    (by rw [finrank_pairs]; exact hD)
  set R := ∑ j, ((v.1 j : F[X]) + (v.2 j : F[X]) * f ^ (Fintype.card F / 2)) *
    X ^ (Fintype.card F * (j : ℕ)) with hRdef
  have hR : R ≠ 0 := by
    intro hR0
    obtain ⟨h1, h2⟩ := eq_zero_of_sum_add_mul_pow_eq_zero hf hq hK
      (r := fun j => (v.1 j : F[X])) (s := fun j => (v.2 j : F[X]))
      (fun j => natDegree_lt_of_mem_degreeLT hK1 (v.1 j).2)
      (fun j => natDegree_lt_of_mem_degreeLT hK1 (v.2 j).2) hR0
    exact hv0 (Prod.ext (funext fun j => Subtype.ext (congrFun h1 j))
      (funext fun j => Subtype.ext (congrFun h2 j)))
  have hvan : ∀ x ∈ ({x | f.eval x ^ (Fintype.card F / 2) = a} : Finset F), ∀ k < ℓ,
      (hasseDeriv k R).eval x = 0 := by
    intro x hx k hk
    have hx' : f.eval x ^ (Fintype.card F / 2) = a := (Finset.mem_filter.mp hx).2
    have hfx : f.eval x ≠ 0 := fun h0 => ha (by rw [← hx', h0, zero_pow (by omega)])
    have h := eval_pow_mul_hasseDeriv_sum (by omega : k < Fintype.card F)
      (by omega : k ≤ Fintype.card F / 2) (fun l => (hG l).1) (fun j => (v.1 j : F[X]))
      (fun j => (v.2 j : F[X])) hx'
    have hvk : stepanovMap f G a J K k v = 0 := hv ⟨k, hk⟩
    rw [← stepanovMap_apply, hvk, eval_zero, eval_mul, eval_pow] at h
    exact (mul_eq_zero.mp h).resolve_left (pow_ne_zero _ hfx)
  exact (mul_card_le_natDegree hR _ hvan).trans (natDegree_auxiliary_le f hK1 v)

example := mul_card_pow_eq_le (F := ZMod 3) (f := X) (by simp)
  (by rw [ZMod.card]; exact ⟨1, rfl⟩) (a := 1) one_ne_zero (ℓ := 1) (J := 2) (K := 1)
  (by simp) (by simp) (by norm_num)

end Transformer.Weil
