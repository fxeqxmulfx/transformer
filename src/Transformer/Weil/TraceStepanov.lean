module

public import Transformer.Weil.TraceIndependence
public import Transformer.Weil.TraceSystem

/-!
# Stepanov's bound for the level sets of the trace polynomial

Let `F` have `q = p^ν` elements, `f ∈ F[X]` of degree `m ≥ 1` prime to `p`, and
`g = T_ν(f)`, of degree `m p^{ν − 1}`. For parameters `ℓ ≤ q`, `K ≤ p^{ν − 1}` and `J` with
`ℓ (K + J + (m − 1)(ℓ − 1)) < p J K`, Siegel's lemma gives `r_{ij} ∈ F[X]_{< K}`, `i < p`,
`j < J`, not all `0`, with `T_k(r) = 0` for every `k < ℓ`. Then `R = ∑ r_{ij} g^i X^{q j}` is
nonzero (independence) and vanishes to order `ℓ` on `{g = c}`, so
`ℓ · #{x : g(x) = c} ≤ deg R ≤ K − 1 + (p − 1) m p^{ν − 1} + q (J − 1)`.

Reference: W. M. Schmidt, *Equations over Finite Fields*, Ch. I (the curve `y^p − y = f(x)`).
The proof here is written from the mathematics.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {p : ℕ} [Fact p.Prime] [CharP F p]

/-- **Stepanov's bound for `T_ν(f) = c`**: for `|F| = p^ν`, `f` of degree `m ≥ 1` prime to `p`,
`ℓ ≤ |F|`, `K ≤ p^{ν − 1}` and `ℓ (K + J + (m − 1)(ℓ − 1)) < p J K`,
`ℓ · #{x : T_ν(f)(x) = c} ≤ K − 1 + (p − 1) m p^{ν − 1} + |F| (J − 1)`. -/
theorem mul_card_tracePoly_eq_le {ν : ℕ} (hcard : Fintype.card F = p ^ ν) {f : F[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) (c : F) {ℓ J K : ℕ}
    (hℓ : ℓ ≤ Fintype.card F) (hK : K ≤ p ^ (ν - 1))
    (hD : ℓ * (K + J + (f.natDegree - 1) * (ℓ - 1)) < p * J * K) :
    ℓ * #{x | (tracePoly p ν f).eval x = c} ≤
      K - 1 + (p - 1) * (f.natDegree * p ^ (ν - 1)) + Fintype.card F * (J - 1) := by
  have hp := (Fact.out : p.Prime)
  have hν : 1 ≤ ν := Nat.pos_of_ne_zero fun h0 => by
    rw [h0, pow_zero] at hcard
    exact absurd hcard (Fintype.one_lt_card (α := F)).ne'
  have hK1 : 1 ≤ K := Nat.pos_of_ne_zero fun h0 => by simp [h0] at hD
  set g := tracePoly p ν f with hgdef
  have hgdeg : g.natDegree = f.natDegree * p ^ (ν - 1) := natDegree_tracePoly hp.two_le hν hm
  choose V hVd hV using fun b i =>
    exists_hasseDeriv_pow_eq_on (fun b hb => natDegree_hasseDeriv_tracePoly_le p ν f hb) c i b
  obtain ⟨v, hv0, hv⟩ := exists_ne_zero_forall_eq_zero
    (fun k : Fin ℓ => traceMap V p J K k)
    (fun k v => natDegree_traceMap_lt hVd hK1 k.2 v)
    (by rw [finrank_traceDomain]; exact hD)
  set R := ∑ ij, (v ij : F[X]) * g ^ (ij.1 : ℕ) * X ^ (Fintype.card F * (ij.2 : ℕ)) with hRdef
  have hR : R ≠ 0 := by
    intro hR0
    have hq : Fintype.card F = p * p ^ (ν - 1) := by
      rw [hcard, ← pow_succ']
      congr 1
      omega
    have h0 := eq_zero_of_sum_mul_pow_mul_X_pow_eq_zero
      ((Nat.Prime.coprime_iff_not_dvd hp).mpr hpm) hm hgdeg hK
      (r := fun i j => (v (i, j) : F[X]))
      (fun i j => natDegree_lt_of_mem_degreeLT hK1 (v (i, j)).2)
      (by rw [← hR0, hRdef, Fintype.sum_prod_type, hq])
    exact hv0 (funext fun ij => Subtype.ext (congrFun (congrFun h0 ij.1) ij.2))
  have hvan : ∀ x ∈ ({x | g.eval x = c} : Finset F), ∀ k < ℓ, (hasseDeriv k R).eval x = 0 := by
    intro x hx k hk
    have h0 : traceMap V p J K k v = 0 := hv ⟨k, hk⟩
    rw [traceMap_apply] at h0
    rw [hRdef, eval_hasseDeriv_traceAux hV (hk.trans_le hℓ) (fun ij => (v ij : F[X]))
      (mem_filter.mp hx).2, h0, eval_zero]
  have hdeg := natDegree_traceAux_le g (Fintype.card F) hK1 v
  rw [hgdeg] at hdeg
  exact (mul_card_le_natDegree hR _ hvan).trans hdeg

example := mul_card_tracePoly_eq_le (F := ZMod 2) (p := 2) (ν := 1)
  (by rw [ZMod.card]; norm_num) (f := X) (by rw [natDegree_X]) (by rw [natDegree_X]; norm_num) 0
  (ℓ := 1) (J := 2) (K := 1) (by rw [ZMod.card]; norm_num) (by norm_num) (by norm_num)

end Transformer.Weil
