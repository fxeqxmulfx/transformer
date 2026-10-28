module

public import Transformer.Weil.TraceSystem

/-!
# Stepanov's linear system for `y^d = f(x)`

Let `f ∈ F[X]` have degree `m ≥ 1`, and for each `i` let `E^{(l)}(f^{c i}) = f^{c i − l} G_{i,l}`
with `deg G_{i,l} ≤ l (m − 1)` (`exists_hasseDeriv_pow_eq`). For `k ≤ c`,
`f^k E^{(k)}(s f^{c i}) = f^{c i} σ_k^{(i)}(s)` for every `i`: for `i ≥ 1` as `k ≤ c i`
(`Auxiliary`), for `i = 0` as `G_{0,l} = E^{(l)}(1)` is `0` for `l ≥ 1`. For `k < q = |F|` the
factors `X^{q j}` pass through `E^{(k)}`, so at a point `x` with `f(x)^c = a`, where `x^q = x`,
`f^k E^{(k)}` of `R = ∑_{i,j} r_{ij} f^{c i} X^{q j}` takes the value of
`T_k(r) = ∑_{i,j} a^i σ_k^{(i)}(r_{ij}) X^j`. On `r_{ij} ∈ F[X]_{< K}`, `i < d`, `j < J`, this
is a linear map of degree `< K + J + (ℓ − 1)(m − 1)` for `k < ℓ`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Module Finset

variable {F : Type*} [Field F]

/-- `f^k E^{(k)}(s f^c) = f^c σ_k(s)` for `k ≤ c`, and also for `c = 0`, where
`G_l = E^{(l)}(1)`. -/
theorem pow_mul_hasseDeriv_mul_pow_of_le_or_eq_zero {f : F[X]} {c k : ℕ} (hk : k ≤ c ∨ c = 0)
    {G : ℕ → F[X]} (hG : ∀ l, hasseDeriv l (f ^ c) = f ^ (c - l) * G l) (s : F[X]) :
    f ^ k * hasseDeriv k (s * f ^ c) = f ^ c * sigma f G k s := by
  rcases hk with hk | rfl
  · exact pow_mul_hasseDeriv_mul_pow hk hG s
  have hG0 (l : ℕ) (hl : 0 < l) : G l = 0 := by
    have h := hG l
    rw [pow_zero, Nat.zero_sub, pow_zero, one_mul, ← C_1, hasseDeriv_C _ _ hl] at h
    exact h.symm
  have hG1 : G 0 = 1 := by
    have h := hG 0
    rw [pow_zero, hasseDeriv_zero, LinearMap.id_apply, one_mul] at h
    exact h.symm
  rw [pow_zero, mul_one, one_mul, sigma_apply, Finset.sum_eq_single k]
  · rw [Nat.sub_self, hG1, mul_one]
  · intro b hb hbk
    rw [hG0 (k - b) (by have := mem_range.mp hb; omega), mul_zero, zero_mul]
  · intro hk
    exact absurd (mem_range.mpr (Nat.lt_succ_self k)) hk

example := pow_mul_hasseDeriv_mul_pow_of_le_or_eq_zero (F := ZMod 3) (f := X) (c := 0) (k := 1)
  (Or.inr rfl) (G := (exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 0).choose)
  (fun l => ((exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero 0).choose_spec l).1) X

/-- Stepanov's linear system for `y^d = f(x)`: `T_k(r) = ∑_{i,j} a^i σ_k^{(i)}(r_{ij}) X^j`. -/
noncomputable def powMap (f : F[X]) (G : ℕ → ℕ → F[X]) (a : F) (d J K k : ℕ) :
    (Fin d × Fin J → degreeLT F K) →ₗ[F] F[X] :=
  ∑ ij : Fin d × Fin J, a ^ (ij.1 : ℕ) • (LinearMap.mulRight F (X ^ (ij.2 : ℕ)) ∘ₗ
    sigma f (G ij.1) k ∘ₗ (degreeLT F K).subtype ∘ₗ LinearMap.proj ij)

theorem powMap_apply (f : F[X]) (G : ℕ → ℕ → F[X]) (a : F) (d J K k : ℕ)
    (v : Fin d × Fin J → degreeLT F K) :
    powMap f G a d J K k v =
      ∑ ij, a ^ (ij.1 : ℕ) • (sigma f (G ij.1) k (v ij : F[X]) * X ^ (ij.2 : ℕ)) := by
  simp [powMap]

/-- `deg T_k < K + J + (ℓ − 1)(m − 1)` for `k < ℓ`. -/
theorem natDegree_powMap_lt {f : F[X]} (hf : 1 ≤ f.natDegree) {G : ℕ → ℕ → F[X]}
    (hG : ∀ i l, (G i l).natDegree ≤ l * (f.natDegree - 1)) (a : F) {d J K ℓ k : ℕ}
    (hK : 1 ≤ K) (hk : k < ℓ) (v : Fin d × Fin J → degreeLT F K) :
    (powMap f G a d J K k v).natDegree < K + J + (ℓ - 1) * (f.natDegree - 1) := by
  rw [powMap_apply]
  refine lt_of_le_of_lt (natDegree_sum_le_of_forall_le _ _
    (n := K - 1 + (ℓ - 1) * (f.natDegree - 1) + (J - 1)) fun ij _ => ?_) (by omega)
  have h1 := natDegree_sigma_le hf (hG ij.1) k (v ij : F[X])
  have h2 : (v ij : F[X]).natDegree ≤ K - 1 :=
    Nat.le_sub_one_of_lt (natDegree_lt_of_mem_degreeLT hK (v ij).2)
  have h3 : k * (f.natDegree - 1) ≤ (ℓ - 1) * (f.natDegree - 1) :=
    Nat.mul_le_mul_right _ (by omega)
  have h4 : (X ^ (ij.2 : ℕ) : F[X]).natDegree ≤ J - 1 :=
    (natDegree_X_pow_le _).trans (by have := ij.2.2; omega)
  exact (natDegree_smul_le _ _).trans (natDegree_mul_le.trans (by omega))

example := natDegree_powMap_lt (F := ZMod 3) (f := X) (by simp) (G := fun _ _ => 0)
  (fun _ _ => by simp) 1 (d := 1) (J := 1) (K := 1) (ℓ := 1) le_rfl (k := 0) one_pos 0

variable [Fintype F]

/-- At `x` with `f(x)^c = a`, for `k < q` and `k ≤ c`,
`f^k E^{(k)}(∑_{i,j} r_{ij} f^{c i} X^{q j})` takes the value of `T_k(r)`. -/
theorem eval_pow_mul_hasseDeriv_powAux {f : F[X]} {c k : ℕ} (hkq : k < Fintype.card F)
    (hkc : k ≤ c) {G : ℕ → ℕ → F[X]}
    (hG : ∀ i l, hasseDeriv l (f ^ (c * i)) = f ^ (c * i - l) * G i l) {d J : ℕ}
    (r : Fin d × Fin J → F[X]) {a x : F} (hx : f.eval x ^ c = a) :
    (f ^ k * hasseDeriv k (∑ ij, r ij * (f ^ c) ^ (ij.1 : ℕ) *
      X ^ (Fintype.card F * (ij.2 : ℕ)))).eval x =
      (∑ ij : Fin d × Fin J, a ^ (ij.1 : ℕ) • (sigma f (G ij.1) k (r ij) * X ^ (ij.2 : ℕ))).eval
        x := by
  rw [map_sum, mul_sum, eval_finsetSum, eval_finsetSum]
  refine sum_congr rfl fun ij _ => ?_
  have hk : k ≤ c * ij.1 ∨ c * ij.1 = 0 := by
    rcases Nat.eq_zero_or_pos (ij.1 : ℕ) with h0 | hpos
    · exact Or.inr (by rw [h0, mul_zero])
    · exact Or.inl (hkc.trans (Nat.le_mul_of_pos_right c hpos))
  rw [hasseDeriv_mul_X_pow_card_mul _ _ hkq, ← pow_mul, ← mul_assoc,
    pow_mul_hasseDeriv_mul_pow_of_le_or_eq_zero hk (hG ij.1) (r ij)]
  simp only [eval_mul, eval_pow, eval_smul, eval_X, smul_eq_mul, pow_mul, hx,
    FiniteField.pow_card]
  ring

example := eval_pow_mul_hasseDeriv_powAux (F := ZMod 3) (f := X) (c := 1) (k := 1) (by simp)
  le_rfl (G := fun i => (exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero (1 * i)).choose)
  (fun i l => ((exists_hasseDeriv_pow_eq (F := ZMod 3) X_ne_zero (1 * i)).choose_spec l).1)
  (d := 2) (J := 1) (fun _ => X) (a := 1) (x := 1) (by simp)

end Transformer.Weil
