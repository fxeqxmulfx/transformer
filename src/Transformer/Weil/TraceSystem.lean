module

public import Transformer.Weil.LinearSystem
public import Transformer.Weil.TracePoly

/-!
# Stepanov's linear system for the trace polynomial

The coefficients `r_{ij} ∈ F[X]_{< K}`, `i < P`, `j < J`, of the auxiliary polynomial
`R = ∑ r_{ij} g^i X^{q j}` form a space of dimension `P J K`. Let `V_{b,i}` agree with
`E^{(b)}(g^i)` on `{g = c}`, with `deg V_{b,i} ≤ d b`. Then on that set, for `k < q = |F|`,
`E^{(k)} R` takes the value of `T_k(r) = ∑_{i,j} ∑_{a ≤ k} E^{(a)}(r_{ij}) V_{k − a, i} X^j`
(Leibniz, `X^{q j}` passes through `E^{(k)}`, and `x^q = x`): a linear map of degree
`< K + J + d (ℓ − 1)` for `k < ℓ`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Module Finset

variable {F : Type*} [Field F]

/-- The coefficients `(r_{ij})_{i < P, j < J}` in `F[X]_{< K}` form a space of dimension
`P J K`. -/
theorem finrank_traceDomain (P J K : ℕ) :
    finrank F (Fin P × Fin J → degreeLT F K) = P * J * K := by
  rw [Module.finrank_pi_fintype]
  simp [finrank_degreeLT]

/-- The linear system `T_k(r) = ∑_{i,j} ∑_{a ≤ k} E^{(a)}(r_{ij}) V_{k − a, i} X^j`. -/
noncomputable def traceMap (V : ℕ → ℕ → F[X]) (P J K k : ℕ) :
    (Fin P × Fin J → degreeLT F K) →ₗ[F] F[X] :=
  ∑ ij : Fin P × Fin J, ∑ a ∈ range (k + 1),
    LinearMap.mulRight F (V (k - a) ij.1 * X ^ (ij.2 : ℕ)) ∘ₗ hasseDeriv a ∘ₗ
      (degreeLT F K).subtype ∘ₗ LinearMap.proj ij

theorem traceMap_apply (V : ℕ → ℕ → F[X]) (P J K k : ℕ) (v : Fin P × Fin J → degreeLT F K) :
    traceMap V P J K k v = ∑ ij, ∑ a ∈ range (k + 1),
      hasseDeriv a (v ij : F[X]) * (V (k - a) ij.1 * X ^ (ij.2 : ℕ)) := by
  simp [traceMap, mul_assoc]

/-- `deg T_k < K + J + d (ℓ − 1)` for `k < ℓ`. -/
theorem natDegree_traceMap_lt {V : ℕ → ℕ → F[X]} {d : ℕ}
    (hV : ∀ b i, (V b i).natDegree ≤ d * b) {P J K ℓ k : ℕ} (hK : 1 ≤ K) (hk : k < ℓ)
    (v : Fin P × Fin J → degreeLT F K) :
    (traceMap V P J K k v).natDegree < K + J + d * (ℓ - 1) := by
  rw [traceMap_apply]
  refine lt_of_le_of_lt (natDegree_sum_le_of_forall_le _ _ (n := K - 1 + d * (ℓ - 1) + (J - 1))
    fun ij _ => natDegree_sum_le_of_forall_le _ _ fun a ha => ?_) (by omega)
  rw [mem_range] at ha
  have h1 : (hasseDeriv a (v ij : F[X])).natDegree ≤ K - 1 :=
    (natDegree_hasseDeriv_le _ _).trans
      (by have := natDegree_lt_of_mem_degreeLT hK (v ij).2; omega)
  have h2 : (V (k - a) ij.1).natDegree ≤ d * (ℓ - 1) :=
    (hV _ _).trans (Nat.mul_le_mul_left _ (by omega))
  have h3 : (X ^ (ij.2 : ℕ) : F[X]).natDegree ≤ J - 1 :=
    (natDegree_X_pow_le _).trans (by have := ij.2.2; omega)
  exact (natDegree_mul_le.trans (add_le_add h1 (natDegree_mul_le.trans (add_le_add h2 h3)))).trans
    (by omega)

example := natDegree_traceMap_lt (F := ZMod 2) (V := fun _ _ => 1) (d := 0) (fun _ _ => by simp)
  (P := 1) (J := 1) (K := 1) (ℓ := 1) le_rfl (k := 0) one_pos 0

/-- The auxiliary polynomial `∑_{ij} r_{ij} g^i X^{q j}` has degree at most
`K − 1 + (P − 1) deg g + q (J − 1)` when the `r_{ij}` lie in `F[X]_{< K}`. -/
theorem natDegree_traceAux_le (g : F[X]) {P J K : ℕ} (q : ℕ) (hK : 1 ≤ K)
    (v : Fin P × Fin J → degreeLT F K) :
    (∑ ij, (v ij : F[X]) * g ^ (ij.1 : ℕ) * X ^ (q * (ij.2 : ℕ))).natDegree ≤
      K - 1 + (P - 1) * g.natDegree + q * (J - 1) := by
  refine natDegree_sum_le_of_forall_le _ _ fun ij _ => natDegree_mul_le.trans
    (add_le_add (natDegree_mul_le.trans (add_le_add ?_ ?_)) ?_)
  · have := natDegree_lt_of_mem_degreeLT hK (v ij).2
    omega
  · exact natDegree_pow_le.trans (Nat.mul_le_mul_right _ (by have := ij.1.2; omega))
  · exact (natDegree_X_pow_le _).trans (Nat.mul_le_mul_left _ (by have := ij.2.2; omega))

example := natDegree_traceAux_le (F := ZMod 2) X (P := 1) (J := 1) 2 le_rfl 0

/-- On `{g = c}`, for `k < q = |F|`, `E^{(k)}(∑ r_{ij} g^i X^{q j})` takes the value of
`∑_{i,j} ∑_{a ≤ k} E^{(a)}(r_{ij}) V_{k − a, i} X^j`. -/
theorem eval_hasseDeriv_traceAux [Fintype F] {g : F[X]} {c : F} {V : ℕ → ℕ → F[X]}
    (hV : ∀ b i x, g.eval x = c → (hasseDeriv b (g ^ i)).eval x = (V b i).eval x)
    {P J k : ℕ} (hk : k < Fintype.card F) (r : Fin P × Fin J → F[X]) {x : F}
    (hx : g.eval x = c) :
    (hasseDeriv k (∑ ij, r ij * g ^ (ij.1 : ℕ) * X ^ (Fintype.card F * (ij.2 : ℕ)))).eval x =
      (∑ ij, ∑ a ∈ range (k + 1), hasseDeriv a (r ij) * (V (k - a) ij.1 * X ^ (ij.2 : ℕ))).eval
        x := by
  rw [map_sum, eval_finsetSum, eval_finsetSum]
  refine sum_congr rfl fun ij _ => ?_
  rw [hasseDeriv_mul_X_pow_card_mul _ _ hk, hasseDeriv_mul,
    Finset.Nat.sum_antidiagonal_eq_sum_range_succ_mk, sum_mul, eval_finsetSum, eval_finsetSum]
  refine sum_congr rfl fun a _ => ?_
  simp only [eval_mul, eval_pow, eval_X, hV _ _ x hx]
  rw [pow_mul, FiniteField.pow_card, mul_assoc]

example := eval_hasseDeriv_traceAux (F := ZMod 2) (g := X) (c := 0)
  (V := fun b i => hasseDeriv b (X ^ i)) (fun _ _ _ _ => rfl) (P := 1) (J := 1) (k := 0)
  (by simp) (fun _ => 1) (x := 0) (by simp)

end Transformer.Weil
