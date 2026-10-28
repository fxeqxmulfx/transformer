module

public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.RingTheory.Polynomial.DegreeLT
public import Transformer.Weil.Auxiliary

/-!
# Stepanov's linear system

The coefficients `r_j, s_j ∈ F[X]_{< K}`, `j < J`, of the auxiliary polynomial form a space of
dimension `2 J K`. Asking that `ℓ` linear maps into polynomials of degree `< D` vanish is
`ℓ D` linear equations, so for `ℓ D < 2 J K` they have a common nonzero solution (Siegel's
lemma). The maps are `T_k(r, s) = ∑_j (f^k E^{(k)}(r_j) + a σ_k(s_j)) X^j`, `k < ℓ`, of
degree `< K + J + (ℓ − 1)(m − 1)`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Module

variable {F : Type*} [Field F]

/-- `dim F[X]_{< K} = K`. -/
theorem finrank_degreeLT (K : ℕ) : finrank F (degreeLT F K) = K := by
  rw [(degreeLTEquiv F K).finrank_eq, Module.finrank_fin_fun]

/-- The pairs `(r, s)` of `J`-tuples in `F[X]_{< K}` form a space of dimension `2 J K`. -/
theorem finrank_pairs (J K : ℕ) :
    finrank F ((Fin J → degreeLT F K) × (Fin J → degreeLT F K)) = 2 * (J * K) := by
  rw [Module.finrank_prod, Module.finrank_pi_fintype]
  simp [finrank_degreeLT]
  ring

/-- `ℓ` linear maps into polynomials of degree `< D` on a space of dimension `> ℓ D` have a
common nonzero zero: their `ℓ D` coefficients below `D` are fewer equations than unknowns. -/
theorem exists_ne_zero_forall_eq_zero {V : Type*} [AddCommGroup V] [Module F V]
    [FiniteDimensional F V] {ℓ D : ℕ} (T : Fin ℓ → V →ₗ[F] F[X])
    (hT : ∀ k v, (T k v).natDegree < D) (hdim : ℓ * D < finrank F V) :
    ∃ v ≠ 0, ∀ k, T k v = 0 := by
  let L : V →ₗ[F] Fin ℓ × Fin D → F := LinearMap.pi fun kd => lcoeff F kd.2 ∘ₗ T kd.1
  have hrank : finrank F (Fin ℓ × Fin D → F) < finrank F V := by
    rwa [Module.finrank_fintype_fun_eq_card, Fintype.card_prod, Fintype.card_fin,
      Fintype.card_fin]
  obtain ⟨v, hv, hv0⟩ :=
    (Submodule.ne_bot_iff _).mp (LinearMap.ker_ne_bot_of_finrank_lt (f := L) hrank)
  refine ⟨v, hv0, fun k => Polynomial.ext fun n => ?_⟩
  rw [coeff_zero]
  rcases lt_or_ge n D with hn | hn
  · simpa [L] using congrFun (LinearMap.mem_ker.mp hv) (k, ⟨n, hn⟩)
  · exact coeff_eq_zero_of_natDegree_lt ((hT k v).trans_le hn)

example := exists_ne_zero_forall_eq_zero (F := ZMod 3) (ℓ := 1) (D := 1)
  (fun _ => monomial 0 ∘ₗ LinearMap.fst (ZMod 3) (ZMod 3) (ZMod 3)) (fun _ v => by simp)
  (by simp)

/-- A polynomial in `F[X]_{< K}`, `K ≥ 1`, has `natDegree < K`. -/
theorem natDegree_lt_of_mem_degreeLT {K : ℕ} (hK : 1 ≤ K) {p : F[X]} (hp : p ∈ degreeLT F K) :
    p.natDegree < K := by
  by_cases h : p = 0
  · rw [h, natDegree_zero]
    omega
  · exact (natDegree_lt_iff_degree_lt h).mpr (mem_degreeLT.mp hp)

example := natDegree_lt_of_mem_degreeLT (F := ZMod 3) le_rfl (p := 0) (Submodule.zero_mem _)

/-- Stepanov's linear system on the pairs `(r, s)`:
`T_k(r, s) = ∑_j (f^k E^{(k)}(r_j) + a σ_k(s_j)) X^j`. -/
noncomputable def stepanovMap (f : F[X]) (G : ℕ → F[X]) (a : F) (J K k : ℕ) :
    ((Fin J → degreeLT F K) × (Fin J → degreeLT F K)) →ₗ[F] F[X] :=
  ∑ j : Fin J, LinearMap.mulRight F (X ^ (j : ℕ)) ∘ₗ
    (LinearMap.mulLeft F (f ^ k) ∘ₗ hasseDeriv k ∘ₗ (degreeLT F K).subtype ∘ₗ
        LinearMap.proj j ∘ₗ LinearMap.fst F _ _ +
      a • (sigma f G k ∘ₗ (degreeLT F K).subtype ∘ₗ LinearMap.proj j ∘ₗ LinearMap.snd F _ _))

theorem stepanovMap_apply (f : F[X]) (G : ℕ → F[X]) (a : F) (J K k : ℕ)
    (v : (Fin J → degreeLT F K) × (Fin J → degreeLT F K)) :
    stepanovMap f G a J K k v =
      ∑ j, (f ^ k * hasseDeriv k (v.1 j : F[X]) + a • sigma f G k (v.2 j : F[X])) *
        X ^ (j : ℕ) := by
  simp [stepanovMap, add_mul]

/-- `deg T_k < K + J + (ℓ − 1)(m − 1)` for `k < ℓ`. -/
theorem natDegree_stepanovMap_lt {f : F[X]} (hf : 1 ≤ f.natDegree) {G : ℕ → F[X]}
    (hG : ∀ l, (G l).natDegree ≤ l * (f.natDegree - 1)) (a : F) {J K ℓ k : ℕ} (hK : 1 ≤ K)
    (hk : k < ℓ) (v : (Fin J → degreeLT F K) × (Fin J → degreeLT F K)) :
    (stepanovMap f G a J K k v).natDegree < K + J + (ℓ - 1) * (f.natDegree - 1) := by
  rw [stepanovMap_apply]
  have h := natDegree_sum_pow_mul_hasseDeriv_le hf hG k a (d := K - 1)
    (fun j => Nat.le_sub_one_of_lt (natDegree_lt_of_mem_degreeLT hK (v.1 j).2))
    (fun j => Nat.le_sub_one_of_lt (natDegree_lt_of_mem_degreeLT hK (v.2 j).2))
  have hk' : k * (f.natDegree - 1) ≤ (ℓ - 1) * (f.natDegree - 1) :=
    Nat.mul_le_mul_right _ (by omega)
  omega

example := natDegree_stepanovMap_lt (F := ZMod 3) (f := X) (by simp) (G := fun _ => 0)
  (fun _ => by simp) 1 (J := 1) (K := 1) (ℓ := 1) le_rfl (k := 0) one_pos 0

end Transformer.Weil
