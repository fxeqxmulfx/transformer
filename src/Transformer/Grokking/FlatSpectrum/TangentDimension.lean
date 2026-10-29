module

public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Transformer.Grokking.FlatSpectrum.LinearGrowth

/-!
# Linear growth needs `r (m + n − r)` cells

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 6.5. Let `U` (`m × r`) and `V`
(`n × r`) have orthonormal columns. The complex-linear map `Ψ (C, B) = (U C + B Vᴴ) on Ω` has, when
no nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, its kernel inside the image of `D ↦ (D Vᴴ, −U D)`: from
`U C + B Vᴴ = 0`, `B = −U (C V)` and `C = (C V) Vᴴ`. By rank–nullity,
`r n + m r ≤ |Ω| + r²`, that is, `|Ω| ≥ r (m + n − r)` (`card_mul_add_le_ncard_of_tangent`), and
so for linear growth (`card_mul_add_le_ncard_of_exists_pos_le_nuclearNorm_add`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Module

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

omit [DecidableEq m] [DecidableEq n] in
/-- **The dimension count**: with `U` (`m × r`) and `V` (`n × r`) of orthonormal columns, if no
nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, then `r n + m r ≤ |Ω| + r²`, that is,
`|Ω| ≥ r (m + n − r)`. -/
theorem card_mul_add_le_ncard_of_tangent {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {Ω : Set (m × n)}
    (hT : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → U * Aᴴ + B * Vᴴ = 0) :
    Fintype.card r * Fintype.card n + Fintype.card m * Fintype.card r ≤
      Ω.ncard + Fintype.card r * Fintype.card r := by
  classical
  let Ψ : Matrix r n ℂ × Matrix m r ℂ →ₗ[ℂ] (Ω → ℂ) :=
    { toFun := fun CB ij => (U * CB.1 + CB.2 * Vᴴ) ij.1.1 ij.1.2
      map_add' := fun CB CB' => by
        ext ij
        simp only [Prod.fst_add, Prod.snd_add, Matrix.mul_add, Matrix.add_mul, Matrix.add_apply,
          Pi.add_apply]
        ring
      map_smul' := fun t CB => by
        ext ij
        simp only [Prod.smul_fst, Prod.smul_snd, Matrix.mul_smul, Matrix.smul_mul,
          Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, RingHom.id_apply, Pi.smul_apply]
        ring }
  let ι : Matrix r r ℂ →ₗ[ℂ] Matrix r n ℂ × Matrix m r ℂ :=
    { toFun := fun D => (D * Vᴴ, -(U * D))
      map_add' := fun D D' => by
        simp only [Matrix.add_mul, Matrix.mul_add, neg_add, Prod.mk_add_mk]
      map_smul' := fun t D => by
        simp only [Matrix.smul_mul, Matrix.mul_smul, smul_neg, RingHom.id_apply, Prod.smul_mk] }
  have hker : LinearMap.ker Ψ ≤ LinearMap.range ι := by
    rintro ⟨C, B⟩ hCB
    have h0 : U * C + B * Vᴴ = 0 := by
      have h := hT Cᴴ B fun ij hij => by
        rw [conjTranspose_conjTranspose]
        exact congrFun (LinearMap.mem_ker.mp hCB) ⟨ij, hij⟩
      rwa [conjTranspose_conjTranspose] at h
    have hB : B = -(U * (C * V)) := by
      have e : (U * C + B * Vᴴ) * V = 0 := by rw [h0, Matrix.zero_mul]
      rw [Matrix.add_mul, Matrix.mul_assoc B, hV, Matrix.mul_one, Matrix.mul_assoc] at e
      exact eq_neg_of_add_eq_zero_left ((add_comm _ _).trans e)
    have hC : C * V * Vᴴ = C := by
      calc C * V * Vᴴ = Uᴴ * U * (C * V * Vᴴ) := by rw [hU, Matrix.one_mul]
        _ = Uᴴ * (-(-(U * (C * V)) * Vᴴ)) := by
          simp only [Matrix.neg_mul, neg_neg, Matrix.mul_assoc]
        _ = Uᴴ * (U * C) := by rw [← hB, ← eq_neg_of_add_eq_zero_left h0]
        _ = C := by rw [← Matrix.mul_assoc, hU, Matrix.one_mul]
    exact ⟨C * V, Prod.ext hC hB.symm⟩
  have hsum := LinearMap.finrank_range_add_finrank_ker Ψ
  have hrange : finrank ℂ (LinearMap.range Ψ) ≤ Ω.ncard := by
    have h := Submodule.finrank_le (LinearMap.range Ψ)
    rwa [Module.finrank_fintype_fun_eq_card, ← Nat.card_eq_fintype_card,
      Nat.card_coe_set_eq] at h
  have hkerle : finrank ℂ (LinearMap.ker Ψ) ≤ Fintype.card r * Fintype.card r := by
    have h := (Submodule.finrank_mono hker).trans (LinearMap.finrank_range_le ι)
    rwa [Module.finrank_matrix, Module.finrank_self, mul_one] at h
  rw [Module.finrank_prod, Module.finrank_matrix, Module.finrank_matrix, Module.finrank_self,
    mul_one, mul_one] at hsum
  omega

/-- On all four cells of a `2 × 2` table the count is tight: `2 · 2 + 2 · 2 = 4 + 2 · 2`. -/
example := card_mul_add_le_ncard_of_tangent (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := Set.univ) (by simp) (by simp) fun A B h => by
    ext i j
    exact h (i, j) trivial

omit [DecidableEq m] in
/-- **Linear growth needs `r (m + n − r)` cells**: with `K = U diag(s) Vᴴ`, `s > 0`, if some
`c > 0` has `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`, then
`r n + m r ≤ |Ω| + r²`. -/
theorem card_mul_add_le_ncard_of_exists_pos_le_nuclearNorm_add {U : Matrix m r ℂ}
    {V : Matrix n r ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k)
    {K : Matrix m n ℂ} (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)}
    {c : ℝ} (hc : 0 < c) (hgrow : ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H)) :
    Fintype.card r * Fintype.card n + Fintype.card m * Fintype.card r ≤
      Ω.ncard + Fintype.card r * Fintype.card r :=
  card_mul_add_le_ncard_of_tangent hU hV
    (eq_zero_of_exists_pos_le_nuclearNorm_add hU hV hs hK hc hgrow)

example := card_mul_add_le_ncard_of_exists_pos_le_nuclearNorm_add
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (V := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (s := fun _ => 1) (K := 1) (Ω := Set.univ) (by simp) (by simp) (fun _ => one_pos)
  (by simp) one_pos fun H hH => by
    obtain rfl : H = 0 := by
      ext i j
      exact hH (i, j) trivial
    simp [nuclearNorm_zero]

end Transformer.Grokking.FlatSpectrum
