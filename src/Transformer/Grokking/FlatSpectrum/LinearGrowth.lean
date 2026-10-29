module

public import Mathlib.LinearAlgebra.Basis.VectorSpace
public import Transformer.Grokking.FlatSpectrum.NuclearBounds
public import Transformer.Grokking.FlatSpectrum.TangentGrowth

/-!
# Linear growth of the nuclear norm around `K`

Source: `review/flat_spectrum.md`, *Rank above one* (added), item 6, its two easy halves. If no
nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, the real-linear map `H ↦ ((1 − U Uᴴ) H (1 − V Vᴴ), H on Ω)`
is injective, since `H = U (Hᴴ U)ᴴ + ((1 − U Uᴴ) H V) Vᴴ + (1 − U Uᴴ) H (1 − V Vᴴ)`. A left inverse
is bounded in the nuclear norm (`exists_nuclearNorm_map_le`), so
`‖H‖_S1 ≤ C ‖(1 − U Uᴴ) H (1 − V Vᴴ)‖_S1` for every `H` vanishing on `Ω`
(`exists_nuclearNorm_le_of_tangent`). With item 1.1 (`le_nuclearNorm_add_of_certificate`), item
1.2's hypotheses then give linear growth: some `c' > 0` has `‖K‖_S1 + c' ‖H‖_S1 ≤ ‖K + H‖_S1`
for every `H` vanishing on `Ω` (`exists_pos_le_nuclearNorm_add_of_certificate`). Conversely,
linear growth leaves no nonzero `H = U Aᴴ + B Vᴴ` vanishing on `Ω`
(`eq_zero_of_exists_pos_le_nuclearNorm_add`): along `H` the nuclear norm grows at most like
`t Re tr((U Vᴴ)ᴴ H) + C t²` on both sides of `t = 0` (`exists_nuclearNorm_add_smul_tangent_le`),
so `c |t| ‖H‖_S1 ≤ C t²`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset
open scoped ComplexOrder

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r] [DecidableEq m] [DecidableEq n]
  [DecidableEq r]

omit [DecidableEq r] in
/-- **Bounded below off the tangent space**: if no nonzero `U Aᴴ + B Vᴴ` vanishes on `Ω`, some
`C` has `‖H‖_S1 ≤ C ‖(1 − U Uᴴ) H (1 − V Vᴴ)‖_S1` for every `H` vanishing on `Ω`. -/
theorem exists_nuclearNorm_le_of_tangent {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    {Ω : Set (m × n)} (hT : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → U * Aᴴ + B * Vᴴ = 0) :
    ∃ C : ℝ, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm H ≤ C * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) := by
  let F : Matrix m n ℂ →ₗ[ℝ] Matrix m n ℂ × (Ω → ℂ) :=
    { toFun := fun H => ((1 - U * Uᴴ) * H * (1 - V * Vᴴ), fun ij => H ij.1.1 ij.1.2)
      map_add' := fun H H' => Prod.ext
        (by simp only [Prod.fst_add, Matrix.mul_add, Matrix.add_mul]) rfl
      map_smul' := fun t H => Prod.ext
        (by simp only [Prod.smul_fst, RingHom.id_apply, Matrix.mul_smul, Matrix.smul_mul]) rfl }
  have hker : LinearMap.ker F = ⊥ := by
    rw [LinearMap.ker_eq_bot']
    intro H hH
    have h1 : (1 - U * Uᴴ) * H * (1 - V * Vᴴ) = 0 := congrArg Prod.fst hH
    have h2 (ij : m × n) (hij : ij ∈ Ω) : H ij.1 ij.2 = 0 :=
      congrFun (congrArg Prod.snd hH) ⟨ij, hij⟩
    have hdec : H = U * (Hᴴ * U)ᴴ + (1 - U * Uᴴ) * H * V * Vᴴ := by
      have e : H = U * Uᴴ * H + (1 - U * Uᴴ) * H * (V * Vᴴ) +
          (1 - U * Uᴴ) * H * (1 - V * Vᴴ) := by
        simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one]
        abel
      rw [h1, add_zero] at e
      rw [conjTranspose_mul, conjTranspose_conjTranspose]
      conv_lhs => rw [e]
      simp only [Matrix.mul_assoc]
    rw [hdec]
    exact hT _ _ fun ij hij => by rw [← hdec]; exact h2 ij hij
  obtain ⟨G, hG⟩ := LinearMap.exists_leftInverse_of_injective F hker
  obtain ⟨C, hC⟩ := exists_nuclearNorm_map_le (G ∘ₗ LinearMap.inl ℝ _ _)
  refine ⟨C, fun H hH => ?_⟩
  have hFH : F H = ((1 - U * Uᴴ) * H * (1 - V * Vᴴ), 0) :=
    Prod.ext rfl (funext fun ij => hH ij.1 ij.2)
  have hGH : G (F H) = H := by rw [← LinearMap.comp_apply, hG, LinearMap.id_apply]
  have h := hC ((1 - U * Uᴴ) * H * (1 - V * Vᴴ))
  rwa [LinearMap.comp_apply, LinearMap.inl_apply, ← hFH, hGH] at h

example := exists_nuclearNorm_le_of_tangent (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := Set.univ)
  fun _ _ hT => by ext i j; exact hT (i, j) (Set.mem_univ _)

/-- **Item 1.2's hypotheses give linear growth**: with a certificate `Y = U Vᴴ + W` as in
`le_nuclearNorm_add_of_certificate` for some `c > 1`, and no nonzero `U Aᴴ + B Vᴴ` vanishing on
`Ω`, some `c' > 0` has `‖K‖_S1 + c' ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`. -/
theorem exists_pos_le_nuclearNorm_add_of_certificate {K : Matrix m n ℂ} {U : Matrix m r ℂ}
    {V : Matrix n r ℂ} (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) (hKU : U * Uᴴ * K = K)
    (hK : ((U * Vᴴ)ᴴ * K).trace = nuclearNorm K) {Ω : Set (m × n)} {W : Matrix m n ℂ}
    (hYΩ : ∀ i j, (i, j) ∉ Ω → (U * Vᴴ + W) i j = 0) (hWU : Uᴴ * W = 0) (hWV : W * V = 0)
    {c : ℝ} (hc : 1 < c) (hW : IsContraction ((c : ℂ) • W))
    (hT : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → U * Aᴴ + B * Vᴴ = 0) :
    ∃ c' > 0, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c' * nuclearNorm H ≤ nuclearNorm (K + H) := by
  obtain ⟨C, hC⟩ := exists_nuclearNorm_le_of_tangent hT
  have hc1 : 0 < 1 - c⁻¹ := sub_pos.mpr (inv_lt_one_of_one_lt₀ hc)
  have hC1 : 0 < |C| + 1 := by positivity
  refine ⟨(1 - c⁻¹) / (|C| + 1), div_pos hc1 hC1, fun H hH => ?_⟩
  have h1 := le_nuclearNorm_add_of_certificate hU hV hKU hK hYΩ hWU hWV
    (zero_lt_one.trans hc) hW hH
  have h0 := nuclearNorm_nonneg ((1 - U * Uᴴ) * H * (1 - V * Vᴴ))
  have h2 : nuclearNorm H ≤ (|C| + 1) * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) :=
    (hC H hH).trans (mul_le_mul_of_nonneg_right (by linarith [le_abs_self C]) h0)
  have h3 : (1 - c⁻¹) / (|C| + 1) * nuclearNorm H ≤
      (1 - c⁻¹) * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) := by
    rw [div_mul_eq_mul_div, div_le_iff₀ hC1]
    nlinarith
  linarith

example :=
  have h := svd_hypotheses (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
    fun k => by fin_cases k <;> norm_num
  exists_pos_le_nuclearNorm_add_of_certificate (by simp) (by simp) h.1 h.2.1 (Ω := Set.univ)
    (W := 0) (fun _ _ hij => absurd (Set.mem_univ _) hij) (by simp) (by simp) one_lt_two
    (by rw [smul_zero]; exact isContraction_zero)
    fun _ _ hT => by ext i j; exact hT (i, j) (Set.mem_univ _)

omit [DecidableEq m] in
/-- **Linear growth leaves no tangent direction on the hidden cells**: with `K = U diag(s) Vᴴ`,
`s > 0`, if some `c > 0` has `‖K‖_S1 + c ‖H‖_S1 ≤ ‖K + H‖_S1` for every `H` vanishing on `Ω`,
then every `U Aᴴ + B Vᴴ` vanishing on `Ω` is zero. -/
theorem eq_zero_of_exists_pos_le_nuclearNorm_add {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    (hU : Uᴴ * U = 1) (hV : Vᴴ * V = 1) {s : r → ℝ} (hs : ∀ k, 0 < s k) {K : Matrix m n ℂ}
    (hK : K = U * diagonal (fun k => (s k : ℂ)) * Vᴴ) {Ω : Set (m × n)} {c : ℝ} (hc : 0 < c)
    (hgrow : ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      nuclearNorm K + c * nuclearNorm H ≤ nuclearNorm (K + H))
    (A : Matrix n r ℂ) (B : Matrix m r ℂ) (hH : ∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) :
    U * Aᴴ + B * Vᴴ = 0 := by
  obtain ⟨C, hC⟩ := exists_nuclearNorm_add_smul_tangent_le hU hV hs hK A B
  set H := U * Aᴴ + B * Vᴴ with hHdef
  have hL (t : ℝ) : ∀ ij ∈ Ω, ((t : ℂ) • H) ij.1 ij.2 = 0 := fun ij hij => by
    rw [Matrix.smul_apply, hH ij hij, smul_zero]
  have hnorm (t : ℝ) : nuclearNorm ((t : ℂ) • H) = |t| * nuclearNorm H := by
    rw [nuclearNorm_smul, Complex.norm_real, Real.norm_eq_abs]
  -- `c t ‖H‖ ≤ ± t Re tr((U Vᴴ)ᴴ H) + C t²` for `t > 0`
  have hpos (t : ℝ) (ht : 0 < t) : c * nuclearNorm H ≤ C * t := by
    have h1 := (hgrow _ (hL t)).trans (hC t)
    have h2 := (hgrow _ (hL (-t))).trans (hC (-t))
    rw [hnorm, abs_of_pos ht] at h1
    rw [hnorm, abs_neg, abs_of_pos ht] at h2
    nlinarith
  have h0 : c * nuclearNorm H ≤ 0 := by
    refine le_of_forall_pos_lt_add fun ε hε => ?_
    have hC1 : 0 < |C| + 1 := by positivity
    have ht : 0 < ε / (|C| + 1) := div_pos hε hC1
    have h1 : C * (ε / (|C| + 1)) < ε := by
      calc C * (ε / (|C| + 1)) ≤ |C| * (ε / (|C| + 1)) :=
            mul_le_mul_of_nonneg_right (le_abs_self C) ht.le
        _ < (|C| + 1) * (ε / (|C| + 1)) := mul_lt_mul_of_pos_right (lt_add_one _) ht
        _ = ε := mul_div_cancel₀ ε hC1.ne'
    linarith [hpos _ ht]
  have hH0 := nuclearNorm_nonneg H
  exact eq_zero_of_nuclearNorm_eq_zero (le_antisymm (by nlinarith) hH0)

example := eq_zero_of_exists_pos_le_nuclearNorm_add (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) (by simp) (s := ![1, 2])
  (fun k => by fin_cases k <;> norm_num) rfl (Ω := Set.univ) one_pos
  (fun H hH => by
    rw [show H = 0 from by ext i j; exact hH (i, j) (Set.mem_univ _), nuclearNorm_zero,
      mul_zero, add_zero, add_zero])
  0 0 fun ij _ => by simp

end Transformer.Grokking.FlatSpectrum
