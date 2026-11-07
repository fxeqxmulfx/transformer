module

public import Mathlib.LinearAlgebra.Basis.VectorSpace
public import Mathlib.LinearAlgebra.Isomorphisms
public import Transformer.Grokking.FlatSpectrum.NuclearBounds

/-!
# Conditioning of a readout modulo unobserved tangent directions

Source: `review/flat_spectrum.md`, *Approximate nonunique recovery*, QR1.
Every real-linear matrix readout has an explicit nuclear-norm bound. If it
annihilates the tangent directions invisible on the observations, its restriction
to all invisible perturbations factors through the normal projection. Extending
that factor from its range supplies a finite conditioning constant. No inverse
for the full matrix, and no tangent injectivity, is required. Existence alone
does not say that the constant is small enough for a useful certificate.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix Finset

variable {m n r : Type*} [Fintype m] [Fintype n] [Fintype r]
  [DecidableEq m] [DecidableEq n]

/-- An explicit, possibly loose bound for a real-linear readout, QR1. -/
def readoutBound (L : Matrix m n ℂ →ₗ[ℝ] ℝ) : ℝ :=
  ∑ i, ∑ j, (|L (single i j 1)| + |L (single i j Complex.I)|)

/-- The explicit conditioning bound is nonnegative, QR1. -/
theorem readoutBound_nonneg (L : Matrix m n ℂ →ₗ[ℝ] ℝ) : 0 ≤ readoutBound L := by
  exact sum_nonneg fun _ _ => sum_nonneg fun _ _ => add_nonneg (abs_nonneg _) (abs_nonneg _)

/-- Every real-linear readout is bounded by the nuclear norm, QR1. -/
theorem abs_readout_le (L : Matrix m n ℂ →ₗ[ℝ] ℝ) (X : Matrix m n ℂ) :
    |L X| ≤ readoutBound L * nuclearNorm X := by
  have hexpand : L X = ∑ i, ∑ j,
      ((X i j).re * L (single i j 1) + (X i j).im * L (single i j Complex.I)) := by
    conv_lhs => rw [eq_sum_re_smul_single_add X]
    simp only [map_sum, map_add, map_smul, smul_eq_mul]
  rw [hexpand, readoutBound, sum_mul]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i _ => ?_)
  rw [sum_mul]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun j _ => ?_)
  calc
    _ ≤ |(X i j).re * L (single i j 1)| +
        |(X i j).im * L (single i j Complex.I)| := abs_add_le _ _
    _ ≤ nuclearNorm X * (|L (single i j 1)| + |L (single i j Complex.I)|) := by
      rw [abs_mul, abs_mul, mul_add]
      exact add_le_add
        (mul_le_mul_of_nonneg_right ((Complex.abs_re_le_norm _).trans
          (norm_apply_le_nuclearNorm X i j)) (abs_nonneg _))
        (mul_le_mul_of_nonneg_right ((Complex.abs_im_le_norm _).trans
          (norm_apply_le_nuclearNorm X i j)) (abs_nonneg _))
    _ = _ := mul_comm _ _

example := abs_readout_le (0 : Matrix (Fin 2) (Fin 2) ℂ →ₗ[ℝ] ℝ) 1

/-- **Finite conditioning without unique recovery**, QR1. Annihilation of hidden
tangent directions implies a finite bound on every hidden readout perturbation
by its normal component. `κ` is not a bound on the whole perturbation. -/
theorem exists_readout_conditioning {U : Matrix m r ℂ} {V : Matrix n r ℂ}
    {Ω : Set (m × n)} (L : Matrix m n ℂ →ₗ[ℝ] ℝ)
    (hL : ∀ (A : Matrix n r ℂ) (B : Matrix m r ℂ),
      (∀ ij ∈ Ω, (U * Aᴴ + B * Vᴴ) ij.1 ij.2 = 0) → L (U * Aᴴ + B * Vᴴ) = 0) :
    ∃ κ ≥ 0, ∀ H : Matrix m n ℂ, (∀ ij ∈ Ω, H ij.1 ij.2 = 0) →
      |L H| ≤ κ * nuclearNorm ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) := by
  let N : Submodule ℝ (Matrix m n ℂ) :=
    { carrier := {H | ∀ ij ∈ Ω, H ij.1 ij.2 = 0}
      add_mem' := fun ha hb ij hij => by
        rw [Matrix.add_apply, ha ij hij, hb ij hij, add_zero]
      zero_mem' := fun _ _ => rfl
      smul_mem' := fun t H hH ij hij => by
        rw [Matrix.smul_apply, hH ij hij, smul_zero] }
  let F : N →ₗ[ℝ] Matrix m n ℂ :=
    { toFun := fun H => (1 - U * Uᴴ) * H.val * (1 - V * Vᴴ)
      map_add' := fun H H' => by simp only [Submodule.coe_add, Matrix.mul_add, Matrix.add_mul]
      map_smul' := fun t H => by
        simp only [Submodule.coe_smul, RingHom.id_apply, Matrix.mul_smul, Matrix.smul_mul] }
  let LN := L.comp N.subtype
  have hker : F.rangeRestrict.ker ≤ LN.ker := by
    intro H hH
    have hz : (1 - U * Uᴴ) * H.val * (1 - V * Vᴴ) = 0 :=
      congrArg Subtype.val hH
    have hdecomp : H.val = U * (H.valᴴ * U)ᴴ +
        ((1 - U * Uᴴ) * H.val * V) * Vᴴ := by
      have hsplit : H.val = U * Uᴴ * H.val +
          (1 - U * Uᴴ) * H.val * (V * Vᴴ) +
          (1 - U * Uᴴ) * H.val * (1 - V * Vᴴ) := by
        simp only [Matrix.sub_mul, Matrix.mul_sub, Matrix.one_mul, Matrix.mul_one]
        abel
      rw [hz, add_zero] at hsplit
      rw [conjTranspose_mul, conjTranspose_conjTranspose]
      simpa only [Matrix.mul_assoc] using hsplit
    change L H.val = 0
    rw [hdecomp]
    exact hL _ _ fun ij hij => by rw [← hdecomp]; exact H.property ij hij
  let e : {g : N →ₗ[ℝ] ℝ // F.rangeRestrict.ker ≤ g.ker} ≃ (F.range →ₗ[ℝ] ℝ) :=
    LinearMap.liftOfSurjective F.surjective_rangeRestrict
  let g : F.range →ₗ[ℝ] ℝ := e ⟨LN, hker⟩
  have hcomp : g.comp F.rangeRestrict = LN := congrArg Subtype.val (e.symm_apply_apply ⟨LN, hker⟩)
  obtain ⟨G, hG⟩ := g.exists_extend
  refine ⟨readoutBound G, readoutBound_nonneg G, fun H hH => ?_⟩
  have hfactor : G ((1 - U * Uᴴ) * H * (1 - V * Vᴴ)) = L H := by
    have h₁ := LinearMap.congr_fun hG (F.rangeRestrict ⟨H, hH⟩)
    have h₂ := LinearMap.congr_fun hcomp ⟨H, hH⟩
    exact h₁.trans h₂
  rw [← hfactor]
  exact abs_readout_le G _

example := exists_readout_conditioning (U := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := Set.univ)
  (0 : Matrix (Fin 2) (Fin 2) ℂ →ₗ[ℝ] ℝ) fun _ _ _ => rfl

end Transformer.Grokking.FlatSpectrum.FeatureCodes
