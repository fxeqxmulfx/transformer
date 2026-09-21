/-
# Bridge: Tokens after `RMSNorm` live on the sphere of radius `√d`

Connects `Transformer.GPTMini.RMSNorm` to the canonical setup of all
formalized clustering theorems (`Transformer.Section1_IPS`,
`Transformer.Section5_HighD`, `Transformer.Causal.MainTheorem`), where the
particles live on the unit sphere `𝕊^{d-1}`.

The bridge is via scaling: after applying `rmsNorm`, every (non-zero)
token has L2 norm exactly `√d`, so dividing by `√d` puts it on the unit
sphere.  Equivalently, the normalized direction `RMSNorm(x) / √d` lies on
`𝕊^{d-1}`.

This is the foundational lemma for all subsequent bridges to spherical-SA
theory.
-/

import Transformer.Basic
import Transformer.GPTMini.RMSNorm

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- **Direction on the unit sphere.**

For any non-zero `x ∈ ℝ^d`, the rescaled normalization `rmsNorm(x) / √d`
is a unit vector. -/
theorem rmsNorm_direction_on_sphere
    {d : ℕ} (hd : 0 < d) (x : EucSpace d) (hx : x ≠ 0) :
    ‖((Real.sqrt (d : ℝ))⁻¹) • rmsNorm x‖ = 1 := by
  rw [norm_smul, Real.norm_eq_abs]
  have h_norm : ‖rmsNorm x‖ = Real.sqrt (d : ℝ) :=
    rmsNorm_norm_eq_sqrt_d x hx
  rw [h_norm]
  have h_sqrt_pos : 0 < Real.sqrt (d : ℝ) :=
    Real.sqrt_pos.mpr (by exact_mod_cast hd)
  rw [abs_of_pos (inv_pos.mpr h_sqrt_pos)]
  field_simp

/-- **Spherical-coordinate map.**  The map

  `Φ(x) := rmsNorm(x) / √d : ℝ^d \ {0} → 𝕊^{d-1}`

sends any non-zero vector to a point on the unit sphere. -/
noncomputable def toSphere (d : ℕ) (x : EucSpace d) : EucSpace d :=
  ((Real.sqrt (d : ℝ))⁻¹) • rmsNorm x

/-- **`toSphere` lands on the unit sphere** (for non-zero `x`). -/
theorem toSphere_norm
    {d : ℕ} (hd : 0 < d) (x : EucSpace d) (hx : x ≠ 0) :
    ‖toSphere d x‖ = 1 := by
  unfold toSphere
  exact rmsNorm_direction_on_sphere hd x hx

/-- **Setup for spherical clustering theorems.**

Given a sequence of non-zero tokens `(x_i)_{i ∈ [n]}`, the directions
`(toSphere d (x_i))_{i ∈ [n]}` are points on `𝕊^{d-1}`.  This is the input
type expected by `Section1_IPS.SA`, `Section5_ConeCollapse.hemisphere_clustering`,
and `Causal.MainTheorem.thm1`. -/
theorem token_sequence_on_sphere
    {d n : ℕ} (hd : 0 < d) (x : Fin n → EucSpace d)
    (hx : ∀ i, x i ≠ 0) :
    ∀ i : Fin n, ‖toSphere d (x i)‖ = 1 := by
  intro i
  exact toSphere_norm hd (x i) (hx i)

/-- On a nonzero vector the direction map is `x / ‖x‖`: the `√d` of `rmsNorm`
cancels. -/
theorem toSphere_eq {d : ℕ} (hd : 0 < d) {x : EucSpace d} (hx : x ≠ 0) :
    toSphere d x = ‖x‖⁻¹ • x := by
  have hs : Real.sqrt (d : ℝ) ≠ 0 := (Real.sqrt_pos.mpr (Nat.cast_pos.mpr hd)).ne'
  rw [toSphere, rmsNorm, ite_eq_right (norm_ne_zero_iff.mpr hx), smul_smul]
  congr 1
  field_simp

/-- The hypotheses of `toSphere_eq` are satisfiable: `e₀ ≠ 0` in `ℝ¹`. -/
example : 0 < 1 ∧ (EuclideanSpace.single (0 : Fin 1) (1 : ℝ) : EucSpace 1) ≠ 0 :=
  ⟨one_pos, by simp⟩

/-- Normalizing moves a vector by at most twice the relative perturbation:
`‖b/‖b‖ - a/‖a‖‖ ≤ 2‖b - a‖/‖a‖`.  So a token far from the origin turns
slowly under bounded updates. -/
theorem norm_inv_smul_sub_le {d : ℕ} {a : EucSpace d} (b : EucSpace d) (ha : a ≠ 0) :
    ‖‖b‖⁻¹ • b - ‖a‖⁻¹ • a‖ ≤ 2 * ‖b - a‖ / ‖a‖ := by
  have hna : 0 < ‖a‖ := norm_pos_iff.mpr ha
  have hsplit : ‖b‖⁻¹ • b - ‖a‖⁻¹ • a = ‖a‖⁻¹ • (b - a) + (‖b‖⁻¹ - ‖a‖⁻¹) • b := by
    rw [smul_sub, sub_smul]; abel
  have h2 : ‖(‖b‖⁻¹ - ‖a‖⁻¹) • b‖ ≤ ‖b - a‖ / ‖a‖ := by
    rw [norm_smul, Real.norm_eq_abs]
    rcases eq_or_ne ‖b‖ 0 with hb | hb
    · rw [hb, mul_zero]; positivity
    have hnb : 0 < ‖b‖ := lt_of_le_of_ne (norm_nonneg _) (Ne.symm hb)
    have : (‖b‖⁻¹ - ‖a‖⁻¹) * ‖b‖ = (‖a‖ - ‖b‖) / ‖a‖ := by field_simp
    rw [← abs_of_pos hnb, ← abs_mul, abs_of_pos hnb, this, abs_div, abs_of_pos hna]
    gcongr
    rw [abs_sub_comm]; exact abs_norm_sub_norm_le _ _
  calc ‖‖b‖⁻¹ • b - ‖a‖⁻¹ • a‖
      ≤ ‖‖a‖⁻¹ • (b - a)‖ + ‖(‖b‖⁻¹ - ‖a‖⁻¹) • b‖ := by rw [hsplit]; exact norm_add_le _ _
    _ ≤ ‖b - a‖ / ‖a‖ + ‖b - a‖ / ‖a‖ := by
        gcongr
        rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_norm, inv_mul_eq_div]
    _ = 2 * ‖b - a‖ / ‖a‖ := by ring

/-- The hypothesis of `norm_inv_smul_sub_le` is satisfiable: `e₀ ≠ 0`. -/
example : (EuclideanSpace.single (0 : Fin 1) (1 : ℝ) : EucSpace 1) ≠ 0 := by simp

end Bridge
end GPTMini
end Transformer
