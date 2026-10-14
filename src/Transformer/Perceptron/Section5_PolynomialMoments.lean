/-
# Original radial and perpendicular polynomial kernel moments

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The polynomial characterization requires original exponential-kernel
tests in every degree. Their radial and perpendicular moments are actual
integrals of the original kernel times actual coordinate powers.

The original uniform law transports these actual moments under every
orthogonal map. A reflection fixes the original radial point and negates
a perpendicular vector, so every odd perpendicular moment vanishes.
Two actual reflections transport any orthogonal pair to another with
the same norms. Hence each original moment has coefficients independent
of the actual orthogonal pair chosen on the original sphere.

Every even radial/perpendicular moment of an orthonormal pair is strictly
positive. The normalized sum of its original vectors gives an actual
sphere point with positive integrand, and the original uniform law has
full support. This proves positivity using the actual measure and powers,
without supplying any eigenvalue, polynomial preservation or inverse
conclusion as a definition. The quadratic computations are included as
the degree-two case; every original degree and vector remains explicit.
-/

import Transformer.Perceptron.Section5_QuadraticGeometry
import Transformer.Perceptron.Section5_UniformSupport

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The actual original kernel moment with radial and perpendicular
coordinate powers. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Orthogonality is a hypothesis of the subsequent identities. -/
noncomputable def radialPerpMomentKernel {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x u : EucSpace d) (p q : ℕ) : ℝ :=
  ∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
    inner (𝕜 := ℝ) x (y : EucSpace d) ^ p * inner (𝕜 := ℝ) u (y : EucSpace d) ^ q ∂ν

/-- Every actual original polynomial moment integrand is continuous.
Source: arXiv:2601.21366v2, §5.1, the finite polynomial tests in
`lem: quadpol` (i). Compactness then gives actual finite integrability. -/
theorem continuous_radialPerpMomentIntegrand {d : ℕ} (β : ℝ) (x u : EucSpace d) (p q : ℕ) :
    Continuous (fun y : SSphere d => exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
      inner (𝕜 := ℝ) x (y : EucSpace d) ^ p * inner (𝕜 := ℝ) u (y : EucSpace d) ^ q) :=
  ((Real.continuous_exp.comp (continuous_const.mul
    (continuous_const.inner continuous_subtype_val))).mul
      ((continuous_const.inner continuous_subtype_val).pow p)).mul
        ((continuous_const.inner continuous_subtype_val).pow q)

/-- Scaling the actual perpendicular vector gives its actual power
factor in the original kernel moment. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, the homogeneous polynomial tests. -/
theorem radialPerpMomentKernel_smul_perp {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x u : EucSpace d) (p q : ℕ) (s : ℝ) :
    radialPerpMomentKernel β ν x (s • u) p q = s ^ q * radialPerpMomentKernel β ν x u p q := by
  unfold radialPerpMomentKernel
  rw [← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; rw [real_inner_smul_left, mul_pow]; ring

/-- The original uniform law transports every original radial and
perpendicular moment under an actual orthogonal transformation.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
This copies `secondMomentKernel_isometry_of_uniform` with actual powers. -/
theorem radialPerpMomentKernel_isometry_of_uniform {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u : EucSpace d) (p q : ℕ) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    radialPerpMomentKernel β ν (U x) (U u) p q = radialPerpMomentKernel β ν x u p q := by
  let f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (U x) (y : EucSpace d)) *
    inner (𝕜 := ℝ) (U x) (y : EucSpace d) ^ p * inner (𝕜 := ℝ) (U u) (y : EucSpace d) ^ q
  have hf : Continuous f := continuous_radialPerpMomentIntegrand β (U x) (U u) p q
  calc
    _ = ∫ y, f y ∂ν.map (Perspective.sphereMap d U) := by rw [hν.2 U]; rfl
    _ = ∫ y, f (Perspective.sphereMap d U y) ∂ν :=
      integral_map (Perspective.continuous_sphereMap d U).measurable.aemeasurable hf.aestronglyMeasurable
    _ = radialPerpMomentKernel β ν x u p q := by
      apply integral_congr_ae
      exact ae_of_all ν fun y => by simp only [f, Perspective.sphereMap, U.inner_map_map]

/-- Every original moment with an actual odd perpendicular power is
zero. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
The actual reflection fixes the radial point and reverses the original
perpendicular vector, so the original integral equals its negative. -/
theorem radialPerpMomentKernel_eq_zero_of_odd {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u : EucSpace d) (p q : ℕ) (hxu : inner (𝕜 := ℝ) x u = 0) (hq : Odd q) :
    radialPerpMomentKernel β ν x u p q = 0 := by
  let U := (ℝ ∙ u)ᗮ.reflection
  have hx : U x = x := Submodule.reflection_mem_subspace_eq_self
    (Submodule.mem_orthogonal_singleton_iff_inner_left.mpr hxu)
  have hu : U u = -u := Submodule.reflection_orthogonalComplement_singleton_eq_neg u
  have h := radialPerpMomentKernel_isometry_of_uniform β ν hν x u p q U
  rw [hx, hu, show -u = (-1 : ℝ) • u by simp, radialPerpMomentKernel_smul_perp,
    hq.neg_one_pow] at h
  linarith

/-- Two actual orthogonal pairs with equal original norms have the
same original radial/perpendicular moments in every degree.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
The first reflection exchanges the radial vectors; the second exchanges
the transported perpendicular vectors while fixing the radial point. -/
theorem radialPerpMomentKernel_eq_of_orthogonal_pairs {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u y v : EucSpace d) (p q : ℕ) (hxy : ‖x‖ = ‖y‖) (huv : ‖u‖ = ‖v‖)
    (hxu : inner (𝕜 := ℝ) x u = 0) (hyv : inner (𝕜 := ℝ) y v = 0) :
    radialPerpMomentKernel β ν x u p q = radialPerpMomentKernel β ν y v p q := by
  let U := (ℝ ∙ (x - y))ᗮ.reflection
  have hx : U x = y := Submodule.reflection_sub hxy
  have hyu : inner (𝕜 := ℝ) y (U u) = 0 := by rw [← hx, U.inner_map_map, hxu]
  let V := (ℝ ∙ (U u - v))ᗮ.reflection
  have hy : V y = y := Submodule.reflection_mem_subspace_eq_self
    (Submodule.mem_orthogonal_singleton_iff_inner_left.mpr (by rw [inner_sub_right, hyu, hyv, sub_self]))
  have hu : V (U u) = v := Submodule.reflection_sub ((U.norm_map u).trans huv)
  have h1 := radialPerpMomentKernel_isometry_of_uniform β ν hν x u p q U
  have h2 := radialPerpMomentKernel_isometry_of_uniform β ν hν y (U u) p q V
  rw [hx] at h1
  rw [hy, hu] at h2
  exact h1.symm.trans h2.symm

/-- Every actual even radial/perpendicular moment of an original
orthonormal pair is strictly positive under the original uniform law.
Source: arXiv:2601.21366v2, §5.1, the positive original kernel in
`eq:FH-basic`. Positivity is proved at the actual normalized sum of
the original vectors and then uses the actual law's full support. -/
theorem radialPerpMomentKernel_even_pos {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u : EucSpace d) (p q : ℕ) (hx : ‖x‖ = 1) (hu : ‖u‖ = 1)
    (hxu : inner (𝕜 := ℝ) x u = 0) : 0 < radialPerpMomentKernel β ν x u (2 * p) (2 * q) := by
  let _ : IsProbabilityMeasure ν := hν.1
  have hxx : inner (𝕜 := ℝ) x x = 1 := by rw [real_inner_self_eq_norm_sq, hx, one_pow]
  have huu : inner (𝕜 := ℝ) u u = 1 := by rw [real_inner_self_eq_norm_sq, hu, one_pow]
  have hux : inner (𝕜 := ℝ) u x = 0 := by rw [real_inner_comm, hxu]
  have hn : ‖x + u‖ ≠ 0 := by
    have hs := norm_add_sq_real x u
    rw [hx, hu, hxu] at hs
    nlinarith
  let z : SSphere d := ⟨‖x + u‖⁻¹ • (x + u), by
    apply mem_sphere_zero_iff_norm.mpr
    rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn]⟩
  have hzx : 0 < inner (𝕜 := ℝ) x (z : EucSpace d) := by
    change 0 < inner (𝕜 := ℝ) x (‖x + u‖⁻¹ • (x + u))
    rw [real_inner_smul_right, inner_add_right, hxx, hxu, add_zero, mul_one]
    exact inv_pos.mpr (lt_of_le_of_ne (norm_nonneg _) hn.symm)
  have hzu : 0 < inner (𝕜 := ℝ) u (z : EucSpace d) := by
    change 0 < inner (𝕜 := ℝ) u (‖x + u‖⁻¹ • (x + u))
    rw [real_inner_smul_right, inner_add_right, hux, huu, zero_add, mul_one]
    exact inv_pos.mpr (lt_of_le_of_ne (norm_nonneg _) hn.symm)
  let f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
    inner (𝕜 := ℝ) x (y : EucSpace d) ^ (2 * p) * inner (𝕜 := ℝ) u (y : EucSpace d) ^ (2 * q)
  have hf : Continuous f := continuous_radialPerpMomentIntegrand β x u _ _
  have hnn : 0 ≤ f := fun y => mul_nonneg
    (mul_nonneg (exp_pos _).le ((even_two_mul p).pow_nonneg _)) ((even_two_mul q).pow_nonneg _)
  apply (integral_pos_iff_support_of_nonneg hnn (Perspective.integrable_of_continuous_compact hf ν)).mpr
  have hfull : ν.support = univ := support_eq_univ_of_uniform (⟨ν, hν.1⟩ : Perspective.ProbSphere d) hν
  have hz : z ∈ ν.support := by rw [hfull]; exact mem_univ _
  exact (Measure.mem_support_iff_forall z).mp hz _ (hf.isOpen_support.mem_nhds
    (show f z ≠ 0 from (mul_pos (mul_pos (exp_pos _) (pow_pos hzx _)) (pow_pos hzu _)).ne'))

/-- Every uniform, orthogonal, equal-norm, even-moment and odd-power
input holds jointly for the actual original circle axes and original
uniform probability. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Both actual pairs are the same nonzero orthonormal pair. -/
example : Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    ‖(basePoint 1 : EucSpace 2)‖ = 1 ∧ ‖secondAxis‖ = 1 ∧
    inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) secondAxis = 0 ∧ Odd (1 : ℕ) :=
  ⟨Metastability.isUniformOn_uniformLaw (by norm_num), norm_basePoint_one,
    norm_secondAxis, inner_basePoint_secondAxis, by decide⟩

end Transformer.Perceptron
