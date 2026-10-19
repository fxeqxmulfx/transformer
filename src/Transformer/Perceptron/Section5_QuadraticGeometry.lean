module

public import Transformer.Perceptron.Section5_QuadraticKernel

/-!
# Orthogonal geometry of the original quadratic test integral

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i),
degree two. At a fixed original kernel point, the actual uniform law
gives the same square integral to all unit perpendicular test vectors.
The actual reflection exchanging two such vectors fixes that point.

Scaling then determines the integral at every perpendicular vector.
Decomposing an arbitrary original vector into its actual radial and
perpendicular parts has no mixed contribution, by the previously proved
reflection identity. The original quadratic integral is therefore a
linear combination of its squared norm and squared radial coordinate.

The auxiliary unit perpendicular vector here is an actual geometric
input, jointly witnessed on the original circle. The subsequent global
coefficient construction supplies this vector at every sphere point by
transporting the original coordinate axes with an actual orthogonal map.
No coefficient, preservation or inverse conclusion is assumed.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- Original points of the same actual norm have the same original
radial square integral. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. The actual reflection exchanging the two
points transports the kernel point and both actual radial test vectors. -/
theorem secondMomentKernel_radial_eq_of_norm_eq {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x y : EucSpace d) (hxy : ‖x‖ = ‖y‖) :
    secondMomentKernel β ν x x x = secondMomentKernel β ν y y y := by
  let U := (ℝ ∙ (x - y))ᗮ.reflection
  have hx : U x = y := Submodule.reflection_sub hxy
  have h := secondMomentKernel_isometry_of_uniform β ν hν x x x U
  simpa only [hx] using h.symm

/-- An actual unit perpendicular vector exists at every original sphere
point in dimension at least two. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, the degree-two geometry. The first actual coordinate axis
is transported to the original point and the second to the witness. -/
theorem exists_unit_perp_sphere {d : ℕ} (hd : 2 ≤ d) (x : SSphere d) :
    ∃ u : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (u : EucSpace d) = 0 := by
  let e₀ : SSphere d := ⟨EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ), by simp⟩
  let e₁ : EucSpace d := EuclideanSpace.single (⟨1, by omega⟩ : Fin d) (1 : ℝ)
  let U := (ℝ ∙ ((e₀ : EucSpace d) - (x : EucSpace d)))ᗮ.reflection
  have hx : U (e₀ : EucSpace d) = (x : EucSpace d) :=
    Submodule.reflection_sub ((norm_eq_of_mem_sphere e₀).trans (norm_eq_of_mem_sphere x).symm)
  refine ⟨⟨U e₁, by simp [e₁]⟩, ?_⟩
  change inner (𝕜 := ℝ) (x : EucSpace d) (U e₁) = 0
  rw [← hx, U.inner_map_map]
  simp [e₀, e₁, EuclideanSpace.inner_single_left]

/-- All original unit perpendicular vectors have the same actual square
integral at a fixed original point. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. The original reflection exchanges the test
vectors while fixing the original kernel point and actual uniform law. -/
theorem secondMomentKernel_unit_perp_eq {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u v : EucSpace d) (hu : ‖u‖ = 1) (hv : ‖v‖ = 1)
    (hxu : inner (𝕜 := ℝ) x u = 0) (hxv : inner (𝕜 := ℝ) x v = 0) :
    secondMomentKernel β ν x u u = secondMomentKernel β ν x v v := by
  let U := (ℝ ∙ (u - v))ᗮ.reflection
  have hx : U x = x := Submodule.reflection_mem_subspace_eq_self
    (Submodule.mem_orthogonal_singleton_iff_inner_left.mpr (by rw [inner_sub_right, hxu, hxv, sub_self]))
  have huv : U u = v := Submodule.reflection_sub (hu.trans hv.symm)
  have h := secondMomentKernel_isometry_of_uniform β ν hν x u u U
  simpa only [hx, huv] using h.symm

/-- The actual square integral at any perpendicular vector scales with
its actual squared norm. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. The zero vector is handled directly; the actual
nonzero vector is normalized before the proved unit-vector identity. -/
theorem secondMomentKernel_perp_eq_norm_sq {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u v : EucSpace d) (hu : ‖u‖ = 1)
    (hxu : inner (𝕜 := ℝ) x u = 0) (hxv : inner (𝕜 := ℝ) x v = 0) :
    secondMomentKernel β ν x v v = ‖v‖ ^ 2 * secondMomentKernel β ν x u u := by
  by_cases hv : v = 0
  · simp [hv, secondMomentKernel]
  have hn : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  have hunit : ‖‖v‖⁻¹ • v‖ = 1 := by
    rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_nonneg (norm_nonneg v), inv_mul_cancel₀ hn]
  have hperp : inner (𝕜 := ℝ) x (‖v‖⁻¹ • v) = 0 := by
    rw [real_inner_smul_right, hxv, mul_zero]
  have h := secondMomentKernel_unit_perp_eq β ν hν x u (‖v‖⁻¹ • v) hu hunit hxu hperp
  rw [secondMomentKernel_smul_square] at h
  have hscale : ‖v‖ ^ 2 * ‖v‖⁻¹ ^ 2 = 1 := by field_simp
  calc
    _ = (‖v‖ ^ 2 * ‖v‖⁻¹ ^ 2) * secondMomentKernel β ν x v v := by rw [hscale, one_mul]
    _ = ‖v‖ ^ 2 * (‖v‖⁻¹ ^ 2 * secondMomentKernel β ν x v v) := by ring
    _ = ‖v‖ ^ 2 * secondMomentKernel β ν x u u := by rw [← h]

/-- The actual perpendicular projection has its original squared norm.
Source: arXiv:2601.21366v2, §5.1, degree-two spherical polynomials.
The proof is copied from `Homogenized.norm_proj_sq`, retaining the actual
unit hypothesis and original projection, without importing that paper. -/
theorem quadratic_norm_proj_sq {d : ℕ} (x b : EucSpace d) (hx : ‖x‖ = 1) :
    ‖proj d x b‖ ^ 2 = ‖b‖ ^ 2 - inner (𝕜 := ℝ) x b ^ 2 := by
  rw [proj, norm_sub_sq_real, real_inner_smul_right, norm_smul, real_inner_comm b x, hx]
  simp
  ring

/-- The original quadratic square integral decomposes into its actual
radial and perpendicular squares, with the mixed term proved zero.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
No radial coefficient or vanishing mixed integral is assumed. -/
theorem secondMomentKernel_radial_perp_decomposition {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x b : EucSpace d) (hx : ‖x‖ = 1) :
    secondMomentKernel β ν x b b =
      inner (𝕜 := ℝ) x b ^ 2 * secondMomentKernel β ν x x x +
        secondMomentKernel β ν x (proj d x b) (proj d x b) := by
  let _ : IsProbabilityMeasure ν := hν.1
  have hb : b = inner (𝕜 := ℝ) x b • x + proj d x b := by unfold proj; abel
  have hp := inner_proj_eq_zero hx b
  have hm : secondMomentKernel β ν x (inner (𝕜 := ℝ) x b • x) (proj d x b) = 0 := by
    rw [secondMomentKernel_smul_left, secondMomentKernel_mixed_eq_zero_of_uniform β ν hν x _ hp,
      mul_zero]
  calc
    _ = secondMomentKernel β ν x (inner (𝕜 := ℝ) x b • x + proj d x b)
        (inner (𝕜 := ℝ) x b • x + proj d x b) := by rw [← hb]
    _ = _ := by rw [secondMomentKernel_add_square, secondMomentKernel_smul_square, hm]; ring

/-- One actual unit perpendicular vector determines the original square
integral at every original test vector. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, degree two. The coefficient expression follows from
the actual radial decomposition and actual perpendicular squared norm. -/
theorem secondMomentKernel_eq_radial_norm_combination {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x u b : EucSpace d) (hx : ‖x‖ = 1) (hu : ‖u‖ = 1)
    (hxu : inner (𝕜 := ℝ) x u = 0) :
    secondMomentKernel β ν x b b = secondMomentKernel β ν x u u * ‖b‖ ^ 2 +
      (secondMomentKernel β ν x x x - secondMomentKernel β ν x u u) * inner (𝕜 := ℝ) x b ^ 2 := by
  rw [secondMomentKernel_radial_perp_decomposition β ν hν x b hx,
    secondMomentKernel_perp_eq_norm_sq β ν hν x u (proj d x b) hu hxu (inner_proj_eq_zero hx b),
    quadratic_norm_proj_sq x b hx]
  ring

/-- Every unit and perpendicular hypothesis holds jointly for the actual
circle axes and genuine original uniform probability.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
The same actual perpendicular axis supplies both unit-vector inputs. -/
example : ∃ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν ∧
    (2 : ℕ) ≤ 2 ∧
    ‖((basePoint 1 : SSphere 2) : EucSpace 2)‖ = 1 ∧
    ‖EuclideanSpace.single (1 : Fin 2) (1 : ℝ)‖ = 1 ∧
    inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  exact ⟨MeanField.uniformLaw 2, hν, le_rfl, norm_eq_of_mem_sphere (basePoint 1),
    by simp, by simp [basePoint, EuclideanSpace.inner_single_left]⟩

end Transformer.Perceptron
