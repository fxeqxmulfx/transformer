module

public import Transformer.Perceptron.Section5_QuadraticGeometry

/-!
# The actual uniform kernel coefficients for degree-two functions

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
An actual unit perpendicular vector exists at each original sphere point
in dimension at least two. The actual orthogonal transport from one
fixed unit pair makes the two original quadratic coefficients constant.

Consequently the original kernel sends the square of every original
linear function to a constant plus the square of that same function.
At positive temperature its squared-coordinate coefficient is nonzero:
otherwise the original difference of two actual orthogonal coordinate
squares would have zero transform and hence vanish identically, contrary
to its actual value one at its own unit coordinate point.

Both degree-zero and degree-two coefficients needed for inversion are
therefore proved nonzero. All coefficients are obtained from original
integrals against the actual original uniform law; no spectral formula,
polynomial preservation, inverse or density is an unproved input.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- The original repeated bilinear test is exactly the original squared
linear-function test. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. No integrand or normalization is changed. -/
theorem secondMomentKernel_eq_integral_sq {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (x b : EucSpace d) :
    secondMomentKernel β ν x b b =
      ∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
        inner (𝕜 := ℝ) b (y : EucSpace d) ^ 2 ∂ν := by
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; ring

/-- The actual original quadratic kernel has two constant coefficients
at all actual unit points. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. An actual fixed perpendicular unit pair is
transported to each point using the original measure's orthogonal symmetry. -/
theorem exists_uniform_quadratic_kernel_coefficients {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ a c : ℝ, ∀ (b : EucSpace d) (x : SSphere d),
      secondMomentKernel β ν (x : EucSpace d) b b =
        a * ‖b‖ ^ 2 + c * inner (𝕜 := ℝ) (x : EucSpace d) b ^ 2 := by
  let _ : IsProbabilityMeasure ν := hν.1
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure ν)
  obtain ⟨u₀, hu₀⟩ := exists_unit_perp_sphere hd x₀
  refine ⟨secondMomentKernel β ν (x₀ : EucSpace d) (u₀ : EucSpace d) (u₀ : EucSpace d),
    secondMomentKernel β ν (x₀ : EucSpace d) (x₀ : EucSpace d) (x₀ : EucSpace d) -
      secondMomentKernel β ν (x₀ : EucSpace d) (u₀ : EucSpace d) (u₀ : EucSpace d), fun b x => ?_⟩
  let U := (ℝ ∙ ((x₀ : EucSpace d) - (x : EucSpace d)))ᗮ.reflection
  have hx : U (x₀ : EucSpace d) = (x : EucSpace d) :=
    Submodule.reflection_sub ((norm_eq_of_mem_sphere x₀).trans (norm_eq_of_mem_sphere x).symm)
  have hxu : inner (𝕜 := ℝ) (x : EucSpace d) (U (u₀ : EucSpace d)) = 0 := by
    rw [← hx, U.inner_map_map]
    exact hu₀
  have hperp := secondMomentKernel_isometry_of_uniform β ν hν
    (x₀ : EucSpace d) (u₀ : EucSpace d) (u₀ : EucSpace d) U
  rw [hx] at hperp
  have hrad := secondMomentKernel_radial_eq_of_norm_eq β ν hν (x : EucSpace d) (x₀ : EucSpace d)
    ((norm_eq_of_mem_sphere x).trans (norm_eq_of_mem_sphere x₀).symm)
  rw [secondMomentKernel_eq_radial_norm_combination β ν hν (x : EucSpace d)
    (U (u₀ : EucSpace d)) b (norm_eq_of_mem_sphere x)
    (by rw [U.norm_map]; exact norm_eq_of_mem_sphere u₀) hxu, hperp, hrad]

/-- At positive temperature the actual squared-coordinate coefficient
is nonzero in every dimension at least two.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
If it vanished, signed-density injectivity would kill the actual nonzero
difference of two orthogonal unit-coordinate squares. -/
theorem exists_nonzero_uniform_quadratic_kernel_coefficients {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ a c : ℝ, c ≠ 0 ∧ ∀ (b : EucSpace d) (x : SSphere d),
      secondMomentKernel β ν (x : EucSpace d) b b =
        a * ‖b‖ ^ 2 + c * inner (𝕜 := ℝ) (x : EucSpace d) b ^ 2 := by
  let _ : IsProbabilityMeasure ν := hν.1
  obtain ⟨a, c, hq⟩ := exists_uniform_quadratic_kernel_coefficients hd β ν hν
  refine ⟨a, c, ?_, hq⟩
  intro hc
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure ν)
  obtain ⟨u₀, hperp⟩ := exists_unit_perp_sphere hd x₀
  have hfx : Continuous (fun y : SSphere d => inner (𝕜 := ℝ) (x₀ : EucSpace d) (y : EucSpace d) ^ 2) :=
    (continuous_const.inner continuous_subtype_val).pow 2
  have hfu : Continuous (fun y : SSphere d => inner (𝕜 := ℝ) (u₀ : EucSpace d) (y : EucSpace d) ^ 2) :=
    (continuous_const.inner continuous_subtype_val).pow 2
  have hz := eq_zero_of_integral_exp_inner_mul_eq_zero_uniform β hβ ν hν
    (fun y => inner (𝕜 := ℝ) (x₀ : EucSpace d) (y : EucSpace d) ^ 2 -
      inner (𝕜 := ℝ) (u₀ : EucSpace d) (y : EucSpace d) ^ 2) (hfx.sub hfu) fun x => by
      have hK : Continuous (fun y : SSphere d =>
          exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))) :=
        Real.continuous_exp.comp (continuous_const.mul
          (continuous_const.inner continuous_subtype_val))
      simp_rw [mul_sub]
      rw [integral_sub
        (f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
          inner (𝕜 := ℝ) (x₀ : EucSpace d) (y : EucSpace d) ^ 2)
        (g := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) *
          inner (𝕜 := ℝ) (u₀ : EucSpace d) (y : EucSpace d) ^ 2)
        (Perspective.integrable_of_continuous_compact (hK.mul hfx) ν)
        (Perspective.integrable_of_continuous_compact (hK.mul hfu) ν),
        ← secondMomentKernel_eq_integral_sq, ← secondMomentKernel_eq_integral_sq,
        hq, hq, hc, norm_eq_of_mem_sphere x₀, norm_eq_of_mem_sphere u₀]
      ring
  have h := congrFun hz x₀
  change inner (𝕜 := ℝ) (x₀ : EucSpace d) (x₀ : EucSpace d) ^ 2 -
    inner (𝕜 := ℝ) (u₀ : EucSpace d) (x₀ : EucSpace d) ^ 2 = 0 at h
  rw [real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere,
    real_inner_comm (x₀ : EucSpace d) (u₀ : EucSpace d), hperp] at h
  norm_num at h

/-- The original constant and squared-coordinate coefficients needed
for inversion are jointly nonzero at positive temperature.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degrees zero and two.
The actual constant integral and quadratic integrals are retained together. -/
theorem exists_uniform_constant_quadratic_coefficients {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ c₀ a c : ℝ, c₀ ≠ 0 ∧ c ≠ 0 ∧
      (∀ x : SSphere d, (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) ∂ν) = c₀) ∧
      ∀ (b : EucSpace d) (x : SSphere d), secondMomentKernel β ν (x : EucSpace d) b b =
        a * ‖b‖ ^ 2 + c * inner (𝕜 := ℝ) (x : EucSpace d) b ^ 2 := by
  obtain ⟨c₀, c₁, hc₀, -, haff⟩ := exists_uniform_affine_kernel_coefficients β hβ ν hν
  obtain ⟨a, c, hc, hq⟩ := exists_nonzero_uniform_quadratic_kernel_coefficients hd β hβ ν hν
  exact ⟨c₀, a, c, hc₀, hc, fun x => by simpa using haff 1 0 x, hq⟩

/-- Differences of original equal-norm coordinate squares share one
actual nonzero kernel coefficient. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`, degree two. Equality of the actual vector norms cancels
the proved constant contribution, retaining the original difference. -/
theorem exists_uniform_equal_norm_quadratic_coefficient {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ∃ c : ℝ, c ≠ 0 ∧ ∀ (u v : EucSpace d), ‖u‖ = ‖v‖ → ∀ x : SSphere d,
      secondMomentKernel β ν (x : EucSpace d) u u - secondMomentKernel β ν (x : EucSpace d) v v =
        c * (inner (𝕜 := ℝ) (x : EucSpace d) u ^ 2 - inner (𝕜 := ℝ) (x : EucSpace d) v ^ 2) := by
  obtain ⟨a, c, hc, hq⟩ := exists_nonzero_uniform_quadratic_kernel_coefficients hd β hβ ν hν
  refine ⟨c, hc, fun u v huv x => ?_⟩
  rw [hq, hq, huv]
  ring

/-- Every positive-temperature, dimensional and uniform-law input holds
jointly for the actual original circle probability.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, one_pos, Metastability.isUniformOn_uniformLaw (by norm_num)⟩

end Transformer.Perceptron
