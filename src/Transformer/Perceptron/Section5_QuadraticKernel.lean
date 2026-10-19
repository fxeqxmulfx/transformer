module

public import Transformer.Perceptron.Section5_LinearKernel

/-!
# The original exponential kernel tested against quadratic functions

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i),
degree two. The actual quadratic test integral is bilinear in its two
original ambient test vectors. The actual uniform law makes it covariant
under every original orthogonal transformation.

These identities are proved by the actual compact integrability and
measure pushforward. Reflecting a vector perpendicular to the original
kernel point fixes that point and reverses that vector. Hence the actual
mixed radial/perpendicular quadratic integral is zero.

This is the geometric input for computing the original quadratic kernel
coefficient and inverting the original finite quadratic attention
transform in the linear stationary-density argument of §5.3.
The actual quadratic integral is defined directly below; neither its
symmetry, coefficients nor polynomial preservation are assumed.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped ENNReal

namespace Transformer.Perceptron

/-- The actual original quadratic test integral, retaining its original
measure, ambient kernel point and both test vectors.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two. -/
noncomputable def secondMomentKernel {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b c : EucSpace d) : ℝ :=
  ∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
    inner (𝕜 := ℝ) b (y : EucSpace d) * inner (𝕜 := ℝ) c (y : EucSpace d) ∂ν

/-- The actual original quadratic integrand is continuous on the sphere.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
The finite original measures can therefore integrate each actual term. -/
theorem continuous_secondMomentIntegrand {d : ℕ} (β : ℝ) (x b c : EucSpace d) :
    Continuous (fun y : SSphere d => exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) *
      inner (𝕜 := ℝ) b (y : EucSpace d) * inner (𝕜 := ℝ) c (y : EucSpace d)) :=
  ((Real.continuous_exp.comp (continuous_const.mul
    (continuous_const.inner continuous_subtype_val))).mul
      (continuous_const.inner continuous_subtype_val)).mul
        (continuous_const.inner continuous_subtype_val)

/-- Exchanging the actual two test vectors preserves the original
quadratic integral. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
This is the symmetry of the actual product, not an assumed kernel symmetry. -/
theorem secondMomentKernel_symm {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b c : EucSpace d) : secondMomentKernel β ν x b c = secondMomentKernel β ν x c b := by
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; ring

/-- Scaling the actual first test vector scales the original integral.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
The original bilinear term retains every actual argument. -/
theorem secondMomentKernel_smul_left {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b c : EucSpace d) (s : ℝ) :
    secondMomentKernel β ν x (s • b) c = s * secondMomentKernel β ν x b c := by
  unfold secondMomentKernel
  rw [← integral_const_mul]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; rw [real_inner_smul_left]; ring

/-- Scaling both actual square-test vectors scales the original square
integral by the square of that scalar. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic`, degree two. This also applies to normalization of an
actual nonzero perpendicular vector in the coefficient computation. -/
theorem secondMomentKernel_smul_square {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    (x b : EucSpace d) (s : ℝ) :
    secondMomentKernel β ν x (s • b) (s • b) = s ^ 2 * secondMomentKernel β ν x b b := by
  rw [secondMomentKernel_smul_left, secondMomentKernel_symm β ν x b (s • b),
    secondMomentKernel_smul_left]
  ring

/-- Adding actual first test vectors adds their actual original quadratic
integrals. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
Continuity proves both needed original finite-measure integrability inputs. -/
theorem secondMomentKernel_add_left {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    [IsFiniteMeasure ν] (x b c v : EucSpace d) :
    secondMomentKernel β ν x (b + c) v =
      secondMomentKernel β ν x b v + secondMomentKernel β ν x c v := by
  unfold secondMomentKernel
  rw [← integral_add
    (Perspective.integrable_of_continuous_compact (continuous_secondMomentIntegrand β x b v) ν)
    (Perspective.integrable_of_continuous_compact (continuous_secondMomentIntegrand β x c v) ν)]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by dsimp; rw [inner_add_left]; ring

/-- The actual uniform law transports the original quadratic integral
covariantly under every actual orthogonal map.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
The identity follows from its actual measure invariance and original kernel. -/
theorem secondMomentKernel_isometry_of_uniform {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x b c : EucSpace d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) :
    secondMomentKernel β ν (U x) (U b) (U c) = secondMomentKernel β ν x b c := by
  let f := fun y : SSphere d => exp (β * inner (𝕜 := ℝ) (U x) (y : EucSpace d)) *
    inner (𝕜 := ℝ) (U b) (y : EucSpace d) * inner (𝕜 := ℝ) (U c) (y : EucSpace d)
  have hf : Continuous f := continuous_secondMomentIntegrand β (U x) (U b) (U c)
  calc
    _ = ∫ y, f y ∂ν.map (Perspective.sphereMap d U) := by rw [hν.2 U]; rfl
    _ = ∫ y, f (Perspective.sphereMap d U y) ∂ν :=
      integral_map (Perspective.continuous_sphereMap d U).measurable.aemeasurable hf.aestronglyMeasurable
    _ = secondMomentKernel β ν x b c := by
      apply integral_congr_ae
      exact ae_of_all ν fun y => by simp only [f, Perspective.sphereMap, U.inner_map_map]

/-- The actual mixed radial/perpendicular quadratic integral is zero.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
The original reflection fixes the kernel point and its radial test vector,
and negates its perpendicular test vector under the actual uniform law. -/
theorem secondMomentKernel_mixed_eq_zero_of_uniform {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (x v : EucSpace d) (hv : inner (𝕜 := ℝ) x v = 0) :
    secondMomentKernel β ν x x v = 0 := by
  let U := (ℝ ∙ v)ᗮ.reflection
  have hx : U x = x := Submodule.reflection_mem_subspace_eq_self
    (Submodule.mem_orthogonal_singleton_iff_inner_left.mpr hv)
  have hneg : U v = -v := Submodule.reflection_orthogonalComplement_singleton_eq_neg v
  have h := secondMomentKernel_isometry_of_uniform β ν hν x x v U
  rw [hx, hneg, secondMomentKernel_symm β ν x x (-v)] at h
  rw [show -v = (-1 : ℝ) • v by simp, secondMomentKernel_smul_left,
    secondMomentKernel_symm β ν x v x] at h
  linarith

/-- The original square at a sum of actual vectors expands into the two
original squares and the actual twice-mixed term.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two.
All three original integrals use the same actual finite measure and point. -/
theorem secondMomentKernel_add_square {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    [IsFiniteMeasure ν] (x b c : EucSpace d) :
    secondMomentKernel β ν x (b + c) (b + c) = secondMomentKernel β ν x b b +
      2 * secondMomentKernel β ν x b c + secondMomentKernel β ν x c c := by
  rw [secondMomentKernel_add_left, secondMomentKernel_symm β ν x b (b + c),
    secondMomentKernel_symm β ν x c (b + c), secondMomentKernel_add_left,
    secondMomentKernel_add_left, secondMomentKernel_symm β ν x c b]
  ring

/-- All uniform, finite-measure and orthogonal-vector inputs hold jointly
for the actual circle law and its original coordinate axes.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`, degree two. -/
example : ∃ ν : Measure (SSphere 2), Metastability.IsUniformOn 2 ν ∧
    IsFiniteMeasure ν ∧ inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2)
      (EuclideanSpace.single (1 : Fin 2) (1 : ℝ)) = 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let _ : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  exact ⟨MeanField.uniformLaw 2, hν, inferInstance,
    by simp [basePoint, EuclideanSpace.inner_single_left]⟩

end Transformer.Perceptron
