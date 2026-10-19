module

public import Transformer.Perceptron.Section5_PolynomialSpan
public import Transformer.Perceptron.Section5_PowerPolynomials
public import Transformer.Perceptron.Section5_DensityIdentification
public import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
public import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-!
# The original kernel's finite-dimensional polynomial inverse

ArXiv:2601.21366v2, §5.1, `eq:FH-basic` and `lem: quadpol` (i).
The actual exponential integral defines a linear map on continuous
functions of the original sphere. Its original moment formulas and the
proved spanning theorem preserve each actual bounded-degree polynomial
space. Injectivity for continuous signed densities makes its restriction
to that finite-dimensional space bijective.

Thus every original polynomial transform has a polynomial signed inverse
of the same degree. Positivity and identification with the original
probability are supplied by the separate proved density theorem.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The original kernel with its actual second argument fixed, viewed
as a continuous function of the original sphere point.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`. -/
noncomputable def sphereKernelSection {d : ℕ} (β : ℝ) (y : SSphere d) : C(SSphere d, ℝ) :=
  ⟨fun x => exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)), by fun_prop⟩

/-- The actual kernel sections vary continuously in the uniform norm.
Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Joint continuity on the actual compact sphere gives this continuity. -/
theorem continuous_sphereKernelSection {d : ℕ} (β : ℝ) : Continuous (sphereKernelSection (d := d) β) := by
  let K : C(SSphere d × SSphere d, ℝ) :=
    ⟨fun z => exp (β * inner (𝕜 := ℝ) (z.2 : EucSpace d) (z.1 : EucSpace d)), by fun_prop⟩
  exact K.curry.continuous

/-- The actual original continuous kernel sections are integrable
against every original finite law, including an actual continuous
signed density. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
theorem integrable_sphereKernelSection_smul {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] (f : C(SSphere d, ℝ)) :
    Integrable (fun y => f y • sphereKernelSection β y) ν :=
  Perspective.integrable_of_continuous_compact
    (f.continuous.smul (continuous_sphereKernelSection β)) ν

/-- The actual original exponential integral as a real linear map on
continuous functions. Source: arXiv:2601.21366v2, §5.1, `eq:FH-basic`.
Its pointwise expression and degree preservation are proved below. -/
noncomputable def sphereKernelOperator {d : ℕ} (β : ℝ)
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] : C(SSphere d, ℝ) →ₗ[ℝ] C(SSphere d, ℝ) where
  toFun f := ∫ y, f y • sphereKernelSection β y ∂ν
  map_add' f g := by
    simp only [ContinuousMap.add_apply, add_smul]
    exact integral_add (integrable_sphereKernelSection_smul β ν f)
      (integrable_sphereKernelSection_smul β ν g)
  map_smul' r f := by
    change (∫ y, (r * f y) • sphereKernelSection β y ∂ν) = r • ∫ y, f y • sphereKernelSection β y ∂ν
    simp_rw [← smul_smul]
    exact integral_smul r (fun y => f y • sphereKernelSection β y)

/-- The actual continuous-map operator is the original pointwise
exponential kernel integral. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic`. Its continuous-map integrability was proved above. -/
theorem sphereKernelOperator_apply {d : ℕ} (β : ℝ) (ν : Measure (SSphere d))
    [IsFiniteMeasure ν] (f : C(SSphere d, ℝ)) (x : SSphere d) :
    sphereKernelOperator β ν f x =
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * f y ∂ν := by
  change (∫ y, f y • sphereKernelSection β y ∂ν) x = _
  rw [ContinuousMap.integral_apply (integrable_sphereKernelSection_smul β ν f)]
  apply integral_congr_ae
  exact ae_of_all ν fun y => by
    change f y * exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) = _
    ring

/-- The actual original exponential kernel preserves every original
bounded-degree spherical polynomial space. Source: arXiv:2601.21366v2,
§5.1, `eq:FH-basic` and `lem: quadpol` (i).
The original homogeneous moment formulas apply to the proved spanning
generators; actual linearity supplies every mixed polynomial. -/
theorem sphereKernelOperator_preserves_polynomials {d k : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (hν : Metastability.IsUniformOn d ν) (f : C(SSphere d, ℝ))
    (hf : f ∈ spherePolynomialSpace d k) :
    sphereKernelOperator β ν f ∈ spherePolynomialSpace d k := by
  have hle : spherePowerSpan d k ≤ (spherePolynomialSpace d k).comap (sphereKernelOperator β ν) := by
    apply Submodule.span_le.mpr
    rintro g ⟨b, n, hn, rfl⟩
    apply (mem_spherePolynomialSpace_iff _).mpr
    have hp := isPolyOfDegreeLE_mono (isPolyOfDegreeLE_powerMomentKernel hd β ν hν b n) hn
    have heq : (sphereKernelOperator β ν (sphereLinearFunction b ^ n) : SSphere d → ℝ) =
        fun x : SSphere d => powerMomentKernel β ν (x : EucSpace d) b n := by
      funext x
      rw [sphereKernelOperator_apply]
      rfl
    rwa [heq]
  rw [spherePowerSpan_eq_polynomialSpace] at hle
  exact hle hf

/-- The actual original kernel is injective on continuous signed
densities against the original uniform law. Source: arXiv:2601.21366v2,
§5.1, injectivity in `lem: quadpol`.
The proved positive/negative-part argument retains the original kernel. -/
theorem sphereKernelOperator_injective {d : ℕ} (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) [IsFiniteMeasure ν] (hν : Metastability.IsUniformOn d ν) :
    Function.Injective (sphereKernelOperator β ν) := by
  intro f g h
  apply ContinuousMap.coe_injective
  apply eq_of_integral_exp_inner_mul_eq_uniform β hβ ν hν f g f.continuous g.continuous
  intro x
  simpa only [sphereKernelOperator_apply] using congrArg (fun q : C(SSphere d, ℝ) => q x) h

/-- Every original polynomial has an actual signed inverse under the
original exponential kernel, with its original degree bound preserved.
Source: arXiv:2601.21366v2, §5.1, the finite part of `lem: quadpol` (i).
Injectivity and the proved original preservation make the actual
finite-dimensional restriction surjective; no inverse is assumed. -/
theorem exists_polynomial_kernel_inverse {d k : ℕ} (hd : 2 ≤ d) (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (f : C(SSphere d, ℝ)) (hf : IsPolyOfDegreeLE k (f : SSphere d → ℝ)) :
    ∃ g : C(SSphere d, ℝ), IsPolyOfDegreeLE k (g : SSphere d → ℝ) ∧
      ∀ x : SSphere d,
        (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) = f x := by
  let _ : IsProbabilityMeasure ν := hν.1
  let V := spherePolynomialSpace d k
  let L : V →ₗ[ℝ] V := ((sphereKernelOperator β ν).domRestrict V).codRestrict V
    (fun g => sphereKernelOperator_preserves_polynomials hd β ν hν g.1 g.2)
  have hi : Function.Injective L := by
    intro g h heq
    apply Subtype.ext
    exact sphereKernelOperator_injective β hβ ν hν (congrArg Subtype.val heq)
  obtain ⟨g, hg⟩ := LinearMap.injective_iff_surjective.mp hi
    ⟨f, (mem_spherePolynomialSpace_iff f).mpr hf⟩
  have heq : sphereKernelOperator β ν g.1 = f := congrArg Subtype.val hg
  refine ⟨g.1, (mem_spherePolynomialSpace_iff g.1).mp g.2, fun x => ?_⟩
  rw [← sphereKernelOperator_apply]
  exact congrArg (fun q : C(SSphere d, ℝ) => q x) heq

/-- Every original dimensional, temperature, uniform-law and polynomial
input holds jointly on the actual original circle with constant one.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The law is the original rotation-invariant spherical probability. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    IsPolyOfDegreeLE 0 (ContinuousMap.const (SSphere 2) (1 : ℝ) : SSphere 2 → ℝ) ∧
    Continuous (fun _ : SSphere 2 => (1 : ℝ)) :=
  ⟨le_rfl, one_pos, Metastability.isUniformOn_uniformLaw (by norm_num),
    isPolyOfDegreeLE_const 1, continuous_const⟩

end Transformer.Perceptron
