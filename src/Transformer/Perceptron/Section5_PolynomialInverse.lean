/-
# The original polynomial kernel inverse in every dimension

ArXiv:2601.21366v2, §5.1, `lem: quadpol` (i).
The paper assumes dimension at least two. The original polynomial
characterization also holds for the actual two-point sphere in dimension
one: every actual function is affine in the original coordinate. There
is no probability on the empty sphere in dimension zero.
Together with the proved kernel preservation in dimensions at least two,
these cases give an actual finite-dimensional inverse in every dimension.
-/

import Transformer.Perceptron.Section5_PolynomialKernel
import Transformer.Perceptron.Section5_ConstantTransform

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- An actual unit sphere point forces positive original dimension.
Source: arXiv:2601.21366v2, §5.1, the dimension extension of `lem: quadpol`.
The zero-dimensional ambient space has only its actual zero vector. -/
theorem pos_dimension_of_sphere {d : ℕ} (x : SSphere d) : 0 < d := by
  by_contra h
  have hd : d = 0 := by omega
  subst d
  have hx : (x : EucSpace 0) = 0 := Subsingleton.elim _ _
  have hn := norm_eq_of_mem_sphere x
  rw [hx, norm_zero] at hn
  norm_num at hn

/-- The actual original one-dimensional sphere consists of the original
positive axis and its actual antipode. Source: arXiv:2601.21366v2, §5.1,
the dimension-one extension of `lem: quadpol` (i). -/
theorem sphere_one_eq_axis_or_antipode (x : SSphere 1) :
    x = basePoint 0 ∨ x = antipodeMap 1 (basePoint 0) := by
  have heq : (x : EucSpace 1) = EuclideanSpace.single (0 : Idx 1) ((x : EucSpace 1) 0) := by
    ext i
    fin_cases i
    simp
  have hn := norm_eq_of_mem_sphere x
  rw [heq, PiLp.norm_single, Real.norm_eq_abs] at hn
  rcases (abs_eq zero_le_one).mp hn with hp | hm
  · left
    apply Subtype.ext
    ext i
    fin_cases i
    simpa [basePoint, PiLp.single_apply] using hp
  · right
    apply Subtype.ext
    ext i
    fin_cases i
    simpa [coe_antipodeMap, PiLp.neg_apply, basePoint, PiLp.single_apply] using hm

/-- Every actual function on the original two-point sphere is an actual
spherical polynomial of degree at most one. Source: arXiv:2601.21366v2,
§5.1, the dimension-one extension of `lem: quadpol` (i).
The affine interpolation uses its original values at both original points. -/
theorem isPolyOfDegreeLE_one_dim (f : SSphere 1 → ℝ) : IsPolyOfDegreeLE 1 f := by
  let p : SSphere 1 := basePoint 0
  let q : SSphere 1 := antipodeMap 1 p
  let C := (f p + f q) / 2
  let D := (f p - f q) / 2
  have hp : IsPolyOfDegreeLE 1 (fun x : SSphere 1 => C + D * inner (𝕜 := ℝ) (p : EucSpace 1) (x : EucSpace 1)) :=
    isPolyOfDegreeLE_add (isPolyOfDegreeLE_mono (isPolyOfDegreeLE_const C) (by omega))
      (isPolyOfDegreeLE_const_mul D (isPolyOfDegreeLE_inner (p : EucSpace 1)))
  have hpp : inner (𝕜 := ℝ) (p : EucSpace 1) (p : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_sq, norm_eq_of_mem_sphere p, one_pow]
  have hpq : inner (𝕜 := ℝ) (p : EucSpace 1) (q : EucSpace 1) = -1 := by
    change inner (𝕜 := ℝ) (p : EucSpace 1) (antipodeMap 1 p : EucSpace 1) = -1
    rw [coe_antipodeMap, inner_neg_right, hpp]
  have hval : f = (fun x : SSphere 1 => C + D * inner (𝕜 := ℝ) (p : EucSpace 1) (x : EucSpace 1)) := by
    funext x
    rcases sphere_one_eq_axis_or_antipode x with hx | hx
    · change x = p at hx
      rw [hx, hpp]
      dsimp [C, D]
      ring
    · change x = q at hx
      rw [hx, hpq]
      dsimp [C, D]
      ring
  rwa [← hval] at hp

/-- Every actual function on the original two-point sphere satisfies
every positive original degree bound. Source: arXiv:2601.21366v2, §5.1,
the dimension-one extension of `lem: quadpol` (i). -/
theorem isPolyOfDegreeLE_one_dim_of_pos {k : ℕ} (hk : 1 ≤ k) (f : SSphere 1 → ℝ) :
    IsPolyOfDegreeLE k f := isPolyOfDegreeLE_mono (isPolyOfDegreeLE_one_dim f) hk

/-- The actual original exponential kernel preserves the original degree
bound in every dimension. Source: arXiv:2601.21366v2, §5.1,
`eq:FH-basic` and `lem: quadpol` (i), extended to the two-point sphere.
Positive degree contains every function there, while degree zero uses
the original uniform transform's proved actual constancy. -/
theorem sphereKernelOperator_preserves_polynomials_all_dimensions {d k : ℕ}
    (β : ℝ) (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (hν : Metastability.IsUniformOn d ν) (f : C(SSphere d, ℝ))
    (hf : f ∈ spherePolynomialSpace d k) :
    sphereKernelOperator β ν f ∈ spherePolynomialSpace d k := by
  by_cases hd : 2 ≤ d
  · exact sphereKernelOperator_preserves_polynomials hd β ν hν f hf
  let _ : IsProbabilityMeasure ν := hν.1
  have hpos := pos_dimension_of_sphere (Classical.choice (nonempty_of_isProbabilityMeasure ν))
  have hd1 : d = 1 := by omega
  subst d
  apply (mem_spherePolynomialSpace_iff _).mpr
  by_cases hk : k = 0
  · subst k
    obtain ⟨C, hC⟩ := (isPolyOfDegreeLE_zero_iff_constant _).mp
      ((mem_spherePolynomialSpace_iff f).mp hf)
    let σ : Perspective.ProbSphere 1 := ⟨ν, hν.1⟩
    apply (isPolyOfDegreeLE_zero_iff_constant _).mpr
    refine ⟨C * attentionTransform β σ (basePoint 0), fun x => ?_⟩
    rw [sphereKernelOperator_apply]
    simp_rw [hC]
    rw [integral_mul_const]
    change attentionTransform β σ x * C = _
    rw [attentionTransform_eq_of_uniform β σ hν x (basePoint 0)]
    ring
  · exact isPolyOfDegreeLE_one_dim_of_pos (by omega) _

/-- Every original spherical polynomial has an actual continuous signed
kernel inverse with the same degree bound, in every dimension.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i), with its valid
dimension-one extension retained. Preservation and signed injectivity
make the actual finite-dimensional restriction surjective. -/
theorem exists_polynomial_kernel_inverse_all_dimensions {d k : ℕ}
    (β : ℝ) (hβ : 0 < β) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (f : C(SSphere d, ℝ)) (hf : IsPolyOfDegreeLE k (f : SSphere d → ℝ)) :
    ∃ g : C(SSphere d, ℝ), IsPolyOfDegreeLE k (g : SSphere d → ℝ) ∧
      ∀ x : SSphere d,
        (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) * g y ∂ν) = f x := by
  let _ : IsProbabilityMeasure ν := hν.1
  let V := spherePolynomialSpace d k
  let L : V →ₗ[ℝ] V := ((sphereKernelOperator β ν).domRestrict V).codRestrict V
    (fun g => sphereKernelOperator_preserves_polynomials_all_dimensions β ν hν g.1 g.2)
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

/-- The positive-degree and original nonempty-sphere inputs hold jointly
at the actual original axis. Source: arXiv:2601.21366v2, §5.1,
the dimension extension of `lem: quadpol` (i). -/
example : (1 : ℕ) ≤ 1 ∧ 0 < 1 ∧ IsPolyOfDegreeLE 1 (fun _ : SSphere 1 => (1 : ℝ)) :=
  ⟨le_rfl, pos_dimension_of_sphere (basePoint 0), isPolyOfDegreeLE_one_dim _⟩

/-- Every inverse input holds jointly at the constructed original uniform
circle law, positive temperature and constant original polynomial.
Source: arXiv:2601.21366v2, §5.1, `lem: quadpol` (i). -/
example : (0 : ℝ) < 1 ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    IsPolyOfDegreeLE 0 (ContinuousMap.const (SSphere 2) (1 : ℝ) : SSphere 2 → ℝ) ∧
    ContinuousMap.const (SSphere 2) (1 : ℝ) ∈ spherePolynomialSpace 2 0 :=
  ⟨one_pos, Metastability.isUniformOn_uniformLaw (by norm_num), isPolyOfDegreeLE_const 1,
    const_mem_spherePolynomialSpace 2 0 1⟩

end Transformer.Perceptron
