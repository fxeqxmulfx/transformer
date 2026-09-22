/-
# Perceptrons and attention's mean-field landscape — the exponential transform

Formalization of `lem: quadpol` of arXiv:2601.21366v2: the map `μ ↦ f^μ`,
`f^μ(x) = ∫ e^{β x·y} dμ(y)`, is injective on `𝒫(𝕊^{d-1})`, and `f^μ` is even
exactly when `μ` is antipodally symmetric.  Part (i) of the lemma, on
polynomial transforms, is `TransformPoly`.  The general attention matrix of
`rem: general-attention` is `TransformMap`; the transform `f_B^μ` it studies,
and the parity argument both cases share, are defined and proved here.

**What the source says and what is carried here.**

* The source's standing `d ≥ 2` is dropped: injectivity and the parity
  characterization hold in every dimension (`𝕊^0 = {±1}`, where `f^μ(1)` is
  strictly increasing in `μ({1})`, and `𝕊^{-1} = ∅`, where `𝒫(𝕊^{-1})` is
  empty).  Dropping a hypothesis only strengthens a statement.

* The source's `β > 0` is relaxed to `β ≠ 0` in injectivity and in the parity
  characterization: `f^μ` at `-β` is `f^μ` at `β` read at the antipode
  (`attentionTransform_neg`).  At `β = 0` both fail once `d ≥ 1`: `f^μ ≡ 1`.

* "`μ(A) = μ(-A)` for every Borel `A`" is the pushforward equality
  `μ ∘ (-id)^{-1} = μ`, the tree's idiom for an invariance — the same one
  `Metastability.IsUniformOn` uses for rotation invariance.

* Injectivity is proved through the positive-definiteness of the kernel
  `e^{β x·y}` (`Perspective.PositiveDefinite`), not through the source's
  Funk–Hecke computation, which needs spherical harmonics.

Source: arXiv:2601.21366v2, `lem: quadpol`, `rem: general-attention`.
-/

import Transformer.Perceptron.Basic
import Transformer.Perspective.PositiveDefinite
import Transformer.Perspective.SphereInvariant

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The transform -/

/-- **`f^μ(x) = ∫ e^{β x·y} dμ(y)`**, the attention transform of a measure on
the sphere.

Source: arXiv:2601.21366v2, `lem: quadpol`. -/
noncomputable def attentionTransform (β : ℝ) (μ : Perspective.ProbSphere d)
    (x : SSphere d) : ℝ :=
  ∫ y, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
    ∂(μ : Measure (SSphere d))

/-- **`f_B(x) = ∫ e^{xᵀ B y} dμ(y)`**, the transform of `rem: general-attention`
for a general attention matrix `B`.

Source: arXiv:2601.21366v2, `rem: general-attention`. -/
noncomputable def attentionTransformMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : SSphere d) : ℝ :=
  ∫ y, Real.exp (inner (𝕜 := ℝ) (x : EucSpace d) (B (y : EucSpace d)))
    ∂(μ : Measure (SSphere d))

/-- `f^μ` is the case `B = β · id` of `f_B`. -/
theorem attentionTransformMap_smul_id (β : ℝ) (μ : Perspective.ProbSphere d) :
    attentionTransformMap (β • LinearMap.id) μ = attentionTransform β μ := by
  funext x
  simp [attentionTransformMap, attentionTransform, real_inner_smul_right]

/-! ### The antipodal map -/

/-- The antipodal map `x ↦ -x` of `𝕊^{d-1}`. -/
noncomputable def antipodeMap (d : ℕ) : SSphere d → SSphere d :=
  Perspective.sphereMap d (LinearIsometryEquiv.neg ℝ)

@[simp] theorem coe_antipodeMap (x : SSphere d) :
    ((antipodeMap d x : SSphere d) : EucSpace d) = -(x : EucSpace d) := rfl

theorem measurable_antipodeMap (d : ℕ) : Measurable (antipodeMap d) :=
  Perspective.measurable_sphereMap d (LinearIsometryEquiv.neg ℝ)

/-- The antipodal image `μ ∘ (-id)^{-1}` of a probability measure on the
sphere: the measure `A ↦ μ(-A)`. -/
noncomputable def antipode (μ : Perspective.ProbSphere d) : Perspective.ProbSphere d :=
  μ.map (antipodeMap d)

@[simp] theorem coe_antipode (μ : Perspective.ProbSphere d) :
    (antipode μ : Measure (SSphere d)) = (μ : Measure (SSphere d)).map (antipodeMap d) :=
  rfl

/-- **`f_B^{μ̌}(x) = f_B^μ(-x)`**, the computation both halves of the parity
characterization run on. -/
theorem attentionTransformMap_antipode (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransformMap B (antipode μ) x = attentionTransformMap B μ (antipodeMap d x) := by
  have hcont : Continuous fun y : SSphere d =>
      Real.exp (inner (𝕜 := ℝ) (x : EucSpace d) (B (y : EucSpace d))) :=
    Real.continuous_exp.comp (continuous_const.inner
      (B.continuous_of_finiteDimensional.comp continuous_subtype_val))
  rw [attentionTransformMap, coe_antipode,
    integral_map (measurable_antipodeMap d).aemeasurable hcont.aestronglyMeasurable]
  simp [attentionTransformMap, map_neg, inner_neg_left, inner_neg_right]

/-- The case `B = β · id`: `f^{μ̌}(x) = f^μ(-x)`. -/
theorem attentionTransform_antipode (β : ℝ) (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransform β (antipode μ) x = attentionTransform β μ (antipodeMap d x) := by
  rw [← attentionTransformMap_smul_id, ← attentionTransformMap_smul_id]
  exact attentionTransformMap_antipode _ μ x

/-- **`f^μ` at `-β` is `f^μ` at `β` read at the antipode:**
`∫ e^{-β x·y} dμ(y) = ∫ e^{β (-x)·y} dμ(y)`. -/
theorem attentionTransform_neg (β : ℝ) (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransform (-β) μ x = attentionTransform β μ (antipodeMap d x) := by
  simp [attentionTransform, inner_neg_left]

/-! ### Injectivity -/

/-- **Lemma (lem: quadpol), injectivity.**  For `β ≠ 0` the map `μ ↦ f^μ` is
injective on `𝒫(𝕊^{d-1})`.

The source states `β > 0`, and that case is
`Perspective.eq_of_integral_exp_inner_eq`.  `β < 0` is added: there `f^μ(x)`
is `f^μ(-x)` at `-β > 0` (`attentionTransform_neg`), and `x ↦ -x` is onto.

Source: arXiv:2601.21366v2, `lem: quadpol`. -/
theorem injective_attentionTransform (β : ℝ) (hβ : β ≠ 0) :
    Function.Injective (attentionTransform (d := d) β) := by
  intro μ₁ μ₂ h
  refine ProbabilityMeasure.toMeasure_injective ?_
  rcases hβ.lt_or_gt with hβ | hβ
  · refine Perspective.eq_of_integral_exp_inner_eq (-β) (neg_pos.2 hβ) _ _ fun x => ?_
    have := (attentionTransform_neg β μ₁ x).trans
      ((congrFun h _).trans (attentionTransform_neg β μ₂ x).symm)
    unfold attentionTransform at this
    exact this
  · refine Perspective.eq_of_integral_exp_inner_eq β hβ _ _ fun x => ?_
    have := congrFun h x
    unfold attentionTransform at this
    exact this

/-- The hypothesis of `injective_attentionTransform` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- `μ ↦ f_B^μ` is injective at `B = β · id`, `β ≠ 0`: it is `μ ↦ f^μ`. -/
theorem injective_attentionTransformMap_smul_id (β : ℝ) (hβ : β ≠ 0) :
    Function.Injective (attentionTransformMap (β • LinearMap.id (M := EucSpace d))) := by
  intro μ₁ μ₂ h
  exact injective_attentionTransform β hβ (by
    rw [← attentionTransformMap_smul_id, ← attentionTransformMap_smul_id, h])

/-- The hypothesis of `injective_attentionTransformMap_smul_id` is satisfiable:
`β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-! ### Parity -/

/-- **The parity argument of `lem: quadpol` (ii)**, for any `B` whose transform
is injective: `f_B^μ` is even if and only if `μ(A) = μ(-A)` for every Borel
`A`.  Injectivity holds for `B = β · id`, `β ≠ 0`
(`injective_attentionTransformMap_smul_id`), and for every invertible `B`
(`injective_attentionTransformMap`, in `TransformMap`).

Source: arXiv:2601.21366v2, `lem: quadpol` (ii), `rem: general-attention`. -/
theorem even_attentionTransformMap_iff_of_injective (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hinj : Function.Injective (attentionTransformMap B)) (μ : Perspective.ProbSphere d) :
    (∀ x : SSphere d, attentionTransformMap B μ (antipodeMap d x)
        = attentionTransformMap B μ x)
      ↔ (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)) := by
  constructor
  · intro h
    have hfun : attentionTransformMap B (antipode μ) = attentionTransformMap B μ :=
      funext fun x => (attentionTransformMap_antipode B μ x).trans (h x)
    exact congrArg (fun ν : Perspective.ProbSphere d => (ν : Measure (SSphere d)))
      (hinj hfun)
  · intro h x
    have hμ : antipode μ = μ := ProbabilityMeasure.toMeasure_injective (by simpa using h)
    rw [← attentionTransformMap_antipode B μ x, hμ]

/-- The hypothesis of `even_attentionTransformMap_iff_of_injective` is
satisfiable: `B = id`. -/
example : Function.Injective (attentionTransformMap (d := d) ((1 : ℝ) • LinearMap.id)) :=
  injective_attentionTransformMap_smul_id 1 one_ne_zero

/-- **Lemma (lem: quadpol) (ii).**  For `β ≠ 0`, `f^μ` is even if and only if
`μ(A) = μ(-A)` for every Borel `A`.

The source states `β > 0`; `β < 0` comes with `injective_attentionTransform`.

Source: arXiv:2601.21366v2, `lem: quadpol` (ii). -/
theorem even_attentionTransform_iff (β : ℝ) (hβ : β ≠ 0) (μ : Perspective.ProbSphere d) :
    (∀ x : SSphere d, attentionTransform β μ (antipodeMap d x) = attentionTransform β μ x)
      ↔ (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)) := by
  have := even_attentionTransformMap_iff_of_injective (β • LinearMap.id (M := EucSpace d))
    (injective_attentionTransformMap_smul_id β hβ) μ
  rwa [attentionTransformMap_smul_id] at this

/-- The hypothesis of `even_attentionTransform_iff` is satisfiable: `β = 1`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

end Perceptron
end Transformer
