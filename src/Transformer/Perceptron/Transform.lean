/-
# Perceptrons and attention's mean-field landscape — the exponential transform

Formalization of `lem: quadpol` and `rem: general-attention` of
arXiv:2601.21366v2: the map `μ ↦ f^μ`, `f^μ(x) = ∫ e^{β x·y} dμ(y)`, is
injective on `𝒫(𝕊^{d-1})`, and `f^μ` is even exactly when `μ` is antipodally
symmetric.  Part (i) of the lemma, on polynomial transforms, is
`TransformPoly`.

**What the source says and what is carried here.**

* The source's standing `d ≥ 2` is dropped: injectivity and the parity
  characterization hold in every dimension (`𝕊^0 = {±1}`, where `f^μ(1)` is
  strictly increasing in `μ({1})`, and `𝕊^{-1} = ∅`, where `𝒫(𝕊^{-1})` is
  empty).  Dropping a hypothesis only strengthens a statement.

* "`μ(A) = μ(-A)` for every Borel `A`" is the pushforward equality
  `μ ∘ (-id)^{-1} = μ`, the tree's idiom for an invariance — the same one
  `Metastability.IsUniformOn` uses for rotation invariance.

Source: arXiv:2601.21366v2, `lem: quadpol`, `rem: general-attention`.
-/

import Transformer.Perceptron.Basic
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

/-! ### Injectivity -/

/-- **Lemma (lem: quadpol), injectivity.**  For `β > 0` the map `μ ↦ f^μ` is
injective on `𝒫(𝕊^{d-1})`.

Not proved here.

Source: arXiv:2601.21366v2, `lem: quadpol`. -/
theorem injective_attentionTransform (β : ℝ) (hβ : 0 < β) :
    Function.Injective (attentionTransform (d := d) β) := by
  sorry

/-- The hypothesis of `injective_attentionTransform` is satisfiable: `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

/-- **Remark (rem: general-attention), injectivity.**  For a symmetric
invertible `B` the map `μ ↦ f_B^μ` is injective on `𝒫(𝕊^{d-1})`.

Not proved here.

Source: arXiv:2601.21366v2, `rem: general-attention`. -/
theorem injective_attentionTransformMap (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hsymm : B.IsSymmetric) (hB : Function.Bijective B) :
    Function.Injective (attentionTransformMap B) := by
  sorry

/-- The hypotheses of `injective_attentionTransformMap` are satisfiable:
`B = id`, for which `f_B = f^μ` at `β = 1`. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace d)).IsSymmetric ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace d)) :=
  ⟨fun _ _ => rfl, Function.bijective_id⟩

/-! ### Parity -/

/-- **Lemma (lem: quadpol) (ii), for a general attention matrix.**  `f_B^μ` is
even if and only if `μ(A) = μ(-A)` for every Borel `A`.

Proved, from the injectivity of `μ ↦ f_B^μ` taken as an explicit hypothesis —
that is `injective_attentionTransformMap`, which is not proved here.

Source: arXiv:2601.21366v2, `lem: quadpol` (ii), `rem: general-attention`. -/
theorem even_attentionTransformMap_iff (B : EucSpace d →ₗ[ℝ] EucSpace d)
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

/-- **Lemma (lem: quadpol) (ii).**  `f^μ` is even if and only if
`μ(A) = μ(-A)` for every Borel `A`.

Proved, from the injectivity of `μ ↦ f^μ` taken as an explicit hypothesis —
that is `injective_attentionTransform`, which is not proved here.

Source: arXiv:2601.21366v2, `lem: quadpol` (ii). -/
theorem even_attentionTransform_iff (β : ℝ)
    (hinj : Function.Injective (attentionTransform (d := d) β))
    (μ : Perspective.ProbSphere d) :
    (∀ x : SSphere d, attentionTransform β μ (antipodeMap d x) = attentionTransform β μ x)
      ↔ (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)) := by
  have hinj' : Function.Injective (attentionTransformMap (β • LinearMap.id (M := EucSpace d))) := by
    intro μ₁ μ₂ h
    exact hinj (by
      rw [← attentionTransformMap_smul_id, ← attentionTransformMap_smul_id, h])
  have := even_attentionTransformMap_iff (β • LinearMap.id (M := EucSpace d)) hinj' μ
  rwa [attentionTransformMap_smul_id] at this

/-- The hypothesis of `even_attentionTransform_iff` is satisfiable: at `β = 1`
it is `injective_attentionTransform 1 one_pos`. -/
example : Function.Injective (attentionTransform (d := d) 1) :=
  injective_attentionTransform 1 one_pos

end Perceptron
end Transformer
