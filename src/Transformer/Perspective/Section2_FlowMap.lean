/-
# §3 — Measure to measure flow map

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §3.1 and §3.2 of the survey; §3.3 (`USA`) and §3.4 (the
modified-metric gradient flow) are in `Perspective.Section2_GradientFlow`.

Main objects:

* `eq: mean.field.ips`     — the mean-field rewriting of `SA`,
* `eq: vfSd`               — the vector field `𝒳[μ]`,
* `eq: partition.function` — the partition function `Z_{β,μ}(x)`,
* `eq: conteqSd`           — the continuity equation,
* `eq: CE`                 — its form for an arbitrary velocity field,
* `eq: interaction.energy` — the interaction energy `𝖤_β[μ]`.

`prop: existence.uniqueness.energy` is proved in two halves: the minimiser in
`Perspective.Section2_EnergyMin`, on the strict convexity of
`Perspective.Section2_EnergyConvex`, and the maximiser — every global
maximiser of `𝖤_β` is a Dirac mass — in `Perspective.Section2_EnergyMax`, on
the analytic groundwork of `Perspective.Section2_EnergyKernel`.  The
dissipation of `𝖤_β` along `SA`, `eq: dissipation.softmax`, is proved in
`Perspective.Section2_Dissipation`.

For the integrals over `SSphere d` we equip the sphere with its induced Borel
measurable space (it is a metric subspace of `ℝ^d`).
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- Equip the sphere with its induced Borel measurable space. -/
noncomputable instance sphereMeasurableSpace : MeasurableSpace (SSphere d) :=
  borel _

/-- That σ-algebra *is* the Borel one, by definition.  Registering the fact
makes the sphere's continuous maps measurable and its singletons measurable
sets, which is what integration against a measure on `𝕊^{d-1}` needs. -/
instance sphereBorelSpace : BorelSpace (SSphere d) := ⟨rfl⟩

/-- Probability measures on `𝕊^{d-1}`. -/
abbrev ProbSphere (d : ℕ) : Type := ProbabilityMeasure (SSphere d)

/-- The Dirac measure at a point of the sphere, as an element of
`𝒫(𝕊^{d-1})`: the simplest inhabitant there is, and the one the satisfiability
witnesses throughout this development are built from. -/
noncomputable def diracProb (x : SSphere d) : ProbSphere d :=
  ⟨Measure.dirac x, inferInstance⟩

/-! ### §3.1 — The continuity equation -/

/-- **Equation (eq: partition.function).** Partition function for a measure:

  `Z_{β,μ}(x) = ∫ exp(β ⟨x, y⟩) dμ(y)`. -/
noncomputable def partitionMu
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : ℝ :=
  ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
    ∂(μ : Measure (SSphere d))

/-- **Equation (eq: vfSd).** The mean-field vector field:

  `𝒳[μ](x) = Proj_x ( Z_{β,μ}(x)⁻¹ · ∫ exp(β ⟨x,y⟩) y dμ(y) )`. -/
noncomputable def vectorField
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : EucSpace d :=
  proj d x ((partitionMu d β μ x)⁻¹ •
    ∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d))
          • (y : EucSpace d) ∂(μ : Measure (SSphere d)))

/-- **Equation (eq: mean.field.ips).** The mean-field rewriting of `SA`:

  `ẋ_i(t) = 𝒳[μ(t)](x_i(t))`,  where `μ(t) = (1/n) Σ_i δ_{x_i(t)}`. -/
def meanFieldIPS
    (β : ℝ) (μ : ℝ → ProbSphere d)
    (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (vectorField d β (μ t) ((X t i : EucSpace d))) t

/-- The empirical measure of the particles `(x_i(t))_{i ∈ [n]}`. -/
noncomputable def empiricalMeasure
    (X : SphereTuple d n) : Measure (SSphere d) :=
  ((n : ℝ)⁻¹ : ℝ).toNNReal • (∑ i : Idx n, (Measure.dirac (X i)))

/-- **Equation (eq: CE), §3.4.**  The continuity equation

  `∂_t μ(t) + div(μ(t) v(t, ·)) = 0`

for an arbitrary (tangent) velocity field `v`, in the distributional form the
paper works with: for every `C¹` test function `φ` on the ambient space,

  `d/dt ∫ φ dμ(t) = ∫ ⟨∇φ(x), v(t, x)⟩ dμ(t)(x)`.

It is stated here rather than in §3.4 because every continuity equation below
— `eq: conteqSd`, `eq: pde.nosoftmaxZ`, `eq: aggregation.eq`,
`eq: first.rewriting` — is this one at a particular `v`. -/
def auxCE (μ : ℝ → ProbSphere d) (v : ℝ → EucSpace d → EucSpace d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 1 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d)) (v t (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t

/-- **Equation (eq: conteqSd).** The continuity equation on the sphere,

  `∂_t μ(t) + div(μ(t) 𝒳[μ(t)]) = 0`,

in the distributional form that is the one the paper works with: for every
`C¹` test function `φ` on the ambient space,

  `d/dt ∫ φ dμ(t) = ∫ ⟨∇φ(x), 𝒳[μ(t)](x)⟩ dμ(t)(x)`.

The ambient gradient is the right pairing even though the flow lives on the
sphere, because `𝒳[μ]` is tangent there: `vectorField` ends in `proj`, so the
normal part of `∇φ` is annihilated. -/
def continuityEquation
    (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 1 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
          (vectorField d β (μ t) (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t

/-- `eq: conteqSd` is `eq: CE` at the velocity field `𝒳[μ(t)]`. -/
theorem continuityEquation_eq_auxCE (β : ℝ) (μ : ℝ → ProbSphere d) :
    continuityEquation d β μ = auxCE d μ (fun t x => vectorField d β (μ t) x) := rfl

/-! ### §3.2 — The interaction energy -/

/-- **Equation (eq: interaction.energy).** Interaction energy:

  `𝖤_β[μ] = (1/(2β)) ∫∫ exp(β ⟨x,x'⟩) dμ(x) dμ(x')`. -/
noncomputable def interactionEnergy
    (β : ℝ) (μ : ProbSphere d) : ℝ :=
  (2 * β)⁻¹ *
    ∫ x, ∫ x',
      Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (x' : EucSpace d))
        ∂(μ : Measure (SSphere d)) ∂(μ : Measure (SSphere d))

/-- **A Dirac mass is a stationary solution of `eq: conteqSd`.**

`𝒳[δ_x](x)` is a multiple of `x` — the only point the measure sees is `x`
itself — and `Proj_x` kills the radial direction, so the velocity vanishes
`δ_x`-almost everywhere and the mass does not move.  This is what makes the
hypothesis of `dissipation_softmax` (in `Section2_Dissipation`) satisfiable,
and satisfiable by something other than a contradiction. -/
theorem vectorField_diracProb_self (β : ℝ) (x : SSphere d) :
    vectorField d β (diracProb d x) (x : EucSpace d) = 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  rw [vectorField, hμ, integral_dirac, smul_smul]
  exact proj_smul_self hx _

/-- The constant curve at a Dirac mass solves `eq: conteqSd`: both sides of the
distributional form vanish, the left because the curve is constant and the
right by `vectorField_diracProb_self`. -/
theorem continuityEquation_const_diracProb (β : ℝ) (x : SSphere d) :
    continuityEquation d β (fun _ => diracProb d x) := by
  intro φ _ t
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  have hrhs :
      (∫ y, inner (𝕜 := ℝ) (gradient φ (y : EucSpace d))
          (vectorField d β (diracProb d x) (y : EucSpace d))
        ∂((diracProb d x : ProbSphere d) : Measure (SSphere d))) = 0 := by
    rw [hμ, integral_dirac, vectorField_diracProb_self, inner_zero_right]
  rw [hrhs]
  exact hasDerivAt_const t _

/-- A linear isometry of the ambient space, restricted to the unit sphere:
the action of `O(d)` on `𝕊^{d-1}` that the uniform measure `σ_d` — and only
it, among probability measures — is invariant under. -/
def sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (x : SSphere d) : SSphere d :=
  ⟨U (x : EucSpace d), by
    rw [mem_sphere_zero_iff_norm, LinearIsometryEquiv.norm_map]
    exact mem_sphere_zero_iff_norm.mp x.2⟩

end Perspective
end Transformer
