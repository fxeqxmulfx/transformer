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
* `eq: interaction.energy` — the interaction energy `𝖤_β[μ]`,
* `eq: dissipation.softmax`— its dissipation along `SA`,
* `Proposition prop: existence.uniqueness.energy`.

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

/-- **Equation (eq: dissipation.softmax).** Energy-dissipation identity:

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳[μ(t)](x)‖² Z_{β,μ(t)}(x) dμ(t,x)`.

In particular the interaction energy is non-decreasing along `SA`.

A `Prop`-valued definition and not a theorem: differentiating the energy under
the integral sign along a solution of the continuity equation is not
formalized here.

Source: arXiv:2312.10794v5, §3.2, `eq: dissipation.softmax`. -/
def DissipationSoftmax (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  continuityEquation d β μ →
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖vectorField d β (μ t) (x : EucSpace d)‖ ^ 2
              * partitionMu d β (μ t) (x : EucSpace d)
          ∂(μ t : Measure (SSphere d))) t

/-- A linear isometry of the ambient space, restricted to the unit sphere:
the action of `O(d)` on `𝕊^{d-1}` that the uniform measure `σ_d` — and only
it, among probability measures — is invariant under. -/
def sphereMap (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (x : SSphere d) : SSphere d :=
  ⟨U (x : EucSpace d), by
    rw [mem_sphere_zero_iff_norm, LinearIsometryEquiv.norm_map]
    exact mem_sphere_zero_iff_norm.mp x.2⟩

/-- **Proposition (prop: existence.uniqueness.energy).**  For `β > 0`
and `d ≥ 2`:

* the unique global minimiser of `𝖤_β` over `𝒫(𝕊^{d-1})` is the uniform
  measure `σ_d` on the sphere;
* every global maximiser is a Dirac mass `δ_{x⋆}`.

The minimiser is pinned down here by rotation invariance rather than by
name — a probability measure on `𝕊^{d-1}` invariant under every linear
isometry of `ℝ^d` *is* `σ_d` — so that no Haar-measure machinery is needed to
state the proposition.

A `Prop`-valued definition and not a theorem: neither half is proved here.

Source: arXiv:2312.10794v5, §3.2, `prop: existence.uniqueness.energy`. -/
def ExistenceUniquenessEnergy (β : ℝ) : Prop :=
  0 < β → 2 ≤ d →
    (∃! μ₀ : ProbSphere d, ∀ μ : ProbSphere d,
        interactionEnergy d β μ₀ ≤ interactionEnergy d β μ) ∧
    (∀ μ₀ : ProbSphere d,
        (∀ μ : ProbSphere d, interactionEnergy d β μ₀ ≤ interactionEnergy d β μ) →
        ∀ U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d,
          (μ₀ : Measure (SSphere d)).map (sphereMap d U)
            = (μ₀ : Measure (SSphere d))) ∧
    (∀ μ₁ : ProbSphere d,
        (∀ μ : ProbSphere d, interactionEnergy d β μ ≤ interactionEnergy d β μ₁) →
        ∃ x : SSphere d, (μ₁ : Measure (SSphere d)) = Measure.dirac x)

end Perspective
end Transformer
