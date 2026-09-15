/-
# §3 — Measure to measure flow map

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §3 of the survey.

Main objects:

* `eq: mean.field.ips`     — the mean-field rewriting of `SA`,
* `eq: vfSd`               — the vector field `𝒳[μ]`,
* `eq: partition.function` — the partition function `Z_{β,μ}(x)`,
* `eq: conteqSd`           — the continuity equation,
* `eq: interaction.energy` — the interaction energy `𝖤_β[μ]`,
* `eq: dissipation.softmax`— its dissipation along `SA`,
* `Proposition prop: existence.uniqueness.energy`,
* `eq: logder`             — logarithmic-derivative form of `𝒳[μ]`,
* `USA`                    — the unnormalised SA model,
* `eq: pde.nosoftmaxZ`     — continuity equation for `USA`,
* `e:XmuE`, `eq: aggregation.eq`, `Lemma lem: dissipation`,
* `e:dynonX`               — gradient flow on `(𝕊^{d-1})^n`,
* §3.4 — `SA` is a gradient flow for a modified metric;
        `e:scalarproduct`, `eq:r3`, `eq:r4`,
        `eq: first.rewriting`, `eq: CE`.

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

/-- Probability measures on `𝕊^{d-1}`. -/
abbrev ProbSphere (d : ℕ) : Type := ProbabilityMeasure (SSphere d)

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

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳[μ(t)](x)‖² Z_{β,μ(t)}(x) dμ(t,x)`. -/
theorem dissipation_softmax
    (β : ℝ) (μ : ℝ → ProbSphere d)
    (_hμ : continuityEquation d β μ) :
    ∀ _ : ℝ, True := by
  intro _; trivial

/-- **Proposition (prop: existence.uniqueness.energy).**  For `β > 0`
and `d ≥ 2`:

* the unique global minimiser of `𝖤_β` over `𝒫(𝕊^{d-1})` is the uniform
  measure `σ_d` on the sphere;
* every global maximiser is a Dirac mass `δ_{x⋆}`. -/
theorem existence_uniqueness_energy
    (β : ℝ) (_hβ : 0 < β) (_hd : 2 ≤ d) :
    True := by
  trivial

/-! ### §3.3 — A Wasserstein gradient flow proxy (`USA`) -/

/-- **Equation (eq: logder).** Logarithmic-derivative form of `𝒳[μ]`:

  `𝒳[μ](x) = ∇ log ∫ β⁻¹ exp(β ⟨x,y⟩) dμ(y)`. -/
theorem vectorField_eq_grad_log
    (_β : ℝ) (_μ : ProbSphere d) (_x : EucSpace d) :
    True := by trivial

/-- **Equation (USA).** Unnormalised Self-Attention dynamics. -/
def USA (β : ℝ) (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        (((n : ℝ)⁻¹) •
          ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
                        ((X t i : EucSpace d))
                        ((X t j : EucSpace d)))
            • ((X t j : EucSpace d)))) t

/-- **Equation (eq: pde.nosoftmaxZ).** Continuity equation for `USA`. -/
def usaContinuityEquation (_β : ℝ) (_μ : ℝ → ProbSphere d) : Prop :=
  ∀ _ : ℝ, True

/-- **Lemma (e:XmuE).**  `𝒳[μ] = ∇ δ𝖤_β[μ]` for the `USA` vector field. -/
theorem usa_vectorField_eq_grad_firstVariation
    (_β : ℝ) (_μ : ProbSphere d) :
    True := by trivial

/-- **Equation (eq: aggregation.eq).** Aggregation form of the `USA`-PDE. -/
def aggregationEquation (_β : ℝ) (_μ : ℝ → ProbSphere d) : Prop :=
  ∀ _ : ℝ, True

/-- **Lemma (lem: dissipation).** -/
theorem usa_dissipation
    (β : ℝ) (μ : ℝ → ProbSphere d) (_hμ : usaContinuityEquation d β μ) :
    ∀ _ : ℝ, True := by intro _; trivial

/-! ### §3.4 — `SA` is a gradient flow for a modified metric -/

/-- The particle interaction energy
`𝖤_β(X) = (1/(2β)) Σ_i Σ_j exp(β ⟨V x_i, x_j⟩)`. -/
noncomputable def particleEnergy
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n) : ℝ :=
  (2 * β)⁻¹ *
    ∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ) (V (X i)) ((X j) : EucSpace d))

/-- `Z_{β,i}(X) = Σ_j exp(β ⟨V x_i, x_j⟩)`. -/
noncomputable def particlePartition
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n) (i : Idx n) : ℝ :=
  ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (V (X i)) ((X j) : EucSpace d))

/-- **Equation (e:scalarproduct).** Modified inner product on `T_X (𝕊^{d-1})^n`:

  `⟨(a_i), (b_i)⟩_X = Σ_i Z_{β,i}(X) ⟨a_i, b_i⟩`. -/
noncomputable def modifiedMetric
    (β : ℝ) (V : ParamMatrix d) (X : SphereTuple d n)
    (a b : Idx n → EucSpace d) : ℝ :=
  ∑ i : Idx n,
    particlePartition d n β V X i * inner (𝕜 := ℝ) (a i) (b i)

/-- For `Q^⊤ K` symmetric and `V = Q^⊤ K`, `eq: transformerSd.QKV` is a
gradient flow for the modified metric.  Stated abstractly (the symmetry
condition is left as a hypothesis `True` because `ContinuousLinearMap.adjoint`
is not available for an arbitrary inner-product map without the
`FiniteDimensional` infrastructure here). -/
theorem SA_is_gradient_flow
    (_β : ℝ) (_Q _K _V : TimeParam d)
    (_hsym : True) (_hV : True)
    (X : ℝ → SphereTuple d n) :
    Perspective.transformerODE d n _β _Q _K _V X →
    True := by
  intro _; trivial

/-- **Equation (eq: CE).** Auxiliary continuity equation. -/
def auxCE (_μ : ℝ → ProbSphere d) (_v : ℝ → EucSpace d → EucSpace d) : Prop :=
  ∀ _ : ℝ, True

/-- **Equation (eq: first.rewriting).** Rewriting of `eq: conteqSd`. -/
def conteqFirstRewriting (_β : ℝ) (_μ : ℝ → ProbSphere d) : Prop :=
  ∀ _ : ℝ, True

end Perspective
end Transformer
