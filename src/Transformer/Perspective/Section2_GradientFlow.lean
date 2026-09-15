/-
# §3.3–§3.4 — `USA` as a Wasserstein gradient flow, `SA` for a modified metric

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

Continues `Perspective.Section2_FlowMap`:

* `eq: logder`             — logarithmic-derivative form of `𝒳[μ]`,
* `USA`                    — the unnormalised `SA` model, and its vector field,
* `eq: pde.nosoftmaxZ`     — continuity equation for `USA`,
* `e:XmuE`, `eq: aggregation.eq`, `Lemma lem: dissipation`,
* §3.4 — `SA` is a gradient flow for a modified metric;
        `e:scalarproduct`, `eq: first.rewriting`.

The three PDEs of §3.3–§3.4 are `Perspective.auxCE` at three velocity fields;
what distinguishes them is exactly the identity `e:XmuE`, which says the first
two coincide.
-/

import Transformer.Perspective.Section2_FlowMap

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-! ### §3.3 — A Wasserstein gradient flow proxy (`USA`) -/

/-- **Equation (eq: logder).** Logarithmic-derivative form of `𝒳[μ]`:

  `𝒳[μ](x) = Proj_x ∇_x ( β⁻¹ log Z_{β,μ}(x) )`,

since `∇_x log Z_{β,μ}(x) = β Z_{β,μ}(x)⁻¹ ∫ exp(β ⟨x,y⟩) y dμ(y)`.

A `Prop`-valued definition and not a theorem: differentiating `partitionMu`
under the integral sign is not formalized here.

Source: arXiv:2312.10794v5, §3.3, `eq: logder`. -/
def VectorFieldEqGradLog (β : ℝ) (μ : ProbSphere d) : Prop :=
  0 < β → ∀ x : EucSpace d,
    vectorField d β μ x
      = proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β μ z)) x)

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

/-- The mean-field vector field of `USA`: `eq: vfSd` with the partition
function dropped,

  `𝒳^{USA}[μ](x) = Proj_x ∫ exp(β ⟨x,y⟩) y dμ(y)`. -/
noncomputable def usaVectorField
    (β : ℝ) (μ : ProbSphere d) (x : EucSpace d) : EucSpace d :=
  proj d x
    (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
      ∂(μ : Measure (SSphere d)))

/-- **Equation (eq: pde.nosoftmaxZ).** Continuity equation for `USA`: `eq: CE`
at the velocity field `𝒳^{USA}[μ(t)]`. -/
def usaContinuityEquation (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x => usaVectorField d β (μ t) x)

/-- **Lemma (e:XmuE).**  `𝒳^{USA}[μ] = Proj ∇ δ𝖤_β[μ]`: the first variation of
the interaction energy `𝖤_β` at `μ` is `x ↦ β⁻¹ Z_{β,μ}(x)`, and its ambient
gradient, projected onto the tangent space, is the `USA` vector field.

A `Prop`-valued definition and not a theorem: the first variation is computed
by differentiating under the integral sign, which is not formalized here.

Source: arXiv:2312.10794v5, §3.3, `e:XmuE`. -/
def UsaVectorFieldEqGradFirstVariation (β : ℝ) (μ : ProbSphere d) : Prop :=
  0 < β → ∀ x : EucSpace d,
    usaVectorField d β μ x
      = proj d x (gradient (fun z => β⁻¹ * partitionMu d β μ z) x)

/-- **Equation (eq: aggregation.eq).** Aggregation form of the `USA`-PDE:
`eq: CE` driven by the gradient of the first variation of `𝖤_β`,

  `∂_t μ = -div( μ Proj ∇ (β⁻¹ Z_{β,μ}) )`.

By `e:XmuE` this is `eq: pde.nosoftmaxZ`; written this way it exhibits the
Wasserstein gradient-flow structure. -/
def aggregationEquation (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x =>
    proj d x (gradient (fun z => β⁻¹ * partitionMu d β (μ t) z) x))

/-- **Lemma (lem: dissipation).**  Along `USA` the interaction energy
dissipates at rate

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳^{USA}[μ(t)](x)‖² dμ(t,x)`

— the same identity as `eq: dissipation.softmax` with the partition-function
weight `Z_{β,μ}` removed, which is precisely what the normalisation costs.

A `Prop`-valued definition and not a theorem: as for
`DissipationSoftmax`, differentiating the energy along the flow is not
formalized here.

Source: arXiv:2312.10794v5, §3.3, `lem: dissipation`. -/
def UsaDissipation (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  usaContinuityEquation d β μ →
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖usaVectorField d β (μ t) (x : EucSpace d)‖ ^ 2
          ∂(μ t : Measure (SSphere d))) t

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

/-- **`SA` is a gradient flow for the modified metric.**

For `Q^⊤ K` self-adjoint and `V = Q^⊤ K`, the dynamics `eq: transformerSd.QKV`
is the gradient *ascent* flow of the particle energy `𝖤_β` with respect to
`e:scalarproduct`: for every curve `Y` through `X(t)` with velocity `b`,

  `d/ds 𝖤_β(Y(s))|_{s=0} = ⟨Ẋ(t), b⟩_{X(t)}`.

Taking `b = Ẋ(t)` recovers `Σ_i Z_{β,i} ‖ẋ_i‖² = d/dt 𝖤_β ≥ 0`, the particle
counterpart of `eq: dissipation.softmax`.

Self-adjointness is a hypothesis on `V` spelled out as
`⟨V x, y⟩ = ⟨x, V y⟩` rather than through `ContinuousLinearMap.adjoint`, which
would need the `FiniteDimensional` infrastructure here.

A `Prop`-valued definition and not a theorem: the identity is not proved here.

Source: arXiv:2312.10794v5, §3.4. -/
def SAIsGradientFlow (β : ℝ) (V : ParamMatrix d) (X : ℝ → SphereTuple d n) : Prop :=
  0 < β →
  (∀ x y : EucSpace d, inner (𝕜 := ℝ) (V x) y = inner (𝕜 := ℝ) x (V y)) →
  Perspective.transformerODE d n β (fun _ => V) (fun _ => V) (fun _ => V) X →
  ∀ (t : ℝ) (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d),
    Y 0 = X t →
    (∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) 0) →
    HasDerivAt (fun s => particleEnergy d n β V (Y s))
      (modifiedMetric d n β V (X t)
        (fun i => deriv (fun s => (X s i : EucSpace d)) t) b) 0

/-- **Equation (eq: first.rewriting).** Rewriting of `eq: conteqSd` through
`eq: logder`: `eq: CE` at the velocity field `Proj_x ∇(β⁻¹ log Z_{β,μ(t)})`.
By `VectorFieldEqGradLog` it is `continuityEquation`; the difference with
`aggregationEquation` is exactly the logarithm, i.e. the normalisation. -/
def conteqFirstRewriting (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x =>
    proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β (μ t) z)) x))

end Perspective
end Transformer
