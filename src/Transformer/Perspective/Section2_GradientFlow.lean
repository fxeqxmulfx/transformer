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

Not proved here: differentiating `partitionMu` under the integral sign is not
formalized.

Source: arXiv:2312.10794v5, §3.3, `eq: logder`. -/
theorem vectorField_eq_grad_log (β : ℝ) (hβ : 0 < β) (μ : ProbSphere d) :
    ∀ x : EucSpace d,
      vectorField d β μ x
        = proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β μ z)) x) := by
  sorry

/-- The hypothesis of `vectorField_eq_grad_log` is satisfiable: `β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

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

Not proved here: the first variation is computed by differentiating under the
integral sign, which is not formalized.

Source: arXiv:2312.10794v5, §3.3, `e:XmuE`. -/
theorem usaVectorField_eq_grad_first_variation (β : ℝ) (hβ : 0 < β)
    (μ : ProbSphere d) :
    ∀ x : EucSpace d,
      usaVectorField d β μ x
        = proj d x (gradient (fun z => β⁻¹ * partitionMu d β μ z) x) := by
  sorry

/-- The hypothesis of `usaVectorField_eq_grad_first_variation` is satisfiable:
`β = 1`. -/
example : (0 : ℝ) < 1 := one_pos

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

Not proved here: as for `dissipation_softmax`, differentiating the energy
along the flow is not formalized.

Source: arXiv:2312.10794v5, §3.3, `lem: dissipation`. -/
theorem usa_dissipation (β : ℝ) (μ : ℝ → ProbSphere d)
    (hCE : usaContinuityEquation d β μ) :
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖usaVectorField d β (μ t) (x : EucSpace d)‖ ^ 2
          ∂(μ t : Measure (SSphere d))) t := by
  sorry

/-- A Dirac mass is a stationary point of the `USA` vector field too: the only
point `δ_x` sees is `x`, so the integral is a multiple of `x` and `Proj_x`
kills it.  Dropping the partition function changes the length of the velocity,
never its direction. -/
theorem usaVectorField_diracProb_self (β : ℝ) (x : SSphere d) :
    usaVectorField d β (diracProb d x) (x : EucSpace d) = 0 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  rw [usaVectorField, hμ, integral_dirac]
  exact proj_smul_self hx _

/-- The constant curve at a Dirac mass solves `eq: pde.nosoftmaxZ`, so the
hypothesis of `usa_dissipation` is satisfiable. -/
theorem usaContinuityEquation_const_diracProb (β : ℝ) (x : SSphere d) :
    usaContinuityEquation d β (fun _ => diracProb d x) := by
  intro φ _ t
  have hμ : ((diracProb d x : ProbSphere d) : Measure (SSphere d))
      = Measure.dirac x := rfl
  have hrhs :
      (∫ y, inner (𝕜 := ℝ) (gradient φ (y : EucSpace d))
          (usaVectorField d β (diracProb d x) (y : EucSpace d))
        ∂((diracProb d x : ProbSphere d) : Measure (SSphere d))) = 0 := by
    rw [hμ, integral_dirac, usaVectorField_diracProb_self, inner_zero_right]
  rw [hrhs]
  exact hasDerivAt_const t _

example : usaContinuityEquation 1 1 (fun _ => diracProb 1 (basePoint 0)) :=
  usaContinuityEquation_const_diracProb 1 1 (basePoint 0)

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

Not proved here: the identity is not.

Source: arXiv:2312.10794v5, §3.4. -/
theorem sa_is_gradient_flow (β : ℝ) (V : ParamMatrix d) (X : ℝ → SphereTuple d n)
    (hβ : 0 < β)
    (hV : ∀ x y : EucSpace d, inner (𝕜 := ℝ) (V x) y = inner (𝕜 := ℝ) x (V y))
    (hX : Perspective.transformerODE d n β (fun _ => V) (fun _ => V) (fun _ => V) X) :
    ∀ (t : ℝ) (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d),
    Y 0 = X t →
    (∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) 0) →
    HasDerivAt (fun s => particleEnergy d n β V (Y s))
      (modifiedMetric d n β V (X t)
        (fun i => deriv (fun s => (X s i : EucSpace d)) t) b) 0 := by
  sorry

/-- A single token sitting still solves `eq: transformerSd.QKV` with
`Q = K = V = I_d`: it attends only to itself, so the attention average is `x`
and `Proj_x x = 0`. -/
theorem transformerODE_const_one (β : ℝ) (p : SSphere d) :
    Perspective.transformerODE d 1 β
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d))
      (fun _ => ContinuousLinearMap.id ℝ (EucSpace d)) (fun _ _ => p) := by
  intro t _
  have hp : ‖(p : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2
  simp only [ContinuousLinearMap.coe_id', id_eq, Fin.sum_univ_one, smul_smul]
  rw [proj_smul_self hp]
  exact hasDerivAt_const t _

/-- The hypotheses of `sa_is_gradient_flow` are satisfiable, and by a genuine
solution: `β = 1`, `V = I_d` — self-adjoint, since `⟨x, y⟩ = ⟨x, y⟩` — and the
stationary one-token solution of `transformerODE_const_one`. -/
example :
    (0 : ℝ) < 1 ∧
      (∀ x y : EucSpace 1,
        inner (𝕜 := ℝ) (ContinuousLinearMap.id ℝ (EucSpace 1) x) y
          = inner (𝕜 := ℝ) x (ContinuousLinearMap.id ℝ (EucSpace 1) y)) ∧
      Perspective.transformerODE 1 1 1
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1))
        (fun _ _ => basePoint 0) :=
  ⟨one_pos, fun _ _ => rfl, transformerODE_const_one 1 1 (basePoint 0)⟩

/-- **Equation (eq: first.rewriting).** Rewriting of `eq: conteqSd` through
`eq: logder`: `eq: CE` at the velocity field `Proj_x ∇(β⁻¹ log Z_{β,μ(t)})`.
By `VectorFieldEqGradLog` it is `continuityEquation`; the difference with
`aggregationEquation` is exactly the logarithm, i.e. the normalisation. -/
def conteqFirstRewriting (β : ℝ) (μ : ℝ → ProbSphere d) : Prop :=
  auxCE d μ (fun t x =>
    proj d x (gradient (fun z => β⁻¹ * Real.log (partitionMu d β (μ t) z)) x))

end Perspective
end Transformer
